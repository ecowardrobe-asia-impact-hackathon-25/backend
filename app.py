import os
import json
import logging
import pickle
import random
from io import BytesIO
from typing import Any, Dict, List, Tuple

from flask import Flask, request, jsonify
from google.cloud import vision
from dotenv import load_dotenv
import PIL.Image

from google import genai
from google.genai import types
# Additional imports for mix-match algorithm
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Validate required environment variables
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
API_KEY = os.getenv("GEMINI_API_KEY")

if not credentials_path or not API_KEY:
    logging.error("Missing required environment variables: GOOGLE_APPLICATION_CREDENTIALS or GEMINI_API_KEY")
    raise EnvironmentError("Missing required environment variables.")

# Initialize Google Cloud Vision & Gemini clients
vision_client = vision.ImageAnnotatorClient()
genai_client = genai.Client(api_key=API_KEY)

# Define clothing categories and their matching relationships
CLOTHING_CATEGORIES = {
    "top": ["shirt", "blouse", "t-shirt", "sweater", "hoodie", "tank top", "polo", "jacket", "blazer", "cardigan"],
    "bottom": ["pants", "jeans", "shorts", "skirt", "trousers", "leggings"],
    "dress": ["dress", "gown", "sundress", "maxi dress"],
}

# Define which categories match with each other
CATEGORY_MATCHING = {
    "top": ["bottom"],
    "bottom": ["top"],
    "dress": [],
}

# Mapping to normalize specific clothing types to general categories
CLOTHING_TYPE_MAPPING = {
    # Tops
    "active shirt": "shirt",
    "athletic shirt": "shirt",
    "polo shirt": "shirt",
    "button-up shirt": "shirt",
    "t-shirt": "shirt",
    "tank top": "shirt",
    "sweater": "sweater",
    "hoodie": "hoodie",
    "blouse": "blouse",
    "jacket": "jacket",
    "blazer": "blazer",
    "cardigan": "cardigan",
    
    # Bottoms
    "jeans": "jeans",
    "denim": "jeans",
    "pants": "pants",
    "trousers": "pants",
    "chinos": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "leggings": "leggings",
    
    # Dresses
    "day dress": "dress",
    "evening dress": "dress",
    "cocktail dress": "dress",
    "maxi dress": "dress",
    "sundress": "dress",
    "gown": "dress",

}


# -----------------------------
# Gemini and Vision Functions
# -----------------------------
def clean_response_text(text: str) -> str:
    """
    Remove markdown code block delimiters (```) from the response text.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def compute_metrics_with_genai(labels: List[Dict[str, Any]], image_content: bytes) -> Dict[str, Any]:
    """
    Uses Gemini to compute sustainability metrics based on the detected labels and image.
    The prompt now also asks for clothing type and material.
    
    Expected JSON output:
      - clothingType: string (e.g., "dress", "shirt")
      - material: string (e.g., "cotton", "polyester")
      - fabricComposition: string description
      - longevityScore: numeric (0–10)
      - co2Consumption: numeric (kg)
      - sustainabilityScore: numeric (0–10)
      - maintenanceTips: list of tips
    """
    prompt = (
        "Based on the following detected fashion labels: "
        f"{json.dumps(labels)}, and using these benchmark guidelines:\n\n"
        "Clothing Type and Category:\n"
        "  - Identify the specific clothing type (e.g., t-shirt, jeans, dress)\n"
        "  - Determine the general clothing category from these options: top, bottom, dress\n\n"
        "Material:\n"
        "  - Infer the primary material or fabric if possible (e.g., cotton, polyester, wool)\n\n"
        "Fabric Composition Score (0–10):\n"
        "  - 8–10 for organic or recycled fibers (e.g., organic cotton)\n"
        "  - 5–7 for natural fibers (e.g., regular cotton, wool)\n"
        "  - 0–4 for synthetics (e.g., polyester)\n\n"
        "Longevity Score (0–10):\n"
        "  - 8–10 for well-constructed, heavy-duty items\n"
        "  - 5–7 for average quality garments\n"
        "  - 0–4 for fast fashion items\n\n"
        "CO₂ Consumption (kg):\n"
        "  - 5–10 kg for eco-friendly production\n"
        "  - 11–20 kg for standard production\n"
        "  - 21+ kg for high-impact manufacturing\n\n"
        "Overall Sustainability Score (0–10):\n"
        "  - Weighted average: 40% fabric, 30% longevity, 30% CO₂ (after normalization)\n\n"
        "Please provide a detailed analysis in JSON format with the following keys:\n"
        "  - clothingType (string, e.g., 't-shirt', 'jeans'),\n"
        "  - clothingCategory (string, one of: top, bottom, dress),\n"
        "  - material (string),\n"
        "  - fabricComposition (string description),\n"
        "  - longevityScore (numeric value 0–10),\n"
        "  - co2Consumption (numeric kg),\n"
        "  - sustainabilityScore (numeric value 0–10),\n"
        "  - maintenanceTips (list of tips).\n"
        "Ensure the output is valid JSON."
    )

    # Prepare image representations for Gemini
    pil_image = PIL.Image.open(BytesIO(image_content))
    
    # Fix: Use the properly imported genai.types namespace
    b64_image = genai.types.Part.from_bytes(
        data=image_content,
        mime_type="image/jpeg"
    )
    
    # Gemini will consider both the text prompt and the image inputs.
    contents = [prompt, pil_image, b64_image]

    response = genai_client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=contents
    )

    logging.info(f"GenAI raw response: {response.text}")
    raw_text = clean_response_text(response.text)
    if not raw_text:
        raise ValueError("Empty response from Gemini API.")

    try:
        metrics = json.loads(raw_text)
    except Exception as e:
        raise ValueError(f"Failed to parse Gemini response: {e}. Raw response: {raw_text}")

    # Normalize the clothing type if it exists in our mapping
    if 'clothingType' in metrics:
        original_type = metrics['clothingType'].lower()
        metrics['clothingTypeOriginal'] = original_type
        metrics['clothingType'] = CLOTHING_TYPE_MAPPING.get(original_type, original_type)

    # If clothing category isn't provided, try to derive it
    if 'clothingCategory' not in metrics and 'clothingType' in metrics:
        normalized_type = metrics['clothingType'].lower()
        for category, types in CLOTHING_CATEGORIES.items():
            if any(t in normalized_type for t in types):
                metrics['clothingCategory'] = category
                break
        # Default to "other" if no category is found
        if 'clothingCategory' not in metrics:
            metrics['clothingCategory'] = "other"

    return metrics

def generate_outfit_ideas(clothing_type: str, clothing_category: str) -> List[Dict[str, str]]:
    """
    Uses Gemini to generate outfit ideas based on the clothing type and category.
    """
    prompt = (
        f"Generate 3 outfit ideas that would go well with a {clothing_type} (category: {clothing_category}).\n\n"
        "For each outfit idea, specify:\n"
        "1. A title for the outfit (e.g., 'Casual Weekend Look', 'Office Chic')\n"
        "2. What specific items would pair well with the {clothing_type}\n"
        "3. A brief description of the overall style\n\n"
        "Return the results as a valid JSON array with each object having the following structure:\n"
        "{\n"
        "  \"title\": \"Outfit title\",\n"
        "  \"items\": [\"item1\", \"item2\", ...],\n"
        "  \"description\": \"Brief style description\"\n"
        "}\n"
    )

    response = genai_client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt
    )

    raw_text = clean_response_text(response.text)
    if not raw_text:
        logging.warning("Empty response from Gemini for outfit ideas")
        return []

    try:
        outfit_ideas = json.loads(raw_text)
        if not isinstance(outfit_ideas, list):
            raise ValueError("Expected a JSON array of outfit ideas")
        return outfit_ideas
    except Exception as e:
        logging.error(f"Failed to parse outfit ideas: {e}. Raw response: {raw_text}")
        return []
    
# -----------------------------
# Load pretrained ResNet model for feature extraction
resnet_model = models.resnet18()
resnet_model.load_state_dict(torch.load("resnet18-f37072fd.pth"))
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove final layer
resnet_model.eval()

def extract_features(image: PIL.Image.Image) -> np.ndarray:
    """
    Extract features from an image using the pretrained ResNet model.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet_model(image_tensor)
    return features.squeeze().numpy()

# Set folder and file names for candidate images
TEST_IMAGES_FOLDER = "test_image"
EMBEDDINGS_FILE = "test_images_embeddings.pkl"

# Load or compute embeddings for all images in test_images folder
if os.path.exists(EMBEDDINGS_FILE):
    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            stored_embeddings, image_paths = pickle.load(f)
        logging.info(f"Loaded {len(image_paths)} embeddings from {EMBEDDINGS_FILE}")
    except Exception as e:
        logging.warning(f"Error loading embeddings: {e}. Recomputing...")
        stored_embeddings, image_paths = [], []
else:
    stored_embeddings, image_paths = [], []

if not image_paths:
    # List all image files in the test_images folder (non-recursive)
    image_paths = [os.path.join(TEST_IMAGES_FOLDER, f) for f in os.listdir(TEST_IMAGES_FOLDER)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for path in image_paths:
        try:
            img = PIL.Image.open(path).convert("RGB")
            feat = extract_features(img)
            stored_embeddings.append(feat)
        except Exception as e:
            logging.warning(f"Error processing {path}: {e}")
    stored_embeddings = np.array(stored_embeddings)
    # Save embeddings for faster loading in future runs
    try:
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump((stored_embeddings, image_paths), f)
        logging.info(f"Saved embeddings to {EMBEDDINGS_FILE}")
    except Exception as e:
        logging.warning(f"Could not save embeddings: {e}")

def get_matching_items(clothing_type: str, clothing_category: str, uploaded_embedding: np.ndarray = None, 
                      top_n: int = 3, similarity_weight: float = 0.3) -> List[Dict[str, Any]]:
    """
    Find matching clothing items that go well with the uploaded item.
    
    This function uses both:
    1. Category-based matching (e.g., tops match with bottoms)
    2. Visual similarity (if embedding is provided)
    
    Returns a list of dictionaries with path and metadata for each matching item.
    """
    # Step 0: Check if category is dress and return empty list
    if clothing_category.lower() == 'dress':
        return []
    
    matching_items = []
    
    # Step 1: Find all items in matching categories
    if clothing_category in CATEGORY_MATCHING:
        matching_categories = CATEGORY_MATCHING[clothing_category]
        category_matches = []
        
        # Use image_paths and stored_embeddings from TEST_IMAGES_FOLDER
        for idx, path in enumerate(image_paths):
            # Get metadata for the image - you may need to adjust how you access metadata
            # Assuming there's a way to get metadata for each path
            item_metadata = {}  # Replace with actual metadata access
            item_category = item_metadata.get('clothingCategory', 'other')
            
            if item_category in matching_categories:
                category_matches.append((idx, path, item_metadata))
        
        # Step 2: If we have an embedding, compute similarity scores
        if uploaded_embedding is not None and len(category_matches) > 0:
            # Extract indices of category matches
            match_indices = [idx for idx, _, _ in category_matches]
            match_embeddings = np.array([stored_embeddings[idx] for idx in match_indices])
            
            # Compute similarity scores
            knn = NearestNeighbors(n_neighbors=min(len(match_embeddings), top_n*2), metric="cosine")
            knn.fit(match_embeddings)
            distances, indices = knn.kneighbors(uploaded_embedding.reshape(1, -1))
            
            # Get the top visually similar items from category matches
            for i in range(min(len(indices[0]), top_n*2)):
                idx = match_indices[indices[0][i]]
                path = image_paths[idx]
                # Get metadata for this path
                metadata = {}  # Replace with actual metadata access
                matching_items.append({
                    'path': path,
                    'metadata': metadata,
                    'matchType': 'visual+category',
                    'similarityScore': float(1.0 - distances[0][i])  # Convert distance to similarity score
                })
        
        # Step 3: If we don't have enough visually similar items, add random category matches
        if len(matching_items) < top_n and category_matches:
            random.shuffle(category_matches)
            for _, path, metadata in category_matches:
                if len(matching_items) >= top_n:
                    break
                    
                # Check if this item is already in the list
                if not any(item['path'] == path for item in matching_items):
                    matching_items.append({
                        'path': path,
                        'metadata': metadata,
                        'matchType': 'category',
                        'similarityScore': 0.0
                    })
    
    # Step 4: If we still don't have enough items, add some random ones
    if len(matching_items) < top_n and image_paths:
        random_indices = random.sample(range(len(image_paths)), min(top_n*2, len(image_paths)))
        for idx in random_indices:
            path = image_paths[idx]
            # Get metadata for this path
            metadata = {}  # Replace with actual metadata access
            
            # Check if this item is already in the list
            if len(matching_items) >= top_n:
                break
            if not any(item['path'] == path for item in matching_items):
                matching_items.append({
                    'path': path,
                    'metadata': metadata,
                    'matchType': 'random',
                    'similarityScore': 0.0
                })
    
    # Return top_n matching items
    return matching_items[:top_n]

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return "Enhanced Google Vision, Gemini & Mix-Match Fashion Backend is running."

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    1. Receives an image file via POST (multipart/form-data).
    2. Uses Google Cloud Vision API for label detection.
    3. Calls Gemini to compute sustainability metrics and additional clothing information.
    4. Uses a mix-match algorithm (via ResNet feature extraction) to find similar items from the test_images folder.
    5. Returns a JSON response containing both the detected labels, Gemini metrics, and matching items.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_content = file.read()
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return jsonify({'error': 'Failed to read image file'}), 500

    # 1. Label detection using Google Cloud Vision
    try:
        image = vision.Image(content=image_content)
        vision_response = vision_client.label_detection(image=image)
        labels = vision_response.label_annotations
    except Exception as e:
        logging.error(f"Google Vision API error: {e}")
        return jsonify({'error': 'Google Vision API error'}), 500

    detected_labels = [
        {'description': label.description, 'score': label.score} for label in labels
    ]

    # 2. Compute sustainability metrics with Gemini
    try:
        metrics = compute_metrics_with_genai(detected_labels, image_content)
    except Exception as e:
        logging.error(f"Gemini call failed: {e}")
        return jsonify({"error": f"Gemini call failed: {str(e)}"}), 500

    # 3. Mix-Match: Extract features from uploaded image and find similar items
    try:
        pil_image = PIL.Image.open(BytesIO(image_content)).convert("RGB")
        uploaded_embedding = extract_features(pil_image)
        matching_items = get_matching_items(
            metrics['clothingType'],
            metrics['clothingCategory'],
            uploaded_embedding=uploaded_embedding,
            top_n=3,
            similarity_weight=0.3
        )
    except Exception as e:
        logging.error(f"Mix-match processing failed: {e}")
        matching_items = []

    # Build the combined response
    response_data = {
        'labels': detected_labels,
        'gemini': {
            'clothingType': metrics.get('clothingType', 'Unknown'),
            'material': metrics.get('material', 'Unknown'),
            'fabricComposition': metrics.get('fabricComposition', 'Unknown'),
            'longevityScore': metrics.get('longevityScore', 0),
            'co2Consumption': metrics.get('co2Consumption', 0),
            'sustainabilityScore': metrics.get('sustainabilityScore', 0),
            'maintenanceTips': metrics.get('maintenanceTips', 'No tips available')
        },
        'matchingItems': matching_items
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

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
import firebase_admin
from firebase_admin import credentials, firestore
# Initialize Firebase Admin SDK
cred = credentials.ApplicationDefault()
# Additional imports for mix-match algorithm

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

default_app = firebase_admin.initialize_app()
db = firestore.client()

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

def load_category_items(category: str) -> List[Dict]:
    """Load items from a category JSON file."""
    try:
        with open(f"categorized_items/{category}_items.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"No items found for category: {category}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error loading {category}_items.json: {e}")
        return []

def get_matching_items(category: str) -> List[Dict]:
    """Get matching items based on the clothing category."""
    matching_items = []
    
    # Get the categories that match with the given category
    matching_categories = CATEGORY_MATCHING.get(category, [])
    
    # For each matching category, load items and select random ones
    for match_category in matching_categories:
        items = load_category_items(match_category)
        if items:
            # Select up to 3 random items
            num_items = min(3, len(items))
            matching_items.extend(random.sample(items, num_items))
    
    return matching_items

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
    # In compute_metrics_with_genai:
    if 'clothingType' in metrics:
        original_type = metrics['clothingType'].lower()
        normalized_type = original_type
        
        # Check each category's types to find the most specific match
        for category, types in CLOTHING_CATEGORIES.items():
            for t in types:
                if t in normalized_type:
                    normalized_type = t
                    break  # Use the first match
            if normalized_type != original_type:
                break
        
        metrics['clothingType'] = normalized_type
        # Determine category based on normalized type
        for category, types in CLOTHING_CATEGORIES.items():
            if any(t == normalized_type for t in types):
                metrics['clothingCategory'] = category
                break
        else:
            metrics['clothingCategory'] = 'other'

    return metrics

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return "Enhanced Google Vision, Gemini & Mix-Match Fashion Backend is running."

ANALYSIS_CACHE = {}

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    1. Receives an image file via POST (multipart/form-data).
    2. Uses Google Cloud Vision API for label detection.
    3. Calls Gemini to compute sustainability metrics and additional clothing information.
    4. Caches the clothing category for later use.
    5. Returns a JSON response containing the detected labels and Gemini metrics.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename

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

    # Store clothing category in cache for the matching endpoint
    clothing_category = metrics.get('clothingCategory', 'Unknown')
    ANALYSIS_CACHE[filename] = clothing_category
    
    # Build the response (without matching items)
    response_data = {
        'clothingType': metrics.get('clothingType', 'Unknown'),
        'clothingCategory': clothing_category,
        'material': metrics.get('material', 'Unknown'),
        'fabricComposition': metrics.get('fabricComposition', 'Unknown'),
        'longevityScore': metrics.get('longevityScore', 0),
        'co2Consumption': metrics.get('co2Consumption', 0),
        'sustainabilityScore': metrics.get('sustainabilityScore', 0),
        'maintenanceTips': metrics.get('maintenanceTips', 'No tips available')

    }

    return jsonify(response_data), 200

@app.route('/matching', methods=['POST'])
def get_matching():
    """
    Uses the cached clothing category to return matching items
    without reprocessing the image.
    """

    # Get item id + userId from request body
    data = request.get_json()
    item_id = data.get('itemId')
    user_id = data.get('userId')

    if not item_id or not user_id:
        return jsonify({'error': 'itemId and userId are required'}), 400
    # Fetch clothingItem data from firebase
    item = db.collection('users').document(user_id).collection('wardrobeItems').document(item_id).get()

    if not item.exists:
        return jsonify({'error': 'Item not found'}), 404
    clothingItem = item.to_dict()

    # Query matching item: if clothingItem is a top, get bottom items, bottom -> top, dress -> none, limit = 3
    clothing_category = clothingItem['clothingCategory']    
    if clothingItem['clothingCategory'] == 'top':
        clothing_category = 'bottom'
    elif clothingItem['clothingCategory'] == 'bottom':
        clothing_category = 'top'
    else:
        return jsonify({ 'matchingItems': [] }), 200
    
    matching_items = db.collection('users').document(user_id).collection('wardrobeItems').where('clothingCategory', '==', clothing_category).limit(3).get()
    
    # Build the matching items response
    response_data = {
        'matchingItems': [item.to_dict() | {'id': item.id} for item in matching_items]
    }

    return jsonify(response_data), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

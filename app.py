import os
import json
import logging
from io import BytesIO

from flask import Flask, request, jsonify
from google.cloud import vision
from dotenv import load_dotenv
import PIL.Image

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
load_dotenv()

credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
API_KEY = os.getenv("GEMINI_API_KEY")

vision_client = vision.ImageAnnotatorClient()

genai_client = genai.Client(api_key=API_KEY)


def compute_metrics_with_genai(labels, image_content):
    """
    Uses Gemini to compute sustainability metrics based solely on the detected labels and image.
    The prompt includes the numerical benchmarks. Modify the prompt here if you wish to update the benchmarks.
    Expected output (as valid JSON):
      - fabricComposition: string description
      - longevityScore: numeric (0-10)
      - co2Consumption: numeric (kg)
      - sustainabilityScore: numeric (0-10)
      - maintenanceTips: list of tips
    """
    prompt = (
        "Based on the following detected fashion labels: "
        f"{json.dumps(labels)}, and using these benchmark guidelines:\n\n"
        "Fabric Composition Score (0–10):\n"
        "  - 8–10 for organic or recycled fibers (e.g. organic cotton)\n"
        "  - 5–7 for natural fibers (e.g. regular cotton, wool)\n"
        "  - 0–4 for synthetics (e.g. polyester)\n\n"
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
        "  - fabricComposition (string description),\n"
        "  - longevityScore (numeric value 0–10),\n"
        "  - co2Consumption (numeric kg),\n"
        "  - sustainabilityScore (numeric value 0–10),\n"
        "  - maintenanceTips (list of tips).\n"
        "Ensure the output is valid JSON."
    )

    # Prepare image representations for Gemini:
    pil_image = PIL.Image.open(BytesIO(image_content))
    b64_image = types.Part.from_bytes(
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

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    if not raw_text:
        raise ValueError("Empty response from Gemini API.")

    try:
        metrics = json.loads(raw_text)
    except Exception as e:
        raise ValueError(f"Failed to parse Gemini response: {e}. Raw response: {raw_text}")

    return metrics


app = Flask(__name__)


@app.route('/')
def index():
    return "Enhanced Google Vision & Gemini Fashion Backend is running."


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    1. Receives an image file via POST (multipart/form-data).
    2. Uses Google Cloud Vision API for label detection.
    3. Calls Gemini to compute sustainability metrics using the prompt that contains benchmark guidelines.
    4. Returns a JSON response containing both the detected labels and generated metrics.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_content = file.read()

    # Label detection using Google Cloud Vision.
    image = vision.Image(content=image_content)
    vision_response = vision_client.label_detection(image=image)
    labels = vision_response.label_annotations

    detected_labels = []
    for label in labels:
        detected_labels.append({
            'description': label.description,
            'score': label.score
        })

    try:
        metrics = compute_metrics_with_genai(detected_labels, image_content)
    except Exception as e:
        return jsonify({"error": f"Gemini call failed: {str(e)}"}), 500

    response_data = {
        'labels': detected_labels,
        'fabricComposition': metrics.get('fabricComposition', 'Unknown'),
        'longevityScore': metrics.get('longevityScore', 0),
        'co2Consumption': metrics.get('co2Consumption', 0),
        'sustainabilityScore': metrics.get('sustainabilityScore', 0),
        'maintenanceTips': metrics.get('maintenanceTips', 'No tips available')
    }

    return jsonify(response_data), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

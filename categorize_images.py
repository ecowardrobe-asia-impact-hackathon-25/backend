import os
import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)



def get_category_from_gemini_response(image_name: str) -> str:
    """Get clothing category from Gemini response file."""
    response_file = f"output/{image_name}_response.json"
    try:
        with open(response_file, 'r') as f:
            data = json.load(f)
            return data['gemini']['clothingCategory']
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        logging.error(f"Error reading category from {response_file}: {e}")
        return None

def categorize_images():
    # Initialize categories
    categories = {
        "top": [],
        "bottom": [],
        "dress": []
    }
    
    # Process each image in test_image folder
    test_image_folder = "test_image"
    for filename in os.listdir(test_image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(test_image_folder, filename)
            base_name = os.path.splitext(filename)[0]  # Get filename without extension
            
            # Get category from Gemini response
            category = get_category_from_gemini_response(base_name)
            
            if category and category in categories:
                categories[category].append({
                    "filename": filename,
                    "path": file_path,
                    "clothingType": base_name.replace('_', ' '),
                    "clothingCategory": category
                })
                logging.info(f"Categorized {filename} as {category}")
            else:
                logging.warning(f"Could not categorize {filename} - invalid or missing category")
    # Save each category to a separate JSON file inside categorized_items folder
    output_dir = "categorized_items"

    # Save each category to a separate JSON file
    for category, items in categories.items():
        output_file = os.path.join(output_dir, f"{category}_items.json")
        with open(output_file, 'w') as f:
            json.dump(items, f, indent=2)
        logging.info(f"Saved {len(items)} {category} items to {output_file}")
        
        # Print the items for debugging
        logging.info(f"Items in {category}:")
        for item in items:
            logging.info(f"  - {item['filename']}")

if __name__ == "__main__":
    categorize_images() 
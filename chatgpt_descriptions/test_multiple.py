import os
import cv2
import mediapipe as mp
import openai
import base64
import json
import re

openai.api_key =  "API_KEY"


def encode_image_to_base64(image_path):
    """
    Encode the image file at the specified path to a base64 string.
    """
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_to_features(image_path, prompt):
    """
    Process the image using OpenAI GPT-4o model to extract specified features into JSON.
    """
    base64_image = encode_image_to_base64(image_path)

    # Send request to GPT-4o
    response = openai.ChatCompletion.create(
        model='gpt-4o',
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
                ]
            }
        ]
    )

    # Extract the response content
    chatgpt_response = response['choices'][0]['message']['content']
    match = re.search(r'{.*}', chatgpt_response, re.DOTALL) 
    if match: 
        return match.group()
    return chatgpt_response

def process_folder_images(folder_path, output_file, prompt):
    """
    Process all images in the folder in order and save their JSON objects to the output file at runtime.
    """
    # Initialize an empty list for storing JSON objects
    all_results = []

    # Get all image files in the folder and sort them
    image_files = sorted(
        [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    # Iterate over sorted image files
    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)
        if os.path.isfile(image_path):
            try:
                # Process the image and get JSON result
                result = process_image_to_features(image_path, prompt)
                result_json = json.loads(result)

                # Add the image file name to the JSON
                result_json["ImageFileName"] = file_name

                # Append the result to the list
                all_results.append(result_json)

                # Save to the JSON file at runtime
                with open(output_file, "w") as file:
                    json.dump(all_results, file, indent=4)
                
                print(f"Processed and saved result for {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                
if __name__ == "__main__":
    # Specify the folder path containing images
    folder_path = "/Users/katherineharvey/Desktop/policesketch/chatgpt_descriptions/original_sketch"

    # Specify the output JSON file
    output_file = "image_analysis.json"

    # Prompt to extract specific features
    prompt = """
                Please analyze the provided image and output a JSON object with the following detailed and specific features:
                {{
                    "Gender": "State the perceived gender in one word (e.g., 'male', 'female', or 'None' if indiscernible).",
                    "ApproximateAge": "Provide a specific age range based on visible features (e.g., '32-35').",
                    "BodyBuild": "Describe the body type concisely but specifically, focusing on proportions and visible traits (e.g., 'lean and athletic', 'broad shoulders and stocky', or 'None' if indiscernible).",
                    "Height": "If the height is discernible, estimate it in one word (e.g., 'tall', 'short', 'average', or 'None' if indiscernible).",
                    "DistinctiveFeatures": "List any unique or standout features in precise detail, such as 'scar running diagonally across left cheek' or 'None' if no distinctive features are visible.",
                    "FaceShape": "Describe the overall face shape clearly, considering jawline, cheekbones, and forehead (e.g., 'oval with defined cheekbones', 'square with a strong jawline').",
                    "HairStyleAndTexture": "Provide a specific description of the hair, including length, texture, and style (e.g., 'short, straight hair parted to the right', 'shoulder-length, wavy hair with a middle part').",
                    "Forehead": "Describe the forehead in detail, focusing on size, shape, and any notable features (e.g., 'broad and smooth with a slight widow's peak', 'narrow and slightly wrinkled').",
                    "Eyebrows": "Describe the eyebrows with attention to their shape, thickness, and prominence (e.g., 'thick and bushy with a natural arch', 'thin and straight with a subtle curve').",
                    "Eyes": "Describe the eyes in terms of shape, size, spacing, and expression (e.g., 'large, almond-shaped eyes set close together', 'deep-set, round eyes with a sharp gaze').",
                    "Nose": "Describe the nose specifically, including its size, shape, and profile (e.g., 'long and narrow with a prominent bridge', 'short and upturned with a wide base').",
                    "Mouth": "Provide details about the mouth, including lip shape, fullness, and expression (e.g., 'full lips with a slight downward curve', 'thin upper lip and fuller lower lip').",
                    "Teeth": "If visible, describe the teeth with attention to alignment, condition, or uniqueness (e.g., 'straight with a small gap in the front', 'slightly crooked with a chipped incisor').",
                    "Jawline": "Describe the jawline in detail, including its definition and shape (e.g., 'sharp and angular with a pronounced chin', 'soft and rounded with minimal definition').",
                    "FacialHair": "Describe any facial hair visible, focusing on its style, length, and placement (e.g., 'short stubble along the jawline', 'none').",
                    "SkinTone": "Indicate the skin tone using general terms and provide additional texture or feature details if visible (e.g., 'light with smooth complexion', 'medium with visible freckles').",
                    "ClothingStyleAndTexture": "Describe the clothing in detail, focusing on style, texture, and visible elements (e.g., 'a collared shirt with fine vertical stripes', 'a knit sweater with a rough texture').",
                    "Accessories": "List any accessories visible, describing them clearly and specifically (e.g., 'thin metal-framed glasses', 'a silver chain around the neck').",
                    "NotableStandoutFeatures": "Describe any other striking features that distinguish the person (e.g., 'a deep scar under the right eye', 'prominent cheekbones that give a gaunt appearance').",
                    "ConfidenceLevel": "Provide a confidence score for the overall description, from 1 (low) to 5 (high), based on the clarity of the image (e.g., '4.5')."
                }}
                Begin output with `{` and end with `}`. Ensure proper JSON syntax.
                """

    try:
        # Process all images in the folder
        process_folder_images(folder_path, output_file, prompt)
        print(f"All image analyses saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

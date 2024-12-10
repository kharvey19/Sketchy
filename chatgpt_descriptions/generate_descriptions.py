import openai
import json
import base64
import os
import tiktoken
from PIL import Image  # Import PIL for image processing

# Your OpenAI API key
openai.api_key = "API_KEY"

input_file = "image_list.txt"
image_folder = "/Users/katherineharvey/Desktop/policesketch/chatgpt_descriptions/test"
output_file = "image_descriptions.json"

# Load existing descriptions if the file exists
if os.path.exists(output_file):
    with open(output_file, 'r') as json_file:
        image_descriptions = json.load(json_file)
else:
    image_descriptions = {}

# Function to preprocess and resize the image using PIL
def preprocess_image(image_path, max_width=256, max_height=256):
    """
    Resize the image to fit within the specified dimensions while maintaining aspect ratio.
    
    Args:
        image_path (str): Path to the original image.
        max_width (int): Maximum width of resized image.
        max_height (int): Maximum height of resized image.

    Returns:
        str: Base64-encoded string of the resized image.
    """
    try:
        with Image.open(image_path) as img:
            # Resize the image while maintaining aspect ratio
            img.thumbnail((max_width, max_height))
            # Save to a temporary file to re-encode it
            temp_path = f"{image_path}_resized.jpg"
            img.save(temp_path, "JPEG")
            # Encode the resized image as base64
            return encode_image_to_base64(temp_path)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to encode an image in base64
def encode_image_to_base64(image_path):
    """
    Encode the image as a base64 string.

    Args:
        image_path (str): Path to the image.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to calculate token count
def calculate_tokens(text, model="gpt-4"):
    """
    Calculate the number of tokens in the input text using tiktoken.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to query ChatGPT for image description
def generate_image_description(image_name, image_data):
    prompt = f"""
    Based on the provided black-and-white image, provide a detailed description of a suspect's appearance. Since the image is black and white, avoid mentioning colors and focus only on visible features, shapes, and textures.
    Only describe features that are clearly visible in the image. If a feature is not visible or indeterminate, state "Not visible."

    Include the following details:
    - Gender
    - Approximate age
    - Body build
    - Height
    - Distinctive features
    - Face shape
    - Hair style and texture
    - Forehead
    - Eyebrows
    - Eyes
    - Nose
    - Mouth
    - Teeth (only if visible)
    - Jawline
    - Facial hair (only if visible)
    - Skin tone (e.g., light, medium, dark) without color descriptors
    - Clothing style and texture (avoid colors)
    - Accessories (e.g., glasses, only if visible)
    - Notable standout features
    - Confidence level in your description

    The image is encoded as a base64 string below:
    {image_data}
    """
    input_tokens = calculate_tokens(prompt, model="gpt-4")
    print(f"Input tokens for {image_name}: {input_tokens}")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates detailed JSON responses based on images."},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract the content from the response
        chatgpt_response = response['choices'][0]['message']['content']
        
        details = json.loads(chatgpt_response)
        output_tokens = calculate_tokens(chatgpt_response, model="gpt-4")
        print(f"Output tokens for {image_name}: {output_tokens}\n")
        print(f"Total tokens (input + output): {input_tokens + output_tokens}\n")

        print("\nRaw details:", details)
        return details
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return {}

# Process all images in the folder
for image_file in sorted(os.listdir(image_folder)):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')) and image_file not in image_descriptions:
        print(f"Processing: {image_file}")

        # Preprocess the image (resize + base64 encode)
        image_path = os.path.join(image_folder, image_file)
        image_data = preprocess_image(image_path)

        if image_data is None:
            print(f"Skipping {image_file} due to preprocessing error.")
            continue

        details = generate_image_description(image_file, image_data)
        
        # Debug: Print details for inspection
        print(f"\nDetails for {image_file}:", details)

        # Add to descriptions dictionary
        if details:
            image_descriptions[image_file] = details

        # Incrementally write to the JSON file
        with open(output_file, 'w') as json_file:
            json.dump(image_descriptions, json_file, indent=4)

print(f"Descriptions saved to {output_file}")

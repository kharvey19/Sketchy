import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def setup_stable_diffusion():
    # Initialize the Stable Diffusion pipeline
    model_id = "CompVis/stable-diffusion-v1-4"  # You can try other models too
    
    # Check if MPS (Metal Performance Shaders) is available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load the pipeline with appropriate settings for Mac
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # MPS works better with float32
        safety_checker=None  # Disable safety checker for faster inference
    )
    
    # Move to appropriate device
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if using MPS
    if device == "mps":
        pipe.enable_attention_slicing()
    
    return pipe

def generate_sketch(pipe, prompt, output_path, num_images=1):
    """
    Generate police sketch-like images using Stable Diffusion
    
    Args:
        pipe: StableDiffusionPipeline instance
        prompt: str, description of the person
        output_path: str, where to save the generated images
        num_images: int, number of images to generate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Enhance prompt to guide towards sketch-like output
    enhanced_prompt = f"black and white police sketch, detailed pencil drawing, portrait of {prompt}"
    
    # Generate images
    images = pipe(
        enhanced_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images
    
    # Save images
    for idx, image in enumerate(images):
        image_path = os.path.join(output_path, f"sketch_{idx}.png")
        image.save(image_path)
        print(f"Saved image to {image_path}")

def main():
    # Setup the pipeline
    pipe = setup_stable_diffusion()
    
    # Example prompts
    test_prompts = [
        "The suspect is described as male around 40-45 years old The face shape is oval with defined cheekbones and the hairstyle is short, straight hair parted to the right with a broad and smooth forehead and thin and straight with a subtle curve eyebrows The eyes are small, round eyes set evenly with a straight and narrow nose a thin lips with a slight upward curve mouth The jawline is rounded with minimal definition with none facial hair The skin tone is light with smooth complexion wearing a collared shirt with a tie accessories include wire-rimmed glasses The confidence level in this description is 4.5",
        "The suspect is described as male around 30-35 years old The face shape is oval with defined cheekbones and the hairstyle is medium-length, straight hair parted slightly off-center with a broad and smooth forehead and thick and slightly curved eyebrows The eyes are medium-sized, almond-shaped eyes with a medium and straight nose a full lips with a neutral expression mouth The jawline is sharp and angular with none facial hair The skin tone is light with smooth complexion wearing collared shirt with fine vertical stripes The confidence level in this description is 4.5",
    ]
    
    # Generate images for each prompt
    output_dir = "generated_sketches"
    for idx, prompt in enumerate(test_prompts):
        print(f"\nGenerating sketch for prompt {idx + 1}:")
        print(prompt)
        generate_sketch(pipe, prompt, output_dir, num_images=2)

if __name__ == "__main__":
    main() 
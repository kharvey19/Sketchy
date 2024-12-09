import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def setup_stable_diffusion():
    # Initialize the Stable Diffusion pipeline
    model_id = "runwayml/stable-diffusion-v1-5"  # Updated to v1-5
    
    # Check if MPS (Metal Performance Shaders) is available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the pipeline with appropriate settings for Mac
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # MPS works better with float32
        safety_checker=None  # Disable safety checker for faster inference
    )
    
    # Move to appropriate device
    pipe = pipe.to(device)
    
    # Enable optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    
    return pipe

def generate_sketch(pipe, prompt, output_path, num_images=4):
    """
    Generate police sketch-like images using Stable Diffusion
    
    Args:
        pipe: StableDiffusionPipeline instance
        prompt: str, description of the person
        output_path: str, where to save the generated images
        num_images: int, number of images to generate (default: 4)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Enhance prompt to guide towards sketch-like output
    enhanced_prompt = (
        "highly detailed police sketch, black and white pencil drawing, "
        "professional forensic artist style, detailed shading, "
        f"portrait of {prompt}"
    )
    
    # Generate images
    print(f"Generating {num_images} sketches...")
    images = pipe(
        enhanced_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images
    
    # Save images
    for idx, image in enumerate(images):
        image_path = os.path.join(output_path, f"sketch_{idx + 1}.png")
        image.save(image_path)
        print(f"Saved image to {image_path}")

def main():
    # Setup the pipeline
    pipe = setup_stable_diffusion()
    
    # Same prompt in two different formats (they produce identical results):
    
    # Format 1: Single line (might be harder to read in code)
    test_prompt = "The suspect is described as male around 40-45 years old The face shape is oval with defined cheekbones and the hairstyle is short, straight hair parted to the right with a broad and smooth forehead and thin and straight with a subtle curve eyebrows The eyes are small, round eyes set evenly with a straight and narrow nose a thin lips with a slight upward curve mouth The jawline is rounded with minimal definition with none facial hair The skin tone is light with smooth complexion wearing a collared shirt with a tie accessories include wire-rimmed glasses"
    
    # We'll use the single-line version for this test
    # Generate images
    output_dir = "generated_sketches"
    print("\nGenerating sketches for prompt:")
    print(test_prompt)
    generate_sketch(pipe, test_prompt, output_dir, num_images=4)

if __name__ == "__main__":
    main() 
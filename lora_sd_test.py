import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

class LoRALinear(torch.nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=8):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Create LoRA matrices on the same device as the linear layer
        device = linear_layer.weight.device
        self.lora_A = torch.nn.Parameter(torch.zeros(self.in_features, rank).to(device))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, self.out_features).to(device))
        self.linear_layer = linear_layer
        
        # Initialize LoRA matrices
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if x.device != self.lora_A.device:
            x = x.to(self.lora_A.device)
        base_output = self.linear_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output

def add_lora_layers(unet, rank=4, alpha=8):
    """Add LoRA layers to the UNet's attention layers"""
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear) and any(x in name for x in ['to_k', 'to_q', 'to_v', 'to_out']):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = unet.get_submodule(parent_name)
            
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            print(f"Added LoRA to {name}")
    
    return unet

def setup_stable_diffusion():
    # Check if MPS is available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)
    
    # Add LoRA layers and load fine-tuned weights
    pipe.unet = add_lora_layers(pipe.unet, rank=4, alpha=8)
    pipe.unet = pipe.unet.to(device)
    
    checkpoint = torch.load('sd_checkpoint_epoch_20.pt', map_location=device)
    pipe.unet.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Enable optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    
    return pipe

def generate_sketch(pipe, prompt, output_path, num_images=4):
    """
    Generate police sketch-like images using fine-tuned Stable Diffusion
    
    Args:
        pipe: StableDiffusionPipeline instance
        prompt: str, description of the person
        output_path: str, where to save the generated images
        num_images: int, number of images to generate (default: 4)
    """
    os.makedirs(output_path, exist_ok=True)
    
    enhanced_prompt = (
        "highly detailed police sketch, black and white pencil drawing, "
        "professional forensic artist style, detailed shading, "
        f"portrait of {prompt}"
    )
    
    print(f"Generating {num_images} sketches...")
    images = pipe(
        enhanced_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images
    
    for idx, image in enumerate(images):
        image_path = os.path.join(output_path, f"sketch_lora_{idx + 1}.png")
        image.save(image_path)
        print(f"Saved image to {image_path}")

def main():
    # Setup the pipeline with fine-tuned model
    pipe = setup_stable_diffusion()
    
    # Use the same test prompt as basic_sd_test.py
    test_prompt = "The suspect is described as male around 40-45 years old The face shape is oval with defined cheekbones and the hairstyle is short, straight hair parted to the right with a broad and smooth forehead and thin and straight with a subtle curve eyebrows The eyes are small, round eyes set evenly with a straight and narrow nose a thin lips with a slight upward curve mouth The jawline is rounded with minimal definition with none facial hair The skin tone is light with smooth complexion wearing a collared shirt with a tie accessories include wire-rimmed glasses"
    
    # Generate images
    output_dir = "generated_sketches_lora"
    print("\nGenerating sketches for prompt:")
    print(test_prompt)
    generate_sketch(pipe, test_prompt, output_dir, num_images=4)

if __name__ == "__main__":
    main() 
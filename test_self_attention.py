import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from pathlib import Path

class LoRALinear(torch.nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=32):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Create LoRA matrices on the same device as linear layer
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

def add_lora_layers(unet, attention_type='self_only'):
    """Add LoRA layers to specific attention mechanisms"""
    modified_layers = []
    
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear) and any(x in name for x in ['to_k', 'to_q', 'to_v', 'to_out']):
            # For self_only, only modify attn1 layers
            if attention_type == 'self_only' and 'attn2' in name:
                continue
                
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = unet.get_submodule(parent_name)
            
            lora_layer = LoRALinear(module)
            setattr(parent, child_name, lora_layer)
            modified_layers.append(name)
            print(f"Added LoRA to {name}")
    
    print(f"\nModified {len(modified_layers)} layers for {attention_type} attention")
    return unet

def setup_pipeline():
    """Initialize and set up the Stable Diffusion pipeline"""
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
    
    # Add LoRA layers
    pipe.unet = add_lora_layers(pipe.unet, attention_type='self_only')
    pipe.unet = pipe.unet.to(device)
    
    # Load the fine-tuned weights
    checkpoint_path = Path("ablation_study_self_only/checkpoints/checkpoint_epoch_20_self_only.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pipe.unet.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Enable optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    
    return pipe

def generate_samples(pipe, prompt, output_dir="generated_samples_self_only", num_images=4):
    """Generate and save sample images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced prompt engineering
    enhanced_prompt = (
        "highly detailed police sketch, black and white pencil drawing, "
        "professional forensic artist style, realistic shading and proportions, "
        "clear facial features, detailed line work, "
        f"{prompt}"
    )
    
    # Generate images with improved parameters
    print(f"Generating {num_images} samples...")
    images = pipe(
        enhanced_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=30,
        guidance_scale=5.0,
        width=512,
        height=512,
        generator=torch.manual_seed(42)
    ).images
    
    # Save images
    for idx, image in enumerate(images):
        image_path = os.path.join(output_dir, f"self_only_sample_{idx + 1}.png")
        image.save(image_path)
        print(f"Saved image to {image_path}")

def main():
    # Set up the pipeline
    pipe = setup_pipeline()
    
    # Test prompt
    test_prompt = "A person with small, round eyes set evenly with a straight and narrow nose. " \
                 "The mouth has thin lips with a slight upward curve. " \
                 "The jawline is rounded with minimal definition and no facial hair. " \
                 "The skin tone is light with a smooth complexion. " \
                 "They are wearing a collared shirt with a tie. " \
                 "Accessories include wire-rimmed glasses."
    
    # Generate samples
    generate_samples(pipe, test_prompt)

if __name__ == "__main__":
    main() 
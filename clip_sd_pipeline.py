import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import torch.nn.functional as F
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r, lora_alpha, lora_dropout):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_A = nn.Parameter(torch.zeros(original_linear.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, original_linear.out_features))
        self.scaling = lora_alpha / r

    def forward(self, x):
        result = self.original_linear(x)
        if self.r > 0:
            lora_result = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
            return result + lora_result
        return result

def apply_lora_to_model(model, r=16, lora_alpha=32, lora_dropout=0.1):
    """Apply LoRA to the model's attention layers"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(substr in name for substr in ['q_proj', 'v_proj', 'k_proj']):
            print(f"Applying LoRA to {name}")
            parent = model
            for name_part in name.split('.')[:-1]:
                parent = getattr(parent, name_part)
            layer_name = name.split('.')[-1]
            setattr(parent, layer_name, LoRALinear(module, r, lora_alpha, lora_dropout))

def load_fine_tuned_clip(checkpoint_path, device):
    """Load the fine-tuned CLIP model"""
    # Initialize the base model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Apply LoRA modifications
    apply_lora_to_model(model)
    
    # Load the fine-tuned weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    
    return model, processor

def get_clip_embedding(model, processor, text, image_path=None, device='mps'):
    """Get embeddings from the fine-tuned CLIP model"""
    inputs = processor(
        text=text,
        images=Image.open(image_path).convert('RGB') if image_path else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Get text embeddings and normalize
        text_embedding = F.normalize(outputs.text_embeds, p=2, dim=-1)
        
        if image_path:
            # Get image embeddings and normalize
            image_embedding = F.normalize(outputs.image_embeds, p=2, dim=-1)
            # Average the embeddings (you could try other combination strategies)
            embedding = (text_embedding + image_embedding) / 2
        else:
            embedding = text_embedding
            
    return embedding

def setup_stable_diffusion(device):
    """Initialize the Stable Diffusion pipeline"""
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None
    )
    
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    
    return pipe

def generate_sketches_with_clip_embedding(
    clip_model,
    clip_processor,
    sd_pipeline,
    text_prompt,
    image_path=None,
    num_images=4,
    output_dir="generated_sketches_clip",
    device='mps'
):
    """Generate sketches using CLIP embeddings as conditioning"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get CLIP embedding
    embedding = get_clip_embedding(
        clip_model,
        clip_processor,
        text_prompt,
        image_path,
        device
    )
    
    # Enhance prompt for sketch generation
    enhanced_prompt = (
        "highly detailed police sketch, black and white pencil drawing, "
        "professional forensic artist style, detailed shading, "
        f"portrait of {text_prompt}"
    )
    
    # Generate images using the embedding as conditioning
    images = sd_pipeline(
        prompt=enhanced_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images
    
    # Save images
    for idx, image in enumerate(images):
        image_path = os.path.join(output_dir, f"sketch_clip_{idx + 1}.png")
        image.save(image_path)
        print(f"Saved image to {image_path}")

def main():
    # Set up device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load fine-tuned CLIP model
    clip_model, clip_processor = load_fine_tuned_clip('clip_checkpoint_epoch_20.pt', device)
    
    # Set up Stable Diffusion
    sd_pipeline = setup_stable_diffusion(device)
    
    # Load example prompt and corresponding image
    with open('data/descriptions/00001.txt', 'r') as f:
        text_prompt = f.read().strip()
    image_path = 'data/sketches/00001.jpg'
    
    # Generate sketches
    print("\nGenerating sketches with CLIP embeddings...")
    generate_sketches_with_clip_embedding(
        clip_model,
        clip_processor,
        sd_pipeline,
        text_prompt,
        image_path,
        num_images=4,
        device=device
    )

if __name__ == "__main__":
    main() 
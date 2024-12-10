import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import numpy as np
from contextlib import nullcontext

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=32):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Create LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        # Store the original layer
        self.linear_layer = linear_layer
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original transformation + LoRA path
        base_output = self.linear_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output

class SketchDataset(Dataset):
    def __init__(self, image_dir, caption_dir, tokenizer, image_size=512):
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        
        # Get sorted lists of files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.caption_files = sorted([f for f in os.listdir(caption_dir) if f.endswith('.txt')])
        
        # Ensure matching pairs
        self.valid_pairs = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            caption_file = f"{base_name}.txt"
            if caption_file in self.caption_files:
                self.valid_pairs.append((img_file, caption_file))
        
        print(f"Found {len(self.valid_pairs)} valid image-caption pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_file, caption_file = self.valid_pairs[idx]
        
        # Load and process image
        image = Image.open(os.path.join(self.image_dir, img_file)).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image)
        
        # Convert to float and normalize to [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        
        # Add alpha channel (fully opaque)
        alpha = np.ones((self.image_size, self.image_size, 1), dtype=np.float32)
        image_rgba = np.concatenate([image, alpha], axis=2)
        
        # Convert to torch tensor and rearrange dimensions to [C, H, W]
        image = torch.from_numpy(image_rgba).permute(2, 0, 1)
        
        # Load and process caption
        with open(os.path.join(self.caption_dir, caption_file), 'r') as f:
            caption = f.read().strip()
        
        # Enhance caption for sketch generation
        enhanced_caption = (
            "highly detailed police sketch, black and white pencil drawing, "
            "professional forensic artist style, detailed shading, "
            f"portrait of {caption}"
        )
        
        # Tokenize caption
        tokens = self.tokenizer(
            enhanced_caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0]
        }

def add_lora_layers(unet, rank=4, alpha=32):
    """Add LoRA layers to the UNet's attention layers"""
    lora_layers = []
    
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and any(x in name for x in ['to_k', 'to_q', 'to_v', 'to_out']):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = unet.get_submodule(parent_name)
            
            # Create and attach LoRA layer
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            lora_layers.append(lora_layer)
            print(f"Added LoRA to {name}")
    
    # Freeze original parameters
    for param in unet.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for layer in lora_layers:
        layer.lora_A.requires_grad = True
        layer.lora_B.requires_grad = True
    
    return unet

def plot_metrics(metrics, save_dir="training_plots"):
    """Plot training metrics"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot training loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch_train_losses'], label='Training Loss', marker='o')
    plt.plot(metrics['epoch_val_losses'], label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'sd_loss_epochs_{timestamp}.png'))
    plt.close()
    
    # Plot step-wise training loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['step_losses'], label='Training Loss', alpha=0.5)
    plt.title('Training Loss Over Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'sd_loss_steps_{timestamp}.png'))
    plt.close()

def main():
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model paths
    model_id = "runwayml/stable-diffusion-v1-5"
    pretrained_model_name_or_path = "openai/clip-vit-large-patch14"
    
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path)
    
    print("Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path).to(device)
    text_encoder.requires_grad_(False)  # Freeze text encoder
    
    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float32,  # Use full precision for MPS
        use_auth_token=True
    )
    
    # Enable memory efficient attention
    print("Enabling memory efficient attention...")
    from diffusers.models.attention_processor import AttnProcessor2_0
    unet.set_attn_processor(AttnProcessor2_0())
    
    # Add LoRA layers to UNet
    print("Adding LoRA layers...")
    unet = add_lora_layers(unet, rank=4, alpha=8)  # Further reduced rank for memory
    unet = unet.to(device)
    
    # Create dataset and dataloaders
    print("Setting up dataset...")
    dataset = SketchDataset(
        image_dir="data/sketches",
        caption_dir="data/descriptions",
        tokenizer=tokenizer,
        image_size=64  # Reduced from 128 to 64 for memory efficiency
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # Keep batch size at 1
        shuffle=True,
        num_workers=0,
        pin_memory=False  # Disable pin_memory for MPS
    )
    
    # Create validation dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Setup optimizer with reduced learning rate
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=5e-5,  # Reduced learning rate
        weight_decay=0.01,
        eps=1e-7
    )
    
    # Training loop
    num_epochs = 20
    metrics = {
        'step_losses': [],
        'epoch_train_losses': [],
        'epoch_val_losses': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            if device == 'mps':
                torch.mps.empty_cache()
            
            try:
                # Move batch to device and keep in full precision
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                input_ids = batch["input_ids"].to(device)
                
                # Get text embeddings
                with torch.no_grad():
                    text_embeddings = text_encoder(input_ids)[0]
                
                # Forward pass with memory-efficient processing
                noise = torch.randn_like(pixel_values)
                timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device)
                noisy_images = noise + timesteps.reshape(-1, 1, 1, 1) * pixel_values
                
                # Compute loss with explicit memory management
                optimizer.zero_grad(set_to_none=True)
                
                # Process in smaller chunks if needed
                with torch.amp.autocast(device_type='mps', dtype=torch.float32) if device == 'mps' else nullcontext():
                    noise_pred = unet(
                        noisy_images,
                        timesteps,
                        encoder_hidden_states=text_embeddings
                    ).sample
                    
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update metrics
                step_loss = loss.item()
                epoch_loss += step_loss
                metrics['step_losses'].append(step_loss)
                progress_bar.set_postfix({"loss": step_loss})
                
                if device == 'mps':
                    torch.mps.empty_cache()
                
            except RuntimeError as e:
                print(f"Error during training step {step}: {str(e)}")
                if "out of memory" in str(e):
                    if device == 'mps':
                        torch.mps.empty_cache()
                    continue
                else:
                    raise e
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        metrics['epoch_train_losses'].append(avg_train_loss)
        
        # Validation phase
        unet.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                if device == 'mps':
                    torch.mps.empty_cache()
                
                try:
                    # Move batch to device
                    pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                    input_ids = batch["input_ids"].to(device)
                    
                    # Get text embeddings
                    text_embeddings = text_encoder(input_ids)[0]
                    
                    # Forward pass
                    noise = torch.randn_like(pixel_values)
                    timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device)
                    noisy_images = noise + timesteps.reshape(-1, 1, 1, 1) * pixel_values
                    
                    with torch.amp.autocast(device_type='mps', dtype=torch.float32) if device == 'mps' else nullcontext():
                        noise_pred = unet(
                            noisy_images,
                            timesteps,
                            encoder_hidden_states=text_embeddings
                        ).sample
                        
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')
                    
                    val_loss += loss.item()
                    
                    if device == 'mps':
                        torch.mps.empty_cache()
                        
                except RuntimeError as e:
                    print(f"Error during validation: {str(e)}")
                    if "out of memory" in str(e):
                        if device == 'mps':
                            torch.mps.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_val_loss = val_loss / len(val_dataloader)
        metrics['epoch_val_losses'].append(avg_val_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Plot current progress
        plot_metrics(metrics)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, f'sd_checkpoint_epoch_{epoch + 1}.pt')
            
            if device == 'mps':
                torch.mps.empty_cache()

if __name__ == "__main__":
    main() 
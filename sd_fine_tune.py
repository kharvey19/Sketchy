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
import seaborn as sns
from datetime import datetime
from PIL import Image
import numpy as np
from contextlib import nullcontext
import json
from pathlib import Path

# Ablation configurations
ABLATION_CONFIGS = [
    {
        'name': 'self_only',
        'attention_type': 'self_only',
        'description': 'LoRA applied only to self-attention layers'
    },
    {
        'name': 'cross_only',
        'attention_type': 'cross_only',
        'description': 'LoRA applied only to cross-attention layers'
    },
    {
        'name': 'both',
        'attention_type': 'both',
        'description': 'LoRA applied to both self and cross-attention layers'
    }
]

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=32):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Create LoRA matrices on the same device as linear layer
        device = linear_layer.weight.device
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank).to(device))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features).to(device))
        self.linear_layer = linear_layer
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if x.device != self.lora_A.device:
            x = x.to(self.lora_A.device)
        base_output = self.linear_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output

def is_attention_layer(name, attention_type):
    """Determine if a layer should be modified based on attention type"""
    if attention_type == 'self_only':
        return 'attn1' in name
    elif attention_type == 'cross_only':
        return 'attn2' in name
    else:  # 'both'
        return 'attn1' in name or 'attn2' in name

def add_lora_layers(unet, rank=4, alpha=32, attention_type='both'):
    """
    Add LoRA layers to specific attention mechanisms
    
    Args:
        unet: The UNet model
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        attention_type: 'self_only', 'cross_only', or 'both'
    """
    lora_layers = []
    modified_layers = []
    
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and any(x in name for x in ['to_k', 'to_q', 'to_v', 'to_out']):
            if not is_attention_layer(name, attention_type):
                continue
                
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = unet.get_submodule(parent_name)
            
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            lora_layers.append(lora_layer)
            modified_layers.append(name)
            print(f"Added LoRA to {name}")
    
    # Freeze all parameters
    for param in unet.parameters():
        param.requires_grad = False
    
    # Unfreeze only LoRA parameters
    for layer in lora_layers:
        layer.lora_A.requires_grad = True
        layer.lora_B.requires_grad = True
    
    print(f"\nModified {len(modified_layers)} layers for {attention_type} attention")
    return unet, modified_layers

def setup_output_dirs(config):
    """Setup directory structure for ablation study outputs"""
    base_dir = Path(f"ablation_study_{config['name']}")
    dirs = {
        'checkpoints': base_dir / 'checkpoints',
        'plots': base_dir / 'plots',
        'samples': base_dir / 'samples',
        'metrics': base_dir / 'metrics'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def save_config_info(config, modified_layers, dirs):
    """Save configuration and layer modification information"""
    info = {
        'config': config,
        'modified_layers': modified_layers,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open(dirs['metrics'] / 'config_info.json', 'w') as f:
        json.dump(info, f, indent=2)

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

def plot_metrics(metrics, save_dir, config_name):
    """Plot training metrics with enhanced visualizations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Training and Validation Loss
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(metrics['epoch_train_losses']) + 1)
    
    plt.plot(epochs, metrics['epoch_train_losses'], 'b-', label='Training Loss', marker='o')
    plt.plot(epochs, metrics['epoch_val_losses'], 'r-', label='Validation Loss', marker='o')
    
    plt.fill_between(epochs, metrics['epoch_train_losses'], metrics['epoch_val_losses'],
                    alpha=0.1, color='gray', label='Generalization Gap')
    
    plt.title(f'Training Progress - {config_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add convergence indicators
    if len(epochs) > 5:
        recent_train_change = abs(metrics['epoch_train_losses'][-1] - 
                                metrics['epoch_train_losses'][-5]) / metrics['epoch_train_losses'][-5]
        recent_val_change = abs(metrics['epoch_val_losses'][-1] - 
                              metrics['epoch_val_losses'][-5]) / metrics['epoch_val_losses'][-5]
        
        status_text = f"Recent Changes:\nTrain: {recent_train_change:.1%}\nVal: {recent_val_change:.1%}"
        plt.text(0.02, 0.98, status_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(save_dir / f'loss_epochs_{timestamp}.png')
    plt.close()
    
    # Plot 2: Step-wise Loss with Moving Average
    plt.figure(figsize=(12, 6))
    window = 50  # Window size for moving average
    steps = range(len(metrics['step_losses']))
    
    # Calculate moving average
    if len(metrics['step_losses']) > window:
        moving_avg = np.convolve(metrics['step_losses'], 
                               np.ones(window)/window, 
                               mode='valid')
        
        plt.plot(steps, metrics['step_losses'], 'b-', alpha=0.3, label='Raw Loss')
        plt.plot(range(window-1, len(steps)), moving_avg, 'r-', 
                label=f'Moving Average (window={window})', linewidth=2)
    else:
        plt.plot(steps, metrics['step_losses'], 'b-', label='Loss')
    
    plt.title(f'Training Stability - {config_name}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(save_dir / f'loss_steps_{timestamp}.png')
    plt.close()

def plot_attention_analysis(metrics_list, save_dir):
    """Plot comparative analysis across different attention configurations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Loss Comparison Across Configurations
    plt.figure(figsize=(15, 8))
    for config, metrics in metrics_list:
        plt.plot(metrics['epoch_train_losses'], 
                label=f'{config["name"]} - Train',
                marker='o')
        plt.plot(metrics['epoch_val_losses'], 
                label=f'{config["name"]} - Val',
                marker='s',
                linestyle='--')
    
    plt.title('Loss Comparison Across Attention Configurations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / f'attention_comparison_{timestamp}.png')
    plt.close()
    
    # Plot 2: Final Performance Comparison
    plt.figure(figsize=(12, 6))
    configs = [config["name"] for config, _ in metrics_list]
    train_losses = [metrics['epoch_train_losses'][-1] for _, metrics in metrics_list]
    val_losses = [metrics['epoch_val_losses'][-1] for _, metrics in metrics_list]
    
    x = np.arange(len(configs))
    width = 0.35
    
    plt.bar(x - width/2, train_losses, width, label='Final Training Loss')
    plt.bar(x + width/2, val_losses, width, label='Final Validation Loss')
    
    plt.title('Final Performance Comparison')
    plt.xlabel('Attention Configuration')
    plt.ylabel('Loss')
    plt.xticks(x, configs)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(save_dir / f'final_comparison_{timestamp}.png')
    plt.close()

def generate_samples(pipe, prompt, save_dir, config_name, num_samples=4):
    """Generate and save sample images"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Enhance prompt for sketch generation
    enhanced_prompt = (
        "highly detailed police sketch, black and white pencil drawing, "
        "professional forensic artist style, detailed shading, "
        f"portrait of {prompt}"
    )
    
    # Generate images
    images = pipe(
        enhanced_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images
    
    # Save individual images
    image_paths = []
    for idx, image in enumerate(images):
        image_path = save_dir / f'sample_{config_name}_{idx + 1}_{timestamp}.png'
        image.save(image_path)
        image_paths.append(image_path)
    
    # Create comparison grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    grid_image = Image.new('RGB', 
                          (grid_size * images[0].width, grid_size * images[0].height),
                          'white')
    
    for idx, image in enumerate(images):
        x = (idx % grid_size) * images[0].width
        y = (idx // grid_size) * images[0].height
        grid_image.paste(image, (x, y))
    
    grid_image.save(save_dir / f'grid_{config_name}_{timestamp}.png')
    
    return image_paths

def train_model(config, device, image_size=64, num_epochs=20):
    """Train model for a specific attention configuration"""
    print(f"\nStarting training for {config['name']} configuration...")
    print(f"Attention type: {config['attention_type']}")
    
    # Setup directories
    dirs = setup_output_dirs(config)
    
    # Model setup
    model_id = "runwayml/stable-diffusion-v1-5"
    pretrained_model_name_or_path = "openai/clip-vit-large-patch14"
    
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path)
    
    print("Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path).to(device)
    text_encoder.requires_grad_(False)
    
    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float32,
        use_auth_token=True
    )
    
    # Enable memory efficient attention
    print("Enabling memory efficient attention...")
    from diffusers.models.attention_processor import AttnProcessor2_0
    unet.set_attn_processor(AttnProcessor2_0())
    
    # Add LoRA layers to UNet with specific attention configuration
    print("Adding LoRA layers...")
    unet, modified_layers = add_lora_layers(
        unet,
        rank=4,
        alpha=2,  # Further reduced alpha for more stable training
        attention_type='cross_only'  # Default to cross-attention only
    )
    unet = unet.to(device)
    
    # Save configuration info
    save_config_info(config, modified_layers, dirs)
    
    # Create dataset and dataloaders
    print("Setting up dataset...")
    dataset = SketchDataset(
        image_dir="data/sketches",
        caption_dir="data/descriptions",
        tokenizer=tokenizer,
        image_size=image_size
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
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Setup optimizer with even lower learning rate
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=5e-6,  # Further reduced learning rate
        weight_decay=0.01,
        eps=1e-7
    )
    
    # Add learning rate scheduler with warmup
    from transformers import get_cosine_schedule_with_warmup
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = len(train_dataloader)  # 1 epoch of warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    metrics = {
        'step_losses': [],
        'epoch_train_losses': [],
        'epoch_val_losses': [],
        'grad_norms': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Loss threshold for early intervention
    loss_spike_threshold = 1.0
    grad_norm_threshold = 1.0
    
    # Keep track of best epoch model
    best_epoch_model = None
    best_epoch = 0

    # Initialize pipeline for sanity checks
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)
    
    test_prompt = "A detailed police sketch of a middle-aged man with sharp features"
    
    # Save initial model state
    initial_model_state = {
        name: param.clone().detach()
        for name, param in unet.named_parameters()
        if param.requires_grad
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
                # Move batch to device
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                input_ids = batch["input_ids"].to(device)
                
                # Get text embeddings
                with torch.no_grad():
                    text_embeddings = text_encoder(input_ids)[0]
                
                # Forward pass with reduced noise scale
                noise = torch.randn_like(pixel_values) * 0.7  # Further reduced noise scale
                timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device)
                noisy_images = noise + timesteps.reshape(-1, 1, 1, 1) * pixel_values
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type='mps', dtype=torch.float32) if device == 'mps' else nullcontext():
                    noise_pred = unet(
                        noisy_images,
                        timesteps,
                        encoder_hidden_states=text_embeddings
                    ).sample
                    
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')
                
                # Check for loss spikes
                if loss.item() > loss_spike_threshold:
                    print(f"\nWarning: Loss spike detected ({loss.item():.4f})")
                    continue
                
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=grad_norm_threshold)
                metrics['grad_norms'].append(grad_norm.item())
                
                # Skip step if gradient norm is too high
                if grad_norm.item() > grad_norm_threshold:
                    print(f"\nWarning: High gradient norm detected ({grad_norm.item():.4f})")
                    continue
                
                optimizer.step()
                scheduler.step()
                
                # Track learning rate
                metrics['learning_rates'].append(scheduler.get_last_lr()[0])
                
                step_loss = loss.item()
                epoch_loss += step_loss
                metrics['step_losses'].append(step_loss)
                progress_bar.set_postfix({
                    "loss": step_loss,
                    "grad_norm": grad_norm.item(),
                    "lr": scheduler.get_last_lr()[0]
                })
                
                # Check for parameter divergence
                for name, param in unet.named_parameters():
                    if param.requires_grad:
                        diff = torch.abs(param - initial_model_state[name]).mean()
                        if diff > 1.0:  # Threshold for parameter divergence
                            print(f"\nWarning: Large parameter divergence detected in {name}")
                
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
                    pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                    input_ids = batch["input_ids"].to(device)
                    text_embeddings = text_encoder(input_ids)[0]
                    
                    noise = torch.randn_like(pixel_values) * 0.7  # Reduced noise scale
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
        print(f"Average Gradient Norm: {sum(metrics['grad_norms'][-len(train_dataloader):]) / len(train_dataloader):.4f}")
        
        # Generate sample images after first epoch and when saving checkpoints
        if epoch == 0 or (epoch + 1) % 2 == 0:  # Increased frequency of sample generation
            print(f"\nGenerating sample images after epoch {epoch + 1}...")
            with torch.no_grad():
                images = pipe(
                    test_prompt,
                    num_images_per_prompt=2,
                    num_inference_steps=30,
                    guidance_scale=7.0
                ).images
                
                for idx, image in enumerate(images):
                    image.save(dirs['samples'] / f'epoch_{epoch + 1}_sample_{idx + 1}.png')
                    
                # If images look good, save as best epoch model
                if epoch == 0:  # Save first epoch model as reference
                    best_epoch_model = {
                        name: param.clone().detach()
                        for name, param in unet.named_parameters()
                        if param.requires_grad
                    }
                    best_epoch = 0
        
        # Plot current progress
        plot_metrics(metrics, dirs['plots'], config['name'])
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
                'metrics': metrics
            }
            torch.save(checkpoint, dirs['checkpoints'] / f'best_model_{config["name"]}.pt')
        else:
            patience_counter += 1
        
        # Early stopping check with model reversion
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print("Reverting to best epoch model...")
            # Revert to best epoch model
            for name, param in unet.named_parameters():
                if param.requires_grad:
                    param.data = best_epoch_model[name].clone()
            break
        
        # Regular checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
                'metrics': metrics
            }
            torch.save(checkpoint, dirs['checkpoints'] / f'checkpoint_epoch_{epoch + 1}_{config["name"]}.pt')
    
    # Save final plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(metrics['epoch_train_losses'], label='Train Loss')
    plt.plot(metrics['epoch_val_losses'], label='Val Loss')
    plt.title('Loss Progress')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics['grad_norms'], label='Gradient Norms')
    plt.title('Gradient Norms')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(dirs['plots'] / 'final_training_metrics.png')
    plt.close()
    
    return metrics, dirs

def main():
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run ablation study for each configuration
    all_metrics = []
    all_dirs = []
    
    for config in ABLATION_CONFIGS:
        metrics, dirs = train_model(config, device)
        all_metrics.append((config, metrics))
        all_dirs.append(dirs)
    
    # Create comparative analysis plots
    analysis_dir = Path("ablation_analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Plot comparative analysis
    plot_attention_analysis(all_metrics, analysis_dir)
    
    # Generate sample images for each configuration
    test_prompt = "The suspect is described as male around 40-45 years old with defined cheekbones"
    
    for config, dirs in zip(ABLATION_CONFIGS, all_dirs):
        # Load best model for this configuration
        checkpoint = torch.load(dirs['checkpoints'] / f'best_model_{config["name"]}.pt')
        
        # Initialize pipeline with best model
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None
        ).to(device)
        
        # Add LoRA layers and load weights
        pipe.unet = add_lora_layers(pipe.unet, attention_type=config['attention_type'])[0]
        pipe.unet.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate samples
        generate_samples(pipe, test_prompt, dirs['samples'], config['name'])
    
    print("\nAblation study complete! Check the following directories for results:")
    for config, dirs in zip(ABLATION_CONFIGS, all_dirs):
        print(f"\n{config['name']}:")
        print(f"- Checkpoints: {dirs['checkpoints']}")
        print(f"- Plots: {dirs['plots']}")
        print(f"- Samples: {dirs['samples']}")
    print(f"\nComparative Analysis: {analysis_dir}")

if __name__ == "__main__":
    main() 
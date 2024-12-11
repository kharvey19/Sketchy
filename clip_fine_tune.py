import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import seaborn as sns
from datetime import datetime

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

class SketchCaptionDataset(Dataset):
    def __init__(self, sketches_dir, captions_dir, processor):
        self.sketches_dir = sketches_dir
        self.captions_dir = captions_dir
        self.processor = processor

        # Collect file identifiers
        sketch_files = sorted([f.split('.')[0] for f in os.listdir(sketches_dir) if f.endswith('.jpg')])
        caption_files = sorted([f.split('.')[0] for f in os.listdir(captions_dir) if f.endswith('.txt')])

        # Synchronize the two lists
        valid_sketch_files = []
        valid_caption_files = []
        i, j = 0, 0

        while i < len(sketch_files) and j < len(caption_files):
            if sketch_files[i] == caption_files[j]:
                valid_sketch_files.append(sketch_files[i])
                valid_caption_files.append(caption_files[j])
                i += 1
                j += 1
            elif int(sketch_files[i]) < int(caption_files[j]):
                i += 1
            else:
                j += 1

        print(f"Found {len(valid_sketch_files)} valid sketch-caption pairs")
        
        self.valid_image_paths = [os.path.join(sketches_dir, f"{file_id}.jpg") for file_id in valid_sketch_files]
        self.valid_caption_paths = [os.path.join(captions_dir, f"{file_id}.txt") for file_id in valid_caption_files]

    def __len__(self):
        return len(self.valid_image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.valid_image_paths[idx]).convert('RGB')
        with open(self.valid_caption_paths[idx], 'r', encoding='utf-8') as f:
            caption = f.read().strip()

        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=77
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r, lora_alpha, lora_dropout):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # Initialize with smaller values for better training stability
        self.lora_A = nn.Parameter(torch.zeros(original_linear.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, original_linear.out_features))
        
        # Initialize using kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Initialize B to zeros for better stability
        
        self.scaling = lora_alpha / r

    def forward(self, x):
        result = self.original_linear(x)
        if self.r > 0:
            lora_result = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
            return result + lora_result
        return result

def apply_lora_to_model(model, r, lora_alpha, lora_dropout):
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(substr in name for substr in ['q_proj', 'v_proj', 'k_proj']):
            print(f"Applying LoRA to {name}")
            lora_layer = LoRALinear(module, r, lora_alpha, lora_dropout)
            names = name.split('.')
            parent = model
            for n in names[:-1]:
                parent = getattr(parent, n)
            setattr(parent, names[-1], lora_layer)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
    return lora_params

def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    
    logits = torch.matmul(image_embeds, text_embeds.t()) / temperature
    labels = torch.arange(image_embeds.size(0)).to(image_embeds.device)
    
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2

def evaluate(model, dataloader, k_values=[1, 5, 10, 25]):
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() 
                     if k in ["input_ids", "pixel_values", "attention_mask"]}
            outputs = model(**inputs)
            
            image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)
            
            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)
    
    image_embeds = torch.cat(all_image_embeds)
    text_embeds = torch.cat(all_text_embeds)
    
    similarity = torch.matmul(text_embeds, image_embeds.t())
    ground_truth = torch.arange(len(similarity)).to(device)
    
    results = {}
    for k in k_values:
        _, indices = similarity.topk(k=k, dim=-1)
        correct = (indices == ground_truth.unsqueeze(1)).any(dim=1).float().mean()
        results[f"top_{k}"] = correct.item()
    
    return results

def plot_training_metrics(metrics, save_dir="training_plots"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_losses'], label='Training Loss', marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'training_loss_{timestamp}.png'))
    plt.close()
    
    # Plot validation accuracies
    plt.figure(figsize=(12, 6))
    for k in metrics['val_accuracies'][0].keys():
        values = [epoch_metrics[k] for epoch_metrics in metrics['val_accuracies']]
        plt.plot(values, label=f'Top-{k.split("_")[1]} Accuracy', marker='o')
    plt.title('Validation Accuracies Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'validation_accuracies_{timestamp}.png'))
    plt.close()

def main():
    # Data directories
    data_dir = "./data"
    sketches_dir = os.path.join(data_dir, "sketches")
    descriptions_dir = os.path.join(data_dir, "descriptions")
    
    # Initialize CLIP
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Set up device (Apple Metal)
    global device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA and move model to device
    lora_params = apply_lora_to_model(model, r=16, lora_alpha=32, lora_dropout=0.1)
    model = model.to(device)
    
    # Prepare dataset
    dataset = SketchCaptionDataset(sketches_dir, descriptions_dir, processor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders with larger batch size for Apple Metal
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(lora_params, lr=2e-4, weight_decay=0.01)
    
    # Training loop
    num_epochs = 20
    metrics = {'train_losses': [], 'val_accuracies': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() 
                     if k in ["input_ids", "pixel_values", "attention_mask"]}
            
            outputs = model(**inputs)
            loss = contrastive_loss(outputs.image_embeds, outputs.text_embeds)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        metrics['train_losses'].append(avg_epoch_loss)
        
        # Evaluate
        val_metrics = evaluate(model, val_dataloader)
        metrics['val_accuracies'].append(val_metrics)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print("Validation Metrics:", val_metrics)
        
        # Plot current progress
        plot_training_metrics(metrics)
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f'clip_checkpoint_epoch_{epoch + 1}.pt')

if __name__ == "__main__":
    main()

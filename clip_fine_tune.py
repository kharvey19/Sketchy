import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import CLIPProcessor, CLIPModel
from dataset import SketchCaptionDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn
from lora_layer import LoRALinear

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize CLIP processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Function to apply LoRA to the CLIP model
def apply_lora_to_model(model, r, lora_alpha, lora_dropout):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in ['q_proj', 'k_proj', 'v_proj']):
            # Split the name to find the parent module
            names = name.split('.')
            parent = model
            for n in names[:-1]:  # Traverse to the parent module
                parent = getattr(parent, n)
            # Replace the linear layer with a LoRALinear layer
            setattr(parent, names[-1], LoRALinear(module, r, lora_alpha, lora_dropout))

# Apply LoRA to the model
apply_lora_to_model(model, r=8, lora_alpha=32, lora_dropout=0.1)

# Move model to device after modifying
model = model.to(device)

# Initialize dataset
dataset = SketchCaptionDataset(
    sketches_dir='data/sketch/sketch/',
    captions_dir='data/text/celeba-caption/',
    processor=processor
)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define contrastive loss function
def contrastive_loss(image_embeds, text_embeds, temperature):
    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Compute logits
    logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * temperature
    logits_per_text = logits_per_image.t()

    # Labels
    batch_size = image_embeds.size(0)
    labels = torch.arange(batch_size).to(device)

    # Compute loss
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2

    return loss

# Define evaluation function
def evaluate(model, dataloader):
    model.eval()
    image_embeddings = []
    text_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "pixel_values", "attention_mask"]
            }
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            image_embeddings.append(image_embeds)
            text_embeddings.append(text_embeds)

    # Concatenate all embeddings
    image_embeddings = torch.cat(image_embeddings)
    text_embeddings = torch.cat(text_embeddings)

    # Compute similarity matrix
    similarity = text_embeddings @ image_embeddings.t()

    # Compute retrieval metrics
    ground_truth = torch.arange(len(similarity)).to(device)
    _, indices = similarity.topk(k=1, dim=-1)
    correct = (indices.squeeze() == ground_truth).float().sum()
    accuracy = correct / len(similarity)

    print(f"Validation Top-1 Accuracy: {accuracy.item():.4f}")

    model.train()
    return accuracy

# Define optimizer
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# Number of epochs
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for batch in progress_bar:
        optimizer.zero_grad()

        # Move inputs to device
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in ["input_ids", "pixel_values", "attention_mask"]
        }

        # Forward pass
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        temperature = model.logit_scale.exp()

        # Compute loss
        loss = contrastive_loss(image_embeds, text_embeds, temperature)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update progress bar and epoch loss
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")

    # Evaluate the model
    evaluate(model, val_dataloader)

# Save the fine-tuned model and processor
model.save_pretrained('fine_tuned_clip_model')
processor.save_pretrained('fine_tuned_clip_processor')

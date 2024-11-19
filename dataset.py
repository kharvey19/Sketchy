import os
from torch.utils.data import Dataset
from PIL import Image

class SketchCaptionDataset(Dataset):
    # Initialize dataset with paths to sketches, captions, and CLIP processor
    def __init__(self, sketches_dir, captions_dir, processor):
        self.sketches_dir = sketches_dir
        self.captions_dir = captions_dir
        self.processor = processor

        # List all the sketch files
        self.sketch_files = sorted([f for f in os.listdir(sketches_dir) if f.endswith('.jpg')])

        # List all the caption files
        self.caption_files = sorted([f for f in os.listdir(captions_dir) if f.endswith('.txt')])

    # Returns total number of samples (30,000)
    def __len__(self):
        return len(self.sketch_files)
    
    # Loads sketch image and corresponding caption
    # Combines all sentences in caption file into one description
    # Uses CLIP processor to tokenize and preprocess inputs
    def __getitem__(self, idx):
        # Get sketch file name
        sketch_file = self.sketch_files[idx]
        sketch_path = os.path.join(self.sketches_dir, sketch_file)

        # Corresponding caption file
        caption_file = self.caption_files[idx]
        caption_path = os.path.join(self.captions_dir, caption_file)

        # Load image
        image = Image.open(sketch_path).convert('RGB')

        # Load and preprocess text
        with open(caption_path, 'r') as f:
            captions = f.readlines()

            # Combine multiple sentences into one description
            description = ' '.join([caption.strip() for caption in captions])

        # Preprocess inputs
        inputs = self.processor(text=description, images=image, return_tensors="pt", padding='max_length', truncation=True, max_length=77)

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs
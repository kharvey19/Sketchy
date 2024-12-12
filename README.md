# Police Sketch Generation Project

⚠️ **IMPORTANT NOTE**: This project has already been run through completion, and all outputs are included in the repository. Rerunning the entire pipeline will take an extremely long time (multiple hours) and significant computational resources. The results of all studies, fine-tuning, and tests are already available in their respective directories.

This project implements a machine learning pipeline for generating police sketches from textual descriptions using fine-tuned Stable Diffusion and CLIP models.

## Step-by-Step Guide

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Preparation
1. Download the CUHK Face Sketch FERET Dataset:
```bash
python download_data.py
```
This will download the dataset to the `data/` directory.

2. Generate text descriptions using OpenAI's API:
```bash
cd chatgpt_descriptions
python gen_descriptions.py
```
This script uses OpenAI's GPT model to generate detailed descriptions for each sketch. The descriptions are saved in `data/descriptions/`.

Note: These steps have already been completed, and the data is included in the repository.

### Step 3: Ablation Study
```bash
cd ..
python sd_fine_tune.py
```
This script performs an ablation study on different LoRA configurations:
- Self-attention only (`ablation_study_self_only/`)
- Cross-attention only (`ablation_study_cross_only/`)
- Both attention types (`ablation_study_both/`)

**Key Finding**: The study concluded that applying LoRA to both self- and cross-attention layers produces the most sketch-like and consistent results. This configuration was used for the final model.

Note: This study has been completed, and results are available in the respective directories.

### Step 4: CLIP Fine-tuning
```bash
python clip_fine_tune.py
```
Fine-tunes the CLIP model for better text-image alignment specific to police sketches. The final checkpoint is available at `clip_checkpoint_epoch_20.pt`.

### Step 5: Model Validation
1. Test base Stable Diffusion:
```bash
python basic_sd_test.py
```
Generates sample images using the base model for comparison.

2. Test integrated pipeline:
```bash
python clip_sd_pipeline.py
```
Tests the combination of fine-tuned CLIP and Stable Diffusion models.

### Step 6: Testing
Run the test suite in the `tests/` directory:

Each test is provided as a standalone Jupyter Notebook. To run a test, simply execute the cells in the corresponding notebook:

1. `img2clip_finetuned.ipynb`: Executes the fine-tuned CLIP-to-img2img Stable Diffusion pipeline on a single image.
2. `img2imgclip.ipynb`: Executes the CLIP-to-img2img Stable Diffusion pipeline on a single image.
3. `img2imgtest.ipynb`: Executes the baseline img2img Stable Diffusion pipeline on a single image.
4. `iteration_img2clip.ipynb`: Runs the CLIP-to-img2img Stable Diffusion pipeline with iterative steps and plots metrics (SSIM, PSNR, CLIP Score, LPIPS) across iterations.
5. `iteration_stable.ipynb`: Runs the baseline img2img Stable Diffusion pipeline with iterative steps and plots metrics across iterations.
6. `iteration_finetuned_img2clip.ipynb`: Runs the fine-tuned CLIP-to-img2img Stable Diffusion pipeline with iterative steps and plots metrics across iterations.
7. `final_metrics`: Runs all iterative tests and computes performance comparisons across models
## Unused Development Files

The following files were created during initial development but were not used in the final implementation:

### Web Application (Unused)
- `app/` directory: Contains a Flask web application that was initially planned for deployment
- `run.py`: Web application entry point
- `tables.py`: Database schema definitions

### Additional Test Files (Unused)
- `test_self_attention.py`: Standalone test for self-attention mechanism

## Project Structure

```
.
├── app/                        # Unused web application
├── data/                      # Dataset and descriptions
│   ├── sketches/             # CUHK Face Sketch FERET Dataset
│   └── descriptions/         # Generated text descriptions
├── tests/                    # Test files
├── ablation_study_*/         # Ablation study results
├── generated_sketches/       # Model outputs
└── requirements.txt         # Project dependencies
```

## Results and Outputs

- Ablation study results: `ablation_study_*/`
- Generated samples: `generated_sketches/`
- Training metrics: `training_plots/`
- CLIP checkpoint: `clip_checkpoint_epoch_20.pt`

## Requirements

See `requirements.txt` for complete list of dependencies. Key requirements:
- PyTorch
- Diffusers
- Transformers
- OpenAI API (for description generation)
- NumPy
- Matplotlib


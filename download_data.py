import kaggle

# Download fine-tune dataset
kaggle.api.dataset_download_files("kashyapkvh/mm-celeba-hq-dataset", path=".", unzip=True)

print("Dataset downloaded to the 'data/' directory.")
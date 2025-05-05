# Standard library imports
import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytz

# Local application imports
from src.model import UNET
from utils.utils import (
    get_loaders_with_augmentation,
    Config,
    inspect_pixel_value_range,
    save_ssl_predictions_as_imgs,
    load_trained_model,
    calculate_dice_score,
    print_current_lr,
    check_dataloader_sizes,
    review_batch,
)

""" Training Hyperparameters """
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 1
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True

""" Load Pre-trained Model"""
PATH_TO_TRAINED_MODEL = 'results/ssl_trained_model/2025-04-19_01-19-21/self_supervised_learning.pt'

""" Load Dataset"""
TRAIN_IMG_DIR = "../ISIC_2017_dataset/data/train_images/"
VAL_IMG_DIR = "../ISIC_2017_dataset/data/val_images/" # not used in feature extraction

""" Saving Directory"""
FEATURE_SPACE_DIRECTORY = "results/ssl_features_space"

""" Feature Extraction Function """
# Fix: Remove the unnecessary optimizer and loss_fn parameters
def extract_features_fn(loader, model, device):
    model.eval()
    all_bottlenecks = []
    file_names = []  # Store file names to verify correct order

    loop = tqdm(loader)
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(device)
            
            # Extract bottleneck features only
            _, bottleneck = model(data)
            
            # Store bottlenecks (move to CPU)
            all_bottlenecks.append(bottleneck.detach().cpu())
            
            # Store file names if available in your dataloader
            if hasattr(loader.dataset, 'img_list'):
                start_idx = batch_idx * loader.batch_size
                end_idx = min(start_idx + loader.batch_size, len(loader.dataset))
                batch_files = [os.path.basename(loader.dataset.img_list[i]) for i in range(start_idx, end_idx)]
                file_names.extend(batch_files)
            
            # Update tqdm loop
            loop.set_postfix(batch=batch_idx)
            
            # Free GPU memory
            del data, bottleneck
            torch.cuda.empty_cache()
    
    # Concatenate all bottlenecks
    all_bottlenecks = torch.cat(all_bottlenecks, dim=0)
    
    # Return both bottleneck features and file names
    return all_bottlenecks, file_names

""" Setup Directories """
def setup_directories():
    """Create and return directories for saving features"""
    timenow = datetime.strftime(datetime.now(pytz.timezone('America/New_York')), '%Y-%m-%d_%H-%M-%S')
    features_space_path = os.path.join(FEATURE_SPACE_DIRECTORY, timenow)
    os.makedirs(features_space_path, exist_ok=True)
    print("The features will be saved at:", features_space_path)
    return features_space_path

""" Transformations """
def create_transforms():
    """Create and return transforms (no augmentation for feature extraction)"""
    transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2(),
    ])
    
    return transforms

""" Main Function """
def main():
    torch.cuda.empty_cache()  # Clear any GPU leftovers
    
    # Setup phase
    features_space_path = setup_directories()
    transforms = create_transforms()
    
    # Model initialization
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    # Load pre-trained model
    print("\nLoading pre-trained model from:", PATH_TO_TRAINED_MODEL)
    checkpoint = torch.load(PATH_TO_TRAINED_MODEL, map_location=DEVICE, weights_only=False)
    
    # The model was saved with the key 'state_dict' based on the provided code
    print("Available keys in checkpoint:", checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    print("Model loaded successfully using 'state_dict' key")
    
    # Configure data loader (without augmentation for feature extraction)
    config = Config(
        flip_rate=0.0,  # No augmentation for feature extraction
        local_rate=0.0,
        nonlinear_rate=0.0, 
        paint_rate=0.0,
        inpaint_rate=0.0
    )

    # Setup data loaders for train and val sets - ensure no shuffling to maintain order
    train_loader, _ = get_loaders_with_augmentation(
        TRAIN_IMG_DIR,
        VAL_IMG_DIR,
        BATCH_SIZE,
        transforms,
        transforms,  # Same transform for both
        num_workers=1,  # Single worker to ensure consistent file order
        pin_memory=PIN_MEMORY,
        config=config,
        shuffle_train=False,  # Important: Don't shuffle to maintain directory order
    )
    
    print("Data loaders prepared, starting feature extraction...")
    
    # Extract features
    model.to(DEVICE)
    # Fix: Call with the correct number of arguments
    features, file_names = extract_features_fn(train_loader, model, DEVICE)
    
    # Save extracted features
    feature_file_path = os.path.join(features_space_path, "features_space.pt")
    torch.save(features, feature_file_path)
    print(f"Features extracted and saved to {feature_file_path}")
    print(f"Feature tensor shape: {features.shape}")
    
    # Also save the file names to ensure correct mapping
    file_names_path = os.path.join(features_space_path, "file_names.txt")
    with open(file_names_path, 'w') as f:
        for name in file_names:
            f.write(f"{name}\n")
    print(f"File names saved to {file_names_path}")

    # Verify we have the expected number of samples
    expected_samples = 1600  # As specified, we expect 1600 training images
    actual_samples = features.shape[0]
    print(f"Expected number of samples: {expected_samples}")
    print(f"Actual number of samples extracted: {actual_samples}")
    if expected_samples != actual_samples:
        print("Warning: The number of extracted features doesn't match the expected count!")
    
    # Verify file names were collected correctly
    if not file_names:
        print("Warning: No file names were collected. Check if the dataset has an 'img_list' attribute.")

if __name__ == "__main__":
    main()
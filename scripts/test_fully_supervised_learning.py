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
    data_loader_for_fully_supervised_learning,
    Config,
    inspect_pixel_value_range,
    save_ssl_predictions_as_imgs,
    save_fss_predictions_images,
    load_trained_model,
    calculate_dice_score,
    print_current_lr,
    check_dataloader_sizes,
    review_batch,
)

""" Testing Configuration """
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True

""" Load Pre-trained Model"""
PATH_TO_TRAINED_MODEL = "results/fully_supervised_trained_model/2025-04-20_01-36-15/fully_supervised_learning.pt"

""" Load Dataset"""
TRAIN_IMG_DIR = "../ISIC_2017_dataset/data/train_images/"
TRAIN_MASK_DIR = "../ISIC_2017_dataset/data/train_masks/" 
VAL_IMG_DIR = "../ISIC_2017_dataset/data/val_images/"
VAL_MASK_DIR = "../ISIC_2017_dataset/data/val_masks/" 
TEST_IMG_DIR = "../ISIC_2017_dataset/data/test_images/"
TEST_MASK_DIR = "../ISIC_2017_dataset/data/test_masks/" 

""" Saving Directory"""
IMAGE_DIRECTORY = "results/fully_supervised_test_images"

""" Test Function """
def test_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device) # [-1,1]
        targets = targets.float().to(device) # {0,1}

        # forward
        predictions, _ = model(data)
        predictions = torch.sigmoid(predictions)  # Convert logits to probabilities [0,1]        
        loss = loss_fn(predictions, targets) # [0,1], {0,1}

        # After calculating loss, convert to binary for metrics
        predictions = (predictions > 0.5).float()  # Convert to binary {0, 1} for metrics/visualization
       
        # Calculate metrics
        total_loss += loss.item()
        preds = predictions.float()
        total_dice += calculate_dice_score(preds, targets) # {0,1}, {0,1}
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())        

        # Free GPU memory
        del data, targets, predictions
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice


""" Main Function """
def main():
    torch.cuda.empty_cache()  # Optional: clear any GPU leftovers from earlier runs
    
    # Setup phase
    image_path = setup_directories()
    train_transform, val_transforms, test_transforms = create_transforms()
    config = setup_augmentation_config()
    
    # Model initialization
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn =  nn.BCELoss()
    
    # Load pre-trained model
    print("\nLoading pre-trained model from:", PATH_TO_TRAINED_MODEL)
    checkpoint = torch.load(PATH_TO_TRAINED_MODEL, map_location=DEVICE, weights_only=False)
    
    # The model was saved with the key 'state_dict' based on the provided code
    print("Available keys in checkpoint:", checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    print("Model loaded successfully using 'state_dict' key")
    model.eval()  # Set model to evaluation mode

    # Data loading
    _, _, test_loader = load_and_review_data(
        train_transform, val_transforms, test_transforms, config
    )

    # Run inference on test set
    print("\nRunning inference on test set...")
    test_loss, test_dice = test_fn(test_loader, model, loss_fn, DEVICE)
    print(f"\nTest Loss: {test_loss:.4f}   Test Dice Score: {test_dice:.4f}")

    # Save test results
    results_file = os.path.join(image_path, "test_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}   Test Dice Score: {test_dice:.4f}\n")
        f.write(f"Model Path: {PATH_TO_TRAINED_MODEL}\n")
        f.write(f"Test Date: {datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Test results saved to {results_file}")
    
    # Save sample predictions as images
    print("\nSaving sample predictions...")
    save_fss_predictions_images(test_loader, model, folder=image_path, device=DEVICE)
    
    print("\nTesting completed successfully.")

""" Setup Directories """
def setup_directories():
    """Create and return directories for saving test results"""
    timenow = datetime.strftime(datetime.now(pytz.timezone('America/New_York')), '%Y-%m-%d_%H-%M-%S')
    output_path = os.path.join(IMAGE_DIRECTORY, timenow)
    os.makedirs(output_path, exist_ok=True)
    print("Test results will be saved at:", output_path)
    return output_path

""" Transformations """
def create_transforms():
    """Create and return training and validation transforms"""
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        ToTensorV2(),
    ])
    
    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2(),
    ])

    test_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2(),
    ])
    
    return train_transform, val_transforms, test_transforms

""" Augmentations """
def setup_augmentation_config():
    """Set up configuration for self-supervised augmentations"""
    return Config(
        flip_rate=0.5,
        local_rate=0.4,
        nonlinear_rate=0.6, 
        paint_rate=0.7,
        inpaint_rate=0.5
    )

""" Load and Review Data """
def load_and_review_data(train_transform, val_transforms, test_transforms, config):
    """Load and review training and validation data"""
    train_loader, val_loader, test_loader = data_loader_for_fully_supervised_learning(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms, # no augmentation. only resizing
        test_transforms, # no augmentation. only resizing
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        config=config,
        shuffle_train=True # shuffle training dataset
    )
    print("train_loader and val_loarder is completed. \n")
    
    # Pixel value range
    inspect_pixel_value_range(test_loader, "Test Loader")
    
    # Review training and validation data
    print("\nReviewing test data")
    check_dataloader_sizes(test_loader)
    review_batch(test_loader, 'test data')
    
    return train_loader, val_loader, test_loader





if __name__ == "__main__":
    main()
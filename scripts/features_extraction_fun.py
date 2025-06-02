# Standard library imports
import os
import sys
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_features(
    path_to_trained_model,
    feature_space_directory,
    train_img_dir,
    valid_img_dir, 
    iter,
    image_height=256,
    image_width=256,
    batch_size=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    pin_memory=True,
    
):
    """
    Extract features from a trained U-NET model and save them to a specified directory.
    
    Args:
        path_to_trained_model (str): Path to the trained model weights
        feature_space_directory (str): Base directory to save extracted features
        train_img_dir (str): Directory containing the training images
        image_height (int): Height to resize images to
        image_width (int): Width to resize images to
        batch_size (int): Batch size for data loading
        device (str): Device to use for computation ('cuda' or 'cpu')
        pin_memory (bool): Whether to pin memory for faster data transfer
        
    Returns:
        tuple: (feature_file_path, file_names_path, txt_file_path) - Paths to the saved features, CSV filenames, and TXT filenames
    """
    # Import these here to avoid circular imports
    from src.model import UNET
    from utils.utils import get_loaders_with_augmentation, Config
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Setup directories
    timenow = datetime.strftime(datetime.now(pytz.timezone('America/New_York')), '%Y-%m-%d_%H-%M-%S')
    # features_space_path = os.path.join(feature_space_directory, timenow)
    features_space_path = feature_space_directory
    os.makedirs(features_space_path, exist_ok=True)
    print("The features will be saved at:", features_space_path)
    
    # Create transforms (no augmentation for feature extraction)
    transforms = A.Compose([
        A.Resize(height=image_height, width=image_width),
        ToTensorV2(),
    ])
    
    # Initialize model
    model = UNET(in_channels=3, out_channels=1).to(device)
    
    # Load pre-trained model
    print("\nLoading pre-trained model from:", path_to_trained_model)
    checkpoint = torch.load(path_to_trained_model, map_location=device, weights_only=False)
    
    # Check checkpoint keys
    print("Available keys in checkpoint:", checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    print("Model loaded successfully using 'state_dict' key")
    
    # Configure data loader (without augmentation)
    config = Config(
        flip_rate=0.0,
        local_rate=0.0,
        nonlinear_rate=0.0,
        paint_rate=0.0,
        inpaint_rate=0.0
    )
    
    # Setup data loader - ensure no shuffling to maintain order
    train_loader, _ = get_loaders_with_augmentation(
        train_img_dir,
        valid_img_dir,  # No validation needed
        batch_size,
        transforms,
        transforms,
        num_workers=1,
        pin_memory=pin_memory,
        config=config,
        shuffle_train=False,
    )
    
    print("Data loader prepared, starting feature extraction...")
    
    # Extract features
    features, file_names = _extract_features_fn(train_loader, model, device)
    
    # Save extracted features
    feature_file_path = os.path.join(features_space_path, f"features_space.pt")
    torch.save(features, feature_file_path)
    print(f"Features extracted and saved to {feature_file_path}")
    print(f"Feature tensor shape: {features.shape}")
    
    # Save file names for reference - CSV with column header
    file_names_path = os.path.join(features_space_path, f"file_names.csv")
    with open(file_names_path, 'w') as f:
        # Write the column header first
        f.write("filename\n")
        # Write each filename on a new line
        for name in file_names:
            f.write(f"{name}\n")
    print(f"File names saved to {file_names_path}")
    
    # Save file names as plain text file without header
    txt_file_path = os.path.join(features_space_path, f"file_names.txt")
    with open(txt_file_path, 'w') as f:
        # Write each filename on a new line without any header
        for name in file_names:
            f.write(f"{name}\n")
    print(f"File names also saved as plain text to {txt_file_path}")
    
    # Return the paths to the saved files
    return feature_file_path, file_names_path, txt_file_path

def _extract_features_fn(loader, model, device):
    """
    Extract bottleneck features from the model for all images in the loader.
    
    Args:
        loader (DataLoader): Data loader with images
        model (nn.Module): The model to extract features from
        device (str): Device to use for computation
        
    Returns:
        tuple: (features tensor, list of file names)
    """
    model.eval()
    all_bottlenecks = []
    file_names = []
    
    loop = tqdm(loader)
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(device)
            
            # Extract bottleneck features only
            _, bottleneck = model(data)
            
            # Store bottlenecks (move to CPU)
            all_bottlenecks.append(bottleneck.detach().cpu())
            
            # Store file names if available
            if hasattr(loader.dataset, 'img_list'):
                start_idx = batch_idx * loader.batch_size
                end_idx = min(start_idx + loader.batch_size, len(loader.dataset))
                batch_files = [os.path.basename(loader.dataset.img_list[i]) for i in range(start_idx, end_idx)]
                file_names.extend(batch_files)
            
            # Update progress bar
            loop.set_postfix(batch=batch_idx)
            
            # Free GPU memory
            del data, bottleneck
            torch.cuda.empty_cache()
    
    # Concatenate all bottlenecks
    all_bottlenecks = torch.cat(all_bottlenecks, dim=0)
    
    return all_bottlenecks, file_names

if __name__ == "__main__":
    """ Load Pre-trained Model"""
    PATH_TO_TRAINED_MODEL = 'results/ssl_trained_model/2025-04-19_01-19-21/self_supervised_learning.pt'

    """ Load Dataset"""
    TRAIN_IMG_DIR = "../ISIC_2017_dataset/data/train_images/"
    VAL_IMG_DIR = "../ISIC_2017_dataset/data/val_images/" # not used in feature extraction

    """ Saving Directory"""
    FEATURE_SPACE_DIRECTORY = "results/ssl_features_space"
    
    """ Iteration """
    iteration_num=1
    # Updated to capture the third return value
    feature_file_path, file_names_path, txt_file_path = extract_features(
        path_to_trained_model=PATH_TO_TRAINED_MODEL,
        feature_space_directory=FEATURE_SPACE_DIRECTORY,
        train_img_dir=TRAIN_IMG_DIR,
        valid_img_dir=VAL_IMG_DIR,
        iter=iteration_num  # Pass iteration_num explicitly
    )
    
    print(f"Feature extraction completed.")
    print(f"Features saved to: {feature_file_path}")
    print(f"CSV file names saved to: {file_names_path}")
    print(f"TXT file names saved to: {txt_file_path}")
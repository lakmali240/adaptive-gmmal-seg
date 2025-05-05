# Standard library imports
import os
import sys
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import pandas as pd
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
    data_loader_for_self_supervised_assisted_active_learning,
    Config,
    inspect_pixel_value_range,
    save_ssl_predictions_as_imgs,
    save_fss_predictions_images,
    save_ssaal_predictions_images,
    load_trained_model,
    load_trained_model_without_scheduler,
    load_trained_model_without_scheduler_and_output_layer,
    calculate_dice_score,
    print_current_lr,
    check_dataloader_sizes,
    review_batch,
)

from loss_function import CombinedLoss

""" Training Hyperparameters """
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
START_EPOCH = 1
NUM_EPOCHS = 20
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True

""" Early Stopping Conditions """
EARLY_STOPPING_EPOCHES = 15
EXPECTED_BEST_LOSS = 0.05

""" Load Self-Supervised Trained Model"""
LOAD_TRAINED_MODEL = False
PATH_TO_TRAINED_MODEL = 'results/ssl_trained_model/2025-04-19_01-19-21/self_supervised_learning.pt'

""" Load Dataset"""
TRAIN_IMG_DIR = "../ISIC_2017_dataset/data/train_images/"
TRAIN_MASK_DIR = "../ISIC_2017_dataset/data/train_masks/" 
VAL_IMG_DIR = "../ISIC_2017_dataset/data/val_images/"
VAL_MASK_DIR = "../ISIC_2017_dataset/data/val_masks/" 
TEST_IMG_DIR = "../ISIC_2017_dataset/data/test_images/"
TEST_MASK_DIR = "../ISIC_2017_dataset/data/test_masks/" 

""" Saving Directory"""
MODEL_DIRECTORY = "results/ssaal_trained_model"
IMAGE_VAL_DIRECTORY = "results/ssaal_validation_images"
IMAGE_TEST_DIRECTORY = "results/ssaal_test_images"

""" Training """
def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_dice = 0
    bottlenecks = []  # Collect bottlenecks across batches
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device) # [-1,1]
        targets = targets.float().to(device) # {0,1}

        # forward
        predictions, bottleneck = model(data)
        predictions = torch.sigmoid(predictions)  # Convert logits to probabilities [0, 1]           
        loss = loss_fn(predictions, targets) # [0,1], {0,1}

        # After calculating loss, convert to binary for metrics
        predictions = (predictions > 0.5).float()  # Convert to binary {0, 1} for metrics/visualization

        # backward - standard approach without scaler
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        total_loss += loss.item()
        preds = predictions.float()
        total_dice += calculate_dice_score(preds, targets)

        # Store bottlenecks (move to CPU)
        bottlenecks.append(bottleneck.detach().cpu())

        # update tqdm loop
        loop.set_postfix(loss=loss.item())        

        # Free GPU memory
        del data, targets, predictions, bottleneck
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice, bottlenecks

""" Validation """
def validate_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    loop = tqdm(loader)  # Add progress bar for validation
    with torch.no_grad():
        for idx, (data, targets) in enumerate(loop):
            data = data.to(device) # [-1,1]
            targets = targets.float().to(device) # {0,1}
            
            predictions, bottleneck = model(data) 
            predictions = torch.sigmoid(predictions)  # Convert logits to probabilities [0, 1] 
            loss = loss_fn(predictions, targets) # [0,1], {0,1}

            # After calculating loss, convert to binary for metrics
            predictions = (predictions > 0.5).float()  # Convert to binary {0, 1} for metrics/visualization
            
            # Calculate metrics
            total_loss += loss.item()
            preds = predictions.float()
            total_dice += calculate_dice_score(preds, targets)
            
            # Update progress bar
            loop.set_postfix(loss=loss.item())
            
            # Free GPU memory
            del data, targets, predictions, bottleneck
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice

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
    model_path, val_image_path, test_image_path = setup_directories()
    train_transform, val_transforms, test_transforms = create_transforms()
    config = setup_augmentation_config()
    
    # Model initialization
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    # loss_fn = nn.MSELoss()
    loss_fn = CombinedLoss(alpha=0.3, beta=0.5, gamma=0.2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define sample sizes for each iteration
    sample_sizes = [300, 335, 370, 405, 440, 475, 510, 545, 580]

    # GMM Cluster information
    base_ranked_clusters_file = "results/gmm_results/2025-04-22_03-38-29/ranked_cluster_assignments.csv"
    updated_ranked_clusters_file = "results/gmm_results/2025-04-22_03-38-29/ranked_cluster_assignments_updated.csv"
    prev_selected_file="results/gmm_results/2025-04-22_03-38-29/ranked_cluster_assignments_prev_selected.csv"
    
    # Previous model path for first iteration
    prev_model_path = PATH_TO_TRAINED_MODEL if LOAD_TRAINED_MODEL else None

    # Model loading if needed
    if LOAD_TRAINED_MODEL and prev_model_path is not None:
        model, start_epoch = load_trained_model_without_scheduler_and_output_layer(model, prev_model_path)
    else:
        print("\nNo pre-trained model loaded. Starting from scratch.\n")
        start_epoch = START_EPOCH
        
    # Dictionary to store test results across iterations
    test_results = {}

    # Iterative training
    for i, sample_size in enumerate(sample_sizes):
        iteration_num = i + 1
        print(f"\n===== ITERATION {iteration_num}: Training with {sample_size} samples =====\n")
        
        # Determine which ranked_clusters_file to use
        if iteration_num == 1:
            current_ranked_clusters_file = base_ranked_clusters_file
            prev_selected_current_file = None
        else:
            current_ranked_clusters_file = updated_ranked_clusters_file
            prev_selected_current_file = prev_selected_file
        print("prev_selected_current_file: ", prev_selected_current_file)

        # Data loading for current iteration
        train_loader, val_loader, test_loader = load_and_review_data(
            train_transform, val_transforms, test_transforms, config,
            ranked_clusters_file=current_ranked_clusters_file,
            prev_selected_cluster_file=prev_selected_current_file,
            top_n_samples=sample_size
        )
    
        # Create directories for this iteration
        iteration_model_path = model_path # os.path.join(model_path, f"iteration{iteration_num}")
        iteration_image_path = val_image_path # os.path.join(val_image_path, f"iteration{iteration_num}")
        os.makedirs(iteration_model_path, exist_ok=True)
        os.makedirs(iteration_image_path, exist_ok=True)
    
        # Load previous iteration's model if available
        if prev_model_path and i > 0:
            print(f"\nLoading model from previous iteration")
            # model, start_epoch = load_trained_model_without_scheduler(model,  prev_model_path)  
            # Model loading if needed
            model, start_epoch = load_trained_model_without_scheduler_and_output_layer(model, PATH_TO_TRAINED_MODEL)     
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
        start_epoch = 1

        # Training loop
        train_model(
            model, train_loader, val_loader, optimizer, 
            loss_fn, iteration_model_path, iteration_image_path, start_epoch=start_epoch, iteration_num=iteration_num, sample_size=sample_size
        )

        # Update model path for next iteration
        prev_model_path = os.path.join(iteration_model_path, "self_supervised_learning.pt")

        # ----- IMPROVED TESTING SECTION -----
        # Create a separate model instance for testing
        test_model = UNET(in_channels=3, out_channels=1).to(DEVICE)

        # Load only the model weights from the best model
        checkpoint = torch.load(prev_model_path, map_location=DEVICE)
        test_model.load_state_dict(checkpoint['state_dict'])

        # Get the actual learning rate used during training
        actual_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

        # Run test function to evaluate the model
        test_loss, test_dice = test_fn(test_loader, test_model, loss_fn, DEVICE)

        # Log test results
        print(f"Test Loss for Iteration {iteration_num} (Samples: {sample_size}): {test_loss:.4f}")
        print(f"Test Dice for Iteration {iteration_num} (Samples: {sample_size}): {test_dice:.4f}")
        
        # Create a unique key for each iteration combination
        test_result_key = f"{iteration_num}"

        # Store results for later comparison
        test_results[test_result_key] = {
            'iteration': iteration_num,
            'sample_size': sample_size,
            'current_learning_rate': actual_lr,  # Use the actual learning rate from training
            'test_loss': f"{test_loss:.5f}",
            'test_dice': f"{test_dice.item():.5f}"
        }

        # Save test results to CSV
        print("Saving test results to .csv")
        results_df = pd.DataFrame.from_dict(test_results, orient='index')
        results_df.to_csv(os.path.join(model_path, 'test_results.csv'))

        # Save images of predictions on test set
        save_ssaal_predictions_images(test_loader, test_model, folder=test_image_path, device=DEVICE)

        # Free memory
        del test_model
        torch.cuda.empty_cache()
        # ----- END IMPROVED TESTING SECTION -----

""" Setup Directories """
def setup_directories():
    """Create and return model directories for saving checkpoints"""
    timenow = datetime.strftime(datetime.now(pytz.timezone('America/New_York')), '%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join(MODEL_DIRECTORY, timenow)
    val_image_path = os.path.join(IMAGE_VAL_DIRECTORY, timenow)
    test_image_path = os.path.join(IMAGE_TEST_DIRECTORY, timenow)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(val_image_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)
    print("The model will be saved at:", model_path)
    print("The valiation images will be saved at:", val_image_path)
    print("The test images will be saved at:", test_image_path)
    return model_path, val_image_path, test_image_path

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


""" Scheduler """
def setup_scheduler(optimizer):
    """Set up learning rate scheduler"""
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, # 0.1
        patience=5, # 10
        # min_lr=1e-7  # Allow learning rate to decrease to this value
    )
    print_current_lr(scheduler)
    return scheduler


""" Load and Review Data """
def load_and_review_data(train_transform, val_transforms, test_transforms, config, ranked_clusters_file, prev_selected_cluster_file, top_n_samples):
    """Load and review training and validation data"""
    train_loader, val_loader, test_loader, updated_ranked_clusters_file, newly_selected_clusters_file, all_selected_clusters_file = data_loader_for_self_supervised_assisted_active_learning(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms, # no augmentation
        test_transforms, # no augmentation
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        config=config,
        shuffle_train=True, # shuffle training dataset
        ranked_clusters_file=ranked_clusters_file, 
        prev_selected_cluster_file=prev_selected_cluster_file,
        top_n_samples=top_n_samples  # Select top 300 samples
    )
    
    # ranked_clusters_file="results/gmm_results/2025-04-22_03-38-29/ranked_cluster_assignments.csv",
    # prev_selected_cluster_file="results/gmm_results/2025-04-22_03-38-29/ranked_cluster_assignments_prev_selected.csv",
    
    df = pd.read_csv(updated_ranked_clusters_file)
    upated_ranked_clusters_file = os.path.join("results/gmm_results/2025-04-22_03-38-29", "ranked_cluster_assignments_updated.csv")
    df.to_csv(upated_ranked_clusters_file, index=False)
    print(f"Saved updated ranked clusters to {upated_ranked_clusters_file}")

  
    df = pd.read_csv(newly_selected_clusters_file)
    selected_clusters_path = os.path.join("results/gmm_results/2025-04-22_03-38-29", "ranked_cluster_assignments_selected.csv")
    df.to_csv(selected_clusters_path, index=False)
    print(f"Saved newly selected clusters to {selected_clusters_path}")

    df = pd.read_csv(all_selected_clusters_file)
    all_selected_clusters_path = os.path.join("results/gmm_results/2025-04-22_03-38-29", "ranked_cluster_assignments_prev_selected.csv")
    df.to_csv(all_selected_clusters_path, index=False)
    print(f"Saved all selected clusters to {all_selected_clusters_path}")

    print(f"Data loading complete with {top_n_samples} samples for train_loader. \n")
    print("train_loader, val_loarder, and test_loader is completed. \n")
    
    # # Pixel value range
    # inspect_pixel_value_range(train_loader, "Train Loader")
    # inspect_pixel_value_range(val_loader, "Val Loader")
    # inspect_pixel_value_range(test_loader, "Test Loader")
    
    # # Review training and validation data
    # print("\nReviewing training data")
    # check_dataloader_sizes(train_loader)
    # review_batch(train_loader, 'training data')
    # print("\nReviewing validation data")
    # check_dataloader_sizes(val_loader)
    # review_batch(val_loader, 'validation data')
    
    return train_loader, val_loader, test_loader


# """ Load Pre-trained Model"""
# def load_pretrained_model_without_scheduler(model):
#     """Load pre-trained model if specified or return default start epoch"""
#     if LOAD_TRAINED_MODEL and PATH_TO_TRAINED_MODEL:
#         print("\nPre-train model loaded. MODEL_PATH:", PATH_TO_TRAINED_MODEL, "\n")
#         model,  start_epoch = load_trained_model_without_scheduler(
#             model, PATH_TO_TRAINED_MODEL
#         )            
#         return model, start_epoch
#     else:
#         print("\nNo pre-trained model loaded. Starting from scratch.\n")
#         return model, START_EPOCH


""" Func for Train Model """
def train_model(model, train_loader, val_loader, optimizer, loss_fn, iteration_model_path, iteration_image_path, start_epoch=1, iteration_num=None, sample_size=None):
    """Execute the main training loop"""
    print('--- train_model ---')
    best_loss = float('inf')
    num_epoch_no_improvement = 0
    
    # Clear memory before training
    sys.stdout.flush()
    torch.cuda.empty_cache()


    # Dictionary to store train and validation results across iterations
    train_results = {}
    
    
    for epoch in range(start_epoch, NUM_EPOCHS+1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        model = model.to(DEVICE)
        
        # Training and validation for this epoch
        print("Train ...")
        train_loss, train_dice, features_space = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        print("Validation ...")
        valid_loss, valid_dice = validate_fn(val_loader, model, loss_fn, DEVICE)
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}")

        # Create a unique key for each iteration-epoch combination
        train_result_key = f"{iteration_num}_{epoch}"
        
        # Store results
        train_results[train_result_key] = {
            'iteration': iteration_num,
            'epoch': epoch,
            'sample_size': sample_size,
            'current_learning_rate': optimizer.param_groups[0]['lr'],
            'train_loss': f"{train_loss:.5f}",
            'train_dice': f"{train_dice.item():.5f}",
            'valid_loss': f"{valid_loss:.5f}",
            'valid_dice': f"{valid_dice.item():.5f}"
        }
        
        # Save results to CSV
        print('current_learning_rate', optimizer.param_groups[0]['lr'])
        print("Saving training results to .csv")
        results_df = pd.DataFrame.from_dict(train_results, orient='index')
        results_df.to_csv(os.path.join(iteration_model_path, f'iteration{iteration_num}_train_results.csv'))
        
        
        # Save model if improved
        if valid_loss < best_loss:
            handle_improved_model(model, optimizer, epoch, valid_loss, best_loss, iteration_model_path)

            # Save sample predictions
            save_ssaal_predictions_images(val_loader, model, folder=iteration_image_path, device=DEVICE)

            best_loss = valid_loss
            num_epoch_no_improvement = 0
        else:
            num_epoch_no_improvement += 1
            print(f"Validation loss does not decrease from {best_loss:.4f}, " 
                  f"num_epoch_no_improvement {num_epoch_no_improvement}")          
        
        # Check early stopping conditions
        if check_early_stopping(num_epoch_no_improvement, best_loss):
            break
        
        # Clean up after epoch
        sys.stdout.flush()
        torch.cuda.empty_cache()

""" Func for Improved Model """
def handle_improved_model(model, optimizer, epoch, valid_loss, best_loss, model_path):
    """Save model when validation loss improves"""
    print(f"Validation loss decreases from {best_loss:.4f} to {valid_loss:.4f}")
    
    # Move model to CPU for saving
    model = model.cpu()
    
    # Save model state
    save_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    save_path = os.path.join(model_path, "self_supervised_learning.pt")
    torch.save(save_dict, save_path)
    print(f"Saving model {save_path}")
    
    # Move model back to GPU
    model = model.to(DEVICE)


""" Func for Early Stopping Conditions """
def check_early_stopping(num_epoch_no_improvement, best_loss):
    """Check if early stopping criteria are met"""
    if num_epoch_no_improvement >= EARLY_STOPPING_EPOCHES or best_loss <= EXPECTED_BEST_LOSS:
        if num_epoch_no_improvement >= EARLY_STOPPING_EPOCHES:
            print(f"Early stopping triggered: No improvement for {EARLY_STOPPING_EPOCHES} epochs")
        else:
            print(f"Early stopping triggered: Best loss of {best_loss:.4f} reached")
        return True
    return False

if __name__ == "__main__":
    main()
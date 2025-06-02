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
    save_ssaal_test_images,
    load_trained_model,
    calculate_dice_score,
    print_current_lr,
    check_dataloader_sizes,
    review_batch,
)

from loss_function import CombinedLoss
from gmm_with_likelihood_fun import run_gmm_clustering
from gmm_with_likelihood_adaptive_gmm_fun_B import run_rmn_clustering_pipeline
from features_extraction_fun import extract_features
from csv_file_fun import extract_and_save_filenames, extract_matching_records, create_new_filename_1

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
EARLY_STOPPING_EPOCHES = 20
EXPECTED_BEST_LOSS = 0.05

""" Load Self-Supervised Trained Model"""
LOAD_TRAINED_MODEL = False
PATH_TO_TRAINED_MODEL = ''

""" Image Ranking """
IMAGE_RANKING_ASCENDING = False # True: Sort by rank in ascending order (high probablilty (low rank) values first); False: Sort by rank in desceding order (low probablilty (high rank) values first)

""" Load Dataset"""
TRAIN_IMG_DIR = " "
TRAIN_MASK_DIR = " " 
VAL_IMG_DIR = " "
VAL_MASK_DIR = " " 
TEST_IMG_DIR = " "
TEST_MASK_DIR = " " 

""" Saving Directory"""
MODEL_DIRECTORY = "results/ssaal_trained_model"
IMAGE_VAL_DIRECTORY = "results/ssaal_validation_images"
IMAGE_TEST_DIRECTORY = "results/ssaal_test_images"

""" Saving Directory"""
FEATURE_SPACE_DIRECTORY = "results/ssaal_trained_model"

FULL_RESET_EACH_ITERATION = True



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
    scheduler = setup_scheduler(optimizer)

    # Define sample sizes for each iteration
    sample_sizes = [300, 335, 370, 405, 440, 475, 510, 545, 580]

    # GMM Cluster information
    base_ranked_clusters_file = "results/gmm_results/dynamic_features/ranked_cluster_assignments.csv"
    updated_ranked_clusters_file = "results/gmm_results/dynamic_features/ranked_cluster_assignments_updated.csv"
    prev_selected_file= "results/gmm_results/dynamic_features/ranked_cluster_assignments_prev_selected.csv"

    # Previous model path for first iteration
    prev_model_path = PATH_TO_TRAINED_MODEL if LOAD_TRAINED_MODEL else None

    # Model loading if needed
    if LOAD_TRAINED_MODEL and prev_model_path is not None:
        model, optimizer, scheduler, start_epoch = load_trained_model(model, optimizer, scheduler, prev_model_path)
    else:
        print("\nNo pre-trained model loaded. Starting from scratch.\n")
        start_epoch = START_EPOCH

    # Dictionary to store test results across iterations
    test_results = {}
    
    # Best valiation loss
    best_loss = float('inf')

    # Iterative training
    for i, sample_size in enumerate(sample_sizes):
        iteration_num = i + 1
        print(f"\n===== ITERATION {iteration_num}: Training with {sample_size} samples =====\n")
        3
        # Determine which ranked_clusters_file to use
        if iteration_num == 1:
            current_ranked_clusters_file = base_ranked_clusters_file
            prev_selected_current_file = None
        else:
            # --- DYNAMIC FEATURES EXTRATION - START ---
            print("--- DYNAMIC FEATURES EXTRATION and Adaptive GMM ---")
            ranked_df_path, output_dir_adpt_gmm = run_rmn_clustering_pipeline(
                input_dim=20,
                hidden_dims=[100, 50],
                latent_dim=20,
                n_components=10,
                alpha=1,
                gamma=1,
                neeta=0.1,
                min_size_weight=0.5,
                min_cluster_size=10,
                proximity_threshold=5,
                n_neighbors=5,
                pca_components=20,
                PATH_TO_TRAINED_MODEL=prev_model_path,
                FEATURE_SPACE_DIRECTORY=FEATURE_SPACE_DIRECTORY,
                train_img_dir=TRAIN_IMG_DIR,
                val_img_dir=VAL_IMG_DIR,
                iteration_num_adpt_gmm=iteration_num,
                n_epochs_adpt_gmm=5,
                lr_adpt_gmm=1e-5
                )
            print("ranked_df_path: ", ranked_df_path)
            print("--- EXTRACT COMMON RECORDS  ---")
            matches = extract_matching_records(
                ranked_df_path,  # Path to first CSV file
                "results/gmm_results/dynamic_features/ranked_cluster_assignments_updated_filenames.csv",  # Path to second CSV file: Which filenames should be added
                updated_ranked_clusters_file  # Path to output CSV file (common file names and other details)
            )
            print(f"Found {matches} matching records.")
            # --- DYNAMIC FEATURES EXTRATION - END ---

            current_ranked_clusters_file = updated_ranked_clusters_file
            prev_selected_current_file = prev_selected_file
        print("current_ranked_clusters_file: ", current_ranked_clusters_file)
        print("prev_selected_current_file: ", prev_selected_current_file)

        # Data loading for current iteration
        print("--- LOAD AND REVIEW DATA  ---")
        train_loader, val_loader, test_loader = load_and_review_data(
            train_transform, val_transforms, test_transforms, config,
            ranked_clusters_file=current_ranked_clusters_file,
            prev_selected_cluster_file=prev_selected_current_file,
            top_n_samples=sample_size,
            image_ranking_ascending=IMAGE_RANKING_ASCENDING
        )
    
        # Create directories for this iteration
        iteration_model_path = model_path # os.path.join(model_path, f"iteration{iteration_num}")
        iteration_image_path = val_image_path # os.path.join(val_image_path, f"iteration{iteration_num}")
        os.makedirs(iteration_model_path, exist_ok=True)
        os.makedirs(iteration_image_path, exist_ok=True)
    
        # Load previous iteration's model if available
        if prev_model_path and i > 0:
            print(f"\nLoading model from previous iteration")
            model, optimizer, scheduler, start_epoch = load_trained_model(model, optimizer, scheduler, prev_model_path)

        # # Reset learning rate explicitly
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE
        # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # Reset certain training parameters for this iteration
        """Most active learning implementations take one of these approaches:
            1) Partial transfer: Keep the model weights, reset both optimizer and scheduler
            2) Full transfer: Keep the model weights and optimizer state, reset only the scheduler
            3) Hybrid approach: Keep the model weights, conditionally reset the optimizer based on performance metrics
        """
        
        # Save the current best value
        best_value_scheduler = scheduler.best if hasattr(scheduler, 'best') else float('inf')

        # Reset scheduler while keeping the best value
        scheduler = reset_scheduler_keep_best(optimizer, best_value_scheduler)
        # scheduler = setup_scheduler(optimizer) 

        """ 
        When setup_scheduler(optimizer) is called, it creates a brand new ReduceLROnPlateau scheduler instance that 
        references the optimizer that was just loaded and modified. This new scheduler has:
            # No memory of previous iterations
            # Fresh internal counters (patience starts at 0)
            # No knowledge of previous best validation losses

        The key insight is that while the optimizer retains its momentum and other internal states from the previous iteration, 
        the scheduler is completely new and only knows about the optimizer's current state.
        Even though the scheduler is watching the same optimizer object that was loaded from the checkpoint, 
        the scheduler itself is a new object with reset internal state. 
        It will start tracking the learning rate anew and make decisions about when to reduce the learning rate based only on what happens in 
        the current active learning iteration.
        """ 

        # """ Full reset Stats"""
        # if FULL_RESET_EACH_ITERATION:
        #     model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        #     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #     scheduler = setup_scheduler(optimizer)
        #     start_epoch = START_EPOCH
        # else:
        #     if prev_model_path and i > 0:
        #         model, optimizer, scheduler, start_epoch = load_trained_model(model, optimizer, scheduler, prev_model_path)
        # """ Full reset Ends"""
        
        start_epoch = 1

        # Training loop - pass and receive best_loss
        print("--- TRAIN  ---")
        best_loss = train_model(
            model, train_loader, val_loader, optimizer, 
            scheduler, loss_fn, iteration_model_path, iteration_image_path, 
            start_epoch=start_epoch, iteration_num=iteration_num, 
            sample_size=sample_size, best_loss=best_loss
        )

        # Update model path for next iteration
        prev_model_path = os.path.join(iteration_model_path, "self_supervised_learning.pt")

        # ----- IMPROVED TESTING SECTION -----
        print("--- TEST ---")
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
        # iteration_test_image_path = os.path.join(test_image_path, f"iteration{iteration_num}")
        # os.makedirs(iteration_test_image_path, exist_ok=True)

        save_ssaal_test_images(test_loader, test_model, folder=test_image_path, device=DEVICE, iteration=iteration_num)

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
        # min_lr=1e-7,  # Allow learning rate to decrease to this value
        verbose=True  # Print when learning rate changes
    )
    print(f"DEBUG: Scheduler created with factor={scheduler.factor}, patience={scheduler.patience}")
    print_current_lr(scheduler)
    return scheduler

def reset_scheduler_keep_best(optimizer, best_value_scheduler):
    # Create a new scheduler
    new_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Transfer the best value from the old scheduler
    new_scheduler.best = best_value_scheduler
    
    print(f"Scheduler reset with preserved best value: {new_scheduler.best}")
    print_current_lr(new_scheduler)
    
    return new_scheduler

def print_scheduler_info(scheduler):
    """Print detailed information about the scheduler's internal state"""
    print(f"Scheduler's internal counter (num_bad_epochs): {scheduler.num_bad_epochs}")
    print(f"Scheduler's best loss so far: {scheduler.best}")
    print(f"Scheduler's patience: {scheduler.patience}")
    print(f"Scheduler's factor: {scheduler.factor}")
    print(f"Current learning rate: {scheduler.optimizer.param_groups[0]['lr']}")

""" Load and Review Data """
def load_and_review_data(train_transform, val_transforms, test_transforms, config, ranked_clusters_file, prev_selected_cluster_file, top_n_samples, image_ranking_ascending):
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
        top_n_samples=top_n_samples,  # Select top 300 samples
        image_ranking_ascending=image_ranking_ascending 
    )
    
    # ranked_clusters_file="results/gmm_results/2025-04-22_03-38-29/ranked_cluster_assignments.csv",
    # prev_selected_cluster_file="results/gmm_results/2025-04-22_03-38-29/ranked_cluster_assignments_prev_selected.csv",
    
    df = pd.read_csv(updated_ranked_clusters_file)
    upated_ranked_clusters_file = os.path.join("results/gmm_results/dynamic_features", "ranked_cluster_assignments_updated.csv")
    df.to_csv(upated_ranked_clusters_file, index=False)
    print(f"Saved updated ranked clusters to {upated_ranked_clusters_file}")  


    # --- EXTRACT FILES NAME FROM THE UNUSED TRAINING DATA - START ---
     # Replace with your actual file paths
    upated_ranked_clusters_file_filenames = create_new_filename_1(upated_ranked_clusters_file)
    success = extract_and_save_filenames(upated_ranked_clusters_file, "results/gmm_results/dynamic_features/ranked_cluster_assignments_updated_filenames.csv")
    
    if success:
        print("Operation completed successfully.")
    else:
        print("Operation failed.")    
    # --- EXTRACT FILES NAME FROM THE UNUSED TRAINING DATA - END ---
  
    df = pd.read_csv(newly_selected_clusters_file)
    selected_clusters_path = os.path.join("results/gmm_results/dynamic_features", "ranked_cluster_assignments_selected.csv")
    df.to_csv(selected_clusters_path, index=False)
    print(f"Saved newly selected clusters to {selected_clusters_path}")

    df = pd.read_csv(all_selected_clusters_file)
    all_selected_clusters_path = os.path.join("results/gmm_results/dynamic_features", "ranked_cluster_assignments_prev_selected.csv")
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
# def load_pretrained_model(model, optimizer, scheduler):
#     """Load pre-trained model if specified or return default start epoch"""
#     if LOAD_TRAINED_MODEL and PATH_TO_TRAINED_MODEL:
#         print("\nPre-train model loaded. MODEL_PATH:", PATH_TO_TRAINED_MODEL, "\n")
#         model, optimizer, scheduler, start_epoch = load_trained_model(
#             model, optimizer, scheduler, PATH_TO_TRAINED_MODEL
#         )            
#         return model, optimizer, scheduler, start_epoch
#     else:
#         print("\nNo pre-trained model loaded. Starting from scratch.\n")
#         return model, optimizer, scheduler, START_EPOCH


""" Func for Train Model """
def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, iteration_model_path, iteration_image_path, start_epoch=1, iteration_num=None, sample_size=None, best_loss=float('inf')):
    """Execute the main training loop"""
    print('--- train_model ---')
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
            'current_learning_rate': scheduler.optimizer.param_groups[0]['lr'], 
            'train_loss': f"{train_loss:.5f}",
            'train_dice': f"{train_dice.item():.5f}",
            'valid_loss': f"{valid_loss:.5f}",
            'valid_dice': f"{valid_dice.item():.5f}"
        }
        
        # Save results to CSV
        print("Saving training results to .csv")
        results_df = pd.DataFrame.from_dict(train_results, orient='index')
        results_df.to_csv(os.path.join(iteration_model_path, f'iteration{iteration_num}_train_results.csv'))
        
        # Update learning rate
        # First, step the scheduler WITH the current validation loss
        # This must happen BEFORE checking for improvement
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(valid_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        new_lr = optimizer.param_groups[0]['lr']

        # Print detailed scheduler information
        print_scheduler_info(scheduler)

        # Check if learning rate changed
        if new_lr != old_lr:
            print(f"Learning rate changed from {old_lr} to {new_lr}")
        
        print(f"Current learning rate: {new_lr}")
        print(f"Scheduler's internal counter: {scheduler.num_bad_epochs}")
        
        
        # Save model if improved
        if valid_loss < best_loss:
            handle_improved_model(model, optimizer, scheduler, epoch, valid_loss, best_loss, iteration_model_path)

            # Save sample predictions
            save_ssaal_predictions_images(val_loader, model, folder=iteration_image_path, device=DEVICE)

            best_loss = valid_loss
            num_epoch_no_improvement = 0
        else:
            num_epoch_no_improvement += 1
            print(f"Validation loss does not decrease from {best_loss:.4f}, " 
                  f"num_epoch_no_improvement {num_epoch_no_improvement} since the last best loss update")    
        
        # Check early stopping conditions
        if check_early_stopping(num_epoch_no_improvement, best_loss):
            break
        
        # Clean up after epoch
        sys.stdout.flush()
        torch.cuda.empty_cache()

    # Return the best_loss to update it in the main function
    return best_loss

""" Func for Improved Model """
def handle_improved_model(model, optimizer, scheduler, epoch, valid_loss, best_loss, model_path):
    """Save model when validation loss improves and return updated best_loss"""
    print(f"Validation loss decreases from {best_loss:.4f} to {valid_loss:.4f}")
    
    # Move model to CPU for saving
    model = model.cpu()
    
    # Save model state
    save_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
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
#!/usr/bin/env python3
"""
This script uses the run_config.py file for all configuration parameters.
Simply modify run_config.py to change any training parameters.

Usage:
    python train.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytz

# Import configuration
from run_config import *

# Import your existing modules (these need to be adapted to work with the new structure)
try:
    # Assuming these modules exist in your original code
    from src.model import UNET
    from utils.utils import (
        get_loaders_with_augmentation,
        data_loader_for_self_supervised_assisted_active_learning,
        Config,
        load_trained_model,
        calculate_dice_score,
        save_ssaal_predictions_images,
        save_ssaal_test_images,
    )
    from loss_function import CombinedLoss
    from gmm_with_likelihood_adaptive_gmm_fun import run_rmn_clustering_pipeline
    from entropy_calculation import MultiClassEntropyCalculator
    from csv_file_fun import extract_and_save_filenames, extract_matching_records
    from replace_likelihood_by_entropy import replace_likelihood_with_entropy_and_preserve_order
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the correct locations.")
    print("You may need to adapt the import paths based on your project structure.")
    sys.exit(1)


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
    print("The validation images will be saved at:", val_image_path)
    print("The test images will be saved at:", test_image_path)
    
    return model_path, val_image_path, test_image_path


def create_transforms():
    """Create and return training and validation transforms"""
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=AUGMENTATION_CONFIG['rotate_limit'], p=1.0),
        A.HorizontalFlip(p=AUGMENTATION_CONFIG['horizontal_flip_prob']),
        A.VerticalFlip(p=AUGMENTATION_CONFIG['vertical_flip_prob']),
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


def setup_augmentation_config():
    """Set up configuration for self-supervised augmentations"""
    return Config(
        flip_rate=AUGMENTATION_CONFIG['flip_rate'],
        local_rate=AUGMENTATION_CONFIG['local_rate'],
        nonlinear_rate=AUGMENTATION_CONFIG['nonlinear_rate'], 
        paint_rate=AUGMENTATION_CONFIG['paint_rate'],
        inpaint_rate=AUGMENTATION_CONFIG['inpaint_rate']
    )


def setup_scheduler(optimizer):
    """Set up learning rate scheduler"""
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        verbose=SCHEDULER_VERBOSE
    )
    print(f"Scheduler created with factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE}")
    return scheduler


def reset_scheduler_keep_best(optimizer, best_value_scheduler):
    """Reset scheduler while keeping the best value"""
    new_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        verbose=SCHEDULER_VERBOSE
    )
    
    new_scheduler.best = best_value_scheduler
    print(f"Scheduler reset with preserved best value: {new_scheduler.best}")
    
    return new_scheduler


def train_fn(loader, model, optimizer, loss_fn, device):
    """Training function for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    bottlenecks = []
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().to(device)

        # Forward pass
        predictions, bottleneck = model(data)
        predictions = torch.sigmoid(predictions)
        loss = loss_fn(predictions, targets)

        # Convert to binary for metrics
        predictions = (predictions > 0.5).float()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        total_loss += loss.item()
        total_dice += calculate_dice_score(predictions, targets)

        # Store bottlenecks
        bottlenecks.append(bottleneck.detach().cpu())

        # Update progress bar
        loop.set_postfix(loss=loss.item())

        # Free GPU memory
        del data, targets, predictions, bottleneck
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice, bottlenecks


def validate_fn(loader, model, loss_fn, device):
    """Validation function"""
    model.eval()
    total_loss = 0
    total_dice = 0
    loop = tqdm(loader)
    
    with torch.no_grad():
        for idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.float().to(device)
            
            predictions, bottleneck = model(data) 
            predictions = torch.sigmoid(predictions)
            loss = loss_fn(predictions, targets)

            # Convert to binary for metrics
            predictions = (predictions > 0.5).float()
            
            # Calculate metrics
            total_loss += loss.item()
            total_dice += calculate_dice_score(predictions, targets)
            
            # Update progress bar
            loop.set_postfix(loss=loss.item())
            
            # Free GPU memory
            del data, targets, predictions, bottleneck
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice


def test_fn(loader, model, loss_fn, device):
    """Test function"""
    model.eval()
    total_loss = 0
    total_dice = 0
    loop = tqdm(loader)
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.float().to(device)

            predictions, _ = model(data)
            predictions = torch.sigmoid(predictions)
            loss = loss_fn(predictions, targets)

            # Convert to binary for metrics
            predictions = (predictions > 0.5).float()
           
            # Calculate metrics
            total_loss += loss.item()
            total_dice += calculate_dice_score(predictions, targets)
            
            # Update progress bar
            loop.set_postfix(loss=loss.item())

            # Free GPU memory
            del data, targets, predictions
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice


def load_and_review_data(train_transform, val_transforms, test_transforms, config, 
                        ranked_clusters_file, prev_selected_cluster_file, 
                        top_n_samples, image_ranking_ascending):
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
        val_transforms,
        test_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        config=config,
        shuffle_train=True,
        ranked_clusters_file=ranked_clusters_file, 
        prev_selected_cluster_file=prev_selected_cluster_file,
        top_n_samples=top_n_samples,
        image_ranking_ascending=image_ranking_ascending 
    )
    
    # Save updated files
    df_urk = pd.read_csv(updated_ranked_clusters_file)
    updated_ranked_clusters_file_path = os.path.join("results/cluster_info", "ranked_cluster_assignments_updated.csv")
    df_urk.to_csv(updated_ranked_clusters_file_path, index=False)

    # Extract filenames from unused training data
    success = extract_and_save_filenames(updated_ranked_clusters_file_path, "results/cluster_info/ranked_cluster_assignments_updated_filenames.csv")
    
    if not success:
        print("Operation failed: EXTRACT FILES NAME FROM THE UNUSED TRAINING DATA")
  
    # Save newly selected clusters
    df_nsc = pd.read_csv(newly_selected_clusters_file)
    selected_clusters_path = os.path.join("results/cluster_info", "ranked_cluster_assignments_selected.csv")
    df_nsc.to_csv(selected_clusters_path, index=False)

    # Save all selected clusters
    df_asc = pd.read_csv(all_selected_clusters_file)
    all_selected_clusters_path = os.path.join("results/cluster_info", "ranked_cluster_assignments_prev_selected.csv")
    df_asc.to_csv(all_selected_clusters_path, index=False)

    print(f"Data loading complete with {top_n_samples} samples for train_loader.")
    print("train_loader, val_loader, and test_loader completed.")
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, 
               iteration_model_path, iteration_image_path, start_epoch=1, 
               iteration_num=None, sample_size=None, best_loss=float('inf')):
    """Execute the main training loop"""
    num_epoch_no_improvement = 0
    
    # Clear memory before training
    sys.stdout.flush()
    torch.cuda.empty_cache()

    train_results = {}
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        model = model.to(DEVICE)
        
        # Training and validation for this epoch
        print("Training...")
        train_loss, train_dice, features_space = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        print("Validation...")
        valid_loss, valid_dice = validate_fn(val_loader, model, loss_fn, DEVICE)
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}")

        # Store results
        train_result_key = f"{iteration_num}_{epoch}"
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
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(valid_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        new_lr = optimizer.param_groups[0]['lr']

        # Check if learning rate changed
        if new_lr != old_lr:
            print(f"Learning rate changed from {old_lr} to {new_lr}")
        
        # Save model if improved
        if valid_loss < best_loss:
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
            save_path = os.path.join(iteration_model_path, "adaptive_gmmal_seg.pt")
            torch.save(save_dict, save_path)
            print(f"Saving model {save_path}")
            
            # Move model back to GPU
            model = model.to(DEVICE)

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

    return best_loss


def check_early_stopping(num_epoch_no_improvement, best_loss):
    """Check if early stopping criteria are met"""
    if num_epoch_no_improvement >= EARLY_STOPPING_EPOCHES or best_loss <= EXPECTED_BEST_LOSS:
        if num_epoch_no_improvement >= EARLY_STOPPING_EPOCHES:
            print(f"Early stopping triggered: No improvement for {EARLY_STOPPING_EPOCHES} epochs")
        else:
            print(f"Early stopping triggered: Best loss of {best_loss:.4f} reached")
        return True
    return False


def main():
    """Main training function"""
    print("="*80)
    print("A WARMER START TO ACTIVE LEARNING WITH ADAPTIVE GAUSSIAN MIXTURE MODELS FOR SKIN LESION SEGMENTATION")
    print("="*80)
    
    # Print and validate configuration
    print_config()
    if not validate_config():
        print("Configuration validation failed. Please check your settings.")
        return
    
    torch.cuda.empty_cache()
    
    # Setup phase
    model_path, val_image_path, test_image_path = setup_directories()
    train_transform, val_transforms, test_transforms = create_transforms()
    config = setup_augmentation_config()
    
    # Model initialization
    model = UNET(in_channels=MODEL_IN_CHANNELS, out_channels=MODEL_OUT_CHANNELS).to(DEVICE)
    loss_fn = CombinedLoss(alpha=LOSS_ALPHA, beta=LOSS_BETA, gamma=LOSS_GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = setup_scheduler(optimizer)

    # Previous model path for first iteration
    PREV_MODEL_PATH = PATH_TO_TRAINED_MODEL if LOAD_TRAINED_SSL_MODEL else None

    # Model loading if needed
    if LOAD_TRAINED_SSL_MODEL and PREV_MODEL_PATH is not None:
        model, optimizer, scheduler, start_epoch = load_trained_model(model, optimizer, scheduler, PREV_MODEL_PATH)
    else:
        print("\nNo pre-trained model loaded. Starting from scratch.\n")
        start_epoch = START_EPOCH

    # Dictionary to store test results across iterations
    test_results = {}
    
    # Best validation loss
    best_loss = float('inf')

    # Iterative training
    for i, sample_size in enumerate(SAMPLE_SIZES):
        iteration_num = i + 1
        print(f"\n===== ITERATION {iteration_num}: Training with {sample_size} samples =====\n")
        
        # Determine which ranked_clusters_file to use
        if iteration_num == 1:
            current_ranked_clusters_file = BASE_RANKED_CLUSTERS_FILE
            prev_selected_current_file = None
        else:
            # Dynamic features extraction - Adaptive GMM
            print("--- DYNAMIC FEATURES EXTRACTION and Adaptive GMM ---")
            ranked_df_path_adaptive_gmm, output_dir_adpt_gmm = run_rmn_clustering_pipeline(
                input_dim=ADAPTIVE_GMM_CONFIG['input_dim'],
                hidden_dims=ADAPTIVE_GMM_CONFIG['hidden_dims'],
                latent_dim=ADAPTIVE_GMM_CONFIG['latent_dim'],
                n_components=ADAPTIVE_GMM_CONFIG['n_components'],
                weight_log_likelihood_loss=ADAPTIVE_GMM_CONFIG['weight_log_likelihood_loss'],
                weight_sep_term=ADAPTIVE_GMM_CONFIG['weight_sep_term'],
                weight_entropy_loss=ADAPTIVE_GMM_CONFIG['weight_entropy_loss'],
                min_size_weight=ADAPTIVE_GMM_CONFIG['min_size_weight'],
                min_cluster_size=ADAPTIVE_GMM_CONFIG['min_cluster_size'],
                proximity_threshold=ADAPTIVE_GMM_CONFIG['proximity_threshold'],
                n_neighbors=ADAPTIVE_GMM_CONFIG['n_neighbors'],
                pca_components=ADAPTIVE_GMM_CONFIG['pca_components'],
                PATH_TO_TRAINED_MODEL=PREV_MODEL_PATH,
                FEATURE_SPACE_DIRECTORY=FEATURE_SPACE_DIRECTORY,
                train_img_dir=TRAIN_IMG_DIR,
                val_img_dir=VAL_IMG_DIR,
                iteration_num_adpt_gmm=iteration_num,
                n_epochs_adpt_gmm=ADAPTIVE_GMM_CONFIG['n_epochs_adpt_gmm'],
                lr_adpt_gmm=ADAPTIVE_GMM_CONFIG['lr_adpt_gmm'],
                freeze_encoder=ADAPTIVE_GMM_CONFIG['freeze_encoder']
            )
            
            print("--- EXTRACT COMMON RECORDS ---")
            matches = extract_matching_records(
                ranked_df_path_adaptive_gmm,
                "results/cluster_info/ranked_cluster_assignments_updated_filenames.csv",
                UPDATED_RANKED_CLUSTERS_FILE
            )
            print(f"Found {matches} matching records.")

            if ACTIVE_SAMPLE_SELECTION_METHOD == 'CBS':
                print(f'Active sample selection method: {ACTIVE_SAMPLE_SELECTION_METHOD}')
                pass
            elif ACTIVE_SAMPLE_SELECTION_METHOD == 'EBR':
                print(f'Active sample selection method: {ACTIVE_SAMPLE_SELECTION_METHOD}')
                # Initialize entropy calculator
                entropy_calc = MultiClassEntropyCalculator(PREV_MODEL_PATH, num_classes=NUM_CLASSES)
                
                # Process dataset and calculate entropy
                _ = entropy_calc.process_dataset(
                    train_img_dir=TRAIN_IMG_DIR,
                    val_img_dir=VAL_IMG_DIR,
                    batch_size=1,
                    save_results=True,
                    output_file=ENTROPY_FILE
                )
                print(f"\nProcessing complete! Results available in '{ENTROPY_FILE}'")
                
                # Low Entropy First (Default - ascending_order=True) -> Most confident predictions (low uncertainty)
                replace_likelihood_with_entropy_and_preserve_order(
                    UPDATED_RANKED_CLUSTERS_FILE, ENTROPY_FILE, UPDATED_RANKED_CLUSTERS_FILE, ascending_order=True
                )
            else:
                print('Error: Wrong active sample selection method')
  
            current_ranked_clusters_file = UPDATED_RANKED_CLUSTERS_FILE
            prev_selected_current_file = PREV_SELECTED_FILE
        
        print("current_ranked_clusters_file:", current_ranked_clusters_file)
        print("prev_selected_current_file:", prev_selected_current_file)

        # Data loading for current iteration
        print("--- LOAD AND REVIEW DATA ---")
        train_loader, val_loader, test_loader = load_and_review_data(
            train_transform, val_transforms, test_transforms, config,
            ranked_clusters_file=current_ranked_clusters_file,
            prev_selected_cluster_file=prev_selected_current_file,
            top_n_samples=sample_size,
            image_ranking_ascending=ACTIVE_SAMPLE_SELECTION_ORDER
        )
        
        # Load previous iteration's model if available
        if PREV_MODEL_PATH and i > 0:
            print(f"\nLoading model from previous iteration")
            model, optimizer, scheduler, start_epoch = load_trained_model(model, optimizer, scheduler, PREV_MODEL_PATH)

        # Reset learning rate explicitly
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE
        
        # Save the current best value
        best_value_scheduler = scheduler.best if hasattr(scheduler, 'best') else float('inf')

        # Reset scheduler while keeping the best value
        scheduler = reset_scheduler_keep_best(optimizer, best_value_scheduler)

        # Training loop
        print("--- TRAIN MODEL ---")
        best_loss = train_model(
            model, train_loader, val_loader, optimizer, 
            scheduler, loss_fn, model_path, val_image_path, 
            start_epoch=1, iteration_num=iteration_num, 
            sample_size=sample_size, best_loss=best_loss
        )

        # Update model path for next iteration
        PREV_MODEL_PATH = os.path.join(model_path, "adaptive_gmmal_seg.pt")

        print("--- TEST MODEL---")
        # Create a separate model instance for testing
        test_model = UNET(in_channels=MODEL_IN_CHANNELS, out_channels=MODEL_OUT_CHANNELS).to(DEVICE)

        # Load only the model weights from the best model
        checkpoint = torch.load(PREV_MODEL_PATH, map_location=DEVICE)
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
            'current_learning_rate': actual_lr,
            'test_loss': f"{test_loss:.5f}",
            'test_dice': f"{test_dice.item():.5f}"
        }

        # Save test results to CSV
        print("Saving test results to .csv")
        results_df_entropy = pd.DataFrame.from_dict(test_results, orient='index')
        results_df_entropy.to_csv(os.path.join(model_path, 'test_results.csv'))

        # Save images of predictions on test set
        save_ssaal_test_images(test_loader, test_model, folder=test_image_path, device=DEVICE, iteration=iteration_num)

        # Free memory
        del test_model
        torch.cuda.empty_cache()

    # Final results summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    
    # Print best results
    best_iteration = max(test_results.items(), key=lambda x: float(x[1]['test_dice']))
    print(f"Best Results:")
    print(f"  Iteration: {best_iteration[1]['iteration']}")
    print(f"  Sample Size: {best_iteration[1]['sample_size']}")
    print(f"  Test Dice: {best_iteration[1]['test_dice']}")
    print(f"  Test Loss: {best_iteration[1]['test_loss']}")
    
    print(f"\nAll results saved to: {model_path}")
    print("="*80)


if __name__ == "__main__":
    main()
# run_config.py
"""
Configuration file for Medical Image Segmentation with Active Learning
All training parameters and paths are defined here.
"""

import torch

# =============================================================================
# Training Hyperparameters
# =============================================================================
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
START_EPOCH = 1
NUM_EPOCHS = 20
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True

# =============================================================================
# Early Stopping Conditions
# =============================================================================
EARLY_STOPPING_EPOCHES = 20
EXPECTED_BEST_LOSS = 0.05

# =============================================================================
# Load Self-Supervised Trained Model
# =============================================================================
LOAD_TRAINED_SSL_MODEL = False
PATH_TO_TRAINED_MODEL = 'results/ssl_trained_model/2025-04-19_01-19-21/self_supervised_learning.pt'

# =============================================================================
# Active Sample Selection Method
# =============================================================================
# TWO OPTIONS:
#   - Cluster-Based Probability (CBS) Selection - 'CBS'
#   - Entropy-Based Ranking (EBR) within Cluster - 'EBR'
# Fix typo in variable name
ACTIVE_SAMPLE_SELECTION_METHOD = 'EBR'

# =============================================================================
# Active Sample Selection Order
# =============================================================================
# TWO OPTIONS:
#   - True: Active sample selection selects the high probability (low entropy) images first
#   - False: Active sample selection selects the low probability (high entropy) images first
ACTIVE_SAMPLE_SELECTION_ORDER = False

# =============================================================================
# Dataset Paths
# =============================================================================
TRAIN_IMG_DIR = "../ISIC_2017_dataset/data/train_images/"
TRAIN_MASK_DIR = "../ISIC_2017_dataset/data/train_masks/"
VAL_IMG_DIR = "../ISIC_2017_dataset/data/val_images/"
VAL_MASK_DIR = "../ISIC_2017_dataset/data/val_masks/"
TEST_IMG_DIR = "../ISIC_2017_dataset/data/test_images/"
TEST_MASK_DIR = "../ISIC_2017_dataset/data/test_masks/"
NUM_CLASSES = 2  # Default: 2 for binary segmentation (background + lesion)

# =============================================================================
# GMM Cluster Saving Information
# =============================================================================
BASE_RANKED_CLUSTERS_FILE = "results/cluster_info/ranked_cluster_assignments.csv"  # The GMM results of the full training dataset. This is for only for the first iteration
UPDATED_RANKED_CLUSTERS_FILE = "results/cluster_info/ranked_cluster_assignments_updated.csv"  # The images in the training dataset that have not been used yet
PREV_SELECTED_FILE = "results/cluster_info/ranked_cluster_assignments_prev_selected.csv"  # The images in the training dataset that have been already used
ENTROPY_FILE = "results/cluster_info/entropy_results.csv"  # The entropy calculation of the full training dataset

# =============================================================================
# Saving Directories
# =============================================================================
MODEL_DIRECTORY = "results/ssaal_trained_model"
IMAGE_VAL_DIRECTORY = "results/ssaal_validation_images"
IMAGE_TEST_DIRECTORY = "results/ssaal_test_images"
FEATURE_SPACE_DIRECTORY = "results/ssaal_trained_model"

# =============================================================================
# Active Learning Configuration
# =============================================================================
# Define sample sizes for each iteration
SAMPLE_SIZES = [300, 335, 370, 405, 440, 475, 510, 545, 580]

# Loss function parameters
LOSS_ALPHA = 0.3  # Weight for BCE loss
LOSS_BETA = 0.5   # Weight for Dice loss
LOSS_GAMMA = 0.2  # Weight for Focal loss

# Scheduler parameters
SCHEDULER_FACTOR = 0.5    # Factor to reduce LR by
SCHEDULER_PATIENCE = 5    # Epochs to wait before reducing LR
SCHEDULER_VERBOSE = True  # Print when LR changes

# Adaptive GMM parameters
ADAPTIVE_GMM_CONFIG = {
    'input_dim': 20,
    'hidden_dims': [100, 50],
    'latent_dim': 20,
    'n_components': 10,
    'weight_log_likelihood_loss': 1,  # Updated parameter name
    'weight_sep_term': 1,             # Updated parameter name
    'weight_entropy_loss': 0.1,       # Updated parameter name
    'min_size_weight': 0.5,
    'min_cluster_size': 10,
    'proximity_threshold': 5,
    'n_neighbors': 5,
    'pca_components': 20,
    'n_epochs_adpt_gmm': 5,
    'lr_adpt_gmm': 1e-5,
    'freeze_encoder': False           # Freeze autoencoder
}

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_IN_CHANNELS = 3
MODEL_OUT_CHANNELS = 1

# =============================================================================
# Data Augmentation Parameters
# =============================================================================
AUGMENTATION_CONFIG = {
    'rotate_limit': 35,
    'horizontal_flip_prob': 0.5,
    'vertical_flip_prob': 0.1,
    'flip_rate': 0.5,
    'local_rate': 0.4,
    'nonlinear_rate': 0.6,
    'paint_rate': 0.7,
    'inpaint_rate': 0.5
}

# =============================================================================
# Utility Functions
# =============================================================================
def print_config():
    """Print current configuration settings."""
    print("="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Image Size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_EPOCHES}")
    print(f"Active Learning Method: {ACTIVE_SAMPLE_SELECTION_METHOD}")
    print(f"Selection Order (High Confidence First): {ACTIVE_SAMPLE_SELECTION_ORDER}")
    print(f"Load Pre-trained Model: {LOAD_TRAINED_SSL_MODEL}")
    print(f"Sample Sizes: {SAMPLE_SIZES}")
    print("="*80)

def validate_config():
    """Validate configuration parameters."""
    import os
    from pathlib import Path
    
    errors = []
    
    # Check dataset paths
    paths_to_check = [
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, 
        VAL_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR
    ]
    
    for path in paths_to_check:
        if not Path(path).exists():
            errors.append(f"Path does not exist: {path}")
    
    # Check active learning method
    if ACTIVE_SAMPLE_SELECTION_METHOD not in ['CBS', 'EBR']:
        errors.append(f"Invalid active learning method: {ACTIVE_SAMPLE_SELECTION_METHOD}")
    
    # Check sample sizes
    if not SAMPLE_SIZES or any(size <= 0 for size in SAMPLE_SIZES):
        errors.append("Sample sizes must be positive integers")
    
    # Check pre-trained model if loading is enabled
    if LOAD_TRAINED_SSL_MODEL and not Path(PATH_TO_TRAINED_MODEL).exists():
        errors.append(f"Pre-trained model not found: {PATH_TO_TRAINED_MODEL}")
    
    if errors:
        print("Configuration Validation Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Configuration validation passed!")
    return True

# =============================================================================
# Export all configuration as a dictionary (for compatibility)
# =============================================================================
CONFIG_DICT = {
    # Training parameters
    'LEARNING_RATE': LEARNING_RATE,
    'DEVICE': DEVICE,
    'BATCH_SIZE': BATCH_SIZE,
    'START_EPOCH': START_EPOCH,
    'NUM_EPOCHS': NUM_EPOCHS,
    'NUM_WORKERS': NUM_WORKERS,
    'IMAGE_HEIGHT': IMAGE_HEIGHT,
    'IMAGE_WIDTH': IMAGE_WIDTH,
    'PIN_MEMORY': PIN_MEMORY,
    
    # Early stopping
    'EARLY_STOPPING_EPOCHES': EARLY_STOPPING_EPOCHES,
    'EXPECTED_BEST_LOSS': EXPECTED_BEST_LOSS,
    
    # Model loading
    'LOAD_TRAINED_SSL_MODEL': LOAD_TRAINED_SSL_MODEL,
    'PATH_TO_TRAINED_MODEL': PATH_TO_TRAINED_MODEL,
    
    # Active learning
    'ACTIVE_SAMPLE_SELECTION_METHOD': ACTIVE_SAMPLE_SELECTION_METHOD,
    'ACTIVE_SAMPLE_SELECTION_ORDER': ACTIVE_SAMPLE_SELECTION_ORDER,
    'SAMPLE_SIZES': SAMPLE_SIZES,
    
    # Dataset paths
    'TRAIN_IMG_DIR': TRAIN_IMG_DIR,
    'TRAIN_MASK_DIR': TRAIN_MASK_DIR,
    'VAL_IMG_DIR': VAL_IMG_DIR,
    'VAL_MASK_DIR': VAL_MASK_DIR,
    'TEST_IMG_DIR': TEST_IMG_DIR,
    'TEST_MASK_DIR': TEST_MASK_DIR,
    'NUM_CLASSES': NUM_CLASSES,
    
    # File paths
    'BASE_RANKED_CLUSTERS_FILE': BASE_RANKED_CLUSTERS_FILE,
    'UPDATED_RANKED_CLUSTERS_FILE': UPDATED_RANKED_CLUSTERS_FILE,
    'PREV_SELECTED_FILE': PREV_SELECTED_FILE,
    'ENTROPY_FILE': ENTROPY_FILE,
    
    # Directories
    'MODEL_DIRECTORY': MODEL_DIRECTORY,
    'IMAGE_VAL_DIRECTORY': IMAGE_VAL_DIRECTORY,
    'IMAGE_TEST_DIRECTORY': IMAGE_TEST_DIRECTORY,
    'FEATURE_SPACE_DIRECTORY': FEATURE_SPACE_DIRECTORY,
    
    # Additional configs
    'LOSS_ALPHA': LOSS_ALPHA,
    'LOSS_BETA': LOSS_BETA,
    'LOSS_GAMMA': LOSS_GAMMA,
    'ADAPTIVE_GMM_CONFIG': ADAPTIVE_GMM_CONFIG,
    'AUGMENTATION_CONFIG': AUGMENTATION_CONFIG
}

if __name__ == "__main__":
    print_config()
    validate_config()
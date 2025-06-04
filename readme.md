# TAIAL Framework - Task-Aligned Iterative Active Learning for Skin Lesion Segmentation

This repository implements the TAIAL (Task-Aligned Iterative Active Learning) framework described in the paper **"A Warmer Start to Active Learning with Adaptive Gaussian Mixture Models for Skin Lesion Segmentation"**.

## 🎯 Overview
ABSTRACT: Active learning is a promising strategy for reducing anno-
tation burdens in medical image segmentation, particularly
for tasks like skin lesion segmentation, where expert annota-
tions are costly and time-intensive. However, existing meth-
ods suffer from cold-start issues and inefficient sample selec-
tion. This paper introduces a novel active learning framework
called Task-Aligned Iterative Active Learning (TAIAL) that
employs clustering and entropy ranking on a progressively re-
fined feature space to select active samples that balance diver-
sity, informativeness, and uncertainty. Coupled with a self-
supervised initialization step, TAIAL provides an effective
solution for both the cold-start problem and sample selection.
Extensive experiments on the ISIC17 dataset demonstrate that
TAIAL achieves early-stage sample selection performance,
representing a 32% improvement over random sampling and
an average improvement of 27% over other active learning
schemes. In the later stage, it reaches 98.7% of fully super-
vised performance with only 38.4% labeled data, outperform-
ing baseline methods. Our approach provides a scalable and
efficient active learning paradigm for annotation-constrained
medical imaging applications.

## 🏗️ Architecture

```
TAIAL Framework
├── Stage 1: Self-Supervised Learning
│   ├── Feature Initialization with U-Net
│   └── Contrastive Learning with Augmentations
├── Stage 2: Initial Clustering
│   ├── Feature Extraction from Bottleneck
│   ├── PCA Dimensionality Reduction
│   └── GMM Clustering with Likelihood Ranking
└── Stage 3: Iterative Active Learning
    ├── Adaptive GMM Re-clustering
    ├── Cluster-Based Probability (CBS) Sample Selection
    ├── Entropy-based (EBR) Sample Selection
    ├── Task-aligned Model Training
    └── Performance Evaluation
```

## 📋 Prerequisites

- Python 3.8+
- PyTorch 12.1+ with CUDA support
- Required dependencies (see installation)
- ISIC 2017 dataset

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/chanakatb/adaptive-gmmal-seg
cd adaptive-gmmal-seg
```

2. **Create virtual environment:**
```bash
python -m venv taial_env
source taial_env/bin/activate  # On Windows: taial_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

4. **Setup dataset:**
```bash
# Download ISIC 2017 dataset and organize as:
ISIC_2017_dataset/
├── data/
│   ├── train_images/
│   ├── train_masks/
│   ├── val_images/
│   ├── val_masks/
│   ├── test_images/
│   └── test_masks/
```

## 🔧 Project Structure

```
taial-framework/
├── src/
│   └── model.py                    # U-Net architecture
├── utils/
│   └── utils.py                   # Data loading and utilities
├── scripts/
│   ├── self_supervised_learning.py        # Stage 1: SSL pre-training
│   ├── features_extraction_fun.py         # Stage 2.1:Feature extraction
│   ├── gmm_with_likelihood_fun.py         # Stage 2.2:Initial GMM clustering
│   ├── run_adaptive_gmmal-seg.py          # Stage 3:Main TAIAL framework
│   ├── run_config.py                      # Configuration parameters
|   ├── gmm_with_likelihood_adaptive_gmm_fun.py  # Adaptive GMM implementation
│   ├── entropy_calculation.py             # Uncertainty estimation
│   ├── loss_function.py                  # Combined loss functions
│   ├── train_fully_superviced_learning.py # Baseline training/(testing)
│   └── test_fully_superviced_learning.py  # Baseline testing
└── requirements.txt              # Dependencies
```

## 📚 Quick Start Guide

To run the complete TAIAL framework, follow these 4 steps in order:

1. **First run `self_supervised_learning.py`** and train the model
2. **Second run `features_extraction_fun.py`** - give the trained model from Step 1 as input along with Train and Validation image directories  
3. **Run `gmm_with_likelihood_fun.py`** to get the initial GMM - give `features_path` and `filenames_path` as inputs which are computed in Step 2
4. **Run `run_adaptive_gmmal-seg.py`** - update the parameters defined in `run_config.py`. `BASE_RANKED_CLUSTERS_FILE` should be the one saved in Step 3 as `ranked_cluster_assignments.csv` and `PATH_TO_TRAINED_MODEL` should be the one saved in Step 1

**Baseline Comparison:** Use `train_fully_superviced_learning.py` and `test_fully_superviced_learning.py` for fully supervised training and testing.

## 📚 Step-by-Step Execution Guide

### Step 1: Self-Supervised Learning (Pre-training)

**First run `self_supervised_learning.py` and train the model.**

Train the U-Net model using self-supervised learning to learn meaningful feature representations.

```bash
python scripts/self_supervised_learning.py
```

**What this does:**
- Trains a U-Net model using self-supervised learning with augmented images
- Saves the trained model checkpoint (`.pt` file)
- Creates validation images to monitor training progress

**Expected Output:**
- Trained model saved in: `results/ssl_trained_model/{timestamp}/self_supervised_learning.pt`
- Validation images saved in: `results/ssl_validation_images/{timestamp}/`

**⚠️ Important:** Note the model path - you'll need it for subsequent steps.

---

### Step 2: Feature Extraction

**Second run `features_extraction_fun.py` - give the trained model from Step 1 as input along with Train and Validation image directories.**

Extract bottleneck features from the trained self-supervised model.

**Before running:**
1. Open `scripts/features_extraction_fun.py`
2. Update the `PATH_TO_TRAINED_MODEL` variable with the model path from Step 1:
   ```python
   PATH_TO_TRAINED_MODEL = 'results/ssl_trained_model/{your_timestamp}/self_supervised_learning.pt'
   ```
3. Ensure your train and validation image directories are correctly set:
   ```python
   TRAIN_IMG_DIR = "../ISIC_2017_dataset/data/train_images/"
   VAL_IMG_DIR = "../ISIC_2017_dataset/data/val_images/"
   ```

```bash
python scripts/features_extraction_fun.py
```

**What this does:**
- Loads the pre-trained model from Step 1
- Extracts bottleneck features for all training and validation images
- Saves features and corresponding filenames

**Expected Output:**
- Features saved in: `results/ssl_features_space/features_space.pt`
- Filenames saved in: `results/ssl_features_space/file_names.txt` and `file_names.csv`

---

### Step 3: Initial GMM Clustering

**Run `gmm_with_likelihood_fun.py` to get the initial GMM - give `features_path` and `filenames_path` as inputs which are computed in Step 2.**

Perform initial Gaussian Mixture Model clustering on the extracted features.

**Before running:**
1. Open `scripts/gmm_with_likelihood_fun.py`
2. Update the paths in the `main()` function with outputs from Step 2:
   ```python
   def main():
       output_path, ranked_df_path = run_gmm_clustering(
           features_path="results/ssl_features_space/features_space.pt",        # From Step 2
           filenames_path="results/ssl_features_space/file_names.txt",          # From Step 2
           n_components=10,
           pca_components=20,
           output_base="results/cluster_info"
       )
       return output_path, ranked_df_path
   ```

```bash
python scripts/gmm_with_likelihood_fun.py
```

**What this does:**
- Applies PCA dimensionality reduction
- Runs GMM clustering with specified number of components
- Ranks samples within each cluster by likelihood
- Creates visualizations

**Expected Output:**
- Clustering results saved in: `results/cluster_info/{timestamp}/`
- **🎯 Key file:** `ranked_cluster_assignments.csv` (needed for Step 4)
- Cluster visualizations and parameter files

---

### Step 4: Configure and Run TAIAL Framework

**Run `run_adaptive_gmmal-seg.py` - update the parameters defined in `run_config.py`. `BASE_RANKED_CLUSTERS_FILE` should be the one saved in Step 3 as `ranked_cluster_assignments.csv` and `PATH_TO_TRAINED_MODEL` should be the one saved in Step 1.**

Configure the parameters and run the main TAIAL framework.

**Configuration:**
1. Open `scripts/run_config.py`
2. Update the following key parameters:

```python
# PATH_TO_TRAINED_MODEL should be the one saved in Step 1
PATH_TO_TRAINED_MODEL = 'results/ssl_trained_model/{your_timestamp}/self_supervised_learning.pt'

# BASE_RANKED_CLUSTERS_FILE should be the one saved in Step 3 as 'ranked_cluster_assignments.csv'
BASE_RANKED_CLUSTERS_FILE = "results/cluster_info/{your_timestamp}/ranked_cluster_assignments.csv"

# Set to True to load the pre-trained model
LOAD_TRAINED_SSL_MODEL = True

# Configure your dataset paths
TRAIN_IMG_DIR = "../ISIC_2017_dataset/data/train_images/"
TRAIN_MASK_DIR = "../ISIC_2017_dataset/data/train_masks/"
VAL_IMG_DIR = "../ISIC_2017_dataset/data/val_images/"
VAL_MASK_DIR = "../ISIC_2017_dataset/data/val_masks/"
TEST_IMG_DIR = "../ISIC_2017_dataset/data/test_images/"
TEST_MASK_DIR = "../ISIC_2017_dataset/data/test_masks/"

# Configure active learning method
ACTIVE_SAMPLE_SELECTION_METHOD = 'EBR'  # 'EBR' or 'CBS'
ACTIVE_SAMPLE_SELECTION_ORDER = False   # False = high entropy first

# Configure sample sizes for each iteration
SAMPLE_SIZES = [300, 335, 370, 405, 440, 475, 510, 545, 580]

# Adaptive GMM parameters
ADAPTIVE_GMM_CONFIG = {
    'input_dim': 20,
    'hidden_dims': [100, 50],
    'latent_dim': 20,
    'n_components': 10,
    'weight_log_likelihood_loss': 1,
    'weight_sep_term': 1,
    'weight_entropy_loss': 0.1,
    'min_size_weight': 0.5,
    'min_cluster_size': 10,
    'proximity_threshold': 5,
    'n_neighbors': 5,
    'pca_components': 20,
    'n_epochs_adpt_gmm': 5,
    'lr_adpt_gmm': 1e-5,
    'freeze_encoder': False
}
```

**Alternative Configuration Approach:**
For easier management, copy the key file to a standard location:
```bash
# After Step 3, copy the file for easier access
cp results/cluster_info/{timestamp}/ranked_cluster_assignments.csv results/cluster_info/ranked_cluster_assignments.csv
```

Then use:
```python
BASE_RANKED_CLUSTERS_FILE = "results/cluster_info/ranked_cluster_assignments.csv"
```

**Run the TAIAL Framework:**
```bash
python scripts/run_adaptive_gmmal-seg.py
```

**What this does:**
- Runs iterative active learning across multiple iterations
- For each iteration:
  - Performs adaptive GMM clustering
  - Calculates entropy for sample selection
  - Selects most informative samples
  - Trains the segmentation model
  - Evaluates on test set
- Saves comprehensive results and visualizations

**Expected Output:**
- Model checkpoints for each iteration
- Training/validation/test results (CSV files)
- Prediction images for visual inspection
- Comprehensive performance metrics

---

## 📊 Baseline Comparisons

### Fully Supervised Learning

Train a model using all available labeled data for comparison:

```bash
python train_fully_superviced_learning.py
```

Test the fully supervised model:

```bash
python test_fully_superviced_learning.py
```

**Before testing:**
Update `PATH_TO_TRAINED_MODEL` in `test_fully_superviced_learning.py` with your trained model path.

---

## 📁 Directory Structure After Execution

```
results/
├── ssl_trained_model/{timestamp}/
│   ├── self_supervised_learning.pt      # Step 1 output
│   └── training_logs/
├── ssl_features_space/
│   ├── features_space.pt                # Step 2 output
│   ├── file_names.txt
│   └── file_names.csv
├── cluster_info/{timestamp}/
│   ├── ranked_cluster_assignments.csv   # Step 3 key output
│   ├── cluster_assignments.csv
│   ├── detailed_cluster_assignments.csv
│   ├── gmm_clusters_2d.png             # Visualizations
│   ├── gmm_clusters_2d.pdf
│   ├── pca_explained_variance.png
│   ├── gmm_summary.txt
│   ├── gmm_weights.npy                 # GMM parameters
│   ├── gmm_means.npy
│   ├── gmm_covariances.npy
│   └── cluster_*_files.txt
├── ssaal_trained_model/{timestamp}/     # Step 4 outputs
│   ├── adaptive_gmmal_seg.pt
│   ├── test_results.csv
│   ├── iteration*_train_results.csv
│   └── model_checkpoints/
├── ssaal_validation_images/{timestamp}/
├── ssaal_test_images/{timestamp}/
├── cluster_info/
│   ├── ranked_cluster_assignments.csv
│   ├── ranked_cluster_assignments_updated.csv
│   ├── ranked_cluster_assignments_prev_selected.csv
│   ├── entropy_results.csv
│   └── selection_logs/
└── fully_supervised_*/                  # Baseline results
```

---

## ⚙️ Configuration Options

### Active Learning Methods

1. **CBS (Cluster-Based Probability Selection):**
   - Selects samples based on cluster membership probabilities
   - Good for maintaining cluster diversity

2. **EBR (Entropy-Based Ranking within Cluster):**
   - Uses prediction entropy for uncertainty estimation
   - Better for identifying truly uncertain samples

### Sample Selection Order

- `ACTIVE_SAMPLE_SELECTION_ORDER = True`: High confidence (low entropy) first
- `ACTIVE_SAMPLE_SELECTION_ORDER = False`: High uncertainty (high entropy) first

### Adaptive GMM Parameters

- `weight_log_likelihood_loss`: Weight for likelihood maximization
- `weight_sep_term`: Weight for cluster separation
- `weight_entropy_loss`: Weight for entropy regularization
- `freeze_encoder`: Whether to freeze encoder during adaptation

---

## 🔧 Troubleshooting

### Common Issues:

1. **Path not found errors:**
   ```bash
   # Ensure all paths in config files point to actual generated files
   # Check timestamps in directory names
   ls results/ssl_trained_model/
   ls results/cluster_info/
   ```

2. **CUDA out of memory:**
   ```python
   # Reduce batch size in configuration
   BATCH_SIZE = 8  # Instead of 16
   
   # Reduce image dimensions
   IMAGE_HEIGHT = 128
   IMAGE_WIDTH = 128
   ```

3. **Dataset path errors:**
   ```bash
   # Verify dataset structure
   ls ../ISIC_2017_dataset/data/
   # Should show: train_images/ train_masks/ val_images/ val_masks/ test_images/ test_masks/
   ```

4. **Import errors:**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Monitoring Progress:

- **Console output:** Real-time training progress
- **CSV files:** Detailed metrics and loss curves
- **Prediction images:** Visual quality assessment
- **Cluster visualizations:** Sample distribution analysis

---

## ⏱️ Expected Runtime

| Step | Description | Estimated Time |
|------|-------------|----------------|
| Step 1 | Self-supervised learning | 2-4 hours |
| Step 2 | Feature extraction | 10-30 minutes |
| Step 3 | Initial GMM clustering | 5-15 minutes |
| Step 4 | TAIAL framework | 4-8 hours |
| **Total** | **Complete pipeline** | **6-12 hours** |

*Times vary based on hardware, dataset size, and configuration parameters.*

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{XXX,
  title={A Warmer Start to Active Learning with Adaptive Gaussian Mixture Models for Skin Lesion Segmentation},
  author={XXX},
  journal={XXX},
  year={XXX}
}
```

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/chanakatb/adaptive-gmmal-seg/issues)
- **Discussions:** [GitHub Discussions](https://github.com/chanakatb/adaptive-gmmal-seg/discussions)
- **Email:** lakmali.nadeesha@uky.edu / chanaka@udel.edu
---

## 🙏 Acknowledgments

- ISIC 2017 Challenge organizers for the dataset
- The open-source community for various tools and libraries

---

**Happy researching! 🚀**

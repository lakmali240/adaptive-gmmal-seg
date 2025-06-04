import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import pytz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import albumentations as A
from albumentations.pytorch import ToTensorV2
# ---------------------- RMNClustering and Trainer ----------------------
class RMNClustering_adaptive_gmm(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_components, weight_log_likelihood_loss=1, weight_sep_term=0.01, weight_entropy_loss=0.1, min_size_weight=0.05, min_cluster_size=10, proximity_threshold=5.0, n_neighbors=None, freeze_encoder=False):
        super().__init__()
        self.autoencoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim)
        )
        self.rmn_mu = nn.Parameter(torch.randn(n_components, latent_dim))
        self.rmn_log_var = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.rmn_log_pi = nn.Parameter(torch.zeros(n_components))

        self.weight_log_likelihood_loss = weight_log_likelihood_loss
        self.weight_sep_term = weight_sep_term
        self.weight_entropy_loss = weight_entropy_loss
        self.min_size_weight = min_size_weight
        self.min_cluster_size = min_cluster_size
        self.proximity_threshold = proximity_threshold
        self.n_neighbors = n_neighbors if n_neighbors is not None else n_components // 2
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.freeze_encoder = freeze_encoder

    def get_params(self):
        pi = torch.softmax(self.rmn_log_pi, dim=0)
        var = torch.exp(self.rmn_log_var)
        return pi, self.rmn_mu, var

    def forward(self, x):
        z = self.autoencoder(x)
        pi, mu, var = self.get_params()
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        inv_var = 1.0 / var.unsqueeze(0)
        mahalanobis = torch.sum(diff ** 2 * inv_var, dim=2)
        log_det = torch.sum(torch.log(var), dim=1)
        log_probs = -0.5 * (self.latent_dim * np.log(2 * np.pi) + log_det + mahalanobis)
        log_probs += torch.log(pi.unsqueeze(0))
        log_likelihood = torch.logsumexp(log_probs, dim=1)
        responsibilities = torch.softmax(log_probs, dim=1)
        return z, log_likelihood, responsibilities

    def compute_loss(self, x):
        z, log_likelihood, responsibilities = self.forward(x)
        assignments = responsibilities.argmax(dim=1).cpu().numpy()
        # print(f"[INFO] Cluster usage: {np.bincount(assignments, minlength=self.n_components)}")
        log_likelihood_loss = -torch.mean(log_likelihood)

        pi, mu, var = self.get_params()
        dist = torch.cdist(mu, mu, p=2)
        neighbors_k = [torch.topk(dist[k], self.n_neighbors + 1, largest=False).indices[1:] for k in range(self.n_components)]

        separation_term = 0.0
        active_pairs = 0
        for k in range(self.n_components):
            for j in neighbors_k[k]:
                dist_kj = torch.norm(mu[k] - mu[j])
                # print(f"neighbors_k[{k}]: {neighbors_k[k]}")
                # print(f"dist_kj: [{k},{j}]: {dist_kj.item()}")
                if dist_kj < self.proximity_threshold:
                    repulsion_weight = (self.proximity_threshold / (dist_kj + 0.1)) ** 2
                    # print(f"repulsion_weight: [{k}]: {repulsion_weight.item()}")
                    repulsion_weight = 1
                    separation_term -= repulsion_weight * torch.sum((mu[k] - mu[j]) ** 2)
                    active_pairs += 1
        if active_pairs > 0:
            separation_term /= active_pairs
        else:
            separation_term = torch.tensor(0.0, device=x.device)

        entropy = -torch.sum(responsibilities * torch.log(responsibilities + 1e-8), dim=1)
        entropy_loss = -torch.mean(entropy)
        cluster_sizes = responsibilities.sum(dim=0)
        size_penalty = torch.sum(torch.relu(self.min_cluster_size - cluster_sizes))

        loss = self.weight_log_likelihood_loss * log_likelihood_loss + self.weight_sep_term * separation_term + self.weight_entropy_loss * entropy_loss + self.min_size_weight * size_penalty
        return loss

    def initialize_with_em(self, dataloader):
        print("Initializing RMN using GMM EM...")
        all_z = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].cuda()
                z = self.autoencoder(x)
                all_z.append(z.cpu().numpy())
        all_z = np.concatenate(all_z)
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='diag',reg_covar=1e-3 )
        gmm.fit(all_z)
        self.rmn_log_pi.data = torch.log(torch.tensor(gmm.weights_, dtype=torch.float32).to(self.rmn_log_pi.device))
        self.rmn_mu.data = torch.tensor(gmm.means_, dtype=torch.float32).to(self.rmn_mu.device)
        self.rmn_log_var.data = torch.log(torch.tensor(gmm.covariances_, dtype=torch.float32).to(self.rmn_log_var.device))
        print("Initialization complete.")


# ---------------------- Feature Extraction ----------------------
def extract_bottleneck_features_adaptive_gmm(model, dataloader):
    model.eval()
    features = []
    with torch.no_grad():
        for x in dataloader:
            x = x[0].cuda()
            _, bottleneck = model(x)
            pooled = F.adaptive_avg_pool2d(bottleneck, 1).squeeze(-1).squeeze(-1)
            features.append(pooled.cpu())
    return torch.cat(features, dim=0).numpy()



def extract_features_adaptive_gmm(
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
    model__adpt = UNET(in_channels=3, out_channels=1).to(device)
    
    # Load pre-trained model
    print("\nLoading pre-trained model from:", path_to_trained_model)
    checkpoint = torch.load(path_to_trained_model, map_location=device, weights_only=False)
    
    # Check checkpoint keys
    print("Available keys in checkpoint:", checkpoint.keys())
    model__adpt.load_state_dict(checkpoint['state_dict'])
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
    features, file_names = _extract_features_fn_adaptive_gmm(train_loader, model__adpt, device)
    
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
    return features, file_names

def _extract_features_fn_adaptive_gmm(loader, model, device):
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


def apply_pca_if_needed_adaptive_gmm(features, pca_components=20):
    """
    Apply PCA to the features if pca_components > 0.
    PCA is applied on flattened features of shape [N, D],
    using PCA(n_components=pca_components) from scikit-learn,
    only if pca_components > 0.
    """
    if isinstance(features, torch.Tensor):
        features = features.view(features.shape[0], -1).cpu().numpy()

    if pca_components == 0:
        print("Skipping PCA dimensionality reduction.")
        return features, None

    print(f"Applying PCA to reduce dimensions from {features.shape[1]} to {pca_components}...")
    pca = PCA(n_components=pca_components)
    reduced_features = pca.fit_transform(features)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"Reduced features shape: {reduced_features.shape}")
    print(f"Total explained variance: {explained_variance[-1]:.4f}")

    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of PCA Components')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_explained_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("PCA explained variance plot saved to 'pca_explained_variance.png'")

    return reduced_features, pca


# ------------------ Visualization ------------------
def visualize_clusters_adaptive_gmm(model, dataloader, epoch, save_dir="gmm_evolution"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_z, all_resp = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].cuda()
            z, _, responsibilities = model(x)
            all_z.append(z.cpu().numpy())
            all_resp.append(responsibilities.cpu().numpy())

    all_z = np.concatenate(all_z)
    all_resp = np.concatenate(all_resp)
    assignments = np.argmax(all_resp, axis=1)

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_z)

    pi, mu, var = model.get_params()
    mu_np = mu.detach().cpu().numpy()
    var_np = var.detach().cpu().numpy()
    mu_2d = pca.transform(mu_np)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, model.n_components))

    for k in range(model.n_components):
        mask = assignments == k
        if np.any(mask):
            plt.scatter(z_2d[mask, 0], z_2d[mask, 1], s=20, color=colors[k], label=f"Cluster {k}", alpha=0.6)
            cov_k = np.diag(var_np[k])
            cov_k_2d = pca.components_ @ cov_k @ pca.components_.T
            evals, evecs = np.linalg.eigh(cov_k_2d)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
            width, height = 2 * np.sqrt(5.991 * evals)
            ellipse = plt.matplotlib.patches.Ellipse(
                xy=mu_2d[k], width=width, height=height, angle=angle,
                edgecolor=colors[k], linestyle='--', fill=False, linewidth=2, alpha=0.6)
            plt.gca().add_patch(ellipse)

    plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c='black', s=100, marker='X', label='Centroids', edgecolors='white')
    for i, (x, y) in enumerate(mu_2d):
        plt.annotate(f"x$_{{{i}}}$", (x, y), xytext=(10, 10), textcoords='offset points',
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))

    plt.title(f"GMM Clustering - Epoch {epoch}")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename = f"{save_dir}/epoch_{epoch:03d}.png"
    plt.savefig(filename)
    plt.close()


def create_gif_adaptive_gmm(save_dir="gmm_evolution", gif_name="clustering_evolution.gif", duration=0.4):
    print("Creating GIF...")
    images = []
    files = sorted([f for f in os.listdir(save_dir) if f.endswith(".png")])
    for file in files:
        filepath = os.path.join(save_dir, file)
        images.append(imageio.imread(filepath))
    gif_path = os.path.join(save_dir, gif_name)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved at {gif_path}")

def generate_cluster_assignments_adaptive_gmm(model, dataloader, file_names, output_dir="results/rmn_clustering"):
    """
    Generate ranked cluster assignments similar to the GMM function output
    
    Args:
        model: Trained RMN clustering model
        dataloader: DataLoader used for training
        file_names: List of filenames corresponding to the data
        output_dir: Directory to save results
        
    Returns:
        Path to the ranked cluster assignments CSV file
    """
    import os
    import torch
    import numpy as np
    from datetime import datetime
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    
    model.eval()
    all_responsibilities = []
    all_filenames = []
    
    # Extract responsibilities (cluster probabilities) for all data
    with torch.no_grad():
        batch_idx = 0
        for batch in dataloader:
            x = batch[0].cuda()
            _, _, responsibilities = model(x)
            all_responsibilities.append(responsibilities.cpu().numpy())
            
            # Get corresponding filenames for this batch
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + dataloader.batch_size, len(file_names))
            batch_files = file_names[start_idx:end_idx]
            all_filenames.extend(batch_files)
            
            batch_idx += 1
    
    # Concatenate all responsibilities
    all_responsibilities = np.concatenate(all_responsibilities, axis=0)
    
    # Get cluster assignments (highest probability cluster for each sample)
    labels = np.argmax(all_responsibilities, axis=1)
    
    # Calculate cluster probabilities (same as responsibilities)
    cluster_probs = all_responsibilities
    
    # Print cluster distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nRMN Cluster Distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    # Create cluster assignments with likelihoods
    cluster_samples = {}
    for i, (filename, label, probs) in enumerate(zip(all_filenames, labels, cluster_probs)):
        likelihood = probs[label]  # Probability of belonging to assigned cluster
        if label not in cluster_samples:
            cluster_samples[label] = []
        cluster_samples[label].append((filename, likelihood, i))
    
    # Rank samples within each cluster by likelihood
    ranked_samples = []
    ranked_by_cluster = {}
    
    for cluster in sorted(cluster_samples.keys()):
        # Sort samples in this cluster by likelihood (descending)
        sorted_samples = sorted(cluster_samples[cluster], key=lambda x: x[1], reverse=True)
        ranked_by_cluster[cluster] = []
        
        # Assign ranks within the cluster (1 for highest likelihood)
        for rank, (filename, likelihood, orig_idx) in enumerate(sorted_samples, 1):
            ranked_samples.append((filename, cluster, likelihood, rank, orig_idx))
            ranked_by_cluster[cluster].append((filename, likelihood, rank))
    
    # Sort back to original order
    ranked_samples.sort(key=lambda x: x[4])
    
    # Save ranked cluster assignments
    ranked_df_path = os.path.join(output_path, 'ranked_cluster_assignments.csv')
    with open(ranked_df_path, 'w') as f:
        f.write('filename,cluster,likelihood,rank\n')
        for filename, cluster, likelihood, rank, _ in ranked_samples:
            f.write(f'{filename},{cluster},{likelihood:.6f},{rank}\n')
    
    # print(f"Ranked cluster assignments saved to {ranked_df_path}")
    
    # Save basic cluster assignments
    cluster_df_path = os.path.join(output_path, 'cluster_assignments.csv')
    with open(cluster_df_path, 'w') as f:
        f.write('filename,cluster,likelihood\n')
        for filename, label, probs in zip(all_filenames, labels, cluster_probs):
            likelihood = probs[label]
            f.write(f'{filename},{label},{likelihood:.6f}\n')
    
    # print(f"Cluster assignments saved to {cluster_df_path}")
    
    # Save detailed assignments with all component probabilities
    detailed_df_path = os.path.join(output_path, 'detailed_cluster_assignments.csv')
    with open(detailed_df_path, 'w') as f:
        header = 'filename,cluster'
        for i in range(model.n_components):
            header += f',prob_component_{i}'
        header += '\n'
        f.write(header)
        
        for filename, label, prob in zip(all_filenames, labels, cluster_probs):
            line = f'{filename},{label}'
            for p in prob:
                line += f',{p:.6f}'
            line += '\n'
            f.write(line)
    
    # print(f"Detailed cluster assignments saved to {detailed_df_path}")
    
    # Save files per cluster (ranked by likelihood)
    for cluster in sorted(ranked_by_cluster.keys()):
        cluster_files = ranked_by_cluster[cluster]
        cluster_file_path = os.path.join(output_path, f'cluster_{cluster}_files.txt')
        with open(cluster_file_path, 'w') as f:
            for filename, likelihood, rank in cluster_files:
                f.write(f'{filename},{rank}\n')
        # print(f"Files for cluster {cluster} saved to {cluster_file_path}")
    
    # Save model parameters
    pi, mu, var = model.get_params()
    
    weights_path = os.path.join(output_path, "rmn_weights.npy")
    means_path = os.path.join(output_path, "rmn_means.npy")
    variances_path = os.path.join(output_path, "rmn_variances.npy")
    
    np.save(weights_path, pi.detach().cpu().numpy())
    np.save(means_path, mu.detach().cpu().numpy())
    np.save(variances_path, var.detach().cpu().numpy())
    
    # print(f"RMN parameters saved to {output_path}")
    
    # Save summary
    summary_path = os.path.join(output_path, "rmn_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"RMN Clustering with {model.n_components} components\n")
        f.write(f"Latent dimension: {model.latent_dim}\n")
        f.write(f"Alpha (likelihood weight): {model.weight_sep_term}\n")
        f.write(f"Gamma (separation weight): {model.weight_entropy_loss}\n")
        f.write(f"Neeta (entropy weight): {model.min_size_weight}\n\n")
        f.write("Mixture Weights:\n")
        pi_np = pi.detach().cpu().numpy()
        for i, weight in enumerate(pi_np):
            f.write(f"Component {i}: {weight:.4f}\n")
        
        f.write("\nCluster distributions:\n")
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            f.write(f"Cluster {label}: {count} samples ({count/len(labels)*100:.2f}%)\n")
    
    # print(f"RMN summary saved to {summary_path}")
    # print(f"All RMN clustering results saved to {output_path}")
    
    return ranked_df_path, output_path

def train_model_adaptive_gmm(model, dataloader, file_names, n_epochs_adpt_gmm=10, lr=1e-3):
    model.train()
    model.initialize_with_em(dataloader)

    # Freeze encoder weights if specified
    if model.freeze_encoder:
        print("[INFO] Freezing encoder weights...")
        for param in model.autoencoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    visualize_clusters_adaptive_gmm(model, dataloader, epoch=0)

    for epoch in range(n_epochs_adpt_gmm):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].cuda()
            loss = model.compute_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Total Loss = {total_loss / len(dataloader):.4f}")
        visualize_clusters_adaptive_gmm(model, dataloader, epoch=epoch+1)

    create_gif_adaptive_gmm()

    ranked_csv_path, output_dir = generate_cluster_assignments_adaptive_gmm(model, dataloader, file_names)
    return ranked_csv_path, output_dir


def run_rmn_clustering_pipeline(
    input_dim=20,
    hidden_dims=[100, 50],
    latent_dim=20,
    n_components=10,
    weight_log_likelihood_loss=1,
    weight_sep_term=1,
    weight_entropy_loss=0.3,
    min_size_weight=0.3,
    min_cluster_size=10,
    proximity_threshold=5,
    n_neighbors=5,
    pca_components=20,
    PATH_TO_TRAINED_MODEL=None,
    FEATURE_SPACE_DIRECTORY=None,
    train_img_dir=None,
    val_img_dir=None,
    iteration_num_adpt_gmm=1,
    n_epochs_adpt_gmm=10,
    lr_adpt_gmm=1e-3,
    freeze_encoder=False
):
    """
    Run the complete RMN clustering pipeline with specified parameters.
    
    Args:
        input_dim (int): Input dimension for the autoencoder
        hidden_dims (list): Hidden layer dimensions for the autoencoder
        latent_dim (int): Latent dimension for the autoencoder
        n_components (int): Number of mixture components
        weight_log_likelihood_loss (float): Weight for log-likelihood loss
        weight_sep_term (float): Weight for separation term
        weight_entropy_loss (float): Weight for entropy loss
        min_size_weight (float): Weight for minimum size penalty
        min_cluster_size (int): Minimum cluster size
        proximity_threshold (float): Threshold for proximity-based separation
        n_neighbors (int): Number of neighbors for separation term
        pca_components (int): Number of PCA components (0 to skip PCA)
        PATH_TO_TRAINED_MODEL (str): Path to the trained model weights
        FEATURE_SPACE_DIRECTORY (str): Directory to save extracted features
        TRAIN_IMG_DIR (str): Directory containing training images
        VAL_IMG_DIR (str): Directory containing validation images
        iteration_num (int): Iteration number for tracking
        n_epochs (int): Number of training epochs
        lr (float): Learning rate for training
        
    Returns:
        tuple: (ranked_csv_path, output_dir) - Paths to the results
    """
    
    print("Starting RMN Clustering Pipeline...")
    print(f"Parameters:")
    print(f"  - Model: input_dim={input_dim}, hidden_dims={hidden_dims}, latent_dim={latent_dim}")
    print(f"  - Clustering: n_components={n_components}, weight_log_likelihood_loss={weight_log_likelihood_loss}, weight_sep_term={weight_sep_term}, weight_entropy_loss={weight_entropy_loss}")
    print(f"  - Constraints: min_size_weight={min_size_weight}, min_cluster_size={min_cluster_size}")
    print(f"  - Proximity: proximity_threshold={proximity_threshold}, n_neighbors={n_neighbors}")
    print(f"  - PCA: pca_components={pca_components}")
    print(f"  - Training: n_epochs={n_epochs_adpt_gmm}, lr={lr_adpt_gmm}")
    
    # Step 1: Extract Features
    # print("\n=== Step 1: Feature Extraction ===")
    features_adpt_gmm, file_names_adpt_gmm = extract_features_adaptive_gmm(
        path_to_trained_model=PATH_TO_TRAINED_MODEL,
        feature_space_directory=FEATURE_SPACE_DIRECTORY,
        train_img_dir=train_img_dir,
        valid_img_dir=val_img_dir,
        iter=iteration_num_adpt_gmm
    )

    # Step 2: Apply PCA
    # print("\n=== Step 2: PCA Dimensionality Reduction ===")
    reduced_features_adpt_gmm, pca_model_adpt_gmm = apply_pca_if_needed_adaptive_gmm(features_adpt_gmm, pca_components=pca_components)

    # Step 3: Prepare DataLoader
    # print("\n=== Step 3: Preparing DataLoader ===")
    reduced_tensor_adpt_gmm = torch.tensor(reduced_features_adpt_gmm, dtype=torch.float32)
    # print("Reduced tensor shape:", reduced_tensor_adpt_gmm.shape)
    clustering_dataset_adpt_gmm = TensorDataset(reduced_tensor_adpt_gmm)
    clustering_loader_adpt_gmm = DataLoader(clustering_dataset_adpt_gmm, batch_size=128, shuffle=False)  # Don't shuffle to maintain order

    # Step 4: Initialize and Train RMN Clustering Model
    # print("\n=== Step 4: RMN Clustering Training ===")
    model_adpt_gmm = RMNClustering_adaptive_gmm(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        n_components=n_components,
        weight_log_likelihood_loss=weight_log_likelihood_loss,
        weight_sep_term=weight_sep_term,
        weight_entropy_loss=weight_entropy_loss,
        min_size_weight=min_size_weight,
        min_cluster_size=min_cluster_size,
        proximity_threshold=proximity_threshold,
        n_neighbors=n_neighbors,
        freeze_encoder=freeze_encoder
    ).cuda()

    # Train model and generate cluster assignments
    print("Starting adaptive gmm model training...")
    ranked_csv_path_adpt_gmm, output_dir_adpt_gmm = train_model_adaptive_gmm(
        model_adpt_gmm, clustering_loader_adpt_gmm, file_names_adpt_gmm, n_epochs_adpt_gmm=n_epochs_adpt_gmm, lr=lr_adpt_gmm
    )
    
    # print(f"\n=== Pipeline Complete ===")
    print(f"Ranked cluster assignments: {ranked_csv_path_adpt_gmm}")
    # print(f"All results directory: {output_dir_adpt_gmm}")
    
    return ranked_csv_path_adpt_gmm, output_dir_adpt_gmm


# Example usage
if __name__ == "__main__":
    # Example with default parameters
    ranked_csv_path_adpt_gmm, output_dir_adpt_gmm = run_rmn_clustering_pipeline(
        input_dim=20,
        hidden_dims=[100, 50],
        latent_dim=20,
        n_components=10,
        weight_log_likelihood_loss=1,
        weight_sep_term=1,
        weight_entropy_loss=0.3,
        min_size_weight=0.3,
        min_cluster_size=10,
        proximity_threshold=5,
        n_neighbors=5,
        pca_components=20,
        PATH_TO_TRAINED_MODEL='results/ssl_trained_model/2025-04-19_01-19-21/self_supervised_learning.pt',
        FEATURE_SPACE_DIRECTORY="results/ssl_features_space",
        train_img_dir="../ISIC_2017_dataset/data/train_images/",
        val_img_dir="../ISIC_2017_dataset/data/val_images/",
        iteration_num_adpt_gmm=1,
        n_epochs_adpt_gmm=10,
        lr_adpt_gmm=1e-3,
        freeze_encoder=False        
    )
    print("ranked_csv_path_adpt_gmm: ", ranked_csv_path_adpt_gmm)
    print("output_dir_adpt_gmm: ", output_dir_adpt_gmm)
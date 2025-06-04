import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import os
from datetime import datetime
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run GMM on feature vectors')
    parser.add_argument('--features', type=str, default="results/ssl_features_space/features_space.pt",
                        help='Path to features_space.pt file')
    parser.add_argument('--filenames', type=str, default="results/ssl_features_space/file_names.txt",
                        help='Path to file_names.txt file')
    parser.add_argument('--components', type=int, default=10,
                        help='Number of Gaussian components')
    parser.add_argument('--pca_components', type=int, default=20,
                        help='Number of PCA components (0 to skip PCA)')
    parser.add_argument('--output_base', type=str, default="results/gmm_results",
                        help='Base output directory for results')
    
    args = parser.parse_args()
    return args

def load_features_and_filenames(features_path, filenames_path):
    """
    Load the extracted features and corresponding filenames
    
    Args:
        features_path: Path to the features_space.pt file
        filenames_path: Path to the file_names.txt file
        
    Returns:
        features: Tensor containing the feature vectors
        filenames: List of corresponding filenames
    """
    # Load features tensor
    try:
        features = torch.load(features_path)
        original_shape = features.shape
        print(f"Features loaded successfully. Original shape: {original_shape}")
        
        # Reshape the features tensor to 2D for GMM
        if len(original_shape) == 4:  # [batch_size, channels, height, width]
            # Flatten all dimensions except batch
            features = features.view(original_shape[0], -1)
            print(f"Reshaped features to: {features.shape}")
        elif len(original_shape) > 2:
            # Flatten any tensor with more than 2 dimensions
            features = features.view(original_shape[0], -1)
            print(f"Reshaped features to: {features.shape}")
        
    except Exception as e:
        print(f"Error loading features: {e}")
        return None, None
    
    # Load filenames
    try:
        with open(filenames_path, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        print(f"Filenames loaded successfully. Total files: {len(filenames)}")
    except Exception as e:
        print(f"Error loading filenames: {e}")
        return features, None
    
    # Verify matching lengths
    if features.shape[0] != len(filenames):
        print(f"Warning: Number of features ({features.shape[0]}) doesn't match number of filenames ({len(filenames)})")
    
    return features, filenames

def apply_pca(features, n_components=100, output_dir=None):
    """
    Apply PCA dimensionality reduction
    
    Args:
        features: Feature vectors (2D tensor or array)
        n_components: Number of PCA components to keep (0 to skip PCA)
        output_dir: Directory to save PCA visualization
        
    Returns:
        reduced_features: PCA-reduced features
        pca: Fitted PCA model
    """
    # Skip PCA if n_components is 0
    if n_components == 0:
        print("Skipping PCA dimensionality reduction.")
        if isinstance(features, torch.Tensor):
            return features.cpu().numpy(), None
        return features, None
    
    # Convert to numpy if needed
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = features
    
    print(f"Applying PCA to reduce dimensions from {features_np.shape[1]} to {n_components}...")
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features_np)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Reduced features shape: {reduced_features.shape}")
    print(f"Total explained variance: {cumulative_variance[-1]:.4f}")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of PCA Components')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the PCA explained variance plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        variance_plot_path = os.path.join(output_dir, "pca_explained_variance.png")
        plt.savefig(variance_plot_path, dpi=300, bbox_inches='tight')
        print(f"PCA explained variance plot saved to {variance_plot_path}")
    
    plt.close()
    
    return reduced_features, pca

def run_gaussian_mixture_model(features, n_components):
    """
    Run Gaussian Mixture Model on the feature vectors with diagonal covariance
    
    Args:
        features: Feature vectors (2D array)
        n_components: Number of Gaussian components to use
        
    Returns:
        model: Trained GMM model
        labels: Cluster assignments for each sample
    """
    print(f"Running GMM with {n_components} components on {features.shape[0]} samples...")
    print(f"Each sample has {features.shape[1]} features")
    print(f"Using covariance type: 'diag' (diagonal)")
    
    # Initialize and fit GMM with diagonal covariance
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='diag',  # Using diagonal covariance to save memory
        max_iter=100,
        random_state=42,
        verbose=1
    )
    
    gmm.fit(features)
    
    # Get cluster assignments
    labels = gmm.predict(features)
    
    # Print GMM results
    print("\nGMM Results:")
    print(f"Converged: {gmm.converged_}")
    print(f"Number of iterations: {gmm.n_iter_}")
    print(f"Lower bound: {gmm.lower_bound_:.4f}")
    
    print("\nMixture Weights:")
    for i, weight in enumerate(gmm.weights_):
        print(f"Component {i}: {weight:.4f}")
    
    print(f"\nMeans shape: {gmm.means_.shape}")
    print(f"Covariances shape: {gmm.covariances_.shape}")
    
    # Count samples per cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nCluster Distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    return gmm, labels

def visualize_clusters_2d(features, labels, output_dir=None, gmm=None):
    """
    Create an enhanced 2D visualization of clusters with centroids labeled from x₀
    
    Args:
        features: Feature vectors
        labels: Cluster assignments
        output_dir: Directory to save visualization
        gmm: The GMM model (optional)
    """
    # If features have more than 2 dimensions, apply PCA
    if features.shape[1] > 2:
        vis_pca = PCA(n_components=2)
        features_2d = vis_pca.fit_transform(features)
        # Calculate explained variance
        explained_variance = vis_pca.explained_variance_ratio_
        print(f"\nVisualization PCA explained variance: {explained_variance[0]:.4f}, {explained_variance[1]:.4f}")
        print(f"Total explained variance: {sum(explained_variance):.4f}")
    else:
        features_2d = features
        explained_variance = [0, 0]  # Placeholder
    
    # Create a more appealing figure
    plt.figure(figsize=(14, 12))
    
    # Set up a nicer plot style - check if 'seaborn-v0_8-whitegrid' is available, if not use 'whitegrid'
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')  # Older versions
        except:
            pass  # Fall back to default style if neither is available
    
    # Get unique labels and count
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Use a better color palette - spectral for more distinguishable colors
    # when there are many clusters
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster with a different color and fixed marker size
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_points = features_2d[mask]
        
        # Use a fixed marker size to avoid kdtree issues
        marker_size = 30  # Medium size that works well for most visualizations
        
        # Scatter plot with fixed marker size
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1],
            s=marker_size,
            label=f'Cluster {label}',
            color=colors[i],
            alpha=0.7,
            edgecolors='w',
            linewidths=0.2
        )
    
    # Add and label cluster centroids if GMM is provided, starting from x₀
    if gmm is not None and features.shape[1] > 2:
        # Project the centroids to 2D space using the same PCA
        centroids_2d = vis_pca.transform(gmm.means_)
        
        # Plot centroids
        plt.scatter(
            centroids_2d[:, 0], 
            centroids_2d[:, 1],
            s=200, 
            marker='X',
            color='black',
            label='Centroids',
            edgecolors='white',
            linewidths=2,
            zorder=10  # Ensure centroids are on top
        )
        
        # Label each centroid with x₀, x₁, etc.
        for i, (x, y) in enumerate(centroids_2d):
            # Use subscript notation for centroid labels, starting from 0
            plt.annotate(
                f'x$_{{{i}}}$',  # Use LaTeX notation for subscript starting from 0
                (x, y),
                xytext=(10, 10),  # Offset text 10 points to the right and up
                textcoords='offset points',
                fontsize=16,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                zorder=11  # Ensure labels are on top
            )
    
    # Enhance plot appearance
    plt.title('GMM Clustering Results (2D Projection)', fontsize=16, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)', fontsize=14)
    
    # Improve legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add grid and adjust layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, "gmm_clusters_2d.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"2D cluster visualization saved to {fig_path}")
        
        # Also save as PDF for high-quality publication
        pdf_path = os.path.join(output_dir, "gmm_clusters_2d.pdf")
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"2D cluster visualization also saved as PDF to {pdf_path}")
    
    plt.close()
    
    return features_2d

def save_cluster_results(labels, filenames, gmm, pca, output_dir):
    """
    Save cluster assignments and create lists of files per cluster
    
    Args:
        labels: Cluster assignments
        filenames: List of filenames
        gmm: Trained GMM model
        pca: PCA model used for dimensionality reduction (or None if no PCA)
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all cluster assignments
    cluster_df_path = os.path.join(output_dir, 'cluster_assignments.csv')
    with open(cluster_df_path, 'w') as f:
        f.write('filename,cluster\n')
        for filename, label in zip(filenames, labels):
            f.write(f'{filename},{label}\n')
    print(f"Cluster assignments saved to {cluster_df_path}")
    
    # Save files per cluster
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_files = [filenames[i] for i, l in enumerate(labels) if l == label]
        cluster_file_path = os.path.join(output_dir, f'cluster_{label}_files.txt')
        with open(cluster_file_path, 'w') as f:
            for filename in cluster_files:
                f.write(f'{filename}\n')
        print(f"Files for cluster {label} saved to {cluster_file_path}")
    
    # Save GMM parameters
    weights_path = os.path.join(output_dir, "gmm_weights.npy")
    means_path = os.path.join(output_dir, "gmm_means.npy")
    covariances_path = os.path.join(output_dir, "gmm_covariances.npy")
    
    np.save(weights_path, gmm.weights_)
    np.save(means_path, gmm.means_)
    np.save(covariances_path, gmm.covariances_)
    print(f"GMM parameters saved to {output_dir}")
    
    # Save PCA model if used
    if pca is not None:
        pca_components_path = os.path.join(output_dir, "pca_components.npy")
        pca_mean_path = os.path.join(output_dir, "pca_mean.npy")
        pca_explained_variance_path = os.path.join(output_dir, "pca_explained_variance.npy")
        
        np.save(pca_components_path, pca.components_)
        np.save(pca_mean_path, pca.mean_)
        np.save(pca_explained_variance_path, pca.explained_variance_)
        print(f"PCA model saved to {output_dir}")
    
    # Save a summary of GMM parameters in text format, using x₀, x₁ notation
    summary_path = os.path.join(output_dir, "gmm_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"GMM with {gmm.n_components} components\n")
        f.write(f"Covariance type: diagonal\n")
        f.write(f"Converged: {gmm.converged_}\n")
        f.write(f"Number of iterations: {gmm.n_iter_}\n")
        f.write(f"Lower bound: {gmm.lower_bound_:.4f}\n\n")
        
        f.write("Weights (mixture proportions):\n")
        for i, weight in enumerate(gmm.weights_):
            f.write(f"x_{i}: {weight:.4f}\n")
        
        f.write("\nMeans shape: {}\n".format(gmm.means_.shape))
        f.write("Covariances shape: {}\n".format(gmm.covariances_.shape))
        
        f.write("\nCluster distributions:\n")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            f.write(f"x_{i} (Cluster {label}): {count} samples ({count/len(labels)*100:.2f}%)\n")
        
        if pca is not None:
            f.write("\nPCA information:\n")
            f.write(f"Number of components: {pca.n_components_}\n")
            f.write(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}\n")
    
    print(f"GMM summary saved to {summary_path}")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory with timestamp in the requested format
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print parameters
    print("\n=== GMM Analysis Parameters ===")
    print(f"Features path: {args.features}")
    print(f"Filenames path: {args.filenames}")
    print(f"Number of GMM components: {args.components}")
    print(f"Number of PCA components: {args.pca_components}")
    print(f"Output directory: {output_dir}")
    print("==============================\n")
    
    # Load data
    features, filenames = load_features_and_filenames(args.features, args.filenames)
    if features is None:
        print("Failed to load features. Exiting.")
        return
    
    # Apply PCA if specified
    reduced_features, pca = apply_pca(features, args.pca_components, output_dir)
    print(f"\nreduced_features shape: {reduced_features.shape}")
    
    # Run GMM with diagonal covariance
    gmm, labels = run_gaussian_mixture_model(reduced_features, args.components)
    
    # Create enhanced 2D visualization - pass the GMM model explicitly
    visualize_clusters_2d(reduced_features, labels, output_dir, gmm)
    
    # Save results
    if filenames is not None:
        save_cluster_results(labels, filenames, gmm, pca, output_dir)
    
    print(f"\nGMM analysis complete. All results saved to {output_dir}")
    
    # Return the path for convenience
    return output_dir

if __name__ == "__main__":
    output_path = main()
    print(f"Results are available at: {output_path}")
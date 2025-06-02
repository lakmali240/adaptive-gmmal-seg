def run_gmm_clustering(
    features_path="results/ssl_features_space/2025-04-19_17-25-19/features_space.pt",
    filenames_path="results/ssl_features_space/2025-04-19_17-25-19/file_names.txt",
    n_components=10,
    pca_components=20,
    output_base="results/gmm_results"
):
    """
    Run Gaussian Mixture Model clustering on feature vectors
    
    Args:
        features_path: Path to the features_space.pt file
        filenames_path: Path to the file_names.txt file
        n_components: Number of Gaussian components for GMM
        pca_components: Number of PCA components (0 to skip PCA)
        output_base: Base output directory for results
        
    Returns:
        output_dir: Path to the directory with results
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    import os
    from datetime import datetime
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print parameters
    print("\n=== GMM Analysis Parameters ===")
    print(f"Features path: {features_path}")
    print(f"Filenames path: {filenames_path}")
    print(f"Number of GMM components: {n_components}")
    print(f"Number of PCA components: {pca_components}")
    print(f"Output directory: {output_dir}")
    print("==============================\n")
    
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
        return None
    
    # Load filenames
    try:
        with open(filenames_path, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        print(f"Filenames loaded successfully. Total files: {len(filenames)}")
    except Exception as e:
        print(f"Error loading filenames: {e}")
        filenames = None
    
    # Verify matching lengths
    if filenames and features.shape[0] != len(filenames):
        print(f"Warning: Number of features ({features.shape[0]}) doesn't match number of filenames ({len(filenames)})")
    
    # Apply PCA if specified
    if pca_components == 0:
        print("Skipping PCA dimensionality reduction.")
        if isinstance(features, torch.Tensor):
            reduced_features = features.cpu().numpy()
        else:
            reduced_features = features
        pca = None
    else:
        # Convert to numpy if needed
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features
        
        print(f"Applying PCA to reduce dimensions from {features_np.shape[1]} to {pca_components}...")
        
        # Initialize and fit PCA
        pca = PCA(n_components=pca_components)
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
        
        # Save the PCA explained variance plot
        os.makedirs(output_dir, exist_ok=True)
        variance_plot_path = os.path.join(output_dir, "pca_explained_variance.png")
        plt.savefig(variance_plot_path, dpi=300, bbox_inches='tight')
        print(f"PCA explained variance plot saved to {variance_plot_path}")
        
        plt.close()
    
    # Run GMM
    print(f"Running GMM with {n_components} components on {reduced_features.shape[0]} samples...")
    print(f"Each sample has {reduced_features.shape[1]} features")
    print(f"Using covariance type: 'diag' (diagonal)")
    
    # Initialize and fit GMM with diagonal covariance
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='diag',  # Using diagonal covariance to save memory
        max_iter=100,
        random_state=42,
        verbose=1
    )
    
    gmm.fit(reduced_features)
    
    # Get cluster assignments
    labels = gmm.predict(reduced_features)
    
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
    
    # Visualize clusters in 2D
    # If features have more than 2 dimensions, apply PCA
    if reduced_features.shape[1] > 2:
        vis_pca = PCA(n_components=2)
        features_2d = vis_pca.fit_transform(reduced_features)
        # Calculate explained variance
        explained_variance = vis_pca.explained_variance_ratio_
        print(f"\nVisualization PCA explained variance: {explained_variance[0]:.4f}, {explained_variance[1]:.4f}")
        print(f"Total explained variance: {sum(explained_variance):.4f}")
    else:
        features_2d = reduced_features
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
    
    # Save the figure
    fig_path = os.path.join(output_dir, "gmm_clusters_2d.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"2D cluster visualization saved to {fig_path}")
    
    # Also save as PDF for high-quality publication
    pdf_path = os.path.join(output_dir, "gmm_clusters_2d.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"2D cluster visualization also saved as PDF to {pdf_path}")
    
    plt.close()
    
    # Save results if filenames are available
    if filenames:
        # Calculate cluster probabilities
        data_for_likelihood = gmm.predict_proba(reduced_features)
        
        # Save all cluster assignments with likelihood values
        cluster_df_path = os.path.join(output_dir, 'cluster_assignments.csv')
        with open(cluster_df_path, 'w') as f:
            f.write('filename,cluster,likelihood\n')
            for filename, label, probs in zip(filenames, labels, data_for_likelihood):
                # The likelihood of belonging to the assigned cluster is the probability for that cluster
                likelihood = probs[label]
                f.write(f'{filename},{label},{likelihood:.6f}\n')
        print(f"Cluster assignments with likelihoods saved to {cluster_df_path}")
        
        # Create a list of samples with their likelihoods for each cluster
        cluster_samples = {}
        for i, (filename, label, probs) in enumerate(zip(filenames, labels, data_for_likelihood)):
            likelihood = probs[label]
            if label not in cluster_samples:
                cluster_samples[label] = []
            cluster_samples[label].append((filename, likelihood, i))  # Store original index
        
        # For each cluster, sort samples by likelihood in descending order and assign ranks
        ranked_samples = []
        ranked_by_cluster = {}  # Store ranked samples by cluster for cluster_files.txt
        
        for cluster in sorted(cluster_samples.keys()):
            # Sort samples in this cluster by likelihood (descending)
            sorted_samples = sorted(cluster_samples[cluster], key=lambda x: x[1], reverse=True)
            
            # Store the sorted samples for this cluster
            ranked_by_cluster[cluster] = []
            
            # Assign ranks within the cluster (1 for highest likelihood)
            for rank, (filename, likelihood, orig_idx) in enumerate(sorted_samples, 1):
                ranked_samples.append((filename, cluster, likelihood, rank, orig_idx))
                ranked_by_cluster[cluster].append((filename, likelihood, rank))
        
        # Sort back to original order using the stored index
        ranked_samples.sort(key=lambda x: x[4])
        
        # Save the ranked cluster assignments
        ranked_df_path = os.path.join(output_dir, 'ranked_cluster_assignments.csv')
        with open(ranked_df_path, 'w') as f:
            f.write('filename,cluster,likelihood,rank\n')
            for filename, cluster, likelihood, rank, _ in ranked_samples:
                f.write(f'{filename},{cluster},{likelihood:.6f},{rank}\n')
        print(f"Ranked cluster assignments saved to {ranked_df_path}")
        
        # Save detailed cluster assignments with component-wise probabilities
        detailed_df_path = os.path.join(output_dir, 'detailed_cluster_assignments.csv')
        with open(detailed_df_path, 'w') as f:
            header = 'filename,cluster'
            for i in range(gmm.n_components):
                header += f',prob_component_{i}'
            header += '\n'
            f.write(header)
            
            for filename, label, prob in zip(filenames, labels, data_for_likelihood):
                line = f'{filename},{label}'
                for p in prob:
                    line += f',{p:.6f}'
                line += '\n'
                f.write(line)
        print(f"Detailed cluster assignments saved to {detailed_df_path}")
        
        # Save files per cluster with ranks (already sorted by rank)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            # Get the ranked samples for this cluster
            cluster_files = ranked_by_cluster[label]
            cluster_file_path = os.path.join(output_dir, f'cluster_{label}_files.txt')
            with open(cluster_file_path, 'w') as f:
                for filename, likelihood, rank in cluster_files:
                    f.write(f'{filename},{rank}\n')
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
    print(f"\nGMM analysis complete. All results saved to {output_dir}")
    
    return output_dir, ranked_df_path, gmm, reduced_features

def print_gmm_parameters(gmm, feature_names=None):
    """
    Print GMM parameters for each component and feature
    
    Args:
        gmm: The fitted GaussianMixture model
        feature_names: List of feature names (optional)
    """
    n_components = gmm.n_components
    n_features = gmm.means_.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    print(f"\n=== GMM Parameters for {n_components} components ===")
    
    for k in range(n_components):
        print(f"\nComponent {k} (Weight: {gmm.weights_[k]:.4f}):")
        print("  Feature               Mean      Variance")
        print("  -------               ----      --------")
        
        for j in range(n_features):
            # Get mean and variance for this feature
            mean_j = gmm.means_[k, j]
            var_j = gmm.covariances_[k, j]  # For 'diag' covariance type
            
            # Print feature statistics
            print(f"  {feature_names[j]:<20} {mean_j:8.4f}   {var_j:10.4f}")

def main():
    output_path, ranked_df_path, gmm, reduced_features = run_gmm_clustering(
        features_path="results/ssl_features_space/2025-04-19_17-25-19/features_space.pt",
        filenames_path="results/ssl_features_space/2025-04-19_17-25-19/file_names.txt",
        n_components=10,
        pca_components=20,
        output_base="results/gmm_results"
    )

    print(f"Ranked cluster assignments are available at: {ranked_df_path}")
    print(f"Results are available at: {output_path}")

    print('\n---Gaussian Components ---')
    # Create feature names for the reduced features
    feature_names = [f"PC_{i}" for i in range(reduced_features.shape[1])]
    print_gmm_parameters(gmm, feature_names)  # Pass feature names, not the actual features

    return output_path, ranked_df_path, gmm, reduced_features

if __name__== "__main__":
    output_path, ranked_df_path, gmm, reduced_features = main()
    

   





    
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the bottleneck features
features_space = torch.load("results/ssl_features_space/2025-04-19_17-25-19/features_space.pt", weights_only=True)

# Load the file names
with open("results/ssl_features_space/2025-04-19_17-25-19/file_names.txt", "r") as f:
    file_names = [line.strip() for line in f.readlines()]

# Find the index of the specific image
target_image = "ISIC_0000026.jpg"
if target_image in file_names:
    index = file_names.index(target_image)
    # Extract the feature for that specific image
    feature = features_space[index]
    print(f"Feature shape for {target_image}: {feature.shape}")  # Should be 1024×16×16
else:
    print(f"{target_image} not found in the dataset")


# Strategy 1: Visualize individual channels
def plot_individual_channels(feature, num_channels=4):
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 3))
    channel_indices = np.linspace(0, feature.shape[0]-1, num_channels, dtype=int)
    
    for i, channel_idx in enumerate(channel_indices):
        channel = feature[channel_idx].detach().cpu().numpy()
        im = axes[i].imshow(channel, cmap='viridis')
        axes[i].set_title(f'Channel {channel_idx}')
        axes[i].axis('off')
    plt.colorbar(im, ax=axes[-1])
    plt.tight_layout()
    plt.savefig('results/ssl_features_space/2025-04-19_17-25-19/individual_channels.png')
    plt.close()

# Strategy 2: PCA to reduce dimensionality across channels
def plot_pca_reduction(feature):
    # Reshape to [1024, 256]
    reshaped = feature.reshape(feature.shape[0], -1).detach().cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(reshaped.T)  # [256, 3]
    
    # Reshape back to spatial dimensions [16, 16, 3]
    pca_img = transformed.reshape(16, 16, 3)
    
    # Normalize for visualization
    pca_img = (pca_img - pca_img.min()) / (pca_img.max() - pca_img.min())
    
    plt.figure(figsize=(6, 6))
    plt.imshow(pca_img)
    plt.title('PCA Visualization (Top 3 Components)')
    plt.axis('off')
    plt.savefig('results/ssl_features_space/2025-04-19_17-25-19/pca_visualization.png')
    plt.close()

# Strategy 3: Feature aggregation - statistics across channels
def plot_feature_statistics(feature):
    # Calculate statistics across channels dimension
    mean_activation = torch.mean(feature, dim=0).detach().cpu().numpy()
    max_activation = torch.max(feature, dim=0)[0].detach().cpu().numpy()
    std_activation = torch.std(feature, dim=0).detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(mean_activation, cmap='plasma')
    axes[0].set_title('Mean Activation')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(max_activation, cmap='plasma')
    axes[1].set_title('Max Activation')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(std_activation, cmap='plasma')
    axes[2].set_title('Std Deviation')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('results/ssl_features_space/2025-04-19_17-25-19/feature_statistics.png')
    plt.close()

# Run all visualization strategies
plot_individual_channels(feature)
plot_pca_reduction(feature)
plot_feature_statistics(feature)

print("Visualizations saved as PNG files")
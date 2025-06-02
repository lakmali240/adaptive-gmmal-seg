# RMN-based Adaptive GMM Clustering
#
# What the Code Currently Does:
# -----------------------------
# It executes an unsupervised clustering pipeline by:
#
# 1. Pretraining an encoder for feature extraction (Stage 1, simplified).
# 2. Fitting a GMM via EM to these features (Stage 2).
# 3. Jointly optimizing encoder + GMM to increase log-likelihood and push clusters apart (Stage 3).
# 4. Visualizing the evolution of clusters across epochs (PCA plots + GIFs).

import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture

# ------------------ Synthetic Data and Dataset ------------------
def generate_synthetic_data(n_samples=1000, input_dim=20, n_clusters=10):
    np.random.seed(0)
    means = np.random.randn(n_clusters, input_dim) * 5
    stds = np.abs(np.random.randn(n_clusters, input_dim)) + 0.5
    X, y = [], []
    for _ in range(n_samples):
        k = np.random.randint(0, n_clusters)
        sample = means[k] + stds[k] * np.random.randn(input_dim)
        X.append(sample)
        y.append(k)
    return np.array(X, dtype=np.float32), np.array(y)

class SimpleDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx]

# ------------------ RMNClustering Model ------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev, h))
            encoder_layers.append(nn.ReLU())
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev, h))
            decoder_layers.append(nn.ReLU())
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

class RMN(nn.Module):
    def __init__(self, latent_dim, n_components):
        super().__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.log_pi = nn.Parameter(torch.zeros(n_components))
        self.mu = nn.Parameter(torch.randn(n_components, latent_dim))
        self.log_var = nn.Parameter(torch.zeros(n_components, latent_dim))

    def get_params(self):
        pi = torch.softmax(self.log_pi, dim=0)
        var = torch.exp(self.log_var)
        return pi, self.mu, var

    def forward(self, z):
        pi, mu, var = self.get_params()
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        inv_var = 1.0 / var.unsqueeze(0)
        mahalanobis = torch.sum(diff**2 * inv_var, dim=2)
        log_det = torch.sum(torch.log(var), dim=1)
        log_probs = -0.5 * (self.latent_dim * np.log(2 * np.pi) + log_det + mahalanobis)
        log_probs += torch.log(pi.unsqueeze(0))
        log_likelihood = torch.logsumexp(log_probs, dim=1)
        responsibilities = torch.softmax(log_probs, dim=1)
        return log_likelihood, responsibilities

class RMNClustering(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_components, gamma=0.01):
        super().__init__()
        self.autoencoder = AutoEncoder(input_dim, hidden_dims, latent_dim)
        self.rmn = RMN(latent_dim, n_components)
        self.gamma = gamma
        self.n_components = n_components

    def forward(self, x):
        z, x_rec = self.autoencoder(x)
        log_likelihood, responsibilities = self.rmn(z)
        return z, x_rec, log_likelihood, responsibilities

    def compute_loss(self, x, neighbors_k=None):
        z, x_rec, log_likelihood, responsibilities = self.forward(x)
        recon_loss = F.mse_loss(x_rec, x)
        log_likelihood_loss = -torch.mean(log_likelihood)

        pi, mu, var = self.rmn.get_params()
        if neighbors_k is None:
            dist = torch.cdist(mu, mu, p=2)
            neighbors_k = [torch.topk(dist[k], self.n_components // 2 + 1, largest=False).indices[1:] for k in range(self.n_components)]

        separation_term = 0.0
        for k in range(self.n_components):
            for j in neighbors_k[k]:
                separation_term -= torch.sum((mu[k] - mu[j])**2)
        separation_term /= (self.n_components * len(neighbors_k[0]))

        loss = recon_loss + log_likelihood_loss + self.gamma * separation_term
        return loss, recon_loss, log_likelihood_loss, -separation_term

    def initialize_with_em(self, dataloader):
        print("Initializing RMN using GMM EM...")
        all_z = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch
                z, _ = self.autoencoder(x)
                all_z.append(z.cpu().numpy())
        all_z = np.concatenate(all_z)
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='diag')
        gmm.fit(all_z)
        self.rmn.log_pi.data = torch.log(torch.tensor(gmm.weights_, dtype=torch.float32))
        self.rmn.mu.data = torch.tensor(gmm.means_, dtype=torch.float32)
        self.rmn.log_var.data = torch.log(torch.tensor(gmm.covariances_, dtype=torch.float32))
        print("Initialization complete.")


# ------------------ Visualization ------------------
def visualize_clusters(model, dataloader, epoch, save_dir="gmm_evolution"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_z, all_resp = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch
            z, _, _, responsibilities = model(x)
            all_z.append(z.cpu().numpy())
            all_resp.append(responsibilities.cpu().numpy())

    all_z = np.concatenate(all_z)
    all_resp = np.concatenate(all_resp)
    assignments = np.argmax(all_resp, axis=1)

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_z)

    pi, mu, var = model.rmn.get_params()
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


def create_gif(save_dir="gmm_evolution", gif_name="clustering_evolution.gif", duration=0.4):
    print("Creating GIF...")
    images = []
    files = sorted([f for f in os.listdir(save_dir) if f.endswith(".png")])
    for file in files:
        filepath = os.path.join(save_dir, file)
        images.append(imageio.imread(filepath))
    gif_path = os.path.join(save_dir, gif_name)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved at {gif_path}")


def train_model(model, dataloader, n_epochs=50, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.initialize_with_em(dataloader)
    visualize_clusters(model, dataloader, epoch=0)  # Initial state
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch
            loss, rloss, llik, sep = model.compute_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Total Loss = {total_loss / len(dataloader):.4f}")
        visualize_clusters(model, dataloader, epoch=epoch+1)
    create_gif()


if __name__ == "__main__":
    X, y = generate_synthetic_data(n_samples=1000, input_dim=20, n_clusters=10)
    dataset = SimpleDataset(X)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = RMNClustering(
        input_dim=20,
        hidden_dims=[100, 50],
        latent_dim=10,
        n_components=10,
        gamma=0.01
    )

    train_model(model, dataloader, n_epochs=30, lr=1e-3)
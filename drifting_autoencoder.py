"""
Drifting Autoencoder from Scratch
==================================

Autoencoder where both encoder and decoder are trained via drifting:
  - Encoder: data -> latent  (drift toward prior)
  - Decoder: latent -> data  (drift toward data)

The latent is detached so encoder and decoder train independently.
Both directions are 1-NFE (single forward pass, no iterative sampling).

Reference: Deng et al., "Generative Modeling via Drifting", ICML 2026
"""

from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange

# =============================================================================
# Device Configuration
# =============================================================================

def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# =============================================================================
# Data Generation
# =============================================================================

def gen_data(n: int, device: torch.device = DEVICE) -> Tensor:
    """Generate 2D mixture of 8 Gaussians arranged in a circle."""
    scale = 4.0
    centers = torch.tensor([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        [-1 / np.sqrt(2), 1 / np.sqrt(2)],
        [-1 / np.sqrt(2), -1 / np.sqrt(2)]
    ], dtype=torch.float32, device=device) * scale
    idx = torch.randint(len(centers), (n,), device=device)
    return centers[idx] + torch.randn(n, 2, device=device) * 0.5


def gen_checkerboard(n: int, device: torch.device = DEVICE) -> Tensor:
    """Generate 2D checkerboard distribution."""
    x1 = torch.rand(n, device=device) * 4 - 2
    x2 = torch.rand(n, device=device) * 4 - 2
    mask = ((x1.floor() + x2.floor()) % 2 == 0).float()
    x1 = x1 * mask + (x1 + 1 - 2 * (x1 > 0).float()) * (1 - mask)
    return torch.stack([x1, x2], dim=1)


# =============================================================================
# Network
# =============================================================================

class Net(nn.Module):
    """Simple MLP: in_dim -> out_dim."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 256, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =============================================================================
# Feature Normalization  (Sec A.6)
# =============================================================================
#
# Standardize to zero mean / unit variance per dimension, then scale so
# average pairwise distance ~ sqrt(D).  This decouples the kernel temperature
# from the raw feature magnitude.
#

def normalize_features(
    features: Tensor,
    mean: Tensor | None = None,
    std: Tensor | None = None,
    scale: float | None = None,
) -> Tuple[Tensor, Tensor, Tensor, float, Tensor | None]:
    """
    Normalize features: standardize per-dim, then scale so avg pairwise
    distance ~ sqrt(D).

    When mean/std/scale are provided (from a reference distribution),
    applies those instead of computing new ones.  This preserves
    cross-distribution offsets.

    Args:
        features: [N, D]
        mean:     Per-dim mean to use (skip computation if given).
        std:      Per-dim std to use (skip computation if given).
        scale:    Distance scale factor to use (skip computation if given).

    Returns:
        (normalized_features, mean, std, scale, self_dists)
        self_dists is the [N, N] pairwise distance matrix in normalized space
        (only computed when scale is None, otherwise None).
    """
    D = features.shape[1]
    self_dists = None
    with torch.no_grad():
        if mean is None:
            mean = features.mean(dim=0, keepdim=True)
        if std is None:
            std = features.std(dim=0, keepdim=True) + 1e-8
    features_std = (features - mean) / std

    if scale is None:
        with torch.no_grad():
            f_sq = (features_std * features_std).sum(dim=-1, keepdim=True)
            dist2 = f_sq + f_sq.t() - 2.0 * features_std @ features_std.t()
            dist2.clamp_(min=0.0)
            N = features.shape[0]
            dist2.fill_diagonal_(0.0)
            avg_dist_sq = dist2.sum() / (N * (N - 1))
            scale = (D / (avg_dist_sq + 1e-8)).sqrt()

    return features_std * scale, mean, std, scale, self_dists


# =============================================================================
# Drifting Field: Compute V  (Algorithm 2 from paper)
# =============================================================================
#
# Mean-shift drifting field (compact form):
#
#   V(x) = (1 / (Z_p Z_q)) E_{y+~p, y-~q}[ k(x,y+) k(x,y-) (y+ - y-) ]
#
# where k(x,y) = exp(-||x-y||^2 / tau)  (Gaussian kernel).
#
# Squared distances via ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x@y^T,
# which reduces to a GEMM and is much faster than cdist.
#

def normalize_drift(V: Tensor, target_variance: float = 1.0) -> Tensor:
    """Normalize drift field so E[V^2] ~ target_variance."""
    current_var = torch.mean(V ** 2)
    scale = (target_variance / (current_var + 1e-8)) ** 0.5
    return V * scale



def compute_drift(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    temp: float = 0.05,
) -> Tensor:
    """
    Compute the mean-shift drifting field V.

    Args:
        x:     Query points (generated samples) [N, D]
        y_pos: Positive samples (data) [N_pos, D]
        y_neg: Negative samples (generated) [N_neg, D]
        temp:  Kernel temperature

    Returns:
        V: Drift vectors [N, D]
    """
    N, N_pos = x.shape[0], y_pos.shape[0]

    
    # pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos)                   # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)                    # [N, N_neg]

    # mask self-interactions (y_neg = x in standard usage)
    if N == y_neg.shape[0]:
        dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

    # logits: -dist / temp
    logit = torch.cat([
        -dist_pos / temp,
        -dist_neg / temp,
    ], dim=1)                                           # [N, N_pos + N_neg]

    # doubly-normalized affinity: softmax over both dims, geometric mean
    A_row = logit.softmax(dim=-1)                       # normalize over y
    A_col = logit.softmax(dim=-2)                       # normalize over x
    A = (A_row * A_col).sqrt()

    # split into positive and negative affinities
    A_pos = A[:, :N_pos]                                # [N, N_pos]
    A_neg = A[:, N_pos:]                                # [N, N_neg]

    # factorized weights for compact form:
    # W_pos[i,j] = A_pos[i,j] * sum_k A_neg[i,k]
    # W_neg[i,k] = A_neg[i,k] * sum_j A_pos[i,j]
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    # V[i] = sum_j sum_k A_pos[i,j] A_neg[i,k] (y_pos[j] - y_neg[k])
    V = W_pos @ y_pos - W_neg @ y_neg

    return V
# =============================================================================
# Drifting Loss  (Algorithm 1 from paper)
# =============================================================================
#
# L = E[ || f(x) - stopgrad(f(x) + V(f(x))) ||^2 ]
#
# V pushes the current distribution toward the target. Gradients flow only
# through f(x), not through the frozen target f(x) + V.
#
def drifting_loss(
    x: Tensor,
    y_pos: Tensor,
) -> Tensor:
    """
    Compute per-sample drifting loss.  y_neg = x (current distribution).

    Features are normalized (Sec A.6) before kernel computation so that
    temperature is independent of raw feature magnitude.  Drift is mapped
    back to original space and variance-normalized.

    Args:
        x:     Current samples [N, D] (with gradient)
        y_pos: Target samples [N_pos, D]

    Returns:
        loss: Per-sample loss [N]
    """
    with torch.no_grad():
        # normalize using x's statistics for all distributions
        x_n, mean, std, scale, _ = normalize_features(x)
        yp_n, _, _, _, _ = normalize_features(y_pos, mean=mean, std=std, scale=scale)
        yn_n, _, _, _, _ = normalize_features(x, mean=mean, std=std, scale=scale)

        V = compute_drift(x_n, yp_n, yn_n)

        # map drift back to original space, normalize variance
        V_orig = normalize_drift(V / scale * std)
        target = (x + V_orig).detach()
    return ((x - target) ** 2).sum(dim=-1)


# =============================================================================
# Drifting Autoencoder
# =============================================================================

class DriftingAutoencoder(nn.Module):
    """
    Drifting Autoencoder.

    Encoder maps data -> latent, drifted toward prior N(0,I).
    Decoder maps latent -> data.
    """

    def __init__(self, data_dim: int = 2, hidden_dim: int = 256, latent_dim: int = 2):
        super().__init__()
        self.encoder = Net(data_dim, hidden_dim, latent_dim)
        self.decoder = Net(latent_dim, hidden_dim, data_dim)
        self.latent_dim = latent_dim
        self.data_dim = data_dim

    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            data: Data samples [N, D]

        Returns:
            (enc_loss, dec_loss, sample_loss): Per-sample losses, each [N]
        """
        encoded = self.encoder(data)
        prior = torch.randn(data.shape[0], self.latent_dim, device=data.device)
        enc_loss = drifting_loss(encoded, prior)

        # L1 reconstruction from encoded inputs
        decoded = self.decoder(encoded)
        dec_loss = F.l1_loss(decoded, data, reduction='none').sum(dim=-1)


        return enc_loss, dec_loss

    @torch.no_grad()
    def encode(self, data: Tensor) -> Tensor:
        """Encode data to latent space (1-NFE)."""
        return self.encoder(data)

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to data space (1-NFE)."""
        return self.decoder(z)


# =============================================================================
# Visualization
# =============================================================================

def viz_2d_data(data: Tensor, filename: str, title: str | None = None):
    """Save 2D scatter plot."""
    plt.figure()
    data = data.cpu()
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    if title:
        plt.title(title)
    plt.axis("scaled")
    plt.savefig(filename, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


def viz_overview(
    data: Tensor,
    encoded: Tensor,
    decoded: Tensor,
    recon: Tensor,
    step: int,
    filename: str = "overview.jpg",
):
    """4-panel overview: data, encoded, generated, reconstruction."""
    d = data.detach().cpu().numpy()
    e = encoded.detach().cpu().numpy()
    g = decoded.detach().cpu().numpy()
    r = recon.detach().cpu().numpy()

    _, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    axes[0].scatter(d[:, 0], d[:, 1], s=2, alpha=0.3, c="black")
    axes[0].set_title("Data")

    axes[1].scatter(e[:, 0], e[:, 1], s=2, alpha=0.3, c="tab:orange")
    axes[1].set_title(f"Encoded (step {step})")

    axes[2].scatter(g[:, 0], g[:, 1], s=2, alpha=0.3, c="tab:green")
    axes[2].set_title(f"Generated (step {step})")

    axes[3].scatter(r[:, 0], r[:, 1], s=2, alpha=0.3, c="tab:purple")
    axes[3].set_title(f"Reconstructed (step {step})")

    plt.tight_layout()
    plt.savefig(filename, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Training
# =============================================================================

def train(
    model: DriftingAutoencoder,
    data_fn: Callable[[int], Tensor],
    tag: str = "train",
    n_iter: int = 5000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    sample_every: int = 100,
    n_samples: int = 4096,
):
    """Train a drifting autoencoder."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = trange(n_iter)

    for i in pbar:
        data = data_fn(batch_size)

        enc_loss, dec_loss = model(data)
        loss = enc_loss.mean() + dec_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            pbar.set_description(
                f"enc: {enc_loss.mean().item():.4f}  "
                f"dec: {dec_loss.mean().item():.4f}  "
            )

        if (i + 1) % sample_every == 0:
            model.eval()

            data_vis = data_fn(n_samples)
            encoded_vis = model.encode(data_vis)
            z_vis = torch.randn(n_samples, model.latent_dim, device=data_vis.device)
            decoded_vis = model.decode(z_vis)
            recon_vis = model.decode(encoded_vis)
            viz_overview(data_vis, encoded_vis, decoded_vis, recon_vis,
                         i + 1, f"overview_{tag}.jpg")

            model.train()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    viz_2d_data(gen_data(8096), filename="data_8gaussians.jpg", title="8 Gaussians")
    viz_2d_data(gen_checkerboard(8096), filename="data_checkerboard.jpg", title="Checkerboard")

    print("\n--- Training on 8 Gaussians ---")
    model = DriftingAutoencoder(data_dim=2, hidden_dim=256, latent_dim=2).to(DEVICE)
    train(model, gen_data, tag="8gauss", n_iter=100_000, batch_size=2048)

    model.eval()
    data_final = gen_data(4096)
    viz_2d_data(model.encode(data_final), filename="final_encoded_8gauss.jpg", title="Encoded")
    viz_2d_data(model.decode(torch.randn(4096, 2, device=DEVICE)),
                filename="final_generated_8gauss.jpg", title="Generated")

    print("\n--- Training on Checkerboard ---")
    model2 = DriftingAutoencoder(data_dim=2, hidden_dim=256, latent_dim=2).to(DEVICE)
    train(model2, gen_checkerboard, tag="checker", n_iter=100_000, batch_size=2048)

    model2.eval()
    data_final2 = gen_checkerboard(4096)
    viz_2d_data(model2.encode(data_final2), filename="final_encoded_checker.jpg", title="Encoded")
    viz_2d_data(model2.decode(torch.randn(4096, 2, device=DEVICE)),
                filename="final_generated_checker.jpg", title="Generated")

    print("Done.")

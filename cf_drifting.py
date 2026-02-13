"""
Generative Modeling via Characteristic Function Drifting
=========================================================

A generator G: z -> x is trained by iteratively drifting its output
distribution toward the data distribution. The drift field is the
functional gradient of the characteristic function (CF) distance.

Given current samples x_1..x_N = G(z_1)..G(z_N) and data y_1..y_M,
the CF distance is estimated by sampling F random frequency vectors
f ~ N(0, sigma^2 I) in R^D:

  D(mu, nu) = (1/F) sum_f |phi_mu(f) - phi_nu(f)|^2

where phi_mu(f) = (1/N) sum_j exp(i f'x_j) is the empirical CF
evaluated at frequency f. Decomposing into real/imaginary parts:

  phi_mu(f) = C_mu(f) + i S_mu(f)

  C_mu(f) = (1/N) sum_j cos(f'x_j),  S_mu(f) = (1/N) sum_j sin(f'x_j)

Differentiating w.r.t. sample x_k gives the per-sample drift:

  V(x_k) = -(2/NF) sum_f [-dC sin(f'x_k) + dS cos(f'x_k)] f

where dC = C_mu(f) - C_nu(f), dS = S_mu(f) - S_nu(f). Negated because
the gradient points uphill; the drift points toward the target.

The training loss uses the drifting framework:

  L_k = ||x_k - sg(x_k + V(x_k))||^2

where sg is stop-gradient. Gradients flow only through x_k = G(z_k),
not through the frozen target x_k + V.

Complexity is O(N * F) where F = number of frequency vectors.
"""

from typing import Callable

import numpy as np
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from tqdm import trange

# =============================================================================
# Device Configuration
# =============================================================================

def get_device() -> torch.device:
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
    x1 = torch.rand(n, device=device) * 4 - 2
    x2 = torch.rand(n, device=device) * 4 - 2
    mask = ((x1.floor() + x2.floor()) % 2 == 0).float()
    x1 = x1 * mask + (x1 + 1 - 2 * (x1 > 0).float()) * (1 - mask)
    return torch.stack([x1, x2], dim=1)


# =============================================================================
# Network
# =============================================================================

class Net(nn.Module):
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

def normalize_drift(V: Tensor, target_variance: float = 1.0) -> Tensor:
    """Rescale drift to target RMS. Direction from CF gradient, magnitude = step size."""
    current_var = torch.mean(V ** 2)
    scale = (target_variance / (current_var + 1e-8)) ** 0.5
    return V * scale


def cf_drifting_loss_empirical(
    x: Tensor,
    y: Tensor,
    num_freq: int = 1024,
    freq_std: float = 2.0,
) -> Tensor:
    """
    Drifting loss using full multivariate CF drift toward empirical y.

    Instead of slicing to 1D and integrating over scalar t, sample F
    random frequency vectors f ~ N(0, freq_std^2 I) in R^D and compare
    the D-dimensional characteristic functions directly:

      D = (1/F) sum_f |phi_mu(f) - phi_nu(f)|^2

    Per-sample drift:

      V(x_k) = -(2/NF) sum_f [-err_C(f) sin(f'x_k) + err_S(f) cos(f'x_k)] f


    Args:
        x: [N, D] current samples (with gradient)
        y: [M, D] target samples
        num_freq: number of random frequency vectors
        freq_std: std of frequency sampling distribution

    Returns:
        loss: [N] per-sample loss
    """
    N, D = x.shape
    device = x.device

    # random frequency vectors in R^D
    F = torch.randn(num_freq, D, device=device, dtype=x.dtype) * freq_std  # [F, D]

    with torch.no_grad():
        # inner products: f'x for all samples and frequencies
        inner_x = x @ F.T                                          # [N, F]
        inner_y = y @ F.T                                          # [M, F]

        # empirical CFs at each frequency
        C_x = torch.cos(inner_x).mean(dim=0)                       # [F]
        S_x = torch.sin(inner_x).mean(dim=0)
        C_y = torch.cos(inner_y).mean(dim=0)                       # [F]
        S_y = torch.sin(inner_y).mean(dim=0)

        err_C = C_x - C_y                                          # [F]
        err_S = S_x - S_y                                          # [F]

        # per-sample gradient coefficient
        cos_k = torch.cos(inner_x)                                  # [N, F]
        sin_k = torch.sin(inner_x)                                  # [N, F]
        coeff = -err_C * sin_k + err_S * cos_k                     # [N, F]

        # drift: sum over frequencies weighted by frequency vector
        V = -(2.0 / (N * num_freq)) * (coeff @ F)                  # [N, D]
        V = normalize_drift(V)
        target = (x + V).detach()

    return ((x - target) ** 2).sum(dim=-1)

class CFDriftGenerator(nn.Module):
    """z ~ N(0,I) -> Net -> x, trained by CF drift toward data."""

    def __init__(self, latent_dim: int = 2, hidden_dim: int = 256, data_dim: int = 2,
                 num_freq: int = 1024, freq_std: float = 2.0):
        super().__init__()
        self.net = Net(latent_dim, hidden_dim, data_dim)
        self.latent_dim = latent_dim
        self.num_freq = num_freq
        self.freq_std = freq_std

    def forward(self, data: Tensor) -> Tensor:
        z = torch.randn(data.shape[0], self.latent_dim, device=data.device)
        generated = self.net(z)
        loss = cf_drifting_loss_empirical(
            generated, data,
            num_freq=self.num_freq, freq_std=self.freq_std,
        )
        return loss

    @torch.no_grad()
    def sample(self, n: int, device: torch.device = DEVICE) -> Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.net(z)


def viz_2d(data: Tensor, generated: Tensor, step: int, filename: str):
    d = data.detach().cpu().numpy()
    g = generated.detach().cpu().numpy()

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    axes[0].scatter(d[:, 0], d[:, 1], s=2, alpha=0.3, c="black")
    axes[0].set_title("Data")

    axes[1].scatter(g[:, 0], g[:, 1], s=2, alpha=0.3, c="tab:green")
    axes[1].set_title(f"Generated (step {step})")

    plt.tight_layout()
    plt.savefig(filename, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Training
# =============================================================================

def train(
    model: CFDriftGenerator,
    data_fn: Callable[[int], Tensor],
    tag: str = "train",
    n_iter: int = 5000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    sample_every: int = 100,
    n_samples: int = 4096,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = trange(n_iter)

    for i in pbar:
        data = data_fn(batch_size)

        loss = model(data)
        total = loss.mean()

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            pbar.set_description(f"loss: {total.item()}")

        if (i + 1) % sample_every == 0:
            model.eval()
            data_vis = data_fn(n_samples)
            gen_vis = model.sample(n_samples, device=data_vis.device)
            viz_2d(data_vis, gen_vis, i + 1, f"gen_{tag}.jpg")
            model.train()


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = CFDriftGenerator(latent_dim=2, hidden_dim=256, data_dim=2,
                             num_freq=2048, freq_std=3.0).to(DEVICE)
    train(model, gen_data, tag="8gauss", n_iter=10_000, batch_size=4096)

    model2 = CFDriftGenerator(latent_dim=2, hidden_dim=256, data_dim=2,
                              num_freq=2048, freq_std=3.0).to(DEVICE)
    train(model2, gen_checkerboard, tag="checker", n_iter=10_000, batch_size=4096)

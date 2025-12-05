import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

@dataclass
class Config:
    base_dir: str = "data/synthetic_dataset_splits_0.02"

    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5

    latent_dim: int = 16
    hidden_dims_enc: Tuple[int, ...] = (256, 256)
    hidden_dims_dec: Tuple[int, ...] = (256, 256)

    dropout: float = 0.1
    beta_kl: float = 1e-3        # weight for KL term
    lambda_abund: float = 10.0   # weight for abundance regression

    patience: int = 20           # early stopping
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir: str = "results_vae"
    model_path: str = "results_vae/best_vae.pt"


# =========================
# Utils
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# Data
# =========================

class SpectralDataset(Dataset):
    def __init__(self, X: np.ndarray, A: np.ndarray):
        assert X.shape[0] == A.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.A = torch.from_numpy(A.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx]


def load_split(base_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        spectra: [N, B]
        abundances: [N, M]
        wavelengths: [B]
    """
    spectra_path = os.path.join(base_dir, f"{split}_spectra.csv")
    abundances_path = os.path.join(base_dir, f"{split}_abundances.csv")

    spec_df = pd.read_csv(spectra_path)
    abund_df = pd.read_csv(abundances_path)

    if "sample_id" in spec_df.columns:
        spec_df = spec_df.drop(columns=["sample_id"])
    if "sample_id" in abund_df.columns:
        abund_df = abund_df.drop(columns=["sample_id"])

    spectra = spec_df.values
    abundances = abund_df.values
    wavelengths = spec_df.columns.astype(float).values

    return spectra, abundances, wavelengths


def standardize_train_val_test(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma == 0.0] = 1.0

    X_train_z = (X_train - mu) / sigma
    X_val_z = (X_val - mu) / sigma
    X_test_z = (X_test - mu) / sigma

    stats = {"mu": mu.squeeze(), "sigma": sigma.squeeze()}
    return X_train_z, X_val_z, X_test_z, stats


# =========================
# VAE with abundance head
# =========================

class VAEWithAbundances(nn.Module):
    """
    Encoder: x -> h -> (mu, logvar, a_hat)
    Latent sample: z ~ N(mu, diag(exp(logvar)))
    Decoder: [z, a_hat] -> x_hat

    a_hat is constrained to the simplex via softmax.
    """

    def __init__(
        self,
        input_dim: int,
        n_materials: int,
        latent_dim: int = 16,
        hidden_dims_enc=(256, 256),
        hidden_dims_dec=(256, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_materials = n_materials
        self.latent_dim = latent_dim

        # Encoder backbone
        enc_layers = []
        prev = input_dim
        for h in hidden_dims_enc:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.BatchNorm1d(h))
            enc_layers.append(nn.ReLU())
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder_backbone = nn.Sequential(*enc_layers)

        # Heads from encoder representation
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        self.fc_abund = nn.Linear(prev, n_materials)
        self.softmax = nn.Softmax(dim=1)

        # Decoder
        dec_input_dim = latent_dim + n_materials
        dec_layers = []
        prev = dec_input_dim
        for h in hidden_dims_dec:
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.BatchNorm1d(h))
            dec_layers.append(nn.ReLU())
            if dropout > 0.0:
                dec_layers.append(nn.Dropout(dropout))
            prev = h
        self.decoder_backbone = nn.Sequential(*dec_layers)
        self.fc_out = nn.Linear(prev, input_dim)
        self.out_activation = nn.Sigmoid()  # reflectance in [0,1] after standardization? optional

    def encode(self, x):
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        abund_logits = self.fc_abund(h)
        a_hat = self.softmax(abund_logits)  # [0,1], sums to 1
        return mu, logvar, a_hat

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, a_hat):
        inp = torch.cat([z, a_hat], dim=1)
        h = self.decoder_backbone(inp)
        x_hat = self.fc_out(h)
        # if spectra are standardized, you might omit sigmoid; if in [0,1], keep it
        return x_hat

    def forward(self, x):
        mu, logvar, a_hat = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, a_hat)
        return x_hat, mu, logvar, a_hat


# =========================
# Loss
# =========================

def vae_loss(
    x,
    x_hat,
    mu,
    logvar,
    a_true,
    a_hat,
    beta_kl: float,
    lambda_abund: float,
):
    # Reconstruction: MSE on standardized spectra
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")

    # KL divergence (per batch mean)
    # 0.5 * sum(μ^2 + σ^2 - log σ^2 - 1)
    kld = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    # Abundance regression loss
    abund_loss = nn.functional.mse_loss(a_hat, a_true, reduction="mean")

    total = recon_loss + beta_kl * kld + lambda_abund * abund_loss
    return total, recon_loss, kld, abund_loss


# =========================
# Training / evaluation
# =========================

def train_one_epoch(
    model,
    loader,
    optimizer,
    cfg: Config,
):
    model.train()
    totals = []
    recons = []
    klds = []
    abunds = []

    for X, A in loader:
        X = X.to(cfg.device)
        A = A.to(cfg.device)

        optimizer.zero_grad()
        x_hat, mu, logvar, a_hat = model(X)
        loss, recon_loss, kld, abund_loss = vae_loss(
            X, x_hat, mu, logvar, A, a_hat,
            beta_kl=cfg.beta_kl,
            lambda_abund=cfg.lambda_abund,
        )
        loss.backward()
        optimizer.step()

        totals.append(loss.item())
        recons.append(recon_loss.item())
        klds.append(kld.item())
        abunds.append(abund_loss.item())

    return (
        np.mean(totals),
        np.mean(recons),
        np.mean(klds),
        np.mean(abunds),
    )


@torch.no_grad()
def eval_one_epoch(model, loader, cfg: Config):
    model.eval()
    totals = []
    recons = []
    klds = []
    abunds = []

    for X, A in loader:
        X = X.to(cfg.device)
        A = A.to(cfg.device)

        x_hat, mu, logvar, a_hat = model(X)
        loss, recon_loss, kld, abund_loss = vae_loss(
            X, x_hat, mu, logvar, A, a_hat,
            beta_kl=cfg.beta_kl,
            lambda_abund=cfg.lambda_abund,
        )

        totals.append(loss.item())
        recons.append(recon_loss.item())
        klds.append(kld.item())
        abunds.append(abund_loss.item())

    return (
        np.mean(totals),
        np.mean(recons),
        np.mean(klds),
        np.mean(abunds),
    )


@torch.no_grad()
def evaluate_test(model, loader, cfg: Config, material_names):
    model.eval()
    y_true_all = []
    y_pred_all = []
    x_true_all = []
    x_recon_all = []

    for X, A in loader:
        X = X.to(cfg.device)
        A = A.to(cfg.device)

        x_hat, mu, logvar, a_hat = model(X)

        y_true_all.append(A.cpu().numpy())
        y_pred_all.append(a_hat.cpu().numpy())
        x_true_all.append(X.cpu().numpy())
        x_recon_all.append(x_hat.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    x_true = np.concatenate(x_true_all, axis=0)
    x_recon = np.concatenate(x_recon_all, axis=0)

    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    print("\nTest metrics per material (abundance head):")
    for i, name in enumerate(material_names):
        print(f"{name:15s}  MAE={mae[i]:.4f}  RMSE={rmse[i]:.4f}")
    print(f"\nOverall MAE:  {mae.mean():.4f}")
    print(f"Overall RMSE: {rmse.mean():.4f}")

    return x_true, x_recon, y_true, y_pred, mae, rmse


# =========================
# Plotting
# =========================

def plot_loss_curves(history, cfg: Config):
    ensure_dir(cfg.results_dir)
    epochs = np.arange(1, len(history["train_total"]) + 1)

    # Total loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_total"], label="Train total")
    plt.plot(epochs, history["val_total"], label="Val total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "loss_total.png"), dpi=300)
    plt.close()

    # Recon loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_recon"], label="Train recon")
    plt.plot(epochs, history["val_recon"], label="Val recon")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Reconstruction loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "loss_recon.png"), dpi=300)
    plt.close()

    # Abundance loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_abund"], label="Train abund")
    plt.plot(epochs, history["val_abund"], label="Val abund")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Abundance regression loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "loss_abund.png"), dpi=300)
    plt.close()

    # KL term
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_kl"], label="Train KL")
    plt.plot(epochs, history["val_kl"], label="Val KL")
    plt.xlabel("Epoch")
    plt.ylabel("KL")
    plt.title("KL divergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "loss_kl.png"), dpi=300)
    plt.close()


def plot_true_vs_pred(y_true, y_pred, material_names, cfg: Config):
    ensure_dir(cfg.results_dir)
    n_mat = y_true.shape[1]
    cols = 3
    rows = int(np.ceil(n_mat / cols))
    x = np.linspace(0, 1, 100)

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i in range(n_mat):
        ax = plt.subplot(rows, cols, i + 1)
        ax.scatter(y_true[:, i], y_pred[:, i], s=5, alpha=0.5)
        ax.plot(x, x, "r--", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(material_names[i])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "true_vs_pred_abund.png"), dpi=300)
    plt.close()


def plot_error_hist(y_true, y_pred, material_names, cfg: Config):
    ensure_dir(cfg.results_dir)
    n_mat = y_true.shape[1]
    cols = 3
    rows = int(np.ceil(n_mat / cols))

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i in range(n_mat):
        ax = plt.subplot(rows, cols, i + 1)
        err = y_pred[:, i] - y_true[:, i]
        ax.hist(err, bins=40, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.set_title(material_names[i])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "error_hist_abund.png"), dpi=300)
    plt.close()


def plot_per_material_errors(mae, rmse, material_names, cfg: Config):
    ensure_dir(cfg.results_dir)
    x = np.arange(len(material_names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, mae, width, label="MAE")
    plt.bar(x + width / 2, rmse, width, label="RMSE")
    plt.xticks(x, material_names, rotation=45, ha="right")
    plt.ylabel("Error")
    plt.title("Abundance errors (test)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "per_material_errors_abund.png"), dpi=300)
    plt.close()


def plot_recon_examples(x_true, x_recon, wavelengths, cfg: Config, n_examples: int = 2):
    ensure_dir(cfg.results_dir)
    n_examples = min(n_examples, x_true.shape[0])

    plt.figure(figsize=(12, 8))
    for i in range(n_examples):
        plt.plot(wavelengths, x_true[i], alpha=0.5, label="True" if i == 0 else None)
        plt.plot(wavelengths, x_recon[i], alpha=0.5, linestyle="--",
                 label="Recon" if i == 0 else None)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Standardized reflectance")
    plt.title("Original vs reconstructed spectra (standardized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "spectra_recon_examples.png"), dpi=300)
    plt.close()


# =========================
# Main
# =========================

def main():
    cfg = Config()
    set_seed(cfg.seed)
    ensure_dir(cfg.results_dir)

    material_names = [
        "solarcell",
        "blackmli",
        "kaptonhn",
        "dunmore",
        "thermalbrightn",
        "kaptonws",
    ]

    # ----- Load data -----
    X_train, A_train, wavelengths = load_split(cfg.base_dir, "train")
    X_val, A_val, _ = load_split(cfg.base_dir, "val")
    X_test, A_test, _ = load_split(cfg.base_dir, "test")

    assert A_train.shape[1] == len(material_names)
    assert A_val.shape[1] == len(material_names)
    assert A_test.shape[1] == len(material_names)

    print(f"Train abundance sum mean: {A_train.sum(axis=1).mean():.4f}")
    print(f"Val abundance sum mean:   {A_val.sum(axis=1).mean():.4f}")
    print(f"Test abundance sum mean:  {A_test.sum(axis=1).mean():.4f}")

    # ----- Standardize spectra -----
    X_train_z, X_val_z, X_test_z, stats = standardize_train_val_test(X_train, X_val, X_test)
    np.savez(
        os.path.join(cfg.results_dir, "standardization_stats.npz"),
        mu=stats["mu"],
        sigma=stats["sigma"],
        wavelengths=wavelengths,
    )
    X_train_z, X_val_z, X_test_z = X_train, X_val, X_test

    train_ds = SpectralDataset(X_train_z, A_train)
    val_ds = SpectralDataset(X_val_z, A_val)
    test_ds = SpectralDataset(X_test_z, A_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    input_dim = X_train_z.shape[1]
    n_materials = A_train.shape[1]

    model = VAEWithAbundances(
        input_dim=input_dim,
        n_materials=n_materials,
        latent_dim=cfg.latent_dim,
        hidden_dims_enc=cfg.hidden_dims_enc,
        hidden_dims_dec=cfg.hidden_dims_dec,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history = {
        "train_total": [],
        "train_recon": [],
        "train_kl": [],
        "train_abund": [],
        "val_total": [],
        "val_recon": [],
        "val_kl": [],
        "val_abund": [],
    }

    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0

    # ----- Training loop -----
    for epoch in range(1, cfg.num_epochs + 1):
        tr_tot, tr_rec, tr_kl, tr_ab = train_one_epoch(model, train_loader, optimizer, cfg)
        va_tot, va_rec, va_kl, va_ab = eval_one_epoch(model, val_loader, cfg)

        history["train_total"].append(tr_tot)
        history["train_recon"].append(tr_rec)
        history["train_kl"].append(tr_kl)
        history["train_abund"].append(tr_ab)

        history["val_total"].append(va_tot)
        history["val_recon"].append(va_rec)
        history["val_kl"].append(va_kl)
        history["val_abund"].append(va_ab)

        print(
            f"Epoch {epoch:03d} | "
            f"train: total={tr_tot:.4f}, recon={tr_rec:.4f}, kl={tr_kl:.4f}, abund={tr_ab:.4f} | "
            f"val: total={va_tot:.4f}, recon={va_rec:.4f}, kl={va_kl:.4f}, abund={va_ab:.4f}"
        )

        if va_tot < best_val - 1e-6:
            best_val = va_tot
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), cfg.model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    plot_loss_curves(history, cfg)

    # ----- Test evaluation -----
    print(f"\nLoading best model from epoch {best_epoch} (val_total={best_val:.4f})")
    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))

    x_true, x_recon, y_true, y_pred, mae, rmse = evaluate_test(model, test_loader, cfg, material_names)

    # ----- Visualizations -----
    plot_true_vs_pred(y_true, y_pred, material_names, cfg)
    plot_error_hist(y_true, y_pred, material_names, cfg)
    plot_per_material_errors(mae, rmse, material_names, cfg)
    plot_recon_examples(x_true, x_recon, wavelengths, cfg, n_examples=2)

    print(f"\nResults saved in: {cfg.results_dir}")
    print(f"Best model path:   {cfg.model_path}")


if __name__ == "__main__":
    main()

import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =========================
# Config & utilities
# =========================

@dataclass
class Config:
    base_dir: str = "data/synthetic_dataset_splits"
    endmember_path: str = "data/endmember_library_clipped.csv"

    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.1
    patience: int = 20

    # loss weights
    lambda_abund: float = 1.0      # abundance MSE
    lambda_recon: float = 1.0      # spectral reconstruction MSE
    lambda_cls: float = 1.0        # MLI/WHITE classification CE

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir: str = "results_group_exclusive_gumbel"
    model_path: str = "results_group_exclusive/best_model.pt"
    seed: int = 42
    
    # Gumbel temperature schedule
    tau_start: float = 1.0
    tau_end: float = 0.1
    tau_anneal_epochs: int = 100



def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# Data loading / standardization
# =========================

class SpectralDataset(Dataset):
    def __init__(self, spectra: np.ndarray, abundances: np.ndarray):
        assert spectra.shape[0] == abundances.shape[0]
        self.X = torch.from_numpy(spectra.astype(np.float32))
        self.y = torch.from_numpy(abundances.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_split(base_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra_path = os.path.join(base_dir, f"{split}_spectra.csv")
    abundances_path = os.path.join(base_dir, f"{split}_abundances.csv")

    spectra_df = pd.read_csv(spectra_path)
    abundances_df = pd.read_csv(abundances_path)

    if "sample_id" in spectra_df.columns:
        spectra_df = spectra_df.drop(columns=["sample_id"])
    if "sample_id" in abundances_df.columns:
        abundances_df = abundances_df.drop(columns=["sample_id"])

    spectra = spectra_df.values
    abundances = abundances_df.values
    wavelengths = spectra_df.columns.astype(float).values

    return spectra, abundances, wavelengths


def standardize_train_val_test(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma == 0.0] = 1.0

    X_train_z = (X_train - mu) / sigma
    X_val_z = (X_val - mu) / sigma
    X_test_z = (X_test - mu) / sigma

    stats = {"mu": mu.squeeze(), "sigma": sigma.squeeze()}
    return X_train_z, X_val_z, X_test_z, stats


def load_endmember_matrix(path: str, material_names):
    df = pd.read_csv(path)
    E = df[material_names].values.T  # [M, B]
    wavelengths = df["wavelength_nm"].values
    return E, wavelengths

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes per-material R² for abundance regression.
    y_true, y_pred: [N, M]
    Returns: [M] array of R² scores.
    """
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return r2

def compute_sam(X_true: np.ndarray, X_recon: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes SAM per sample and mean SAM.
    Returns:
        sam_per_sample: [N]
        sam_mean: float
    """
    dot = np.sum(X_true * X_recon, axis=1)
    norm_true = np.linalg.norm(X_true, axis=1)
    norm_recon = np.linalg.norm(X_recon, axis=1)
    
    cos_angle = dot / (norm_true * norm_recon + 1e-12)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    sam = np.arccos(cos_angle)  # radians
    return sam, sam.mean()



# =========================
# Group-exclusive model
# =========================

class GumbelGroupedUnmixer(nn.Module):
    """
    Group-exclusive model with discrete identities selected by Gumbel-Softmax.

    - group_head:   continuous fractions for (solar, MLI, WHITE) via softmax
    - mli_head:     logits for 3-way categorical over (blackmli, kaptonhn, dunmore)
    - white_head:   logits for 2-way categorical over (thermalbrightn, kaptonws)

    For each sample:
        g = softmax(group_logits)            ∈ Δ^3
        m_onehot = gumbel_softmax(...)       ∈ {one-hot in R^3}
        w_onehot = gumbel_softmax(...)       ∈ {one-hot in R^2}

        a_solar = g_solar
        a_mli   = g_mli * m_onehot
        a_white = g_white * w_onehot

    So exactly one MLI and one WHITE material are non-zero per pixel.
    """

    def __init__(self, input_dim: int, hidden_dims=(256, 256, 128),
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h),
                       nn.BatchNorm1d(h),
                       nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)

        self.group_head = nn.Linear(prev, 3)   # solar, MLI, WHITE
        self.mli_head   = nn.Linear(prev, 3)   # blackmli, kaptonhn, dunmore
        self.white_head = nn.Linear(prev, 2)   # thermalbrightn, kaptonws

    @staticmethod
    def _gumbel_one_hot(logits: torch.Tensor, tau: float,
                        training: bool) -> torch.Tensor:
        """
        During training: Gumbel-Softmax (hard = one-hot, straight-through).
        During eval: deterministic one-hot from argmax.
        """
        if training:
            return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            idx = logits.argmax(dim=-1)
            return F.one_hot(idx, num_classes=logits.size(-1)).float()

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        h = self.backbone(x)

        group_logits = self.group_head(h)          # [N,3]
        mli_logits   = self.mli_head(h)            # [N,3]
        white_logits = self.white_head(h)          # [N,2]

        # continuous group fractions
        g = F.softmax(group_logits, dim=-1)        # [N,3]
        g_solar = g[:, 0:1]                        # [N,1]
        g_mli   = g[:, 1:2]
        g_white = g[:, 2:3]

        # discrete identities within each group
        m_onehot = self._gumbel_one_hot(mli_logits,   tau, self.training)  # [N,3]
        w_onehot = self._gumbel_one_hot(white_logits, tau, self.training)  # [N,2]

        a_solar = g_solar                           # [N,1]
        a_mli   = g_mli * m_onehot                  # [N,3]
        a_white = g_white * w_onehot                # [N,2]

        abundances = torch.cat([a_solar, a_mli, a_white], dim=1)  # [N,6]

        # we still return logits for CE losses
        return abundances, (group_logits, mli_logits, white_logits)

def gumbel_tau_schedule(cfg: Config, epoch: int):
    """
    Linear annealing of Gumbel temperature.
    """
    if epoch >= cfg.tau_anneal_epochs:
        return cfg.tau_end
    frac = epoch / cfg.tau_anneal_epochs
    return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac

# =========================
# Losses & training
# =========================

def build_group_labels(y_true: torch.Tensor):
    """
    y_true: [N,6] in fixed order
      0: solarcell
      1,2,3: blackmli, kaptonhn, dunmore
      4,5: thermalbrightn, kaptonws
    Returns:
      mli_idx:   [N] in {0,1,2}
      white_idx: [N] in {0,1}
    """
    mli_block = y_true[:, 1:4]
    white_block = y_true[:, 4:6]
    mli_idx = mli_block.argmax(dim=1)
    white_idx = white_block.argmax(dim=1)
    return mli_idx, white_idx


def train_one_epoch_gumbel_group(model, loader, optimizer, device,
                                 E_z_torch: torch.Tensor, cfg, tau: float):
    model.train()
    total_loss = total_abund = total_recon = total_cls = 0.0
    n_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # NOTE: tau from config
        y_pred, (group_logits, mli_logits, white_logits) = model(X, tau=tau)

        # abundance supervision
        loss_abund = F.mse_loss(y_pred, y)

        # reconstruction supervision
        X_hat = y_pred @ E_z_torch
        loss_recon = F.mse_loss(X_hat, X)

        # identity classification within MLI and WHITE groups
        mli_idx, white_idx = build_group_labels(y)
        loss_mli   = F.cross_entropy(mli_logits,   mli_idx)
        loss_white = F.cross_entropy(white_logits, white_idx)
        loss_cls = loss_mli + loss_white

        loss = (cfg.lambda_abund * loss_abund +
                cfg.lambda_recon * loss_recon +
                cfg.lambda_cls   * loss_cls )

        loss.backward()
        optimizer.step()

        bs = X.size(0)
        total_loss  += loss.item() * bs
        total_abund += loss_abund.item() * bs
        total_recon += loss_recon.item() * bs
        total_cls   += loss_cls.item() * bs
        n_samples   += bs

    return (total_loss / n_samples,
            total_abund / n_samples,
            total_recon / n_samples,
            total_cls / n_samples)



@torch.no_grad()
def eval_one_epoch_gumbel_group(model, loader, device,
                                E_z_torch: torch.Tensor, cfg, tau):
    model.eval()
    total_loss = total_abund = total_recon = total_cls = 0.0
    n_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        y_pred, (group_logits, mli_logits, white_logits)= model(X, tau=tau)

        loss_abund = F.mse_loss(y_pred, y)
        X_hat = y_pred @ E_z_torch
        loss_recon = F.mse_loss(X_hat, X)

        mli_idx, white_idx = build_group_labels(y)
        loss_mli   = F.cross_entropy(mli_logits, mli_idx)
        loss_white = F.cross_entropy(white_logits, white_idx)
        loss_cls = loss_mli + loss_white

        loss = (cfg.lambda_abund * loss_abund +
                cfg.lambda_recon * loss_recon +
                cfg.lambda_cls   * loss_cls )

        bs = X.size(0)
        total_loss  += loss.item() * bs
        total_abund += loss_abund.item() * bs
        total_recon += loss_recon.item() * bs
        total_cls   += loss_cls.item() * bs
        n_samples   += bs

    return (total_loss / n_samples,
            total_abund / n_samples,
            total_recon / n_samples,
            total_cls / n_samples)



# =========================
# Evaluation / plotting
# =========================

@torch.no_grad()
def evaluate_test(model, loader, device, material_names):
    model.eval()
    y_true_all, y_pred_all = [], []

    for X, y in loader:
        X = X.to(device)
        y_true_all.append(y.numpy())
        y_pred, _ = model(X)
        y_pred_all.append(y_pred.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    r2 = compute_r2(y_true, y_pred)

    print("\nTest metrics per material:")
    for i, name in enumerate(material_names):
        print(f"{name:15s}  MAE={mae[i]:.4f}  RMSE={rmse[i]:.4f}  R2={r2[i]:.4f}")

    print(f"\nOverall MAE:  {mae.mean():.4f}")
    print(f"Overall RMSE: {rmse.mean():.4f}")
    print(f"Overall R2:   {r2.mean():.4f}")

    return y_true, y_pred, mae, rmse, r2


def plot_loss_curves(train_losses, val_losses, cfg: Config):
    ensure_dir(cfg.results_dir)
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Composite loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "loss_curves.png"), dpi=300)
    plt.close()


def plot_true_vs_pred(y_true, y_pred, material_names, cfg: Config):
    ensure_dir(cfg.results_dir)
    n_mat = y_true.shape[1]
    cols = 3
    rows = int(np.ceil(n_mat / cols))
    xline = np.linspace(0, 1, 100)

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i in range(n_mat):
        ax = plt.subplot(rows, cols, i + 1)
        ax.scatter(y_true[:, i], y_pred[:, i], s=5, alpha=0.5)
        ax.plot(xline, xline, "r--", linewidth=1)
        ax.set_title(material_names[i])
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "true_vs_pred.png"), dpi=300)
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
        ax.hist(err, bins=40, edgecolor="black", alpha=0.7)
        ax.set_title(material_names[i])
        ax.set_xlabel("Prediction error")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "error_histograms.png"), dpi=300)
    plt.close()


def reconstruct_and_plot(X_test, y_pred, E, wavelengths, cfg: Config):
    ensure_dir(cfg.results_dir)
    X_recon = y_pred @ E  # unstandardized domain

    # SAM between raw spectra
    sam, sam_mean = compute_sam(X_test, X_recon)
    print(f"Mean SAM (radians): {sam_mean:.6f}")
    print(f"Mean SAM (degrees): {np.degrees(sam_mean):.6f}")
    # Spectral R2 (per wavelength)
    ss_res = np.sum((X_test - X_recon) ** 2, axis=0)
    ss_tot = np.sum((X_test - X_test.mean(axis=0)) ** 2, axis=0)
    r2_spec = 1 - ss_res / (ss_tot + 1e-12)

    print(f"Mean spectral R2: {r2_spec.mean():.6f}")


    diff = X_recon - X_test
    rmse = np.sqrt(np.mean(diff ** 2, axis=1))
    per_band_rmse = np.sqrt(np.mean(diff ** 2, axis=0))

    print(f"\nSpectral reconstruction: mean RMSE={rmse.mean():.6f}")

    # example spectra + SAM info
    idxs = np.random.choice(len(X_test), size=min(5, len(X_test)), replace=False)

    plt.figure(figsize=(12, 6))
    for idx in idxs:
        plt.plot(wavelengths, X_test[idx], alpha=0.8, label=f"True {idx}")
        plt.plot(wavelengths, X_recon[idx], "--", alpha=0.8, label=f"Recon {idx}")

    # Compute SAM for selected samples
    sam_examples, _ = compute_sam(X_test[idxs], X_recon[idxs])
    sam_deg = np.degrees(sam_examples)

    # Add SAM text block inside the figure
    sam_text = "\n".join([f"Sample {idx}: SAM = {sam_examples[i]:.4f} rad ({sam_deg[i]:.2f}°)"
                        for i, idx in enumerate(idxs)])

    plt.gcf().text(
        0.72, 0.35,      # position inside figure
        sam_text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("True vs reconstructed spectra (examples)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.65, 1])  # make room for SAM text box
    plt.savefig(os.path.join(cfg.results_dir, "spectra_reconstruction_examples.png"), dpi=300)
    plt.close()


    # per-band RMSE
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, per_band_rmse)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("RMSE")
    plt.title("Reconstruction error per wavelength")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "reconstruction_error_per_wavelength.png"), dpi=300)
    plt.close()

    # per-band R2
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, r2_spec)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("R²")
    plt.title("Spectral R² per wavelength")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "spectral_r2_per_wavelength.png"), dpi=300)
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

    # load data
    X_train, y_train, wavelengths = load_split(cfg.base_dir, "train")
    X_val, y_val, _ = load_split(cfg.base_dir, "val")
    X_test, y_test, _ = load_split(cfg.base_dir, "test")

    # standardize spectra
    X_train_z, X_val_z, X_test_z, stats = standardize_train_val_test(X_train, X_val, X_test)

    # load endmembers and standardize them in same way
    E, wavelengths_lib = load_endmember_matrix(cfg.endmember_path, material_names)
    assert np.allclose(wavelengths, wavelengths_lib), "Wavelength grids mismatch."

    mu = stats["mu"][None, :]
    sigma = stats["sigma"][None, :]
    E_z = (E - mu) / sigma
    E_z_torch = torch.from_numpy(E_z.astype(np.float32)).to(cfg.device)

    # datasets / loaders
    train_ds = SpectralDataset(X_train_z, y_train)
    val_ds = SpectralDataset(X_val_z, y_val)
    test_ds = SpectralDataset(X_test_z, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # model
    model = GumbelGroupedUnmixer(
        input_dim=X_train_z.shape[1],
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # training loop
    best_val = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.num_epochs + 1):
        tau = gumbel_tau_schedule(cfg, epoch)
        tr_loss, tr_ab, tr_rec, tr_cls = train_one_epoch_gumbel_group(
            model, train_loader, optimizer, cfg.device, E_z_torch, cfg, tau
        )
        va_loss, va_ab, va_rec, va_cls = eval_one_epoch_gumbel_group(
            model, val_loader, cfg.device, E_z_torch, cfg, tau
        )

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        print(
            f"Epoch {epoch:03d}  "
            f"train={tr_loss:.6f} (ab={tr_ab:.6f}, rec={tr_rec:.6f}, cls={tr_cls:.6f})  "
            f"val={va_loss:.6f} (ab={va_ab:.6f}, rec={va_rec:.6f}, cls={va_cls:.6f})"
        )

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg.model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("Early stopping.")
                break

    plot_loss_curves(train_losses, val_losses, cfg)

    # test evaluation with best model
    print(f"\nLoading best model (val_loss={best_val:.6f})")
    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))

    y_true, y_pred, mae, rmse, _ = evaluate_test(model, test_loader, cfg.device, material_names)
    plot_true_vs_pred(y_true, y_pred, material_names, cfg)
    plot_error_hist(y_true, y_pred, material_names, cfg)

    # spectral reconstruction in original domain
    reconstruct_and_plot(X_test, y_pred, E, wavelengths, cfg)
    print(f"\nResults written to: {cfg.results_dir}")


if __name__ == "__main__":
    main()

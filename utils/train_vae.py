import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.vae import PhysicsConstrainedVAE


# Dataset

class SpectralDataset(Dataset):
    def __init__(self, spectra_path, abundances_path):
        spectra_df = pd.read_csv(spectra_path)
        abundances_df = pd.read_csv(abundances_path)

        self.spectra = spectra_df.drop('sample_id', axis=1).values.astype(np.float32)
        self.abundances = abundances_df.drop('sample_id', axis=1).values.astype(np.float32)
        self.material_names = abundances_df.drop('sample_id', axis=1).columns.tolist()

        print(f"Loaded {len(self.spectra)} samples")

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return {
            'spectrum': torch.from_numpy(self.spectra[idx]),
            'abundance': torch.from_numpy(self.abundances[idx])
        }


# Loss Function

def vae_loss(recon_spectrum, input_spectrum, abundances_pred, abundances_true, 
             mu, log_var, group_info, alpha=1.0, beta=0.1, gamma=1.0, delta=1.0):
    """    
    Loss = α·L_recon + β·L_KL + γ·L_group + δ·L_categorical
    """
    # reconstruction loss = spectral MSE
    recon_loss = F.mse_loss(recon_spectrum, input_spectrum, reduction='mean')

    # KLD(q(z|x) || N(0,I)) = -0.5*Σ(1+log(σ²)-μ²-σ²)
    kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

    # group fraction supervision loss: [solar, mli, white]
    group_true = torch.stack([
        abundances_true[:, 0],  # solar cell
        abundances_true[:, 1:4].sum(dim=1),  # MLI materials
        abundances_true[:, 4:6].sum(dim=1),  # white materials
    ], dim=1)
    group_loss = F.mse_loss(group_info['group_fracs'], group_true, reduction='mean')

    # categorical supervision for within-group selection
    mli_labels = torch.argmax(abundances_true[:, 1:4], dim=1)
    white_labels = torch.argmax(abundances_true[:, 4:6], dim=1)

    categorical_loss = (F.cross_entropy(group_info['mli_logits'], mli_labels) +
                        F.cross_entropy(group_info['white_logits'], white_labels))

    total_loss = (
        alpha * recon_loss + 
        beta * kld_loss + 
        gamma * group_loss + 
        delta * categorical_loss
        )

    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'kld': kld_loss.item(),
        'group': group_loss.item(),
        'categorical': categorical_loss.item(),
    }

    return total_loss, loss_dict


# Training Functions

def train_epoch(model, train_loader, optimizer, device, alpha, beta, gamma, delta):
    model.train()
    total_loss = 0
    loss_components = {'recon': 0, 'kld': 0, 'group': 0, 'categorical': 0}

    for batch in train_loader:
        spectra = batch['spectrum'].to(device)
        abundances_true = batch['abundance'].to(device)

        recon_spectra, abundances_pred, mu, log_var, group_info = model(spectra)

        loss, loss_dict = vae_loss(recon_spectra, spectra, abundances_pred, abundances_true,
                                   mu, log_var, group_info, alpha, beta, gamma, delta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = len(spectra)
        total_loss += loss_dict['total'] * batch_size
        for key in loss_components:
            loss_components[key] += loss_dict[key] * batch_size

    n_samples = len(train_loader.dataset)
    avg_loss = total_loss / n_samples
    for key in loss_components:
        loss_components[key] /= n_samples

    return avg_loss, loss_components


def validate_epoch(model, val_loader, device, alpha, beta, gamma, delta):
    model.eval()
    total_loss = 0
    loss_components = {'recon': 0, 'kld': 0, 'group': 0, 'categorical': 0}

    with torch.no_grad():
        for batch in val_loader:
            spectra = batch['spectrum'].to(device)
            abundances_true = batch['abundance'].to(device)

            recon_spectra, abundances_pred, mu, log_var, group_info = model(spectra)

            _, loss_dict = vae_loss(recon_spectra, spectra, abundances_pred, abundances_true,
                                       mu, log_var, group_info, alpha, beta, gamma, delta)

            batch_size = len(spectra)
            total_loss += loss_dict['total'] * batch_size
            for key in loss_components:
                loss_components[key] += loss_dict[key] * batch_size

    n_samples = len(val_loader.dataset)
    avg_loss = total_loss / n_samples
    for key in loss_components:
        loss_components[key] /= n_samples

    return avg_loss, loss_components


# Evaluation Functions

def spectral_angle_mapper(y_true, y_pred):
    y_true_norm = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-10)
    y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-10)
    cos_sim = np.sum(y_true_norm * y_pred_norm, axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    sam = np.arccos(cos_sim) * 180.0 / np.pi
    return sam

def evaluate_model(model, test_loader, device, material_names):
    model.eval()

    all_spectra_true = []
    all_spectra_pred = []
    all_abundances_true = []
    all_abundances_pred = []

    with torch.no_grad():
        for batch in test_loader:
            spectra = batch['spectrum'].to(device)
            abundances_true = batch['abundance'].to(device)

            recon_spectra, abundances_pred, _, _, _ = model(spectra, hard=True)

            all_spectra_true.append(spectra.cpu().numpy())
            all_spectra_pred.append(recon_spectra.cpu().numpy())
            all_abundances_true.append(abundances_true.cpu().numpy())
            all_abundances_pred.append(abundances_pred.cpu().numpy())

    spectra_true = np.concatenate(all_spectra_true, axis=0)
    spectra_pred = np.concatenate(all_spectra_pred, axis=0)
    abundances_true = np.concatenate(all_abundances_true, axis=0)
    abundances_pred = np.concatenate(all_abundances_pred, axis=0)

    results = {
        'abundance_mae': mean_absolute_error(abundances_true, abundances_pred),
        'abundance_mse': mean_squared_error(abundances_true, abundances_pred),
        'abundance_rmse': np.sqrt(mean_squared_error(abundances_true, abundances_pred)),

        'spectral_mae': mean_absolute_error(spectra_true, spectra_pred),
        'spectral_mse': mean_squared_error(spectra_true, spectra_pred),
        'spectral_rmse': np.sqrt(mean_squared_error(spectra_true, spectra_pred)),

        'per_material': {}
    }

    for i, material in enumerate(material_names):
        results['per_material'][material] = {
            'mae': mean_absolute_error(abundances_true[:, i], abundances_pred[:, i]),
            'mse': mean_squared_error(abundances_true[:, i], abundances_pred[:, i]),
            'rmse': np.sqrt(mean_squared_error(abundances_true[:, i], abundances_pred[:, i])),
        }

    sam = spectral_angle_mapper(spectra_true, spectra_pred)
    results['sam_mean'] = float(np.mean(sam))
    results['sam_std'] = float(np.std(sam))
    results['sam_median'] = float(np.median(sam))

    results['predictions'] = {
        'spectra_true': spectra_true,
        'spectra_pred': spectra_pred,
        'abundances_true': abundances_true,
        'abundances_pred': abundances_pred
    }

    return results


# Plotting Functions

def plot_training_history(history, save_path):
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    epochs = range(1, len(history['train_loss']) + 1)

    # total loss
    ax_total = fig.add_subplot(gs[0, :])
    ax_total.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax_total.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax_total.set_xlabel('Epoch')
    ax_total.set_ylabel('Total Loss')
    ax_total.set_title('Total Loss')
    ax_total.legend()
    ax_total.grid(True, alpha=0.3)

    # reconstruction loss
    ax_recon = fig.add_subplot(gs[1, 0])
    ax_recon.plot(epochs, history['train_recon'], label='Train', linewidth=2)
    ax_recon.plot(epochs, history['val_recon'], label='Validation', linewidth=2)
    ax_recon.set_xlabel('Epoch')
    ax_recon.set_ylabel('Reconstruction Loss')
    ax_recon.set_title('Spectral Reconstruction Loss')
    ax_recon.legend()
    ax_recon.grid(True, alpha=0.3)

    # KL Divergence
    ax_kld = fig.add_subplot(gs[1, 1])
    ax_kld.plot(epochs, history['train_kld'], label='Train', linewidth=2)
    ax_kld.plot(epochs, history['val_kld'], label='Validation', linewidth=2)
    ax_kld.set_xlabel('Epoch')
    ax_kld.set_ylabel('KL Divergence')
    ax_kld.set_title('KL Divergence')
    ax_kld.legend()
    ax_kld.grid(True, alpha=0.3)

    # group loss
    ax_group = fig.add_subplot(gs[2, 0])
    ax_group.plot(epochs, history['train_group'], label='Train', linewidth=2)
    ax_group.plot(epochs, history['val_group'], label='Validation', linewidth=2)
    ax_group.set_xlabel('Epoch')
    ax_group.set_ylabel('Group Loss')
    ax_group.set_title('Group Supervision Loss')
    ax_group.legend()
    ax_group.grid(True, alpha=0.3)

    # categorical loss
    ax_cat = fig.add_subplot(gs[2, 1])
    ax_cat.plot(epochs, history['train_categorical'], label='Train', linewidth=2)
    ax_cat.plot(epochs, history['val_categorical'], label='Validation', linewidth=2)
    ax_cat.set_xlabel('Epoch')
    ax_cat.set_ylabel('Categorical Loss')
    ax_cat.set_title('Categorical Supervision Loss')
    ax_cat.legend()
    ax_cat.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_test_results(results, material_names, wavelengths, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    abundances_true = results['predictions']['abundances_true']
    abundances_pred = results['predictions']['abundances_pred']
    spectra_true = results['predictions']['spectra_true']
    spectra_pred = results['predictions']['spectra_pred']

    # abundance predictions scatter plot
    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, material in enumerate(material_names):
        ax = axes[i]
        true = abundances_true[:, i]
        pred = abundances_pred[:, i]

        ax.scatter(true, pred, alpha=0.5, s=30)

        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        mae = results['per_material'][material]['mae']
        rmse = results['per_material'][material]['rmse']

        ax.set_xlabel('True Abundance')
        ax.set_ylabel('Predicted Abundance')
        ax.set_title(f'{material}\nMAE={mae:.4f}, RMSE={rmse:.4f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'abundance_predictions.png', dpi=300)
    plt.close()

    # spectral reconstructions plot
    _, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    n_examples = min(10, len(spectra_true))
    indices = np.random.choice(len(spectra_true), n_examples, replace=False)

    for idx, sample_idx in enumerate(indices):
        ax = axes[idx]
        ax.plot(wavelengths, spectra_true[sample_idx], label='True', linewidth=2, alpha=0.7)
        ax.plot(wavelengths, spectra_pred[sample_idx], label='Predicted', linewidth=2, alpha=0.7)

        sam = spectral_angle_mapper(spectra_true[sample_idx:sample_idx+1], 
                                    spectra_pred[sample_idx:sample_idx+1])[0]

        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        ax.set_ylabel('Reflectance', fontsize=9)
        ax.set_title(f'Sample {sample_idx}, SAM={sam:.2f}°', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'spectral_reconstructions.png', dpi=300)
    plt.close()

    # error distributions plot
    errors = abundances_pred - abundances_true

    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, material in enumerate(material_names):
        ax = axes[i]
        error = errors[:, i]

        ax.hist(error, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2)

        mean_error = np.mean(error)
        std_error = np.std(error)

        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{material}\nMean={mean_error:.4f}, Std={std_error:.4f}')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'error_distributions.png', dpi=300)
    plt.close()


# Training Pipeline

def train(train_spectra, train_abundances, val_spectra, val_abundances, test_spectra, test_abundances, 
          endmembers, batch_size, epochs, lr, alpha, beta, gamma, delta,
          patience=20, seed=42, results_dir=None, device=None):

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"α: {alpha}, β: {beta}, γ: {gamma}, δ: {delta}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)

    train_dataset = SpectralDataset(train_spectra, train_abundances)
    val_dataset = SpectralDataset(val_spectra, val_abundances)
    test_dataset = SpectralDataset(test_spectra, test_abundances)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    endmembers_df = pd.read_csv(endmembers)
    material_names = train_dataset.material_names
    endmember_matrix = endmembers_df[material_names].values.astype(np.float32)
    wavelengths = endmembers_df['wavelength_nm'].values

    print(f"Endmembers: {endmember_matrix.shape}")
    print(f"Wavelengths: {len(wavelengths)}")

    model = PhysicsConstrainedVAE(endmember_spectra=endmember_matrix, input_dim=len(wavelengths)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=patience//2)

    history = {
        'train_loss': [], 'train_recon': [], 'train_kld': [], 'train_group': [], 'train_categorical': [],
        'val_loss': [], 'val_recon': [], 'val_kld': [], 'val_group': [], 'val_categorical': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss, train_components = train_epoch(model, train_loader, optimizer, device, 
                                                   alpha, beta, gamma, delta)

        val_loss, val_components = validate_epoch(model, val_loader, device, 
                                                  alpha, beta, gamma, delta)

        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_components['recon'])
        history['train_kld'].append(train_components['kld'])
        history['train_group'].append(train_components['group'])
        history['train_categorical'].append(train_components['categorical'])
        history['val_loss'].append(val_loss)
        history['val_recon'].append(val_components['recon'])
        history['val_kld'].append(val_components['kld'])
        history['val_group'].append(val_components['group'])
        history['val_categorical'].append(val_components['categorical'])

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training => "
              f"Total Loss: {train_loss:.4f}, Reconstruction Loss: {train_components['recon']:.4f}, "
              f"KLD Loss: {train_components['kld']:.4f}, Group Loss: {train_components['group']:.4f}, "
              f"Categorical Loss: {train_components['categorical']:.4f}")
        print(f"Validation => "
              f"Total Loss {val_loss:.4f}, Reconstruction Loss: {val_components['recon']:.4f}, "
              f"KLD Loss: {val_components['kld']:.4f}, Group Loss: {val_components['group']:.4f}, "
              f"Categorical Loss: {val_components['categorical']:.4f}")

        scheduler.step(val_loss)

        # saving best model in memory
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"Best validation loss: {best_val_loss:.4f}")

    if results_dir is not None:
        with open(results_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        plot_training_history(history, results_dir / 'training_curves.png')

    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    test_results = evaluate_model(model, test_loader, device, material_names)

    results_log = {
        'abundance_mae': test_results['abundance_mae'],
        'abundance_rmse': test_results['abundance_rmse'],
        'spectral_mse': test_results['spectral_mse'],
        'spectral_rmse': test_results['spectral_rmse'],
        'sam_mean': test_results['sam_mean'],
        'sam_std': test_results['sam_std'],
        'sam_median': test_results['sam_median'],
        'per_material': test_results['per_material']
    }

    if results_dir is not None:
        with open(results_dir / 'test_results.json', 'w') as f:
            json.dump(results_log, f, indent=2)

        plot_test_results(test_results, material_names, wavelengths, results_dir)

    return results_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data params
    parser.add_argument('--train-spectra', default='data/synthetic_dataset_splits/train_spectra.csv')
    parser.add_argument('--train-abundances', default='data/synthetic_dataset_splits/train_abundances.csv')
    parser.add_argument('--val-spectra', default='data/synthetic_dataset_splits/val_spectra.csv')
    parser.add_argument('--val-abundances', default='data/synthetic_dataset_splits/val_abundances.csv')
    parser.add_argument('--test-spectra', default='data/synthetic_dataset_splits/test_spectra.csv')
    parser.add_argument('--test-abundances', default='data/synthetic_dataset_splits/test_abundances.csv')
    parser.add_argument('--endmembers', default='data/endmember_library_clipped.csv')

    # training params
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=1.0, help='recon loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='kld loss weight')
    parser.add_argument('--gamma', type=float, default=1.0, help='group loss weight')
    parser.add_argument('--delta', type=float, default=1.0, help='categorical loss weight')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    results = train(train_spectra=args.train_spectra, train_abundances=args.train_abundances,
                    val_spectra=args.val_spectra, val_abundances=args.val_abundances,
                    test_spectra=args.test_spectra, test_abundances=args.test_abundances,
                    endmembers=args.endmembers, batch_size=args.batch_size, 
                    epochs=args.epochs, lr=args.lr,
                    alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
                    patience=args.patience, seed=args.seed)

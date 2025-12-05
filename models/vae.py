import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """
    Encoder network: spectrum → MLP → latent distribution (μ_z, log_σ_z)
    8-dimensional latent space with physical interpretation
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[256, 128]):
        super().__init__()
    
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class LatentToAbundance(nn.Module):
    """
    Converts 8-dimensional interpretable latent space to 6-dimensional abundances
    Latent structure:
        z[0:3] = group fraction logits → softmax → [solar_frac, mli_frac, white_frac]
        z[3:6] = MLI selection logits → softmax → [p(blackmli), p(kapton), p(dunmore)]
        z[6:8] = white selection logits → softmax → [p(thermalbrightn), p(kaptonws)]
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, hard=False):
        group_logits = z[:, 0:3]  # group [solar, mli, white] fractions
        mli_logits = z[:, 3:6]  # MLI material
        white_logits = z[:, 6:8]  # white material

        group_fracs = F.softmax(group_logits, dim=1)

        if hard:  # discrete one-hot selection for inference
            mli_probs = F.gumbel_softmax(mli_logits, tau=self.temperature, hard=hard)
            white_probs = F.gumbel_softmax(white_logits, tau=self.temperature, hard=hard)
        else:  # soft differentiable selection for training
            mli_probs = F.softmax(mli_logits / self.temperature, dim=1)
            white_probs = F.softmax(white_logits / self.temperature, dim=1)

        # final abundances
        abundances = torch.cat([
            group_fracs[:, 0:1],  # solar cell
            group_fracs[:, 1:2] * mli_probs,  # MLI materials
            group_fracs[:, 2:3] * white_probs,  # white materials
        ], dim=1)

        group_info = {
            'group_fracs': group_fracs,
            'mli_logits': mli_logits,
            'mli_probs': mli_probs,
            'white_logits': white_logits,
            'white_probs': white_probs
        }

        return abundances, group_info


class PhysicsDecoder(nn.Module):
    """
    Physics-based decoder: fixed linear mixing model with no learnable parameters. 
    Enforces the physical constraint that observed spectra are linear mixtures of endmembers:
        x(λ) = Σ_i a_i * s_i(λ),
    where: a_i = abundance of material i, s_i(λ) = spectral signature of material i at wavelength λ
    """
    def __init__(self, endmember_spectra):
        super().__init__()

        if isinstance(endmember_spectra, np.ndarray):
            endmember_spectra = torch.from_numpy(endmember_spectra).float()

        self.register_buffer('endmembers', endmember_spectra)  # registers endmembers as buffer (untrainable)

    def forward(self, abundances):
        """
        Linear mixing: spectrum = abundances @ endmembers^T
        """
        spectrum = torch.matmul(abundances, self.endmembers.T)
        return spectrum


class PhysicsConstrainedVAE(nn.Module):
    """
    Encoder: spectrum (501) → latent (μ_z, log_σ_z) ∈ ℝ⁸ -> Reparameterization: z ~ N(μ, σ²) -> 
    LatentToAbundance: z (8) → abundances (6) -> Physics Decoder: abundances (6) → reconstructed spectrum (501)
    """
    def __init__(self, endmember_spectra, input_dim, latent_dim=8, 
                 temperature=1.0, encoder_hidden=[256, 128]):
        super().__init__()
        assert latent_dim == 8, (f"Latent dim must be 8 for hierarchical structure: "
                                 f"3 (group) + 3 (MLI) + 2 (white), got {latent_dim}")

        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, latent_dim, encoder_hidden)
        self.latent_to_abundance = LatentToAbundance(temperature)
        self.physics_decoder = PhysicsDecoder(endmember_spectra)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = μ + σ ⊙ ε, where ε ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x, hard=False):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        abundances, group_info = self.latent_to_abundance(z, hard=hard)
        recon_spectrum = self.physics_decoder(abundances)

        return recon_spectrum, abundances, mu, log_var, group_info


if __name__ == "__main__":
    # dummy data
    batch_size = 32
    n_bands = 501
    latent_dim = 8
    n_materials = 6

    endmembers = np.random.rand(n_bands, n_materials).astype(np.float32)
    x = torch.randn(batch_size, n_bands)

    model = PhysicsConstrainedVAE(endmembers, n_bands, latent_dim, 
                                  temperature=1.0, encoder_hidden=[256, 128])

    recon_spectrum, abundances_pred, mu, log_var, group_info = model(x)  # forward pass

    print(f"Input shape: {x.shape}")
    print(f"Latent mean shape: {mu.shape} (8 = 3 group + 3 MLI + 2 white)")
    print(f"Latent log_var shape: {log_var.shape}")
    print(f"Predicted abundances shape: {abundances_pred.shape}")
    print(f"Reconstructed spectrum shape: {recon_spectrum.shape}")
    print(f"Abundances sum: {abundances_pred.sum(dim=1).mean():.3f}")
    print(f"Group fractions sum: {group_info['group_fracs'].sum(dim=1).mean():.3f}")
    print(f"MLI probs sum: {group_info['mli_probs'].sum(dim=1).mean():.3f}")
    print(f"White probs sum: {group_info['white_probs'].sum(dim=1).mean():.3f}")

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
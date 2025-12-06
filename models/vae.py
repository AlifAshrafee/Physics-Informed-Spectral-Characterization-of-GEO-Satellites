import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """
    Encoder network: spectrum → MLP → latent distribution (μ_z, log_σ_z)
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128]):
        super(Encoder, self).__init__()
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


class HierarchicalAbundanceProjector(nn.Module):
    """
    Abundance projector network: z → MLP → logits → softmax → abundances
    """
    def __init__(self, latent_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.shared_projector = nn.Sequential(*layers)

        # group fraction prediction head: [solar_frac, mli_frac, white_frac]
        self.group_fractions = nn.Linear(hidden_dims[-1], 3)

        # within-group prediction heads
        self.mli_selector = nn.Linear(hidden_dims[-1], 3)  # MLI materials
        self.white_selector = nn.Linear(hidden_dims[-1], 2)  # white materials

    def forward(self, z, hard=False):
        h = self.shared_projector(z)

        group_logits = self.group_fractions(h)  # [solar, mli, white]
        group_fracs = F.softmax(group_logits, dim=1)  # ensures sum to 1

        # within-group categorical selections
        mli_logits = self.mli_selector(h)
        white_logits = self.white_selector(h)

        if hard:  # hard one-hot selection for inference
            mli_probs = F.gumbel_softmax(mli_logits, tau=1, hard=hard)
            white_probs = F.gumbel_softmax(white_logits, tau=1, hard=hard)
        else:  # soft differentiable selection for training
            mli_probs = F.softmax(mli_logits, dim=1)
            white_probs = F.softmax(white_logits, dim=1)

        abundances = torch.cat([
            group_fracs[:, 0:1],  # solar cell
            group_fracs[:, 1:2] * mli_probs,  # MLI materials
            group_fracs[:, 2:3] * white_probs  # white materials
        ], dim=1)

        return abundances


class PhysicsDecoder(nn.Module):
    """
    Physics-based decoder: fixed linear mixing model with no learnable parameters. 
    Enforces the physical constraint that observed spectra are linear mixtures of endmembers:
        x(λ) = Σ_i a_i * s_i(λ),
    where: a_i = abundance of material i, s_i(λ) = spectral signature of material i at wavelength λ
    """
    def __init__(self, endmember_spectra):
        super(PhysicsDecoder, self).__init__()

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
    Encoder: spectrum → (μ_z, log_σ_z) -> Reparameterization: z ~ N(μ, σ²) -> 
    Abundance Projector: z → abundances -> Physics Decoder: abundances → reconstructed spectrum
    """
    def __init__(self, endmember_spectra, input_dim, latent_dim, 
                 encoder_hidden=[256, 128], projector_hidden=[128, 64]):
        super(PhysicsConstrainedVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, latent_dim, encoder_hidden)
        self.abundance_projector = HierarchicalAbundanceProjector(latent_dim, projector_hidden)
        self.physics_decoder = PhysicsDecoder(endmember_spectra)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = μ + σ ⊙ ε, where ε ~ N(0, 1)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x, hard=False):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        abundances = self.abundance_projector(z, hard=hard)
        recon_spectrum = self.physics_decoder(abundances)

        return recon_spectrum, abundances, mu, log_var


if __name__ == "__main__":
    # dummy data
    batch_size = 32
    n_bands = 501
    latent_dim = 32
    n_materials = 6
    endmembers = np.random.rand(n_bands, n_materials).astype(np.float32)
    x = torch.randn(batch_size, n_bands)
    abundances_true = torch.randn(batch_size, n_materials).softmax(dim=1)

    model = PhysicsConstrainedVAE(endmembers, n_bands, latent_dim, 
                                  encoder_hidden=[256, 128], projector_hidden=[128, 64])

    recon_spectrum, abundances_pred, mu, log_var = model(x)  # forward pass

    print(f"Input shape: {x.shape}")
    print(f"Reconstructed spectrum shape: {recon_spectrum.shape}")
    print(f"Predicted abundances shape: {abundances_pred.shape}")
    print(f"Abundances sum: {abundances_pred.sum(dim=1).mean():.3f}")
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent log_var shape: {log_var.shape}")

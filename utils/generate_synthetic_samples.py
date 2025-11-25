"""
This script generates synthetic spectral samples by mixing endmember materials
according to physically realistic constraints for GEO satellite characterization.

Physical Constraints:
- Solar Cell: 5-60% (always present)
- MLI materials (mutually exclusive): Black MLI (5-30%), Kapton HN (5-30%), or Dunmore (5-30%)
- White materials (mutually exclusive): ThermalBright N (5-20%) or Kapton WS (5-20%)
- All abundances must sum to 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

# random seed for reproducibility
np.random.seed(42)


class ConstrainedMixingGenerator:
    def __init__(self, endmember_library_path: str):
        self.df = pd.read_csv(endmember_library_path)
        self.wavelengths = self.df["wavelength_nm"].values
        self.mli_materials = ["blackmli", "kaptonhn", "dunmore"]
        self.white_materials = ["thermalbrightn", "kaptonws"]

        # material groups and constraints
        self.materials = {
            "solarcell": {"min": 0.05, "max": 0.60, "required": True},
            "blackmli": {"min": 0.05, "max": 0.30, "required": False, "group": "MLI"},
            "kaptonhn": {"min": 0.05, "max": 0.30, "required": False, "group": "MLI"},
            "dunmore": {"min": 0.05, "max": 0.30, "required": False, "group": "MLI"},
            "thermalbrightn": {"min": 0.05, "max": 0.20, "required": False, "group": "WHITE"},
            "kaptonws": {"min": 0.05, "max": 0.20, "required": False, "group": "WHITE"},
        }

        print(f"\nMaterial constraints:")
        for material, constraints in self.materials.items():
            print(f"{material}: {constraints['min']*100:.1f}% - {constraints['max']*100:.1f}%")

    def sample_abundances(self):
        abundances = {}

        # randomly selects one MLI material
        selected_mli = np.random.choice(self.mli_materials)

        # randomly selects one white material
        selected_white = np.random.choice(self.white_materials)

        # samples abundances using Dirichlet-like approach with constraints
        while True:
            # samples from uniform distributions within allowed ranges
            a_solar = np.random.uniform(
                self.materials["solarcell"]["min"], 
                self.materials["solarcell"]["max"]
            )
            a_mli = np.random.uniform(
                self.materials[selected_mli]["min"], 
                self.materials[selected_mli]["max"]
            )
            a_white = np.random.uniform(
                self.materials[selected_white]["min"],
                self.materials[selected_white]["max"],
            )

            total = a_solar + a_mli + a_white

            if total <= 1.0:  # normalizes to sum to exactly 1.0
                abundances["solarcell"] = a_solar / total
                abundances[selected_mli] = a_mli / total
                abundances[selected_white] = a_white / total

                # sets excluded materials to 0
                for material in self.mli_materials:
                    if material != selected_mli:
                        abundances[material] = 0.0
                for material in self.white_materials:
                    if material != selected_white:
                        abundances[material] = 0.0

                # verifies constraints are still satisfied after normalization
                if self._verify_constraints(abundances):
                    return abundances

    def _verify_constraints(self, abundances: Dict[str, float]):
        for material, value in abundances.items():
            if value > 0:  # only checks active materials
                constraints = self.materials[material]
                if value < constraints["min"] or value > constraints["max"]:
                    return False
        return abs(sum(abundances.values()) - 1.0) < 1e-6

    def generate_mixed_spectrum(self, abundances: Dict[str, float]):
        mixed_spectrum = np.zeros(len(self.wavelengths))

        for material, abundance in abundances.items():
            if abundance > 0:
                material_spectrum = self.df[material].values
                mixed_spectrum += abundance * material_spectrum

        return mixed_spectrum

    def generate_samples(self, n_samples: int, add_noise: bool = True, noise_level: float = 0.02):
        print(f"\Generating {n_samples} synthetic samples")

        material_names = [
            "solarcell",
            "blackmli",
            "kaptonhn",
            "dunmore",
            "thermalbrightn",
            "kaptonws",
        ]

        spectra = np.zeros((n_samples, len(self.wavelengths)))
        abundances_matrix = np.zeros((n_samples, len(material_names)))
        metadata = []

        for i in range(n_samples):
            abundances = self.sample_abundances()

            spectrum = self.generate_mixed_spectrum(abundances)

            if add_noise:
                noise = np.random.normal(0, noise_level * spectrum.mean(), len(spectrum))
                spectrum = np.clip(spectrum + noise, 0, 1)

            spectra[i, :] = spectrum
            abundances_matrix[i, :] = [abundances[material] for material in material_names]

            active_materials = [material for material, val in abundances.items() if val > 0]
            metadata.append(
                {
                    "sample_id": i,
                    "active_materials": active_materials,
                    "abundances": abundances.copy(),
                    "noise_added": add_noise,
                    "noise_level": noise_level if add_noise else 0.0,
                }
            )

            # progress logging
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")

        print(f"Generated {n_samples} samples")
        return spectra, abundances_matrix, metadata

    def save_dataset(self, spectra: np.ndarray, abundances: np.ndarray, 
                     metadata: List[Dict], output_prefix: str = "synthetic_dataset"):

        spectra_df = pd.DataFrame(spectra, columns=self.wavelengths)
        spectra_df.insert(0, "sample_id", range(len(spectra)))
        spectra_path = f"data/{output_prefix}_spectra.csv"
        spectra_df.to_csv(spectra_path, index=False)

        material_names = [
            "solarcell",
            "blackmli",
            "kaptonhn",
            "dunmore",
            "thermalbrightn",
            "kaptonws",
        ]

        abundances_df = pd.DataFrame(abundances, columns=material_names)
        abundances_df.insert(0, "sample_id", range(len(abundances)))
        abundances_path = f"data/{output_prefix}_abundances.csv"
        abundances_df.to_csv(abundances_path, index=False)

        metadata_path = f"data/{output_prefix}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def visualize_samples(self, spectra: np.ndarray, abundances: np.ndarray, n_examples: int = 10):
        material_names = [
            "solarcell",
            "blackmli",
            "kaptonhn",
            "dunmore",
            "thermalbrightn",
            "kaptonws",
        ]

        # plots example spectra
        _, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes = axes.ravel()

        for i in range(min(n_examples, len(spectra))):
            axes[0].plot(
                self.wavelengths,
                spectra[i],
                alpha=0.7,
                linewidth=1.5,
            )
        axes[0].set_xlabel("Wavelength (nm)", fontsize=12)
        axes[0].set_ylabel("Reflectance", fontsize=12)
        axes[0].set_title(
            f"All Synthetic Samples Spectra (n={min(n_examples, len(spectra))})",
            fontsize=14,
        )
        axes[0].grid(True, alpha=0.3)

        # plots abundance distribution
        abundance_means = abundances.mean(axis=0)
        abundance_stds = abundances.std(axis=0)

        x_pos = np.arange(len(material_names))
        axes[1].bar(
            x_pos,
            abundance_means,
            yerr=abundance_stds,
            capsize=5,
            alpha=0.7,
            color="steelblue",
        )
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(material_names, rotation=45, ha="right")
        axes[1].set_ylabel("Mean Abundance Fraction", fontsize=12)
        axes[1].set_title(
            "Mean Material Abundances Across All Samples",
            fontsize=14,
        )
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("data/visuals/synthetic_samples_overview.png", dpi=300, bbox_inches="tight")
        plt.close()

        # abundance distribution histograms
        _, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for idx, material in enumerate(material_names):
            ax = axes[idx]
            material_abundances = abundances[:, idx]

            # only plots non-zero values
            non_zero_abundances = material_abundances[material_abundances > 0]

            if len(non_zero_abundances) > 0:
                ax.hist(
                    non_zero_abundances,
                    bins=30,
                    alpha=0.7,
                    color="steelblue",
                    edgecolor="black",
                )
                ax.axvline(
                    non_zero_abundances.mean(),
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label=f"Mean={non_zero_abundances.mean():.3f}",
                )
                ax.set_xlabel("Abundance Fraction", fontsize=10)
                ax.set_ylabel("Frequency", fontsize=10)
                ax.set_title(
                    f"{material} ({len(non_zero_abundances)} non-zero samples)",
                    fontsize=11,
                )
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                raise ValueError(f"No non-zero abundances found for material: {material}")

        plt.tight_layout()
        plt.savefig("data/visuals/abundance_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

        # material co-occurrence matrix
        _, ax = plt.subplots(figsize=(10, 8))

        # creates binary presence matrix
        presence = (abundances > 0).astype(int)
        cooccurrence = presence.T @ presence

        sns.heatmap(
            cooccurrence,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            xticklabels=material_names,
            yticklabels=material_names,
            cbar_kws={"label": "Co-occurrence Count"},
            ax=ax,
        )
        ax.set_title(
            "Material Co-occurrence Matrix (How often materials appear together)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.savefig("data/visuals/material_cooccurrence.png", dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    generator = ConstrainedMixingGenerator("data/endmember_library_clipped.csv")

    n_samples = 1000
    spectra, abundances, metadata = generator.generate_samples(
        n_samples=n_samples,
        add_noise=True,
        noise_level=0.02,  # 2% noise relative to mean signal
    )

    print("\nDataset Statistics:")
    print(f"Number of samples: {n_samples}")
    print(f"Spectral bands: {len(generator.wavelengths)}")
    print(f"Wavelength range: {generator.wavelengths[0]:.0f} - {generator.wavelengths[-1]:.0f} nm")

    print(f"\nAbundance statistics:")
    print(f"Min abundance (excluding zeros): {abundances[abundances > 0].min():.4f}")
    print(f"Max abundance: {abundances.max():.4f}")
    print(f"Sum check (should all be 1.0):")
    print(f"Min={abundances.sum(axis=1).min():.4f}, Max={abundances.sum(axis=1).max():.4f}")

    generator.save_dataset(spectra, abundances, metadata)

    generator.visualize_samples(spectra, abundances, n_examples=n_samples)

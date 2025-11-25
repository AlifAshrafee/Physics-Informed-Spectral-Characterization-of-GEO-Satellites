"""
This script runs the complete synthetic spectral sample generation pipeline 
with physical constraints for GEO satellite material unmixing.

Steps:
- Data sanity check and normalization
- Synthetic sample generation
- Train/valid/test split
"""

from data_sanity_check import *
from generate_synthetic_samples import *
from create_dataset_split import *


if __name__ == "__main__":
    # step 1: data sanity check and normalization
    df = load_and_inspect_data("data/endmember_library_gtri_ggx.csv")
    material_cols = perform_sanity_checks(df)
    visualize_spectra(df, material_cols)
    df_clipped = clip_data(df, material_cols)
    df_clipped.to_csv("data/endmember_library_clipped.csv", index=False)

    # step 2: synthetic sample generation
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

    # step 3: train/valid/test split
    spectra_df, abundances_df, metadata = load_synthetic_dataset(
        'data/synthetic_dataset_spectra.csv',
        'data/synthetic_dataset_abundances.csv',
        'data/synthetic_dataset_metadata.json'
    )

    splits = split_dataset(spectra_df, abundances_df, metadata, 
                           train_ratio=0.80, val_ratio=0.10, test_ratio=0.10)

    visualize_splits(splits)

    save_splits(splits)

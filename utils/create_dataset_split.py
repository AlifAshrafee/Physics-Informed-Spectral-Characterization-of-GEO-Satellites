"""
This script splits the synthetic dataset into training, validation, and test sets 
for machine learning model development.
"""

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# random seed for reproducibility
np.random.seed(42)

def load_synthetic_dataset(spectra_path, abundances_path, metadata_path):
    spectra_df = pd.read_csv(spectra_path)
    abundances_df = pd.read_csv(abundances_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return spectra_df, abundances_df, metadata

def split_dataset(spectra_df, abundances_df, metadata, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    n_samples = len(spectra_df)

    sample_ids = spectra_df['sample_id'].values

    train_ids, temp_ids = train_test_split(sample_ids, test_size=(val_ratio + test_ratio), random_state=42)

    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(temp_ids, test_size=(1 - val_ratio_adjusted), random_state=42)

    train_spectra = spectra_df[spectra_df['sample_id'].isin(train_ids)]
    val_spectra = spectra_df[spectra_df['sample_id'].isin(val_ids)]
    test_spectra = spectra_df[spectra_df['sample_id'].isin(test_ids)]

    train_abundances = abundances_df[abundances_df['sample_id'].isin(train_ids)]
    val_abundances = abundances_df[abundances_df['sample_id'].isin(val_ids)]
    test_abundances = abundances_df[abundances_df['sample_id'].isin(test_ids)]

    train_metadata = [m for m in metadata if m['sample_id'] in train_ids]
    val_metadata = [m for m in metadata if m['sample_id'] in val_ids]
    test_metadata = [m for m in metadata if m['sample_id'] in test_ids]

    print(f"Total samples: {n_samples}")
    print(f"Training: {len(train_ids):3d} samples ({len(train_ids)/n_samples*100:.2f}%)")
    print(f"Validation: {len(val_ids):3d} samples ({len(val_ids)/n_samples*100:.2f}%)")
    print(f"Test: {len(test_ids):3d} samples ({len(test_ids)/n_samples*100:.2f}%)")

    return {
        'train': {'spectra': train_spectra, 'abundances': train_abundances, 'metadata': train_metadata},
        'val': {'spectra': val_spectra, 'abundances': val_abundances, 'metadata': val_metadata},
        'test': {'spectra': test_spectra, 'abundances': test_abundances, 'metadata': test_metadata}
    }

def visualize_splits(splits):
    material_names = ['solarcell', 'blackmli', 'kaptonhn', 'dunmore', 'thermalbrightn', 'kaptonws']

    # abundance distributions across splits
    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for idx, material in enumerate(material_names):
        ax = axes[idx]
        for split_name, split_data in splits.items():
            abundances = split_data['abundances'][material].values
            non_zero = abundances[abundances > 0]
            if len(non_zero) > 0:
                ax.hist(non_zero, bins=20, alpha=0.5, label=split_name, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Abundance Fraction', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(material, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Material Abundance Distributions Across Splits', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('data/visuals/split_abundance_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # split size comparison
    _, ax = plt.subplots(figsize=(10, 6))

    split_sizes = [len(splits['train']['spectra']), 
                  len(splits['val']['spectra']), 
                  len(splits['test']['spectra'])]
    split_names = ['Training', 'Validation', 'Test']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    ax.bar(split_names, split_sizes, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Dataset Split Sizes', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('data/visuals/split_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_splits(splits, output_dir='data/synthetic_dataset_splits'):
    for split_name, split_data in splits.items():
        spectra_path = f'{output_dir}/{split_name}_spectra.csv'
        split_data['spectra'].to_csv(spectra_path, index=False)

        abundances_path = f'{output_dir}/{split_name}_abundances.csv'
        split_data['abundances'].to_csv(abundances_path, index=False)
        
        metadata_path = f'{output_dir}/{split_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(split_data['metadata'], f, indent=2)


if __name__ == "__main__":
    spectra_df, abundances_df, metadata = load_synthetic_dataset(
        'data/synthetic_dataset_spectra.csv',
        'data/synthetic_dataset_abundances.csv',
        'data/synthetic_dataset_metadata.json'
    )

    splits = split_dataset(spectra_df, abundances_df, metadata, 
                           train_ratio=0.80, val_ratio=0.10, test_ratio=0.10)

    visualize_splits(splits)

    save_splits(splits)
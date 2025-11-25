"""
This script loads the endmember library, performs sanity checks,
and prepares the data for synthetic sample generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_and_inspect_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: \n{df.columns.tolist()}\n")

    return df


def perform_sanity_checks(df):
    material_cols = [col for col in df.columns if col != "wavelength_nm"]

    # checks for negative values
    print("Checking for negative values:")
    for col in material_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"{col}: {neg_count} negative values found!")
        else:
            print(f"{col}: No negative values found!")

    # checks for values > 1
    print("\nChecking for values greater than 1.0:")
    for col in material_cols:
        over_one = (df[col] > 1.0).sum()
        if over_one > 0:
            print(f"{col}: {over_one} values exceed 1.0!")
        else:
            print(f"{col}: All values ≤ 1.0!")

    # checks for NaN/Inf values
    print("\nChecking for NaN/Inf values:")
    for col in material_cols:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"{col}: {nan_count} NaN, {inf_count} Inf values!")
        else:
            print(f"{col}: No NaN/Inf values found!")

    # statistical summary
    print("\nStatistical Summary:")
    print(df[material_cols].describe())
    print("")

    return material_cols


def visualize_spectra(df, material_cols):
    # plots all spectra
    _, ax = plt.subplots(figsize=(14, 8))

    # raw spectra
    for col in material_cols:
        ax.plot(df["wavelength_nm"], df[col], label=col, linewidth=2)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance", fontsize=12)
    ax.set_title("Raw Endmember Spectra", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Reflectance = 1.0")

    plt.tight_layout()
    plt.savefig("data/visuals/endmember_spectra_visualization_raw.png", dpi=300, bbox_inches="tight")
    plt.close()


def clip_data(df, material_cols):
    """
    Clips the data to ensure all reflectance values are in [0, 1]
    """
    df_clipped = df.copy()

    for col in material_cols:
        original_max = df[col].max()
        original_min = df[col].min()
        print(f"Processing {col}")
        print(f"Original: min={original_min:.4f}, max={original_max:.4f}")

        df_clipped[col] = np.clip(df[col], 0, 1)  # clips to [0, 1]
        clipped_count = ((df[col] < 0) | (df[col] > 1)).sum()
        if clipped_count > 0:
            print(f"Clipped {clipped_count} values")
            new_max = df_clipped[col].max()
            new_min = df_clipped[col].min()
            print(f"New: min={new_min:.4f}, max={new_max:.4f}")
        print("")

    # visualizes clipped spectra
    _, ax = plt.subplots(figsize=(14, 8))
    for col in material_cols:
        ax.plot(df_clipped["wavelength_nm"], df_clipped[col], label=col, linewidth=2)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance", fontsize=12)
    ax.set_title(f"Clipped Endmember Spectra", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig(f"data/visuals/endmember_spectra_visualization_clipped.png", dpi=300, bbox_inches="tight")
    plt.close()

    return df_clipped


if __name__ == "__main__":
    df = load_and_inspect_data("data/endmember_library_gtri_ggx.csv")

    material_cols = perform_sanity_checks(df)

    visualize_spectra(df, material_cols)

    df_clipped = clip_data(df, material_cols)

    df_clipped.to_csv("data/endmember_library_clipped.csv", index=False)

# Physics-Informed-Spectral-Characterization-of-GEO-Satellites
Application of ML in Remote Sensing Project on Space Domain Awareness (SDA) and Space Situational Awareness (SSA) using DIRSIG-simulated and telescope-collected data

## Synthetic Spectral Dataset Generation Pipeline

This pipeline generates synthetic spectral samples for GEO satellite material unmixing. It implements physics-based constraints to ensure realistic material mixing patterns consistent with actual spacecraft construction.

---

### Table of Contents
1. [Quick Start](#quick-start)
2. [Physical Constraints](#physical-constraints)
3. [Pipeline Components](#pipeline-components)
4. [Output Files](#output-files)
5. [Data Format](#data-format)

---

### Quick Start

#### Option 1: Run Complete Pipeline
```bash
python utils/run_data_generation_pipeline.py
```

This runs all three steps automatically:
1. Data sanity check and preprocessing
2. Synthetic sample generation (1000 samples)
3. Train/validation/test split (80%/10%/10%)

#### Option 2: Run Individual Steps
```bash
# step 1: data sanity check and preprocessing
python utils/data_sanity_check.py

# step 2: synthetic sample generation
python utils/generate_synthetic_samples.py

# step 3: train/valid/test split
python utils/create_dataset_split.py
```
---

### Physical Constraints

#### Material Abundance Ranges
The following constraints ensure physically realistic satellite surface compositions:

| Material | Range | Notes |
|----------|-------|-------|
| **Solar Cell** | 5-60% | Always present; largest contributor during glint |
| **Black MLI** | 5-30% | Kapton-coated black Multi-Layer Insulation |
| **Kapton HN** | 5-30% | Gold/amber Kapton MLI (older satellites) |
| **Dunmore** | 5-30% | Silver MLI with high reflectance |
| **ThermalBright N** | 5-20% | Reflective white radiators/coatings |
| **Kapton WS** | 5-20% | White Kapton for thermal coatings |

#### Mutual Exclusivity Rules
Real satellites don't mix incompatible materials:
- **Solar Cell**: Solar panels are always present for power generation
- **MLI Materials**: Different MLI types serve the same purpose → mutually exclusive. Only ONE of {Black MLI, Kapton HN, Dunmore} per sample
- **White Materials**: White materials serve the same thermal control purpose → mutually exclusive. Only ONE of {ThermalBright N, Kapton WS} per sample
- **Abundance Sum**: All materials must sum to exactly 1.0 (100%)

---

### Pipeline Components

#### 1. Data Sanity Check (`utils/data_sanity_check.py`)

- Checks for negative values
- Identifies values > 1.0 (e.g., solar cell oversaturation)
- Detects NaN/Inf values
- Generates statistical summaries
-  Clips reflectance to [0, 1] to preserve physical interpretation (reflectance ≤ 1.0)

---

#### 2. Synthetic Sample Generation (`utils/generate_synthetic_samples.py`)

- Generates 1000 synthetic spectral samples with physical constraints
- Adds realistic and configurable sensor noise
- Validates all physical constraints are satisfied
- Generates comprehensive metadata

**Algorithm**:
```
For each sample:
  - Randomly select ONE MLI material
  - Randomly select ONE white material
  - Sample abundances from allowed ranges using rejection sampling
  - Ensure abundances sum to 1.0
  - Mix endmember spectra using linear mixing model: 
      mixed_spectrum(λ) = Σ abundance_i × endmember_i(λ)
  - Add 2% Gaussian noise to simulate measurement uncertainty
```

---

#### 3. Train/Val/Test Split (`utils/create_dataset_split.py`)

- Splits the synthetic dataset for the machine learning pipeline
- Training: 80% (800 samples)
- Validation: 10% (100 samples)
- Test: 10% (100 samples)

---

### Output Files

#### Data Files

##### Preprocessed Endmember Library
```
data/endmember_library_clipped.csv
├── wavelength_nm: 380-880 nm (501 bands)
├── thermalbrightn: ThermalBright N reflectance
├── kaptonws: Kapton WS reflectance
├── kaptonhn: Kapton HN reflectance
├── blackmli: Black MLI reflectance
├── dunmore: Dunmore reflectance
└── solarcell: Solar cell reflectance
```

##### Synthetic Spectra
```
data/synthetic_dataset_spectra.csv (or train/val/test_spectra.csv)
├── sample_id: Unique sample identifier
└── 380.0, 381.0, ..., 880.0: Reflectance at each wavelength
```

##### Ground Truth Abundances
```
data/synthetic_dataset_abundances.csv (or train/val/test_abundances.csv)
├── sample_id: Unique sample identifier
├── solarcell: Abundance fraction [0, 1]
├── blackmli: Abundance fraction [0, 1]
├── kaptonhn: Abundance fraction [0, 1]
├── dunmore: Abundance fraction [0, 1]
├── thermalbrightn: Abundance fraction [0, 1]
└── kaptonws: Abundance fraction [0, 1]

Note: Sum of abundances = 1.0 for each sample
```

##### Metadata
```json
[
  {
    "sample_id": 0,
    "active_materials": [
      "solarcell",
      "dunmore",
      "kaptonws"
    ],
    "abundances": {
      "solarcell": 0.5962225817045373,
      "dunmore": 0.22684540815662846,
      "kaptonws": 0.17693201013883422,
      "blackmli": 0.0,
      "kaptonhn": 0.0,
      "thermalbrightn": 0.0
    },
    "noise_added": true,
    "noise_level": 0.02
  }
  ...
]
```

---

### Data Format

#### Spectral Data Shape
- **Training**: (800, 501) - 800 samples × 501 wavelength bands
- **Validation**: (100, 501) - 100 samples × 501 wavelength bands
- **Test**: (100, 501) - 100 samples × 501 wavelength bands

#### Abundance Data Shape
- **Training**: (800, 6) - 800 samples × 6 materials
- **Validation**: (100, 6) - 100 samples × 6 materials
- **Test**: (100, 6) - 100 samples × 6 materials

#### Material Order
The materials are always in the following order:

```python
material_names = [
    'solarcell',      # index 0
    'blackmli',       # index 1
    'kaptonhn',       # index 2
    'dunmore',        # index 3
    'thermalbrightn', # index 4
    'kaptonws'        # index 5
]
```
---

## Citation
If you use this codebase in your research, please cite:

```
Jason T. Kirkendall, Alif Ashrafee, and Akib Khan.
"Physics-Informed Spectral Characterization of GEO Satellites Using DIRSIG-Simulated and Telescope-Collected Data." 
Chester F. Carlson Center for Imaging Science, Rochester Institute of Technology, October 2025.
```

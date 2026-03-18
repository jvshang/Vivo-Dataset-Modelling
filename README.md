# VivoProject

Activity recognition pipeline for EMG and IMU data collected during exoskeleton-assisted sit-to-stand trials.

## Data

**The `data/` folder is not included in this repository** (excluded via `.gitignore` due to file size).

Raw data is stored on Isambard at: `/lus/lfs1aip2/projects/b5bb/public`

```
data/
├── EMG/        # Electromyography recordings (~1926 Hz)
└── IMU/        # Inertial Measurement Unit recordings (100 Hz)
```

### File naming convention

```
{Subject}_{Condition}_{Trial}_{Stage}_{Modality}.csv

e.g. SBJ1_EXO_A_final_EMG.csv
```

| Field | Values | Description |
|-------|--------|-------------|
| Subject | SBJ1 – SBJ6 | Participant ID |
| Condition | `EXO` / `NoEXO` | With or without exoskeleton |
| Trial | A – G | Trial letter |
| Stage | `final` / `trimmed` | Use `final` — contains Activity labels |
| Modality | `EMG` / `IMU` | Sensor type |

> **Note:** SBJ3 has no EXO data (participant did not wear the exoskeleton during data collection).

### Sensor channels

| Modality | Channels | Sampling Rate |
|----------|----------|---------------|
| EMG | Rectus Femoris, Vastus Medialis, Vastus Lateralis, Bicep Femoris, Tibialis, Gastrocnemius Medialis | ~1926 Hz |
| IMU | R Thigh Angle, R Shank Angle, L Thigh Angle, L Shank Angle, Hip Angle | 100 Hz |

### Activity labels

`Sat`, `Stood`, `Standing up`, `Sitting down`, `Unknown`

---

## Project structure

```
VivoProject_local/
├── data/               # Local only — not in git
│   ├── EMG/
│   └── IMU/
├── src/
│   ├── dataloader.py   # Data loading, sliding window, benchmark splits
│   ├── train.py        # Training + wandb experiment tracking
│   └── main.py         # Example usage
└── sweep.yaml          # wandb sweep config (lead time vs. accuracy grid search)
```

---

## Implemented Features

### 1. Data Loading
- `list_sessions(subject, condition, modality)` — list all available `final` sessions
- `load_emg(subject, condition, trial)` — load EMG CSV(s) into a DataFrame
- `load_imu(subject, condition, trial)` — load IMU CSV(s) into a DataFrame (consolidates the 5 separate time columns into one)

### 2. Sliding Window
- `sliding_windows(df, L, S, fs, feature_cols)` — applies a sliding window to a single session
  - `L`: window length in seconds
  - `S`: stride in seconds (`S < L` gives overlapping windows)
  - Label per window: majority vote; windows where the majority label is `Unknown` are discarded
  - Returns `X` of shape `(N_windows, window_size, n_features)` and `y` of shape `(N_windows,)`

### 3. Dataset Builder
- `make_dataset(subjects, conditions, L, S)` — builds a windowed dataset for given subjects and conditions
  - EMG and IMU are windowed independently at their native sampling rates, then concatenated along the feature axis
  - Returns flattened `X` of shape `(N_windows, n_features)` and `y`
  - At `L=0.1` s: feature dim = 192×6 (EMG) + 10×5 (IMU) = **1202**

### 5. Experiment Tracking (wandb)

- `train.py` — trains a configurable sklearn classifier and logs four metrics to wandb after each run:

| Metric | wandb key | Description |
|--------|-----------|-------------|
| Accuracy | `accuracy` | Test-set classification accuracy |
| Lead Time | `lead_time_s` | = window length L — minimum data needed before a prediction |
| Latency | `latency_ms` | Mean single-sample inference time in milliseconds |
| Model Size | `model_size_mb` | Serialised size in MB; `size_ok = True` if < 5 MB |

- Supported classifiers: Random Forest (`rf`), Gradient Boosting (`gb`), Logistic Regression (`lr`)
- `sweep.yaml` — grid search over window lengths `[0.05, 0.1, 0.2, 0.5]` s to visualise the lead time vs. accuracy trade-off in the wandb dashboard

### 4. Benchmark Splits
- **Task 1 – General Purpose Training**
  - Train: 3 subjects (EXO + NoEXO) + SBJ3 (NoEXO only, always included) | Test: 2 subjects (EXO + NoEXO)
  - Three fixed benchmark splits via `split` argument: `"hard"` (test SBJ5+SBJ6), `"mixed"` (test SBJ1+SBJ5), `"easy"` (test SBJ1+SBJ2)
  - `task1_split(split="hard", L, S)`

- **Task 2 – Exo Challenge**
  - Train: all 6 subjects, NoEXO only | Test: SBJ1/2/4/5/6, EXO only
  - `task2_split(L, S)`

---

## Usage

Run the demo script from the project root:

```bash
python src/main.py
```

Or import directly in your own script (run from project root):

```python
import sys
sys.path.insert(0, "src")

from dataloader import (
    list_sessions, load_emg, load_imu,
    sliding_windows, make_dataset,
    task1_split, task2_split,
    EMG_FS, IMU_FS, EMG_MUSCLES, IMU_ANGLES,
)

# List all available sessions
sessions = list_sessions(modality="EMG")

# Load raw data for a single session
emg = load_emg(subject="SBJ1", condition="EXO", trial="A")
imu = load_imu(subject="SBJ1", condition="EXO", trial="A")

# Apply sliding window to a single session
X_emg, y = sliding_windows(emg, L=0.1, S=0.05, fs=EMG_FS, feature_cols=EMG_MUSCLES)
# X_emg shape: (N_windows, 192, 6)

# Build a custom dataset for specific subjects/conditions
X, y = make_dataset(subjects=["SBJ1", "SBJ2"], conditions=["EXO", "NoEXO"], L=0.1, S=0.05)

# Task 1 – General Purpose (3 train subjects / 2 test subjects)
X_train, y_train, X_test, y_test, train_sbj, test_sbj = task1_split(L=0.1, S=0.05)

# Task 1 with explicit subject selection
X_train, y_train, X_test, y_test, _, _ = task1_split(
    train_subjects=["SBJ1", "SBJ2", "SBJ4"], L=0.1, S=0.05
)

# Task 2 – Exo Challenge (NoEXO → train, EXO → test)
X_train, y_train, X_test, y_test = task2_split(L=0.1, S=0.05)
```

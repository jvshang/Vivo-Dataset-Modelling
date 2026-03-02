# VivoProject

Analysis of EMG and IMU data collected during exoskeleton-assisted walking trials.

## Data

**The `data/` folder is not included in this repository** (excluded via `.gitignore` due to file size).

The raw data files are stored locally at Isambard: /lus/lfs1aip2/projects/b5bb/public

```
data/
├── EMG/        # Electromyography recordings
└── IMU/        # Inertial Measurement Unit recordings
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
| Stage | `final` / `trimmed` | Processing stage |
| Modality | `EMG` / `IMU` | Sensor type |

### EMG channels

Rectus Femoris, Vastus Medialis, Vastus Lateralis, Bicep Femoris, Tibialis, Gastrocnemius Medialis

### IMU channels

R Thigh Angle, R Shank Angle, L Thigh Angle, L Shank Angle, Hip Angle

## Project structure

```
VivoProject_local/
├── data/               # Local only — not in git
│   ├── EMG/
│   └── IMU/
└── src/
    ├── dataloader.py   # Data loading utilities
    └── main.py         # Example usage
```

## Usage

```python
from src.dataloader import list_sessions, load_emg, load_imu

# List all available sessions
sessions = list_sessions(modality="EMG")

# Load a single session
emg = load_emg(subject="SBJ1", condition="EXO", trial="A")
imu = load_imu(subject="SBJ1", condition="EXO", trial="A")

# Load all NoEXO sessions for a subject
emg_all = load_emg(subject="SBJ2", condition="NoEXO")
```

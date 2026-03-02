import os
import re
from typing import List, Optional
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

EMG_MUSCLES = [
    "Rectus Femoris", "Vastus Medialis", "Vastus Lateralis",
    "Bicep Femoris", "Tibialis", "Gastrocnemius Medialis",
]
IMU_ANGLES = ["R Thigh Angle", "R Shank Angle", "L Thigh Angle", "L Shank Angle", "Hip Angle"]

# Matches e.g. SBJ1_EXO_A, SBJ3_NoExo_G (case-insensitive condition)
_FILENAME_RE = re.compile(
    r"(SBJ\d+)_((?:No)?[Ee][Xx][Oo])_([A-G])_final_(EMG|IMU)\.csv",
    re.IGNORECASE,
)


def _parse_filename(fname: str) -> Optional[dict]:
    m = _FILENAME_RE.match(fname)
    if not m:
        return None
    subject, condition, trial, modality = m.groups()
    condition = "EXO" if condition.upper() == "EXO" else "NoEXO"
    return {"subject": subject, "condition": condition, "trial": trial, "modality": modality}


def list_sessions(subject=None, condition=None, modality="EMG") -> List[dict]:
    """Return metadata dicts for all matching sessions."""
    folder = DATA_DIR / modality
    sessions = []
    for fname in sorted(os.listdir(folder)):
        info = _parse_filename(fname)
        if info is None or info["modality"] != modality:
            continue
        if subject and info["subject"] != subject:
            continue
        if condition and info["condition"] != condition:
            continue
        info["path"] = folder / fname
        sessions.append(info)
    return sessions


def load_emg(subject=None, condition=None, trial=None) -> pd.DataFrame:
    """Load one or more final_EMG files into a single DataFrame."""
    sessions = list_sessions(subject=subject, condition=condition, modality="EMG")
    if trial:
        sessions = [s for s in sessions if s["trial"] == trial]

    frames = []
    for s in sessions:
        df = pd.read_csv(s["path"])
        df["subject"] = s["subject"]
        df["condition"] = s["condition"]
        df["trial"] = s["trial"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No EMG files found for subject={subject}, condition={condition}, trial={trial}")
    return pd.concat(frames, ignore_index=True)


def load_imu(subject=None, condition=None, trial=None) -> pd.DataFrame:
    """Load one or more final_IMU files into a single DataFrame."""
    sessions = list_sessions(subject=subject, condition=condition, modality="IMU")
    if trial:
        sessions = [s for s in sessions if s["trial"] == trial]

    frames = []
    for s in sessions:
        df = pd.read_csv(s["path"])
        # Consolidate the 5 separate time columns into a single Time column
        df["Time"] = df["rtTime"]
        drop_cols = [c for c in df.columns if c.endswith("Time") and c != "Time"]
        df = df.drop(columns=drop_cols)
        df["subject"] = s["subject"]
        df["condition"] = s["condition"]
        df["trial"] = s["trial"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No IMU files found for subject={subject}, condition={condition}, trial={trial}")
    return pd.concat(frames, ignore_index=True)

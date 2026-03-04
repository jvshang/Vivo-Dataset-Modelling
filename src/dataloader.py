import os
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Sampling frequencies ─────────────────────────────────────────────────────
# EMG:  ~1/0.000519 s ≈ 1926 Hz
# IMU:  1/0.01 s = 100 Hz
EMG_FS = 1926
IMU_FS = 100

# ── Feature columns ──────────────────────────────────────────────────────────
EMG_MUSCLES = [
    "Rectus Femoris", "Vastus Medialis", "Vastus Lateralis",
    "Bicep Femoris", "Tibialis", "Gastrocnemius Medialis",
]
IMU_ANGLES = [
    "R Thigh Angle", "R Shank Angle", "L Thigh Angle", "L Shank Angle", "Hip Angle",
]
VALID_ACTIVITIES = ["Sat", "Stood", "Standing up", "Sitting down"]

# ── Subject groupings ────────────────────────────────────────────────────────
# SBJ3 was too tall and did NOT wear the exoskeleton → no EXO data collected.
ALL_SUBJECTS   = ["SBJ1", "SBJ2", "SBJ3", "SBJ4", "SBJ5", "SBJ6"]
EXO_SUBJECTS   = ["SBJ1", "SBJ2", "SBJ4", "SBJ5", "SBJ6"]  # SBJ3 excluded from EXO
TASK1_SUBJECTS = EXO_SUBJECTS   # Task 1 uses only these 5 (3 train + 2 test)

# ── Filename parser ──────────────────────────────────────────────────────────
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


# ── Session listing ──────────────────────────────────────────────────────────

def list_sessions(subject=None, condition=None, modality="EMG") -> List[dict]:
    """Return metadata dicts for all matching *final* sessions."""
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


# ── Raw loaders ──────────────────────────────────────────────────────────────

def load_emg(subject=None, condition=None, trial=None) -> pd.DataFrame:
    """Load one or more final_EMG files into a single DataFrame."""
    sessions = list_sessions(subject=subject, condition=condition, modality="EMG")
    if trial:
        sessions = [s for s in sessions if s["trial"] == trial]

    frames = []
    for s in sessions:
        df = pd.read_csv(s["path"])
        df["subject"]   = s["subject"]
        df["condition"] = s["condition"]
        df["trial"]     = s["trial"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No EMG files found for subject={subject}, condition={condition}, trial={trial}"
        )
    return pd.concat(frames, ignore_index=True)


def load_imu(subject=None, condition=None, trial=None) -> pd.DataFrame:
    """Load one or more final_IMU files into a single DataFrame."""
    sessions = list_sessions(subject=subject, condition=condition, modality="IMU")
    if trial:
        sessions = [s for s in sessions if s["trial"] == trial]

    frames = []
    for s in sessions:
        df = pd.read_csv(s["path"])
        # Five separate time columns → single 'Time' column (all identical within a row)
        df["Time"] = df["rtTime"]
        drop_cols = [c for c in df.columns if c.endswith("Time") and c != "Time"]
        df = df.drop(columns=drop_cols)
        df["subject"]   = s["subject"]
        df["condition"] = s["condition"]
        df["trial"]     = s["trial"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No IMU files found for subject={subject}, condition={condition}, trial={trial}"
        )
    return pd.concat(frames, ignore_index=True)


# ── Sliding window ────────────────────────────────────────────────────────────

def sliding_windows(
    df: pd.DataFrame,
    L: float,
    S: float,
    fs: float,
    feature_cols: List[str],
    label_col: str = "Activity",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a sliding window to a single-session DataFrame.

    Parameters
    ----------
    df           : DataFrame for one continuous recording segment.
    L            : Window length in seconds.
    S            : Stride (step) in seconds. S < L gives overlapping windows.
    fs           : Sampling frequency of this modality (EMG_FS or IMU_FS).
    feature_cols : Column names to use as features.
    label_col    : Column containing activity labels.

    Returns
    -------
    X : float32 array of shape (N_windows, window_size, n_features)
    y : string array  of shape (N_windows,)
        Windows where the majority label is not in VALID_ACTIVITIES are dropped.
    """
    window_size = int(L * fs)
    stride      = int(S * fs)
    if window_size < 1 or stride < 1:
        raise ValueError(
            f"L={L}s, S={S}s → window_size={window_size}, stride={stride}: too small for fs={fs} Hz."
        )

    data   = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df[label_col].to_numpy()

    X_list, y_list = [], []
    for start in range(0, len(data) - window_size + 1, stride):
        end            = start + window_size
        majority_label = pd.Series(labels[start:end]).mode().iloc[0]
        if majority_label not in VALID_ACTIVITIES:
            continue  # discard Unknown-majority windows
        X_list.append(data[start:end])
        y_list.append(majority_label)

    if not X_list:
        return (
            np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            np.array([], dtype=str),
        )
    return np.stack(X_list).astype(np.float32), np.array(y_list)


# ── Per-session windowing (EMG + IMU combined) ────────────────────────────────

def _windows_for_session(
    subject: str,
    condition: str,
    trial: str,
    L: float,
    S: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load one EMG + IMU session, apply sliding windows to each modality
    independently at its native sampling rate, then concatenate features
    along the feature axis.

    EMG window  : (window_size_emg, 6)  →  flattened → window_size_emg * 6 features
    IMU window  : (window_size_imu, 5)  →  flattened → window_size_imu * 5 features
    Combined X  : (N_windows, window_size_emg*6 + window_size_imu*5)

    NOTE: Temporal alignment strategy between EMG (~1926 Hz) and IMU (100 Hz)
    is TBD. For now both modalities are windowed at their own rates and trimmed
    to the shorter window count. Replace this function to change the strategy.

    Returns (X, y) or (None, None) if no windows could be produced.
    """
    try:
        emg = load_emg(subject=subject, condition=condition, trial=trial)
        imu = load_imu(subject=subject, condition=condition, trial=trial)
    except FileNotFoundError:
        return None, None

    X_emg, y_emg = sliding_windows(emg, L, S, EMG_FS, EMG_MUSCLES)
    X_imu, _ = sliding_windows(imu, L, S, IMU_FS, IMU_ANGLES)

    n = min(len(X_emg), len(X_imu))
    if n == 0:
        return None, None

    # Flatten each window then concatenate modalities along feature axis
    X = np.concatenate(
        [X_emg[:n].reshape(n, -1), X_imu[:n].reshape(n, -1)],
        axis=1,
    )
    return X, y_emg[:n]   # labels from EMG (higher temporal resolution)


# ── Dataset builder ───────────────────────────────────────────────────────────

def make_dataset(
    subjects: List[str],
    conditions: Union[str, List[str]],
    L: float = 0.1,
    S: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a windowed, flattened dataset for the given subjects and conditions.

    Parameters
    ----------
    subjects   : e.g. ["SBJ1", "SBJ2", "SBJ4"]
    conditions : "EXO", "NoEXO", or ["EXO", "NoEXO"]
    L          : Window length in seconds (default 0.1 s).
    S          : Stride in seconds (default 0.05 s → 50% overlap).

    Returns
    -------
    X : float32 array (N_total_windows, n_features)
    y : string  array (N_total_windows,)
    """
    if isinstance(conditions, str):
        conditions = [conditions]

    all_X, all_y = [], []
    for subject in subjects:
        for condition in conditions:
            sessions = list_sessions(subject=subject, condition=condition, modality="EMG")
            for s in sessions:
                X, y = _windows_for_session(subject, condition, s["trial"], L, S)
                if X is not None and len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)

    if not all_X:
        return np.empty((0,), dtype=np.float32), np.array([], dtype=str)
    return np.vstack(all_X).astype(np.float32), np.concatenate(all_y)


# ── Benchmark splits ──────────────────────────────────────────────────────────

def task1_split(
    train_subjects: Optional[List[str]] = None,
    L: float = 0.1,
    S: float = 0.05,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Task 1 – General Purpose Training.

    Pool: TASK1_SUBJECTS = {SBJ1, SBJ2, SBJ4, SBJ5, SBJ6}  (SBJ3 excluded).
    Train: 3 subjects (EXO + NoEXO)
    Test : 2 subjects (EXO + NoEXO)

    Parameters
    ----------
    train_subjects : Explicit list of 3 subjects. If None, randomly picks 3.
    random_seed    : Seed for reproducible random selection.

    Returns
    -------
    X_train, y_train, X_test, y_test, train_subjects, test_subjects
    """
    rng = np.random.default_rng(random_seed)
    if train_subjects is None:
        train_subjects = list(rng.choice(TASK1_SUBJECTS, size=3, replace=False))
    test_subjects = [s for s in TASK1_SUBJECTS if s not in train_subjects]

    print(f"[Task 1] Train subjects : {train_subjects}")
    print(f"[Task 1] Test  subjects : {test_subjects}")

    X_train, y_train = make_dataset(train_subjects, ["EXO", "NoEXO"], L, S)
    X_test,  y_test  = make_dataset(test_subjects,  ["EXO", "NoEXO"], L, S)

    print(f"[Task 1] X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, train_subjects, test_subjects


def task2_split(
    L: float = 0.1,
    S: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Task 2 – Exo Challenge / Transference.

    Train: ALL_SUBJECTS (SBJ1–6)  NoEXO data only.
    Test : EXO_SUBJECTS (SBJ1,2,4,5,6)  EXO data only.
           SBJ3 is excluded from test because no EXO data was collected.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    print(f"[Task 2] Train: {ALL_SUBJECTS}  (NoEXO only)")
    print(f"[Task 2] Test : {EXO_SUBJECTS}  (EXO only)")

    X_train, y_train = make_dataset(ALL_SUBJECTS, "NoEXO", L, S)
    X_test,  y_test  = make_dataset(EXO_SUBJECTS, "EXO",   L, S)

    print(f"[Task 2] X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

import os
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = "/lus/lfs1aip2/projects/b5bb/public/final_data"

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

# ── Task 1 fixed benchmark splits ────────────────────────────────────────────
# SBJ1 is the least erratic; SBJ5 and SBJ6 are the most erratic.
# Fixed test sets ensure comparability across experiments.
TASK1_SPLITS = {
    # SBJ3 is always added to train (NoEXO only) in addition to the 3 listed subjects.
    # Hardest benchmark: both test subjects have fast/erratic transitions
    "hard":  {"test": ["SBJ5", "SBJ6"], "train": ["SBJ1", "SBJ2", "SBJ4"]},
    # Mixed benchmark: one erratic + one easy subject in test set
    "mixed": {"test": ["SBJ1", "SBJ5"], "train": ["SBJ2", "SBJ4", "SBJ6"]},
    # Easy benchmark: least erratic subject in test set (sanity check)
    "easy":  {"test": ["SBJ1", "SBJ2"], "train": ["SBJ4", "SBJ5", "SBJ6"]},
}

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
    folder = f"{DATA_DIR}/{modality}"
    sessions = []
    for fname in sorted(os.listdir(folder)):
        info = _parse_filename(fname)
        if info is None or info["modality"] != modality:
            continue
        if subject and info["subject"] != subject:
            continue
        if condition and info["condition"] != condition:
            continue
        info["path"] = f"{folder}/{fname}"
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
    """
    Load one or more final_IMU files into a single DataFrame.

    Missing IMU values (sensor dropouts) are zero-filled, guaranteeing a
    fixed-length feature vector across all sessions.

    Returns
    -------
    df : Cleaned DataFrame. All IMU_ANGLES columns are always present with no NaNs.
    """
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
    df = pd.concat(frames, ignore_index=True)

    df[IMU_ANGLES] = df[IMU_ANGLES].fillna(0.0)

    return df


# ── Sliding window ────────────────────────────────────────────────────────────

def sliding_windows(
    df: pd.DataFrame,
    L: float,
    S: float,
    fs: float,
    feature_cols: List[str],
    label_col: str = "Activity",
    unknown_handling: str = "drop",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a sliding window to a single-session DataFrame.

    Parameters
    ----------
    df               : DataFrame for one continuous recording segment.
    L                : Window length in seconds.
    S                : Stride (step) in seconds. S < L gives overlapping windows.
    fs               : Sampling frequency of this modality (EMG_FS or IMU_FS).
    feature_cols     : Column names to use as features.
    label_col        : Column containing activity labels.
    unknown_handling : How to treat "Unknown" labels.
                       "drop"     – discard any window whose majority label is not
                                    in VALID_ACTIVITIES (original behavior).
                       "backfill" – before windowing, replace each "Unknown" sample
                                    with the previous known valid label (i.e. the
                                    state the person is coming *from*).  Any leading
                                    unknowns with no prior valid label are still
                                    dropped.

    Returns
    -------
    X : float32 array of shape (N_windows, window_size, n_features)
    y : string array  of shape (N_windows,)
    """
    window_size = int(L * fs)
    stride      = int(S * fs)
    if window_size < 1 or stride < 1:
        raise ValueError(
            f"L={L}s, S={S}s → window_size={window_size}, stride={stride}: too small for fs={fs} Hz."
        )

    data   = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df[label_col].to_numpy().copy().astype(object)

    if unknown_handling == "backfill":
        # Replace Unknown with NaN, then backfill with the next valid label.
        # Samples with no following valid label (trailing unknowns) remain NaN
        # and will be treated as invalid during windowing.
        label_series = pd.Series(labels)
        label_series[~label_series.isin(VALID_ACTIVITIES)] = np.nan
        label_series = label_series.bfill()
        labels = label_series.to_numpy()
    elif unknown_handling != "drop":
        raise ValueError(f"unknown_handling must be 'drop' or 'backfill', got {unknown_handling!r}")

    X_list, y_list = [], []
    # Need enough data for the feature window AND the next label window
    for start in range(0, len(data) - 2 * window_size + 1, stride):
        end            = start + window_size
        label_start    = end
        label_end      = end + window_size
        
        window_labels  = pd.Series(labels[label_start:label_end])
        valid_only     = window_labels[window_labels.isin(VALID_ACTIVITIES)]
        if valid_only.empty:
            continue  # no valid label in this next window → discard
        majority_label = valid_only.mode().iloc[0]
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
    unknown_handling: str = "drop",
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

    X_emg, y_emg = sliding_windows(emg, L, S, EMG_FS, EMG_MUSCLES, unknown_handling=unknown_handling)
    X_imu, _ = sliding_windows(imu, L, S, IMU_FS, IMU_ANGLES, unknown_handling=unknown_handling)

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
    unknown_handling: str = "drop",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a windowed, flattened dataset for the given subjects and conditions.

    Parameters
    ----------
    subjects          : e.g. ["SBJ1", "SBJ2", "SBJ4"]
    conditions        : "EXO", "NoEXO", or ["EXO", "NoEXO"]
    L                 : Window length in seconds (default 0.1 s).
    S                 : Stride in seconds (default 0.05 s → 50% overlap).
    unknown_handling  : "drop" (discard Unknown windows) or "backfill"
                        (replace Unknown with the next known label).

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
                X, y = _windows_for_session(subject, condition, s["trial"], L, S, unknown_handling)
                if X is not None and len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)

    if not all_X:
        return np.empty((0,), dtype=np.float32), np.array([], dtype=str)
    return np.vstack(all_X).astype(np.float32), np.concatenate(all_y)


# ── Benchmark splits ──────────────────────────────────────────────────────────

def task1_split(
    split: Optional[str] = "hard",
    train_subjects: Optional[List[str]] = None,
    L: float = 0.1,
    S: float = 0.05,
    random_seed: int = 42,
    unknown_handling: str = "drop",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Task 1 – Generalization across participants.

    Pool: TASK1_SUBJECTS = {SBJ1, SBJ2, SBJ4, SBJ5, SBJ6}  (SBJ3 excluded from test).
    Train: 3 subjects (EXO + NoEXO) + SBJ3 (NoEXO only, always included).
    Test : 2 subjects (EXO + NoEXO).

    Parameters
    ----------
    split            : Named benchmark split from TASK1_SPLITS.
                       "hard"  – test on SBJ5 + SBJ6 (most erratic transitions).
                       "mixed" – test on SBJ1 + SBJ5.
                       "easy"  – test on SBJ1 + SBJ2 (least erratic, sanity check).
                       None    – ignore named splits; use train_subjects or random.
    train_subjects   : Explicit list of 3 train subjects. Only used when split=None.
                       SBJ3 is always added on top of this list (NoEXO only).
                       If split=None and train_subjects=None, randomly picks 3.
    random_seed      : Seed for random selection (only used when split=None).
    unknown_handling : "drop" or "backfill" (see make_dataset).

    Returns
    -------
    X_train, y_train, X_test, y_test, train_subjects, test_subjects
        train_subjects includes "SBJ3" to reflect the full training pool.
    """
    if split is not None:
        if split not in TASK1_SPLITS:
            raise ValueError(f"split must be one of {list(TASK1_SPLITS)}, got {split!r}")
        train_subjects = TASK1_SPLITS[split]["train"]
        test_subjects  = TASK1_SPLITS[split]["test"]
    else:
        rng = np.random.default_rng(random_seed)
        if train_subjects is None:
            train_subjects = list(rng.choice(TASK1_SUBJECTS, size=3, replace=False))
        test_subjects = [s for s in TASK1_SUBJECTS if s not in train_subjects]

    print(f"[Task 1] Split          : {split or 'custom'}")
    print(f"[Task 1] Train subjects : {train_subjects} + SBJ3 (NoEXO only)")
    print(f"[Task 1] Test  subjects : {test_subjects}")

    # Load 3 main train subjects (both conditions) + SBJ3 (NoEXO only)
    X_main,  y_main  = make_dataset(train_subjects, ["EXO", "NoEXO"], L, S, unknown_handling)
    X_sbj3,  y_sbj3  = make_dataset(["SBJ3"],       "NoEXO",          L, S, unknown_handling)
    X_train = np.vstack([X_main, X_sbj3]) if len(X_sbj3) > 0 else X_main
    y_train = np.concatenate([y_main, y_sbj3]) if len(y_sbj3) > 0 else y_main

    X_test,  y_test  = make_dataset(test_subjects,  ["EXO", "NoEXO"], L, S, unknown_handling)

    print(f"[Task 1] X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, train_subjects + ["SBJ3"], test_subjects


def task2_split(
    L: float = 0.1,
    S: float = 0.05,
    unknown_handling: str = "drop",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Task 2 – Exo Challenge / Transference.

    Train: ALL_SUBJECTS (SBJ1–6)  NoEXO data only.
    Test : EXO_SUBJECTS (SBJ1,2,4,5,6)  EXO data only.
           SBJ3 is excluded from test because no EXO data was collected.

    Parameters
    ----------
    unknown_handling : "drop" or "backfill" (see make_dataset).

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    print(f"[Task 2] Train: {ALL_SUBJECTS}  (NoEXO only)")
    print(f"[Task 2] Test : {EXO_SUBJECTS}  (EXO only)")

    X_train, y_train = make_dataset(ALL_SUBJECTS, "NoEXO", L, S, unknown_handling)
    X_test,  y_test  = make_dataset(EXO_SUBJECTS, "EXO",   L, S, unknown_handling)

    print(f"[Task 2] X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

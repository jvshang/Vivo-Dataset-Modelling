import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataloader import (
    list_sessions,
    load_emg,
    load_imu,
    sliding_windows,
    task1_split,
    task2_split,
    EMG_FS, IMU_FS,
    EMG_MUSCLES, IMU_ANGLES,
)

# ── 1. List all available sessions ───────────────────────────────────────────
print("=== Available EMG sessions ===")
sessions = list_sessions(modality="EMG")
for s in sessions:
    print(f"  {s['subject']}  {s['condition']}  trial={s['trial']}")

# ── 2. Sliding window sanity check (single session) ──────────────────────────
print("\n=== Sliding window: SBJ1 / EXO / trial A ===")
emg = load_emg(subject="SBJ1", condition="EXO", trial="A")
imu = load_imu(subject="SBJ1", condition="EXO", trial="A")

L, S = 0.1, 0.05   # 0.1 s window, 50% overlap

X_emg, y_emg = sliding_windows(emg, L, S, EMG_FS, EMG_MUSCLES)
X_imu, y_imu = sliding_windows(imu, L, S, IMU_FS, IMU_ANGLES)

print(f"EMG windows : {X_emg.shape}   (N, {int(L*EMG_FS)} samples, 6 muscles)")
print(f"IMU windows : {X_imu.shape}   (N, {int(L*IMU_FS)} samples, 5 angles)")
print(f"EMG labels  : {dict(zip(*[list(v) for v in __import__('numpy').unique(y_emg, return_counts=True)]))}")

# ── 3. Task 1 split ───────────────────────────────────────────────────────────
print("\n=== Task 1: General Purpose Training ===")
X_tr1, y_tr1, X_te1, y_te1, train_sbj, test_sbj = task1_split(L=L, S=S)
print(f"Feature dim : {X_tr1.shape[1]}  "
      f"(EMG: {int(L*EMG_FS)}×6={int(L*EMG_FS)*6}  +  IMU: {int(L*IMU_FS)}×5={int(L*IMU_FS)*5})")

import numpy as np
for split_name, y in [("Train", y_tr1), ("Test", y_te1)]:
    labels, counts = np.unique(y, return_counts=True)
    print(f"  {split_name}: {dict(zip(labels, counts))}")

# ── 4. Task 2 split ───────────────────────────────────────────────────────────
print("\n=== Task 2: Exo Challenge ===")
X_tr2, y_tr2, X_te2, y_te2 = task2_split(L=L, S=S)
for split_name, y in [("Train (NoEXO)", y_tr2), ("Test  (EXO)  ", y_te2)]:
    labels, counts = np.unique(y, return_counts=True)
    print(f"  {split_name}: {dict(zip(labels, counts))}")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataloader import list_sessions, load_emg, load_imu

# ── 1. List all available sessions ──────────────────────────────────────────
print("=== Available EMG sessions ===")
sessions = list_sessions(modality="EMG")
for s in sessions:
    print(f"  {s['subject']}  {s['condition']}  trial={s['trial']}")

# ── 2. Load a single EMG session ─────────────────────────────────────────────
print("\n=== Load SBJ1 / EXO / trial A (EMG) ===")
emg = load_emg(subject="SBJ1", condition="EXO", trial="A")
print(emg.shape)
print(emg.dtypes)
print(emg["Activity"].value_counts())
print(emg.head(3))

# ── 3. Load a single IMU session ─────────────────────────────────────────────
print("\n=== Load SBJ1 / EXO / trial A (IMU) ===")
imu = load_imu(subject="SBJ1", condition="EXO", trial="A")
print(imu.shape)
print(imu.head(3))

# ── 4. Load all NoEXO sessions for SBJ2 ─────────────────────────────────────
print("\n=== Load all SBJ2 NoEXO EMG sessions ===")
emg_all = load_emg(subject="SBJ2", condition="NoEXO")
print(emg_all.shape)
print(emg_all.groupby("trial")["Activity"].value_counts())

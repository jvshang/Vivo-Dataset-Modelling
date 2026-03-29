"""
train.py – Experiment runner with Weights & Biases tracking.

Tracked metrics
---------------
accuracy       : Classification accuracy on the held-out test split.
lead_time_s    : Minimum data window required before a prediction (= L seconds).
                 Longer windows → higher potential accuracy but greater delay.
latency_ms     : Mean single-sample inference time in milliseconds.
model_size_mb  : Serialised model size in megabytes (target: < 5 MB).

Usage
-----
# Run all models and hyperparameter combinations defined in config.yaml
python src/train.py

# Point to a different config file
python src/train.py --config experiments/task1.yaml
"""

import argparse
import io
import pickle
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import wandb
import yaml
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from dataloader import task1_split, task2_split  # noqa: E402
from models import build_model, available_models  # noqa: E402


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_grid(cfg: dict) -> list[dict]:
    """
    Expand per-model hyperparameter grids into a flat list of run configs.

    Each entry in cfg["models"] is a dict of param → value | [values].
    Returns one config dict per combination, with "model" set to the model key.
    """
    base = {k: v for k, v in cfg.items() if k != "models"}
    runs = []
    for model_key, param_grid in cfg["models"].items():
        keys   = list(param_grid.keys())
        values = [v if isinstance(v, list) else [v] for v in param_grid.values()]
        for combo in product(*values):
            runs.append({**base, "model": model_key, **dict(zip(keys, combo))})
    return runs


# ── Metric helpers ────────────────────────────────────────────────────────────

def model_size_mb(model) -> float:
    """Serialise model with pickle and return size in MB."""
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return buf.tell() / (1024 ** 2)


def inference_latency_ms(model, X: np.ndarray, n_repeats: int = 200) -> float:
    """
    Mean single-sample inference latency in milliseconds.
    Draws random single samples to mimic real-time prediction.
    """
    # Warm-up pass (avoids JIT / first-call overhead)
    model.predict(X[:min(10, len(X))])

    rng = np.random.default_rng(0)
    durations = []
    for _ in range(n_repeats):
        idx = int(rng.integers(0, len(X)))
        sample = X[idx : idx + 1]
        t0 = time.perf_counter()
        model.predict(sample)
        durations.append(time.perf_counter() - t0)

    return float(np.mean(durations)) * 1000  # convert to ms


# ── Core experiment ───────────────────────────────────────────────────────────

def run(config: dict | None = None):
    """
    Run one experiment and log all metrics to wandb.

    Parameters
    ----------
    config : Flat dict of hyperparameters for a single run, produced by expand_grid.
    """
    with wandb.init(config=config) as run:
        cfg = wandb.config

        L = cfg.window_length
        S = cfg.stride

        # ── Lead time ─────────────────────────────────────────────────────────
        # The model cannot produce a prediction until L seconds of sensor data
        # have been collected, so lead_time == window length.
        lead_time_s = L

        # ── Load benchmark split ──────────────────────────────────────────────
        print(f"\n[wandb run: {run.name}]  task={cfg.task}  L={L}s  model={cfg.model}")
        if cfg.task == 1:
            X_train, y_train, X_test, y_test, train_sbj, test_sbj = task1_split(L=L, S=S)
            wandb.config.update({"train_subjects": train_sbj, "test_subjects": test_sbj},
                                allow_val_change=True)
        elif cfg.task == 2:
            X_train, y_train, X_test, y_test = task2_split(L=L, S=S)
        else:
            raise ValueError(f"task must be 1 or 2, got {cfg.task}")

        # ── Label encoding ────────────────────────────────────────────────────
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)

        # ── Train ─────────────────────────────────────────────────────────────
        model = build_model(cfg)
        t_train_start = time.perf_counter()
        model.fit(X_train, y_train_enc)
        train_time_s = time.perf_counter() - t_train_start

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test_enc, y_pred)

        # ── Latency ───────────────────────────────────────────────────────────
        latency_ms = inference_latency_ms(model, X_test)

        # ── Model size ────────────────────────────────────────────────────────
        size_mb        = model_size_mb(model)
        size_within_5mb = size_mb < 5.0

        # ── Log to wandb ──────────────────────────────────────────────────────
        wandb.log({
            # Primary benchmark metrics
            "accuracy":        accuracy,
            "lead_time_s":     lead_time_s,
            "latency_ms":      latency_ms,
            "model_size_mb":   size_mb,
            # Constraint flag
            "size_ok":         size_within_5mb,
            # Dataset info
            "n_train":         len(X_train),
            "n_test":          len(X_test),
            "n_features":      X_train.shape[1],
            # Training throughput
            "train_time_s":    train_time_s,
        })

        # ── Console summary ───────────────────────────────────────────────────
        size_flag = "✓" if size_within_5mb else "✗ EXCEEDS 5 MB LIMIT"
        print(f"  Accuracy   : {accuracy:.4f}")
        print(f"  Lead time  : {lead_time_s:.3f} s  (window = {L} s)")
        print(f"  Latency    : {latency_ms:.3f} ms / sample")
        print(f"  Model size : {size_mb:.3f} MB  {size_flag}")
        print(f"  Train time : {train_time_s:.1f} s")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train + wandb logging for VIVO activity recognition")
    p.add_argument("--config", default="../configs/task1.yaml", help="Path to YAML experiment config")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_config(args.config)
    runs   = expand_grid(cfg)
    print(f"[train] Config : {args.config}")
    print(f"[train] Total runs : {len(runs)}")
    for run_cfg in runs:
        run(run_cfg)
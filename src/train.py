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
# Single run with default config
python src/train.py

# Override any config value from the command line
python src/train.py --task 2 --window_length 0.2 --model rf

# wandb sweep (define sweep.yaml first, then):
# wandb sweep sweep.yaml
# wandb agent <sweep-id>
"""

import argparse
import io
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import wandb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from dataloader import task1_split, task2_split  # noqa: E402


# ── Default experiment configuration ─────────────────────────────────────────

DEFAULT_CONFIG = {
    "task": 1,            # 1 = general-purpose, 2 = exo-challenge
    "window_length": 0.1, # L in seconds — directly determines lead time
    "stride": 0.05,       # S in seconds (50% overlap)
    "model": "rf",        # "rf" | "gb" | "lr"
    # RandomForest / GradientBoosting hyper-params
    "n_estimators": 50,
    "max_depth": 10,
    # Logistic Regression hyper-params
    "lr_C": 1.0,
    "lr_max_iter": 500,
}


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg):
    if cfg.model == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=42,
            n_jobs=-1,
        )
    if cfg.model == "gb":
        return GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=42,
        )
    if cfg.model == "lr":
        return LogisticRegression(
            C=cfg.lr_C,
            max_iter=cfg.lr_max_iter,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model type: {cfg.model!r}. Choose rf | gb | lr.")


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
    config : dict of hyperparameters. If None, uses DEFAULT_CONFIG.
             When invoked by a wandb sweep agent the agent injects its own config.
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
    p.add_argument("--task",          type=int,   default=DEFAULT_CONFIG["task"])
    p.add_argument("--window_length", type=float, default=DEFAULT_CONFIG["window_length"])
    p.add_argument("--stride",        type=float, default=DEFAULT_CONFIG["stride"])
    p.add_argument("--model",         type=str,   default=DEFAULT_CONFIG["model"],
                   choices=["rf", "gb", "lr"])
    p.add_argument("--n_estimators",  type=int,   default=DEFAULT_CONFIG["n_estimators"])
    p.add_argument("--max_depth",     type=int,   default=DEFAULT_CONFIG["max_depth"])
    p.add_argument("--lr_C",          type=float, default=DEFAULT_CONFIG["lr_C"])
    p.add_argument("--lr_max_iter",   type=int,   default=DEFAULT_CONFIG["lr_max_iter"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(config=vars(args))

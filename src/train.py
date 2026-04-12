"""
train.py – Experiment runner with Weights & Biases tracking.

Tracked metrics
---------------
accuracy       : Classification accuracy on the held-out test split.
cv_best_score  : Best mean cross-validated accuracy from grid search.
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
import os
import pickle
import random
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import wandb
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from dataloader import task1_split, task2_split  # noqa: E402
from models import _CUML, build_model            # noqa: E402

# ── Reproducibility ───────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and the OS hash seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize_grid(param_grid: dict) -> dict:
    """Wrap any scalar grid values in a list so GridSearchCV and _torch_cv always receive lists."""
    return {k: v if isinstance(v, list) else [v] for k, v in param_grid.items()}


def iter_window_combos(cfg: dict):
    """Yield (window_length, stride) pairs from the base config."""
    lengths = cfg["window_length"] if isinstance(cfg["window_length"], list) else [cfg["window_length"]]
    strides = cfg["stride"]        if isinstance(cfg["stride"],        list) else [cfg["stride"]]
    yield from product(lengths, strides)


# ── GridSearchCV wrapper for sklearn / sktime models ─────────────────────────

class _GridSearchEstimator(BaseEstimator, ClassifierMixin):
    """
    Wraps a raw sklearn/sktime/cuML estimator for GridSearchCV, with an
    optional (N,F) → (N,1,F) reshape for sktime classifiers.

    get_params(deep=False) returns only the constructor args so that
    sklearn's clone() can reconstruct this wrapper correctly.
    get_params(deep=True)  additionally surfaces the inner clf's params
    so GridSearchCV can validate the param_grid keys.
    set_params() routes unknown keys directly to the inner clf.
    """

    def __init__(self, clf, needs_3d: bool = False):
        self.clf      = clf
        self.needs_3d = needs_3d

    def _maybe_3d(self, X: np.ndarray) -> np.ndarray:
        return X[:, np.newaxis, :] if self.needs_3d else X

    def fit(self, X, y):
        self.clf.fit(self._maybe_3d(X), y)
        return self

    def predict(self, X):
        return self.clf.predict(self._maybe_3d(X))

    def get_params(self, deep: bool = True) -> dict:
        out = {"clf": self.clf, "needs_3d": self.needs_3d}
        if deep and hasattr(self.clf, "get_params"):
            out.update(self.clf.get_params(deep=True))
        return out

    def set_params(self, **params):
        clf_params = {}
        for k, v in params.items():
            if k == "clf":
                self.clf = v
            elif k == "needs_3d":
                self.needs_3d = v
            else:
                clf_params[k] = v
        if clf_params:
            self.clf.set_params(**clf_params)
        return self


# ── Custom CV loop for PyTorch models ─────────────────────────────────────────

def _torch_cv(
    model_key: str,
    param_grid: dict,
    base_cfg: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    seed: int,
) -> tuple[dict, float]:
    """
    Manual stratified k-fold CV for PyTorch models that cannot use GridSearchCV.
    Iterates over the full cartesian product of param_grid and returns the
    best param combo and its mean CV accuracy.
    """
    keys   = list(param_grid.keys())
    values = [v if isinstance(v, list) else [v] for v in param_grid.values()]
    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    best_score, best_params = -1.0, {}
    for combo in product(*values):
        params  = dict(zip(keys, combo))
        run_cfg = {**base_cfg, "model": model_key, **params}
        fold_accs = []
        for train_idx, val_idx in skf.split(X, y):
            m = build_model(run_cfg)
            m.fit(X[train_idx], y[train_idx])
            fold_accs.append(accuracy_score(y[val_idx], m.predict(X[val_idx])))
        mean_acc = float(np.mean(fold_accs))
        print(f"    {params}  →  CV acc: {mean_acc:.4f}")
        if mean_acc > best_score:
            best_score, best_params = mean_acc, params

    return best_params, best_score


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

def run(model_key: str, param_grid: dict, base_cfg: dict, L: float, S: float):
    """
    Run grid search for one model + (L, S) combo and log all metrics to wandb.

    Parameters
    ----------
    model_key  : Registered model key (e.g. "rf").
    param_grid : Hyperparameter grid.
                 sklearn/sktime/xgb models → passed to GridSearchCV directly.
                 PyTorch models           → iterated manually in _torch_cv.
    base_cfg   : Shared experiment settings (task, seed, cv_folds, etc.).
    L          : Window length in seconds.
    S          : Stride in seconds.
    """
    run_cfg  = {**base_cfg, "model": model_key, "window_length": L, "stride": S}
    run_name = f"{model_key}_L{L}_S{S}"
    run_dir  = Path("outputs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with wandb.init(config=run_cfg, name=run_name, dir=run_dir) as wrun:
        cfg = wandb.config
        set_global_seed(cfg.seed)

        print(f"\n[wandb run: {wrun.name}]  task={cfg.task}  L={L}s  model={model_key}")

        # ── Load benchmark split ──────────────────────────────────────────────
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

        # ── Grid search — dispatch by model type ──────────────────────────────
        probe = build_model({"model": model_key, "seed": cfg.seed})
        param_grid = _normalize_grid(param_grid)

        t_train_start = time.perf_counter()

        if probe._is_torch:
            # PyTorch: manual stratified CV over the full param cartesian product
            best_params, cv_best_score = _torch_cv(
                model_key, param_grid, dict(cfg),
                X_train, y_train_enc, cfg.cv_folds, cfg.seed,
            )
            best_model = build_model({**dict(cfg), "model": model_key, **best_params})
            best_model.fit(X_train, y_train_enc)
            best_estimator = best_model   # BaseClassifier — has .predict()
        else:
            # sklearn / sktime / XGBoost: GridSearchCV on the inner estimator.
            # _GridSearchEstimator handles the 3-D reshape for sktime models.
            wrapper = _GridSearchEstimator(probe._clf, needs_3d=probe._needs_3d)
            gs = GridSearchCV(
                wrapper, param_grid,
                cv=cfg.cv_folds, scoring="accuracy",
                refit=True, n_jobs=1 if _CUML else -1,
            )
            gs.fit(X_train, y_train_enc)
            best_estimator = gs.best_estimator_   # _GridSearchEstimator — has .predict()
            best_params    = gs.best_params_
            cv_best_score  = gs.best_score_

        train_time_s = time.perf_counter() - t_train_start

        wandb.config.update({"best_params": best_params}, allow_val_change=True)
        print(f"  Best params : {best_params}  (CV accuracy: {cv_best_score:.4f})")

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred   = best_estimator.predict(X_test)
        accuracy = accuracy_score(y_test_enc, y_pred)

        # ── Latency & size ────────────────────────────────────────────────────
        latency_ms      = inference_latency_ms(best_estimator, X_test)
        size_mb         = model_size_mb(best_estimator)
        size_within_5mb = size_mb < 5.0

        # ── Log to wandb ──────────────────────────────────────────────────────
        wandb.log({
            # Best hyperparameters as individual columns
            **{f"best_{k}": v for k, v in best_params.items()},
            # Primary benchmark metrics
            "accuracy":        accuracy,
            "cv_best_score":   cv_best_score,
            "lead_time_s":     L,
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
        print(f"  Accuracy   : {accuracy:.4f}  (CV best: {cv_best_score:.4f})")
        print(f"  Lead time  : {L:.3f} s  (window = {L} s)")
        print(f"  Latency    : {latency_ms:.3f} ms / sample")
        print(f"  Model size : {size_mb:.3f} MB  {size_flag}")
        print(f"  Train time : {train_time_s:.1f} s")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train + wandb logging for VIVO activity recognition")
    p.add_argument("--config", default="../configs/task1.yaml", help="Path to YAML experiment config")
    return p.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    cfg      = load_config(args.config)
    base_cfg = {k: v for k, v in cfg.items() if k != "models"}
    combos   = list(iter_window_combos(cfg))
    n_runs   = len(combos) * len(cfg["models"])

    print(f"[train] Config       : {args.config}")
    print(f"[train] Models       : {list(cfg['models'].keys())}")
    print(f"[train] (L, S) pairs : {combos}")
    print(f"[train] Total runs   : {n_runs}  ({len(combos)} window combos × {len(cfg['models'])} models)")

    for L, S in combos:
        for model_key, model_cfg in cfg["models"].items():
            run(model_key, model_cfg["param_grid"], base_cfg, L, S)
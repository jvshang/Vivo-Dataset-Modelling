"""
train.py – Experiment runner with Weights & Biases tracking.

Tracked metrics
---------------
accuracy       : Classification accuracy on the held-out test split.
auc            : Macro-average one-vs-rest ROC AUC on the held-out test split.
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
import concurrent.futures
import multiprocessing as mp
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import wandb
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from dataloader import task1_split, task2_split  # noqa: E402
from models import _CUML, build_model            # noqa: E402

from utils_hyperparameters import _GridSearchEstimator, _torch_cv
from utils_plot import _log_confusion_matrix, _log_roc_curves, _log_imbalance_and_performance, compare_models

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

def get_cached_dataset(task: int, L: float, S: float, split: str = "hard", cache_dir: str = ".data_cache"):
    """Loads (X_train, y_train, X_test, y_test, ...) from disk, generating it if necessary."""
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if task == 1:
        cache_file = out_dir / f"task1_{split}_L{L}_S{S}.npz"
        if cache_file.exists():
            print(f" [Data] Loading cached dataset: {cache_file.name}")
            data = np.load(cache_file, allow_pickle=True)
            return (
                data["X_train"], data["y_train"],
                data["X_test"], data["y_test"],
                data["train_subjects"].tolist(),
                data["test_subjects"].tolist()
            )
        else:
            print(f" [Data] Cache miss. Generating Task 1 data for L={L}, S={S}...")
            X_tr, y_tr, X_te, y_te, tr_sbj, te_sbj = task1_split(split=split, L=L, S=S)
            np.savez_compressed(
                cache_file,
                X_train=X_tr, y_train=y_tr,
                X_test=X_te, y_test=y_te,
                train_subjects=np.array(tr_sbj),
                test_subjects=np.array(te_sbj)
            )
            return X_tr, y_tr, X_te, y_te, tr_sbj, te_sbj

    elif task == 2:
        cache_file = out_dir / f"task2_L{L}_S{S}.npz"
        if cache_file.exists():
            print(f" [Data] Loading cached dataset: {cache_file.name}")
            data = np.load(cache_file, allow_pickle=True)
            return data["X_train"], data["y_train"], data["X_test"], data["y_test"]
        else:
            print(f" [Data] Cache miss. Generating Task 2 data for L={L}, S={S}...")
            X_tr, y_tr, X_te, y_te = task2_split(L=L, S=S)
            np.savez_compressed(
                cache_file,
                X_train=X_tr, y_train=y_tr,
                X_test=X_te, y_test=y_te
            )
            return X_tr, y_tr, X_te, y_te
    else:
        raise ValueError(f"Unknown task {task}")

def run(model_key: str, param_grid: dict, base_cfg: dict, L: float, S: float):
    """
    Run grid search for one model + (L, S) combo and log all metrics to wandb.

    Returns a results dict for use in compare_models().
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
        split_name = cfg.get("split", "hard") # Default to hard if not specified
        
        if cfg.task == 1:
            X_train, y_train, X_test, y_test, train_sbj, test_sbj = get_cached_dataset(1, L, S, split=split_name)
            wandb.config.update({"train_subjects": train_sbj, "test_subjects": test_sbj},
                                allow_val_change=True)
        elif cfg.task == 2:
            X_train, y_train, X_test, y_test = get_cached_dataset(2, L, S)
        else:
            raise ValueError(f"task must be 1 or 2, got {cfg.task}")

        # ── Label encoding ────────────────────────────────────────────────────
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)
        class_names = le.classes_.tolist()

        # ── Undersampling ─────────────────────────────────────────────────────
        if cfg.get("undersample", False):
            print("  [Data] Applying random undersampling to training set...")
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=cfg.seed)
            X_train, y_train_enc = rus.fit_resample(X_train, y_train_enc)
            print(f"  [Data] Resampled training set shape: {X_train.shape}")

        # ── Grid search — dispatch by model type ──────────────────────────────
        probe = build_model(dict(cfg))
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
        y_proba  = best_estimator.predict_proba(X_test)
        accuracy = accuracy_score(y_test_enc, y_pred)

        # ── Per-run plots ──────────────────────────────────────────────────────
        undersample_flag = cfg.get("undersample", False)
        class_weights_flag = cfg.get("use_class_weights", False)
        figures_dir = Path("figures") / f"Task_{cfg.task}" / f"undersampling_{undersample_flag}_class_weights_{class_weights_flag}"
        model_figures_dir = figures_dir / run_name
        model_figures_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"Task{cfg.task}_US{undersample_flag}_CW{class_weights_flag}_{model_key}_L{L}_S{S}_"

        _log_confusion_matrix(y_test_enc, y_pred, class_names, model_figures_dir, prefix=prefix)
        auc = _log_roc_curves(y_test_enc, y_proba, class_names, model_figures_dir, prefix=prefix)
        _log_imbalance_and_performance(
            y_train_enc, y_test_enc, y_pred, class_names, model_figures_dir, prefix=prefix
        )

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
            "auc":             auc,
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
        print(f"  AUC        : {auc:.4f}")
        print(f"  Lead time  : {L:.3f} s  (window = {L} s)")
        print(f"  Latency    : {latency_ms:.3f} ms / sample")
        print(f"  Model size : {size_mb:.3f} MB  {size_flag}")
        print(f"  Train time : {train_time_s:.1f} s")

    return {
        "model":          model_key,
        "L":              L,               # Added L
        "S":              S,               # Added S
        "accuracy":       accuracy,
        "auc":            auc,
        "latency_ms":     latency_ms,
        "model_size_mb":  size_mb,
    }

# ── Parallel Execution Helpers ────────────────────────────────────────────────

def _init_gpu_worker(gpu_queue):
    """Assigns an isolated GPU to the worker process upon initialization."""
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def _run_gpu_task(kwargs):
    """Wrapper mapping dictionary arguments back into the run function."""
    return run(**kwargs)


# ── CLI entry-point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train + wandb logging for VIVO activity recognition")
    p.add_argument("--config", default="../configs/task1.yaml", help="Path to YAML experiment config")
    return p.parse_args()


if __name__ == "__main__":
    # Crucial for CUDA in multiprocessing contexts to prevent context deadlocks
    mp.set_start_method("spawn", force=True)

    args     = parse_args()
    cfg      = load_config(args.config)
    base_cfg = {k: v for k, v in cfg.items() if k != "models"}
    combos   = list(iter_window_combos(cfg))
    out_dir  = Path("outputs")
    n_runs   = len(combos) * len(cfg["models"])

    print(f"[train] Config       : {args.config}")
    print(f"[train] Models       : {list(cfg['models'].keys())}")
    print(f"[train] (L, S) pairs : {combos}")
    print(f"[train] Total runs   : {n_runs}  ({len(combos)} window combos × {len(cfg['models'])} models)")

    print("\n[Pre-computation] Checking dataset caches to prevent parallel file collisions...")
    task_id = cfg.get("task", 1)
    split_name = cfg.get("split", "hard")
    for L, S in combos:
        # Calling this sequentially guarantees the files are safely written to disk once.
        # Later, parallel workers will just read the generated .npz files instantly.
        get_cached_dataset(task_id, L, S, split_name)
    print("[Pre-computation] Dataset caching complete.\n")

    # Identify GPU vs CPU bound models dynamically
    gpu_models_set = {"xgb", "lstmcnn"}
    if _CUML:
        gpu_models_set.update({"rf", "lr"})

    gpu_tasks = []
    cpu_tasks = []

    for L, S in combos:
        for model_key, model_cfg in cfg["models"].items():
            task_kwargs = {
                "model_key": model_key,
                "param_grid": model_cfg["param_grid"],
                "base_cfg": base_cfg,
                "L": L,
                "S": S
            }
            if model_key in gpu_models_set:
                gpu_tasks.append(task_kwargs)
            else:
                cpu_tasks.append(task_kwargs)

    combo_results = []

    # 1. Parallelize GPU tasks across available hardware
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except ImportError:
        n_gpus = 0

    if n_gpus > 0 and gpu_tasks:
        print(f"\n[Parallel] Distributing {len(gpu_tasks)} GPU-bound tasks across {n_gpus} GPUs...")
        m = mp.Manager()
        gpu_q = m.Queue()
        for i in range(n_gpus):
            gpu_q.put(i)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_gpus,
            initializer=_init_gpu_worker,
            initargs=(gpu_q,)
        ) as exc:
            gpu_results = list(exc.map(_run_gpu_task, gpu_tasks))
        combo_results.extend(gpu_results)

    elif gpu_tasks:
        print("\n[Sequential] No GPUs detected. Running GPU-targeted tasks serially...")
        for task in gpu_tasks:
            combo_results.append(run(**task))

    # 2. Sequential CPU Execution
    # CPU models run serially so their internal n_jobs=-1 grid search can max out 
    # the processor without OS scheduler thrashing.
    if cpu_tasks:
        print(f"\n[Sequential] Executing {len(cpu_tasks)} CPU-bound tasks (maxing CPU via inner n_jobs=-1)...")
        for task in cpu_tasks:
            combo_results.append(run(**task))

    # 3. Regroup results and dispatch to plotting
    grouped_results = defaultdict(list)
    for res in combo_results:
        grouped_results[(res["L"], res["S"])].append(res)

    task_id = cfg.get("task", 1)
    undersample_flag = cfg.get("undersample", False)
    class_weights_flag = cfg.get("use_class_weights", False)
    figures_dir = Path("figures") / f"Task_{task_id}" / f"undersampling_{undersample_flag}_class_weights_{class_weights_flag}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for (L, S), results in grouped_results.items():
        prefix = f"Task{task_id}_US{undersample_flag}_CW{class_weights_flag}_L{L}_S{S}_"
        compare_models(results, L, S, figures_dir, wandb_dir=out_dir, prefix=prefix)
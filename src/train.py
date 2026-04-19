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
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import wandb
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

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

    def predict_proba(self, X):
        return self.clf.predict_proba(self._maybe_3d(X))

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

# ── Per-run plots ─────────────────────────────────────────────────────────────

def _log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    """Log a labelled confusion matrix to the active wandb run and save as PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=class_names,
    ).plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)


def _log_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> float:
    """
    Log per-class ROC curves and return macro-average AUC.
    Uses one-vs-rest binarisation for multiclass problems.
    """
    lb      = LabelBinarizer().fit(y_true)
    y_bin   = lb.transform(y_true)
    n_cls   = len(class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    aucs    = []
    for i, name in enumerate(class_names):
        col   = y_bin[:, i] if n_cls > 2 else y_bin[:, 0]
        prob  = y_proba[:, i]
        fpr, tpr, _ = roc_curve(col, prob)
        auc_val     = roc_auc_score(col, prob)
        aucs.append(auc_val)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (one-vs-rest)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    wandb.log({"roc_curves": wandb.Image(fig)})
    plt.close(fig)

    return float(np.mean(aucs))


def _log_imbalance_and_performance(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    """
    Two-panel figure that links class imbalance to per-class performance.

    Top panel  – Stacked bar chart: absolute sample counts per class in the
                 training split (shows imbalance) with the test-set support
                 overlaid as a hatch pattern.
    Bottom panel – Per-class Precision / Recall / F1 bars side-by-side so the
                   reader can immediately see whether minority classes are the
                   ones suffering on all three metrics.

    Saved to out_dir/imbalance_performance.png and logged to the active wandb run.
    """
    report = classification_report(
        y_test, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    n_cls       = len(class_names)
    train_counts = np.array([np.sum(y_train == i) for i in range(n_cls)])
    test_counts  = np.array([np.sum(y_test  == i) for i in range(n_cls)])
    precision    = np.array([report[c]["precision"] for c in class_names])
    recall       = np.array([report[c]["recall"]    for c in class_names])
    f1           = np.array([report[c]["f1-score"]  for c in class_names])

    x      = np.arange(n_cls)
    width  = 0.25
    # Color-code bars by train imbalance ratio so minority classes stand out
    imb_ratio = train_counts / train_counts.max()               # 1.0 = majority class
    bar_colors = plt.cm.RdYlGn(imb_ratio)                       # red = minority, green = majority

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(max(8, n_cls * 1.2), 9),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.45},
    )
    fig.suptitle("Class Imbalance  ×  Per-class Performance", fontsize=13, fontweight="bold")

    # ── Top: class distribution ───────────────────────────────────────────────
    bars_train = ax_top.bar(
        x, train_counts, color=bar_colors, edgecolor="white", linewidth=0.6,
        label="Train samples",
    )
    bars_test = ax_top.bar(
        x, test_counts,
        color="none", edgecolor="steelblue", linewidth=1.4,
        hatch="///", label="Test samples",
    )
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax_top.set_ylabel("Sample count")
    ax_top.set_title("Class distribution (train fill · test outline)")
    ax_top.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_top.legend(fontsize=8, loc="upper right")

    # Annotate each bar with absolute counts
    for bar, tc, vc in zip(bars_train, train_counts, test_counts):
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + train_counts.max() * 0.01,
            f"tr:{tc}\nte:{vc}",
            ha="center", va="bottom", fontsize=7, linespacing=1.3,
        )

    # Imbalance ratio annotation (minority class = 1.0 reference)
    min_cnt = train_counts.min()
    for i, cnt in enumerate(train_counts):
        ratio = cnt / min_cnt
        ax_top.text(
            x[i], -train_counts.max() * 0.08,
            f"×{ratio:.1f}",
            ha="center", va="top", fontsize=7, color="dimgray",
        )
    ax_top.text(
        -0.6, -train_counts.max() * 0.08,
        "ratio:", ha="right", va="top", fontsize=7, color="dimgray",
    )

    # ── Bottom: per-class precision / recall / F1 ─────────────────────────────
    ax_bot.bar(x - width, precision, width, label="Precision", color="steelblue",   alpha=0.85)
    ax_bot.bar(x,         recall,    width, label="Recall",    color="darkorange",  alpha=0.85)
    ax_bot.bar(x + width, f1,        width, label="F1-score",  color="mediumseagreen", alpha=0.85)

    # Overlay imbalance ratio as a step line so the correlation is visible
    ax_r = ax_bot.twinx()
    ax_r.step(x, imb_ratio, where="mid", color="crimson", linewidth=1.5,
              linestyle="--", label="Train imbalance ratio")
    ax_r.set_ylim(0, 1.4)
    ax_r.set_ylabel("Imbalance ratio  (1 = majority)", fontsize=8, color="crimson")
    ax_r.tick_params(axis="y", labelcolor="crimson", labelsize=8)
    ax_r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax_bot.set_ylabel("Score")
    ax_bot.set_ylim(0, 1.15)
    ax_bot.set_title("Per-class Precision / Recall / F1  (dashed = train imbalance ratio)")
    ax_bot.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Merge legends from both axes
    h1, l1 = ax_bot.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_bot.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    fig.tight_layout()
    path = out_dir / "imbalance_performance.png"
    fig.savefig(path, dpi=150)
    wandb.log({"imbalance_performance": wandb.Image(fig)})
    plt.close(fig)

    # Also log per-class metrics as a wandb Table for filtering in the UI
    table = wandb.Table(
        columns=["class", "train_count", "test_count", "imbalance_ratio",
                 "precision", "recall", "f1"],
        data=[
            [
                class_names[i],
                int(train_counts[i]),
                int(test_counts[i]),
                round(float(imb_ratio[i]), 4),
                round(float(precision[i]), 4),
                round(float(recall[i]), 4),
                round(float(f1[i]), 4),
            ]
            for i in range(n_cls)
        ],
    )
    wandb.log({"per_class_metrics": table})


# ── Comparative plots ─────────────────────────────────────────────────────────

def compare_models(results: list[dict], L: float, S: float, out_dir: Path) -> None:
    """
    Save comparative bar charts for all models evaluated at the same (L, S).

    Plots: accuracy, AUC, latency (ms), model size (MB).
    Saved to out_dir/comparison_L{L}_S{S}.png and also logged to a dedicated
    wandb run so they appear alongside the per-model runs.
    """
    metrics  = ["accuracy", "auc", "latency_ms", "model_size_mb"]
    titles   = ["Accuracy", "Macro AUC", "Latency (ms)", "Model size (MB)"]
    models   = [r["model"] for r in results]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    fig.suptitle(f"Model comparison  —  L={L}s  S={S}s", fontsize=13)

    for ax, metric, title in zip(axes, metrics, titles):
        values = [r[metric] for r in results]
        bars   = ax.bar(models, values)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.2)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    fig.tight_layout()
    path = out_dir / f"comparison_L{L}_S{S}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[compare] Saved {path}")

    run_dir = out_dir / f"comparison_L{L}_S{S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        name=f"comparison_L{L}_S{S}",
        dir=run_dir,
        config={"window_length": L, "stride": S, "type": "comparison"},
    ):
        wandb.log({"model_comparison": wandb.Image(str(path))})
        # Also log each metric as a bar-chart table for the wandb UI
        for metric, title in zip(metrics, titles):
            wandb.log({
                f"comparison/{metric}": wandb.plot.bar(
                    wandb.Table(
                        columns=["model", metric],
                        data=[[r["model"], r[metric]] for r in results],
                    ),
                    "model", metric, title=title,
                )
            })


# ── Core experiment ───────────────────────────────────────────────────────────

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
        class_names = le.classes_.tolist()

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
        y_proba  = best_estimator.predict_proba(X_test)
        accuracy = accuracy_score(y_test_enc, y_pred)

        # ── Per-run plots ──────────────────────────────────────────────────────
        _log_confusion_matrix(y_test_enc, y_pred, class_names, run_dir)
        auc = _log_roc_curves(y_test_enc, y_proba, class_names, run_dir)
        _log_imbalance_and_performance(
            y_train_enc, y_test_enc, y_pred, class_names, run_dir
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
        "accuracy":       accuracy,
        "auc":            auc,
        "latency_ms":     latency_ms,
        "model_size_mb":  size_mb,
    }


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
    out_dir  = Path("outputs")
    n_runs   = len(combos) * len(cfg["models"])

    print(f"[train] Config       : {args.config}")
    print(f"[train] Models       : {list(cfg['models'].keys())}")
    print(f"[train] (L, S) pairs : {combos}")
    print(f"[train] Total runs   : {n_runs}  ({len(combos)} window combos × {len(cfg['models'])} models)")

    for L, S in combos:
        combo_results = []
        for model_key, model_cfg in cfg["models"].items():
            result = run(model_key, model_cfg["param_grid"], base_cfg, L, S)
            combo_results.append(result)
        compare_models(combo_results, L, S, out_dir)
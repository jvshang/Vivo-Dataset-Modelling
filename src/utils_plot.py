import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
import wandb
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

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

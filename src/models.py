"""
models.py – Model registry for VIVO activity recognition.

Adding a new model
------------------
1. Subclass BaseClassifier and implement fit / predict / predict_proba.
2. Register it with @register_model("your_key").
3. Add its hyperparameters to DEFAULT_CONFIG in train.py if needed.

Available keys: rf | gb | lr | xgb | rocket | minirocket | tsf | lstmcnn

sktime models (rocket, minirocket)
-----------------------------------
sktime classifiers expect 3-D input (n_samples, n_channels, n_timepoints).
The current pipeline provides flat 2-D windows (n_samples, n_features) that
are the concatenation of EMG and IMU windows at their native sampling rates.
These wrappers reshape to (n_samples, 1, n_features), treating the entire
feature vector as a single-channel time series.  A future improvement would
be to pass the original multi-channel 3-D windows directly.
"""

from __future__ import annotations

import abc
from typing import Dict, Type

import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

import torch
import torch.nn as nn

try:
    from cuml.ensemble import RandomForestClassifier
    from cuml.linear_model import LogisticRegression
    _CUML = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    _CUML = False

_DEVICE = "GPU (cuML)" if _CUML else "CPU (sklearn)"


# ── Base class ────────────────────────────────────────────────────────────────

class BaseClassifier(abc.ABC):
    """Common interface for all activity classifiers."""

    name: str        # human-readable name shown in logs / wandb
    _needs_3d: bool = False  # True for sktime classifiers that expect (N, C, T) input
    _is_torch: bool = False  # True for PyTorch models that need a custom CV loop

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Train on (X, y). Returns self."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for each sample in X."""

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (N, n_classes)."""


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Type[BaseClassifier]] = {}


def register_model(key: str):
    """Decorator to register a classifier class under a short key."""
    def decorator(cls: Type[BaseClassifier]):
        _REGISTRY[key] = cls
        return cls
    return decorator


def build_model(cfg) -> BaseClassifier:
    """Instantiate the model specified by cfg.model (wandb config or dict-like)."""
    key = cfg.model if hasattr(cfg, "model") else cfg["model"]
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model {key!r}. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[key](cfg)


def available_models():
    return list(_REGISTRY.keys())


# ── Implementations ───────────────────────────────────────────────────────────

@register_model("rf")
class RandomForestModel(BaseClassifier):
    name = "Random Forest"

    def __init__(self, cfg):
        n_est  = cfg.n_estimators if hasattr(cfg, "n_estimators") else cfg.get("n_estimators", 50)
        depth  = cfg.max_depth    if hasattr(cfg, "max_depth")    else cfg.get("max_depth", 10)
        seed   = cfg.seed         if hasattr(cfg, "seed")         else cfg.get("seed", 42)
        kwargs = dict(n_estimators=n_est, max_depth=depth, random_state=seed)
        if cfg.get("use_class_weights", False):
            kwargs["class_weight"] = "balanced"
        print(f"Loading Random Forest with {_DEVICE}")
        if not _CUML:
            kwargs["n_jobs"] = -1
        self._clf = RandomForestClassifier(**kwargs)

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


@register_model("gb")
class GradientBoostingModel(BaseClassifier):
    name = "Gradient Boosting"

    def __init__(self, cfg):
        n_est  = cfg.n_estimators if hasattr(cfg, "n_estimators") else cfg.get("n_estimators", 50)
        depth  = cfg.max_depth    if hasattr(cfg, "max_depth")    else cfg.get("max_depth", 10)
        seed   = cfg.seed         if hasattr(cfg, "seed")         else cfg.get("seed", 42)
        self._clf = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=depth, random_state=seed
        )

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

@register_model("hgb")
class HistGradientBoostingModel(BaseClassifier):
    """
    Histogram-based Gradient Boosting — 10–50× faster than vanilla GB.
    Supports early stopping and handles missing values natively.
    CPU-only (sklearn); parallelism via GridSearchCV n_jobs=-1.
    """
    name = "Hist Gradient Boosting"

    def __init__(self, cfg):
        max_iter  = cfg.max_iter if hasattr(cfg, "max_iter") else cfg.get("max_iter", 300)
        depth     = cfg.max_depth if hasattr(cfg, "max_depth") else cfg.get("max_depth", None)
        seed      = cfg.seed         if hasattr(cfg, "seed")         else cfg.get("seed", 42)
        kwargs = dict(max_iter=max_iter, max_depth=depth, early_stopping=True, random_state=seed)
        if cfg.get("use_class_weights", False):
            kwargs["class_weight"] = "balanced"
        self._clf = HistGradientBoostingClassifier(**kwargs)

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


@register_model("xgb")
class XGBoostModel(BaseClassifier):
    name = "XGBoost"

    def __init__(self, cfg):
        n_est  = cfg.n_estimators if hasattr(cfg, "n_estimators") else cfg.get("n_estimators", 50)
        depth  = cfg.max_depth    if hasattr(cfg, "max_depth")    else cfg.get("max_depth", 10)
        seed   = cfg.seed         if hasattr(cfg, "seed")         else cfg.get("seed", 42)
        print(f"Loading XGBoost with {_DEVICE}")
        self._clf = XGBClassifier(
            n_estimators=n_est, max_depth=depth, random_state=seed,
            device="cuda" if _CUML else "cpu",
            eval_metric="mlogloss", verbosity=0,
        )

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


@register_model("lr")
class LogisticRegressionModel(BaseClassifier):
    name = "Logistic Regression"

    def __init__(self, cfg):
        C        = cfg.C        if hasattr(cfg, "C")        else cfg.get("C", 1.0)
        max_iter = cfg.max_iter if hasattr(cfg, "max_iter") else cfg.get("max_iter", 500)
        seed     = cfg.seed     if hasattr(cfg, "seed")     else cfg.get("seed", 42)
        kwargs   = dict(C=C, max_iter=max_iter)
        if cfg.get("use_class_weights", False):
            kwargs["class_weight"] = "balanced"
        print(f"Loading Logistic Regression with {_DEVICE}")
        if not _CUML:
            kwargs["n_jobs"] = -1
            kwargs["random_state"] = seed
        self._clf = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


# ── sktime wrappers ───────────────────────────────────────────────────────────
# Input X is 2-D (n_samples, n_features). sktime classifiers need 3-D
# (n_samples, n_channels, n_timepoints), so we add a channel dimension.

def _to_3d(X: np.ndarray) -> np.ndarray:
    """Reshape (N, F) → (N, 1, F) for sktime classifiers."""
    return X[:, np.newaxis, :]


@register_model("rocket")
class RocketModel(BaseClassifier):
    name      = "ROCKET"
    _needs_3d = True

    def __init__(self, cfg):
        from sktime.classification.kernel_based import RocketClassifier
        n_kernels = (
            cfg.num_kernels if hasattr(cfg, "num_kernels") else cfg.get("num_kernels", 10_000)
        )
        seed   = cfg.seed         if hasattr(cfg, "seed")         else cfg.get("seed", 42)
        
        # Safely resolve n_jobs for Numba (cap at 72 to avoid ValueError)
        n_jobs = min(os.cpu_count() or 1, 72)
        
        self._clf = RocketClassifier(num_kernels=n_kernels, random_state=seed, n_jobs=n_jobs)

    def fit(self, X, y):
        self._clf.fit(_to_3d(X), y)
        return self

    def predict(self, X):
        return self._clf.predict(_to_3d(X))

    def predict_proba(self, X):
        return self._clf.predict_proba(_to_3d(X))


@register_model("minirocket")
class MiniRocketModel(BaseClassifier):
    name      = "MiniROCKET"
    _needs_3d = True

    def __init__(self, cfg):
        from sktime.classification.kernel_based import RocketClassifier
        n_kernels = (
            cfg.num_kernels if hasattr(cfg, "num_kernels") else cfg.get("num_kernels", 10_000)
        )
        seed   = cfg.seed         if hasattr(cfg, "seed")         else cfg.get("seed", 42)
        
        # Safely resolve n_jobs for Numba (cap at 72 to avoid ValueError)
        n_jobs = min(os.cpu_count() or 1, 72)
        
        self._clf = RocketClassifier(num_kernels=n_kernels, random_state=seed, n_jobs=n_jobs, rocket_transform="minirocket")

    def fit(self, X, y):
        self._clf.fit(_to_3d(X), y)
        return self

    def predict(self, X):
        return self._clf.predict(_to_3d(X))

    def predict_proba(self, X):
        return self._clf.predict_proba(_to_3d(X))


@register_model("tsf")
class TimeSeriesForestModel(BaseClassifier):
    name      = "Time Series Forest"
    _needs_3d = True

    def __init__(self, cfg):
        from sktime.classification.interval_based import TimeSeriesForestClassifier
        n_est = (
            cfg.n_estimators if hasattr(cfg, "n_estimators") else cfg.get("n_estimators", 50)
        )
        seed   = cfg.seed         if hasattr(cfg, "seed")         else cfg.get("seed", 42)
        self._clf = TimeSeriesForestClassifier(
            n_estimators=n_est, random_state=seed, n_jobs=-1
        )

    def fit(self, X, y):
        self._clf.fit(_to_3d(X), y)
        return self

    def predict(self, X):
        return self._clf.predict(_to_3d(X))

    def predict_proba(self, X):
        return self._clf.predict_proba(_to_3d(X))


# ── LSTM-CNN hybrid (PyTorch) ─────────────────────────────────────────────────
# Architecture:
#   Conv1d stack  – extract local patterns across the feature/time axis
#   LSTM          – model sequential dependencies across the convolved output
#   FC            – map last hidden state to class logits
#
# Input X is 2-D (N, F); we treat the F-length vector as a univariate time
# series fed into Conv1d channels-first: (N, 1, F).

class LSTMCNNPyTorchNet(nn.Module):
    """Global module so Python can pickle it for the size check."""
    def __init__(self, n_classes: int, hidden: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden,
            num_layers=2, batch_first=True, dropout=0.3,
        )
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):          # x: (B, 1, F)
        x = self.cnn(x)            # (B, 64, F')
        x = x.permute(0, 2, 1)     # (B, F', 64) — seq-first for LSTM
        _, (h, _) = self.lstm(x)   # h: (layers, B, hidden)
        return self.fc(h[-1])      # (B, n_classes)


@register_model("lstmcnn")
class LSTMCNNModel(BaseClassifier):
    name      = "LSTM-CNN"
    _is_torch = True

    def __init__(self, cfg):
        self._hidden     = cfg.lstm_hidden  if hasattr(cfg, "lstm_hidden")  else cfg.get("lstm_hidden",  128)
        self._epochs     = cfg.epochs       if hasattr(cfg, "epochs")       else cfg.get("epochs",       30)
        self._batch_size = cfg.batch_size   if hasattr(cfg, "batch_size")   else cfg.get("batch_size",   256)
        self._seed       = cfg.seed         if hasattr(cfg, "seed")         else cfg.get("seed",         None)
        self._use_class_weights = cfg.get("use_class_weights", False)
        self._net        = None
        self._classes    = None

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    def _seed_everything(self) -> None:
        """Seed all RNGs that affect training.  No-op when seed is None."""
        if self._seed is None:
            return
        import random, torch
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        # Sacrifice a little speed for deterministic CUDA kernels
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    def _make_generator(self):
        """Return a seeded (or unseeded) torch.Generator for the DataLoader."""
        g = torch.Generator()
        if self._seed is not None:
            g.manual_seed(self._seed)
        return g

    # ------------------------------------------------------------------

    def _device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMCNNModel":

        from torch.utils.data import DataLoader, TensorDataset

        self._seed_everything()  # seed before any weight init or data shuffling

        self._classes = np.unique(y)
        n_classes  = len(self._classes)
        n_features = X.shape[1]
        device     = self._device()

        self._net = LSTMCNNPyTorchNet(n_classes, self._hidden).to(device)

        X_t = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)  # (N,1,F)
        y_t = torch.tensor(y.astype(np.int64))
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=False,
            generator=self._make_generator(),   # reproducible shuffle order
            worker_init_fn=(                    # seed DataLoader workers too
                (lambda wid: np.random.seed(self._seed + wid))
                if self._seed is not None else None
            ),
        )

        optimiser = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        if self._use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            weights = compute_class_weight("balanced", classes=self._classes, y=y)
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        self._net.train()
        for _ in range(self._epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                criterion(self._net(xb), yb).backward()
                optimiser.step()
        return self

    def _forward(self, X: np.ndarray):
        self._net.eval()
        device = self._device()
        
        # Keep the full tensor on the CPU initially to save VRAM
        X_t = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)
        
        logits_list = []
        with torch.no_grad():
            # Process the data in chunks
            for i in range(0, len(X_t), self._batch_size):
                # Move only the current batch to the GPU
                xb = X_t[i : i + self._batch_size].to(device)
                logits = self._net(xb)
                
                # Immediately move the output back to CPU and store it
                logits_list.append(logits.cpu().numpy())
                
        # Combine all batches into a single array
        return np.concatenate(logits_list, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._classes[self._forward(X).argmax(axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._forward(X)
        exp    = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

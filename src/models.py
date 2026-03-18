"""
models.py – Model registry for VIVO activity recognition.

Adding a new model
------------------
1. Subclass BaseClassifier and implement fit / predict / predict_proba.
2. Register it with @register_model("your_key").
3. Add its hyperparameters to DEFAULT_CONFIG in train.py if needed.

Available keys: rf | gb | lr
"""

from __future__ import annotations

import abc
from typing import Dict, Type

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ── Base class ────────────────────────────────────────────────────────────────

class BaseClassifier(abc.ABC):
    """Common interface for all activity classifiers."""

    name: str  # human-readable name shown in logs / wandb

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
        self._clf = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1
        )

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
        self._clf = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=depth, random_state=42
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
        C        = cfg.lr_C        if hasattr(cfg, "lr_C")        else cfg.get("lr_C", 1.0)
        max_iter = cfg.lr_max_iter if hasattr(cfg, "lr_max_iter") else cfg.get("lr_max_iter", 500)
        self._clf = LogisticRegression(
            C=C, max_iter=max_iter, random_state=42, n_jobs=-1
        )

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

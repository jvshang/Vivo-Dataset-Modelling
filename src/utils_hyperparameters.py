from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from itertools import product
from models import build_model
from sklearn.metrics import accuracy_score

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
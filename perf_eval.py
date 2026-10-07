"""
perf_eval.py
============
Covers:
    - accuracy, precision, recall, F1-score
    - ROC-AUC score
    - confusion matrix
    - training time and inference time
    - model size (parameter count + memory footprint)

Usage with a PyTorch model (nn_model.py)
------------------------------------------
    from perf_eval import PerfEvaluator, model_size_torch, torch_predict_proba

    pe = PerfEvaluator("nn_mlp_64")
    pe.time_fit(train_and_evaluate, hidden_size=64, train_path=..., test_path=...)
    y_proba = pe.time_predict(torch_predict_proba(model, device), X_test_tensor)
    pe.record_model_size(*model_size_torch(model))
    result = pe.evaluate(y_test.numpy(), y_proba)
    pe.print_report()

Comparing every model that has been evaluated
-----------------------------------------------
    from perf_eval import comparison_table
    comparison_table()   # one row per results/*.json, sorted by model name
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RESULTS_DIR = Path("results")


# ============================================================
# Result container
# ============================================================
@dataclass
class EvalResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list  # [[TN, FP], [FN, TP]]
    train_time_s: float
    inference_time_s: float
    inference_ms_per_1000_samples: float
    n_params: int
    memory_kb: float
    memory_mb: float = field(init=False)

    def __post_init__(self):
        self.memory_mb = self.memory_kb / 1024

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        cm = self.confusion_matrix
        lines = [
            f"=== Performance Report: {self.model} ===",
            f"Accuracy:        {self.accuracy:.4f}",
            f"Precision:       {self.precision:.4f}",
            f"Recall:          {self.recall:.4f}",
            f"F1-score:        {self.f1:.4f}",
            f"ROC-AUC:         {self.roc_auc:.4f}",
            f"Confusion matrix [[TN, FP], [FN, TP]]: {cm}",
            f"Training time:   {self.train_time_s:.4f} s",
            f"Inference time:  {self.inference_time_s:.6f} s "
            f"({self.inference_ms_per_1000_samples:.4f} ms / 1000 samples)",
            f"Model size:      {self.n_params:,} params, "
            f"{self.memory_kb:.2f} KB ({self.memory_mb:.3f} MB)",
        ]
        return "\n".join(lines)


# ============================================================
# Main evaluator
# ============================================================
class PerfEvaluator:
    """Times training/inference, computes metrics, and persists results.

    Call order: time_fit -> time_predict -> record_model_size -> evaluate.
    Any step can be skipped in favor of setting the attribute directly
    (e.g. `pe.train_time_s = 12.3`) if already measured it elsewhere.
    """

    def __init__(self, name: str, results_dir: Path | str = RESULTS_DIR,
                threshold: float = 0.5, n_timing_runs: int = 5):
        self.name = name
        self.results_dir = Path(results_dir)
        self.threshold = threshold
        self.n_timing_runs = n_timing_runs

        self.train_time_s: float | None = None
        self.inference_time_s: float | None = None
        self.n_samples_timed: int | None = None
        self.n_params: int | None = None
        self.memory_bytes: int | None = None
        self.result: EvalResult | None = None

    # ---- timing -----------------------------------------------------
    def time_fit(self, fit_fn: Callable, *args, **kwargs):
        """Runs fit_fn(*args, **kwargs) once, recording wall-clock time."""
        t0 = time.perf_counter()
        out = fit_fn(*args, **kwargs)
        self.train_time_s = time.perf_counter() - t0
        return out

    def time_predict(self, predict_proba_fn: Callable[[object], np.ndarray], X,
                    n_runs: int | None = None) -> np.ndarray:
        """Runs predict_proba_fn(X) several times and records the median time.

        predict_proba_fn must return an array of P(class == 1), one per row
        of X. Returns that array (from the final run) for use in evaluate().
        """
        n_runs = n_runs or self.n_timing_runs
        times, proba = [], None
        for _ in range(n_runs):
            t0 = time.perf_counter()
            proba = np.asarray(predict_proba_fn(X))
            times.append(time.perf_counter() - t0)
        self.inference_time_s = float(np.median(times))
        self.n_samples_timed = len(proba)
        return proba

    # ---- model size ---------------------------------------------------
    def record_model_size(self, n_params: int, memory_bytes: int):
        self.n_params = int(n_params)
        self.memory_bytes = int(memory_bytes)

    # ---- metrics --------------------------------------------------------
    def evaluate(self, y_true, y_proba, threshold: float | None = None, save: bool = True) -> EvalResult:
        if self.train_time_s is None or self.inference_time_s is None:
            raise RuntimeError("call time_fit() and time_predict() before evaluate()")
        if self.n_params is None or self.memory_bytes is None:
            raise RuntimeError("call record_model_size() before evaluate()")

        threshold = threshold if threshold is not None else self.threshold
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        y_pred = (y_proba >= threshold).astype(int)

        n_samples = self.n_samples_timed or len(y_true)
        self.result = EvalResult(
            model=self.name,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            f1=f1_score(y_true, y_pred),
            roc_auc=roc_auc_score(y_true, y_proba),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
            train_time_s=float(self.train_time_s),
            inference_time_s=float(self.inference_time_s),
            inference_ms_per_1000_samples=self.inference_time_s / n_samples * 1000 * 1000,
            n_params=self.n_params,
            memory_kb=self.memory_bytes / 1024,
        )
        if save:
            self.save()
        return self.result

    # ---- reporting / persistence ------------------------------------
    def print_report(self):
        if self.result is None:
            raise RuntimeError("call evaluate() first")
        print(self.result)

    def plot_confusion_matrix(self, ax=None, labels=("<=50K", ">50K")):
        import matplotlib.pyplot as plt

        if self.result is None:
            raise RuntimeError("call evaluate() first")
        cm = np.array(self.result.confusion_matrix)
        ax = ax or plt.gca()
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(self.name)
        return ax

    def save(self, path: Path | str | None = None):
        if self.result is None:
            raise RuntimeError("call evaluate() first")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = Path(path) if path else self.results_dir / f"{self.name}.json"
        json.dump(self.result.to_dict(), open(path, "w"), indent=2)
        return path


# ============================================================
# Model-size helpers
# ============================================================
def model_size_sklearn(model) -> tuple[int, int]:
    """Returns (n_params, memory_bytes) for a fitted scikit-learn estimator.

    memory_bytes is the pickled size (a reasonable proxy for footprint).
    n_params is estimator-specific: linear models count coefficients,
    trees count nodes, ensembles sum over their members.
    """
    memory_bytes = len(pickle.dumps(model))
    return _count_sklearn_params(model), memory_bytes


def _count_sklearn_params(model) -> int:
    if hasattr(model, "coef_"):
        n = int(np.asarray(model.coef_).size)
        if hasattr(model, "intercept_"):
            n += int(np.asarray(model.intercept_).size)
        return n
    if hasattr(model, "tree_"):
        return int(model.tree_.node_count)
    if hasattr(model, "estimators_"):
        return int(sum(_count_sklearn_params(e) for e in model.estimators_))
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    return 0


def model_size_torch(model) -> tuple[int, int]:
    """Returns (n_params, memory_bytes) for a PyTorch nn.Module."""
    n_params = sum(p.numel() for p in model.parameters())
    memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    memory_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return int(n_params), int(memory_bytes)


# ============================================================
# Predict-proba adapters
# ============================================================
def sklearn_predict_proba(model) -> Callable[[object], np.ndarray]:
    """X -> P(class == 1) for any scikit-learn classifier with predict_proba."""
    return lambda X: model.predict_proba(X)[:, 1]


def torch_predict_proba(model, device=None) -> Callable[[object], np.ndarray]:
    """X -> P(class == 1) for a 2-logit PyTorch classifier (softmax over dim=1).

    X may be a tensor or anything torch.as_tensor accepts. Puts the model
    in eval mode and disables gradients for the call.
    """
    import torch
    import torch.nn.functional as F

    def _predict(X) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            X_t = X if torch.is_tensor(X) else torch.as_tensor(X, dtype=torch.float32)
            if device is not None:
                X_t = X_t.to(device)
            logits = model(X_t)
            proba = F.softmax(logits, dim=1)[:, 1]
            return proba.cpu().numpy()

    return _predict


# ============================================================
# Cross-model comparison
# ============================================================
def comparison_table(results_dir: Path | str = RESULTS_DIR) -> pd.DataFrame:
    """Merges every results/*.json saved by PerfEvaluator.save() into one table."""
    results_dir = Path(results_dir)
    rows = [json.load(open(p)) for p in sorted(results_dir.glob("*.json"))]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop(columns=["confusion_matrix"]).set_index("model")
    return df.round(4)

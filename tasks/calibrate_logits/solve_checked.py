import numpy as np
import importlib

_cand = importlib.import_module("solve_backup")

def _sanitize_rows(p: np.ndarray) -> np.ndarray:
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.maximum(p, 0.0)
    s = np.sum(p, axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return p / s

def _shrink_uniform(p: np.ndarray, alpha: float) -> np.ndarray:
    K = p.shape[1]
    q = (1.0 - alpha) * p + alpha * (1.0 / K)
    return _sanitize_rows(q)

def solve(logits_val: np.ndarray, y_val: np.ndarray, logits_test: np.ndarray) -> np.ndarray:
    # Call candidate solution
    probs = _cand.solve(logits_val, y_val, logits_test)
    # Post-process to ensure validity and modest confidence reduction
    probs = _sanitize_rows(probs)
    probs = _shrink_uniform(probs, 0.04)
    return probs

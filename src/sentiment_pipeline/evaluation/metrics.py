"""Agreement between a model's labels and the human labels, with bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES = ("negative", "neutral", "positive")


def confusion(truth, pred, classes=CLASSES) -> np.ndarray:
    """Rows: human label. Columns: model label."""
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(truth, pred, strict=True):
        m[idx[t], idx[p]] += 1
    return m


def accuracy(m: np.ndarray) -> float:
    return float(np.trace(m) / m.sum()) if m.sum() else float("nan")


def cohen_kappa(m: np.ndarray) -> float:
    n = m.sum()
    if n == 0:
        return float("nan")
    po = np.trace(m) / n
    pe = float((m.sum(axis=1) * m.sum(axis=0)).sum()) / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def macro_f1(m: np.ndarray) -> float:
    f1s = []
    for i in range(len(m)):
        tp, fp, fn = m[i, i], m[:, i].sum() - m[i, i], m[i, :].sum() - m[i, i]
        if tp + fp + fn == 0:
            continue  # class absent from both: no information
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


@dataclass
class Score:
    n: int
    accuracy: float
    kappa: float
    macro_f1: float
    ci: dict[str, tuple[float, float]]
    matrix: np.ndarray


def score(truth, pred, n_boot: int = 2000, seed: int = 7) -> Score:
    truth, pred = np.asarray(truth), np.asarray(pred)
    m = confusion(truth, pred)
    rng = np.random.default_rng(seed)
    boots = {"accuracy": [], "kappa": [], "macro_f1": []}
    for _ in range(n_boot if len(truth) else 0):
        i = rng.integers(0, len(truth), len(truth))
        bm = confusion(truth[i], pred[i])
        boots["accuracy"].append(accuracy(bm))
        boots["kappa"].append(cohen_kappa(bm))
        boots["macro_f1"].append(macro_f1(bm))
    ci = {}
    for k, v in boots.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]  # e.g. kappa is undefined on a resample with a single class
        ci[k] = tuple(np.percentile(v, [2.5, 97.5])) if v.size else (float("nan"), float("nan"))
    return Score(len(truth), accuracy(m), cohen_kappa(m), macro_f1(m), ci, m)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with a Wilson 95% interval (good for small n and proportions near 0)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)

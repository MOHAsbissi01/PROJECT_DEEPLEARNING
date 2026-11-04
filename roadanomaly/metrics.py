"""Pixel-level evaluation metrics.

Threshold-dependent: precision, recall, F1, Dice, IoU (from pooled TP/FP/FN/TN).
Threshold-free (as in Lis et al. 2019): AUROC, average precision (AP) and
FPR at 95 % TPR, computed exactly over all pixels.
"""

from __future__ import annotations

import numpy as np


def safe_divide(numerator, denominator) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    tp = int(np.count_nonzero(y_true & y_pred))
    fp = int(np.count_nonzero(~y_true & y_pred))
    fn = int(np.count_nonzero(y_true & ~y_pred))
    tn = int(y_true.size - tp - fp - fn)
    return tp, fp, fn, tn


def scores_from_counts(tp, fp, fn, tn) -> dict:
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "f1": f1,
        "dice": f1,  # identical to F1 for binary masks
        "iou": safe_divide(tp, tp + fp + fn),
        "accuracy": safe_divide(tp + tn, tp + fp + fn + tn),
    }


def image_scores(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    return scores_from_counts(*confusion_counts(y_true, probabilities > threshold))


def threshold_sweep(pairs, thresholds) -> list[dict]:
    """Pooled metrics for each threshold. ``pairs`` = [(y_true_bool, probabilities), ...]."""
    rows = []
    for threshold in thresholds:
        totals = np.zeros(4, dtype=np.int64)
        for y_true, probabilities in pairs:
            totals += confusion_counts(y_true, probabilities > threshold)
        row = scores_from_counts(*totals)
        row["threshold"] = float(threshold)
        rows.append(row)
    return rows


def select_threshold(rows: list[dict]) -> dict:
    """Best pooled F1; ties broken by recall, then by closeness to 0.5."""
    return max(rows, key=lambda r: (r["f1"], r["recall"], -abs(r["threshold"] - 0.5)))


def evaluate(pairs, threshold: float, names=None) -> dict:
    totals = np.zeros(4, dtype=np.int64)
    per_image = []
    for index, (y_true, probabilities) in enumerate(pairs):
        counts = confusion_counts(y_true, probabilities > threshold)
        totals += counts
        row = scores_from_counts(*counts)
        row["threshold"] = float(threshold)
        if names is not None:
            row["name"] = names[index]
        per_image.append(row)
    return {"threshold": float(threshold), "global_metrics": scores_from_counts(*totals), "per_image_metrics": per_image}


def ranking_metrics(pairs, max_pixels: int | None = None, seed: int = 0) -> dict:
    """Exact pixel-level AUROC, average precision and FPR@95%TPR.

    Ties in scores are grouped so the result does not depend on pixel order.
    ``max_pixels`` optionally subsamples pixels uniformly (for very large sets).
    """
    scores = np.concatenate([p.ravel() for _, p in pairs]).astype(np.float32)
    labels = np.concatenate([t.ravel() for t, _ in pairs]).astype(bool)
    if max_pixels and scores.size > max_pixels:
        index = np.random.default_rng(seed).choice(scores.size, max_pixels, replace=False)
        scores, labels = scores[index], labels[index]

    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return {"auroc": float("nan"), "average_precision": float("nan"), "fpr_at_95_tpr": float("nan")}

    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    distinct = np.r_[np.flatnonzero(np.diff(scores)), scores.size - 1]  # last index of each tie group
    tp = np.cumsum(labels)[distinct].astype(np.float64)
    fp = (distinct + 1 - tp).astype(np.float64)

    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    precision = tp / (tp + fp)
    recall = tp / positives

    auroc = float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    reach = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[reach[0]]) if reach.size else 1.0
    return {"auroc": auroc, "average_precision": average_precision, "fpr_at_95_tpr": fpr95}


def roc_pr_curves(pairs, n_points: int = 200, max_pixels: int = 2_000_000, seed: int = 0) -> dict:
    """Down-sampled ROC and PR curves for plotting."""
    scores = np.concatenate([p.ravel() for _, p in pairs]).astype(np.float32)
    labels = np.concatenate([t.ravel() for t, _ in pairs]).astype(bool)
    if scores.size > max_pixels:
        index = np.random.default_rng(seed).choice(scores.size, max_pixels, replace=False)
        scores, labels = scores[index], labels[index]
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, n_points)))[::-1]
    positives, negatives = max(labels.sum(), 1), max((~labels).sum(), 1)
    tpr, fpr, precision = [], [], []
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = np.count_nonzero(predicted & labels)
        fp = np.count_nonzero(predicted & ~labels)
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
        precision.append(safe_divide(tp, tp + fp) if tp + fp else 1.0)
    return {"fpr": np.array(fpr), "tpr": np.array(tpr), "precision": np.array(precision), "recall": np.array(tpr)}

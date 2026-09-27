"""
固定指标：Macro-F1、PR-AUC、危机召回、误报率（FPR）。

优先使用 scikit-learn（与论文常用实现一致）；若未安装则回退到 NumPy 等价实现。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_recall_curve,
        recall_score,
    )

    _HAS_SK = True
except ImportError:
    _HAS_SK = False


def _f1_per_class(tp: float, fp: float, fn: float) -> float:
    if tp <= 0 and fp <= 0 and fn <= 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec <= 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _compute_emotion_metrics_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 8) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    f1s = []
    supports = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        f1s.append(_f1_per_class(tp, fp, fn))
        supports.append(float(np.sum(y_true == c)))
    sup = np.asarray(supports, dtype=float)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    micro_f1 = float(np.mean(y_true == y_pred))
    if sup.sum() > 0:
        weighted_f1 = float(np.sum(np.asarray(f1s) * sup) / sup.sum())
    else:
        weighted_f1 = 0.0
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
    }


def _average_precision_binary_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """二元 average precision（与 sklearn 在常见设定下一致）。"""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    if n_pos == len(y):
        return 1.0
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    precisions = tps / np.maximum(tps + fps, 1)
    recalls = tps / n_pos
    # AP = sum (delta_recall * precision)，在 recall 变化处累加
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def _precision_recall_curve_np(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    desc = np.argsort(-s, kind="mergesort")
    y = y[desc]
    s = s[desc]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    n_pos = max(int(y.sum()), 1)
    precisions = tps / np.maximum(tps + fps, 1)
    recalls = tps / n_pos
    thresholds = s
    return precisions, recalls, thresholds


def compute_emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    多分类情绪：固定报告 Macro-F1（论文主表之一）。
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if _HAS_SK:
        return {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    return _compute_emotion_metrics_np(y_true, y_pred)


def compute_crisis_detection_metrics(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    二值危机检测（正类=危机）：

    - PR-AUC：average_precision_score(y_true, y_score)
    - 危机召回：recall（正类）
    - 误报率：FPR = FP / (FP + TN)，即在真实非危机中被判为危机的比例
    """
    y_true = np.asarray(y_true_binary).astype(int)
    s = np.asarray(y_score, dtype=float)
    if y_true.min() < 0 or y_true.max() > 1:
        raise ValueError("y_true_binary 必须为 0/1")
    y_pred = (s >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if y_true.sum() == 0:
        pr_auc = float("nan")
    elif _HAS_SK:
        pr_auc = float(average_precision_score(y_true, s))
    else:
        pr_auc = _average_precision_binary_np(y_true, s)

    return {
        "pr_auc": pr_auc,
        "crisis_recall": float(recall_pos),
        "false_positive_rate": float(fpr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "threshold": float(threshold),
    }


def labels_to_crisis_binary(
    y_labels: np.ndarray, crisis_indices: Tuple[int, ...] = (7,)
) -> np.ndarray:
    """将多类标签转为二值危机（任一危机类为 1）。"""
    y = np.asarray(y_labels).astype(int)
    m = np.zeros_like(y, dtype=int)
    for idx in crisis_indices:
        m |= (y == idx).astype(int)
    return m


def crisis_score_from_probs(
    proba: np.ndarray, crisis_indices: Tuple[int, ...] = (7,)
) -> np.ndarray:
    """将多类概率中危机类概率之和作为 y_score（用于 PR 曲线）。"""
    p = np.asarray(proba, dtype=float)
    s = np.zeros(p.shape[0], dtype=float)
    for i in crisis_indices:
        s += p[:, i]
    return np.clip(s, 0.0, 1.0)


def best_f1_threshold(y_true_binary: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """在验证集上扫阈值取 F1 最大点（可选，用于报告公平对比）。"""
    y = np.asarray(y_true_binary).astype(int)
    s = np.asarray(y_score, dtype=float)
    if _HAS_SK:
        prec, rec, thr = precision_recall_curve(y, s)
        f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    else:
        prec, rec, thr = _precision_recall_curve_np(y, s)
        f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    if len(f1s) == 0:
        return 0.5, 0.0
    j = int(np.nanargmax(f1s))
    if len(thr) == 0:
        return 0.5, float(f1s[j])
    j_thr = min(j, len(thr) - 1)
    return float(thr[j_thr]), float(f1s[j])

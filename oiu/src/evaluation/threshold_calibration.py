"""
危机预警阈值校准模块

在验证集上锁定阈值，测试集只应用已锁定阈值。
支持：
  - best_f1: 最大化 F1
  - fixed_recall: 在达到目标召回率的前提下最小化 FPR
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    strategy: str
    threshold: float
    f1: float
    precision: float
    recall: float
    fpr: float
    target_recall: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ThresholdCalibrator:
    """验证集阈值校准器。"""

    def __init__(
        self,
        strategy: str = "best_f1",
        target_recall: float = 0.90,
        num_grid: int = 101,
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
    ):
        if strategy not in {"best_f1", "fixed_recall"}:
            raise ValueError("strategy must be 'best_f1' or 'fixed_recall'")
        self.strategy = strategy
        self.target_recall = float(target_recall)
        self.num_grid = num_grid
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.threshold_: Optional[float] = None
        self.result_: Optional[CalibrationResult] = None
        self._locked: bool = False

    @staticmethod
    def _binary_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float
    ) -> Tuple[float, float, float, float]:
        y_pred = (y_score >= threshold).astype(np.int32)
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true == 0)))
        fn = float(np.sum((y_pred == 0) & (y_true == 1)))
        tn = float(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        return f1, precision, recall, fpr

    def _candidate_thresholds(self, y_score: np.ndarray) -> np.ndarray:
        grid = np.linspace(self.min_threshold, self.max_threshold, self.num_grid)
        # 同时并入分位点，提高稀疏正样本场景的稳定性
        quantiles = np.quantile(y_score, np.linspace(0.05, 0.95, 19))
        candidates = np.unique(np.concatenate([grid, quantiles]))
        return candidates

    def fit(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        strategy: Optional[str] = None,
        target_recall: Optional[float] = None,
    ) -> CalibrationResult:
        """
        仅在验证集上调用。拟合后阈值锁定，测试集应调用 predict / transform。
        """
        y_true = np.asarray(y_true).astype(np.int32).ravel()
        y_score = np.asarray(y_score).astype(np.float64).ravel()
        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError("y_true and y_score must have the same length")
        if y_true.size == 0:
            raise ValueError("empty validation set")

        strategy = strategy or self.strategy
        target_recall = self.target_recall if target_recall is None else float(target_recall)
        candidates = self._candidate_thresholds(y_score)

        rows: List[Tuple[float, float, float, float, float]] = []
        for thr in candidates:
            f1, precision, recall, fpr = self._binary_metrics(y_true, y_score, float(thr))
            rows.append((float(thr), f1, precision, recall, fpr))

        if strategy == "best_f1":
            # F1 优先，其次更高召回、更低 FPR
            best = max(rows, key=lambda r: (r[1], r[3], -r[4]))
            result = CalibrationResult(
                strategy=strategy,
                threshold=best[0],
                f1=best[1],
                precision=best[2],
                recall=best[3],
                fpr=best[4],
            )
        else:
            feasible = [r for r in rows if r[3] >= target_recall]
            if feasible:
                best = min(feasible, key=lambda r: (r[4], -r[1], -r[2]))
            else:
                # 达不到目标召回时，退化为最高召回，再最小化 FPR
                best = max(rows, key=lambda r: (r[3], -r[4], r[1]))
            result = CalibrationResult(
                strategy=strategy,
                threshold=best[0],
                f1=best[1],
                precision=best[2],
                recall=best[3],
                fpr=best[4],
                target_recall=target_recall,
            )

        self.threshold_ = result.threshold
        self.result_ = result
        self._locked = True
        return result

    @property
    def threshold(self) -> float:
        if self.threshold_ is None:
            raise RuntimeError("ThresholdCalibrator has not been fit/locked yet")
        return float(self.threshold_)

    def is_locked(self) -> bool:
        return self._locked

    def predict(self, y_score: np.ndarray) -> np.ndarray:
        """测试集：仅应用已锁定阈值。"""
        scores = np.asarray(y_score).astype(np.float64).ravel()
        return (scores >= self.threshold).astype(np.int32)

    def transform(self, y_score: np.ndarray) -> np.ndarray:
        return self.predict(y_score)

    def evaluate(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true).astype(np.int32).ravel()
        f1, precision, recall, fpr = self._binary_metrics(
            y_true, np.asarray(y_score).astype(np.float64).ravel(), self.threshold
        )
        return {
            'threshold': self.threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
        }


def calibrate_crisis_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strategy: str = "best_f1",
    target_recall: float = 0.90,
) -> CalibrationResult:
    """便捷函数：验证集上校准并返回结果。"""
    calibrator = ThresholdCalibrator(strategy=strategy, target_recall=target_recall)
    return calibrator.fit(y_true, y_score)

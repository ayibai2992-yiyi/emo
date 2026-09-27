"""
标准实验协议：用户级划分、固定指标、基线注册。

指标计算在 metrics_protocol 中（依赖 scikit-learn），请:
  pip install scikit-learn
后使用:
  from src.experiment.metrics_protocol import compute_emotion_metrics, ...
"""

from src.experiment.protocol import ExperimentProtocol, BaselineKind
from src.experiment.splits import summarize_split, user_stratified_masks

__all__ = [
    "ExperimentProtocol",
    "BaselineKind",
    "user_stratified_masks",
    "summarize_split",
]

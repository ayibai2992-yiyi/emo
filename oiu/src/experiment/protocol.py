"""
论文级实验协议：随机种子、划分比例、危机定义、基线枚举。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class BaselineKind(str, Enum):
    """固定对比基线（论文表格行）。"""

    TRADITIONAL = "traditional"  # 词袋/TF-IDF + LR 或 SVM 等浅层模型
    SINGLE_DEEP = "single_deep"  # 单一深度模型（如 BERT 直接 8 分类）
    THREE_LAYER = "three_layer"  # 当前系统：轻量筛选 + 深度 + 预警


@dataclass
class ExperimentProtocol:
    """
    标准实验协议（固定超参外仅允许改数据路径与输出目录）。

    数据划分：按 user_id 分层到 train/val/test，同一用户只出现在一个 split，防泄漏。
    指标：见 metrics_protocol。
    """

    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 8 类情绪下标与 deep_analyzer 一致：绝望=7 作为「危机相关」主类（可按数据改）
    crisis_emotion_indices: Tuple[int, ...] = (7,)
    # 二值危机检测：若样本带 is_crisis 列则优先用；否则由标签是否属于 crisis_emotion_indices 派生

    emotion_labels: Tuple[str, ...] = (
        "中性",
        "高兴",
        "惊讶",
        "悲伤",
        "愤怒",
        "恐惧",
        "厌恶",
        "绝望",
    )

    # 预警二值评估默认阈值（与 y_score 量纲一致时再调）
    alert_score_threshold: float = 0.5

    baselines: Tuple[BaselineKind, ...] = field(
        default_factory=lambda: (
            BaselineKind.TRADITIONAL,
            BaselineKind.SINGLE_DEEP,
            BaselineKind.THREE_LAYER,
        )
    )

    def __post_init__(self) -> None:
        r = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(r - 1.0) > 1e-6:
            raise ValueError(f"train/val/test 比例之和应为 1，当前为 {r}")

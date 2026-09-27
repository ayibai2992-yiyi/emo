"""训练模块"""

from .losses import (
    UnifiedLoss,
    EmotionClassificationLoss,
    CrisisDetectionLoss,
    CausalConsistencyLoss,
    MetaLearningLoss,
    GraphRegularizationLoss,
)

__all__ = [
    'UnifiedLoss',
    'EmotionClassificationLoss',
    'CrisisDetectionLoss',
    'CausalConsistencyLoss',
    'MetaLearningLoss',
    'GraphRegularizationLoss',
]

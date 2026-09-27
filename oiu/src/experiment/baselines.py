"""
基线说明（与 BaselineKind 对齐，供论文「对比方法」小节直接引用）。
"""

from __future__ import annotations

from typing import Dict

from src.experiment.protocol import BaselineKind

BASELINE_REGISTRY: Dict[BaselineKind, Dict[str, str]] = {
    BaselineKind.TRADITIONAL: {
        "name_zh": "传统浅层基线",
        "description": "例如 TF-IDF/词袋 + 逻辑回归或线性 SVM，作为低成本对照。",
        "inputs": "单条文本（可选拼接统计特征）",
    },
    BaselineKind.SINGLE_DEEP: {
        "name_zh": "单一深度基线",
        "description": "同一预训练中文 BERT 编码器 + 线性分类头，端到端 8 类情绪分类。",
        "inputs": "单条文本",
    },
    BaselineKind.THREE_LAYER: {
        "name_zh": "三层监测系统（本文方法）",
        "description": "轻量筛选 → 深度个性化（统一模型/启发式降级）→ 危机预警链路。",
        "inputs": "当前句 + 用户 ID + 对话历史",
    },
}


def baseline_table_rows() -> Dict[str, Dict[str, str]]:
    """返回 {kind.value: {...}} 便于导出 CSV/Markdown。"""
    return {k.value: v for k, v in BASELINE_REGISTRY.items()}

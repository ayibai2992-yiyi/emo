"""
统一模型内部消融配置（论文消融表）。

约定（推理时）：
- ``use_cpeb``：为 False 时融合第 1 路使用 ``emotion_raw``，不再使用因果去偏后的分布。
- ``use_mea``：为 False 时 **不调用** 元学习适配器，第 2 路替换为均匀分布（无信息先验）。
- ``use_thegn``：为 False 时 **不调用** 图网络，第 3 路为全零向量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModuleAblation:
    use_cpeb: bool = True
    use_mea: bool = True
    use_thegn: bool = True

    @staticmethod
    def full() -> "ModuleAblation":
        return ModuleAblation(True, True, True)

    @staticmethod
    def no_cpeb() -> "ModuleAblation":
        return ModuleAblation(False, True, True)

    @staticmethod
    def no_mea() -> "ModuleAblation":
        return ModuleAblation(True, False, True)

    @staticmethod
    def no_thegn() -> "ModuleAblation":
        return ModuleAblation(True, True, False)

    @staticmethod
    def no_cpeb_no_mea_no_thegn() -> "ModuleAblation":
        """三路外部信号均关闭，仅依赖融合层对固定先验与 raw 的纠偏（极端对照）。"""
        return ModuleAblation(False, False, False)


@dataclass
class PipelineAblation:
    """系统级：第一层快筛是否参与路由与计时。"""

    use_layer1: bool = True

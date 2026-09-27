"""
第二层：深度个性化分析器
优先调用 UnifiedEmotionModel；不可用时显式报错（不再静默关键词模拟充当「我们的方法」）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UserBaseline:
    """用户个性化基线（展示/阈值偏移用）"""
    user_id: str
    mean_emotion: np.ndarray
    std_emotion: np.ndarray
    expression_style: str
    history_count: int
    last_update: str
    risk_threshold: float


class PersonalizedDeepAnalyzer:
    """
    深度个性化分析器。

    真实推理路径：UnifiedEmotionModel.predict
    heuristic 仅在 explicitly allow_heuristic=True 时启用（调试用，不可作论文主结果）。
    """

    def __init__(
        self,
        cpeb_model=None,
        meta_model=None,
        graph_model=None,
        device: str = "cpu",
        unified_model=None,
        user_to_idx: Optional[Dict[str, int]] = None,
        allow_heuristic: bool = False,
        crisis_threshold: float = 0.5,
    ):
        self.cpeb_model = cpeb_model
        self.meta_model = meta_model
        self.graph_model = graph_model
        self.device = device
        self.unified_model = unified_model
        self.user_to_idx = user_to_idx or {}
        self.allow_heuristic = allow_heuristic
        self.crisis_threshold = float(crisis_threshold)

        self.user_baselines: Dict[str, UserBaseline] = {}
        self.default_thresholds = {
            "green": 3.0,
            "blue": 5.0,
            "orange": 7.0,
            "red": 9.0,
        }
        self.emotion_labels = ["中性", "高兴", "惊讶", "悲伤", "愤怒", "恐惧", "厌恶", "绝望"]

    def set_unified_model(self, model, user_to_idx: Optional[Dict[str, int]] = None):
        self.unified_model = model
        if user_to_idx is not None:
            self.user_to_idx = user_to_idx

    def set_crisis_threshold(self, threshold: float):
        self.crisis_threshold = float(threshold)

    def load_user_baseline(self, user_id: str) -> Optional[UserBaseline]:
        return self.user_baselines.get(user_id)

    def _resolve_user_idx(self, user_id: str) -> int:
        if user_id in self.user_to_idx:
            return int(self.user_to_idx[user_id])
        # 尝试纯数字
        try:
            return int(user_id)
        except Exception:
            return 0

    def calculate_dynamic_threshold(
        self,
        user_id: str,
        current_emotion: np.ndarray,
    ) -> Dict[str, float]:
        """
        基于用户基线与当前情绪偏离的阈值偏移。
        weighted_deviation 实际参与调整（修复死代码）。
        """
        baseline = self.load_user_baseline(user_id)
        if baseline is None:
            return dict(self.default_thresholds)

        deviation = (current_emotion - baseline.mean_emotion) / (baseline.std_emotion + 1e-6)
        negative_deviation = deviation[[3, 4, 5, 7]]
        weights = np.array([2.0, 1.5, 2.5, 3.0])
        weighted_deviation = float(np.average(negative_deviation, weights=weights))

        style_factor = 0.7 if baseline.expression_style == "reserved" else 1.3
        # 负向偏离越大，阈值略降（更敏感）
        deviation_factor = float(np.clip(1.0 - 0.05 * weighted_deviation, 0.7, 1.3))

        adjusted = {
            "green": baseline.risk_threshold * 0.3 * style_factor * deviation_factor,
            "blue": baseline.risk_threshold * 0.5 * style_factor * deviation_factor,
            "orange": baseline.risk_threshold * 0.7 * style_factor * deviation_factor,
            "red": baseline.risk_threshold * 0.9 * style_factor * deviation_factor,
        }
        confidence = min(baseline.history_count / 50, 1.0)
        return {
            k: confidence * adjusted[k] + (1 - confidence) * self.default_thresholds[k]
            for k in adjusted
        }

    def calculate_risk_score(
        self,
        emotion_vector: np.ndarray,
        crisis_prob: Optional[float] = None,
    ) -> float:
        """
        风险分 0-10。
        优先使用模型危机头概率；情绪加权仅作辅助且避免结构性饱和。
        """
        if crisis_prob is not None:
            return float(np.clip(crisis_prob * 10.0, 0.0, 10.0))

        sadness = float(emotion_vector[3])
        anger = float(emotion_vector[4])
        fear = float(emotion_vector[5])
        despair = float(emotion_vector[7])
        # 权重和归一，避免 *10 后轻易顶满
        risk = (
            0.20 * sadness
            + 0.15 * anger
            + 0.25 * fear
            + 0.40 * despair
        )
        return float(np.clip(risk * 10.0, 0.0, 10.0))

    def _analyze_unified(
        self,
        text: str,
        user_id: str,
        conversation_history: List[str],
        layer2_threshold_factor: float = 1.0,
    ) -> Dict[str, Any]:
        start = time.time()
        uid = self._resolve_user_idx(user_id)
        result = self.unified_model.predict(
            text=text,
            user_id=uid,
            conversation_history=conversation_history or None,
            support_examples=None,
        )
        emotion_final = np.array(
            [result["emotion_distribution"][lab] for lab in self.emotion_labels],
            dtype=float,
        )
        crisis_prob = float(result.get("crisis_prob", emotion_final[7]))
        risk_score = self.calculate_risk_score(emotion_final, crisis_prob=crisis_prob)

        thresholds = self.calculate_dynamic_threshold(user_id, emotion_final)
        thresholds = {k: v * layer2_threshold_factor for k, v in thresholds.items()}

        # 危机判定：以校准后的 crisis_threshold 为主，risk_level 供系统展示
        need_alert = crisis_prob >= self.crisis_threshold
        if risk_score >= thresholds["red"] or need_alert:
            risk_level, risk_label = "red", "危机"
        elif risk_score >= thresholds["orange"]:
            risk_level, risk_label = "orange", "警告"
        elif risk_score >= thresholds["blue"]:
            risk_level, risk_label = "blue", "关注"
        else:
            risk_level, risk_label = "green", "正常"

        return {
            "user_id": user_id,
            "text": text,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_label": risk_label,
            "emotion_vector": emotion_final.tolist(),
            "emotion_distribution": {
                lab: float(emotion_final[i]) for i, lab in enumerate(self.emotion_labels)
            },
            "crisis_prob": crisis_prob,
            "fusion_weights": result.get("fusion_weights"),
            "graph_mode": result.get("graph_mode"),
            "thresholds": thresholds,
            "attention_weights": {},
            "latency_ms": (time.time() - start) * 1000.0,
            "need_alert": bool(need_alert or risk_level == "red"),
            "backend": "unified_model",
            "intermediate_results": result.get("intermediate_results"),
        }

    def _analyze_heuristic(
        self,
        text: str,
        user_id: str,
        conversation_history: List[str],
        layer2_threshold_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """仅调试用关键词路径，backend 标记为 heuristic。"""
        start = time.time()
        emotion = np.zeros(8, dtype=float)
        emotion[0] = 0.4
        if any(kw in text for kw in ["高兴", "开心", "快乐", "不错"]):
            emotion[1] = 0.5
        if any(kw in text for kw in ["悲伤", "难过", "伤心", "痛苦"]):
            emotion[3] = 0.5
        if any(kw in text for kw in ["愤怒", "生气", "气愤", "烦"]):
            emotion[4] = 0.4
        if any(kw in text for kw in ["恐惧", "害怕", "担心", "不安"]):
            emotion[5] = 0.4
        if any(kw in text for kw in ["绝望", "崩溃", "放弃", "自杀", "不想活"]):
            emotion[7] = 0.7
        emotion = emotion / max(emotion.sum(), 1e-8)

        crisis_prob = float(emotion[7])
        risk_score = self.calculate_risk_score(emotion, crisis_prob=crisis_prob)
        thresholds = {
            k: v * layer2_threshold_factor
            for k, v in self.calculate_dynamic_threshold(user_id, emotion).items()
        }
        need_alert = crisis_prob >= self.crisis_threshold
        if risk_score >= thresholds["red"] or need_alert:
            risk_level, risk_label = "red", "危机"
        elif risk_score >= thresholds["orange"]:
            risk_level, risk_label = "orange", "警告"
        elif risk_score >= thresholds["blue"]:
            risk_level, risk_label = "blue", "关注"
        else:
            risk_level, risk_label = "green", "正常"

        return {
            "user_id": user_id,
            "text": text,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_label": risk_label,
            "emotion_vector": emotion.tolist(),
            "emotion_distribution": {
                lab: float(emotion[i]) for i, lab in enumerate(self.emotion_labels)
            },
            "crisis_prob": crisis_prob,
            "fusion_weights": None,
            "graph_mode": "heuristic",
            "thresholds": thresholds,
            "attention_weights": {
                "temporal": [0.3] * min(len(conversation_history), 5),
            },
            "latency_ms": (time.time() - start) * 1000.0,
            "need_alert": bool(need_alert or risk_level == "red"),
            "backend": "heuristic",
            "intermediate_results": None,
        }

    def analyze(
        self,
        text: str,
        user_id: str,
        conversation_history: Optional[List[str]] = None,
        module_ablation=None,
        layer2_threshold_factor: float = 1.0,
    ) -> Dict[str, Any]:
        history = list(conversation_history or [])
        # module_ablation 预留：当前完整模型路径；重训消融在实验脚本中完成
        _ = module_ablation

        if self.unified_model is not None:
            return self._analyze_unified(
                text, user_id, history, layer2_threshold_factor=layer2_threshold_factor
            )
        if self.allow_heuristic:
            return self._analyze_heuristic(
                text, user_id, history, layer2_threshold_factor=layer2_threshold_factor
            )
        raise RuntimeError(
            "PersonalizedDeepAnalyzer 未接入 UnifiedEmotionModel，且未允许 heuristic。"
            "请先训练并加载 models/unified_emotion_model.pt。"
        )


if __name__ == "__main__":
    print("deep_analyzer 模块加载成功（需外部注入 unified_model 才能推理）")

"""
同步情绪监测流水线：第一层轻量筛选 + 第二层 UnifiedEmotionModel 深度分析。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.ablation import ModuleAblation, PipelineAblation
from src.monitoring.deep_analyzer import PersonalizedDeepAnalyzer, UserBaseline
from src.monitoring.lightweight_filter import LightweightEmotionFilter
from src.utils.config import get_default_config


def _default_baselines() -> Dict[str, UserBaseline]:
    return {
        "user001": UserBaseline(
            user_id="user001",
            mean_emotion=np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]),
            std_emotion=np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02]),
            expression_style="reserved",
            history_count=50,
            last_update="2024-01-01",
            risk_threshold=7.0,
        ),
        "user002": UserBaseline(
            user_id="user002",
            mean_emotion=np.array([0.2, 0.15, 0.1, 0.15, 0.15, 0.1, 0.1, 0.05]),
            std_emotion=np.array([0.15, 0.15, 0.1, 0.1, 0.1, 0.08, 0.05, 0.03]),
            expression_style="expressive",
            history_count=80,
            last_update="2024-01-01",
            risk_threshold=8.5,
        ),
    }


class MonitoringPipeline:
    """轻量筛选 + 统一模型深度分析。"""

    def __init__(
        self,
        device: Optional[str] = None,
        allow_heuristic: bool = False,
        force_deep: bool = False,
        model_dir: Optional[str] = None,
    ):
        cfg = get_default_config()
        if model_dir:
            cfg.model_dir = model_dir
        self.model_dir = cfg.model_dir
        self.device = device or cfg.device
        self.force_deep = force_deep
        try:
            import torch

            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
        except Exception:
            self.device = "cpu"

        self.filter = LightweightEmotionFilter(
            model_name=cfg.monitoring.layer1_model,
            device=self.device,
            deep_threshold=cfg.monitoring.layer1_threshold,
        )
        self.unified_model = self._try_build_unified_model(cfg)
        user_to_idx = self._load_user_to_idx(cfg)
        crisis_thr = self._load_crisis_threshold(cfg)

        self.analyzer = PersonalizedDeepAnalyzer(
            device=self.device,
            unified_model=self.unified_model,
            user_to_idx=user_to_idx,
            allow_heuristic=allow_heuristic and self.unified_model is None,
            crisis_threshold=crisis_thr,
        )
        self.analyzer.user_baselines.update(_default_baselines())
        self.preferred_model_backend = (
            "unified_model" if self.unified_model is not None else "heuristic"
        )

    def _load_user_to_idx(self, cfg) -> Dict[str, int]:
        path = Path(cfg.model_dir) / "user_to_idx.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {str(k): int(v) for k, v in raw.items()}
        return {}

    def _load_crisis_threshold(self, cfg) -> float:
        path = Path(cfg.model_dir) / "crisis_threshold.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return float(raw.get("threshold", 0.5))
        return 0.5

    def _try_build_unified_model(self, cfg):
        try:
            import torch
            from src.models.unified_model import UnifiedEmotionModel
        except Exception as e:
            print(f"[MonitoringPipeline] unified model import failed: {e}")
            return None

        try:
            model = UnifiedEmotionModel(
                bert_model_name=cfg.model.bert_model,
                num_emotions=cfg.model.num_emotion_labels,
                hidden_size=cfg.model.hidden_size,
                graph_hidden_size=cfg.model.graph_hidden_size,
                num_graph_layers=cfg.model.num_graph_layers,
                num_attention_heads=cfg.model.num_attention_heads,
                meta_adapter_type=cfg.model.meta_learning_algorithm,
                inner_lr=cfg.model.inner_lr,
                num_inner_steps=cfg.model.num_inner_steps,
                temporal_decay_lambda=cfg.model.temporal_decay_lambda,
                dropout=cfg.model.dropout,
                device=self.device,
                proto_max_shots=cfg.model.proto_max_shots,
                maml_full_shots=cfg.model.maml_full_shots,
                despair_index=cfg.model.despair_index,
            )
            model.to(self.device)
            model.eval()
        except Exception as e:
            print(f"[MonitoringPipeline] unified model build failed: {e}")
            return None

        weight_path = Path(cfg.model_dir) / "unified_emotion_model.pt"
        if weight_path.exists():
            try:
                state = torch.load(weight_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                model.eval()
                print(f"[MonitoringPipeline] loaded unified weights: {weight_path}")
            except Exception as e:
                print(f"[MonitoringPipeline] load unified weights failed: {e}")
        else:
            print(
                f"[MonitoringPipeline] WARNING: 未找到权重 {weight_path}，"
                "模型为随机初始化，论文评估前请先运行 experiments/train_unified_model.py"
            )

        try:
            _ = model.predict(
                text="系统自检",
                user_id=0,
                conversation_history=["测试历史1", "测试历史2"],
                support_examples=None,
            )
        except Exception as e:
            print(f"[MonitoringPipeline] unified model warmup failed: {e}")
            return None

        return model

    def analyze_text(
        self,
        text: str,
        user_id: str = "user001",
        conversation_history: Optional[List[str]] = None,
        module_ablation: Optional[ModuleAblation] = None,
        pipeline_ablation: Optional[PipelineAblation] = None,
        layer1_threshold: Optional[float] = None,
        layer2_threshold_factor: float = 1.0,
    ) -> Dict[str, Any]:
        history = list(conversation_history or [])
        pa = pipeline_ablation if pipeline_ablation is not None else PipelineAblation()

        if pa.use_layer1 and not self.force_deep:
            layer1 = self.filter.predict(
                text, user_id, deep_threshold_override=layer1_threshold
            )
        else:
            t0 = time.perf_counter()
            layer1 = {
                "user_id": user_id,
                "text": text,
                "score": 10.0,
                "level": "force_deep" if self.force_deep else "ablation_no_layer1",
                "need_deep_analysis": True,
                "is_extreme": False,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "components": {
                    "keyword_score": 0.0,
                    "pattern_score": 0.0,
                    "bert_score": 0.0,
                    "note": "layer1_skipped",
                },
            }

        out: Dict[str, Any] = {
            "user_id": user_id,
            "text": text,
            "conversation_history": history,
            "layer1": layer1,
            "need_deep_analysis": layer1["need_deep_analysis"],
            "preferred_model_backend": self.preferred_model_backend,
            "ablation": {
                "pipeline_use_layer1": pa.use_layer1,
                "module": (
                    {
                        "use_cpeb": module_ablation.use_cpeb,
                        "use_mea": module_ablation.use_mea,
                        "use_thegn": module_ablation.use_thegn,
                    }
                    if module_ablation is not None
                    else None
                ),
            },
            "overrides": {
                "layer1_threshold": layer1_threshold,
                "layer2_threshold_factor": float(layer2_threshold_factor),
            },
        }

        if layer1["need_deep_analysis"]:
            layer2 = self.analyzer.analyze(
                text,
                user_id,
                history,
                module_ablation=module_ablation,
                layer2_threshold_factor=layer2_threshold_factor,
            )
            out["layer2"] = layer2
            out["model_backend"] = layer2.get("backend", "heuristic")
            out["final"] = {
                "risk_score": layer2["risk_score"],
                "risk_level": layer2["risk_level"],
                "risk_label": layer2["risk_label"],
                "need_alert": layer2["need_alert"],
                "crisis_prob": layer2.get("crisis_prob"),
                "latency_ms_layer1": layer1["latency_ms"],
                "latency_ms_layer2": layer2["latency_ms"],
            }
        else:
            score = float(layer1["score"])
            out["layer2"] = None
            out["model_backend"] = "layer1_only"
            out["final"] = {
                "risk_score": score,
                "risk_level": "green" if score < 5 else "blue",
                "risk_label": "正常" if score < 5 else "关注",
                "need_alert": False,
                "crisis_prob": None,
                "latency_ms_layer1": layer1["latency_ms"],
                "latency_ms_layer2": 0.0,
            }

        return out

    def user_baselines_summary(self) -> List[Dict[str, Any]]:
        rows = []
        for _, b in self.analyzer.user_baselines.items():
            rows.append(
                {
                    "user_id": b.user_id,
                    "mean_emotion": np.asarray(b.mean_emotion).tolist(),
                    "std_emotion": np.asarray(b.std_emotion).tolist(),
                    "expression_style": b.expression_style,
                    "history_count": b.history_count,
                    "last_update": b.last_update,
                    "risk_threshold": b.risk_threshold,
                }
            )
        return rows

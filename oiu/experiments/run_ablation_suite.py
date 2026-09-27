#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验脚本：统一模型各模块 + 第一层快筛，输出延迟与（可选）标签准确率占位。

用法（在 oiu 目录）:
  python experiments/run_ablation_suite.py

若需与标注对比，请自行加载 test 集，对每条样本调用
``UnifiedEmotionModel.predict(..., ablation=ModuleAblation(...))`` 后接 ``compute_emotion_metrics``。
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _bench_predict(model, cases: List[Tuple[str, int, List[str]]], ablation) -> Dict[str, Any]:
    import torch

    if torch.cuda.is_available() and getattr(model, "device", "cpu") == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for text, uid, hist in cases:
        model.predict(text=text, user_id=uid, conversation_history=hist, ablation=ablation)
    if torch.cuda.is_available() and getattr(model, "device", "cpu") == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {"total_ms": elapsed_ms, "per_sample_ms": elapsed_ms / max(len(cases), 1)}


def main():
    from src.models.ablation import ModuleAblation, PipelineAblation
    from src.models.unified_model import UnifiedEmotionModel
    from src.service.monitoring_pipeline import MonitoringPipeline

    cases = [
        ("今天心情还可以", 1, []),
        ("压力好大，有点崩溃", 1, ["昨晚失眠", "作业很多"]),
        ("我不想活了", 2, ["很难受", "一直哭"]),
    ]

    rows = []

    # ---------- 统一模型内部消融 ----------
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = UnifiedEmotionModel(device=device)
        model.to(device)
        model.eval()
    except Exception as e:
        print(json.dumps({"error": f"unified_model_init_failed: {e}"}, ensure_ascii=False))
        return

    ablations: List[Tuple[str, ModuleAblation]] = [
        ("full", ModuleAblation.full()),
        ("no_cpeb", ModuleAblation.no_cpeb()),
        ("no_mea", ModuleAblation.no_mea()),
        ("no_thegn", ModuleAblation.no_thegn()),
        (
            "no_cpeb_no_mea_no_thegn",
            ModuleAblation(False, False, False),
        ),
    ]

    for name, ab in ablations:
        st = _bench_predict(model, cases, ab)
        rows.append(
            {
                "suite": "unified_model",
                "variant": name,
                "ablation": {"use_cpeb": ab.use_cpeb, "use_mea": ab.use_mea, "use_thegn": ab.use_thegn},
                "latency_total_ms": round(st["total_ms"], 3),
                "latency_per_sample_ms": round(st["per_sample_ms"], 3),
                "num_samples": len(cases),
            }
        )

    # ---------- 系统流水线：第一层快筛开/关 ----------
    try:
        pipe = MonitoringPipeline(device="cpu")
        for use_l1 in (True, False):
            t0 = time.perf_counter()
            for text, _, hist in cases:
                uid = "user001"
                pipe.analyze_text(
                    text,
                    user_id=uid,
                    conversation_history=hist,
                    module_ablation=None,
                    pipeline_ablation=PipelineAblation(use_layer1=use_l1),
                )
            elapsed = (time.perf_counter() - t0) * 1000.0
            rows.append(
                {
                    "suite": "monitoring_pipeline",
                    "variant": "layer1_on" if use_l1 else "layer1_off_ablation",
                    "pipeline_use_layer1": use_l1,
                    "latency_total_ms": round(elapsed, 3),
                    "latency_per_sample_ms": round(elapsed / len(cases), 3),
                    "num_samples": len(cases),
                }
            )
    except Exception as e:
        rows.append({"suite": "monitoring_pipeline", "error": str(e)})

    print(json.dumps({"ablation_runs": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

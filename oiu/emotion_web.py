#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪监测 · 论文展示面板。
在 `oiu` 目录下:  python emotion_web.py  →  http://127.0.0.1:8765

API:
  GET  /api/config           默认第一层阈值等
  POST /api/analyze          支持 layer1_threshold、layer2_threshold_factor、pipeline_ablation、module_ablation
  POST /api/batch_analyze    JSON 批量（items 数组）
  POST /api/batch_csv        CSV 批量
  GET  /api/baselines
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from flask import Flask, jsonify, render_template, request

from src.models.ablation import ModuleAblation, PipelineAblation
from src.service.monitoring_pipeline import MonitoringPipeline
from src.utils.config import get_default_config

app = Flask(__name__, template_folder=str(ROOT / "templates"))
_pipeline: Optional[MonitoringPipeline] = None


def get_pipeline() -> MonitoringPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MonitoringPipeline()
    return _pipeline


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (float, int, str)) or obj is None:
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


@app.route("/")
def index():
    return render_template("emotion_dashboard.html")


@app.get("/api/baselines")
def api_baselines():
    pipe = get_pipeline()
    return jsonify({"ok": True, "users": json_safe(pipe.user_baselines_summary())})


@app.get("/api/config")
def api_config():
    cfg = get_default_config()
    return jsonify(
        {
            "ok": True,
            "layer1_threshold_default": float(cfg.monitoring.layer1_threshold),
            "layer1_model": cfg.monitoring.layer1_model,
            "layer2_threshold_factor_default": 1.0,
        }
    )


@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "text 不能为空"}), 400

    user_id = (data.get("user_id") or "user001").strip()
    history = data.get("conversation_history")
    if history is None:
        raw_hist = data.get("history_text")
        if isinstance(raw_hist, str) and raw_hist.strip():
            history = [ln.strip() for ln in raw_hist.splitlines() if ln.strip()]
        else:
            history = []
    if not isinstance(history, list):
        return jsonify({"ok": False, "error": "conversation_history 必须是字符串数组"}), 400

    history = [str(h).strip() for h in history if str(h).strip()]

    module_ablation = None
    if isinstance(data.get("module_ablation"), dict):
        ma = data["module_ablation"]
        module_ablation = ModuleAblation(
            use_cpeb=bool(ma.get("use_cpeb", True)),
            use_mea=bool(ma.get("use_mea", True)),
            use_thegn=bool(ma.get("use_thegn", True)),
        )

    pipeline_ablation = None
    if isinstance(data.get("pipeline_ablation"), dict):
        pa = data["pipeline_ablation"]
        pipeline_ablation = PipelineAblation(use_layer1=bool(pa.get("use_layer1", True)))

    layer1_threshold = data.get("layer1_threshold")
    if layer1_threshold is not None:
        try:
            layer1_threshold = float(layer1_threshold)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "layer1_threshold 必须为数字"}), 400

    try:
        layer2_threshold_factor = float(data.get("layer2_threshold_factor", 1.0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "layer2_threshold_factor 必须为数字"}), 400

    pipe = get_pipeline()
    result = pipe.analyze_text(
        text,
        user_id=user_id,
        conversation_history=history,
        module_ablation=module_ablation,
        pipeline_ablation=pipeline_ablation,
        layer1_threshold=layer1_threshold,
        layer2_threshold_factor=layer2_threshold_factor,
    )
    return jsonify({"ok": True, "result": json_safe(result)})


@app.post("/api/batch_analyze")
def api_batch_analyze():
    """JSON 批量：{ \"items\": [ { \"text\", \"user_id\", \"conversation_history\": [] } ], ... }"""
    data = request.get_json(silent=True) or {}
    items = data.get("items")
    if not isinstance(items, list) or len(items) == 0:
        return jsonify({"ok": False, "error": "items 必须为非空数组"}), 400

    layer1_threshold = data.get("layer1_threshold")
    if layer1_threshold is not None:
        try:
            layer1_threshold = float(layer1_threshold)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "layer1_threshold 必须为数字"}), 400

    try:
        layer2_threshold_factor = float(data.get("layer2_threshold_factor", 1.0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "layer2_threshold_factor 必须为数字"}), 400

    module_ablation = None
    if isinstance(data.get("module_ablation"), dict):
        ma = data["module_ablation"]
        module_ablation = ModuleAblation(
            use_cpeb=bool(ma.get("use_cpeb", True)),
            use_mea=bool(ma.get("use_mea", True)),
            use_thegn=bool(ma.get("use_thegn", True)),
        )

    pipeline_ablation = None
    if isinstance(data.get("pipeline_ablation"), dict):
        pa = data["pipeline_ablation"]
        pipeline_ablation = PipelineAblation(use_layer1=bool(pa.get("use_layer1", True)))

    pipe = get_pipeline()
    results = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            return jsonify({"ok": False, "error": f"items[{i}] 必须为对象"}), 400
        text = (it.get("text") or "").strip()
        if not text:
            continue
        uid = (it.get("user_id") or "user001").strip()
        hist = it.get("conversation_history")
        if hist is None:
            hist = []
        if not isinstance(hist, list):
            return jsonify({"ok": False, "error": f"items[{i}].conversation_history 须为数组"}), 400
        hist = [str(h).strip() for h in hist if str(h).strip()]
        results.append(
            pipe.analyze_text(
                text,
                user_id=uid,
                conversation_history=hist,
                module_ablation=module_ablation,
                pipeline_ablation=pipeline_ablation,
                layer1_threshold=layer1_threshold,
                layer2_threshold_factor=layer2_threshold_factor,
            )
        )

    return jsonify({"ok": True, "count": len(results), "results": json_safe(results)})


@app.post("/api/batch_csv")
def api_batch_csv():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "请上传 file 字段（CSV）"}), 400
    f = request.files["file"]
    raw = f.read()
    try:
        text_io = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="")
    except Exception:
        text_io = io.StringIO(raw.decode("utf-8", errors="replace"))

    reader = csv.DictReader(text_io)
    if not reader.fieldnames or "text" not in [c.strip().lower() for c in reader.fieldnames]:
        return jsonify({"ok": False, "error": "CSV 必须包含 text 列"}), 400

    def norm_key(k: str) -> str:
        return (k or "").strip().lower()

    fieldmap = {norm_key(c): c for c in reader.fieldnames}

    def row_get(row: Dict[str, str], *names: str) -> str:
        for n in names:
            k = fieldmap.get(n.lower())
            if k and row.get(k) is not None:
                return str(row.get(k) or "")
        return ""

    rows_out: List[Dict[str, Any]] = []
    pipe = get_pipeline()
    for row in reader:
        text = row_get(row, "text", "content", "message").strip()
        if not text:
            continue
        uid = row_get(row, "user_id", "userid", "user").strip() or "user001"
        hist_cell = row_get(row, "history", "conversation_history", "context")
        history: List[str] = []
        if hist_cell:
            for sep in ["\n", "|", "；", ";"]:
                if sep in hist_cell:
                    history = [p.strip() for p in hist_cell.replace("；", ";").split(sep) if p.strip()]
                    break
            if not history:
                history = [hist_cell.strip()]

        rows_out.append(pipe.analyze_text(text, user_id=uid, conversation_history=history))

    return jsonify({"ok": True, "count": len(rows_out), "results": json_safe(rows_out)})


if __name__ == "__main__":
    print("论文展示面板: http://127.0.0.1:8765")
    print("  GET  /api/config  POST /api/batch_analyze  …")
    app.run(host="0.0.0.0", port=8765, debug=False)

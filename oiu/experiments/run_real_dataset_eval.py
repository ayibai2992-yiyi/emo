#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据评估脚本：
- 读取 CSV（至少包含 user_id/text/label）
- 按用户划分 train/val/test，防止同用户跨集合泄漏
- 使用 MonitoringPipeline 逐条推理
- 在验证集上自动选择危机阈值（best F1）
- 在测试集上输出固定指标

在 oiu 目录下运行示例：
  python experiments/run_real_dataset_eval.py --csv data/processed/train.csv
  python experiments/run_real_dataset_eval.py --csv data/processed/train.csv --text-col content --label-col emotion_id
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.experiment.metrics_protocol import (
    best_f1_threshold,
    compute_crisis_detection_metrics,
    compute_emotion_metrics,
    labels_to_crisis_binary,
)
from src.experiment.protocol import ExperimentProtocol
from src.experiment.splits import summarize_split, user_stratified_masks
from src.service.monitoring_pipeline import MonitoringPipeline


EMOTION_LABELS: Tuple[str, ...] = (
    "中性",
    "高兴",
    "惊讶",
    "悲伤",
    "愤怒",
    "恐惧",
    "厌恶",
    "绝望",
)


def _pick_col(df: pd.DataFrame, preferred: str, candidates: List[str], required: bool = True) -> str:
    cols = list(df.columns)
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"未找到列。可选列名尝试: {candidates}，当前列: {cols}")
    return ""


def _safe_int_label(v) -> int:
    iv = int(v)
    if iv < 0 or iv > 7:
        raise ValueError(f"label 必须在 [0,7]，当前: {iv}")
    return iv


def _split_history(raw: str, sep: str) -> List[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    if sep == "\\n":
        sep = "\n"
    return [x.strip() for x in raw.split(sep) if x.strip()]


def _emotion_pred_from_layer2(layer2: Dict, fallback_risk: float = 0.0) -> Tuple[int, float]:
    """从 Layer2 提取情绪预测与危机分数；优先 crisis_prob。"""
    if not isinstance(layer2, dict) or not layer2:
        return 0, float(np.clip(fallback_risk / 10.0, 0.0, 1.0))

    if "crisis_prob" in layer2 and layer2["crisis_prob"] is not None:
        crisis_score = float(np.clip(layer2["crisis_prob"], 0.0, 1.0))
    else:
        crisis_score = None

    vec = layer2.get("emotion_vector")
    if isinstance(vec, list) and len(vec) == 8:
        arr = np.array(vec, dtype=float)
        idx = int(arr.argmax())
        if crisis_score is None:
            crisis_score = float(np.clip(arr[7], 0.0, 1.0))
        return idx, float(crisis_score)

    if crisis_score is not None:
        return 0, crisis_score
    return 0, float(np.clip(fallback_risk / 10.0, 0.0, 1.0))



def run_eval(args: argparse.Namespace) -> Dict:
    df = pd.read_csv(args.csv, encoding=args.encoding)
    if len(df) == 0:
        raise ValueError("CSV 为空，无法评估。")

    user_col = _pick_col(df, args.user_col, ["user_id", "uid", "user", "userid"])
    text_col = _pick_col(df, args.text_col, ["text", "content", "sentence", "utterance"])
    label_col = _pick_col(df, args.label_col, ["label", "emotion_id", "emotion_label", "y"])
    crisis_col = _pick_col(
        df,
        args.crisis_col,
        ["is_crisis", "crisis", "label_crisis", "y_crisis"],
        required=False,
    )

    history_col = ""
    if args.history_col:
        if args.history_col not in df.columns:
            raise ValueError(f"--history-col 指定列不存在: {args.history_col}")
        history_col = args.history_col
    else:
        for c in ("conversation_history", "history", "context"):
            if c in df.columns:
                history_col = c
                break

    # 基础清洗
    data = df[[user_col, text_col, label_col] + ([crisis_col] if crisis_col else []) + ([history_col] if history_col else [])].copy()
    data = data.dropna(subset=[user_col, text_col, label_col])
    data[user_col] = data[user_col].astype(str)
    data[text_col] = data[text_col].astype(str)
    data[label_col] = data[label_col].apply(_safe_int_label)
    data = data[data[text_col].str.strip() != ""].reset_index(drop=True)
    if len(data) < 10:
        raise ValueError(f"有效样本太少（{len(data)}），建议至少 >= 10。")

    proto = ExperimentProtocol(seed=args.seed)
    user_ids = data[user_col].to_numpy()
    y_true_all = data[label_col].to_numpy(dtype=int)

    train_m, val_m, test_m = user_stratified_masks(
        user_ids=user_ids,
        train_ratio=proto.train_ratio,
        val_ratio=proto.val_ratio,
        test_ratio=proto.test_ratio,
        seed=proto.seed,
    )

    pipeline = MonitoringPipeline(device=args.device, force_deep=args.force_deep)

    y_pred = np.zeros(len(data), dtype=int)
    y_score_crisis = np.zeros(len(data), dtype=float)
    backends: List[str] = []

    for i in range(len(data)):
        text = data.at[i, text_col]
        uid = data.at[i, user_col]
        history = _split_history(data.at[i, history_col], args.history_sep) if history_col else []

        out = pipeline.analyze_text(
            text=text,
            user_id=uid,
            conversation_history=history,
            layer2_threshold_factor=args.layer2_threshold_factor,
        )

        final = out.get("final", {})
        layer2 = out.get("layer2") or {}
        pred_id, crisis_score = _emotion_pred_from_layer2(layer2, fallback_risk=float(final.get("risk_score", 0.0)))
        y_pred[i] = pred_id
        y_score_crisis[i] = crisis_score
        backends.append(str(out.get("model_backend", "unknown")))

    # 危机真值：优先使用显式列，否则由标签派生
    if crisis_col:
        y_crisis = data[crisis_col].astype(int).to_numpy()
        y_crisis = np.clip(y_crisis, 0, 1)
    else:
        y_crisis = labels_to_crisis_binary(y_true_all, proto.crisis_emotion_indices)

    # 在验证集选阈值，再在测试集报告
    val_thr, val_best_f1 = best_f1_threshold(y_crisis[val_m], y_score_crisis[val_m])
    if args.fixed_threshold is not None:
        test_threshold = float(args.fixed_threshold)
        threshold_source = "fixed"
    else:
        test_threshold = float(val_thr)
        threshold_source = "val_best_f1"

    emo_test = compute_emotion_metrics(y_true_all[test_m], y_pred[test_m])
    crisis_test = compute_crisis_detection_metrics(y_crisis[test_m], y_score_crisis[test_m], threshold=test_threshold)
    emo_val = compute_emotion_metrics(y_true_all[val_m], y_pred[val_m])
    crisis_val = compute_crisis_detection_metrics(y_crisis[val_m], y_score_crisis[val_m], threshold=test_threshold)

    out = {
        "input": {
            "csv": args.csv,
            "rows_raw": int(len(df)),
            "rows_valid": int(len(data)),
            "columns_used": {
                "user_col": user_col,
                "text_col": text_col,
                "label_col": label_col,
                "crisis_col": crisis_col or None,
                "history_col": history_col or None,
            },
        },
        "protocol": {
            "seed": proto.seed,
            "split": {"train": proto.train_ratio, "val": proto.val_ratio, "test": proto.test_ratio},
            "crisis_emotion_indices": list(proto.crisis_emotion_indices),
        },
        "split_summary": summarize_split(user_ids, train_m, val_m, test_m),
        "backend_stats": {
            "model_backend_counts": {k: int(v) for k, v in pd.Series(backends).value_counts().to_dict().items()},
            "preferred_model_backend": pipeline.preferred_model_backend,
        },
        "threshold_selection": {
            "source": threshold_source,
            "val_best_f1_threshold": float(val_thr),
            "val_best_f1": float(val_best_f1),
            "test_threshold_used": float(test_threshold),
        },
        "val_metrics_emotion": emo_val,
        "val_metrics_crisis": crisis_val,
        "test_metrics_emotion": emo_test,
        "test_metrics_crisis": crisis_test,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="真实数据评估脚本（与 src/experiment 指标协议一致）")
    p.add_argument("--csv", required=True, help="输入 CSV 路径")
    p.add_argument("--encoding", default="utf-8", help="CSV 编码（默认 utf-8）")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    p.add_argument("--user-col", default="", help="用户列名（默认自动识别）")
    p.add_argument("--text-col", default="", help="文本列名（默认自动识别）")
    p.add_argument("--label-col", default="", help="情绪标签列名（0-7，默认自动识别）")
    p.add_argument("--crisis-col", default="", help="危机二值列名（0/1，默认自动识别，可不传）")
    p.add_argument("--history-col", default="", help="历史对话列名（可选）")
    p.add_argument("--history-sep", default="\\n", help="历史对话分隔符，默认换行")

    p.add_argument("--layer2-threshold-factor", type=float, default=1.0, help="第二层阈值缩放因子")
    p.add_argument("--fixed-threshold", type=float, default=None, help="固定危机阈值；不传则使用验证集 best-F1")
    p.add_argument(
        "--force-deep",
        action="store_true",
        help="跳过第一层，强制全部样本走 UnifiedEmotionModel（论文主表推荐）",
    )
    p.add_argument(
        "--output-json",
        default="results/real_eval_metrics.json",
        help="输出 JSON 路径（默认 results/real_eval_metrics.json）",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    result = run_eval(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

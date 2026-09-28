#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对已人工填写的弱标抽检表计算一致率。

  python experiments/score_weak_label_audit.py \\
    --audit-csv ../data/experiment_sets/weak_label_audit_filled.csv \\
    --output-json results/weak_label_audit_score.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from src.experiment.metrics_protocol import compute_crisis_detection_metrics, compute_emotion_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-csv", required=True)
    p.add_argument("--output-json", default="results/weak_label_audit_score.json")
    args = p.parse_args()

    df = pd.read_csv(args.audit_csv)
    for c in ("model_label_id", "human_label_id"):
        if c not in df.columns:
            raise ValueError(f"缺少列 {c}")

    work = df.dropna(subset=["human_label_id"]).copy()
    work["human_label_id"] = work["human_label_id"].astype(int)
    work["model_label_id"] = work["model_label_id"].astype(int)
    if "human_is_crisis" in work.columns:
        work = work.dropna(subset=["human_is_crisis"])
        work["human_is_crisis"] = work["human_is_crisis"].astype(int)
        work["model_is_crisis"] = work.get("model_is_crisis", 0).astype(int)

    y_m = work["model_label_id"].to_numpy()
    y_h = work["human_label_id"].to_numpy()
    acc = float((y_m == y_h).mean()) if len(work) else 0.0
    emo = compute_emotion_metrics(y_h, y_m)

    out = {
        "n_labeled": int(len(work)),
        "n_total_rows": int(len(df)),
        "label_accuracy_model_vs_human": acc,
        "emotion_metrics_treating_human_as_gold": emo,
        "note": "此处把人工标当作金标，衡量弱标与人工一致程度",
    }
    if "human_is_crisis" in work.columns:
        score = work["model_is_crisis"].to_numpy(dtype=float)
        out["crisis_metrics"] = compute_crisis_detection_metrics(
            work["human_is_crisis"].to_numpy(), score, threshold=0.5
        )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

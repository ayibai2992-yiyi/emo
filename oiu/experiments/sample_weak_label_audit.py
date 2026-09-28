#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弱标注人工抽检：从学生 enriched / S1r 中分层抽样，导出待标注表。

用途（EI 答辩/审稿）：
  报告「弱标与人工一致率」，避免把 CPCD 弱标当成金标。

示例：
  python experiments/sample_weak_label_audit.py \\
    --csv ../data/experiment_sets/upload_ready_student_enriched_rebalanced.csv \\
    --n-per-label 40 --output ../data/experiment_sets/weak_label_audit_sample.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EMOTION = ("中性", "高兴", "惊讶", "悲伤", "愤怒", "恐惧", "厌恶", "绝望")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--label-col", default="label_id")
    p.add_argument("--n-per-label", type=int, default=40, help="每类抽样条数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="../data/experiment_sets/weak_label_audit_sample.csv")
    p.add_argument(
        "--guide-json",
        default="../data/experiment_sets/weak_label_audit_guide.json",
        help="标注说明 JSON",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    lab = args.label_col if args.label_col in df.columns else "label"
    parts = []
    for y in range(8):
        sub = df[df[lab].astype(int) == y]
        if len(sub) == 0:
            continue
        take = min(args.n_per_label, len(sub))
        parts.append(sub.sample(n=take, random_state=args.seed + y))
    sample = pd.concat(parts, axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "audit_id": [f"A{i:04d}" for i in range(len(sample))],
            "row_id": sample.get("row_id", pd.Series([""] * len(sample))).astype(str),
            "user_id": sample["user_id"].astype(str),
            "text": sample["text"].astype(str),
            "conversation_history": sample.get(
                "conversation_history", pd.Series([""] * len(sample))
            ).astype(str),
            "model_label_id": sample[lab].astype(int),
            "model_label_name": sample[lab].astype(int).map(lambda i: EMOTION[i] if 0 <= i < 8 else "?"),
            "model_is_crisis": sample.get("is_crisis", 0).astype(int),
            "source_file": sample.get("source_file", "").astype(str),
            "human_label_id": "",  # 人工填写 0-7
            "human_is_crisis": "",  # 人工填写 0/1
            "human_note": "",
        }
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    guide = {
        "purpose": "核对弱标注与人工判断一致性，供论文局限/附录使用",
        "label_legend": {str(i): n for i, n in enumerate(EMOTION)},
        "instructions": [
            "仅根据 text（可参考 history）判断求助者当前情绪，不要被咨询师安慰语带偏",
            "human_label_id 填 0-7；拿不准填最接近类并在 human_note 写不确定",
            "危机：明确自伤/自杀意图或极度绝望无力感标 1，否则 0",
            "完成后运行: python experiments/score_weak_label_audit.py --audit-csv <填好的表>",
        ],
        "sample_size": int(len(out)),
        "n_per_label_target": args.n_per_label,
        "source_csv": args.csv,
        "agreement_formula": "accuracy = mean(human_label_id == model_label_id); crisis_f1 另算",
    }
    guide_path = Path(args.guide_json)
    with open(guide_path, "w", encoding="utf-8") as f:
        json.dump(guide, f, ensure_ascii=False, indent=2)

    print(f"audit sample -> {out_path} (n={len(out)})")
    print(f"guide -> {guide_path}")


if __name__ == "__main__":
    main()

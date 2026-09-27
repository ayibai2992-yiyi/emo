#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练前数据重平衡脚本（针对“中性样本过多、危机样本过少”）。

功能：
1) 对中性类（默认 label=0）做下采样
2) 对危机样本（默认 is_crisis=1）做上采样
3) 输出重平衡后的 CSV 与前后分布统计 JSON

示例：
python experiments/rebalance_dataset.py \
  --input-csv /root/autodl-tmp/upload_ready_5k.csv \
  --output-csv /root/autodl-tmp/upload_ready_5k_rebalanced.csv \
  --label-col label \
  --crisis-col is_crisis \
  --neutral-label 0 \
  --neutral-keep-ratio 0.4 \
  --crisis-upsample-factor 3.0 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class RebalanceConfig:
    neutral_label: int = 0
    neutral_keep_ratio: float = 0.4
    crisis_upsample_factor: float = 3.0
    seed: int = 42


def _validate_ratio(name: str, value: float, low: float, high: float) -> None:
    if value < low or value > high:
        raise ValueError(f"{name} 需在 [{low}, {high}]，当前: {value}")


def _count_distribution(df: pd.DataFrame, label_col: str, crisis_col: str) -> Dict:
    label_dist = df[label_col].value_counts(dropna=False).sort_index().to_dict()
    crisis_dist = df[crisis_col].astype(int).value_counts(dropna=False).sort_index().to_dict()
    return {
        "rows": int(len(df)),
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "crisis_distribution": {str(k): int(v) for k, v in crisis_dist.items()},
    }


def rebalance_dataframe(
    df: pd.DataFrame,
    label_col: str,
    crisis_col: str,
    cfg: RebalanceConfig,
) -> Tuple[pd.DataFrame, Dict]:
    # 1) 中性下采样
    neutral_mask = df[label_col].astype(int) == cfg.neutral_label
    neutral_df = df[neutral_mask]
    other_df = df[~neutral_mask]

    if len(neutral_df) > 0:
        neutral_target = max(1, int(len(neutral_df) * cfg.neutral_keep_ratio))
        neutral_kept = neutral_df.sample(n=neutral_target, replace=False, random_state=cfg.seed)
    else:
        neutral_kept = neutral_df

    step1_df = pd.concat([neutral_kept, other_df], axis=0, ignore_index=True)

    # 2) 危机样本上采样（只扩增 crisis=1）
    crisis_mask = step1_df[crisis_col].astype(int) == 1
    crisis_df = step1_df[crisis_mask]
    non_crisis_df = step1_df[~crisis_mask]

    if len(crisis_df) > 0 and cfg.crisis_upsample_factor > 1.0:
        target_crisis = int(len(crisis_df) * cfg.crisis_upsample_factor)
        extra_n = max(0, target_crisis - len(crisis_df))
        if extra_n > 0:
            crisis_extra = crisis_df.sample(n=extra_n, replace=True, random_state=cfg.seed)
            crisis_aug = pd.concat([crisis_df, crisis_extra], axis=0, ignore_index=True)
        else:
            crisis_aug = crisis_df
    else:
        crisis_aug = crisis_df

    out_df = pd.concat([non_crisis_df, crisis_aug], axis=0, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    stats = {
        "after_neutral_downsample_rows": int(len(step1_df)),
        "neutral_kept_rows": int(len(neutral_kept)),
        "crisis_rows_before_upsample": int(len(crisis_df)),
        "crisis_rows_after_upsample": int(len(crisis_aug)),
    }
    return out_df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="重平衡训练数据：中性下采样 + 危机上采样")
    parser.add_argument("--input-csv", required=True, help="输入 CSV 路径")
    parser.add_argument("--output-csv", required=True, help="输出 CSV 路径")
    parser.add_argument("--label-col", default="label", help="情绪标签列（0~7）")
    parser.add_argument("--crisis-col", default="is_crisis", help="危机二值列（0/1）")
    parser.add_argument("--neutral-label", type=int, default=0, help="中性类标签 id，默认 0")
    parser.add_argument("--neutral-keep-ratio", type=float, default=0.4, help="中性类保留比例，默认 0.4")
    parser.add_argument("--crisis-upsample-factor", type=float, default=3.0, help="危机类扩增倍数，默认 3.0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--stats-json",
        default="",
        help="统计输出 JSON 路径；不传则自动写到 output 同目录",
    )
    args = parser.parse_args()

    _validate_ratio("neutral_keep_ratio", args.neutral_keep_ratio, 0.01, 1.0)
    _validate_ratio("crisis_upsample_factor", args.crisis_upsample_factor, 1.0, 20.0)

    df = pd.read_csv(args.input_csv)
    for col in (args.label_col, args.crisis_col):
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}，当前列: {list(df.columns)}")

    # 基础清洗：确保关键列有效
    work_df = df.copy()
    work_df = work_df.dropna(subset=[args.label_col, args.crisis_col]).reset_index(drop=True)
    work_df[args.label_col] = work_df[args.label_col].astype(int)
    work_df[args.crisis_col] = work_df[args.crisis_col].astype(int).clip(0, 1)

    before_stats = _count_distribution(work_df, args.label_col, args.crisis_col)
    cfg = RebalanceConfig(
        neutral_label=args.neutral_label,
        neutral_keep_ratio=args.neutral_keep_ratio,
        crisis_upsample_factor=args.crisis_upsample_factor,
        seed=args.seed,
    )
    out_df, process_stats = rebalance_dataframe(work_df, args.label_col, args.crisis_col, cfg)
    after_stats = _count_distribution(out_df, args.label_col, args.crisis_col)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    stats_path = args.stats_json.strip()
    if not stats_path:
        out_base = os.path.splitext(args.output_csv)[0]
        stats_path = f"{out_base}_rebalance_stats.json"
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)

    summary = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "config": {
            "label_col": args.label_col,
            "crisis_col": args.crisis_col,
            "neutral_label": args.neutral_label,
            "neutral_keep_ratio": args.neutral_keep_ratio,
            "crisis_upsample_factor": args.crisis_upsample_factor,
            "seed": args.seed,
        },
        "before": before_stats,
        "process": process_stats,
        "after": after_stats,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n已输出重平衡数据: {args.output_csv}")
    print(f"已输出统计信息: {stats_path}")


if __name__ == "__main__":
    main()


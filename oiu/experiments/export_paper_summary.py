#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 real_eval_metrics.json 导出论文可用的结果摘要。

示例（在 oiu 目录）:
  python experiments/export_paper_summary.py
  python experiments/export_paper_summary.py --input-json results/real_eval_metrics.json --output-prefix results/paper_result
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict


def _f(v: Any, digits: int = 4, na: str = "N/A") -> str:
    try:
        x = float(v)
    except Exception:
        return na
    if math.isnan(x) or math.isinf(x):
        return na
    return f"{x:.{digits}f}"


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_texts(metrics: Dict[str, Any]) -> Dict[str, str]:
    inp = metrics.get("input", {})
    cols = inp.get("columns_used", {})
    proto = metrics.get("protocol", {})
    thr = metrics.get("threshold_selection", {})
    split = metrics.get("split_summary", {})
    backend = metrics.get("backend_stats", {})

    val_emo = metrics.get("val_metrics_emotion", {})
    val_cri = metrics.get("val_metrics_crisis", {})
    test_emo = metrics.get("test_metrics_emotion", {})
    test_cri = metrics.get("test_metrics_crisis", {})

    zh = (
        "我们在真实数据上评估了所提系统，并采用用户级划分以避免同一用户跨集合泄漏。"
        f"有效样本数为 {inp.get('rows_valid', 'N/A')}（原始 {inp.get('rows_raw', 'N/A')}），"
        f"随机种子为 {proto.get('seed', 'N/A')}，划分比例为 "
        f"train/val/test={_get(proto, 'split', 'train', default='N/A')}/"
        f"{_get(proto, 'split', 'val', default='N/A')}/"
        f"{_get(proto, 'split', 'test', default='N/A')}。"
        f"危机阈值来源为 {thr.get('source', 'N/A')}，测试阈值为 {_f(thr.get('test_threshold_used'))}。"
        f"在验证集上，情绪分类 Macro-F1={_f(val_emo.get('macro_f1'))}，"
        f"危机检测 PR-AUC={_f(val_cri.get('pr_auc'))}、召回率={_f(val_cri.get('crisis_recall'))}、"
        f"误报率(FPR)={_f(val_cri.get('false_positive_rate'))}。"
        f"在测试集上，情绪分类 Macro-F1={_f(test_emo.get('macro_f1'))}、"
        f"Micro-F1={_f(test_emo.get('micro_f1'))}、Weighted-F1={_f(test_emo.get('weighted_f1'))}；"
        f"危机检测 PR-AUC={_f(test_cri.get('pr_auc'))}、召回率={_f(test_cri.get('crisis_recall'))}、"
        f"误报率(FPR)={_f(test_cri.get('false_positive_rate'))}。"
        f"当前后端统计为 {backend.get('model_backend_counts', {})}，"
        f"优先后端为 {backend.get('preferred_model_backend', 'N/A')}。"
        f"本次实验使用列映射 user={cols.get('user_col')} text={cols.get('text_col')} "
        f"label={cols.get('label_col')} crisis={cols.get('crisis_col')} history={cols.get('history_col')}。"
    )

    en = (
        "We evaluate the proposed system on a real dataset with user-level splitting to avoid cross-split leakage from the same user. "
        f"The number of valid samples is {inp.get('rows_valid', 'N/A')} (raw: {inp.get('rows_raw', 'N/A')}), "
        f"with random seed {proto.get('seed', 'N/A')} and split ratio "
        f"train/val/test={_get(proto, 'split', 'train', default='N/A')}/"
        f"{_get(proto, 'split', 'val', default='N/A')}/"
        f"{_get(proto, 'split', 'test', default='N/A')}. "
        f"The crisis threshold source is {thr.get('source', 'N/A')}, and the test threshold is {_f(thr.get('test_threshold_used'))}. "
        f"On the validation set, emotion classification reaches Macro-F1={_f(val_emo.get('macro_f1'))}, while crisis detection achieves "
        f"PR-AUC={_f(val_cri.get('pr_auc'))}, recall={_f(val_cri.get('crisis_recall'))}, and FPR={_f(val_cri.get('false_positive_rate'))}. "
        f"On the test set, emotion classification obtains Macro-F1={_f(test_emo.get('macro_f1'))}, Micro-F1={_f(test_emo.get('micro_f1'))}, "
        f"and Weighted-F1={_f(test_emo.get('weighted_f1'))}; crisis detection reports PR-AUC={_f(test_cri.get('pr_auc'))}, "
        f"recall={_f(test_cri.get('crisis_recall'))}, and FPR={_f(test_cri.get('false_positive_rate'))}. "
        f"Backend statistics are {backend.get('model_backend_counts', {})}, with preferred backend "
        f"{backend.get('preferred_model_backend', 'N/A')}. "
        f"Column mapping in this run: user={cols.get('user_col')} text={cols.get('text_col')} "
        f"label={cols.get('label_col')} crisis={cols.get('crisis_col')} history={cols.get('history_col')}."
    )

    md = (
        "# Paper-Ready Result Summary\n\n"
        "## Chinese Draft\n\n"
        f"{zh}\n\n"
        "## English Draft\n\n"
        f"{en}\n"
    )

    return {"zh": zh, "en": en, "md": md, "split_json": json.dumps(split, ensure_ascii=False, indent=2)}


def main() -> None:
    p = argparse.ArgumentParser(description="导出论文可用结果摘要")
    p.add_argument("--input-json", default="results/real_eval_metrics.json", help="评估结果 JSON 路径")
    p.add_argument("--output-prefix", default="results/paper_result_summary", help="输出文件前缀（不含扩展名）")
    args = p.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    texts = build_texts(metrics)
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    out_zh = args.output_prefix + ".zh.txt"
    out_en = args.output_prefix + ".en.txt"
    out_md = args.output_prefix + ".md"
    out_split = args.output_prefix + ".split.json"

    with open(out_zh, "w", encoding="utf-8") as f:
        f.write(texts["zh"] + "\n")
    with open(out_en, "w", encoding="utf-8") as f:
        f.write(texts["en"] + "\n")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(texts["md"])
    with open(out_split, "w", encoding="utf-8") as f:
        f.write(texts["split_json"] + "\n")

    print(
        json.dumps(
            {
                "input_json": args.input_json,
                "outputs": [out_zh, out_en, out_md, out_split],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


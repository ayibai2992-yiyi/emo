#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 oiu/results 中读取实验 JSON 并绘制论文图表。

输出目录：
  oiu/results/figures/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close()


def plot_baseline_comparison() -> None:
    p = RESULTS_DIR / "baseline_20k_stratified_users.json"
    if not p.exists():
        return
    data = load_json(p)
    rows = data.get("results", {})
    methods = ["traditional", "single_deep", "three_layer"]

    macro = [rows.get(m, {}).get("emotion_test", {}).get("macro_f1", 0.0) for m in methods]
    prauc = [rows.get(m, {}).get("crisis_test", {}).get("pr_auc", 0.0) for m in methods]
    recall = [rows.get(m, {}).get("crisis_test", {}).get("crisis_recall", 0.0) for m in methods]
    fpr = [rows.get(m, {}).get("crisis_test", {}).get("false_positive_rate", 0.0) for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics: List[Tuple[str, List[float], str]] = [
        ("Emotion Macro-F1", macro, "macro_f1"),
        ("Crisis PR-AUC", prauc, "pr_auc"),
        ("Crisis Recall", recall, "recall"),
        ("Crisis FPR", fpr, "fpr"),
    ]
    for ax, (title, vals, ylabel) in zip(axes.ravel(), metrics):
        ax.bar(methods, vals)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    save_fig("fig_baseline_comparison.png")


def plot_fixed_threshold_scan() -> None:
    files = sorted(RESULTS_DIR.glob("real_eval_20k_fixed_*.json"))
    if not files:
        return

    thrs: List[float] = []
    recalls: List[float] = []
    fprs: List[float] = []
    praucs: List[float] = []
    for p in files:
        d = load_json(p)
        t = d.get("threshold_selection", {}).get("test_threshold_used")
        tm = d.get("test_metrics_crisis", {})
        if t is None:
            continue
        thrs.append(float(t))
        recalls.append(float(tm.get("crisis_recall", 0.0)))
        fprs.append(float(tm.get("false_positive_rate", 0.0)))
        praucs.append(float(tm.get("pr_auc", 0.0)))

    if not thrs:
        return

    zipped = sorted(zip(thrs, recalls, fprs, praucs), key=lambda x: x[0])
    thrs, recalls, fprs, praucs = map(list, zip(*zipped))

    plt.figure(figsize=(8, 5))
    plt.plot(thrs, recalls, marker="o", label="Recall")
    plt.plot(thrs, fprs, marker="o", label="FPR")
    plt.plot(thrs, praucs, marker="o", label="PR-AUC")
    plt.xlabel("Fixed threshold")
    plt.ylabel("Score")
    plt.title("Threshold sensitivity (20k)")
    plt.grid(alpha=0.25)
    plt.legend()
    save_fig("fig_threshold_sensitivity.png")


def plot_strategy_comparison() -> None:
    candidates = [
        ("default", RESULTS_DIR / "real_eval_20k_stratified_users.json"),
        ("l1_5.5", RESULTS_DIR / "real_eval_20k_l1_5p5.json"),
        ("no_layer1", RESULTS_DIR / "real_eval_20k_no_layer1.json"),
    ]
    items = []
    for name, path in candidates:
        if path.exists():
            d = load_json(path)
            b = d.get("backend_stats", {}).get("model_backend_counts", {})
            total = sum(int(v) for v in b.values()) or 1
            unified = int(b.get("unified_model", 0))
            tm = d.get("test_metrics_crisis", {})
            items.append(
                (
                    name,
                    unified / total,
                    float(tm.get("crisis_recall", 0.0)),
                    float(tm.get("false_positive_rate", 0.0)),
                )
            )
    if not items:
        return

    names = [x[0] for x in items]
    unified_ratio = [x[1] for x in items]
    recall = [x[2] for x in items]
    fpr = [x[3] for x in items]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(names, unified_ratio)
    axes[0].set_title("Unified model ratio")
    axes[0].set_ylim(0, 1.0)
    axes[1].bar(names, recall)
    axes[1].set_title("Crisis recall")
    axes[1].set_ylim(0, 1.0)
    axes[2].bar(names, fpr)
    axes[2].set_title("Crisis FPR")
    axes[2].set_ylim(0, 1.0)
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    save_fig("fig_strategy_comparison.png")


def plot_main_result_radar() -> None:
    p = RESULTS_DIR / "baseline_20k_stratified_users.json"
    if not p.exists():
        return
    data = load_json(p)
    rows = data.get("results", {})
    methods = ["traditional", "single_deep", "three_layer"]

    # 将“越低越好”的 FPR 转成 1-FPR，便于统一在雷达图中展示（越高越好）
    metric_labels = ["Emotion Macro-F1", "Crisis PR-AUC", "Crisis Recall", "1 - Crisis FPR"]
    vectors: List[List[float]] = []
    for m in methods:
        emo = rows.get(m, {}).get("emotion_test", {})
        cri = rows.get(m, {}).get("crisis_test", {})
        vectors.append(
            [
                float(emo.get("macro_f1", 0.0)),
                float(cri.get("pr_auc", 0.0)),
                float(cri.get("crisis_recall", 0.0)),
                1.0 - float(cri.get("false_positive_rate", 0.0)),
            ]
        )

    n = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for method, vals in zip(methods, vectors):
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, linewidth=2, label=method)
        ax.fill(angles, vals_closed, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.0)
    ax.set_title("Main result radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    save_fig("fig_main_result_radar.png")


def plot_backend_stacked_ratio() -> None:
    candidates = [
        ("default", RESULTS_DIR / "real_eval_20k_stratified_users.json"),
        ("l1_5.5", RESULTS_DIR / "real_eval_20k_l1_5p5.json"),
        ("no_layer1", RESULTS_DIR / "real_eval_20k_no_layer1.json"),
    ]

    names: List[str] = []
    layer1_only: List[float] = []
    heuristic: List[float] = []
    unified_model: List[float] = []
    for name, path in candidates:
        if not path.exists():
            continue
        d = load_json(path)
        b = d.get("backend_stats", {}).get("model_backend_counts", {})
        total = float(sum(int(v) for v in b.values()) or 1)
        names.append(name)
        layer1_only.append(float(b.get("layer1_only", 0)) / total)
        heuristic.append(float(b.get("heuristic", 0)) / total)
        unified_model.append(float(b.get("unified_model", 0)) / total)

    if not names:
        return

    x = np.arange(len(names))
    w = 0.6
    plt.figure(figsize=(8, 5))
    plt.bar(x, layer1_only, width=w, label="layer1_only")
    plt.bar(x, heuristic, bottom=layer1_only, width=w, label="heuristic")
    bottom2 = [a + b for a, b in zip(layer1_only, heuristic)]
    plt.bar(x, unified_model, bottom=bottom2, width=w, label="unified_model")
    plt.xticks(x, names)
    plt.ylim(0, 1.0)
    plt.ylabel("Ratio")
    plt.title("Backend routing ratio (stacked)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_fig("fig_backend_stacked_ratio.png")


def main() -> None:
    plot_baseline_comparison()
    plot_fixed_threshold_scan()
    plot_strategy_comparison()
    plot_main_result_radar()
    plot_backend_stacked_ratio()
    print(f"Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示标准实验协议：合成数据 -> 用户级划分 -> 固定指标。
在 oiu 目录下:  python experiments/run_protocol_demo.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.experiment.protocol import ExperimentProtocol, BaselineKind
from src.experiment.splits import summarize_split, user_stratified_masks
from src.experiment.metrics_protocol import (
    compute_crisis_detection_metrics,
    compute_emotion_metrics,
    crisis_score_from_probs,
    labels_to_crisis_binary,
)
from src.experiment.baselines import baseline_table_rows


def main():
    proto = ExperimentProtocol(seed=42)
    rng = np.random.RandomState(proto.seed)

    # 合成：20 用户，每人 8 条，标签 0-7
    users = []
    y_true = []
    for u in range(20):
        for _ in range(8):
            users.append(f"U{u:02d}")
            y_true.append(int(rng.randint(0, 8)))
    user_ids = np.array(users)
    y_true = np.array(y_true, dtype=int)

    train_m, val_m, test_m = user_stratified_masks(
        user_ids,
        proto.train_ratio,
        proto.val_ratio,
        proto.test_ratio,
        seed=proto.seed,
    )
    split_info = summarize_split(user_ids, train_m, val_m, test_m)

    # 模拟预测：在测试集上轻微扰动
    y_pred = y_true.copy()
    te = test_m
    flip = rng.rand(te.sum()) < 0.15
    y_pred_te = y_pred[te].copy()
    y_pred_te[flip] = rng.randint(0, 8, size=flip.sum())
    y_pred[te] = y_pred_te

    # 多分类 Macro-F1（仅在 test）
    emo = compute_emotion_metrics(y_true[te], y_pred[te])

    # 危机：二值 + 用 one-hot 概率之和模拟 score
    y_crisis = labels_to_crisis_binary(y_true[te], proto.crisis_emotion_indices)
    n_te = int(te.sum())
    proba = rng.dirichlet(np.ones(8), size=n_te)
    # 让正样本略偏高危机维
    for i in range(n_te):
        if y_crisis[i] == 1:
            proba[i, 7] += 0.35
    proba /= proba.sum(axis=1, keepdims=True)
    scores = crisis_score_from_probs(proba, proto.crisis_emotion_indices)
    crisis_m = compute_crisis_detection_metrics(
        y_crisis, scores, threshold=proto.alert_score_threshold
    )

    out = {
        "protocol": {
            "seed": proto.seed,
            "split": {"train": proto.train_ratio, "val": proto.val_ratio, "test": proto.test_ratio},
            "crisis_emotion_indices": list(proto.crisis_emotion_indices),
            "baselines": [b.value for b in proto.baselines],
        },
        "split_summary": split_info,
        "test_metrics_emotion": emo,
        "test_metrics_crisis": crisis_m,
        "baseline_descriptions": baseline_table_rows(),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一基线对比实验：
- traditional: TF-IDF + LogisticRegression
- single_deep: BERT(冻结编码) + 线性分类头(LogisticRegression)
- three_layer: MonitoringPipeline（本文方法）

在 oiu 目录示例：
  python experiments/run_baseline_comparison.py --csv data/processed/train.csv
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.experiment.metrics_protocol import (
    best_f1_threshold,
    compute_crisis_detection_metrics,
    compute_emotion_metrics,
    labels_to_crisis_binary,
)
from src.experiment.protocol import ExperimentProtocol
from src.experiment.splits import summarize_split, user_stratified_masks
from src.service.monitoring_pipeline import MonitoringPipeline


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
    if not isinstance(layer2, dict) or not layer2:
        return 0, float(np.clip(fallback_risk / 10.0, 0.0, 1.0))
    crisis_score = None
    if layer2.get("crisis_prob") is not None:
        crisis_score = float(np.clip(layer2["crisis_prob"], 0.0, 1.0))
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



def _bert_frozen_embeddings(texts: List[str], model_name: str, batch_size: int = 16) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
            feats.append(cls)
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 768), dtype=np.float32)


def _run_traditional(
    x_train: List[str],
    y_train: np.ndarray,
    y_cri_train: np.ndarray,
    x_val: List[str],
    y_cri_val: np.ndarray,
    x_test: List[str],
    y_test: np.ndarray,
    y_cri_test: np.ndarray,
) -> Dict:
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    xtr = vec.fit_transform(x_train)
    xva = vec.transform(x_val)
    xte = vec.transform(x_test)

    clf_emo = LogisticRegression(max_iter=1000, multi_class="auto")
    clf_emo.fit(xtr, y_train)
    y_pred_test = clf_emo.predict(xte)
    emo_test = compute_emotion_metrics(y_test, y_pred_test)

    clf_cri = LogisticRegression(max_iter=1000)
    clf_cri.fit(xtr, y_cri_train)
    score_val = clf_cri.predict_proba(xva)[:, 1]
    score_test = clf_cri.predict_proba(xte)[:, 1]
    thr, best_f1 = best_f1_threshold(y_cri_val, score_val)
    cri_test = compute_crisis_detection_metrics(y_cri_test, score_test, threshold=float(thr))

    return {
        "emotion_test": emo_test,
        "crisis_test": cri_test,
        "threshold_selection": {"source": "val_best_f1", "val_best_f1_threshold": float(thr), "val_best_f1": float(best_f1)},
    }


def _run_single_deep(
    x_train: List[str],
    y_train: np.ndarray,
    y_cri_train: np.ndarray,
    x_val: List[str],
    y_cri_val: np.ndarray,
    x_test: List[str],
    y_test: np.ndarray,
    y_cri_test: np.ndarray,
    bert_model_name: str,
) -> Dict:
    ztr = _bert_frozen_embeddings(x_train, model_name=bert_model_name)
    zva = _bert_frozen_embeddings(x_val, model_name=bert_model_name)
    zte = _bert_frozen_embeddings(x_test, model_name=bert_model_name)

    clf_emo = LogisticRegression(max_iter=1000, multi_class="auto")
    clf_emo.fit(ztr, y_train)
    y_pred_test = clf_emo.predict(zte)
    emo_test = compute_emotion_metrics(y_test, y_pred_test)

    clf_cri = LogisticRegression(max_iter=1000)
    clf_cri.fit(ztr, y_cri_train)
    score_val = clf_cri.predict_proba(zva)[:, 1]
    score_test = clf_cri.predict_proba(zte)[:, 1]
    thr, best_f1 = best_f1_threshold(y_cri_val, score_val)
    cri_test = compute_crisis_detection_metrics(y_cri_test, score_test, threshold=float(thr))

    return {
        "emotion_test": emo_test,
        "crisis_test": cri_test,
        "threshold_selection": {"source": "val_best_f1", "val_best_f1_threshold": float(thr), "val_best_f1": float(best_f1)},
    }


def _run_three_layer(
    data: pd.DataFrame,
    user_col: str,
    text_col: str,
    history_col: str,
    y_true_all: np.ndarray,
    y_crisis_all: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
    test_m: np.ndarray,
    device: str,
    history_sep: str,
    layer2_threshold_factor: float,
) -> Dict:
    pipeline = MonitoringPipeline(device=device)
    y_pred = np.zeros(len(data), dtype=int)
    y_score = np.zeros(len(data), dtype=float)
    backends = []

    for i in range(len(data)):
        text = str(data.at[i, text_col])
        uid = str(data.at[i, user_col])
        history = _split_history(data.at[i, history_col], history_sep) if history_col else []
        out = pipeline.analyze_text(
            text=text,
            user_id=uid,
            conversation_history=history,
            layer2_threshold_factor=layer2_threshold_factor,
        )
        final = out.get("final", {})
        layer2 = out.get("layer2") or {}
        pred_id, crisis_score = _emotion_pred_from_layer2(layer2, fallback_risk=float(final.get("risk_score", 0.0)))
        y_pred[i] = pred_id
        y_score[i] = crisis_score
        backends.append(str(out.get("model_backend", "unknown")))

    thr, best_f1 = best_f1_threshold(y_crisis_all[val_m], y_score[val_m])
    emo_test = compute_emotion_metrics(y_true_all[test_m], y_pred[test_m])
    cri_test = compute_crisis_detection_metrics(y_crisis_all[test_m], y_score[test_m], threshold=float(thr))
    return {
        "emotion_test": emo_test,
        "crisis_test": cri_test,
        "threshold_selection": {"source": "val_best_f1", "val_best_f1_threshold": float(thr), "val_best_f1": float(best_f1)},
        "backend_stats": {k: int(v) for k, v in pd.Series(backends).value_counts().to_dict().items()},
    }


def run(args: argparse.Namespace) -> Dict:
    df = pd.read_csv(args.csv, encoding=args.encoding)
    if len(df) == 0:
        raise ValueError("CSV 为空，无法评估。")

    user_col = _pick_col(df, args.user_col, ["user_id", "uid", "user", "userid"])
    text_col = _pick_col(df, args.text_col, ["text", "content", "sentence", "utterance"])
    label_col = _pick_col(df, args.label_col, ["label", "emotion_id", "emotion_label", "y"])
    crisis_col = _pick_col(df, args.crisis_col, ["is_crisis", "crisis", "label_crisis", "y_crisis"], required=False)

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

    data = df[[user_col, text_col, label_col] + ([crisis_col] if crisis_col else []) + ([history_col] if history_col else [])].copy()
    data = data.dropna(subset=[user_col, text_col, label_col])
    data[user_col] = data[user_col].astype(str)
    data[text_col] = data[text_col].astype(str)
    data[label_col] = data[label_col].apply(_safe_int_label)
    data = data[data[text_col].str.strip() != ""].reset_index(drop=True)
    if len(data) < 30:
        raise ValueError(f"有效样本太少（{len(data)}），建议至少 >= 30 以支持基线对比。")

    proto = ExperimentProtocol(seed=args.seed)
    user_ids = data[user_col].to_numpy()
    y_true_all = data[label_col].to_numpy(dtype=int)
    if crisis_col:
        y_crisis_all = np.clip(data[crisis_col].astype(int).to_numpy(), 0, 1)
    else:
        y_crisis_all = labels_to_crisis_binary(y_true_all, proto.crisis_emotion_indices)

    train_m, val_m, test_m = user_stratified_masks(
        user_ids=user_ids,
        train_ratio=proto.train_ratio,
        val_ratio=proto.val_ratio,
        test_ratio=proto.test_ratio,
        seed=proto.seed,
    )

    x_train = data.loc[train_m, text_col].astype(str).tolist()
    x_val = data.loc[val_m, text_col].astype(str).tolist()
    x_test = data.loc[test_m, text_col].astype(str).tolist()
    y_train = y_true_all[train_m]
    y_val = y_true_all[val_m]
    y_test = y_true_all[test_m]
    y_cri_train = y_crisis_all[train_m]
    y_cri_val = y_crisis_all[val_m]
    y_cri_test = y_crisis_all[test_m]

    rows: Dict[str, Dict] = {}
    errors: Dict[str, str] = {}

    if args.run_traditional:
        try:
            rows["traditional"] = _run_traditional(
                x_train,
                y_train,
                y_cri_train,
                x_val,
                y_cri_val,
                x_test,
                y_test,
                y_cri_test,
            )
        except Exception as e:
            errors["traditional"] = str(e)

    if args.run_single_deep:
        try:
            rows["single_deep"] = _run_single_deep(
                x_train,
                y_train,
                y_cri_train,
                x_val,
                y_cri_val,
                x_test,
                y_test,
                y_cri_test,
                bert_model_name=args.bert_model,
            )
        except Exception as e:
            errors["single_deep"] = str(e)

    if args.run_three_layer:
        try:
            rows["three_layer"] = _run_three_layer(
                data=data,
                user_col=user_col,
                text_col=text_col,
                history_col=history_col,
                y_true_all=y_true_all,
                y_crisis_all=y_crisis_all,
                train_m=train_m,
                val_m=val_m,
                test_m=test_m,
                device=args.device,
                history_sep=args.history_sep,
                layer2_threshold_factor=args.layer2_threshold_factor,
            )
        except Exception as e:
            errors["three_layer"] = str(e)

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
        "results": rows,
        "errors": errors,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="基线对比实验：traditional/single_deep/three_layer")
    p.add_argument("--csv", required=True, help="输入 CSV 路径")
    p.add_argument("--encoding", default="utf-8", help="CSV 编码（默认 utf-8）")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="three_layer 推理设备")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    p.add_argument("--user-col", default="", help="用户列名（默认自动识别）")
    p.add_argument("--text-col", default="", help="文本列名（默认自动识别）")
    p.add_argument("--label-col", default="", help="情绪标签列名（0-7，默认自动识别）")
    p.add_argument("--crisis-col", default="", help="危机二值列名（0/1，默认自动识别）")
    p.add_argument("--history-col", default="", help="历史对话列名（可选）")
    p.add_argument("--history-sep", default="\\n", help="历史对话分隔符，默认换行")

    p.add_argument("--bert-model", default="bert-base-chinese", help="single_deep 使用的编码器名称")
    p.add_argument("--layer2-threshold-factor", type=float, default=1.0, help="three_layer 第二层阈值缩放")

    p.add_argument("--run-traditional", action="store_true", default=True, help="运行 traditional")
    p.add_argument("--run-single-deep", action="store_true", default=True, help="运行 single_deep")
    p.add_argument("--run-three-layer", action="store_true", default=True, help="运行 three_layer")
    p.add_argument("--output-json", default="results/baseline_comparison.json", help="输出 JSON 路径")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = run(args)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


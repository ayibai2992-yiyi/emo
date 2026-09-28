#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端训练 UnifiedEmotionModel。

在 oiu 目录运行示例：
  python experiments/train_unified_model.py \\
    --csv ../data_all/train/converted_csv/upload_ready_20k_stratified_users.csv \\
    --epochs 3 --batch-size 8 --device cpu --max-samples 2000

产出：
  models/unified_emotion_model.pt
  models/user_to_idx.json
  models/crisis_threshold.json
  results/train_run_*.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.dataset import EmotionCrisisDataset, collate_emotion_batch, load_and_split_csv
from src.evaluation.threshold_calibration import ThresholdCalibrator
from src.experiment.metrics_protocol import compute_crisis_detection_metrics, compute_emotion_metrics
from src.models.thegn import create_dialogue_graph
from src.models.unified_model import UnifiedEmotionModel
from src.training.losses import UnifiedLoss
from src.utils.config import get_default_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_bert_bottom_layers(model: UnifiedEmotionModel, n_layers: int) -> None:
    bert = getattr(model.cpeb_model, "bert", None)
    if bert is None or n_layers <= 0:
        return
    # embeddings
    for p in bert.embeddings.parameters():
        p.requires_grad = False
    encoder_layers = getattr(getattr(bert, "encoder", None), "layer", None)
    if encoder_layers is None:
        return
    for i, layer in enumerate(encoder_layers):
        if i < n_layers:
            for p in layer.parameters():
                p.requires_grad = False


def build_optimizer(model: UnifiedEmotionModel, lr_bert: float, lr_head: float, weight_decay: float):
    bert_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "cpeb_model.bert" in name:
            bert_params.append(p)
        else:
            head_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": bert_params, "lr": lr_bert},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def _build_graphs_for_batch(
    model: UnifiedEmotionModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    history_texts: List[List[str]],
    device: torch.device,
    history_roles: Optional[List[List[int]]] = None,
    current_role_types: Optional[List[int]] = None,
    use_role_graph: bool = False,
):
    """为有历史的样本构图；无历史返回 None 占位。"""
    graphs = []
    with torch.no_grad():
        # 当前句特征
        cur_feat = model.cpeb_model.encode_text(input_ids, attention_mask)
    for i, hist in enumerate(history_texts):
        if not hist:
            graphs.append(None)
            continue
        turns = hist + ["[CUR]"]
        # 用历史文本重新编码（简化：历史句 + 当前句特征拼接）
        tok = model.cpeb_model.tokenizer
        enc = tok(
            hist,
            truncation=True,
            max_length=64,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.set_grad_enabled(model.training):
            hist_feat = model.cpeb_model.encode_text(enc["input_ids"], enc["attention_mask"])
        feats = torch.cat([hist_feat, cur_feat[i : i + 1]], dim=0)
        node_type_ids = None
        if use_role_graph:
            hist_r = (history_roles[i] if history_roles else []) or []
            # 与 hist 对齐长度
            if len(hist_r) < len(hist):
                hist_r = list(hist_r) + [0] * (len(hist) - len(hist_r))
            hist_r = list(hist_r)[: len(hist)]
            cur_t = 0
            if current_role_types is not None and i < len(current_role_types):
                cur_t = int(current_role_types[i])
            node_type_ids = hist_r + [cur_t]
        graph = create_dialogue_graph(
            turns,
            feats.detach() if not model.training else feats,
            node_type_ids=node_type_ids,
        )
        graphs.append(graph)
    return graphs


def _encode_support(
    model: UnifiedEmotionModel,
    support_ids_list: List[List[torch.Tensor]],
    support_mask_list: List[List[torch.Tensor]],
    support_labels_list: List[List[int]],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """把 batch 内所有 support 拼成共享池（简化版 MEA）。"""
    all_ids, all_mask, all_lab = [], [], []
    for ids, masks, labs in zip(support_ids_list, support_mask_list, support_labels_list):
        for x, m, y in zip(ids, masks, labs):
            all_ids.append(x)
            all_mask.append(m)
            all_lab.append(y)
    if not all_ids:
        return None, None
    ids_t = torch.stack(all_ids, dim=0).to(device)
    mask_t = torch.stack(all_mask, dim=0).to(device)
    labs_t = torch.tensor(all_lab, dtype=torch.long, device=device)
    feats = model.cpeb_model.encode_text(ids_t, mask_t)
    return feats, labs_t


def _weighted_mean(per_sample: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """对 [B] 或可 reduce 到 [B] 的损失做样本加权平均。"""
    w = weights / weights.sum().clamp_min(1e-6)
    return (per_sample * w).sum()


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    class_weights: Optional[torch.Tensor],
    grad_clip: float,
    log_interval: int,
    use_role_graph: bool = False,
    use_sample_weight: bool = False,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    n = 0
    for step, batch in enumerate(loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        user_ids = batch["user_ids"].to(device)
        labels = batch["label_id"].to(device)
        crisis = batch["is_crisis"].to(device)
        sample_w = batch.get("sample_weight")
        if sample_w is not None:
            sample_w = sample_w.to(device)
        else:
            sample_w = torch.ones(labels.size(0), device=device)

        graphs = _build_graphs_for_batch(
            model,
            input_ids,
            attention_mask,
            batch["history_texts"],
            device,
            history_roles=batch.get("history_roles"),
            current_role_types=batch.get("current_role_types"),
            use_role_graph=use_role_graph,
        )
        support_feat, support_lab = None, None
        if "support_input_ids" in batch:
            support_feat, support_lab = _encode_support(
                model,
                batch["support_input_ids"],
                batch["support_attention_mask"],
                batch["support_labels"],
                device,
            )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            conversation_graphs=graphs,
            support_features=support_feat,
            support_labels=support_lab,
            return_intermediate=True,
        )

        # per-sample emotion / crisis，再按 sample_weight 聚合
        emo_per = F.cross_entropy(
            outputs["fusion_logits"],
            labels,
            weight=class_weights,
            reduction="none",
        )
        crisis_logits = outputs["crisis_logits"].view(-1)
        crisis_per = F.binary_cross_entropy_with_logits(
            crisis_logits, crisis.view(-1), reduction="none"
        )

        losses = criterion(
            outputs,
            labels,
            crisis_labels=crisis,
            compute_causal=True,
            compute_crisis=True,
            compute_meta=False,
            compute_graph=False,
        )
        if use_sample_weight:
            emo_loss = _weighted_mean(emo_per, sample_w)
            crisis_loss = _weighted_mean(crisis_per, sample_w)
        else:
            emo_loss = emo_per.mean()
            crisis_loss = crisis_per.mean()

        losses["emotion"] = emo_loss
        losses["crisis"] = crisis_loss
        losses["total"] = (
            criterion.loss_weights.get("emotion", 1.0) * emo_loss
            + criterion.loss_weights.get("crisis", 0.5) * crisis_loss
            + criterion.loss_weights.get("causal", 0.5) * losses["causal"]
        )

        optimizer.zero_grad()
        losses["total"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total += float(losses["total"].item())
        n += 1
        if step % log_interval == 0:
            print(
                f"  step={step} loss={losses['total'].item():.4f} "
                f"emo={losses['emotion'].item():.4f} "
                f"crisis={losses['crisis'].item():.4f} "
                f"causal={losses['causal'].item():.4f}"
            )
    return {"loss": total / max(n, 1)}


@torch.no_grad()
def evaluate(model, loader, device, use_role_graph: bool = False) -> Dict[str, Any]:
    model.eval()
    y_true, y_pred = [], []
    y_crisis, y_score = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        user_ids = batch["user_ids"].to(device)
        labels = batch["label_id"].cpu().numpy()
        crisis = batch["is_crisis"].cpu().numpy()

        graphs = _build_graphs_for_batch(
            model,
            input_ids,
            attention_mask,
            batch["history_texts"],
            device,
            history_roles=batch.get("history_roles"),
            current_role_types=batch.get("current_role_types"),
            use_role_graph=use_role_graph,
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            conversation_graphs=graphs,
            return_intermediate=False,
        )
        pred = outputs["emotion_final"].argmax(dim=1).cpu().numpy()
        score = outputs["crisis_prob"].cpu().numpy()
        y_true.append(labels)
        y_pred.append(pred)
        y_crisis.append(crisis.astype(int))
        y_score.append(score)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_crisis = np.concatenate(y_crisis)
    y_score = np.concatenate(y_score)
    emo = compute_emotion_metrics(y_true, y_pred)
    return {
        "emotion": emo,
        "y_crisis": y_crisis,
        "y_score": y_score,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    p = argparse.ArgumentParser(description="Train UnifiedEmotionModel end-to-end")
    p.add_argument(
        "--csv",
        default="../data_all/train/converted_csv/upload_ready_20k_stratified_users.csv",
        help="训练 CSV（需含 user_id/text/label_id/is_crisis）",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=0, help=">0 时截断训练集加速冒烟")
    p.add_argument("--num-support", type=int, default=None)
    p.add_argument("--freeze-bert-layers", type=int, default=None)
    p.add_argument("--output-dir", default="models")
    p.add_argument("--result-dir", default="results")
    p.add_argument(
        "--specialty",
        action="store_true",
        help="学生/咨询特色：client-only + 域加权 + score 加权 + role 构图",
    )
    p.add_argument("--client-only", action="store_true", help="划分前丢弃 counselor 轮")
    p.add_argument("--domain-weight", action="store_true", help="按 source_file 域加权")
    p.add_argument("--score-weight", action="store_true", help="按 score/危机 样本加权")
    p.add_argument("--role-graph", action="store_true", help="THEGN 使用 client/counselor 节点类型")
    args = p.parse_args()

    if args.specialty:
        args.client_only = True
        args.domain_weight = True
        args.score_weight = True
        args.role_graph = True

    cfg = get_default_config()
    cfg.device = args.device
    cfg.seed = args.seed
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.num_support is not None:
        cfg.training.num_support = args.num_support
    if args.freeze_bert_layers is not None:
        cfg.training.freeze_bert_layers = args.freeze_bert_layers

    set_seed(cfg.seed)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}")
    specialty_cfg = {
        "specialty": bool(args.specialty),
        "client_only": bool(args.client_only),
        "domain_weight": bool(args.domain_weight),
        "score_weight": bool(args.score_weight),
        "role_graph": bool(args.role_graph),
    }
    print("specialty_cfg:", json.dumps(specialty_cfg, ensure_ascii=False))

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(os.path.join(ROOT, csv_path))

    bundles = load_and_split_csv(
        csv_path,
        seed=cfg.seed,
        client_only=args.client_only,
        use_domain_weight=args.domain_weight,
        use_score_weight=args.score_weight,
    )
    print("split meta:", json.dumps(bundles.meta, ensure_ascii=False, indent=2))

    train_df = bundles.train_df
    if args.max_samples and args.max_samples > 0:
        train_df = train_df.sample(
            n=min(args.max_samples, len(train_df)), random_state=cfg.seed
        ).reset_index(drop=True)
        print(f"truncated train to {len(train_df)} for smoke/debug")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model)
    train_ds = EmotionCrisisDataset(
        train_df,
        bundles.user_to_idx,
        tokenizer,
        max_length=cfg.training.max_length,
        max_history_turns=cfg.training.max_history_turns,
        num_support=cfg.training.num_support,
        seed=cfg.seed,
        use_role_graph=args.role_graph,
    )
    val_ds = EmotionCrisisDataset(
        bundles.val_df,
        bundles.user_to_idx,
        tokenizer,
        max_length=cfg.training.max_length,
        max_history_turns=cfg.training.max_history_turns,
        num_support=0,
        seed=cfg.seed + 1,
        use_role_graph=args.role_graph,
    )
    test_ds = EmotionCrisisDataset(
        bundles.test_df,
        bundles.user_to_idx,
        tokenizer,
        max_length=cfg.training.max_length,
        max_history_turns=cfg.training.max_history_turns,
        num_support=0,
        seed=cfg.seed + 2,
        use_role_graph=args.role_graph,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_emotion_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_emotion_batch,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_emotion_batch,
        num_workers=0,
    )

    # 扩大 user embedding 容量
    model = UnifiedEmotionModel(
        bert_model_name=cfg.model.bert_model,
        num_emotions=cfg.model.num_emotion_labels,
        hidden_size=cfg.model.hidden_size,
        graph_hidden_size=cfg.model.graph_hidden_size,
        num_graph_layers=cfg.model.num_graph_layers,
        num_attention_heads=max(1, cfg.model.num_attention_heads),
        meta_adapter_type=cfg.model.meta_learning_algorithm,
        inner_lr=cfg.model.inner_lr,
        num_inner_steps=min(cfg.model.num_inner_steps, 2),
        temporal_decay_lambda=cfg.model.temporal_decay_lambda,
        dropout=cfg.model.dropout,
        device=str(device),
        proto_max_shots=cfg.model.proto_max_shots,
        maml_full_shots=cfg.model.maml_full_shots,
        despair_index=cfg.model.despair_index,
    )
    # 保证用户 embedding 足够大
    n_users = max(bundles.user_to_idx.values(), default=1) + 2
    emb = model.cpeb_model.baseline_estimator.user_embeddings
    if emb.num_embeddings < n_users:
        new_emb = torch.nn.Embedding(n_users, emb.embedding_dim)
        with torch.no_grad():
            new_emb.weight[: emb.num_embeddings] = emb.weight
        model.cpeb_model.baseline_estimator.user_embeddings = new_emb

    model.to(device)
    freeze_bert_bottom_layers(model, cfg.training.freeze_bert_layers)

    class_weights = None
    if cfg.training.class_weight:
        counts = train_df["label_id"].value_counts().reindex(range(8), fill_value=0).to_numpy()
        inv = 1.0 / np.maximum(counts, 1)
        w = inv / inv.sum() * 8.0
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
        print("class_weights:", w.round(4).tolist())

    criterion = UnifiedLoss(
        num_emotions=8,
        loss_weights=cfg.training.loss_weights,
        label_smoothing=cfg.training.label_smoothing,
        crisis_pos_weight=cfg.training.crisis_pos_weight,
        despair_index=cfg.model.despair_index,
    )
    optimizer = build_optimizer(
        model,
        lr_bert=cfg.training.bert_learning_rate,
        lr_head=cfg.training.head_learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = max(1, cfg.training.num_epochs * len(train_loader))
    warmup = int(total_steps * cfg.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_macro = -1.0
    best_path = out_dir / "unified_emotion_model.pt"
    history = []
    use_sample_weight = bool(args.domain_weight or args.score_weight)

    for epoch in range(1, cfg.training.num_epochs + 1):
        t0 = time.time()
        print(f"\n===== Epoch {epoch}/{cfg.training.num_epochs} =====")
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            class_weights,
            cfg.training.grad_clip,
            cfg.training.log_interval,
            use_role_graph=args.role_graph,
            use_sample_weight=use_sample_weight,
        )
        val = evaluate(model, val_loader, device, use_role_graph=args.role_graph)
        cal = ThresholdCalibrator(strategy=cfg.model.threshold_strategy, target_recall=cfg.model.target_recall)
        cal_res = cal.fit(val["y_crisis"], val["y_score"])
        crisis_val = compute_crisis_detection_metrics(
            val["y_crisis"], val["y_score"], threshold=cal.threshold
        )
        macro = val["emotion"]["macro_f1"]
        print(
            f"train_loss={tr['loss']:.4f} val_macro_f1={macro:.4f} "
            f"val_pr_auc={crisis_val['pr_auc']:.4f} thr={cal.threshold:.4f} "
            f"time={time.time()-t0:.1f}s"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "val_emotion": val["emotion"],
                "val_crisis": crisis_val,
                "threshold": cal_res.to_dict(),
            }
        )
        if macro > best_macro:
            best_macro = macro
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "bert_model": cfg.model.bert_model,
                        "meta_adapter_type": cfg.model.meta_learning_algorithm,
                        "seed": cfg.seed,
                        "specialty": specialty_cfg,
                    },
                    "val_macro_f1": macro,
                    "crisis_threshold": cal.threshold,
                },
                best_path,
            )
            with open(out_dir / "user_to_idx.json", "w", encoding="utf-8") as f:
                json.dump(bundles.user_to_idx, f, ensure_ascii=False, indent=2)
            with open(out_dir / "crisis_threshold.json", "w", encoding="utf-8") as f:
                json.dump(cal_res.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"  saved best checkpoint -> {best_path}")

    # 测试集：加载最佳权重，应用锁定阈值
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    thr = float(ckpt.get("crisis_threshold", 0.5))
    test = evaluate(model, test_loader, device, use_role_graph=args.role_graph)
    crisis_test = compute_crisis_detection_metrics(test["y_crisis"], test["y_score"], threshold=thr)
    print("\n===== TEST =====")
    print("emotion:", test["emotion"])
    print("crisis:", crisis_test)
    print("locked_threshold:", thr)

    run_report = {
        "csv": csv_path,
        "specialty": specialty_cfg,
        "split_meta": bundles.meta,
        "history": history,
        "best_val_macro_f1": best_macro,
        "test_emotion": test["emotion"],
        "test_crisis": crisis_test,
        "crisis_threshold": thr,
        "checkpoint": str(best_path),
        "seed": cfg.seed,
        "device": str(device),
        "max_samples": args.max_samples,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = result_dir / f"train_run_{stamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, ensure_ascii=False, indent=2)
    cfg.to_yaml(str(out_dir / "train_config.yaml"))
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
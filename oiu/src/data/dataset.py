"""
情绪-危机对话数据集。

必需字段（CSV）：
  user_id, text, label_id (0-7), is_crisis (0/1), conversation_history (可选)

划分：按 user_id 分层，同一用户不跨 train/val/test。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.experiment.splits import user_stratified_masks


EMOTION_LABELS = ("中性", "高兴", "惊讶", "悲伤", "愤怒", "恐惧", "厌恶", "绝望")


def parse_conversation_history(raw: Any, max_turns: int = 5) -> List[str]:
    """把 CSV 中的历史字段解析为轮次列表。"""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        turns = [str(x).strip() for x in raw if str(x).strip()]
        return turns[-max_turns:]
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "[]", "{}"}:
        return []
    # JSON / Python list
    if text[0] in "([":
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                turns = [str(x).strip() for x in obj if str(x).strip()]
                return turns[-max_turns:]
        except Exception:
            try:
                obj = ast.literal_eval(text)
                if isinstance(obj, list):
                    turns = [str(x).strip() for x in obj if str(x).strip()]
                    return turns[-max_turns:]
            except Exception:
                pass
    # 换行或中文句号切分
    if "\n" in text:
        turns = [x.strip() for x in text.split("\n") if x.strip()]
    else:
        turns = [x.strip() for x in re.split(r"[。！？!?；;]+", text) if x.strip()]
    if len(turns) <= 1 and len(text) > 0:
        turns = [text]
    return turns[-max_turns:]


def _pick_col(df: pd.DataFrame, preferred: str, candidates: Sequence[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"未找到列 {preferred}/{list(candidates)}，当前列={list(df.columns)}")


@dataclass
class SplitBundles:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    user_to_idx: Dict[str, int]
    meta: Dict[str, Any]


def load_and_split_csv(
    csv_path: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    user_col: str = "",
    text_col: str = "",
    label_col: str = "",
    crisis_col: str = "",
    history_col: str = "",
    encoding: str = "utf-8",
) -> SplitBundles:
    """读取 CSV 并按用户划分。"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(path, encoding=encoding)
    u = _pick_col(df, user_col, ["user_id", "uid", "user", "userid"])
    t = _pick_col(df, text_col, ["text", "content", "sentence", "utterance"])
    y = _pick_col(df, label_col, ["label_id", "label", "emotion_id", "emotion_label"])
    try:
        c = _pick_col(df, crisis_col, ["is_crisis", "crisis", "label_crisis", "y_crisis"])
    except ValueError:
        c = ""
    try:
        h = _pick_col(df, history_col, ["conversation_history", "history", "context"])
    except ValueError:
        h = ""

    keep = [u, t, y] + ([c] if c else []) + ([h] if h else [])
    data = df[keep].copy()
    data = data.dropna(subset=[u, t, y]).reset_index(drop=True)
    data[u] = data[u].astype(str)
    data[t] = data[t].astype(str)
    data[y] = data[y].astype(int)
    data = data[(data[y] >= 0) & (data[y] <= 7)]
    data = data[data[t].str.strip() != ""].reset_index(drop=True)

    if c:
        data["is_crisis"] = data[c].astype(int).clip(0, 1)
    else:
        data["is_crisis"] = (data[y] == 7).astype(int)
    data["conversation_history"] = data[h] if h else ""

    data = data.rename(columns={u: "user_id", t: "text", y: "label_id"})
    data = data[["user_id", "text", "label_id", "is_crisis", "conversation_history"]]

    train_m, val_m, test_m = user_stratified_masks(
        data["user_id"].to_numpy(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_df = data.loc[train_m].reset_index(drop=True)
    val_df = data.loc[val_m].reset_index(drop=True)
    test_df = data.loc[test_m].reset_index(drop=True)

    # 用户索引仅由训练集构建；未知用户 → 0
    users = sorted(train_df["user_id"].unique().tolist())
    user_to_idx = {uid: i + 1 for i, uid in enumerate(users)}  # 0 = unk

    meta = {
        "csv": str(path),
        "n_total": int(len(data)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_users_train": int(len(users)),
        "n_users_val": int(val_df["user_id"].nunique()),
        "n_users_test": int(test_df["user_id"].nunique()),
        "label_dist_train": train_df["label_id"].value_counts().sort_index().to_dict(),
        "crisis_rate_train": float(train_df["is_crisis"].mean()),
        "crisis_rate_val": float(val_df["is_crisis"].mean()),
        "crisis_rate_test": float(test_df["is_crisis"].mean()),
        "seed": seed,
    }
    return SplitBundles(train_df, val_df, test_df, user_to_idx, meta)


class EmotionCrisisDataset(Dataset):
    """单条话语样本；可选同用户 K-shot support 索引。"""

    def __init__(
        self,
        df: pd.DataFrame,
        user_to_idx: Dict[str, int],
        tokenizer,
        max_length: int = 128,
        max_history_turns: int = 5,
        num_support: int = 0,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.user_to_idx = user_to_idx
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_history_turns = max_history_turns
        self.num_support = num_support
        self.rng = np.random.RandomState(seed)

        self._user_indices: Dict[str, List[int]] = {}
        for i, uid in enumerate(self.df["user_id"].tolist()):
            self._user_indices.setdefault(uid, []).append(i)

    def __len__(self) -> int:
        return len(self.df)

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        uid = str(row["user_id"])
        text = str(row["text"])
        label = int(row["label_id"])
        crisis = int(row["is_crisis"])
        history = parse_conversation_history(row["conversation_history"], self.max_history_turns)

        enc = self._encode(text)
        item: Dict[str, Any] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "user_idx": torch.tensor(self.user_to_idx.get(uid, 0), dtype=torch.long),
            "label_id": torch.tensor(label, dtype=torch.long),
            "is_crisis": torch.tensor(crisis, dtype=torch.float32),
            "history_texts": history,
            "text": text,
            "user_id": uid,
        }

        if self.num_support > 0:
            pool = [j for j in self._user_indices.get(uid, []) if j != idx]
            support_ids: List[torch.Tensor] = []
            support_mask: List[torch.Tensor] = []
            support_labels: List[int] = []
            if pool:
                take = min(self.num_support, len(pool))
                chosen = self.rng.choice(pool, size=take, replace=False).tolist()
                for j in chosen:
                    r2 = self.df.iloc[j]
                    e2 = self._encode(str(r2["text"]))
                    support_ids.append(e2["input_ids"])
                    support_mask.append(e2["attention_mask"])
                    support_labels.append(int(r2["label_id"]))
            item["support_input_ids"] = support_ids
            item["support_attention_mask"] = support_mask
            item["support_labels"] = support_labels

        return item


def collate_emotion_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DataLoader collate：定长张量堆叠，历史以 list 保留。"""
    out: Dict[str, Any] = {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "user_ids": torch.stack([b["user_idx"] for b in batch], dim=0),
        "label_id": torch.stack([b["label_id"] for b in batch], dim=0),
        "is_crisis": torch.stack([b["is_crisis"] for b in batch], dim=0),
        "history_texts": [b["history_texts"] for b in batch],
        "texts": [b["text"] for b in batch],
        "user_id_str": [b["user_id"] for b in batch],
    }

    if "support_input_ids" in batch[0]:
        # 变长 support：训练循环里按样本处理；这里只打包 list
        out["support_input_ids"] = [b["support_input_ids"] for b in batch]
        out["support_attention_mask"] = [b["support_attention_mask"] for b in batch]
        out["support_labels"] = [b["support_labels"] for b in batch]

    return out

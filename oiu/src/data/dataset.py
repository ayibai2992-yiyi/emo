"""
情绪-危机对话数据集。

必需字段（CSV）：
  user_id, text, label_id (0-7), is_crisis (0/1), conversation_history (可选)

可选特色字段：
  role, score, source_file

划分：按 user_id 分层，同一用户不跨 train/val/test。
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.experiment.splits import user_stratified_masks


EMOTION_LABELS = ("中性", "高兴", "惊讶", "悲伤", "愤怒", "恐惧", "厌恶", "绝望")

# 节点类型：0=client/user，1=counselor/bot，2=emotion（预留）
NODE_TYPE_CLIENT = 0
NODE_TYPE_COUNSELOR = 1

# 域加权：source_file 子串 → 乘数
DOMAIN_WEIGHT_RULES: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    (("时序性对话", "student"), 2.0),
    (("梗", "表情"), 1.0),
    (("nlpcc", "chnsenti", "train.json"), 0.5),
)


def role_to_node_type(role: Any) -> int:
    """把说话人角色映射为图节点类型。"""
    if role is None or (isinstance(role, float) and np.isnan(role)):
        return NODE_TYPE_CLIENT
    s = str(role).strip().lower()
    if s in {"counselor", "bot", "assistant", "therapist", "顾问", "咨询师"}:
        return NODE_TYPE_COUNSELOR
    return NODE_TYPE_CLIENT


def is_counselor_role(role: Any) -> bool:
    return role_to_node_type(role) == NODE_TYPE_COUNSELOR


def parse_conversation_history(raw: Any, max_turns: int = 5) -> List[str]:
    """把 CSV 中的历史字段解析为轮次列表。"""
    turns_with_roles = parse_history_with_roles(raw, current_role="client", max_turns=max_turns)
    return [t for t, _ in turns_with_roles]


def parse_history_with_roles(
    raw: Any,
    current_role: Any = "client",
    max_turns: int = 5,
) -> List[Tuple[str, int]]:
    """
    解析历史为 (text, node_type) 列表。

    若历史无逐轮 role：从当前句 role 向前交替标注（咨询对话常见交替）。
    """
    texts = _parse_history_texts(raw, max_turns=max_turns)
    if not texts:
        return []
    cur_type = role_to_node_type(current_role)
    # 历史最后一轮应与当前句交替，即与 cur_type 相反
    # texts[-1] 是最近历史 → 与当前交替 → 1 - cur_type（仅对 0/1）
    out: List[Tuple[str, int]] = []
    n = len(texts)
    for i, text in enumerate(texts):
        # 距当前句的步数：最近历史为 1，再往前为 2...
        steps_back = n - i
        # 当前为 client(0) 时，最近历史多为 counselor(1)
        node_type = cur_type if (steps_back % 2 == 0) else (1 - cur_type)
        out.append((text, int(node_type)))
    return out


def _parse_history_texts(raw: Any, max_turns: int = 5) -> List[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        turns: List[str] = []
        for x in raw:
            if isinstance(x, dict):
                t = str(x.get("content") or x.get("text") or "").strip()
            else:
                t = str(x).strip()
            if t:
                turns.append(t)
        return turns[-max_turns:]
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "[]", "{}"}:
        return []
    if text[0] in "([":
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return _parse_history_texts(obj, max_turns=max_turns)
        except Exception:
            try:
                obj = ast.literal_eval(text)
                if isinstance(obj, list):
                    return _parse_history_texts(obj, max_turns=max_turns)
            except Exception:
                pass
    if "\n" in text:
        turns = [x.strip() for x in text.split("\n") if x.strip()]
    else:
        turns = [x.strip() for x in re.split(r"[。！？!?；;]+", text) if x.strip()]
    if len(turns) <= 1 and len(text) > 0:
        turns = [text]
    return turns[-max_turns:]


def domain_weight_from_source(source_file: Any) -> float:
    """按 source_file 子串匹配域权重。"""
    s = "" if source_file is None else str(source_file)
    s_lower = s.lower()
    for keys, w in DOMAIN_WEIGHT_RULES:
        for k in keys:
            if k.lower() in s_lower or k in s:
                return float(w)
    return 1.0


def compute_sample_weight(
    source_file: Any = "",
    score: Any = 0.0,
    is_crisis: int = 0,
    use_domain_weight: bool = True,
    use_score_weight: bool = True,
) -> float:
    """域权重 × score 权重（危机样本额外 ×1.5）。"""
    w = 1.0
    if use_domain_weight:
        w *= domain_weight_from_source(source_file)
    if use_score_weight:
        try:
            sc = float(score)
        except (TypeError, ValueError):
            sc = 0.0
        if np.isnan(sc):
            sc = 0.0
        sc = float(np.clip(sc, 0.0, 9.0))
        w *= 1.0 + 0.1 * sc
        if int(is_crisis) == 1:
            w *= 1.5
    return float(max(w, 1e-6))


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
    client_only: bool = False,
    use_domain_weight: bool = False,
    use_score_weight: bool = False,
) -> SplitBundles:
    """读取 CSV 并按用户划分；可选 client-only 过滤与样本权重。"""
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

    role_col = "role" if "role" in df.columns else ""
    score_col = "score" if "score" in df.columns else ""
    source_col = "source_file" if "source_file" in df.columns else ""

    keep = [u, t, y]
    if c:
        keep.append(c)
    if h:
        keep.append(h)
    if role_col:
        keep.append(role_col)
    if score_col:
        keep.append(score_col)
    if source_col:
        keep.append(source_col)

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
    data["role"] = data[role_col].astype(str) if role_col else ""
    if score_col:
        data["score"] = pd.to_numeric(data[score_col], errors="coerce").fillna(0.0)
    else:
        data["score"] = 0.0
    data["source_file"] = data[source_col].astype(str) if source_col else ""

    n_before_filter = int(len(data))
    client_only_applied = False
    if client_only:
        if not role_col:
            warnings.warn("client_only=True 但 CSV 无 role 列，跳过过滤", UserWarning)
        else:
            mask_keep = ~data["role"].map(is_counselor_role)
            data = data.loc[mask_keep].reset_index(drop=True)
            client_only_applied = True

    data = data.rename(columns={u: "user_id", t: "text", y: "label_id"})
    data = data[
        [
            "user_id",
            "text",
            "label_id",
            "is_crisis",
            "conversation_history",
            "role",
            "score",
            "source_file",
        ]
    ]

    data["sample_weight"] = [
        compute_sample_weight(
            source_file=sf,
            score=sc,
            is_crisis=int(cr),
            use_domain_weight=use_domain_weight,
            use_score_weight=use_score_weight,
        )
        for sf, sc, cr in zip(data["source_file"], data["score"], data["is_crisis"])
    ]

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

    users = sorted(train_df["user_id"].unique().tolist())
    user_to_idx = {uid: i + 1 for i, uid in enumerate(users)}  # 0 = unk

    role_dist = (
        data["role"].value_counts(dropna=False).astype(int).to_dict() if role_col else {}
    )
    source_dist = (
        data["source_file"].value_counts().head(20).astype(int).to_dict()
        if source_col
        else {}
    )

    meta = {
        "csv": str(path),
        "n_total": int(len(data)),
        "n_before_client_filter": n_before_filter,
        "n_after_client_filter": int(len(data)),
        "client_only": bool(client_only),
        "client_only_applied": bool(client_only_applied),
        "use_domain_weight": bool(use_domain_weight),
        "use_score_weight": bool(use_score_weight),
        "history_role_heuristic": "alternate_from_current_role",
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_users_train": int(len(users)),
        "n_users_val": int(val_df["user_id"].nunique()),
        "n_users_test": int(test_df["user_id"].nunique()),
        "label_dist_train": train_df["label_id"].value_counts().sort_index().to_dict(),
        "crisis_rate_train": float(train_df["is_crisis"].mean()) if len(train_df) else 0.0,
        "crisis_rate_val": float(val_df["is_crisis"].mean()) if len(val_df) else 0.0,
        "crisis_rate_test": float(test_df["is_crisis"].mean()) if len(test_df) else 0.0,
        "role_dist": role_dist,
        "source_dist": source_dist,
        "sample_weight_mean_train": float(train_df["sample_weight"].mean()) if len(train_df) else 1.0,
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
        use_role_graph: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.user_to_idx = user_to_idx
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_history_turns = max_history_turns
        self.num_support = num_support
        self.use_role_graph = use_role_graph
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
        role = str(row["role"]) if "role" in self.df.columns else ""
        weight = float(row["sample_weight"]) if "sample_weight" in self.df.columns else 1.0

        if self.use_role_graph:
            hist_pairs = parse_history_with_roles(
                row["conversation_history"],
                current_role=role or "client",
                max_turns=self.max_history_turns,
            )
            history = [t for t, _ in hist_pairs]
            history_roles = [r for _, r in hist_pairs]
        else:
            history = parse_conversation_history(row["conversation_history"], self.max_history_turns)
            history_roles = [NODE_TYPE_CLIENT] * len(history)

        # 当前句节点类型追加在构图侧；此处 history_roles 仅对应历史轮
        cur_role_type = role_to_node_type(role) if self.use_role_graph else NODE_TYPE_CLIENT

        enc = self._encode(text)
        item: Dict[str, Any] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "user_idx": torch.tensor(self.user_to_idx.get(uid, 0), dtype=torch.long),
            "label_id": torch.tensor(label, dtype=torch.long),
            "is_crisis": torch.tensor(crisis, dtype=torch.float32),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "history_texts": history,
            "history_roles": history_roles,
            "current_role_type": cur_role_type,
            "text": text,
            "user_id": uid,
            "role": role,
            "source_file": str(row["source_file"]) if "source_file" in self.df.columns else "",
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
        "sample_weight": torch.stack([b["sample_weight"] for b in batch], dim=0),
        "history_texts": [b["history_texts"] for b in batch],
        "history_roles": [b["history_roles"] for b in batch],
        "current_role_types": [b["current_role_type"] for b in batch],
        "texts": [b["text"] for b in batch],
        "user_id_str": [b["user_id"] for b in batch],
        "roles": [b.get("role", "") for b in batch],
        "source_files": [b.get("source_file", "") for b in batch],
    }

    if "support_input_ids" in batch[0]:
        out["support_input_ids"] = [b["support_input_ids"] for b in batch]
        out["support_attention_mask"] = [b["support_attention_mask"] for b in batch]
        out["support_labels"] = [b["support_labels"] for b in batch]

    return out

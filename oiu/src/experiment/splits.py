"""
按用户划分 train/val/test，避免同用户跨集合泄漏。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def user_stratified_masks(
    user_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据 user_ids 生成 train/val/test 布尔掩码（样本级）。

    规则：每个 user 整体只落入一个 split；用户按固定种子打乱后顺序切分。
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")

    user_ids = np.asarray(user_ids)
    rng = np.random.RandomState(seed)
    unique_users = np.unique(user_ids)
    rng.shuffle(unique_users)

    n_u = len(unique_users)
    if n_u < 3:
        raise ValueError(
            "按用户划分至少需要 3 个不同 user_id；当前 unique 用户数不足，无法形成非空 train/val/test。"
        )
    n_train = max(1, int(np.floor(n_u * train_ratio)))
    n_val = max(1, int(np.floor(n_u * val_ratio)))
    n_test = n_u - n_train - n_val
    # 舍入可能导致 test 为 0：从 train/val 各借用户直到 test>=1
    while n_test < 1 and n_train > 1:
        n_train -= 1
        n_test = n_u - n_train - n_val
    while n_test < 1 and n_val > 1:
        n_val -= 1
        n_test = n_u - n_train - n_val
    if n_test < 1:
        raise ValueError("无法在保持 train/val/test 均非空的前提下完成划分，请增加用户数或调整比例。")

    train_users = set(unique_users[:n_train].tolist())
    val_users = set(unique_users[n_train : n_train + n_val].tolist())
    test_users = set(unique_users[n_train + n_val :].tolist())

    def _in(users: set) -> np.ndarray:
        return np.array([u in users for u in user_ids], dtype=bool)

    return _in(train_users), _in(val_users), _in(test_users)


def summarize_split(user_ids: np.ndarray, train_m: np.ndarray, val_m: np.ndarray, test_m: np.ndarray) -> Dict[str, object]:
    """用于日志/论文附录：各 split 用户数与样本数。"""
    uid = np.asarray(user_ids)

    def _stats(m: np.ndarray) -> Dict[str, int]:
        u = uid[m]
        return {"num_samples": int(m.sum()), "num_users": int(len(np.unique(u)))}

    return {"train": _stats(train_m), "val": _stats(val_m), "test": _stats(test_m)}


def users_in_split(user_ids: np.ndarray, mask: np.ndarray) -> List[str]:
    return np.unique(np.asarray(user_ids)[mask]).tolist()

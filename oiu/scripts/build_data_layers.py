#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 data_all 整理为 oiu/data 三层结构（与实验 README 约定一致）：

1) oiu/data/processed/main/
   - train.csv / val.csv / test.csv
   - 字段：user_id, user_id_type, text, label, is_crisis, conversation_history,
           source, emotion_raw, score, original_label（后四列为溯源，评估脚本可忽略）

   数据源优先级：
   - 若存在 data_all/data/{train,val,test}.json 则优先使用；
   - 否则使用 data_all/train/train.json、data_all/val/val.json、data_all/test/test.json。

2) oiu/data/processed/extra_unlabeled/
   - annotation_data.jsonl
   - test/0.json, 5.json, 8.json（待标注/轮次格式，不参与本轮主监督）

3) oiu/data/raw_backup/
   - data_all/train、data_all/test 下 *.tsv / *.txt / *.xml
   - 同源 jsonl：train.jsonl、val.jsonl、test.jsonl（小体积索引用）

运行（在仓库根目录）：
  python oiu/scripts/build_data_layers.py
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# 与 oiu/src/experiment/protocol.py 中 8 类顺序一致：0 中性 … 7 绝望
EMOTION_TO_LABEL: Dict[str, int] = {
    "neutral": 0,
    "none": 0,
    "happy": 1,
    "happiness": 1,
    "like": 1,
    "confident": 1,
    "relieved": 1,
    "excitement": 1,
    "surprise": 2,
    "embarrassed": 2,
    "confused": 2,
    "sadness": 3,
    "sad": 3,
    "angry": 4,
    "anger": 4,
    "fear": 5,
    "anxiety": 5,
    "stress": 5,
    "impatience": 5,
    "disgust": 6,
    "despair": 7,
    "worthless": 7,
    "lonely": 7,
    "frustrated": 7,
}


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DATA_ALL = ROOT / "data_all"
OIU_DATA = ROOT / "oiu" / "data"
MAIN = OIU_DATA / "processed" / "main"
EXTRA = OIU_DATA / "processed" / "extra_unlabeled"
RAW = OIU_DATA / "raw_backup"


CSV_FIELDS = [
    "user_id",
    "user_id_type",
    "text",
    "label",
    "is_crisis",
    "conversation_history",
    "source",
    "emotion_raw",
    "score",
    "original_label",
]


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def _emotion_to_label(emotion: Any) -> Optional[int]:
    if emotion is None:
        return None
    key = str(emotion).strip().lower()
    if "（" in key:
        key = key.split("（", 1)[0].strip()
    return EMOTION_TO_LABEL.get(key)


def _resolve_json(split: str) -> Path:
    """split: train | val | test"""
    p_data = DATA_ALL / "data" / f"{split}.json"
    if p_data.exists():
        return p_data
    if split == "train":
        return DATA_ALL / "train" / "train.json"
    if split == "val":
        return DATA_ALL / "val" / "val.json"
    return DATA_ALL / "test" / "test.json"


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组: {path}")
    return data


def convert_split(split: str, out_csv: Path) -> Tuple[int, Counter]:
    src = _resolve_json(split)
    rows = _load_json_array(src)
    out: List[Dict[str, str]] = []
    dropped = Counter()
    for i, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            dropped["not_dict"] += 1
            continue
        text = _norm_text(str(item.get("text", "")))
        if len(text) < 2:
            dropped["empty_text"] += 1
            continue
        emo = item.get("emotion")
        lid = _emotion_to_label(emo)
        if lid is None:
            dropped[f"unmapped_emotion:{emo}"] += 1
            continue
        uid = f"{split}_{i:06d}"
        score = item.get("score", "")
        ol = item.get("original_label", "")
        src_tag = str(item.get("source", "") or "")
        out.append(
            {
                "user_id": uid,
                "user_id_type": "synthetic_split_index",
                "text": text,
                "label": str(lid),
                "is_crisis": "1" if lid == 7 else "0",
                "conversation_history": "",
                "source": src_tag,
                "emotion_raw": str(emo).strip() if emo is not None else "",
                "score": str(score) if score != "" and score is not None else "",
                "original_label": str(ol) if ol != "" and ol is not None else "",
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(out)
    return len(out), dropped


def copy_extra_unlabeled() -> None:
    EXTRA.mkdir(parents=True, exist_ok=True)
    ann = DATA_ALL / "annotation_data.jsonl"
    if ann.exists():
        shutil.copy2(ann, EXTRA / "annotation_data.jsonl")
    for name in ("0.json", "5.json", "8.json"):
        p = DATA_ALL / "test" / name
        if p.exists():
            shutil.copy2(p, EXTRA / name)


def copy_raw_backup() -> None:
    for sub in ("train", "test"):
        src_dir = DATA_ALL / sub
        if not src_dir.is_dir():
            continue
        dst_dir = RAW / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in src_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() in (".tsv", ".txt", ".xml"):
                shutil.copy2(p, dst_dir / p.name)
    # 同源 jsonl
    jsonl_map = [
        (DATA_ALL / "train" / "train.jsonl", RAW / "train" / "train.jsonl"),
        (DATA_ALL / "val" / "val.jsonl", RAW / "val" / "val.jsonl"),
        (DATA_ALL / "test" / "test.jsonl", RAW / "test" / "test.jsonl"),
    ]
    for src, dst in jsonl_map:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_report(
    train_n: int,
    val_n: int,
    test_n: int,
    train_drop: Counter,
    val_drop: Counter,
    test_drop: Counter,
) -> None:
    rep = {
        "main_csv": {
            "train": str(MAIN / "train.csv"),
            "val": str(MAIN / "val.csv"),
            "test": str(MAIN / "test.csv"),
            "rows": {"train": train_n, "val": val_n, "test": test_n},
        },
        "json_sources": {
            "train": str(_resolve_json("train")),
            "val": str(_resolve_json("val")),
            "test": str(_resolve_json("test")),
        },
        "dropped_counts": {
            "train": dict(train_drop),
            "val": dict(val_drop),
            "test": dict(test_drop),
        },
        "extra_unlabeled": str(EXTRA),
        "raw_backup": str(RAW),
    }
    report_path = OIU_DATA / "processed" / "_build_data_layers_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def main() -> None:
    if not DATA_ALL.exists():
        raise FileNotFoundError(f"未找到 data_all: {DATA_ALL}")

    train_n, td = convert_split("train", MAIN / "train.csv")
    val_n, vd = convert_split("val", MAIN / "val.csv")
    test_n, xtd = convert_split("test", MAIN / "test.csv")

    copy_extra_unlabeled()
    copy_raw_backup()
    write_report(train_n, val_n, test_n, td, vd, xtd)

    print(f"OK: main CSV rows train={train_n} val={val_n} test={test_n}")


if __name__ == "__main__":
    main()

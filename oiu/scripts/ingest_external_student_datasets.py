#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集并转换「学生–AI 对话 + 学生日常热语」外部/本地数据，产出统一 CSV。

主要来源：
1) CPCD（Psy-Chronicle，2026 校园长程心理咨询）via HuggingFace 镜像
   https://huggingface.co/datasets/EdwinUstb/CPCD
2) 本地梗指南 md → 学生口吻热语短句
3) 可选：从本地时序咨询对话中筛校园主题 client 句

输出目录：data/external_processed/
  - cpcd_student_ai_turns.csv
  - slang_hot_speech.csv
  - upload_ready_student_enriched.csv  （与主实验字段对齐的混合集）

用法（仓库根目录）：
  python oiu/scripts/ingest_external_student_datasets.py
  python oiu/scripts/ingest_external_student_datasets.py --cpcd-students 30 --max-enriched 20000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "external_raw"
OUT_DIR = ROOT / "data" / "external_processed"
MEME_MD = ROOT / "数据" / "data" / "raw" / "梗数据集（思考怎么处理中）.md"
LOCAL_DIALOG_DIR = ROOT / "数据" / "data" / "light_deal" / "train" / "时序性对话"
BASE_CSV = ROOT / "data" / "upload_ready_20k_stratified_users.csv"

HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
CPCD_REPO = "EdwinUstb/CPCD"

CSV_FIELDS = [
    "user_id",
    "user_id_type",
    "text",
    "label_id",
    "is_crisis",
    "source_file",
    "label_raw",
    "score",
    "row_id",
    "conversation_history",
    "role",
]

# 校园主题关键词（筛本地咨询 + 弱标）
CAMPUS_KW = [
    "高考", "期末", "挂科", "重修", "绩点", "考研", "宿舍", "室友", "舍友",
    "辅导员", "导师", "实习", "秋招", "春招", "毕业论文", "答辩", "军训",
    "社团", "奖学金", "助学金", "学费", "选课", "专业课", "实验室", "图书馆",
]

CRISIS_KW = ["自杀", "轻生", "不想活", "跳楼", "割腕", "自残", "结束生命", "活不下去"]

# domain / stress → 粗粒度情绪
DOMAIN_TO_LABEL = {
    "学业压力": 5,  # 恐惧/焦虑倾向
    "人际关系": 3,  # 悲伤
    "职业发展": 5,
    "家庭经济": 3,
    "身心健康": 7,
}


def _http_get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 emo-ingest/1.0"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read()


def _hf_list(path_in_repo: str) -> List[str]:
    """列出数据集仓库某路径下文件（依赖 huggingface_hub + 镜像）。"""
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    from huggingface_hub import HfApi

    api = HfApi(endpoint=HF_MIRROR)
    items = api.list_repo_tree(
        CPCD_REPO, repo_type="dataset", path_in_repo=path_in_repo, recursive=True
    )
    out = []
    for x in items:
        p = getattr(x, "path", None)
        if p and str(p).endswith(".json"):
            out.append(str(p))
    return out


def _hf_download(rel_path: str, dest: Path) -> Path:
    parts = rel_path.split("/")
    enc = "/".join(urllib.parse.quote(p) for p in parts)
    url = f"{HF_MIRROR}/datasets/{CPCD_REPO}/resolve/main/{enc}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    data = _http_get(url)
    dest.write_bytes(data)
    return dest


def download_cpcd(student_ids: Iterable[int], sleep_s: float = 0.15) -> List[Path]:
    """下载 CPCD conversation/{bucket}/*.json。

    仓库结构为 conversation/1..10（每桶约 10 名学生，共约 100 人），
    不是 conversation/1..100。传入的 student_ids 按桶号理解。
    """
    raw_root = RAW_DIR / "cpcd"
    raw_root.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    # 只保留有效桶 1..10
    buckets = sorted({int(x) for x in student_ids if 1 <= int(x) <= 10})
    if not buckets:
        buckets = list(range(1, 11))
    for sid in buckets:
        repo_dir = f"conversation/{sid}"
        try:
            files = _hf_list(repo_dir)
        except Exception as e:
            print(f"[warn] list {repo_dir} failed: {e}")
            continue
        print(f"[cpcd] student={sid} files={len(files)}")
        for rel in files:
            local = raw_root / rel.replace("/", "_")
            try:
                _hf_download(rel, local)
                paths.append(local)
                time.sleep(sleep_s)
            except Exception as e:
                print(f"[warn] download {rel}: {e}")
    return paths


def _weak_label_from_text(text: str, stress: Optional[int] = None, domain: str = "") -> Tuple[int, int, float]:
    """返回 (label_id, is_crisis, score)。偏保守，避免弱标注把危机比例抬太高。"""
    t = text or ""
    if any(k in t for k in CRISIS_KW):
        return 7, 1, 9.0
    # 高压力会话：多数标恐惧/悲伤，仅极高压力才危机
    if stress is not None and stress >= 9 and any(k in t for k in ("撑不", "受不了", "绝望", "崩溃", "放弃")):
        return 7, 1, float(min(9, stress))
    if stress is not None and stress >= 6:
        base = DOMAIN_TO_LABEL.get(domain, 5)
        if base == 7:
            base = 5
        return base, 0, float(stress)
    # 简单词典
    if any(k in t for k in ("开心", "高兴", "太好了", "终于", "放松")):
        return 1, 0, 2.0
    if any(k in t for k in ("生气", "愤怒", "烦死", "气死")):
        return 4, 0, 4.0
    if any(k in t for k in ("害怕", "焦虑", "紧张", "压力", "慌", "不安")):
        return 5, 0, 5.0
    if any(k in t for k in ("难过", "伤心", "哭", "失落", "孤独", "委屈")):
        return 3, 0, 4.0
    if any(k in t for k in ("恶心", "讨厌", "厌恶")):
        return 6, 0, 3.0
    if any(k in t for k in ("惊讶", "居然", "没想到")):
        return 2, 0, 2.0
    if domain in DOMAIN_TO_LABEL and stress and stress >= 4:
        lab = DOMAIN_TO_LABEL[domain]
        if lab == 7:
            lab = 5
        return lab, 0, float(stress)
    return 0, 0, float(stress or 1.0)


def _role_norm(role: str) -> str:
    r = (role or "").strip().lower()
    if r in {"student", "client", "user", "求助者", "来访者"}:
        return "client"
    if r in {"counselor", "therapist", "assistant", "ai", "bot", "咨询师"}:
        return "counselor"
    return r or "client"


def convert_cpcd_file(path: Path, client_only: bool = False) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] parse {path.name}: {e}")
        return []
    profile = data.get("student_profile") or {}
    name = str(profile.get("name") or "stu")
    uid = f"cpcd_{hashlib.md5(name.encode('utf-8')).hexdigest()[:10]}"
    rows: List[Dict[str, Any]] = []
    for sess in data.get("sessions") or []:
        event = sess.get("event") or {}
        domain = str(event.get("domain") or "")
        try:
            stress = int(event.get("stress_level") or 0)
        except Exception:
            stress = 0
        dialogue = sess.get("dialogue") or []
        hist: List[str] = []
        for i, turn in enumerate(dialogue):
            role = _role_norm(str(turn.get("role") or ""))
            # 学生–AI：counselor 记为 counselor（训练 specialty 会过滤）；AI 叙事保留 role
            text = str(turn.get("content") or "").strip()
            if not text:
                continue
            if client_only and role != "client":
                hist.append(text)
                continue
            label_id, is_crisis, score = _weak_label_from_text(text, stress=stress, domain=domain)
            # 对学生轮：危机仅当高压力或危机词；咨询师轮不标危机监督优先 client
            if role != "client":
                is_crisis = 0
                if label_id == 7:
                    label_id = 0
                    score = 1.0
            history = "\n".join(hist[-5:])
            rows.append(
                {
                    "user_id": uid,
                    "user_id_type": "cpcd_student",
                    "text": text,
                    "label_id": label_id,
                    "is_crisis": is_crisis,
                    "source_file": f"CPCD/{path.name}",
                    "label_raw": f"domain={domain};stress={stress}",
                    "score": score,
                    "row_id": f"{uid}_s{sess.get('session_id', 0)}_t{i}",
                    "conversation_history": history,
                    "role": role,
                }
            )
            hist.append(text)
    return rows


def parse_meme_md(md_path: Path) -> List[Dict[str, Any]]:
    """把梗指南 md 转成学生口吻热语短句（非百科摘要）。"""
    if not md_path.exists():
        print(f"[warn] meme md missing: {md_path}")
        return []
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    # 块：以「X是什么梗」标题分隔
    blocks = re.split(r"\n(?=\d+:\d+\n)", text)
    templates_pos = [
        "今天终于{meme}了，心情一下子好起来",
        "和室友聊到{meme}，大家都笑疯了",
        "刷到{meme}笑出猪叫，明天考试也不慌了",
        "朋友用{meme}逗我，整个人都轻松了",
        "小组作业做完，晚上靠{meme}快乐一下",
    ]
    templates_neu = [
        "群里又在玩{meme}，我只能回一个6",
        "同学整天{meme}，我有点跟不上梗",
        "朋友发了个{meme}，我假装看懂了",
        "宿舍夜谈又提到{meme}，我躺着听",
        "课间大家聊{meme}，我插不上话",
        "网上全是{meme}，感觉信息过载",
    ]
    templates_neg = [
        "又被导师{meme}，我直接破防了",
        "宿舍那事搞得我很{meme}，真的绷不住",
        "期末周还遇到{meme}，心态崩了",
        "看到{meme}突然emo了，不想社交",
        "被同学阴阳{meme}，内耗一整天",
        "秋招碰壁，刷到{meme}更焦虑了",
        "挂科预警加上{meme}，今晚睡不着",
    ]
    templates_crisis = [
        "最近满脑子都是{meme}，感觉活着没意思",
        "被{meme}搞到撑不下去了，真的好累",
        "旁边人都在{meme}，我却觉得自己很失败",
    ]

    # 额外固定校园热语（不依赖梗名）
    FIXED_HOT = [
        ("今天又摆烂了，根本不想学习", 3, 0, 4.0),
        ("绩点太难看，我直接破防", 5, 0, 6.0),
        ("室友内卷到凌晨，我也焦虑了", 5, 0, 5.0),
        ("秋招挂了一堆，心态崩了", 3, 0, 6.0),
        ("导师已读不回，我emo了一下午", 3, 0, 4.0),
        ("宿舍矛盾升级，真的社恐发作", 5, 0, 5.0),
        ("期末周靠咖啡续命，人麻了", 0, 0, 3.0),
        ("考证失败，有点绝望", 7, 1, 7.0),
        ("被辅导员约谈，压力好大", 5, 0, 5.0),
        ("奖学金没拿到，有点丧", 3, 0, 3.0),
        ("网课挂机被点名，社死", 2, 0, 3.0),
        ("答辩被问懵了，当场破防", 5, 0, 6.0),
        ("恋爱被分手，宿舍哭了一晚", 3, 0, 6.0),
        ("家教被放鸽子，烦死了", 4, 0, 3.0),
        ("社团拉新好累，只想躺平", 0, 0, 2.0),
        ("yyds！这科终于过了", 1, 0, 2.0),
        ("awsl 这只小猫太可爱了", 1, 0, 1.0),
        ("这操作我直接6", 0, 0, 1.0),
        ("又被PUA了，好无语", 4, 0, 5.0),
        ("别鸡娃了，我真的累", 3, 0, 4.0),
    ]

    rows: List[Dict[str, Any]] = []
    meme_names: List[str] = []
    for blk in blocks:
        m = re.search(r"([^\n]{1,40})是什么梗", blk)
        if not m:
            continue
        meme = m.group(1).strip()
        meme = re.sub(r"^[\d:\s]+", "", meme).strip()
        if len(meme) < 1 or len(meme) > 20:
            continue
        if meme in meme_names:
            continue
        meme_names.append(meme)

    rng = random.Random(42)
    for i, meme in enumerate(meme_names):
        uid = f"slang_{i % 80:03d}"
        # 每条梗生成多条口语句，覆盖情绪
        packs = [
            (templates_neu, 0, 0, 2.0),
            (templates_pos, 1, 0, 2.0),
            (templates_neg, 3, 0, 5.0),
        ]
        if i % 17 == 0:
            packs.append((templates_crisis, 7, 1, 8.0))
        for tpls, lab, cri, sc in packs:
            # 每梗多抽几条模板，增加热语覆盖
            chosen = rng.sample(tpls, k=min(3, len(tpls)))
            for tpl in chosen:
                sent = tpl.format(meme=meme)
                rows.append(
                    {
                        "user_id": uid,
                        "user_id_type": "student_slang",
                        "text": sent,
                        "label_id": lab,
                        "is_crisis": cri,
                        "source_file": "梗数据集_hot_speech_synth.txt",
                        "label_raw": f"meme={meme}",
                        "score": sc,
                        "row_id": f"slang_{i}_{lab}_{hashlib.md5(sent.encode()).hexdigest()[:8]}",
                        "conversation_history": "",
                        "role": "client",
                    }
                )
    for j, (sent, lab, cri, sc) in enumerate(FIXED_HOT):
        rows.append(
            {
                "user_id": f"slang_fixed_{j % 40:03d}",
                "user_id_type": "student_slang",
                "text": sent,
                "label_id": lab,
                "is_crisis": cri,
                "source_file": "campus_hot_speech_fixed.txt",
                "label_raw": "fixed_hot",
                "score": sc,
                "row_id": f"fixed_{j}",
                "conversation_history": "",
                "role": "client",
            }
        )
    print(f"[slang] memes={len(meme_names)} utterances={len(rows)}")
    return rows


def filter_local_campus_dialogs(max_files: int = 8000, max_rows: int = 6000) -> List[Dict[str, Any]]:
    """从本地时序咨询中筛校园主题 client 轮（补充学生–AI 语境）。"""
    if not LOCAL_DIALOG_DIR.exists():
        print(f"[warn] local dialog dir missing: {LOCAL_DIALOG_DIR}")
        return []
    files = sorted(LOCAL_DIALOG_DIR.glob("*.json"))[:max_files]
    rows: List[Dict[str, Any]] = []
    for fi, fp in enumerate(files):
        try:
            turns = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(turns, list):
            continue
        joined = "".join(str(t.get("content") or "") for t in turns)
        if not any(k in joined for k in CAMPUS_KW):
            continue
        uid = f"campus_{fp.stem}"
        hist: List[str] = []
        for i, t in enumerate(turns):
            role = _role_norm(str(t.get("role") or ""))
            text = str(t.get("content") or "").strip()
            if not text:
                continue
            # 情绪：优先 annotation
            label_id, score = 0, 1.0
            anns = t.get("annotation") or []
            EMOTION_MAP = {
                "neutral": 0, "none": 0,
                "happy": 1, "happiness": 1, "like": 1, "confident": 1, "relieved": 1,
                "surprise": 2, "embarrassed": 2, "confused": 2,
                "sadness": 3, "sad": 3,
                "angry": 4, "anger": 4,
                "fear": 5, "anxiety": 5, "stress": 5, "impatience": 5,
                "disgust": 6,
                "despair": 7, "worthless": 7, "lonely": 7, "frustrated": 7,
            }
            if anns and isinstance(anns, list) and isinstance(anns[0], dict):
                labmap = anns[0].get("label") or {}
                if isinstance(labmap, dict) and labmap:
                    key = max(labmap.keys(), key=lambda k: float(labmap.get(k) or 0))
                    label_id = EMOTION_MAP.get(str(key).lower(), 0)
                    try:
                        score = float(labmap.get(key) or 1)
                    except Exception:
                        score = 1.0
            is_crisis = 1 if (label_id == 7 or any(k in text for k in CRISIS_KW)) else 0
            if role != "client":
                hist.append(text)
                # 仍保留少量 AI/counselor 作为上下文角色样本（低权重场景用 role 区分）
                if len(rows) < max_rows // 5:
                    rows.append(
                        {
                            "user_id": uid,
                            "user_id_type": "campus_local",
                            "text": text,
                            "label_id": 0,
                            "is_crisis": 0,
                            "source_file": f"时序性对话_campus/{fp.name}",
                            "label_raw": "counselor_turn",
                            "score": 1.0,
                            "row_id": f"{uid}_t{i}",
                            "conversation_history": "\n".join(hist[-5:]),
                            "role": "counselor",
                        }
                    )
                continue
            rows.append(
                {
                    "user_id": uid,
                    "user_id_type": "campus_local",
                    "text": text,
                    "label_id": label_id,
                    "is_crisis": is_crisis,
                    "source_file": f"时序性对话_campus/{fp.name}",
                    "label_raw": "client_campus",
                    "score": score,
                    "row_id": f"{uid}_t{i}",
                    "conversation_history": "\n".join(hist[-5:]),
                    "role": "client",
                }
            )
            hist.append(text)
            if len(rows) >= max_rows:
                print(f"[campus] rows={len(rows)} files_scanned={fi+1}")
                return rows
    print(f"[campus] rows={len(rows)} files_scanned={len(files)}")
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"[write] {path} n={len(rows)}")


def load_base_client_rows(max_n: int = 8000) -> List[Dict[str, Any]]:
    if not BASE_CSV.exists():
        return []
    import pandas as pd

    df = pd.read_csv(BASE_CSV)
    if "role" in df.columns:
        df = df[df["role"].astype(str).str.lower() == "client"]
    df = df.head(max_n)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "user_id": str(r.get("user_id")),
                "user_id_type": str(r.get("user_id_type") or "base"),
                "text": str(r.get("text") or ""),
                "label_id": int(r.get("label_id", r.get("label", 0)) or 0),
                "is_crisis": int(r.get("is_crisis") or 0),
                "source_file": str(r.get("source_file") or "upload_ready_20k"),
                "label_raw": str(r.get("label_raw") or ""),
                "score": float(r.get("score") or 0),
                "row_id": str(r.get("row_id") or ""),
                "conversation_history": str(r.get("conversation_history") or ""),
                "role": "client",
            }
        )
    return rows


def build_enriched(
    cpcd_rows: List[Dict[str, Any]],
    slang_rows: List[Dict[str, Any]],
    campus_rows: List[Dict[str, Any]],
    max_total: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """配比：CPCD 40% + 热语 20% + 校园本地 25% + 原 client 15%。"""
    rng = random.Random(seed)
    base = load_base_client_rows(8000)
    n = max_total
    quotas = {
        "cpcd": int(n * 0.38),
        "slang": int(n * 0.22),
        "campus": int(n * 0.25),
        "base": int(n * 0.15),
    }

    def sample(rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        if not rows:
            return []
        if len(rows) <= k:
            return list(rows)
        return rng.sample(rows, k)

    out: List[Dict[str, Any]] = []
    out.extend(sample(cpcd_rows, quotas["cpcd"]))
    out.extend(sample(slang_rows, quotas["slang"]))
    out.extend(sample(campus_rows, quotas["campus"]))
    out.extend(sample(base, quotas["base"]))
    rng.shuffle(out)
    # 补齐
    if len(out) < max_total:
        pool = cpcd_rows + campus_rows + slang_rows + base
        need = max_total - len(out)
        extra = sample([r for r in pool if r not in out], min(need, len(pool)))
        out.extend(extra)
    return out[:max_total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpcd-students", type=int, default=25, help="下载 CPCD 学生轨迹数 1..N")
    ap.add_argument("--skip-download", action="store_true", help="只用已下载的 CPCD 缓存")
    ap.add_argument("--max-enriched", type=int, default=20000)
    ap.add_argument("--client-only-cpcd", action="store_true", help="CPCD 只导出 Student 轮")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1) CPCD
    if not args.skip_download:
        download_cpcd(range(1, args.cpcd_students + 1))
    cpcd_files = sorted((RAW_DIR / "cpcd").glob("*.json"))
    print(f"[cpcd] cached_files={len(cpcd_files)}")
    cpcd_rows: List[Dict[str, Any]] = []
    for fp in cpcd_files:
        cpcd_rows.extend(convert_cpcd_file(fp, client_only=args.client_only_cpcd))
    write_csv(OUT_DIR / "cpcd_student_ai_turns.csv", cpcd_rows)

    # 2) 热语
    slang_rows = parse_meme_md(MEME_MD)
    write_csv(OUT_DIR / "slang_hot_speech.csv", slang_rows)

    # 3) 本地校园
    campus_rows = filter_local_campus_dialogs()
    write_csv(OUT_DIR / "campus_local_client_turns.csv", campus_rows)

    # 4) 混合 enriched
    enriched = build_enriched(cpcd_rows, slang_rows, campus_rows, args.max_enriched)
    # 优先 client 监督：过滤后仍保留少量 counselor 便于 role 构图
    write_csv(OUT_DIR / "upload_ready_student_enriched.csv", enriched)

    # 统计
    from collections import Counter

    roles = Counter(str(r.get("role")) for r in enriched)
    sources = Counter(str(r.get("source_file")).split("/")[0] for r in enriched)
    print("[summary] enriched roles=", dict(roles))
    print("[summary] enriched source_prefix=", dict(sources))
    meta = {
        "cpcd_turns": len(cpcd_rows),
        "slang_turns": len(slang_rows),
        "campus_turns": len(campus_rows),
        "enriched": len(enriched),
        "cpcd_students_requested": args.cpcd_students,
        "sources": {
            "CPCD": "https://huggingface.co/datasets/EdwinUstb/CPCD (Psy-Chronicle, 2026)",
            "meme_md": str(MEME_MD),
            "local_campus": str(LOCAL_DIALOG_DIR),
        },
        "note": "CPCD 轮次情绪为弱标注(domain/stress/词典)；论文主表请与人工抽检结合。",
    }
    (OUT_DIR / "ingest_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done]", meta)


if __name__ == "__main__":
    main()

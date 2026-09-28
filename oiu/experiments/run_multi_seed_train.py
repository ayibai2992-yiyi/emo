#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多 seed 训练包装：对同一配置跑多个种子，并汇总 Macro-F1 / PR-AUC 均值±标准差。

示例：
  python experiments/run_multi_seed_train.py \\
    --csv ../data/experiment_sets/upload_ready_20k_rebalanced.csv \\
    --seeds 42,43,44 --epochs 3 --device cuda \\
    --output-root models/baseline_rebalanced_multiseed \\
    --result-dir results/multiseed_B1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def _mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    import math

    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return {"mean": m, "std": 0.0, "n": 1}
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return {"mean": m, "std": math.sqrt(var), "n": len(xs)}


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed wrapper for train_unified_model.py")
    p.add_argument("--csv", required=True)
    p.add_argument("--seeds", default="42,43,44", help="逗号分隔种子")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--freeze-bert-layers", type=int, default=10)
    p.add_argument("--specialty", action="store_true")
    p.add_argument("--output-root", default="models/multiseed")
    p.add_argument("--result-dir", default="results/multiseed")
    p.add_argument("--extra-args", default="", help="透传额外参数，空格分隔")
    args = p.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    train_py = ROOT / "experiments" / "train_unified_model.py"
    runs: List[Dict[str, Any]] = []

    for seed in seeds:
        out_dir = Path(args.output_root) / f"seed_{seed}"
        cmd = [
            sys.executable,
            str(train_py),
            "--csv",
            args.csv,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--freeze-bert-layers",
            str(args.freeze_bert_layers),
            "--seed",
            str(seed),
            "--output-dir",
            str(out_dir),
            "--result-dir",
            args.result_dir,
        ]
        if args.specialty:
            cmd.append("--specialty")
        if args.extra_args.strip():
            cmd.extend(args.extra_args.split())

        print("\n==== seed", seed, "====")
        print(" ".join(cmd))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if proc.returncode != 0:
            runs.append({"seed": seed, "ok": False, "returncode": proc.returncode})
            continue

        # 找最新 train_run
        rdir = Path(args.result_dir)
        reports = sorted(rdir.glob("train_run_*.json"), key=lambda x: x.stat().st_mtime)
        report = None
        if reports:
            with open(reports[-1], encoding="utf-8") as f:
                report = json.load(f)
            # 仅当 seed 匹配时采信；否则仍记录路径
            if int(report.get("seed", -1)) != seed:
                # 扫描含该 seed 的
                for cand in reversed(reports):
                    with open(cand, encoding="utf-8") as f:
                        j = json.load(f)
                    if int(j.get("seed", -1)) == seed:
                        report = j
                        break

        entry: Dict[str, Any] = {"seed": seed, "ok": True, "output_dir": str(out_dir)}
        if report:
            entry["report_seed"] = report.get("seed")
            te = report.get("test_emotion") or {}
            tc = report.get("test_crisis") or {}
            entry["test_macro_f1"] = te.get("macro_f1")
            entry["test_pr_auc"] = tc.get("pr_auc")
            entry["test_crisis_recall"] = tc.get("recall")
            entry["test_crisis_fpr"] = tc.get("fpr")
            entry["best_val_macro_f1"] = report.get("best_val_macro_f1")
        runs.append(entry)

    macros = [float(r["test_macro_f1"]) for r in runs if r.get("ok") and r.get("test_macro_f1") is not None]
    praucs = [float(r["test_pr_auc"]) for r in runs if r.get("ok") and r.get("test_pr_auc") is not None]
    summary = {
        "csv": args.csv,
        "seeds": seeds,
        "specialty": bool(args.specialty),
        "runs": runs,
        "aggregate": {
            "test_macro_f1": _mean_std(macros),
            "test_pr_auc": _mean_std(praucs),
        },
        "reproducibility_note": "论文可报 mean±std；若时间不足至少固定 seed=42 并声明单次可复现。",
    }
    out_path = Path(args.result_dir) / "multiseed_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("summary ->", out_path)


if __name__ == "__main__":
    main()

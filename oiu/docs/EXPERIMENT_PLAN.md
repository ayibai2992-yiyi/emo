# 后续实验计划（配比修正 × 学生特色）

> 版本：2026-10-02  
> 前提：原 `upload_ready_20k` 全量 3 epoch 训练已完成（B0）。  
> 数据目录：`data/experiment_sets/`（本计划所需 CSV 已生成）。  
> 代码分支：`cloud-minimal`（含 `--specialty`、bert_finetune、跨域 `--model-dir`、弱标抽检脚本）。

---

## 0. 下一步执行清单（2026-10-02 起立刻执行）

**当前进度**：B0 已训完；数据集与 EI 代码已就绪。按下面顺序在 **AutoDL GPU**（`/root/emo/oiu`）执行。  
**最小 EI 闭环**：B0（已有）+ B1 + S1r + bert_finetune + 消融 + 弱标一致率。

### 0.1 同步代码 + 备份 B0（约 5 分钟，必做）

```bash
cd /root/emo && git pull origin cloud-minimal
cd oiu
mkdir -p models_baseline_20k_orig results/baseline_orig
cp -r models/* models_baseline_20k_orig/ 2>/dev/null || true
cp results/train_run_*.json results/baseline_orig/ 2>/dev/null || true
ls ../data/experiment_sets/*.csv | head
```

确认存在：`upload_ready_20k_rebalanced.csv`、`upload_ready_student_enriched_rebalanced.csv`。

### 0.2 Day1 — 训评 B1（配比修正，约 1.5–3 h）

```bash
python experiments/train_unified_model.py \
  --csv ../data/experiment_sets/upload_ready_20k_rebalanced.csv \
  --epochs 3 --batch-size 8 --device cuda --freeze-bert-layers 10 \
  --output-dir models/baseline_rebalanced \
  --result-dir results
```

```bash
python experiments/run_real_dataset_eval.py \
  --csv ../data/experiment_sets/upload_ready_20k_rebalanced.csv \
  --model-dir models/baseline_rebalanced \
  --device cuda --force-deep --client-only-metrics \
  --output-json results/real_eval_B1_rebalanced.json
```

验收：稀有类 F1 相对 B0 改善或更稳；Macro 不明显崩。

### 0.3 Day1 — 强基线 bert_finetune（B1 同数据，约 1–2 h）

B1 结束后立刻串行跑：

```bash
python experiments/run_baseline_comparison.py \
  --csv ../data/experiment_sets/upload_ready_20k_rebalanced.csv \
  --device cuda --run-bert-finetune --finetune-epochs 3 \
  --output-json results/baseline_B1_with_finetune.json
```

### 0.4 Day1–2 — 训评 S1r + 跨域（约 2–4 h）

```bash
python experiments/train_unified_model.py \
  --csv ../data/experiment_sets/upload_ready_student_enriched_rebalanced.csv \
  --epochs 3 --batch-size 8 --device cuda --freeze-bert-layers 10 \
  --specialty \
  --output-dir models/specialty_enriched_rebal \
  --result-dir results
```

```bash
python experiments/run_real_dataset_eval.py \
  --csv ../data/experiment_sets/upload_ready_student_enriched_rebalanced.csv \
  --model-dir models/specialty_enriched_rebal \
  --device cuda --force-deep --client-only-metrics \
  --output-json results/real_eval_S1r.json
```

跨域（S1r 权重 → 原 20k）：

```bash
python experiments/run_real_dataset_eval.py \
  --csv ../data/experiment_sets/upload_ready_20k_original.csv \
  --model-dir models/specialty_enriched_rebal \
  --device cuda --force-deep --client-only-metrics \
  --output-json results/cross_domain_S1r_on_B0.json
```

### 0.5 Day2 — 消融（在 S1r 权重上）

```bash
python experiments/run_ablation_suite.py
```

### 0.6 本机可并行 — 弱标抽检（不占 GPU）

填写 `data/experiment_sets/weak_label_audit_sample.csv` 的 `human_*` 列后：

```bash
python experiments/score_weak_label_audit.py \
  --audit-csv ../data/experiment_sets/weak_label_audit_filled.csv \
  --output-json results/weak_label_audit_score.json
```

### 0.7 建议日程一览

| 时间 | 任务 | 必要性 |
|------|------|--------|
| 现在 | §0.1 同步 + 备份 B0 | 必做 |
| Day1 | B1 训+评 → bert_finetune | 必做 |
| Day1–2 | S1r 训+评 + 跨域 | 必做 |
| Day2 | 消融 | 必做 |
| 空档 | 本机弱标抽检 | 必做 |
| 赶时间可砍 | 多 seed、RoBERTa、S0/S1 完整对照、Web | 选做 |

**现在立刻做**：云端 `git pull` → 备份 B0 → 启动 **B1 训练**。B1 跑完核对 `train_run_*.json` 中 Macro-F1 / 稀有类 F1，再开 S1r。

---

## 1. 问题与目标

### 1.1 已暴露问题

| 问题 | 证据（原 20k） | 影响 |
|------|----------------|------|
| 类别严重失衡 | 中性 16215/20000（81%）；惊讶 47、愤怒 158、厌恶 184 | 稀有类 F1 不稳；Macro-F1 易被多数类带动 |
| 危机与绝望重合 | 危机 649 条且与 label=7 对齐 | 危机召回可偏高、误报偏多；需在论文写清操作化 |
| 缺少学生特色训 | 主训以咨询混杂为主，未用 enriched | 无法支撑「学生–AI / 热语」叙事的主实验 |
| counselor 占比高 | 原数据 counselor≈11894 | 安慰语干扰求助者情绪学习（`--specialty` 可滤） |

### 1.2 总目标

1. **方法有效性**：在配比更合理的数据上复现/巩固统一模型相对旧基线的优势（B0→B1）。  
2. **场景适配**：在学生增强数据上完成特色训练与评测（S0/S1/S1r）。  
3. **可写论文**：主表 + 消融 + 局限说明齐全，不强行把弱标说成金标。

---

## 2. 实验矩阵

| ID | 数据 | 训练开关 | 权重目录建议 | 论文角色 |
|----|------|----------|--------------|----------|
| **B0** | `upload_ready_20k_original.csv` | 无 specialty（已完成） | `models/` 或备份 `models_baseline_20k_orig/` | 方法基线（已有） |
| **B1** | `upload_ready_20k_rebalanced.csv` | 无 specialty | `models/baseline_rebalanced/` | 配比修正后方法基线 |
| **S0** | `upload_ready_student_enriched.csv` | 无 specialty | `models/enriched_plain/` | 只换数据 |
| **S1** | `upload_ready_student_enriched.csv` | `--specialty` | `models/specialty_enriched/` | 数据 + 特色算法 |
| **S1r** | `upload_ready_student_enriched_rebalanced.csv` | `--specialty` | `models/specialty_enriched_rebal/` | **推荐主报（学生场景）** |

辅助集（可选，不单独占主表）：

- `upload_ready_20k_client_only.csv`：原数据去 counselor  
- `upload_ready_student_enriched_client_only.csv`：enriched 去 counselor  

---

## 3. 已生成数据概况

### 3.1 B0 原始 20k（训练已完成）

- 行数 20000，用户 800  
- 标签：`{0:16215, 1:1363, 2:47, 3:235, 4:158, 5:1149, 6:184, 7:649}`  
- 危机率 ≈ 3.2%  
- 参考：第 1 epoch 验证 Macro-F1≈0.651，PR-AUC≈0.926（最终以测试集 `train_run_*.json` 为准）

### 3.2 B1 重平衡后（新建）

路径：`data/experiment_sets/upload_ready_20k_rebalanced.csv`

| 指标 | Before → After |
|------|----------------|
| 行数 | 20000 → **13334** |
| 中性 | 16215 → **5675**（保留 35%） |
| 惊讶/悲伤/愤怒/厌恶 | 极稀有 → 各 **800** |
| 绝望/危机 | 649 → **1947**（×3） |
| 危机率 | 3.2% → **≈14.6%** |

### 3.3 学生 enriched（已有）与 S1r（新建）

路径：

- 原始：`data/experiment_sets/upload_ready_student_enriched.csv`（同 `external_processed/`）  
- 重平衡：`data/experiment_sets/upload_ready_student_enriched_rebalanced.csv`

| 指标 | Enriched → S1r |
|------|----------------|
| 行数 | 20000 → **≈14245** |
| 中性 | 7592 → **≈4200** |
| 恐惧（原过多） | 8479 → **3200** |
| 惊讶/愤怒/厌恶 | 极稀有 → **≈800** |
| 危机率 | 2.3% → **≈7.9%** |

**限制**：CPCD 轮次情绪多为弱标注；主表 B 需与 B0/B1 对照，并在局限中说明。

---

## 4. 执行步骤与命令

以下均在 **`oiu/`** 目录、GPU 机器上执行。先备份 B0：

```bash
cd /root/emo/oiu   # 或本机 oiu/
mkdir -p models_baseline_20k_orig results/baseline_orig
cp -r models/* models_baseline_20k_orig/ 2>/dev/null || true
cp results/train_run_*.json results/baseline_orig/ 2>/dev/null || true
```

### 4.1 阶段 A — B0 收尾（若测试报告未整理）

```bash
python experiments/run_real_dataset_eval.py \
  --csv ../data/experiment_sets/upload_ready_20k_original.csv \
  --device cuda --force-deep --client-only-metrics \
  --output-json results/real_eval_B0_original.json
```

### 4.2 阶段 B — 训 B1（配比修正，约 1.5–3 h）

```bash
python experiments/train_unified_model.py \
  --csv ../data/experiment_sets/upload_ready_20k_rebalanced.csv \
  --epochs 3 --batch-size 8 --device cuda --freeze-bert-layers 10 \
  --output-dir models/baseline_rebalanced \
  --result-dir results
```

```bash
python experiments/run_real_dataset_eval.py \
  --csv ../data/experiment_sets/upload_ready_20k_rebalanced.csv \
  --model-dir models/baseline_rebalanced \
  --device cuda --force-deep --client-only-metrics \
  --output-json results/real_eval_B1_rebalanced.json
```

### 4.3 阶段 C — 学生特色（优先 S1r，约 2–4 h）

```bash
# 需已 git pull cloud-minimal，且含 --specialty 代码
python experiments/train_unified_model.py \
  --csv ../data/experiment_sets/upload_ready_student_enriched_rebalanced.csv \
  --epochs 3 --batch-size 8 --device cuda --freeze-bert-layers 10 \
  --specialty \
  --output-dir models/specialty_enriched_rebal \
  --result-dir results
```

```bash
python experiments/run_real_dataset_eval.py \
  --csv ../data/experiment_sets/upload_ready_student_enriched_rebalanced.csv \
  --model-dir models/specialty_enriched_rebal \
  --device cuda --force-deep --client-only-metrics \
  --output-json results/real_eval_S1r.json
```

可选对照 S0 / S1（时间紧可只做 S1r）：

```bash
# S0：只换数据
python experiments/train_unified_model.py \
  --csv ../data/experiment_sets/upload_ready_student_enriched.csv \
  --epochs 3 --batch-size 8 --device cuda --freeze-bert-layers 10 \
  --output-dir models/enriched_plain --result-dir results

# S1：数据 + specialty，不重平衡
python experiments/train_unified_model.py \
  --csv ../data/experiment_sets/upload_ready_student_enriched.csv \
  --epochs 3 --batch-size 8 --device cuda --freeze-bert-layers 10 \
  --specialty \
  --output-dir models/specialty_enriched --result-dir results
```

### 4.4 阶段 D — 消融与基线对比（S1r 权重上）

```bash
python experiments/run_ablation_suite.py
# 以及（原协议基线表）
python experiments/run_baseline_comparison.py \
  --csv ../data/experiment_sets/upload_ready_20k_rebalanced.csv \
  --device cuda \
  --output-json results/baseline_cmp_B1.json
```

### 4.5 阶段 E — 系统接入（可选）

将 `models/specialty_enriched_rebal/unified_emotion_model.pt` 接到 `monitoring_pipeline` / `emotion_web.py`，仅作演示，不替代主表数字。

---

## 5. 验收标准

| 检查项 | 通过条件 |
|--------|----------|
| B0 测试报告 | 存在 `train_run_*.json`，含 8 类 F1 与危机指标 |
| B1 vs B0 | 稀有类 F1 总体改善或更稳；Macro 不明显崩 |
| S1r | client 子集指标可报；危机 PR-AUC/召回可解释 |
| 目录隔离 | specialty 未覆盖 `models_baseline_20k_orig` |
| 论文表述 | 写明弱标、危机=绝望操作化、重平衡协议 |

参考线（来自项目既有结果，非本次绝对门槛）：旧 three_layer Macro-F1≈0.16；single_deep（冻结）≈0.42；B0 验证已明显高于前者。公平强基线请用 **bert_finetune**。

---

## 5b. EI 投稿：必做 / 选做（已落地代码）

### 必做

| 项 | 说明 | 状态 |
|----|------|------|
| B0 备份 + 测试集每类 F1 | `models_baseline_20k_orig/` | 云端操作 |
| B1 重平衡训评 | `upload_ready_20k_rebalanced.csv` | 数据集已生成 |
| S1r 特色训评 | enriched_rebalanced + `--specialty` | 数据集已生成 |
| 强基线 bert_finetune | 与主模型同一用户划分 | **代码已加** |
| 模块消融 | `run_ablation_suite.py` | 已有 |
| 弱标人工抽检 | `weak_label_audit_sample.csv`（320 条） | **表已生成，待填写** |
| 跨域评测 | S1r 权重 × 原 20k `--model-dir` | **代码已加** |

**强基线：**

```bash
python experiments/run_baseline_comparison.py \
  --csv ../data/experiment_sets/upload_ready_20k_rebalanced.csv \
  --device cuda --run-bert-finetune --finetune-epochs 3 \
  --output-json results/baseline_B1_with_finetune.json
# 可选第二编码器：
#   --run-roberta-finetune --roberta-model hfl/chinese-roberta-wwm-ext
```

**弱标抽检：**

```bash
# 填写 data/experiment_sets/weak_label_audit_sample.csv 中 human_* 列后：
python experiments/score_weak_label_audit.py \
  --audit-csv ../data/experiment_sets/weak_label_audit_filled.csv \
  --output-json results/weak_label_audit_score.json
```

**跨域（S1r → 原 20k）：**

```bash
python experiments/run_real_dataset_eval.py \
  --csv ../data/experiment_sets/upload_ready_20k_original.csv \
  --model-dir models/specialty_enriched_rebal \
  --device cuda --force-deep --client-only-metrics \
  --output-json results/cross_domain_S1r_on_B0.json
```

### 选做

| 项 | 命令 |
|----|------|
| 多 seed 均值±方差 | `python experiments/run_multi_seed_train.py --seeds 42,43,44 ...` |
| RoBERTa 第二强基线 | `--run-roberta-finetune` |
| S0/S1 完整对照 | 见 §4.3 |
| Web 演示 | `emotion_web.py` |

---

## 6. 建议日程

详见 **§0 下一步执行清单**。摘要：

```text
Day0  备份 B0；确认 experiment_sets；并行填弱标抽检
Day1  训 B1 + bert_finetune 基线
Day1–2  训 S1r + 评测 + 跨域
Day2  消融；多 seed（赶时间可只 seed=42 并声明）
Day3+ 可选 RoBERTa / Web
```

**最小 EI 路径**：B0 + B1 + S1r + bert_finetune + 消融 + 弱标一致率 + 局限段。

---

## 7. 论文表格建议

**表 A**：B0、B1、traditional、single_deep、**bert_finetune**、three_layer — Macro-F1 / 危机指标  

**表 B**：S0、S1、S1r — 整体 + subset_client  

**表 C**：模块消融 + specialty 开关消融  

**表 D（附录）**：弱标 vs 人工一致率  

**局限**：重采样、CPCD 弱标、危机=绝望操作化、热语模板、种子策略。

---

## 8. 同步到云端 / 服务器拉取

本机推送后，在 AutoDL 执行：

```bash
cd /root/emo && git pull origin cloud-minimal
# 仅确认本文件：
ls -l oiu/docs/EXPERIMENT_PLAN.md
# 或只拉该文件（已在仓库内时）：
git checkout origin/cloud-minimal -- oiu/docs/EXPERIMENT_PLAN.md
```

---

## 9. 相关文档与新脚本

- `data/experiment_sets/README.md` / `weak_label_audit_*.csv|json`
- `oiu/experiments/run_multi_seed_train.py`
- `oiu/experiments/sample_weak_label_audit.py` / `score_weak_label_audit.py`
- `oiu/experiments/run_baseline_comparison.py`（含 bert_finetune）
- `oiu/docs/P0_RUNBOOK.md`

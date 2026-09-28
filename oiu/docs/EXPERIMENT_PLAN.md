# 后续实验计划（配比修正 × 学生特色）

> 版本：2026-09-28  
> 前提：原 `upload_ready_20k` 全量 3 epoch 训练已完成（B0）。  
> 数据目录：`data/experiment_sets/`（本计划所需 CSV 已生成）。

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

参考线（来自项目既有结果，非本次绝对门槛）：旧 three_layer Macro-F1≈0.16；single_deep≈0.42；B0 验证已明显高于前者。

---

## 6. 建议日程

```text
Day0（今日）  备份 B0；整理测试集每类 F1；确认 experiment_sets 已在云端
Day1          训 B1 + 评测
Day1–2        训 S1r + 评测（主攻）
Day2          可选 S0/S1；消融；写表
Day3+         Web 接入（可选）
```

**最小论文路径**：B0（完成）+ B1 + S1r + 一组模块消融。

---

## 7. 论文表格建议

**表 A — 方法主结果（原域/重平衡）**

- 行：B0、B1、（可选）traditional / single_deep / three_layer  
- 列：Macro-F1、各类 F1（或附录）、危机 PR-AUC / Recall / FPR  

**表 B — 学生场景**

- 行：S0、S1、S1r  
- 列：整体 Macro-F1、subset_client Macro-F1、危机指标  

**表 C — 消融**

- 行：full / no_cpeb / no_mea / no_thegn；（算法开关）client-only / domain-weight / role-graph  

**局限段**：数据失衡与重采样、CPCD 弱标、危机操作化为绝望、热语为模板生成。

---

## 8. 同步到云端

本机生成后，将 `data/experiment_sets/` 与更新后的本文件纳入 `cloud-minimal` 再 `git pull`，例如：

```bash
# 本机（在仓库根，cloud-minimal 分支）
git add data/experiment_sets oiu/docs/EXPERIMENT_PLAN.md oiu/experiments/rebalance_dataset.py
git commit -m "Add rebalanced experiment sets and experiment plan"
git push origin cloud-minimal

# AutoDL
cd /root/emo && git pull origin cloud-minimal
ls -lh data/experiment_sets/
```

---

## 9. 相关文档

- `data/experiment_sets/README.md` — 数据集字段与复现命令  
- `data/external_processed/README.md` — 学生 enriched 来源说明  
- `oiu/docs/P0_RUNBOOK.md` — 训练/评测基础命令  

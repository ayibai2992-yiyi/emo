# P0 运行清单（端到端训练 → 接入流水线 → 正式评测）

## 已就绪代码

| 组件 | 路径 |
|------|------|
| Dataset + 按用户划分 | `src/data/dataset.py` |
| 端到端训练 | `experiments/train_unified_model.py` |
| 深度分析接统一模型 | `src/monitoring/deep_analyzer.py` |
| 流水线加载权重/阈值 | `src/service/monitoring_pipeline.py` |
| 验证集阈值校准 | `src/evaluation/threshold_calibration.py` |
| 学生/咨询特色（`--specialty`） | `dataset.py` + `thegn.py` + `student_lexicon.py` |

## 你需要执行的命令（在 `oiu/` 目录）

### 1) 冒烟训练（先确认能跑通，约几分钟～数十分钟，视机器而定）

```bash
cd oiu
python experiments/train_unified_model.py \
  --csv ../data/upload_ready_20k_stratified_users.csv \
  --epochs 1 --batch-size 4 --device cuda --max-samples 200 --freeze-bert-layers 10
```

### 2) 正式训练（方法基线，无特色）

```bash
python experiments/train_unified_model.py \
  --csv ../data/upload_ready_20k_stratified_users.csv \
  --epochs 3 --batch-size 8 --device cuda --freeze-bert-layers 10
```

产出：
- `models/unified_emotion_model.pt`
- `models/user_to_idx.json`
- `models/crisis_threshold.json`
- `results/train_run_*.json`

### 2b) 学生/咨询特色重训（基线训完后对照）

`--specialty` = client-only（丢弃 counselor）+ 域加权 + score 加权 + role 异构构图。  
权重写入独立目录，避免覆盖基线：

```bash
python experiments/train_unified_model.py \
  --csv ../data/upload_ready_20k_stratified_users.csv \
  --epochs 3 --batch-size 8 --device cuda \
  --freeze-bert-layers 10 --specialty \
  --output-dir models/specialty \
  --result-dir results
```

单独消融开关：`--client-only` / `--domain-weight` / `--score-weight` / `--role-graph`。

### 2c) 学生增强数据（CPCD + 热语 + 校园筛选）

采集脚本（仓库根目录）：

```bash
# 默认走 hf-mirror；可设 HF_ENDPOINT
python oiu/scripts/ingest_external_student_datasets.py --cpcd-students 40 --max-enriched 20000 --client-only-cpcd
```

产出：
- `data/external_processed/cpcd_student_ai_turns.csv` — CPCD 校园学生–咨询/AI 对话轮
- `data/external_processed/slang_hot_speech.csv` — 学生口吻热语
- `data/external_processed/upload_ready_student_enriched.csv` — 混合 20k（建议下一轮 `--specialty` 训练用）

来源说明：CPCD = Psy-Chronicle（2026，HuggingFace `EdwinUstb/CPCD`）；情绪标签为弱标注，正式论文建议抽检。

```bash
python experiments/train_unified_model.py \
  --csv ../data/external_processed/upload_ready_student_enriched.csv \
  --epochs 3 --batch-size 8 --device cuda \
  --freeze-bert-layers 10 --specialty \
  --output-dir models/specialty_enriched \
  --result-dir results
```

### 3) 用训练后权重重跑评测（强制走深度模型，避免 layer1_only 淹没指标）

```bash
python experiments/run_real_dataset_eval.py \
  --csv ../data/upload_ready_20k_stratified_users.csv \
  --device cuda --force-deep \
  --client-only-metrics \
  --output-json results/real_eval_after_train.json
```

`--client-only-metrics` 额外报告 test 上 `subset_client` 与 `subset_student_source`（后者样本很少，仅作附录）。

对比基线：

```bash
python experiments/run_baseline_comparison.py \
  --csv ../data/upload_ready_20k_stratified_users.csv \
  --device cuda \
  --output-json results/baseline_after_train.json
```

## 验收门槛（P0 完成判定）

1. `models/unified_emotion_model.pt` 存在且非随机初始化推理
2. `backend_stats` 中深度路径为 `unified_model`（或 force_deep 全量）
3. 测试集 Macro-F1 **显著高于** 旧 three_layer（≈0.16），并接近/超过 single_deep（≈0.42）趋势合理
4. 危机阈值来自 `crisis_threshold.json`（验证集锁定），测试集不再重调
5. 报告数字全部来自本次真实运行 JSON
6. 特色版：`split_meta.client_only_applied=true`，train 中 counselor 显著减少；`specialty` 字段写入 `train_run_*.json`

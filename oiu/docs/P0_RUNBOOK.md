# P0 运行清单（端到端训练 → 接入流水线 → 正式评测）

## 已就绪代码

| 组件 | 路径 |
|------|------|
| Dataset + 按用户划分 | `src/data/dataset.py` |
| 端到端训练 | `experiments/train_unified_model.py` |
| 深度分析接统一模型 | `src/monitoring/deep_analyzer.py` |
| 流水线加载权重/阈值 | `src/service/monitoring_pipeline.py` |
| 验证集阈值校准 | `src/evaluation/threshold_calibration.py` |

## 你需要执行的命令（在 `oiu/` 目录）

### 1) 冒烟训练（先确认能跑通，约几分钟～数十分钟，视机器而定）

```bash
cd oiu
python experiments/train_unified_model.py ^
  --csv ../data_all/train/converted_csv/upload_ready_20k_stratified_users.csv ^
  --epochs 1 --batch-size 4 --device cpu --max-samples 200 --freeze-bert-layers 10
```

有 GPU 时把 `--device cuda`，并去掉 `--max-samples` 或设大一些。

### 2) 正式训练（论文主结果）

```bash
python experiments/train_unified_model.py ^
  --csv ../data_all/train/converted_csv/upload_ready_20k_stratified_users.csv ^
  --epochs 5 --batch-size 8 --device cuda
```

产出：
- `models/unified_emotion_model.pt`
- `models/user_to_idx.json`
- `models/crisis_threshold.json`
- `results/train_run_*.json`

### 3) 用训练后权重重跑评测（强制走深度模型，避免 layer1_only 淹没指标）

```bash
python experiments/run_real_dataset_eval.py ^
  --csv ../data_all/train/converted_csv/upload_ready_20k_stratified_users.csv ^
  --device cuda --force-deep ^
  --output-json results/real_eval_after_train.json
```

对比基线：

```bash
python experiments/run_baseline_comparison.py ^
  --csv ../data_all/train/converted_csv/upload_ready_20k_stratified_users.csv ^
  --device cuda ^
  --output-json results/baseline_after_train.json
```

## 验收门槛（P0 完成判定）

1. `models/unified_emotion_model.pt` 存在且非随机初始化推理
2. `backend_stats` 中深度路径为 `unified_model`（或 force_deep 全量）
3. 测试集 Macro-F1 **显著高于** 旧 three_layer（≈0.16），并接近/超过 single_deep（≈0.42）趋势合理
4. 危机阈值来自 `crisis_threshold.json`（验证集锁定），测试集不再重调
5. 报告数字全部来自本次真实运行 JSON

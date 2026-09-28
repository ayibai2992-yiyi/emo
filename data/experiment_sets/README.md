# 实验数据集目录（experiment_sets）

本目录存放后续对照实验所需 CSV，均由本地数据生成，字段与主训练一致。

## 文件一览

| 文件 | 用途 | 约行数 | 说明 |
|------|------|--------|------|
| `upload_ready_20k_original.csv` | **B0** 方法基线（已训完） | 20000 | 原分层 20k，中性约 81% |
| `upload_ready_20k_rebalanced.csv` | **B1** 配比修正基线 | ~13334 | 中性↓ + 稀有类↑ + 危机↑ |
| `upload_ready_20k_client_only.csv` | 辅助：只留非 counselor | ~8106 | 未重平衡，角色过滤 |
| `upload_ready_student_enriched.csv` | **S0/S1** 学生增强 | 20000 | CPCD+热语+校园+原 client |
| `upload_ready_student_enriched_rebalanced.csv` | **S1r** 推荐主训 | ~14245 | enriched 再平衡（恐惧↓、稀有↑） |
| `upload_ready_student_enriched_client_only.csv` | 辅助 | ~19430 | enriched 去 counselor |
| `*_rebalance_stats.json` | 重平衡前后统计 | — | 论文附录可引用 |

## 标签约定

情绪 `label_id`：0 中性 · 1 高兴 · 2 惊讶 · 3 悲伤 · 4 愤怒 · 5 恐惧 · 6 厌恶 · 7 绝望  
危机：`is_crisis`；当前数据中危机与绝望高度重合，论文需写明操作化定义。

## 重平衡参数（可复现）

**B1（原 20k）**

```bash
python oiu/experiments/rebalance_dataset.py \
  --input-csv data/upload_ready_20k_stratified_users.csv \
  --output-csv data/experiment_sets/upload_ready_20k_rebalanced.csv \
  --label-col label_id --neutral-keep-ratio 0.35 --crisis-upsample-factor 3.0 \
  --rare-labels 2,3,4,6 --rare-min-count 800 --seed 42
```

**S1r（学生 enriched）**

```bash
python oiu/experiments/rebalance_dataset.py \
  --input-csv data/external_processed/upload_ready_student_enriched.csv \
  --output-csv data/experiment_sets/upload_ready_student_enriched_rebalanced.csv \
  --label-col label_id --neutral-keep-ratio 0.55 --crisis-upsample-factor 2.5 \
  --rare-labels 2,4,6 --rare-min-count 800 --majority-max-count 3200 --seed 42
```

详见上级文档：`oiu/docs/EXPERIMENT_PLAN.md`。

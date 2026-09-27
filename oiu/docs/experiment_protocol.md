# 标准实验协议（数据划分 · 指标 · 基线）

## 1. 数据划分（防泄漏）

- **原则**：同一 `user_id` 的样本只出现在 **train / val / test 之一**，避免「同一用户说话模式」跨集合泄漏。
- **实现**：`src.experiment.splits.user_stratified_masks`，需 **至少 3 个不同用户**。
- **固定随机性**：`ExperimentProtocol.seed` 控制用户打乱顺序（论文中写死种子便于复现）。

### 建议表格字段（CSV/JSONL 均可）

| 字段 | 说明 |
|------|------|
| `user_id` | 用户标识（字符串或整型转字符串） |
| `text` | 当前轮文本 |
| `label` | 8 类情绪整数标签 0–7（与 `emotion_labels` 顺序一致） |
| `is_crisis` | （可选）0/1，人工标注的危机标签；若无则用 `label ∈ crisis_emotion_indices` 派生 |

## 2. 固定指标

模块：`src.experiment.metrics_protocol`

| 指标 | 含义 | 函数 |
|------|------|------|
| Macro-F1 | 8 类情绪宏平均 F1 | `compute_emotion_metrics` |
| Micro / Weighted F1 | 辅助对照 | 同上 |
| PR-AUC | 二值危机检测的 Average Precision | `compute_crisis_detection_metrics` |
| 危机召回 | 正类召回 TP/(TP+FN) | 同上 |
| 误报率 FPR | FP/(FP+TN)，非危机被判危机的比例 | 同上 |

- **危机类**：默认绝望类 `label==7`，可在 `ExperimentProtocol.crisis_emotion_indices` 修改。
- **危机分数**：多分类模型可用 `crisis_score_from_probs(proba, indices)` 将危机维概率求和作为 `y_score`。
- **依赖**：若已安装 `scikit-learn`，指标与论文常用实现一致；否则自动使用 **NumPy 回退**（建议正式投稿前用 sklearn 再核对一次）。

## 3. 固定对比基线

枚举：`src.experiment.protocol.BaselineKind`

- `traditional`：传统浅层（TF-IDF + LR/SVM 等）
- `single_deep`：单一 BERT 分类头
- `three_layer`：本文三层监测流水线

说明表：`src.experiment.baselines.BASELINE_REGISTRY`（`baseline_table_rows()` 可导出）。

## 4. 一键演示

在 `oiu` 目录下：

```bash
python experiments/run_protocol_demo.py
```

输出 JSON：划分摘要、测试集 Macro-F1、危机检测 PR-AUC/召回/FPR（演示为随机分数，仅展示字段）。

## 5. 单测

```bash
python tests/test_experiment_protocol.py
```

## 6. 消融实验（与主表配合）

见 `docs/ablation_experiments.md` 与 `experiments/run_ablation_suite.py`：统一模型 CPEB / MEA / THEGN 与第一层快筛的开关与延迟统计。

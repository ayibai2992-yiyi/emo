# 消融实验设计（论文核心证据）

## 1. 统一模型内部三路（`UnifiedEmotionModel`）

配置类：`src.models.ablation.ModuleAblation`

| 变体 | `use_cpeb` | `use_mea` | `use_thegn` | 含义 |
|------|------------|-----------|-------------|------|
| full | ✓ | ✓ | ✓ | 完整模型 |
| no_cpeb | ✗ | ✓ | ✓ | 融合第 1 路改为 **原始 softmax**（不经因果去偏） |
| no_mea | ✓ | ✗ | ✓ | **不调用** MEA，第 2 路为 **均匀分布** |
| no_thegn | ✓ | ✓ | ✗ | **不调用** THEGN，第 3 路为 **全零** |
| all_off | ✗ | ✗ | ✗ | 三路极端对照（仍过融合层） |

调用方式：

```python
from src.models.ablation import ModuleAblation
from src.models.unified_model import UnifiedEmotionModel

model = UnifiedEmotionModel(device="cpu")
out = model.predict(text="...", user_id=1, conversation_history=[...], ablation=ModuleAblation.no_mea())
```

前向接口：`forward(..., ablation=ModuleAblation(...))`。

**性能指标**：在固定 test 集上报告 Macro-F1 等（见 `experiment_protocol.md`），每个变体 **单独评测**（或固定种子重训融合层——属进阶；当前实现为 **同一套已训练权重** 下的推理消融，论文中需明确说明）。

**延迟**：`experiments/run_ablation_suite.py` 对固定若干条样本统计 `predict` 总耗时与单条均值。

## 2. 第一层快筛（系统级）

配置：`src.models.ablation.PipelineAblation(use_layer1=False)`，经 `MonitoringPipeline.analyze_text(..., pipeline_ablation=...)`：

- `use_layer1=False`：**不执行** `LightweightEmotionFilter`，强制 `need_deep_analysis=True`，第一层延迟近似为 0（仅计时开销）。
- 用于对比 **端到端延迟** 与 **进入深层的比例 / 漏检风险**（若仅用 layer1 会省算力但可能漏危机，可在讨论中写清）。

## 3. 与标注集结合的推荐流程

1. 按用户划分 test（`user_stratified_masks`）。
2. 对每个消融变体，在 test 上跑推理得到 `y_pred`。
3. 用 `compute_emotion_metrics` 报 Macro-F1；危机维用 `labels_to_crisis_binary` + `compute_crisis_detection_metrics`。
4. 主表：行=变体，列=Macro-F1 / PR-AUC / 危机召回 / FPR / 平均延迟。

## 4. 自动化脚本

```bash
python experiments/run_ablation_suite.py
```

输出 JSON 行表，可粘贴到论文附录或画柱状图。

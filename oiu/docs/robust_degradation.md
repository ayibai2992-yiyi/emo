# 鲁棒性：统一模型自动降级机制（论文工程章节素材）

## 设计动机

在线情绪监测需要兼顾 **深度模型能力** 与 **服务可用性**。统一模型 `UnifiedEmotionModel` 依赖 Transformers 与图模块等多子系统，任一环节在部署环境失败（权重损坏、算子不兼容、维度配置错误、显存不足等）都不应导致整条流水线不可用。

## 实现要点

1. **启动自检（Warmup）**  
   `MonitoringPipeline` 在构造时尝试构建 `UnifiedEmotionModel`，并执行一次与线上一致的 `predict()` 自检（含非空对话历史，覆盖 THEGN 构图路径）。若抛出异常，则 **不挂载** 统一模型，并记录日志。

2. **推理时双保险**  
   `PersonalizedDeepAnalyzer` 在 `unified_model` 已挂载的前提下优先走 `_analyze_with_unified_model()`；若单次推理失败，则 **捕获异常并回退** 到启发式分支，保证 API/前端仍返回结构化结果。

3. **前端可观测性**  
   API 返回字段建议区分：
   - `preferred_model_backend`：配置/期望使用的后端（`unified_model` 或 `heuristic`）。
   - `model_backend`：本次深度层实际使用的后端（`unified_model` / `heuristic` / `layer1_only`）。

## 论文表述建议

可将本节概括为 **「推理路径熔断与降级（graceful degradation）」**：在复杂多模块系统中，通过 **离线自检 + 在线回退** 降低故障面，同时保留可解释的后备估计，便于审计与临床/教育场景下的安全部署讨论。

## 训练权重与恢复统一模型

将训练得到的 `state_dict` 保存为：

`models/unified_emotion_model.pt`

流水线在 `MonitoringPipeline` 初始化时会尝试 `load_state_dict(..., strict=False)` 加载；若 Warmup 通过，则 `model_backend` 在深度分析触发时应为 `unified_model`。

## 附录：THEGN 多头输出维度（与 `hidden_size` / `graph_hidden_size` 对齐）

`THEGNModel` 中 `HeterogeneousGATLayer` 在 `concat=False` 时原先对多头取均值，得到 `out_features = hidden_size // num_heads`（如 32），与后续 `LayerNorm(hidden_size)` 及 `Linear(hidden_size, hidden_size)` 不一致。现改为 **先展平各头再经 `Linear(num_heads * out_features, in_features)` 投影回 `in_features`（即 `graph_hidden_size`）**，保证与 `UnifiedEmotionModel` 中 `hidden_size=768`（BERT 特征）经 `input_projection` 映射到 `graph_hidden_size=256` 后的张量流一致。

"""
统一情绪监测模型
整合 CPEB + MEA + THEGN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer

from .cpeb import CPEBModel
from .meta_learner import MetaEmotionAdapter
from .thegn import THEGNModel, create_dialogue_graph, HeteroGraph


class EvidenceAwareFusion(nn.Module):
    """
    Reliability-Aware Contextual Fusion (RACF)
    证据可靠性感知动态融合模块。

    不为三路输出做固定拼接，而是结合文本上下文、分支置信度/熵和图可用性，
    为 CPEB、MEA、THEGN 动态学习样本级融合权重，并输出可解释 fusion_weights。
    """

    def __init__(
        self,
        num_emotions: int,
        context_size: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_branches = 3
        self.num_emotions = num_emotions

        self.branch_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_emotions, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(self.num_branches)
        ])

        self.context_projector = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        reliability_size = self.num_branches * (num_emotions + 3)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size + reliability_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_branches)
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_emotions)
        )

    def _normalise_distribution(self, probs: torch.Tensor) -> torch.Tensor:
        """把任意非负分支证据归一化为概率分布，减少上游尺度差异。"""
        probs = torch.clamp(probs, min=1e-8)
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _reliability_features(
        self,
        probs: torch.Tensor,
        availability: torch.Tensor
    ) -> torch.Tensor:
        confidence = probs.max(dim=1, keepdim=True).values
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / np.log(self.num_emotions)
        return torch.cat([probs, confidence, 1.0 - entropy, availability], dim=1)

    def forward(
        self,
        emotion_debiased: torch.Tensor,
        emotion_adapted: torch.Tensor,
        emotion_graph: torch.Tensor,
        text_features: torch.Tensor,
        graph_availability: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        cpeb_probs = self._normalise_distribution(emotion_debiased)
        meta_probs = self._normalise_distribution(emotion_adapted)
        graph_probs = self._normalise_distribution(emotion_graph)

        branch_probs = [cpeb_probs, meta_probs, graph_probs]
        context = self.context_projector(text_features)

        projected = torch.stack([
            projector(probs)
            for projector, probs in zip(self.branch_projectors, branch_probs)
        ], dim=1)

        always_available = torch.ones_like(graph_availability)
        reliability = torch.cat([
            self._reliability_features(cpeb_probs, always_available),
            self._reliability_features(meta_probs, always_available),
            self._reliability_features(graph_probs, graph_availability),
        ], dim=1)

        gate_logits = self.gate_network(torch.cat([context, reliability], dim=1))
        fusion_weights = F.softmax(gate_logits, dim=1)

        fused_evidence = torch.sum(projected * fusion_weights.unsqueeze(-1), dim=1)
        fusion_logits = self.output_layer(fused_evidence + context)
        emotion_final = F.softmax(fusion_logits, dim=1)

        return {
            'fusion_logits': fusion_logits,
            'emotion_final': emotion_final,
            'fusion_weights': fusion_weights,
            'branch_probabilities': torch.stack(branch_probs, dim=1),
            'reliability_features': reliability
        }


class UnifiedEmotionModel(nn.Module):
    """
    统一情绪监测模型
    整合因果推断、元学习和图神经网络
    """
    
    def __init__(
        self,
        bert_model_name: str = "bert-base-chinese",
        num_emotions: int = 8,
        hidden_size: int = 768,
        graph_hidden_size: int = 256,
        num_graph_layers: int = 3,
        num_attention_heads: int = 8,
        meta_adapter_type: str = "hybrid",
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        temporal_decay_lambda: float = 0.1,
        dropout: float = 0.1,
        device: str = "cpu",
        proto_max_shots: int = 5,
        maml_full_shots: int = 20,
        despair_index: int = 7,
    ):
        """
        初始化统一模型
        
        Args:
            bert_model_name: BERT模型名称
            num_emotions: 情绪类别数
            hidden_size: BERT隐藏层大小
            graph_hidden_size: 图网络隐藏层大小
            num_graph_layers: 图卷积层数
            num_attention_heads: 注意力头数
            meta_adapter_type: 元学习适配器类型 (hybrid/maml/prototypical/lora)
            inner_lr: 元学习内层学习率
            num_inner_steps: 元学习内层步数
            temporal_decay_lambda: 时间衰减系数
            dropout: Dropout概率
            device: 设备
            proto_max_shots: hybrid 模式下纯原型最大 K
            maml_full_shots: hybrid 模式下完全启用 MAML 的 K
            despair_index: 高危情绪「绝望」索引（危机辅任务对齐）
        """
        super().__init__()
        self.num_emotions = num_emotions
        self.hidden_size = hidden_size
        self.device = device
        self.despair_index = despair_index
        
        # 1. CPEB模块：因果个性化基线
        self.cpeb_model = CPEBModel(
            bert_model_name=bert_model_name,
            num_emotions=num_emotions,
            hidden_size=hidden_size,
            dropout=dropout,
            device=device,
            despair_index=despair_index,
        )
        
        # 2. MEA模块：原型+MAML 混合元适配（默认 hybrid）
        self.meta_adapter = MetaEmotionAdapter(
            input_size=hidden_size,
            hidden_size=graph_hidden_size,
            num_emotions=num_emotions,
            adapter_type=meta_adapter_type,
            inner_lr=inner_lr,
            num_inner_steps=num_inner_steps,
            proto_max_shots=proto_max_shots,
            maml_full_shots=maml_full_shots,
        )
        
        # 3. THEGN模块：时序异构图网络
        self.thegn_model = THEGNModel(
            input_size=hidden_size,
            hidden_size=graph_hidden_size,
            num_emotions=num_emotions,
            num_graph_layers=num_graph_layers,
            num_attention_heads=num_attention_heads,
            temporal_decay_lambda=temporal_decay_lambda,
            dropout=dropout
        )
        
        # 无历史或图缺失时的可学习退化路径，避免 THEGN 分支以零向量污染融合。
        self.graph_fallback = nn.Sequential(
            nn.Linear(hidden_size, graph_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden_size, num_emotions)
        )

        # RACF: 证据可靠性感知融合层
        self.fusion_layer = EvidenceAwareFusion(
            num_emotions=num_emotions,
            context_size=hidden_size,
            hidden_size=graph_hidden_size,
            dropout=dropout
        )

        # 危机预警辅助任务头（共享 BERT/CPEB/MEA/THEGN 表征）
        self.crisis_head = nn.Sequential(
            nn.Linear(hidden_size + num_emotions, graph_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden_size, 1)
        )
        
        # 情绪标签
        self.emotion_labels = ['中性', '高兴', '惊讶', '悲伤', '愤怒', '恐惧', '厌恶', '绝望']

    def _move_graph_to_device(self, graph: HeteroGraph, device: torch.device) -> HeteroGraph:
        """保证图张量与当前 batch 在同一设备上，便于 GPU 推理和训练。"""
        return HeteroGraph(
            node_features=graph.node_features.to(device),
            node_types=graph.node_types.to(device),
            node_times=graph.node_times.to(device),
            edge_index=graph.edge_index.to(device),
            edge_types=graph.edge_types.to(device),
            edge_weights=graph.edge_weights.to(device),
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            num_node_types=graph.num_node_types,
            num_edge_types=graph.num_edge_types
        )

    def _graph_fallback_distribution(self, text_features: torch.Tensor) -> torch.Tensor:
        """基于当前文本语义生成 THEGN 缺失时的可训练替代证据。"""
        return F.softmax(self.graph_fallback(text_features), dim=1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        user_ids: torch.Tensor,
        conversation_graphs: Optional[List[HeteroGraph]] = None,
        support_features: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            user_ids: 用户ID [batch_size]
            conversation_graphs: 对话图列表
            support_features: 支持集特征（元学习）
            support_labels: 支持集标签（元学习）
            return_intermediate: 是否返回中间结果
            
        Returns:
            输出字典
        """
        batch_size = input_ids.size(0)
        
        # Stage 1: CPEB - 因果基线建模（含反事实去偏）
        cpeb_outputs = self.cpeb_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            return_baseline=True,
            return_counterfactual=True,
        )
        
        emotion_debiased = cpeb_outputs['emotion_debiased']  # [batch_size, num_emotions]
        text_features = cpeb_outputs['text_features']  # [batch_size, hidden_size]
        
        # Stage 2: MEA - 原型+MAML 混合快速适应
        if support_features is not None and support_labels is not None:
            emotion_adapted = self.meta_adapter(
                text_features,
                support_features,
                support_labels
            )
        else:
            emotion_adapted = self.meta_adapter(text_features)
        
        # Stage 3: THEGN - 图网络时序建模；缺失图走可学习退化路径（cold-start）
        fallback_graph = self._graph_fallback_distribution(text_features)
        graph_emotions = []
        graph_availability = []
        for i in range(batch_size):
            has_graph = (
                conversation_graphs is not None
                and i < len(conversation_graphs)
                and conversation_graphs[i] is not None
                and conversation_graphs[i].num_nodes > 0
                and conversation_graphs[i].num_edges > 0
            )

            if has_graph:
                graph = self._move_graph_to_device(conversation_graphs[i], text_features.device)
                graph_output = self.thegn_model(graph)
                # 取最后一个节点的情绪（当前轮次）
                graph_emotions.append(graph_output['probabilities'][-1])
                graph_availability.append(1.0)
            else:
                graph_emotions.append(fallback_graph[i])
                graph_availability.append(0.0)

        emotion_graph = torch.stack(graph_emotions)
        graph_availability = torch.tensor(
            graph_availability,
            dtype=text_features.dtype,
            device=text_features.device
        ).unsqueeze(1)
        
        # Stage 4: RACF 证据可靠性感知融合
        fusion_outputs = self.fusion_layer(
            emotion_debiased=emotion_debiased,
            emotion_adapted=emotion_adapted,
            emotion_graph=emotion_graph,
            text_features=text_features,
            graph_availability=graph_availability
        )
        fusion_logits = fusion_outputs['fusion_logits']
        emotion_final = fusion_outputs['emotion_final']

        # Stage 5: 危机预警辅助头（共享表征）
        crisis_input = torch.cat([text_features, fusion_logits], dim=-1)
        crisis_logits = self.crisis_head(crisis_input).squeeze(-1)
        crisis_prob = torch.sigmoid(crisis_logits)
        
        outputs = {
            'emotion_final': emotion_final,
            'fusion_logits': fusion_logits,
            'fusion_weights': fusion_outputs['fusion_weights'],
            'graph_availability': graph_availability,
            'crisis_logits': crisis_logits,
            'crisis_prob': crisis_prob,
            # 训练损失默认需要的 CPEB 中间量
            'emotion_raw': cpeb_outputs['emotion_raw'],
            'emotion_debiased': emotion_debiased,
            'emotion_adapted': emotion_adapted,
            'emotion_graph': emotion_graph,
            'baseline_mean': cpeb_outputs['baseline_mean'],
            'baseline_logvar': cpeb_outputs['baseline_logvar'],
            'text_features': text_features,
        }

        if 'causal_aux_losses' in cpeb_outputs:
            outputs['causal_aux_losses'] = cpeb_outputs['causal_aux_losses']
            outputs['emotion_counterfactual'] = cpeb_outputs['emotion_counterfactual']
        
        if return_intermediate:
            outputs.update({
                'branch_probabilities': fusion_outputs['branch_probabilities'],
                'reliability_features': fusion_outputs['reliability_features'],
                'baseline': cpeb_outputs.get('baseline'),
                'despair_prob': emotion_final[:, self.despair_index],
            })
        
        return outputs
    
    def predict(
        self,
        text: str,
        user_id: int,
        conversation_history: Optional[List[str]] = None,
        support_examples: Optional[List[Tuple[str, int]]] = None,
        conversation_role_types: Optional[List[int]] = None,
    ) -> Dict:
        """
        预测单个文本的情绪
        
        Args:
            text: 输入文本
            user_id: 用户ID
            conversation_history: 对话历史
            support_examples: 支持集样例 [(text, label), ...]
            conversation_role_types: 与 history+[text] 等长的节点类型
                （0=client，1=counselor）；缺省则全 0
            
        Returns:
            预测结果字典
        """
        self.eval()
        
        # 准备输入
        if self.cpeb_model.tokenizer is None:
            raise RuntimeError(
                "UnifiedEmotionModel: BERT tokenizer 未加载成功，拒绝返回随机分布。"
                "请检查 bert_model_name 与网络/本地缓存。"
            )
        
        tokenizer = self.cpeb_model.tokenizer
        
        # 分词
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        user_ids = torch.tensor([user_id]).to(self.device)
        
        # 准备支持集
        support_features = None
        support_labels = None
        if support_examples is not None and len(support_examples) > 0:
            support_texts = [ex[0] for ex in support_examples]
            support_labels_list = [ex[1] for ex in support_examples]
            
            support_inputs = tokenizer(
                support_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # 编码支持集
            with torch.no_grad():
                support_ids = support_inputs['input_ids'].to(self.device)
                support_mask = support_inputs['attention_mask'].to(self.device)
                support_features = self.cpeb_model.encode_text(support_ids, support_mask)
                support_labels = torch.tensor(support_labels_list).to(self.device)
        
        # 准备对话图
        conversation_graphs = None
        if conversation_history is not None and len(conversation_history) > 0:
            # 编码历史对话
            history_inputs = tokenizer(
                conversation_history + [text],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                history_ids = history_inputs['input_ids'].to(self.device)
                history_mask = history_inputs['attention_mask'].to(self.device)
                history_features = self.cpeb_model.encode_text(history_ids, history_mask)
            
            turns = conversation_history + [text]
            node_type_ids = None
            if conversation_role_types is not None and len(conversation_role_types) == len(turns):
                node_type_ids = list(conversation_role_types)
            # 创建图
            graph = create_dialogue_graph(
                turns,
                history_features,
                node_type_ids=node_type_ids,
            )
            conversation_graphs = [graph]
        
        # 前向传播
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask,
                user_ids,
                conversation_graphs,
                support_features,
                support_labels,
                return_intermediate=True
            )
        
        # 解析结果
        emotion_probs = outputs['emotion_final'][0].cpu().numpy()
        dominant_idx = emotion_probs.argmax()
        
        result = {
            'emotion_distribution': dict(zip(self.emotion_labels, emotion_probs)),
            'dominant_emotion': self.emotion_labels[dominant_idx],
            'confidence': float(emotion_probs[dominant_idx]),
            'crisis_prob': float(outputs['crisis_prob'][0].item()),
            'fusion_weights': {
                'CPEB': float(outputs['fusion_weights'][0, 0].item()),
                'MEA': float(outputs['fusion_weights'][0, 1].item()),
                'THEGN': float(outputs['fusion_weights'][0, 2].item()),
            },
            'graph_mode': (
                'history-aware'
                if outputs['graph_availability'][0].item() > 0.5
                else 'cold-start'
            ),
            'intermediate_results': {
                'cpeb_emotion': outputs['emotion_debiased'][0].cpu().numpy().tolist(),
                'meta_emotion': outputs['emotion_adapted'][0].cpu().numpy().tolist(),
                'graph_emotion': outputs['emotion_graph'][0].cpu().numpy().tolist(),
                'fusion_weights': outputs['fusion_weights'][0].cpu().numpy().tolist(),
                'graph_available': bool(outputs['graph_availability'][0].item() > 0.5)
            }
        }
        
        return result
    
    def get_model_size(self) -> Dict[str, int]:
        """获取模型大小信息"""
        cpeb_params = sum(p.numel() for p in self.cpeb_model.parameters())
        meta_params = sum(p.numel() for p in self.meta_adapter.parameters())
        thegn_params = sum(p.numel() for p in self.thegn_model.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())
        crisis_params = sum(p.numel() for p in self.crisis_head.parameters())
        fallback_params = sum(p.numel() for p in self.graph_fallback.parameters())
        total_params = (
            cpeb_params + meta_params + thegn_params
            + fusion_params + crisis_params + fallback_params
        )
        
        return {
            'cpeb': cpeb_params,
            'meta': meta_params,
            'thegn': thegn_params,
            'fusion': fusion_params,
            'crisis': crisis_params,
            'graph_fallback': fallback_params,
            'total': total_params
        }


# 使用示例
if __name__ == '__main__':
    print("正在初始化统一情绪监测模型...")
    
    # 创建模型
    model = UnifiedEmotionModel(
        bert_model_name="bert-base-chinese",
        num_emotions=8,
        hidden_size=768,
        graph_hidden_size=256,
        num_graph_layers=3,
        num_attention_heads=8,
        meta_adapter_type="hybrid",
        device='cpu'
    )
    
    # 模型大小
    model_size = model.get_model_size()
    print("\n模型参数量:")
    print(f"  CPEB: {model_size['cpeb'] / 1e6:.2f}M")
    print(f"  MEA: {model_size['meta'] / 1e6:.2f}M")
    print(f"  THEGN: {model_size['thegn'] / 1e6:.2f}M")
    print(f"  Fusion(RACF): {model_size['fusion'] / 1e6:.2f}M")
    print(f"  Crisis: {model_size['crisis'] / 1e6:.2f}M")
    print(f"  GraphFallback: {model_size['graph_fallback'] / 1e6:.2f}M")
    print(f"  总计: {model_size['total'] / 1e6:.2f}M")
    
    # 测试预测
    print("\n" + "="*60)
    print("测试情绪预测")
    print("="*60)
    
    test_cases = [
        {
            'text': "今天心情很好",
            'user_id': 1,
            'conversation_history': None,
            'support_examples': None
        },
        {
            'text': "我真的撑不住了",
            'user_id': 1,
            'conversation_history': ["最近压力好大", "睡不好觉", "感觉很累"],
            'support_examples': None
        },
        {
            'text': "有点难过",
            'user_id': 2,
            'conversation_history': None,
            'support_examples': [("今天开心", 1), ("很伤心", 3), ("有点生气", 4)]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n案例 {i}:")
        print(f"文本: {case['text']}")
        print(f"用户ID: {case['user_id']}")
        if case['conversation_history']:
            print(f"对话历史: {case['conversation_history']}")
        if case['support_examples']:
            print(f"支持集: {case['support_examples']}")
        
        result = model.predict(**case)
        
        print(f"\n预测结果:")
        print(f"  主要情绪: {result['dominant_emotion']}")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"  情绪分布:")
        for emotion, prob in result['emotion_distribution'].items():
            if prob > 0.05:
                print(f"    {emotion}: {prob:.2%}")
    
    print("\n" + "="*60)
    print("测试完成!")


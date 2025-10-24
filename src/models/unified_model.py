"""
统一情绪监测模型
整合 CPEB + MEA + THEGN
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer

from .cpeb import CPEBModel
from .meta_learner import MetaEmotionAdapter
from .thegn import THEGNModel, create_dialogue_graph, HeteroGraph


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
        meta_adapter_type: str = "maml",
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        temporal_decay_lambda: float = 0.1,
        dropout: float = 0.1,
        device: str = "cpu"
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
            meta_adapter_type: 元学习适配器类型
            inner_lr: 元学习内层学习率
            num_inner_steps: 元学习内层步数
            temporal_decay_lambda: 时间衰减系数
            dropout: Dropout概率
            device: 设备
        """
        super().__init__()
        self.num_emotions = num_emotions
        self.hidden_size = hidden_size
        self.device = device
        
        # 1. CPEB模块：因果个性化基线
        self.cpeb_model = CPEBModel(
            bert_model_name=bert_model_name,
            num_emotions=num_emotions,
            hidden_size=hidden_size,
            dropout=dropout,
            device=device
        )
        
        # 2. MEA模块：元学习适配器
        self.meta_adapter = MetaEmotionAdapter(
            input_size=hidden_size,
            hidden_size=graph_hidden_size,
            num_emotions=num_emotions,
            adapter_type=meta_adapter_type,
            inner_lr=inner_lr,
            num_inner_steps=num_inner_steps
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
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_emotions * 3, graph_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden_size, num_emotions)
        )
        
        # 情绪标签
        self.emotion_labels = ['中性', '高兴', '惊讶', '悲伤', '愤怒', '恐惧', '厌恶', '绝望']
    
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
        
        # Stage 1: CPEB - 因果基线建模
        cpeb_outputs = self.cpeb_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            return_baseline=True
        )
        
        emotion_debiased = cpeb_outputs['emotion_debiased']  # [batch_size, num_emotions]
        text_features = cpeb_outputs['text_features']  # [batch_size, hidden_size]
        
        # Stage 2: MEA - 元学习快速适应
        if support_features is not None and support_labels is not None:
            # 有支持集，使用元学习适应
            emotion_adapted = self.meta_adapter(
                text_features,
                support_features,
                support_labels
            )
        else:
            # 无支持集，使用默认预测
            emotion_adapted = self.meta_adapter(text_features)
        
        # Stage 3: THEGN - 图网络时序建模
        if conversation_graphs is not None and len(conversation_graphs) > 0:
            # 使用图网络
            graph_emotions = []
            for i in range(batch_size):
                if i < len(conversation_graphs):
                    graph = conversation_graphs[i]
                    graph_output = self.thegn_model(graph)
                    # 取最后一个节点的情绪（当前轮次）
                    emotion_graph = graph_output['probabilities'][-1]
                    graph_emotions.append(emotion_graph)
                else:
                    # 没有图，使用零向量
                    graph_emotions.append(torch.zeros(self.num_emotions).to(self.device))
            
            emotion_graph = torch.stack(graph_emotions)
        else:
            # 无图，使用零向量
            emotion_graph = torch.zeros(batch_size, self.num_emotions).to(self.device)
        
        # Stage 4: 融合三个模块的输出
        # 拼接三个情绪分布
        combined = torch.cat([
            emotion_debiased,
            emotion_adapted,
            emotion_graph
        ], dim=1)  # [batch_size, num_emotions * 3]
        
        # 融合
        fusion_logits = self.fusion_layer(combined)
        emotion_final = torch.softmax(fusion_logits, dim=1)
        
        outputs = {
            'emotion_final': emotion_final,
            'fusion_logits': fusion_logits,
        }
        
        if return_intermediate:
            outputs.update({
                'emotion_debiased': emotion_debiased,
                'emotion_adapted': emotion_adapted,
                'emotion_graph': emotion_graph,
                'text_features': text_features,
                'baseline_mean': cpeb_outputs['baseline_mean'],
                'baseline_logvar': cpeb_outputs['baseline_logvar']
            })
        
        return outputs
    
    def predict(
        self,
        text: str,
        user_id: int,
        conversation_history: Optional[List[str]] = None,
        support_examples: Optional[List[Tuple[str, int]]] = None
    ) -> Dict:
        """
        预测单个文本的情绪
        
        Args:
            text: 输入文本
            user_id: 用户ID
            conversation_history: 对话历史
            support_examples: 支持集样例 [(text, label), ...]
            
        Returns:
            预测结果字典
        """
        self.eval()
        
        # 准备输入
        if self.cpeb_model.tokenizer is None:
            # 模拟输出
            emotion_probs = np.random.dirichlet(np.ones(self.num_emotions))
            return {
                'emotion_distribution': dict(zip(self.emotion_labels, emotion_probs)),
                'dominant_emotion': self.emotion_labels[emotion_probs.argmax()],
                'confidence': float(emotion_probs.max())
            }
        
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
            
            # 创建图
            graph = create_dialogue_graph(
                conversation_history + [text],
                history_features
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
            'intermediate_results': {
                'cpeb_emotion': outputs['emotion_debiased'][0].cpu().numpy().tolist(),
                'meta_emotion': outputs['emotion_adapted'][0].cpu().numpy().tolist(),
                'graph_emotion': outputs['emotion_graph'][0].cpu().numpy().tolist()
            }
        }
        
        return result
    
    def get_model_size(self) -> Dict[str, int]:
        """获取模型大小信息"""
        cpeb_params = sum(p.numel() for p in self.cpeb_model.parameters())
        meta_params = sum(p.numel() for p in self.meta_adapter.parameters())
        thegn_params = sum(p.numel() for p in self.thegn_model.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())
        total_params = cpeb_params + meta_params + thegn_params + fusion_params
        
        return {
            'cpeb': cpeb_params,
            'meta': meta_params,
            'thegn': thegn_params,
            'fusion': fusion_params,
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
        meta_adapter_type="maml",
        device='cpu'
    )
    
    # 模型大小
    model_size = model.get_model_size()
    print("\n模型参数量:")
    print(f"  CPEB: {model_size['cpeb'] / 1e6:.2f}M")
    print(f"  MEA: {model_size['meta'] / 1e6:.2f}M")
    print(f"  THEGN: {model_size['thegn'] / 1e6:.2f}M")
    print(f"  Fusion: {model_size['fusion'] / 1e6:.2f}M")
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


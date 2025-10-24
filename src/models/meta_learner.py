"""
MEA: Meta-Emotion Adapter
元学习驱动的快速适应

基于MAML实现少样本快速个性化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


class PrototypicalNetwork(nn.Module):
    """
    原型网络
    学习用户情绪表达的原型表示
    """
    
    def __init__(self, input_size: int = 768, hidden_size: int = 256, num_emotions: int = 8):
        """
        初始化原型网络
        
        Args:
            input_size: 输入特征大小
            hidden_size: 隐藏层大小
            num_emotions: 情绪类别数
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions
        
        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码特征
        
        Args:
            x: 输入特征 [batch_size, input_size]
            
        Returns:
            编码后的特征 [batch_size, hidden_size]
        """
        return self.encoder(x)
    
    def compute_prototypes(
        self, 
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个类别的原型
        
        Args:
            support_features: 支持集特征 [n_support, hidden_size]
            support_labels: 支持集标签 [n_support]
            
        Returns:
            原型向量 [num_emotions, hidden_size]
        """
        prototypes = []
        for emotion_id in range(self.num_emotions):
            mask = (support_labels == emotion_id)
            if mask.sum() > 0:
                prototype = support_features[mask].mean(dim=0)
            else:
                # 如果该类别没有样本，使用零向量
                prototype = torch.zeros(self.hidden_size).to(support_features.device)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def compute_distances(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        计算查询样本到原型的距离
        
        Args:
            query_features: 查询集特征 [n_query, hidden_size]
            prototypes: 原型向量 [num_emotions, hidden_size]
            
        Returns:
            距离矩阵 [n_query, num_emotions]
        """
        # 欧氏距离
        n_query = query_features.size(0)
        n_proto = prototypes.size(0)
        
        query_expanded = query_features.unsqueeze(1).expand(n_query, n_proto, -1)
        proto_expanded = prototypes.unsqueeze(0).expand(n_query, n_proto, -1)
        
        distances = torch.pow(query_expanded - proto_expanded, 2).sum(dim=2)
        return distances
    
    def predict(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor
    ) -> torch.Tensor:
        """
        基于原型进行预测
        
        Args:
            support_features: 支持集特征
            support_labels: 支持集标签
            query_features: 查询集特征
            
        Returns:
            预测概率 [n_query, num_emotions]
        """
        # 计算原型
        prototypes = self.compute_prototypes(support_features, support_labels)
        
        # 计算距离
        distances = self.compute_distances(query_features, prototypes)
        
        # 转换为概率（负距离的softmax）
        logits = -distances
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities


class MAMLAdapter(nn.Module):
    """
    MAML适配器
    基于Model-Agnostic Meta-Learning的快速适应模块
    """
    
    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 256,
        num_emotions: int = 8,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5
    ):
        """
        初始化MAML适配器
        
        Args:
            input_size: 输入特征大小
            hidden_size: 隐藏层大小
            num_emotions: 情绪类别数
            inner_lr: 内层学习率
            num_inner_steps: 内层更新步数
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        # 快速适应网络（轻量级）
        self.adapter = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.1)),
            ('fc2', nn.Linear(hidden_size, hidden_size // 2)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.1)),
            ('fc3', nn.Linear(hidden_size // 2, num_emotions))
        ]))
    
    def forward(self, x: torch.Tensor, params: Optional[OrderedDict] = None) -> torch.Tensor:
        """
        前向传播（可使用自定义参数）
        
        Args:
            x: 输入特征 [batch_size, input_size]
            params: 自定义参数（用于MAML）
            
        Returns:
            输出logits [batch_size, num_emotions]
        """
        if params is None:
            return self.adapter(x)
        
        # 使用自定义参数进行前向传播
        x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        
        return x
    
    def adapt(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> OrderedDict:
        """
        使用支持集进行快速适应
        
        Args:
            support_features: 支持集特征 [n_support, input_size]
            support_labels: 支持集标签 [n_support]
            
        Returns:
            适应后的参数
        """
        # 复制当前参数
        adapted_params = OrderedDict()
        for name, param in self.adapter.named_parameters():
            adapted_params[name] = param.clone()
        
        # 内层循环：使用支持集更新参数
        for _ in range(self.num_inner_steps):
            # 前向传播
            logits = self.forward(support_features, adapted_params)
            loss = F.cross_entropy(logits, support_labels)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=True  # 保留计算图用于元学习
            )
            
            # 更新参数
            adapted_params = OrderedDict([
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(adapted_params.items(), grads)
            ])
        
        return adapted_params
    
    def predict_with_adaptation(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor
    ) -> torch.Tensor:
        """
        先适应，后预测
        
        Args:
            support_features: 支持集特征
            support_labels: 支持集标签
            query_features: 查询集特征
            
        Returns:
            预测概率
        """
        # 快速适应
        adapted_params = self.adapt(support_features, support_labels)
        
        # 使用适应后的参数进行预测
        logits = self.forward(query_features, adapted_params)
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities


class LoRAAdapter(nn.Module):
    """
    LoRA低秩适配器
    使用低秩矩阵进行参数高效的微调
    """
    
    def __init__(
        self,
        input_size: int = 768,
        output_size: int = 768,
        rank: int = 8,
        alpha: float = 16.0
    ):
        """
        初始化LoRA适配器
        
        Args:
            input_size: 输入大小
            output_size: 输出大小
            rank: 低秩维度
            alpha: 缩放因子
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 低秩矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.randn(input_size, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, output_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, input_size]
            
        Returns:
            输出 [batch_size, output_size]
        """
        # LoRA: x @ (A @ B) * scaling
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class MetaEmotionAdapter(nn.Module):
    """
    元情绪适配器（整合模块）
    结合原型网络、MAML和LoRA
    """
    
    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 256,
        num_emotions: int = 8,
        adapter_type: str = "maml",  # "maml" or "prototypical" or "lora"
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        lora_rank: int = 8
    ):
        """
        初始化元情绪适配器
        
        Args:
            input_size: 输入特征大小
            hidden_size: 隐藏层大小
            num_emotions: 情绪类别数
            adapter_type: 适配器类型
            inner_lr: 内层学习率
            num_inner_steps: 内层更新步数
            lora_rank: LoRA秩
        """
        super().__init__()
        self.adapter_type = adapter_type
        self.num_emotions = num_emotions
        
        # 选择适配器
        if adapter_type == "maml":
            self.adapter = MAMLAdapter(
                input_size, hidden_size, num_emotions,
                inner_lr, num_inner_steps
            )
        elif adapter_type == "prototypical":
            self.adapter = PrototypicalNetwork(
                input_size, hidden_size, num_emotions
            )
        elif adapter_type == "lora":
            self.lora = LoRAAdapter(input_size, input_size, lora_rank)
            self.classifier = nn.Linear(input_size, num_emotions)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    def forward(
        self,
        features: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch_size, input_size]
            support_features: 支持集特征（用于少样本适应）
            support_labels: 支持集标签
            
        Returns:
            情绪概率分布
        """
        if self.adapter_type == "lora":
            # LoRA适配
            adapted_features = features + self.lora(features)
            logits = self.classifier(adapted_features)
            return F.softmax(logits, dim=1)
        
        elif self.adapter_type == "maml":
            if support_features is not None and support_labels is not None:
                # 使用支持集进行快速适应
                return self.adapter.predict_with_adaptation(
                    support_features, support_labels, features
                )
            else:
                # 直接预测
                logits = self.adapter(features)
                return F.softmax(logits, dim=1)
        
        elif self.adapter_type == "prototypical":
            if support_features is not None and support_labels is not None:
                # 原型网络预测
                query_encoded = self.adapter(features)
                support_encoded = self.adapter(support_features)
                return self.adapter.predict(
                    support_encoded, support_labels, query_encoded
                )
            else:
                # 无支持集时，返回均匀分布
                batch_size = features.size(0)
                return torch.ones(batch_size, self.num_emotions).to(features.device) / self.num_emotions


# 使用示例
if __name__ == '__main__':
    print("正在测试元学习适配器...")
    
    # 参数
    input_size = 768
    num_emotions = 8
    n_support = 5
    n_query = 3
    
    # 模拟数据
    support_features = torch.randn(n_support, input_size)
    support_labels = torch.randint(0, num_emotions, (n_support,))
    query_features = torch.randn(n_query, input_size)
    
    print("\n" + "="*60)
    print("测试不同类型的元学习适配器")
    print("="*60)
    
    # 测试MAML
    print("\n1. MAML适配器")
    maml_adapter = MetaEmotionAdapter(
        input_size=input_size,
        num_emotions=num_emotions,
        adapter_type="maml",
        inner_lr=0.01,
        num_inner_steps=5
    )
    maml_adapter.eval()
    
    predictions = maml_adapter(
        query_features,
        support_features,
        support_labels
    )
    print(f"输出形状: {predictions.shape}")
    print(f"预测概率 (样本1): {predictions[0].detach().numpy()}")
    
    # 测试原型网络
    print("\n2. 原型网络")
    proto_adapter = MetaEmotionAdapter(
        input_size=input_size,
        num_emotions=num_emotions,
        adapter_type="prototypical"
    )
    proto_adapter.eval()
    
    predictions = proto_adapter(
        query_features,
        support_features,
        support_labels
    )
    print(f"输出形状: {predictions.shape}")
    print(f"预测概率 (样本1): {predictions[0].detach().numpy()}")
    
    # 测试LoRA
    print("\n3. LoRA适配器")
    lora_adapter = MetaEmotionAdapter(
        input_size=input_size,
        num_emotions=num_emotions,
        adapter_type="lora",
        lora_rank=8
    )
    lora_adapter.eval()
    
    predictions = lora_adapter(query_features)
    print(f"输出形状: {predictions.shape}")
    print(f"预测概率 (样本1): {predictions[0].detach().numpy()}")
    
    # 参数量统计
    print("\n" + "="*60)
    print("参数量统计")
    print("="*60)
    for name, adapter in [("MAML", maml_adapter), ("Prototypical", proto_adapter), ("LoRA", lora_adapter)]:
        total_params = sum(p.numel() for p in adapter.parameters())
        trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"{name}: 总参数={total_params:,}, 可训练={trainable_params:,}")
    
    print("\n" + "="*60)
    print("测试完成!")


"""
损失函数模块
多任务损失函数组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EmotionClassificationLoss(nn.Module):
    """情绪分类损失"""
    
    def __init__(self, num_emotions: int = 8, label_smoothing: float = 0.0):
        """
        初始化
        
        Args:
            num_emotions: 情绪类别数
            label_smoothing: 标签平滑系数
        """
        super().__init__()
        self.num_emotions = num_emotions
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            logits: 预测logits [batch_size, num_emotions]
            labels: 真实标签 [batch_size]
            weights: 样本权重 [batch_size]
            
        Returns:
            损失值
        """
        if self.label_smoothing > 0:
            # 标签平滑
            confidence = 1.0 - self.label_smoothing
            smooth_value = self.label_smoothing / (self.num_emotions - 1)
            
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smooth_value)
            true_dist.scatter_(1, labels.unsqueeze(1), confidence)
            
            loss = F.kl_div(
                F.log_softmax(logits, dim=1),
                true_dist,
                reduction='none'
            ).sum(dim=1)
        else:
            # 标准交叉熵
            loss = F.cross_entropy(logits, labels, reduction='none')
        
        # 应用权重
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


class CausalConsistencyLoss(nn.Module):
    """
    因果一致性损失
    确保因果干预后的预测与反事实预测一致
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        emotion_raw: torch.Tensor,
        emotion_debiased: torch.Tensor,
        baseline_mean: torch.Tensor,
        baseline_logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            emotion_raw: 原始情绪分布 [batch_size, num_emotions]
            emotion_debiased: 去偏后的情绪分布 [batch_size, num_emotions]
            baseline_mean: 基线均值 [batch_size, num_emotions]
            baseline_logvar: 基线对数方差 [batch_size, num_emotions]
            
        Returns:
            损失值
        """
        # KL散度损失：约束去偏后的分布不要偏离原始分布太远
        kl_loss = F.kl_div(
            torch.log(emotion_debiased + 1e-10),
            emotion_raw,
            reduction='batchmean'
        )
        
        # VAE的KL散度正则化：约束基线分布接近标准正态分布
        kl_reg = -0.5 * torch.sum(
            1 + baseline_logvar - baseline_mean.pow(2) - baseline_logvar.exp(),
            dim=1
        ).mean()
        
        return kl_loss + 0.1 * kl_reg


class MetaLearningLoss(nn.Module):
    """
    元学习损失
    MAML二阶梯度损失
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        query_logits: torch.Tensor,
        query_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query_logits: 查询集预测logits [batch_size, num_emotions]
            query_labels: 查询集标签 [batch_size]
            
        Returns:
            损失值
        """
        return F.cross_entropy(query_logits, query_labels)


class GraphRegularizationLoss(nn.Module):
    """
    图结构正则化损失
    平滑性约束：相连节点的特征应该相似
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [num_nodes, feature_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weights: 边权重 [num_edges]
            
        Returns:
            损失值
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0).to(node_features.device)
        
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        source_features = node_features[source_nodes]
        target_features = node_features[target_nodes]
        
        # L2距离
        distances = torch.pow(source_features - target_features, 2).sum(dim=1)
        
        # 应用边权重
        if edge_weights is not None:
            distances = distances * edge_weights
        
        return distances.mean()


class UnifiedLoss(nn.Module):
    """
    统一损失函数
    组合多个损失项
    """
    
    def __init__(
        self,
        num_emotions: int = 8,
        loss_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.1
    ):
        """
        初始化
        
        Args:
            num_emotions: 情绪类别数
            loss_weights: 损失权重字典
            label_smoothing: 标签平滑系数
        """
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {
                'emotion': 1.0,
                'causal': 0.5,
                'meta': 0.3,
                'graph': 0.2
            }
        
        self.loss_weights = loss_weights
        
        # 各个损失函数
        self.emotion_loss = EmotionClassificationLoss(num_emotions, label_smoothing)
        self.causal_loss = CausalConsistencyLoss()
        self.meta_loss = MetaLearningLoss()
        self.graph_loss = GraphRegularizationLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        compute_causal: bool = True,
        compute_meta: bool = False,
        compute_graph: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            outputs: 模型输出字典
            labels: 真实标签
            compute_causal: 是否计算因果损失
            compute_meta: 是否计算元学习损失
            compute_graph: 是否计算图正则化损失
            **kwargs: 其他参数
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 1. 主情绪分类损失
        if 'fusion_logits' in outputs:
            losses['emotion'] = self.emotion_loss(
                outputs['fusion_logits'],
                labels
            )
        elif 'emotion_logits' in outputs:
            losses['emotion'] = self.emotion_loss(
                outputs['emotion_logits'],
                labels
            )
        else:
            losses['emotion'] = torch.tensor(0.0)
        
        # 2. 因果一致性损失
        if compute_causal and all(k in outputs for k in ['emotion_raw', 'emotion_debiased', 'baseline_mean', 'baseline_logvar']):
            losses['causal'] = self.causal_loss(
                outputs['emotion_raw'],
                outputs['emotion_debiased'],
                outputs['baseline_mean'],
                outputs['baseline_logvar']
            )
        else:
            losses['causal'] = torch.tensor(0.0)
        
        # 3. 元学习损失
        if compute_meta and 'query_logits' in kwargs:
            losses['meta'] = self.meta_loss(
                kwargs['query_logits'],
                kwargs['query_labels']
            )
        else:
            losses['meta'] = torch.tensor(0.0)
        
        # 4. 图正则化损失
        if compute_graph and all(k in kwargs for k in ['node_features', 'edge_index']):
            losses['graph'] = self.graph_loss(
                kwargs['node_features'],
                kwargs['edge_index'],
                kwargs.get('edge_weights', None)
            )
        else:
            losses['graph'] = torch.tensor(0.0)
        
        # 5. 总损失
        total_loss = sum(
            self.loss_weights.get(name, 1.0) * loss
            for name, loss in losses.items()
        )
        
        losses['total'] = total_loss
        
        return losses


# 使用示例
if __name__ == '__main__':
    print("测试损失函数...")
    
    batch_size = 8
    num_emotions = 8
    
    # 模拟数据
    fusion_logits = torch.randn(batch_size, num_emotions)
    labels = torch.randint(0, num_emotions, (batch_size,))
    emotion_raw = torch.softmax(torch.randn(batch_size, num_emotions), dim=1)
    emotion_debiased = torch.softmax(torch.randn(batch_size, num_emotions), dim=1)
    baseline_mean = torch.randn(batch_size, num_emotions)
    baseline_logvar = torch.randn(batch_size, num_emotions)
    
    # 创建统一损失函数
    criterion = UnifiedLoss(num_emotions=num_emotions)
    
    # 模拟输出
    outputs = {
        'fusion_logits': fusion_logits,
        'emotion_raw': emotion_raw,
        'emotion_debiased': emotion_debiased,
        'baseline_mean': baseline_mean,
        'baseline_logvar': baseline_logvar
    }
    
    # 计算损失
    losses = criterion(
        outputs,
        labels,
        compute_causal=True,
        compute_meta=False,
        compute_graph=False
    )
    
    print("\n损失值:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    print("\n测试完成!")


"""
损失函数模块
多任务损失：
  L = L_emotion + λ1 L_crisis + λ2 L_causal + λ3 L_graph + λ4 L_meta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EmotionClassificationLoss(nn.Module):
    """情绪分类损失"""
    
    def __init__(self, num_emotions: int = 8, label_smoothing: float = 0.0):
        super().__init__()
        self.num_emotions = num_emotions
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.label_smoothing > 0:
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
            loss = F.cross_entropy(logits, labels, reduction='none')
        
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


class CrisisDetectionLoss(nn.Module):
    """危机二分类辅助损失（支持类别不平衡的 pos_weight）"""

    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        crisis_logits: torch.Tensor,
        crisis_labels: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.tensor(
            [self.pos_weight],
            device=crisis_logits.device,
            dtype=crisis_logits.dtype
        )
        return F.binary_cross_entropy_with_logits(
            crisis_logits,
            crisis_labels.float(),
            pos_weight=weight
        )


class CausalConsistencyLoss(nn.Module):
    """
    因果一致性损失
    - 去偏分布相对原始分布的轻度 KL
    - 用户基线 KL 正则（防过拟合）
    - 反事实 / 绝望一致性（若模型提供 causal_aux_losses）
    """
    
    def __init__(
        self,
        baseline_kl_weight: float = 0.1,
        despair_weight: float = 0.5,
        counterfactual_weight: float = 0.2,
    ):
        super().__init__()
        self.baseline_kl_weight = baseline_kl_weight
        self.despair_weight = despair_weight
        self.counterfactual_weight = counterfactual_weight
    
    def forward(
        self,
        emotion_raw: torch.Tensor,
        emotion_debiased: torch.Tensor,
        baseline_mean: torch.Tensor,
        baseline_logvar: torch.Tensor,
        causal_aux_losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        kl_loss = F.kl_div(
            torch.log(emotion_debiased.clamp_min(1e-10)),
            emotion_raw.clamp_min(1e-10),
            reduction='batchmean'
        )
        
        kl_reg = -0.5 * torch.mean(
            1 + baseline_logvar - baseline_mean.pow(2) - baseline_logvar.exp()
        )

        total = kl_loss + self.baseline_kl_weight * kl_reg

        if causal_aux_losses:
            if 'despair_consistency' in causal_aux_losses:
                total = total + self.despair_weight * causal_aux_losses['despair_consistency']
            if 'counterfactual_consistency' in causal_aux_losses:
                total = total + self.counterfactual_weight * causal_aux_losses['counterfactual_consistency']
            if 'baseline_kl' in causal_aux_losses:
                # 与上方 kl_reg 同源时可再轻度叠加，强化论文中的基线约束叙述
                total = total + 0.05 * causal_aux_losses['baseline_kl']

        return total


class MetaLearningLoss(nn.Module):
    """元学习损失：查询集交叉熵"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        query_logits: torch.Tensor,
        query_labels: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(query_logits, query_labels)


class GraphRegularizationLoss(nn.Module):
    """图结构正则化：相连节点特征应相似"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=node_features.device)
        
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        source_features = node_features[source_nodes]
        target_features = node_features[target_nodes]
        
        distances = torch.pow(source_features - target_features, 2).sum(dim=1)
        
        if edge_weights is not None:
            distances = distances * edge_weights
        
        return distances.mean()


class UnifiedLoss(nn.Module):
    """
    统一多任务损失：
      L = L_emotion + λ1 L_crisis + λ2 L_causal + λ3 L_graph + λ4 L_meta
    """
    
    def __init__(
        self,
        num_emotions: int = 8,
        loss_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.1,
        crisis_pos_weight: float = 2.0,
        despair_index: int = 7,
    ):
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {
                'emotion': 1.0,
                'crisis': 0.5,
                'causal': 0.5,
                'meta': 0.3,
                'graph': 0.2,
            }
        
        self.loss_weights = loss_weights
        self.despair_index = despair_index
        
        self.emotion_loss = EmotionClassificationLoss(num_emotions, label_smoothing)
        self.crisis_loss = CrisisDetectionLoss(pos_weight=crisis_pos_weight)
        self.causal_loss = CausalConsistencyLoss()
        self.meta_loss = MetaLearningLoss()
        self.graph_loss = GraphRegularizationLoss()

    @staticmethod
    def derive_crisis_labels(
        emotion_labels: torch.Tensor,
        despair_index: int = 7,
        explicit_crisis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """若无显式危机标签，用「绝望」类作为弱监督危机标签。"""
        if explicit_crisis is not None:
            return explicit_crisis.float()
        return (emotion_labels == despair_index).float()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        crisis_labels: Optional[torch.Tensor] = None,
        compute_causal: bool = True,
        compute_meta: bool = False,
        compute_graph: bool = False,
        compute_crisis: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        zero = torch.tensor(0.0, device=labels.device if isinstance(labels, torch.Tensor) else 'cpu')
        
        # 1. 主情绪分类损失
        if 'fusion_logits' in outputs:
            losses['emotion'] = self.emotion_loss(outputs['fusion_logits'], labels)
        elif 'emotion_logits' in outputs:
            losses['emotion'] = self.emotion_loss(outputs['emotion_logits'], labels)
        else:
            losses['emotion'] = zero
        
        # 2. 危机二分类辅助损失
        if compute_crisis and 'crisis_logits' in outputs:
            derived = self.derive_crisis_labels(
                labels, self.despair_index, crisis_labels
            )
            losses['crisis'] = self.crisis_loss(outputs['crisis_logits'], derived)
        else:
            losses['crisis'] = zero
        
        # 3. 因果一致性损失
        if compute_causal and all(
            k in outputs for k in [
                'emotion_raw', 'emotion_debiased', 'baseline_mean', 'baseline_logvar'
            ]
        ):
            losses['causal'] = self.causal_loss(
                outputs['emotion_raw'],
                outputs['emotion_debiased'],
                outputs['baseline_mean'],
                outputs['baseline_logvar'],
                causal_aux_losses=outputs.get('causal_aux_losses'),
            )
        else:
            losses['causal'] = zero
        
        # 4. 元学习损失
        if compute_meta and 'query_logits' in kwargs:
            losses['meta'] = self.meta_loss(
                kwargs['query_logits'],
                kwargs['query_labels']
            )
        else:
            losses['meta'] = zero
        
        # 5. 图正则化损失
        if compute_graph and all(k in kwargs for k in ['node_features', 'edge_index']):
            losses['graph'] = self.graph_loss(
                kwargs['node_features'],
                kwargs['edge_index'],
                kwargs.get('edge_weights', None)
            )
        else:
            losses['graph'] = zero
        
        # 6. 总损失
        total_loss = sum(
            self.loss_weights.get(name, 1.0) * loss
            for name, loss in losses.items()
        )
        losses['total'] = total_loss
        
        return losses


if __name__ == '__main__':
    print("测试多任务损失函数...")
    
    batch_size = 8
    num_emotions = 8
    
    fusion_logits = torch.randn(batch_size, num_emotions)
    crisis_logits = torch.randn(batch_size)
    labels = torch.randint(0, num_emotions, (batch_size,))
    emotion_raw = torch.softmax(torch.randn(batch_size, num_emotions), dim=1)
    emotion_debiased = torch.softmax(torch.randn(batch_size, num_emotions), dim=1)
    baseline_mean = torch.randn(batch_size, num_emotions)
    baseline_logvar = torch.randn(batch_size, num_emotions)
    
    criterion = UnifiedLoss(num_emotions=num_emotions)
    
    outputs = {
        'fusion_logits': fusion_logits,
        'crisis_logits': crisis_logits,
        'emotion_raw': emotion_raw,
        'emotion_debiased': emotion_debiased,
        'baseline_mean': baseline_mean,
        'baseline_logvar': baseline_logvar,
        'causal_aux_losses': {
            'baseline_kl': torch.tensor(0.1),
            'despair_consistency': torch.tensor(0.05),
            'counterfactual_consistency': torch.tensor(0.02),
        }
    }
    
    losses = criterion(outputs, labels, compute_causal=True, compute_crisis=True)
    
    print("\n损失值:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    print("\n测试完成!")

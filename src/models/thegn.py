"""
THEGN: Temporal Heterogeneous Emotion Graph Network
时序异构图注意力网络

建模对话中的情绪演化和长距离依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HeteroGraph:
    """异构图数据结构"""
    # 节点特征
    node_features: torch.Tensor  # [num_nodes, feature_dim]
    node_types: torch.Tensor  # [num_nodes] 节点类型ID
    node_times: torch.Tensor  # [num_nodes] 节点时间戳
    
    # 边信息
    edge_index: torch.Tensor  # [2, num_edges] (source, target)
    edge_types: torch.Tensor  # [num_edges] 边类型ID
    edge_weights: torch.Tensor  # [num_edges] 边权重
    
    # 元信息
    num_nodes: int
    num_edges: int
    num_node_types: int  # user, bot, emotion
    num_edge_types: int  # temporal, reply, influence


class HeterogeneousGATLayer(nn.Module):
    """
    异构图注意力层
    对不同类型的边使用不同的注意力机制
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_edge_types: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        concat: bool = True
    ):
        """
        初始化异构GAT层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            num_edge_types: 边类型数量
            num_heads: 注意力头数
            dropout: Dropout概率
            concat: 是否拼接多头输出
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_edge_types = num_edge_types
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        
        # 为每种边类型创建独立的权重矩阵
        self.W_types = nn.ModuleList([
            nn.Linear(in_features, out_features * num_heads, bias=False)
            for _ in range(num_edge_types)
        ])
        
        # 注意力参数（为每种边类型和每个头）
        self.a_types = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_heads, 2 * out_features))
            for _ in range(num_edge_types)
        ])
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        if concat:
            self.out_dim = out_features * num_heads
        else:
            self.out_dim = out_features
            self.combine_heads = nn.Linear(out_features * num_heads, out_features)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_features]
            edge_index: 边索引 [2, num_edges]
            edge_types: 边类型 [num_edges]
            edge_weights: 边权重 [num_edges]
            
        Returns:
            更新后的节点特征
        """
        num_nodes = x.size(0)
        
        # 为每种边类型计算注意力
        outputs = []
        
        for edge_type in range(self.num_edge_types):
            # 找到该类型的边
            mask = (edge_types == edge_type)
            if mask.sum() == 0:
                continue
            
            type_edge_index = edge_index[:, mask]
            type_edge_weights = edge_weights[mask] if edge_weights is not None else None
            
            # 变换特征
            h = self.W_types[edge_type](x)  # [num_nodes, out_features * num_heads]
            h = h.view(num_nodes, self.num_heads, self.out_features)
            
            # 计算注意力系数
            source_nodes = type_edge_index[0]
            target_nodes = type_edge_index[1]
            
            h_source = h[source_nodes]  # [num_edges, num_heads, out_features]
            h_target = h[target_nodes]  # [num_edges, num_heads, out_features]
            
            # 拼接源和目标节点特征
            h_concat = torch.cat([h_source, h_target], dim=2)  # [num_edges, num_heads, 2*out_features]
            
            # 计算注意力分数
            e = self.leakyrelu((h_concat * self.a_types[edge_type]).sum(dim=2))  # [num_edges, num_heads]
            
            # 如果有边权重，乘上去
            if type_edge_weights is not None:
                e = e * type_edge_weights.unsqueeze(1)
            
            # 对每个目标节点的入边进行softmax
            # 创建注意力矩阵
            attention = torch.zeros(num_nodes, num_nodes, self.num_heads).to(x.device)
            for i, (src, tgt) in enumerate(type_edge_index.t()):
                attention[tgt, src] = e[i]
            
            # Softmax (对每个节点的入边)
            attention = F.softmax(attention, dim=1)
            attention = self.dropout_layer(attention)
            
            # 聚合邻居特征
            h_out = torch.einsum('ijk,jkl->ikl', attention, h)  # [num_nodes, num_heads, out_features]
            
            outputs.append(h_out)
        
        if len(outputs) == 0:
            # 没有边，返回零张量
            return torch.zeros(num_nodes, self.out_dim).to(x.device)
        
        # 合并不同边类型的输出
        h_out = torch.stack(outputs).mean(dim=0)  # [num_nodes, num_heads, out_features]
        
        if self.concat:
            # 拼接多头
            return h_out.reshape(num_nodes, -1)
        else:
            # 平均多头
            h_out = h_out.mean(dim=1)
            return h_out


class TemporalAttention(nn.Module):
    """
    时序注意力模块
    带时间衰减的注意力机制
    """
    
    def __init__(self, hidden_size: int, decay_lambda: float = 0.1):
        """
        初始化时序注意力
        
        Args:
            hidden_size: 隐藏层大小
            decay_lambda: 时间衰减系数
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.decay_lambda = decay_lambda
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, hidden_size]
            timestamps: 时间戳 [num_nodes]
            
        Returns:
            (加权后的特征, 注意力权重)
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_size)
        
        # 计算时间衰减
        time_diffs = timestamps.unsqueeze(0) - timestamps.unsqueeze(1)
        time_diffs = torch.abs(time_diffs).float()
        decay = torch.exp(-self.decay_lambda * time_diffs)
        
        # 应用时间衰减
        attention_scores = attention_scores * decay
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class THEGNModel(nn.Module):
    """
    时序异构图情绪网络（完整模型）
    """
    
    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 256,
        num_emotions: int = 8,
        num_graph_layers: int = 3,
        num_attention_heads: int = 8,
        num_node_types: int = 3,  # user, bot, emotion
        num_edge_types: int = 3,  # temporal, reply, influence
        temporal_decay_lambda: float = 0.1,
        dropout: float = 0.1
    ):
        """
        初始化THEGN模型
        
        Args:
            input_size: 输入特征大小
            hidden_size: 隐藏层大小
            num_emotions: 情绪类别数
            num_graph_layers: 图卷积层数
            num_attention_heads: 注意力头数
            num_node_types: 节点类型数
            num_edge_types: 边类型数
            temporal_decay_lambda: 时间衰减系数
            dropout: Dropout概率
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions
        self.num_graph_layers = num_graph_layers
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 节点类型嵌入
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_size)
        
        # 异构图注意力层
        self.gat_layers = nn.ModuleList([
            HeterogeneousGATLayer(
                in_features=hidden_size,
                out_features=hidden_size // num_attention_heads,
                num_edge_types=num_edge_types,
                num_heads=num_attention_heads,
                dropout=dropout,
                concat=True if i < num_graph_layers - 1 else False
            )
            for i in range(num_graph_layers)
        ])
        
        # 时序注意力
        self.temporal_attention = TemporalAttention(hidden_size, temporal_decay_lambda)
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_graph_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_emotions)
        )
    
    def forward(
        self,
        graph: HeteroGraph,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            graph: 异构图
            return_attention: 是否返回注意力权重
            
        Returns:
            输出字典
        """
        # 投影输入特征
        x = self.input_projection(graph.node_features)
        
        # 添加节点类型嵌入
        type_emb = self.node_type_embedding(graph.node_types)
        x = x + type_emb
        
        attention_weights = {}
        
        # 多层图卷积
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            # 图注意力
            x_new = gat_layer(
                x,
                graph.edge_index,
                graph.edge_types,
                graph.edge_weights
            )
            
            # 残差连接
            if i > 0:
                x_new = x_new + x
            
            # 层归一化
            x = layer_norm(x_new)
        
        # 时序注意力
        x_temporal, temporal_attn = self.temporal_attention(x, graph.node_times)
        
        if return_attention:
            attention_weights['temporal'] = temporal_attn
        
        # 残差连接
        x = x + x_temporal
        
        # 输出投影
        logits = self.output_projection(x)
        probabilities = F.softmax(logits, dim=-1)
        
        outputs = {
            'logits': logits,
            'probabilities': probabilities,
            'node_features': x
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def predict_emotion(
        self,
        graph: HeteroGraph,
        target_node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测情绪
        
        Args:
            graph: 输入图
            target_node_indices: 目标节点索引（如果为None则返回所有节点）
            
        Returns:
            情绪概率分布
        """
        outputs = self.forward(graph)
        probabilities = outputs['probabilities']
        
        if target_node_indices is not None:
            probabilities = probabilities[target_node_indices]
        
        return probabilities


def create_dialogue_graph(
    dialogue_history: List[str],
    text_features: torch.Tensor,
    temporal_decay: float = 0.1
) -> HeteroGraph:
    """
    从对话历史创建异构图
    
    Args:
        dialogue_history: 对话历史
        text_features: 文本特征 [num_turns, feature_dim]
        temporal_decay: 时间衰减参数
        
    Returns:
        异构图
    """
    num_turns = len(dialogue_history)
    
    if num_turns == 0:
        # 空图
        return HeteroGraph(
            node_features=torch.zeros(1, text_features.size(1)),
            node_types=torch.zeros(1, dtype=torch.long),
            node_times=torch.zeros(1, dtype=torch.long),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_types=torch.zeros(0, dtype=torch.long),
            edge_weights=torch.zeros(0),
            num_nodes=1,
            num_edges=0,
            num_node_types=3,
            num_edge_types=3
        )
    
    # 节点特征（所有对话轮次）
    node_features = text_features
    
    # 节点类型（这里简化：都是user类型，实际应区分user和bot）
    node_types = torch.zeros(num_turns, dtype=torch.long)
    
    # 节点时间戳
    node_times = torch.arange(num_turns, dtype=torch.long)
    
    # 构建边
    edge_list = []
    edge_type_list = []
    edge_weight_list = []
    
    # 时序边（连接相邻的对话轮次）
    for i in range(num_turns - 1):
        edge_list.append([i, i + 1])
        edge_type_list.append(0)  # temporal type
        # 时间衰减权重
        time_diff = 1
        weight = np.exp(-temporal_decay * time_diff)
        edge_weight_list.append(weight)
    
    # 长距离连接（连接所有历史节点，权重随时间衰减）
    for i in range(num_turns):
        for j in range(i + 2, min(i + 6, num_turns)):  # 最多连接后续5个节点
            edge_list.append([i, j])
            edge_type_list.append(1)  # reply type
            time_diff = j - i
            weight = np.exp(-temporal_decay * time_diff)
            edge_weight_list.append(weight)
    
    if len(edge_list) == 0:
        # 至少要有一条边
        edge_list = [[0, 0]]
        edge_type_list = [0]
        edge_weight_list = [1.0]
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_types = torch.tensor(edge_type_list, dtype=torch.long)
    edge_weights = torch.tensor(edge_weight_list, dtype=torch.float)
    
    return HeteroGraph(
        node_features=node_features,
        node_types=node_types,
        node_times=node_times,
        edge_index=edge_index,
        edge_types=edge_types,
        edge_weights=edge_weights,
        num_nodes=num_turns,
        num_edges=len(edge_list),
        num_node_types=3,
        num_edge_types=3
    )


# 使用示例
if __name__ == '__main__':
    print("正在测试THEGN模型...")
    
    # 参数
    input_size = 768
    hidden_size = 256
    num_emotions = 8
    num_turns = 5
    
    # 模拟对话历史
    dialogue_history = [
        "今天心情不错",
        "作业有点多",
        "感觉有点累",
        "压力好大",
        "真的撑不住了"
    ]
    
    # 模拟文本特征
    text_features = torch.randn(num_turns, input_size)
    
    # 创建图
    print("\n创建对话图...")
    graph = create_dialogue_graph(dialogue_history, text_features)
    print(f"节点数: {graph.num_nodes}")
    print(f"边数: {graph.num_edges}")
    print(f"节点类型: {graph.node_types}")
    print(f"边类型: {graph.edge_types}")
    
    # 创建模型
    print("\n创建THEGN模型...")
    model = THEGNModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_emotions=num_emotions,
        num_graph_layers=3,
        num_attention_heads=8
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 前向传播
    print("\n前向传播...")
    model.eval()
    with torch.no_grad():
        outputs = model.forward(graph, return_attention=True)
    
    print(f"输出logits形状: {outputs['logits'].shape}")
    print(f"输出概率形状: {outputs['probabilities'].shape}")
    
    # 预测最后一轮的情绪
    print("\n预测最后一轮对话的情绪...")
    last_turn_emotion = outputs['probabilities'][-1]
    emotion_labels = ['中性', '高兴', '惊讶', '悲伤', '愤怒', '恐惧', '厌恶', '绝望']
    
    print("情绪分布:")
    for label, prob in zip(emotion_labels, last_turn_emotion):
        if prob > 0.05:
            print(f"  {label}: {prob:.2%}")
    
    print("\n" + "="*60)
    print("测试完成!")


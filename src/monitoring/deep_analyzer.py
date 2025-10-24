"""
第二层：深度个性化分析器
整合 CPEB + MEA + THEGN
目标延迟: < 500ms
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UserBaseline:
    """用户个性化基线"""
    user_id: str
    mean_emotion: np.ndarray  # 8维情绪均值向量
    std_emotion: np.ndarray   # 8维情绪标准差向量
    expression_style: str     # 表达风格: 'reserved'(内敛) / 'expressive'(外放)
    history_count: int        # 历史对话数
    last_update: str          # 最后更新时间
    risk_threshold: float     # 个性化风险阈值


class PersonalizedDeepAnalyzer:
    """
    深度个性化分析器
    整合 CPEB + MEA + THEGN
    """
    
    def __init__(
        self, 
        cpeb_model=None,
        meta_model=None, 
        graph_model=None,
        device: str = "cpu"
    ):
        """
        初始化深度分析器
        
        Args:
            cpeb_model: 因果基线模型
            meta_model: 元学习模型
            graph_model: 图网络模型
            device: 设备
        """
        self.cpeb_model = cpeb_model
        self.meta_model = meta_model
        self.graph_model = graph_model
        self.device = device
        
        # 从数据库/缓存加载用户基线
        self.user_baselines: Dict[str, UserBaseline] = {}
        
        # 固定阈值 (用于新用户)
        self.default_thresholds = {
            'green': 3.0,    # 正常
            'blue': 5.0,     # 关注
            'orange': 7.0,   # 警告
            'red': 9.0       # 危机
        }
        
        # 情绪标签
        self.emotion_labels = ['中性', '高兴', '惊讶', '悲伤', '愤怒', '恐惧', '厌恶', '绝望']
    
    def load_user_baseline(self, user_id: str) -> Optional[UserBaseline]:
        """
        从缓存/数据库加载用户基线
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户基线或None
        """
        if user_id in self.user_baselines:
            return self.user_baselines[user_id]
        
        # 实际实现中从Redis/数据库读取
        # 这里返回None表示新用户
        return None
    
    def calculate_dynamic_threshold(
        self, 
        user_id: str, 
        current_emotion: np.ndarray
    ) -> Dict[str, float]:
        """
        动态阈值计算（核心算法）
        
        根据用户个性化基线和表达风格动态调整阈值
        
        Args:
            user_id: 用户ID
            current_emotion: 当前情绪向量 (8维)
            
        Returns:
            动态阈值字典
        """
        baseline = self.load_user_baseline(user_id)
        
        if baseline is None:
            # 新用户使用默认阈值
            return self.default_thresholds
        
        # 计算情绪偏离度 (标准化)
        deviation = (current_emotion - baseline.mean_emotion) / (baseline.std_emotion + 1e-6)
        
        # 负面情绪索引: [3:悲伤, 4:愤怒, 5:恐惧, 7:绝望]
        negative_deviation = deviation[[3, 4, 5, 7]]
        
        # 加权平均偏离度
        weights = np.array([2.0, 1.5, 2.5, 3.0])  # 绝望权重最高
        weighted_deviation = np.average(negative_deviation, weights=weights)
        
        # 根据表达风格调整
        if baseline.expression_style == 'reserved':
            # 内敛型用户：轻微负面就要警惕
            style_factor = 0.7
        else:  # 'expressive'
            # 外放型用户：需要更强的负面才警惕
            style_factor = 1.3
        
        # 动态调整阈值
        adjusted_thresholds = {
            'green': baseline.risk_threshold * 0.3 * style_factor,
            'blue': baseline.risk_threshold * 0.5 * style_factor,
            'orange': baseline.risk_threshold * 0.7 * style_factor,
            'red': baseline.risk_threshold * 0.9 * style_factor
        }
        
        # 如果历史数据少，向默认值回归
        confidence = min(baseline.history_count / 50, 1.0)
        final_thresholds = {}
        for key in adjusted_thresholds:
            final_thresholds[key] = (
                confidence * adjusted_thresholds[key] + 
                (1 - confidence) * self.default_thresholds[key]
            )
        
        return final_thresholds
    
    def encode_emotion(self, text: str) -> np.ndarray:
        """
        编码文本为情绪向量 (模拟实现)
        
        实际实现应使用 CPEB 模型
        
        Args:
            text: 输入文本
            
        Returns:
            8维情绪向量
        """
        # 模拟情绪编码
        # 实际应该: emotion_raw = self.cpeb_model.encode_emotion(text)
        
        # 简单的关键词匹配模拟
        emotion_vector = np.zeros(8)
        emotion_vector[0] = 0.5  # 基础中性
        
        if any(kw in text for kw in ['高兴', '开心', '快乐', '不错']):
            emotion_vector[1] = 0.8
        if any(kw in text for kw in ['悲伤', '难过', '伤心', '痛苦']):
            emotion_vector[3] = 0.7
        if any(kw in text for kw in ['愤怒', '生气', '气愤', '烦']):
            emotion_vector[4] = 0.6
        if any(kw in text for kw in ['恐惧', '害怕', '担心', '不安']):
            emotion_vector[5] = 0.6
        if any(kw in text for kw in ['绝望', '崩溃', '放弃', '自杀']):
            emotion_vector[7] = 0.9
        
        # 归一化
        if emotion_vector.sum() > 1.0:
            emotion_vector = emotion_vector / emotion_vector.sum()
        
        return emotion_vector
    
    def causal_intervention(
        self, 
        emotion_raw: np.ndarray, 
        user_id: str
    ) -> np.ndarray:
        """
        因果干预 (CPEB)
        
        消除个体表达习惯的混淆偏差
        
        Args:
            emotion_raw: 原始情绪向量
            user_id: 用户ID
            
        Returns:
            去偏后的情绪向量
        """
        # 实际实现: emotion_debiased = self.cpeb_model.causal_intervention(emotion_raw, user_id)
        
        # 模拟实现：简单调整
        baseline = self.load_user_baseline(user_id)
        if baseline is not None:
            # 减去用户基线偏差
            emotion_debiased = emotion_raw - 0.1 * baseline.mean_emotion
            emotion_debiased = np.clip(emotion_debiased, 0, 1)
        else:
            emotion_debiased = emotion_raw
        
        return emotion_debiased
    
    def meta_adapt(
        self,
        emotion: np.ndarray,
        user_id: str,
        recent_history: List[str]
    ) -> np.ndarray:
        """
        元学习快速适应 (MEA)
        
        用少量样本快速适应用户风格
        
        Args:
            emotion: 情绪向量
            user_id: 用户ID
            recent_history: 最近对话历史
            
        Returns:
            适应后的情绪向量
        """
        # 实际实现: emotion_adapted = self.meta_model.adapt(emotion, user_id, recent_history)
        
        # 模拟实现：如果用户数据少，进行轻微调整
        baseline = self.load_user_baseline(user_id)
        if baseline is not None and baseline.history_count < 20:
            # 新用户，使用元学习适应
            emotion_adapted = emotion * 1.1  # 模拟适应效果
            emotion_adapted = np.clip(emotion_adapted, 0, 1)
        else:
            emotion_adapted = emotion
        
        return emotion_adapted
    
    def graph_model_inference(
        self,
        emotion: np.ndarray,
        conversation_history: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """
        图网络建模时序依赖 (THEGN)
        
        Args:
            emotion: 当前情绪向量
            conversation_history: 对话历史
            
        Returns:
            (最终情绪向量, 注意力权重)
        """
        # 实际实现:
        # graph = self.graph_model.build_graph(conversation_history)
        # emotion_final, attention_weights = self.graph_model.forward(graph, emotion)
        
        # 模拟实现：考虑历史情绪趋势
        if len(conversation_history) > 3:
            # 如果持续负面，增强负面情绪
            emotion_final = emotion * 1.2
            emotion_final = np.clip(emotion_final, 0, 1)
        else:
            emotion_final = emotion
        
        # 模拟注意力权重
        attention_weights = {
            'temporal': [0.3] * min(len(conversation_history), 5),
            'importance': [0.1] * min(len(conversation_history), 5)
        }
        
        return emotion_final, attention_weights
    
    def calculate_risk_score(self, emotion_vector: np.ndarray) -> float:
        """
        计算综合风险分数 (0-10)
        
        Args:
            emotion_vector: 情绪向量 (8维)
            
        Returns:
            风险分数
        """
        # 负面情绪加权
        sadness = emotion_vector[3]
        anger = emotion_vector[4]
        fear = emotion_vector[5]
        despair = emotion_vector[7]
        
        risk = (
            2.0 * sadness +
            1.5 * anger +
            2.5 * fear +
            3.5 * despair  # 绝望权重最高
        )
        
        return np.clip(risk * 10, 0, 10)
    
    def analyze(
        self, 
        text: str, 
        user_id: str, 
        conversation_history: Optional[List[str]] = None
    ) -> Dict:
        """
        深度分析主函数
        
        Args:
            text: 输入文本
            user_id: 用户ID
            conversation_history: 对话历史
            
        Returns:
            分析结果字典
        """
        start_time = time.time()
        
        if conversation_history is None:
            conversation_history = []
        
        # Step 1: 编码情绪
        emotion_raw = self.encode_emotion(text)
        
        # Step 2: 因果基线调整 (CPEB)
        emotion_debiased = self.causal_intervention(emotion_raw, user_id)
        
        # Step 3: 元学习快速适应 (MEA)
        emotion_adapted = self.meta_adapt(
            emotion_debiased, 
            user_id, 
            conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history
        )
        
        # Step 4: 图网络建模时序依赖 (THEGN)
        emotion_final, attention_weights = self.graph_model_inference(
            emotion_adapted,
            conversation_history
        )
        
        # Step 5: 动态阈值判定
        thresholds = self.calculate_dynamic_threshold(user_id, emotion_final)
        
        # Step 6: 计算最终风险分数
        risk_score = self.calculate_risk_score(emotion_final)
        
        # Step 7: 判定风险等级
        if risk_score >= thresholds['red']:
            risk_level = 'red'
            risk_label = '危机'
        elif risk_score >= thresholds['orange']:
            risk_level = 'orange'
            risk_label = '警告'
        elif risk_score >= thresholds['blue']:
            risk_level = 'blue'
            risk_label = '关注'
        else:
            risk_level = 'green'
            risk_label = '正常'
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'user_id': user_id,
            'text': text,
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'risk_label': risk_label,
            'emotion_vector': emotion_final.tolist(),
            'emotion_distribution': {
                label: float(emotion_final[i]) 
                for i, label in enumerate(self.emotion_labels)
            },
            'thresholds': thresholds,
            'attention_weights': attention_weights,
            'latency_ms': latency,
            'need_alert': risk_level == 'red'
        }


# 使用示例
if __name__ == '__main__':
    print("正在初始化深度分析器...")
    analyzer = PersonalizedDeepAnalyzer(device='cpu')
    
    # 添加模拟用户基线
    analyzer.user_baselines['user001'] = UserBaseline(
        user_id='user001',
        mean_emotion=np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]),
        std_emotion=np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02]),
        expression_style='reserved',
        history_count=50,
        last_update='2024-01-01',
        risk_threshold=7.0
    )
    
    test_cases = [
        ("今天心情还不错", []),
        ("作业有点多，感觉有点累", ["今天上课很无聊"]),
        ("我真的撑不住了，太痛苦了", ["压力好大", "睡不着觉", "很难受"]),
    ]
    
    print("\n" + "="*60)
    print("测试深度个性化分析器")
    print("="*60)
    
    for text, history in test_cases:
        result = analyzer.analyze(text, user_id='user001', conversation_history=history)
        print(f"\n文本: {text}")
        print(f"风险评分: {result['risk_score']:.2f}")
        print(f"风险等级: {result['risk_level']} - {result['risk_label']}")
        print(f"需要预警: {result['need_alert']}")
        print(f"延迟: {result['latency_ms']:.2f}ms")
        print(f"情绪分布: {result['emotion_distribution']}")
    
    print("\n" + "="*60)
    print("测试完成!")


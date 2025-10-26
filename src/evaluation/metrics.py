"""
评估指标模块
待修改部分：
1. 缺少对不同模型之间的对比
2.compute_confusion_matrix 方法未在main中测试
3.EmotionMetrics.compute_metrics 中标签顺序不匹配

key main methods:
1. EmotionMetrics: 情绪分类的基础指标(准确率、精确率、召回率、F1等)
2. PersonalizationMetrics: 个性化模型的评估指标
3. FewShotMetrics: 少样本学习的性能评估
4. AlertMetrics: 预警系统的评估指标和延迟统计
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,#计算精确率、召回率、F1分数
    confusion_matrix,
    classification_report
)


class EmotionMetrics:
    """
    情绪分类评估指标
    1. 评估指标：准确率、精确率、召回率、F1分数
    2. 混淆矩阵
    3. 分类报告
    """
    
    def __init__(self, emotion_labels: Optional[List[str]] = None):
        """
        初始化
        
        Args:
            emotion_labels: 情绪标签列表 映射表 or 默认
        """
        if emotion_labels is None:
            emotion_labels = ['中性', '高兴', '惊讶', '悲伤', '愤怒', '恐惧', '厌恶', '绝望']
        
        self.emotion_labels = emotion_labels
        self.num_emotions = len(emotion_labels)
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            average: 平均方式 ('macro', 'micro', 'weighted')
            macro: 计算所有类别的指标的未加权平均值
            micro: 计算所有类级别的指标的加权平均值
            weighted: 计算所有类级别的指标的加权平均值，权重为每个类的支持度
            
        Returns:
            指标字典
            accuracy: 准确率
            precision_macro: 宏平均精确率
            recall_macro: 宏平均召回率
            f1_macro: 宏平均F1分数
            precision_<label>: 每个类别的精确率
            recall_<label>: 每个类别的召回率
            f1_<label>: 每个类别的F1分数
            support_<label>: 每个类别的支持度
        Note:
            精确率(Precision): 在所有被预测为正类的样本中，实际为正类的比例。
            召回率(Recall): 在所有实际为正类的样本中，被正确预测为正类的比例。
            F1分数(F1 Score): 精确率和召回率的调和平均数，综合考虑了两者的表现，越高说明效果越好。
            准确率(Accuracy): 所有预测中，正确预测的比例。
            混淆矩阵: 显示分类结果与真实结果的对比，矩阵的对角线上的元素表示预测正确的数量，
                      非对角线上的元素表示预测错误的数量。
        """
        # 准确率
        accuracy = accuracy_score(y_true, y_pred)
        
        # 精确率、召回率、F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            f'precision_{average}': float(precision),
            f'recall_{average}': float(recall),
            f'f1_{average}': float(f1)
        }
        
        # 每个类别的指标 返回的是每个类别的指标数组
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
        
        for i, label in enumerate(self.emotion_labels):
            if i < len(precision_per_class):
                # 每个情绪类别的四个指标
                metrics[f'precision_{label}'] = float(precision_per_class[i])
                metrics[f'recall_{label}'] = float(recall_per_class[i])
                metrics[f'f1_{label}'] = float(f1_per_class[i])
                metrics[f'support_{label}'] = int(support_per_class[i])
        
        return metrics
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        计算混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            混淆矩阵

        Note:
            混淆矩阵的行表示真实标签，列表示预测标签。
        """
        return confusion_matrix(y_true, y_pred)
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            格式化字符串：
            - 每个类别的准确率、精确率、召回率、F1分数
            - 整体的准确率
            - 宏平均和加权
        Note:
            分类报告是sklearn的标准格式
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.emotion_labels,
            zero_division=0 # 忽略除零错误
        )


class PersonalizationMetrics:
    """
    个性化评估指标(与通用模型对比)
    Asgs:
        generic_f1: 通用模型F1分数
        personalized_f1: 个性化模型F1分数
    Returns:
        个性化增益（personalization f1）:
        - 正值表示个性化模型优于通用模型
        - 负值表示个性化模型不如通用模型
        - 0 表示两个模型效果相同

    """
    
    @staticmethod
    def compute_personalization_gain(
        generic_f1: float,
        personalized_f1: float
    ) -> float:
        return personalized_f1 - generic_f1
    
    @staticmethod
    def compute_user_level_metrics(
        y_true_by_user: Dict[str, np.ndarray],
        y_pred_by_user: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        计算用户级别的指标
        
        Args:
            y_true_by_user: 按用户分组的真实标签
            y_pred_by_user: 按用户分组的预测标签
            
        Returns:
            用户级别指标字典
        """
        user_metrics = {}
        # 遍历用户
        for user_id in y_true_by_user:
            if user_id not in y_pred_by_user:
                continue
            
            y_true = y_true_by_user[user_id]
            y_pred = y_pred_by_user[user_id]
            
            if len(y_true) == 0:
                continue
            
            accuracy = accuracy_score(y_true, y_pred)
            #_忽略不需要的返回值
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            
            user_metrics[user_id] = {
                'accuracy': float(accuracy),
                'f1_macro': float(f1),
                'num_samples': len(y_true)
            }
        
        return user_metrics


class FewShotMetrics:
    """少样本学习评估指标"""
    
    @staticmethod
    def compute_k_shot_performance(
        support_sizes: List[int],
        f1_scores: List[float]
    ) -> Dict[str, float]:
        """
        计算不同shot数下的性能
        
        Args:
            support_sizes: 支持集大小列表
            f1_scores: 对应的F1分数列表
            
        Returns:
            性能字典
        Note:
            绘制学习曲线，样本数的影响
            improvement_rate: 学习效果
            曲线平稳说明模型对样本数不敏感
        """
        performance = {}
        
        for k, f1 in zip(support_sizes, f1_scores):
            performance[f'{k}-shot_f1'] = f1
        
        # 计算性能增益曲线
        if len(f1_scores) > 1:
            performance['improvement_rate'] = (f1_scores[-1] - f1_scores[0]) / len(f1_scores)
        
        return performance
    
    @staticmethod
    def compute_sample_efficiency(
        few_shot_f1: float,
        full_shot_f1: float,
        few_shot_size: int,
        full_shot_size: int
    ) -> float:
        """
        计算样本效率
        
        Args:
            few_shot_f1: 少样本F1分数
            full_shot_f1: 全样本F1分数
            few_shot_size: 少样本数量
            full_shot_size: 全样本数量
            
        Returns:
            样本效率（达到多少比例的性能用了多少比例的数据）
            efficiency = (few_shot_f1 / full_shot_f1) / (few_shot_size / full_shot_size)
            - 值>1: 表示样本利用效率很高,用少量数据就能达到较好效果
            - 值=1: 表示性能提升与数据增加成正比
            - 值<1: 表示需要大量数据才能提升性能
        """
        #计算样本效率 少量样本性能
        performance_ratio = few_shot_f1 / full_shot_f1 if full_shot_f1 > 0 else 0
        #计算样本效率 用了多少比率的数据
        data_ratio = few_shot_size / full_shot_size if full_shot_size > 0 else 0
        
        if data_ratio > 0:
            return performance_ratio / data_ratio
        else:
            return 0.0


class AlertMetrics:
    """
    预警系统评估指标
    1. 预警指标：精确率、召回率、F1分数、特异性等
    2. 延迟统计：平均延迟、中位数延迟  
    """
    
    @staticmethod
    def compute_alert_metrics(
        y_true_risk: np.ndarray,
        y_pred_risk: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        计算预警指标
        
        Args:
            y_true_risk: 真实风险标签（0/1）
            y_pred_risk: 预测风险分数（0-1）
            threshold: 预警阈值
            
        Returns:
            指标字典
        """
        y_pred_binary = (y_pred_risk >= threshold).astype(int)
        
        # 真阳性、假阳性、真阴性、假阴性
        tp = np.sum((y_true_risk == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_risk == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true_risk == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true_risk == 1) & (y_pred_binary == 0))
        
        # 精确率、召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 特异性
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'specificity': float(specificity),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    @staticmethod
    def compute_latency_metrics(
        latencies: List[float]
    ) -> Dict[str, float]:
        """
        计算延迟指标
        Args:
            latencies: 延迟列表（毫秒）
            
        Returns:
            延迟统计
        """
        latencies = np.array(latencies)
        
        return {
            'mean': float(np.mean(latencies)),
            'median': float(np.median(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99))
        }


# 使用示例
if __name__ == '__main__':
    print("测试评估指标...")
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 100
    num_emotions = 8
    
    y_true = np.random.randint(0, num_emotions, n_samples)
    y_pred = np.random.randint(0, num_emotions, n_samples)
    
    # 情绪分类指标
    print("\n1. 情绪分类指标")
    emotion_metrics = EmotionMetrics()
    metrics = emotion_metrics.compute_metrics(y_true, y_pred)
    
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(emotion_metrics.generate_classification_report(y_true, y_pred))
    
    # 少样本指标
    print("\n2. 少样本性能")
    few_shot_metrics = FewShotMetrics()
    support_sizes = [1, 3, 5, 10, 20]
    f1_scores = [0.45, 0.58, 0.65, 0.72, 0.78]
    
    performance = few_shot_metrics.compute_k_shot_performance(support_sizes, f1_scores)
    for k, v in performance.items():
        print(f"{k}: {v:.4f}")
    
    # 预警指标
    print("\n3. 预警系统指标")
    y_true_risk = np.random.randint(0, 2, 50)
    y_pred_risk = np.random.rand(50)
    
    alert_metrics = AlertMetrics()
    alert_results = alert_metrics.compute_alert_metrics(y_true_risk, y_pred_risk)
    
    print(f"精确率: {alert_results['precision']:.4f}")
    print(f"召回率: {alert_results['recall']:.4f}")
    print(f"F1: {alert_results['f1']:.4f}")
    print(f"真阳性: {alert_results['true_positives']}")
    print(f"假阳性: {alert_results['false_positives']}")
    
    # 延迟指标
    print("\n4. 延迟指标")
    latencies = np.random.gamma(2, 15, 1000)  # 模拟延迟分布
    latency_metrics = alert_metrics.compute_latency_metrics(latencies)
    
    print(f"平均延迟: {latency_metrics['mean']:.2f}ms")
    print(f"P50延迟: {latency_metrics['p50']:.2f}ms")
    print(f"P95延迟: {latency_metrics['p95']:.2f}ms")
    print(f"P99延迟: {latency_metrics['p99']:.2f}ms")
    
    print("\n测试完成!")


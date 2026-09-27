"""
第一层：轻量级情绪预筛选器
目标延迟: < 50ms
"""

import re
import torch
import time
from typing import Dict, Tuple, List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class LightweightEmotionFilter:
    """
    轻量级情绪预筛选器
    使用规则 + 轻量BERT快速判断是否需要深度分析
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        device: str = "cpu",
        deep_threshold: float = 7.0,
    ):
        """
        初始化过滤器
        
        Args:
            model_name: 预训练模型名称
            device: 设备 (cpu/cuda)
            deep_threshold: 触发深度分析的分数阈值
        """
        self.device = device
        self.deep_threshold = float(deep_threshold)
        
        # 使用轻量级模型 (实际部署时可用 distilbert)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # 注意：未微调时分类头为随机初始化；仅作粗筛辅助，主筛查仍靠关键词
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=8,
                ignore_mismatched_sizes=True
            )
            self.model.to(device)
            self.model.eval()
            self._bert_head_untrained = True
        except Exception as e:
            print(f"警告: 无法加载模型 {model_name}, 将只使用规则匹配。错误: {e}")
            self.tokenizer = None
            self.model = None
            self._bert_head_untrained = True
        
        # 极端关键词库 (三级严重程度)
        self.extreme_keywords = {
            'critical': [  # 危机级别
                '自杀', '轻生', '结束生命', '不想活', '去死',
                '跳楼', '割腕', '服毒', '了断', '自残'
            ],
            'severe': [  # 严重级别
                '活不下去', '没有希望', '彻底崩溃', '撑不住',
                '痛不欲生', '生无可恋', '一了百了'
            ],
            'warning': [  # 警告级别
                '抑郁', '绝望', '崩溃', '无助', '孤独',
                '压力大', '想哭', '痛苦', '难受'
            ]
        }
        
        # 负面词汇模式
        self.negative_patterns = {
            r'(太|很|非常|特别|超级)\s*(累|烦|难过|痛苦)': 2.0,
            r'不(想|愿意|要)\s*(活|继续|坚持)': 3.0,
            r'(完全|彻底|真的)\s*(绝望|崩溃|失望)': 2.5,
        }
    
    def check_extreme_keywords(self, text: str) -> Tuple[bool, str, float]:
        """
        检查极端关键词
        
        Args:
            text: 输入文本
            
        Returns:
            (是否包含极端词, 匹配级别, 基础分数)
        """
        text_lower = text.lower()
        
        # 危机级别 - 直接触发
        for keyword in self.extreme_keywords['critical']:
            if keyword in text_lower:
                return True, 'critical', 10.0
        
        # 严重级别
        for keyword in self.extreme_keywords['severe']:
            if keyword in text_lower:
                return True, 'severe', 8.5
        
        # 警告级别 (至少出现2个警告词才触发)
        warning_count = sum(1 for kw in self.extreme_keywords['warning'] 
                           if kw in text_lower)
        if warning_count >= 2:
            return True, 'warning', 7.0
        
        return False, 'normal', 0.0
    
    def calculate_pattern_score(self, text: str) -> float:
        """
        基于正则模式计算额外分数
        
        Args:
            text: 输入文本
            
        Returns:
            模式匹配分数
        """
        score = 0.0
        for pattern, weight in self.negative_patterns.items():
            if re.search(pattern, text):
                score += weight
        return min(score, 5.0)  # 最多加5分
    
    def get_bert_emotion_score(self, text: str) -> float:
        """
        使用轻量BERT获取情绪分数
        
        Args:
            text: 输入文本
            
        Returns:
            情绪分数 (0-10)
        """
        if self.tokenizer is None or self.model is None:
            return 0.0
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,  # 限制长度提升速度
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
            
            # 情绪类别: [中性, 高兴, 惊讶, 悲伤, 愤怒, 恐惧, 厌恶, 绝望]
            # 计算负面情绪加权分数
            negative_weights = [0, 0, 0, 2.0, 1.5, 2.5, 1.0, 3.0]
            score = sum(p.item() * w for p, w in zip(probs, negative_weights))
            
            return score * 10  # 归一化到0-10
        except Exception as e:
            print(f"BERT评分失败: {e}")
            return 0.0
    
    def predict(
        self,
        text: str,
        user_id: Optional[str] = None,
        deep_threshold_override: Optional[float] = None,
    ) -> Dict:
        """
        主预测函数
        
        Args:
            text: 输入文本
            user_id: 用户ID (可选)
            deep_threshold_override: 覆盖默认深度分析阈值
            
        Returns:
            预测结果字典
        """
        start_time = time.time()
        thr = float(self.deep_threshold if deep_threshold_override is None else deep_threshold_override)
        
        # Step 1: 关键词检查 (最快)
        has_extreme, level, keyword_score = self.check_extreme_keywords(text)
        
        # Step 2: 模式匹配
        pattern_score = self.calculate_pattern_score(text)
        
        # Step 3: BERT评分 — 未微调头仅作弱参考，权重压低
        if not has_extreme or level != 'critical':
            bert_score = self.get_bert_emotion_score(text)
            if getattr(self, "_bert_head_untrained", True):
                bert_score = min(bert_score * 0.3, 3.0)
        else:
            bert_score = 10.0  # 危机级别直接给满分
        
        # 综合评分：关键词优先，避免未训练 BERT 主导
        final_score = max(keyword_score, pattern_score + 0.5 * bert_score)
        final_score = min(final_score, 10.0)
        
        need_deep_analysis = (
            final_score >= thr or 
            level in ['critical', 'severe']
        )
        
        latency = (time.time() - start_time) * 1000  # ms
        
        return {
            'user_id': user_id,
            'text': text,
            'score': final_score,
            'level': level,
            'need_deep_analysis': need_deep_analysis,
            'is_extreme': has_extreme,
            'latency_ms': latency,
            'components': {
                'keyword_score': keyword_score,
                'pattern_score': pattern_score,
                'bert_score': bert_score,
                'deep_threshold': thr,
            }
        }
    
    def batch_predict(self, texts: List[str], user_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        批量预测
        
        Args:
            texts: 文本列表
            user_ids: 用户ID列表
            
        Returns:
            预测结果列表
        """
        if user_ids is None:
            user_ids = [None] * len(texts)
        
        return [self.predict(text, uid) for text, uid in zip(texts, user_ids)]


# 使用示例
if __name__ == '__main__':
    print("正在初始化轻量级过滤器...")
    filter_model = LightweightEmotionFilter(device='cpu')
    
    test_cases = [
        "今天天气不错",  # 正常
        "作业好多啊，有点累",  # 正常
        "我真的很累，压力好大，完全崩溃了",  # 警告
        "活不下去了，太痛苦了",  # 严重
        "我想自杀，活着没意思",  # 危机
    ]
    
    print("\n" + "="*60)
    print("测试轻量级情绪过滤器")
    print("="*60)
    
    for text in test_cases:
        result = filter_model.predict(text)
        print(f"\n文本: {text}")
        print(f"分数: {result['score']:.2f}")
        print(f"级别: {result['level']}")
        print(f"需要深度分析: {result['need_deep_analysis']}")
        print(f"延迟: {result['latency_ms']:.2f}ms")
        print(f"组件分数: 关键词={result['components']['keyword_score']:.2f}, "
              f"模式={result['components']['pattern_score']:.2f}, "
              f"BERT={result['components']['bert_score']:.2f}")
    
    print("\n" + "="*60)
    print("测试完成!")


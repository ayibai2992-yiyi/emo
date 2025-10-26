"""
CPEB: Causal Personalized Emotion Baseline
因果个性化情绪基线模型

使用因果推断消除个体表达习惯的混淆偏差
讲情绪分解为：
1.用户基线表达习惯
2.用户真实的情绪表示

通过因果推断消除表达习惯影响，获得更准确的情绪识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from transformers import AutoModel, AutoTokenizer


class BaselineEstimator(nn.Module):
    """
    用户情绪基线估计器
    使用变分推断学习用户的情绪表达基线分布
    Returns:
        基线均值和对数方差
    """
    
    def __init__(self, hidden_size: int = 768, num_emotions: int = 8):
        """
        初始化基线估计器
        
        Args:
            hidden_size: 隐藏层大小
            num_emotions: 情绪类别数
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions
        
        # 用户嵌入
        self.user_embedding_size = 64
        self.user_embeddings = nn.Embedding(50000, self.user_embedding_size)
        
        # 基线均值和方差估计网络
        self.mean_net = nn.Sequential(
            nn.Linear(self.user_embedding_size + hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions)
        )
        
        self.logvar_net = nn.Sequential(
            nn.Linear(self.user_embedding_size + hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions)
        )
    
    def forward(
        self, 
        user_ids: torch.Tensor, 
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            user_ids: 用户ID张量 [batch_size]
            text_features: 文本特征 [batch_size, hidden_size]
            
        Returns:
            (baseline_mean, baseline_logvar): 基线均值和对数方差
        """
        # 获取用户嵌入
        user_emb = self.user_embeddings(user_ids)  # [batch_size, user_embedding_size]
        
        # 拼接用户嵌入和文本特征
        combined = torch.cat([user_emb, text_features], dim=1)
        
        # 估计基线分布
        baseline_mean = self.mean_net(combined)  # [batch_size, num_emotions]
        baseline_logvar = self.logvar_net(combined)  # [batch_size, num_emotions]
        
        return baseline_mean, baseline_logvar
    
    def sample_baseline(
        self, 
        baseline_mean: torch.Tensor, 
        baseline_logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        从基线分布采样
        Returns:
            baseline_mean: 基线均值
            baseline_logvar: 基线对数方差
        考虑到用户习惯的变化和防止过拟合
        """
        #标准化
        std = torch.exp(0.5 * baseline_logvar)
        #正态分布
        eps = torch.randn_like(std)
        #重参数
        return baseline_mean + eps * std


class CausalInterventionModule(nn.Module):
    """
    因果干预模块
    使用前门调整消除混淆偏差
    Args:
    - 原始情绪表示
    - 用户基线
    情绪=真实情绪+表达习惯
    """
    def __init__(self, num_emotions: int = 8, hidden_size: int = 256):
        """
        初始化因果干预模块
        
        Args:
            num_emotions: 情绪类别数
            hidden_size: 隐藏层大小
        """
        super().__init__()
        self.num_emotions = num_emotions
        
        # 表达习惯编码器 用户基数向量->表达习惯特征向量
        self.habit_encoder = nn.Sequential(
            nn.Linear(num_emotions, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_emotions)
        )
        
        # 真实情绪解码器
        self.emotion_decoder = nn.Sequential(
            nn.Linear(num_emotions * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_emotions),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        emotion_raw: torch.Tensor,
        baseline: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播 - 因果干预
        原始情绪 → 识别习惯偏差 → 移除偏差 → 真实情绪
        Args:
            emotion_raw: 原始情绪表示 [batch_size, num_emotions]
            baseline: 用户基线 [batch_size, num_emotions]
            
        Returns:
            去偏后的情绪表示
        """
        # 估计表达习惯
        habit = self.habit_encoder(baseline)
        
        # 干预：移除表达习惯的影响
        combined = torch.cat([emotion_raw, habit], dim=1)
        emotion_debiased = self.emotion_decoder(combined)
        
        # 残差连接
        emotion_debiased = emotion_debiased + emotion_raw * 0.1
        
        return emotion_debiased


class CPEBModel(nn.Module):
    """
    完整的CPEB模型
    因果个性化情绪基线模型
    输入文本 → BERT编码 → 原始情绪分类 → 用户基线估计 → 因果干预 → 去偏情绪输出

    1. 情绪分类损失:预测正确的情绪类别
    2. 基线正则化损失:学习合理的用户基线
    3. 因果干预效果:去偏后的结果应该更准确
    """
    
    def __init__(
        self,
        bert_model_name: str = "bert-base-chinese",
        num_emotions: int = 8,
        hidden_size: int = 768,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        """
        初始化CPEB模型
        
        Args:
            bert_model_name: BERT模型名称
            num_emotions: 情绪类别数
            hidden_size: 隐藏层大小
            dropout: Dropout概率
            device: 设备
        """
        super().__init__()
        self.num_emotions = num_emotions
        self.hidden_size = hidden_size
        self.device = device
        
        # BERT编码器
        try:
            self.bert = AutoModel.from_pretrained(bert_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        except Exception as e:
            print(f"警告: 无法加载BERT模型 {bert_model_name}: {e}")
            self.bert = None
            self.tokenizer = None
        
        # 基线估计器
        self.baseline_estimator = BaselineEstimator(hidden_size, num_emotions)
        
        # 因果干预模块
        self.causal_intervention = CausalInterventionModule(num_emotions, 256)
        
        # 情绪分类器
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_emotions)
        )
        
        # 情绪嵌入层
        self.emotion_embedding = nn.Linear(num_emotions, hidden_size)
    
    def encode_text(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        编码文本
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            
        Returns:
            文本特征
        """
        if self.bert is None:
            # 模拟BERT输出
            batch_size = input_ids.size(0)
            return torch.randn(batch_size, self.hidden_size).to(self.device)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用[CLS]标记的输出
        text_features = outputs.last_hidden_state[:, 0, :]
        return text_features
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        user_ids: torch.Tensor,
        return_baseline: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        1. BERT编码文本 → 获得文本语义表示
        2. 估计用户基线 → 了解用户的表达习惯
        3. 原始情绪分类 → 直接预测情绪(不考虑习惯)
        4. 因果干预 → 移除习惯影响,得到真实情绪
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            user_ids: 用户ID [batch_size]
            return_baseline: 是否返回基线
            
        Returns:
            emotion_raw: 原始情绪表示 [batch_size, num_emotions]
            emotion_debiased: 去偏后的情绪表示 [batch_size, num_emotions]
            emotion_logits: 情绪分类结果 [batch_size, num_emotions]
            baseline_mean: 用户基线均值 [batch_size, num_emotions]
            baseline_logvar: 用户基线方差 [batch_size, num_emotions]
        """
        # 1. 编码文本
        text_features = self.encode_text(input_ids, attention_mask)
        
        # 2. 估计用户基线
        baseline_mean, baseline_logvar = self.baseline_estimator(user_ids, text_features)
        baseline = self.baseline_estimator.sample_baseline(baseline_mean, baseline_logvar)
        baseline = torch.sigmoid(baseline)  # 归一化到[0,1]
        
        # 3. 原始情绪分类
        emotion_logits = self.emotion_classifier(text_features)
        emotion_raw = torch.softmax(emotion_logits, dim=-1)
        
        # 4. 因果干预
        emotion_debiased = self.causal_intervention(emotion_raw, baseline)
        
        outputs = {
            'emotion_raw': emotion_raw,
            'emotion_debiased': emotion_debiased,
            'emotion_logits': emotion_logits,
            'baseline_mean': baseline_mean,#用于KL散度计算
            'baseline_logvar': baseline_logvar,#用于KL散度计算
            'text_features': text_features
        }
        
        if return_baseline:
            outputs['baseline'] = baseline
        
        return outputs
    
    def predict_emotion(
        self,
        text: str,
        user_id: int,
        use_causal: bool = True
    ) -> np.ndarray:
        """
        预测情绪
        
        Args:
            text: 输入文本
            user_id: 用户ID
            use_causal: 是否使用因果干预
            
        Returns:
            情绪概率分布
        """
        self.eval()
        
        if self.tokenizer is None:
            # 返回模拟结果
            return np.random.dirichlet(np.ones(self.num_emotions))
        
        # 分词
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        user_ids = torch.tensor([user_id]).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, user_ids)
            
            if use_causal:
                emotion_probs = outputs['emotion_debiased']
            else:
                emotion_probs = outputs['emotion_raw']
        
        return emotion_probs.cpu().numpy()[0]
    
    def get_user_baseline(self, user_id: int, text: str = "") -> np.ndarray:
        """
        获取用户情绪基线
        
        Args:
            user_id: 用户ID
            text: 参考文本（可选）
            
        Returns:
            用户基线向量
        """
        self.eval()
        
        if text == "":
            text = "今天天气不错"  # 默认中性文本
        
        if self.tokenizer is None:
            return np.random.rand(self.num_emotions)
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        user_ids = torch.tensor([user_id]).to(self.device)
        
        with torch.no_grad():
            text_features = self.encode_text(input_ids, attention_mask)
            baseline_mean, _ = self.baseline_estimator(user_ids, text_features)
            baseline = torch.sigmoid(baseline_mean)
        
        return baseline.cpu().numpy()[0]


# 使用示例
if __name__ == '__main__':
    print("正在初始化CPEB模型...")
    
    # 创建模型
    model = CPEBModel(
        bert_model_name="bert-base-chinese",
        num_emotions=8,
        hidden_size=768,
        device='cpu'
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试
    test_texts = [
        "今天心情很好",
        "我感觉很难过",
        "真的撑不住了"
    ]
    
    emotion_labels = ['中性', '高兴', '惊讶', '悲伤', '愤怒', '恐惧', '厌恶', '绝望']
    
    print("\n" + "="*60)
    print("测试CPEB模型")
    print("="*60)
    
    for text in test_texts:
        # 预测情绪（不使用因果干预）
        emotion_raw = model.predict_emotion(text, user_id=1, use_causal=False)
        
        # 预测情绪（使用因果干预）
        emotion_debiased = model.predict_emotion(text, user_id=1, use_causal=True)
        
        print(f"\n文本: {text}")
        print("原始情绪分布:")
        for label, prob in zip(emotion_labels, emotion_raw):
            if prob > 0.1:
                print(f"  {label}: {prob:.2%}")
        
        print("因果干预后:")
        for label, prob in zip(emotion_labels, emotion_debiased):
            if prob > 0.1:
                print(f"  {label}: {prob:.2%}")
    
    # 获取用户基线
    print("\n" + "="*60)
    print("用户基线")
    print("="*60)
    baseline = model.get_user_baseline(user_id=1)
    print("用户1的情绪基线:")
    for label, value in zip(emotion_labels, baseline):
        if value > 0.1:
            print(f"  {label}: {value:.2%}")
    
    print("\n" + "="*60)
    print("测试完成!")


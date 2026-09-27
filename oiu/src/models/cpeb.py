"""
CPEB: Causal Personalized Emotion Baseline
因果个性化情绪基线模型

使用因果推断消除个体表达习惯的混淆偏差
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
        self.user_embeddings = nn.Embedding(10000, self.user_embedding_size)
        
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
        
        Args:
            baseline_mean: 基线均值
            baseline_logvar: 基线对数方差
            
        Returns:
            采样的基线
        """
        std = torch.exp(0.5 * baseline_logvar)
        eps = torch.randn_like(std)
        return baseline_mean + eps * std


class CausalInterventionModule(nn.Module):
    """
    因果干预模块
    使用前门调整消除混淆偏差，并支持反事实用户基线对照。
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
        
        # 表达习惯编码器
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

    def counterfactual_debias(
        self,
        emotion_raw: torch.Tensor,
        factual_baseline: torch.Tensor,
        counterfactual_baseline: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        同一句话在不同用户表达习惯基线下的去偏对照。

        用于分离「稳定表达习惯」与「瞬时情绪状态」。
        """
        emotion_factual = self.forward(emotion_raw, factual_baseline)
        emotion_cf = self.forward(emotion_raw, counterfactual_baseline)
        return {
            'emotion_factual': emotion_factual,
            'emotion_counterfactual': emotion_cf,
            'habit_gap': (factual_baseline - counterfactual_baseline).abs().mean(dim=1)
        }


class CPEBModel(nn.Module):
    """
    完整的CPEB模型
    因果个性化情绪基线模型
    """
    
    # 默认情绪顺序中「绝望」索引，用于危机相关因果一致性约束
    DESPAIR_INDEX = 7

    def __init__(
        self,
        bert_model_name: str = "bert-base-chinese",
        num_emotions: int = 8,
        hidden_size: int = 768,
        dropout: float = 0.1,
        device: str = "cpu",
        despair_index: int = 7,
        despair_consistency_threshold: float = 0.25,
    ):
        """
        初始化CPEB模型
        
        Args:
            bert_model_name: BERT模型名称
            num_emotions: 情绪类别数
            hidden_size: 隐藏层大小
            dropout: Dropout概率
            device: 设备
            despair_index: 高危情绪「绝望」类别索引
            despair_consistency_threshold: 触发绝望一致性约束的内容侧阈值
        """
        super().__init__()
        self.num_emotions = num_emotions
        self.hidden_size = hidden_size
        self.device = device
        self.despair_index = despair_index
        self.despair_consistency_threshold = despair_consistency_threshold
        
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

    def _build_counterfactual_baseline(
        self,
        baseline: torch.Tensor,
        user_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        构造反事实用户基线：优先在 batch 内置换不同用户的基线；
        若 batch 内用户单一，则回退到标准正态先验采样。
        """
        batch_size = baseline.size(0)
        if batch_size == 1:
            return torch.sigmoid(torch.randn_like(baseline))

        # 循环移位，尽量换成其他用户习惯
        shifted = torch.roll(baseline, shifts=1, dims=0)
        same_user = (user_ids == torch.roll(user_ids, shifts=1, dims=0)).unsqueeze(1)
        prior = torch.sigmoid(torch.randn_like(baseline))
        return torch.where(same_user, prior, shifted)

    def compute_causal_aux_losses(
        self,
        emotion_raw: torch.Tensor,
        emotion_debiased: torch.Tensor,
        emotion_cf: torch.Tensor,
        baseline_mean: torch.Tensor,
        baseline_logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        CPEB 辅助损失：
        - baseline_kl: 约束用户基线不过拟合（贴近 N(0,I)）
        - counterfactual_consistency: 内容驱动的高危情绪在习惯干预下应保持稳定
        - despair_consistency: 对「绝望」类的专门因果一致性
        """
        baseline_kl = -0.5 * torch.mean(
            1 + baseline_logvar - baseline_mean.pow(2) - baseline_logvar.exp()
        )

        # 内容侧已显出绝望倾向时，去偏结果应对用户习惯干预更稳健
        despair_raw = emotion_raw[:, self.despair_index]
        content_mask = (despair_raw >= self.despair_consistency_threshold).float()
        despair_gap = (
            emotion_debiased[:, self.despair_index] - emotion_cf[:, self.despair_index]
        ).abs()
        if content_mask.sum() > 0:
            despair_consistency = (despair_gap * content_mask).sum() / content_mask.sum().clamp_min(1.0)
        else:
            despair_consistency = despair_gap.mean() * 0.0

        # 整体反事实一致性（轻度）：防止习惯编码吞掉语义
        cf_consistency = F.kl_div(
            torch.log(emotion_debiased.clamp_min(1e-8)),
            emotion_cf.clamp_min(1e-8),
            reduction='batchmean'
        )

        return {
            'baseline_kl': baseline_kl,
            'despair_consistency': despair_consistency,
            'counterfactual_consistency': cf_consistency,
        }
    
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
        return_baseline: bool = False,
        return_counterfactual: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            user_ids: 用户ID [batch_size]
            return_baseline: 是否返回基线
            return_counterfactual: 是否计算反事实去偏结果与因果辅助损失
            
        Returns:
            输出字典
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
        
        # 4. 因果干预（事实基线）
        emotion_debiased = self.causal_intervention(emotion_raw, baseline)
        
        outputs = {
            'emotion_raw': emotion_raw,
            'emotion_debiased': emotion_debiased,
            'emotion_logits': emotion_logits,
            'baseline_mean': baseline_mean,
            'baseline_logvar': baseline_logvar,
            'text_features': text_features
        }
        
        if return_baseline:
            outputs['baseline'] = baseline

        # 5. 反事实用户习惯干预：同一语义、不同表达习惯
        if return_counterfactual:
            cf_baseline = self._build_counterfactual_baseline(baseline, user_ids)
            cf_outputs = self.causal_intervention.counterfactual_debias(
                emotion_raw, baseline, cf_baseline
            )
            aux_losses = self.compute_causal_aux_losses(
                emotion_raw=emotion_raw,
                emotion_debiased=emotion_debiased,
                emotion_cf=cf_outputs['emotion_counterfactual'],
                baseline_mean=baseline_mean,
                baseline_logvar=baseline_logvar,
            )
            outputs.update({
                'baseline_counterfactual': cf_baseline,
                'emotion_counterfactual': cf_outputs['emotion_counterfactual'],
                'habit_gap': cf_outputs['habit_gap'],
                'causal_aux_losses': aux_losses,
            })
        
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


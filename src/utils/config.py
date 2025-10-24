"""
配置管理模块
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # BERT配置
    bert_model: str = "bert-base-chinese"
    hidden_size: int = 768
    num_emotion_labels: int = 8
    dropout: float = 0.1
    
    # CPEB配置
    baseline_estimation_method: str = "variational"  # variational / moment
    causal_intervention_method: str = "front_door"  # front_door / back_door
    
    # MEA配置
    meta_learning_algorithm: str = "maml"  # maml / prototypical
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    num_support_samples: int = 5
    num_query_samples: int = 10
    
    # THEGN配置
    graph_hidden_size: int = 256
    num_graph_layers: int = 3
    num_attention_heads: int = 8
    temporal_decay_lambda: float = 0.1
    edge_types: List[str] = field(default_factory=lambda: ['temporal', 'reply', 'influence'])


@dataclass
class MonitoringConfig:
    """监测系统配置"""
    # 第一层配置
    layer1_model: str = "distilbert-base-chinese"
    layer1_max_length: int = 128
    layer1_threshold: float = 7.0
    
    # 第二层配置
    layer2_threshold: Dict[str, float] = field(default_factory=lambda: {
        'green': 3.0,
        'blue': 5.0,
        'orange': 7.0,
        'red': 9.0
    })
    
    # 极端关键词
    extreme_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        'critical': ['自杀', '轻生', '结束生命', '不想活', '去死'],
        'severe': ['活不下去', '没有希望', '彻底崩溃', '撑不住'],
        'warning': ['抑郁', '绝望', '崩溃', '无助', '孤独']
    })


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # 损失权重
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'emotion': 1.0,
        'causal': 0.5,
        'meta': 0.3,
        'graph': 0.2
    })
    
    # 优化器
    optimizer: str = "adamw"
    scheduler: str = "linear"
    
    # 早停
    early_stopping_patience: int = 10
    
    # 日志
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000


@dataclass
class AlertConfig:
    """预警系统配置"""
    smtp_server: str = "smtp.example.com"
    smtp_port: int = 587
    sender_email: str = "alert@example.com"
    sender_password: str = ""
    admin_emails: List[str] = field(default_factory=list)
    
    # 预警阈值
    alert_threshold: float = 9.0
    alert_cooldown_minutes: int = 30  # 同一用户预警冷却时间


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    
    # 路径
    data_dir: str = "data"
    model_dir: str = "models"
    result_dir: str = "results"
    log_dir: str = "logs"
    
    # 设备
    device: str = "cuda"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f, allow_unicode=True)


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


if __name__ == '__main__':
    # 测试配置
    config = get_default_config()
    print("默认配置创建成功!")
    print(f"模型隐藏层大小: {config.model.hidden_size}")
    print(f"监测阈值: {config.monitoring.layer1_threshold}")
    print(f"训练批次大小: {config.training.batch_size}")


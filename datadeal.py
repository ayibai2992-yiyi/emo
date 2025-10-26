"""
情感数据处理通用Pipeline
支持数据下载、预处理、标注、划分的完整流程
"""

import os
import json
import re
import unicodedata
import logging
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import random
import requests
from tqdm import tqdm
from pathlib import Path
from typing import NamedTuple
import jieba
import jieba.posseg as pseg
from jieba import analyse
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import json
import numpy as np
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置类 ====================
@dataclass
class PipelineConfig:
    """Pipeline配置"""
    # 数据目录
    base_dir: str = "./emotion_data"
    raw_dir: str = "./emotion_data/raw"
    output_dir: str = "./emotion_data/output"
    
    # 数据集划分比例
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # BERT模型
    bert_model: str = "bert-base-chinese"
    use_gpu: bool = True
    
    # 随机种子
    random_seed: int = 42


# ==================== 数据下载器 ====================
class DatasetDownloader:
    """数据集下载器"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集URL配置
        self.dataset_urls = {
            'nlpcc2014':{
                'url': 'https://dataset-bj.cdn.bcebos.com/qianyan/THUCNews.zip',
                'type': 'zip'
            },
            'nlpcc2013': {
                'url': 'https://dataset-bj.cdn.bcebos.com/qianyan/ChnSentiCorp.zip',
                'type': 'zip'
            },
            'psy_insight': {
                'url': 'https://github.com/ckqqqq/Psy-Insight',
                'type': 'github_repo'
            }

        }
    
    def download_file(self, url: str, dest_path: str) -> bool:
        """下载文件"""
        try:
            logger.info(f"开始下载: {url}")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            # 使用tqdm或更简洁的日志
                            print(f"\r下载进度: {progress:.1f}%", end="")
            
            print() # 换行
            logger.info(f"下载完成: {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False
    
    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        """解压文件"""
        try:
            logger.info(f"解压文件: {archive_path}")
            
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.error(f"不支持的压缩格式: {archive_path}")
                return False
            
            logger.info(f"解压完成: {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"解压失败: {e}")
            return False
    
    def download_dataset(self, dataset_name: str) -> bool:
        """下载指定数据集"""
        if dataset_name not in self.dataset_urls:
            logger.error(f"未知数据集: {dataset_name}")
            return False
        
        dataset_info = self.dataset_urls[dataset_name]
        url = dataset_info['url']
        file_ext = dataset_info['type']

        # 特殊处理github_repo
        if file_ext == 'github_repo':
            logger.info(f"数据集 '{dataset_name}' 是一个GitHub仓库: {url}")
            logger.info("请手动访问该页面以下载数据。跳过自动下载。")
            return False # 返回False表示未自动下载

        # 目标文件路径
        archive_name = f"{dataset_name}.{file_ext}"
        archive_path = self.raw_dir / archive_name
        extract_dir = self.raw_dir / dataset_name
        
        # 检查是否已下载
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"数据集已存在: {dataset_name}")
            return True
        
        # 下载文件
        if not archive_path.exists():
            if not self.download_file(url, str(archive_path)):
                return False
        
        # 如果不是压缩文件 (例如 .json), 直接移动
        if file_ext in ['json', 'txt', 'csv', 'tsv']:
             logger.info(f"文件 {archive_name} 不是压缩包, 将其移动到目标目录...")
             extract_dir.mkdir(parents=True, exist_ok=True)
             target_file = extract_dir / archive_name
             if target_file.exists():
                 target_file.unlink()
             archive_path.rename(target_file)
             return True

        # 解压文件
        if not self.extract_archive(str(archive_path), str(extract_dir)):
            return False
        
        return True
    
    def download_all(self, datasets: List[str] = None) -> Dict[str, bool]:
        """下载所有数据集"""
        if datasets is None:
            datasets = list(self.dataset_urls.keys())
        
        results = {}
        for dataset_name in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"处理数据集: {dataset_name}")
            logger.info(f"{'='*60}")
            results[dataset_name] = self.download_dataset(dataset_name)
        
        return results


# ==================== 文本预处理器 ====================
class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self):
        self.emoji_mapping = self._init_emoji_mapping()
        self.netspeak_mapping = self._init_netspeak_mapping()
        self.noise_patterns = self._init_noise_patterns()
        
    def _init_emoji_mapping(self) -> Dict[str, str]:
        """表情符号映射"""
        return {
            '😀': '[开心]', '😃': '[开心]', '😄': '[开心]', '😊': '[微笑]',
            '😢': '[哭泣]', '😭': '[大哭]', '😠': '[生气]', '😡': '[愤怒]',
            '😨': '[害怕]', '😰': '[焦虑]', '😱': '[惊恐]', '😴': '[困倦]',
            '👍': '[赞]', '👎': '[踩]', '❤️': '[心]', '💕': '[心]',
            '🔥': '[火]', '💯': '[满分]', '⭐': '[星星]', '✨': '[闪亮]'
        }
    
    def _init_netspeak_mapping(self) -> Dict[str, str]:
        """网络用语映射"""
        return {
            '666': '很厉害', '233': '哈哈哈', '555': '呜呜呜', '88': '拜拜',
            'yyds': '永远的神', 'awsl': '我死了', 'nsdd': '你说的对', 'xswl': '笑死我了',
            'emo': '情绪低落', 'yysy': '有一说一', 'nb': '牛逼', 'gkd': '搞快点',
            'orz': '跪了', 'QAQ': '哭哭', 'TAT': '哭哭', '>_<': '郁闷', '-_-': '无语'
        }
    
    def _init_noise_patterns(self) -> List[Tuple]:
        """噪声清理模式"""
        return [
            (re.compile(r'<[^>]+>'), ''),  # HTML标签
            (re.compile(r'http[s]?://\S+'), '[链接]'),  # URL
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[邮箱]'),
            (re.compile(r'1[3-9]\d{9}'), '[电话]'),  # 手机号
            (re.compile(r'[!！]{3,}'), '!!!'),  # 重复标点
            (re.compile(r'[?？]{3,}'), '???'),
            (re.compile(r'[.。]{3,}'), '...'),
            (re.compile(r'\s{2,}'), ' ')  # 多余空格
        ]
    
    def standardize(self, text: str) -> str:
        """文本标准化"""
        if not text:
            return ''
        
        # 清理噪声
        for pattern, replacement in self.noise_patterns:
            text = pattern.sub(replacement, text)
        
        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        
        # 全角转半角
        text = text.translate(str.maketrans(
            'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
            'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
            '０１２３４５６７８９',
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789'
        ))
        
        # 标点统一
        punct_map = {
            '，': ',', '。': '.', '！': '!', '？': '?', '：': ':',
            '；': ';', '"': '"', '"': '"', ''': "'", ''': "'"
        }
        for full, half in punct_map.items():
            text = text.replace(full, half)
        
        # 表情符号转换
        for emoji, label in self.emoji_mapping.items():
            text = text.replace(emoji, label)
        
        # 网络用语标准化
        for netspeak, standard in self.netspeak_mapping.items():
            pattern = r'\b' + re.escape(netspeak) + r'\b'
            text = re.sub(pattern, standard, text, flags=re.IGNORECASE)
        
        # 重复字符规整
        text = re.compile(r'([\u4e00-\u9fff])\1{2,}').sub(r'\1\1\1', text)
        text = re.compile(r'([a-zA-Z])\1{3,}').sub(r'\1\1\1', text)
        
        return text.strip()


# ==================== 情绪分类器 ====================
class EmotionClassifier:
    """情绪分类器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(
            'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu'
        )
        
        # 初始化BERT模型
        logger.info(f"加载BERT模型: {config.bert_model}")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_model)
            self.model = BertModel.from_pretrained(config.bert_model)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"BERT模型加载成功! 使用设备: {self.device}")
        except Exception as e:
            logger.warning(f"BERT模型加载失败: {e}")
            logger.warning("将使用关键词模式进行分类")
            self.model = None
        
        # 情绪映射表
        self.emotion_scores = self._init_emotion_scores()
        self.chinese_emotion_mapping = self._init_chinese_mapping()
        self.emotion_keywords = self._init_emotion_keywords()
    
    def _init_emotion_scores(self) -> Dict[str, float]:
        """初始化情绪得分映射"""
        return {
            # 无风险区间 (0-2分)
            'none': 0, 'neutral': 1, 'happy': 2, 'like': 2, 'confident': 2,
            
            # 低风险区间 (3-4分)
            'surprise': 3, 'embarrassed': 3, 'stress': 4, 'confused': 4, 'excitement': 4,
            
            # 中风险区间 (5-6分)
            'disgust': 5, 'anxiety': 5, 'lonely': 5, 'sadness': 6,
            'overwhelmed': 6, 'rejected': 6, 'frustrated': 6,
            
            # 中高风险区间 (7-8分)
            'fear': 7, 'burnout': 7, 'angry': 8, 'bullied': 8,
            
            # 高风险区间 (9-10分)
            'despair': 9, 'worthless': 9, 'rage': 9,
            'extreme': 10, 'suicidal': 10, 'self_harm': 10
        }
    
    def _init_chinese_mapping(self) -> Dict[str, str]:
        """中文情绪关键词映射"""
        return {
            '普通': 'neutral', '平静': 'neutral', '淡然': 'neutral', '一般': 'neutral',
            '开心': 'happy', '高兴': 'happy', '快乐': 'happy', '喜悦': 'happy', '满意': 'happy',
            '无聊': 'none', '惊讶': 'surprise', '意外': 'surprise',
            '兴奋': 'excitement', '激动': 'excitement', '紧张': 'stress', '震惊': 'surprise',
            '郁闷': 'frustrated', '烦躁': 'frustrated', '担心': 'anxiety', '焦虑': 'anxiety',
            '讨厌': 'disgust', '厌恶': 'disgust', '反感': 'disgust',
            '难过': 'sadness', '伤心': 'sadness', '沮丧': 'sadness', '失落': 'sadness',
            '痛苦': 'despair', '生气': 'angry', '恼火': 'angry', '害怕': 'fear',
            '愤怒': 'angry', '恐惧': 'fear'
        }
    
    def _init_emotion_keywords(self) -> Dict[str, List[str]]:
        """初始化情绪关键词"""
        return {
            'positive': ['开心', '高兴', '快乐', '喜悦', '满意', '棒', '好', '爱', '喜欢'],
            'negative': ['难过', '伤心', '痛苦', '绝望', '讨厌', '厌恶', '生气', '愤怒'],
            'neutral': ['普通', '一般', '平常', '还好', '还行'],
            'high_risk': ['想死', '自杀', '绝望', '没救', '完了', '自残', '伤害自己']
        }
    
    def analyze_emotion_keywords(self, text: str) -> str:
        """基于关键词分析情感"""
        for keyword, emotion in self.chinese_emotion_mapping.items():
            if keyword in text:
                return emotion
        return 'neutral'
    
    def analyze_emotion_context(self, text: str) -> str:
        """基于上下文分析情感"""
        positive_words = ['决心', '目标', '必胜', '好办', '计划', '自信', '满意', '开心', '高兴']
        negative_words = ['焦虑', '压力', '痛苦', '难过', '恐惧', '愤怒', '绝望', '无助']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if negative_count > positive_count:
            if '焦虑' in text or '担心' in text:
                return 'anxiety'
            elif '愤怒' in text or '生气' in text:
                return 'angry'
            elif '难过' in text or '伤心' in text:
                return 'sadness'
            elif '害怕' in text or '恐惧' in text:
                return 'fear'
            else:
                return 'frustrated'
        elif positive_count > negative_count:
            if '兴奋' in text or '激动' in text:
                return 'excitement'
            elif '自信' in text or '决心' in text:
                return 'confident'
            else:
                return 'happy'
        else:
            return 'neutral'
    
    def get_bert_embedding(self, text: str) -> Optional[torch.Tensor]:
        """获取BERT嵌入向量"""
        if self.model is None:
            return None
        
        try:
            text = text.strip() if text else "无内容"
            
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            return cls_embedding
        except Exception as e:
            logger.warning(f"BERT嵌入提取失败: {e}")
            return None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """分析文本情感倾向"""
        sentiment_scores = {
            'positive': 0.5,
            'negative': 0.5,
            'high_risk': 0.0,
            'intensity': 0.5
        }
        
        # BERT分析
        embedding = self.get_bert_embedding(text)
        if embedding is not None:
            embedding_np = embedding.cpu().numpy()[0]
            
            pos_part = embedding_np[embedding_np > 0]
            neg_part = embedding_np[embedding_np < 0]
            
            positive_score = np.mean(pos_part) if pos_part.size > 0 else 0.0
            negative_score = -np.mean(neg_part) if neg_part.size > 0 else 0.0
            
            positive_score = max(0, positive_score)
            negative_score = max(0, negative_score)
            
            total = positive_score + negative_score
            if total > 0:
                sentiment_scores['positive'] = positive_score / total
                sentiment_scores['negative'] = negative_score / total
            
            sentiment_scores['intensity'] = np.std(embedding_np)
        
        # 关键词辅助判断
        high_risk_count = sum(1 for word in self.emotion_keywords['high_risk'] if word in text)
        positive_count = sum(1 for word in self.emotion_keywords['positive'] if word in text)
        negative_count = sum(1 for word in self.emotion_keywords['negative'] if word in text)
        
        sentiment_scores['positive'] += positive_count * 0.1
        sentiment_scores['negative'] += negative_count * 0.1
        sentiment_scores['high_risk'] = high_risk_count * 0.3
        
        return sentiment_scores
    
    def classify_emotion(self, text: str) -> Tuple[str, float]:
        """分类单个句子的情感"""
        # 首先尝试关键词匹配
        emotion = self.analyze_emotion_keywords(text)
        
        # 如果没有找到关键词,使用上下文分析
        if emotion == 'neutral':
            emotion = self.analyze_emotion_context(text)
        
        # 特殊规则处理
        if '焦虑' in text:
            emotion = 'anxiety'
        elif '决心' in text or '目标' in text:
            emotion = 'confident'
        elif '放松' in text and '?' in text:
            emotion = 'frustrated'
        elif '计划' in text and '!' in text:
            emotion = 'confident'
        
        # 获取基础分数
        base_score = self.emotion_scores.get(emotion, 1.0)
        
        # 使用情感分析调整分数
        sentiment = self.analyze_sentiment(text)
        adjustment = 0.0
        
        if sentiment['high_risk'] > 0:
            adjustment += min(sentiment['high_risk'] * 3, 3.0)
        
        intensity = sentiment['intensity']
        if intensity > 0.5:
            adjustment += (intensity - 0.5) * 2
        elif intensity < 0.3:
            adjustment -= (0.3 - intensity) * 1
        
        if base_score <= 2 and sentiment['negative'] > sentiment['positive']:
            adjustment += (sentiment['negative'] - sentiment['positive']) * 2
        elif base_score >= 6 and sentiment['positive'] > sentiment['negative']:
            adjustment -= (sentiment['positive'] - sentiment['negative']) * 1.5
        
        final_score = max(0.0, min(10.0, base_score + adjustment))
        
        # 显式转换为 Python float 类型
        return emotion, round(float(final_score), 2)


# ==================== 数据处理Pipeline ====================
class EmotionDataPipeline:
    """情感数据处理Pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.downloader = DatasetDownloader(config)
        self.preprocessor = TextPreprocessor()
        self.classifier = EmotionClassifier(config)
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def run(self, datasets: List[str] = None, skip_download: bool = False):
        """运行完整Pipeline"""
        logger.info("="*60)
        logger.info("情感数据处理Pipeline启动")
        logger.info("="*60)
        
        # Step 1: 下载数据集
        if not skip_download:
            logger.info("\n[步骤 1/4] 下载数据集")
            download_results = self.downloader.download_all(datasets)
            logger.info(f"下载结果: {download_results}")
            
            # 检查是否有下载成功的数据集
            if not any(download_results.values()):
                logger.error("所有数据集下载/检查失败，程序终止")
                logger.info("\n建议:")
                logger.info("  1. 检查网络连接")
                logger.info("  2. 检查 'psy_insight' (如果包含在内) 是否需要手动下载")
                logger.info("  3. 手动下载数据集并放入 raw/ 目录")
                logger.info("  4. 然后使用 --skip-download 参数重新运行")
                return
        else:
            logger.info("\n[步骤 1/4] 跳过下载")
        
        # Step 2: 加载和预处理数据
        logger.info("\n[步骤 2/4] 加载和预处理数据")
        all_data = self.load_and_preprocess_data()
        
        if len(all_data) == 0:
            logger.error("没有加载到任何数据，程序终止")
            logger.info("\n请检查:")
            logger.info(f"  1. {self.config.raw_dir} 目录下是否有数据文件")
            logger.info("  2. 数据文件格式是否正确 (.txt, .tsv, .csv, .json)")
            logger.info("  3. 是否需要先运行下载步骤（不使用 --skip-download）")
            return
        
        logger.info(f"成功加载 {len(all_data)} 条数据")
        
        # 显示数据样例
        logger.info("\n数据样例（前3条）:")
        for i, item in enumerate(all_data[:3], 1):
            logger.info(f"  {i}. [{item['emotion']}|{item['score']}] {item['text'][:50]}...")
        
        # Step 3: 划分数据集
        logger.info("\n[步骤 3/4] 划分数据集")
        train_data, val_data, test_data = self.split_dataset(all_data)
        
        if len(train_data) == 0:
            logger.error("数据集划分失败，程序终止")
            return
        
        # Step 4: 保存处理后的数据
        logger.info("\n[步骤 4/4] 保存处理后的数据")
        self.save_datasets(train_data, val_data, test_data)
        
        logger.info("\n"+"="*60)
        logger.info(" Pipeline执行完成!")
        logger.info("="*60)
        logger.info(f"\n 输出目录: {self.config.output_dir}")
        logger.info(f" 数据统计:")
        logger.info(f"   - 训练集: {len(train_data)} 条")
        logger.info(f"   - 验证集: {len(val_data)} 条")
        logger.info(f"   - 测试集: {len(test_data)} 条")
        logger.info(f"   - 总计: {len(all_data)} 条")
    
    def load_and_preprocess_data(self) -> List[Dict]:
        """加载和预处理所有数据"""
        all_data = []
        raw_dir = Path(self.config.raw_dir)
        
        # 检查目录是否存在
        if not raw_dir.exists():
            logger.error(f"原始数据目录不存在: {raw_dir}")
            logger.info("请先下载数据或将数据放入该目录")
            return all_data
        
        # 遍历所有数据集目录
        processed_files = 0
        for dataset_dir in raw_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            logger.info(f"扫描数据集目录: {dataset_dir.name}")
            
            # 递归查找所有文本文件
            for file_path in dataset_dir.rglob('*'):
                if file_path.suffix in ['.txt', '.tsv', '.csv', '.json']:
                    logger.info(f"  处理文件: {file_path.name}")
                    try:
                        data = self.process_file(file_path)
                        if data:
                            all_data.extend(data)
                            processed_files += 1
                            logger.info(f"    ✓ 成功加载 {len(data)} 条数据")
                        else:
                            logger.warning(f"    ✗ 文件为空或格式不正确")
                    except Exception as e:
                        logger.error(f"    ✗ 处理失败: {e}")
        
        logger.info(f"\n总计处理 {processed_files} 个文件，加载 {len(all_data)} 条数据")
        
        if len(all_data) == 0:
            logger.warning("\n 警告: 没有加载到任何数据!")
            logger.info("\n可能的原因:")
            logger.info("  1. 数据文件不存在于 raw 目录")
            logger.info("  2. 数据文件格式不正确")
            logger.info("  3. 文件编码问题")
            logger.info("\n建议操作:")
            logger.info("  1. 检查 raw 目录是否有数据文件")
            logger.info("  2. 确保文件格式为 .txt, .tsv, .csv 或 .json")
            logger.info("  3. 运行时不要使用 --skip-download (除非已手动放置数据)")
        
        return all_data
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """处理单个文件"""
        data = []
        
        try:
            if file_path.suffix == '.txt':
                data = self.process_txt_file(file_path)
            elif file_path.suffix in ['.tsv', '.csv']:
                data = self.process_tsv_file(file_path)
            elif file_path.suffix == '.json':
                data = self.process_json_file(file_path)
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
        
        return data
    
    def process_txt_file(self, file_path: Path) -> List[Dict]:
        """处理TXT文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    standardized = self.preprocessor.standardize(line)
                    emotion, score = self.classifier.classify_emotion(standardized)
                    
                    data.append({
                        'text': standardized,
                        'original': line,
                        'emotion': emotion,
                        'score': score,
                        'source': file_path.stem
                    })
        return data
    
    def process_tsv_file(self, file_path: Path) -> List[Dict]:
        """处理TSV/CSV文件"""
        data = []
        delimiter = '\t' if file_path.suffix == '.tsv' else ','
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 0:  # 跳过表头
                    continue
                
                parts = line.strip().split(delimiter)
                if len(parts) >= 2:
                    text = parts[-1]  # 最后一列通常是文本
                    original_label = parts[0] if len(parts) >= 2 else None
                    
                    standardized = self.preprocessor.standardize(text)
                    emotion, score = self.classifier.classify_emotion(standardized)
                    
                    data.append({
                        'text': standardized,
                        'original': text,
                        'emotion': emotion,
                        'score': score,
                        'original_label': original_label,
                        'source': file_path.stem
                    })
        return data
    
    def process_json_file(self, file_path: Path) -> List[Dict]:
        """处理JSON文件"""
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if isinstance(json_data, list):
                for item in json_data:
                    text = item.get('text', item.get('content', ''))
                    if text:
                        standardized = self.preprocessor.standardize(text)
                        emotion, score = self.classifier.classify_emotion(standardized)
                        
                        data.append({
                            'text': standardized,
                            'original': text,
                            'emotion': emotion,
                            'score': score,
                            'original_label': item.get('label'),
                            'source': file_path.stem
                        })
        except Exception as e:
            logger.error(f"处理JSON文件失败: {e}")
        
        return data
    
    def split_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """划分数据集"""
        # 检查数据是否为空
        if len(data) == 0:
            logger.error("数据为空，无法划分数据集!")
            return [], [], []
        
        # 检查数据量是否足够
        min_samples = 10  # 最少需要10条数据
        if len(data) < min_samples:
            logger.error(f"数据量太少（{len(data)}条），至少需要{min_samples}条数据")
            logger.info("建议: 添加更多数据或调整数据源")
            return [], [], []
        
        logger.info(f"开始划分数据集，总数据量: {len(data)}")
        
        # 检查是否所有数据都有相同的emotion标签（如果是，分层抽样会失败）
        emotions = [d['emotion'] for d in data]
        emotion_counts = Counter(emotions)
        
        # 如果某个类别样本数小于2，无法进行分层抽样
        min_class_count = min(emotion_counts.values())
        if min_class_count < 2:
            logger.warning(f"某些情绪类别样本数太少（最少{min_class_count}个），使用随机划分代替分层抽样")
            # 使用简单随机划分
            random.shuffle(data)
            train_size = int(len(data) * self.config.train_ratio)
            val_size = int(len(data) * self.config.val_ratio)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]
            
            logger.info(f"随机划分完成:")
            logger.info(f"  训练集: {len(train_data)} 条")
            logger.info(f"  验证集: {len(val_data)} 条")
            logger.info(f"  测试集: {len(test_data)} 条")
            
            return train_data, val_data, test_data
        
        try:
            # 首先划分训练集和临时集
            train_data, temp_data = train_test_split(
                data,
                test_size=(self.config.val_ratio + self.config.test_ratio),
                random_state=self.config.random_seed,
                stratify=emotions
            )
            
            # 再划分验证集和测试集
            val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
            temp_emotions = [d['emotion'] for d in temp_data]

            # 检查临时集中的类别数量
            temp_emotion_counts = Counter(temp_emotions)
            min_temp_class_count = min(temp_emotion_counts.values())

            if min_temp_class_count < 2:
                logger.warning(f"临时数据集中类别样本数太少（最少{min_temp_class_count}个），对 val/test 进行随机划分")
                random.shuffle(temp_data)
                val_size = int(len(temp_data) * val_ratio_adjusted)
                val_data = temp_data[:val_size]
                test_data = temp_data[val_size:]
            
            else:
                val_data, test_data = train_test_split(
                    temp_data,
                    test_size=(1 - val_ratio_adjusted),
                    random_state=self.config.random_seed,
                    stratify=temp_emotions
                )
            
            logger.info(f"分层抽样划分完成:")
            logger.info(f"  训练集: {len(train_data)} 条 ({len(train_data)/len(data)*100:.1f}%)")
            logger.info(f"  验证集: {len(val_data)} 条 ({len(val_data)/len(data)*100:.1f}%)")
            logger.info(f"  测试集: {len(test_data)} 条 ({len(test_data)/len(data)*100:.1f}%)")
            
        except Exception as e:
            logger.warning(f"分层抽样失败: {e}")
            logger.info("使用随机划分...")
            
            # 降级到随机划分
            random.shuffle(data)
            train_size = int(len(data) * self.config.train_ratio)
            val_size = int(len(data) * self.config.val_ratio)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def save_datasets(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """保存数据集"""
        output_dir = Path(self.config.output_dir)
        
        # 保存为JSON格式
        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        def convert_to_python_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                # 额外处理NaN，将其转为None (JSON中的null)
                if np.isnan(obj):
                    return None
                return float(obj)  
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  
            elif isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            else:
                return obj  
        
        for name, data in datasets.items():
            
            data_serializable = convert_to_python_types(data)
            
            # 保存JSON
            json_path = output_dir / f"{name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                # 修复: 使用转换后的数据
                json.dump(data_serializable, f, ensure_ascii=False, indent=2)
            logger.info(f"保存 {name} 数据集到: {json_path}")
            
            # 保存JSONL格式(每行一个JSON对象,方便流式读取)
            jsonl_path = output_dir / f"{name}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                # 修复: 遍历转换后的数据
                for item in data_serializable:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"保存 {name} 数据集到: {jsonl_path}")
        

            # 保存TSV格式(方便查看)
            tsv_path = output_dir / f"{name}.tsv"
            with open(tsv_path, 'w', encoding='utf-8') as f:
                # 写表头
                f.write("score\temotion\ttext\tsource\n")
                for item in data: # TSV写入时不需要转换
                    f.write(f"{item['score']}\t{item['emotion']}\t{item['text']}\t{item['source']}\n")
            logger.info(f"保存 {name} 数据集到: {tsv_path}")
        
        # 生成统计报告
        self.generate_statistics_report(train_data, val_data, test_data)
    
    def generate_statistics_report(self, train_data: List[Dict], 
                                   val_data: List[Dict], 
                                   test_data: List[Dict]):
        """生成统计报告"""
        report_path = Path(self.config.output_dir) / "statistics_report.txt"
        
        all_data = train_data + val_data + test_data
        if not all_data:
            logger.warning("没有数据可用于生成统计报告。")
            return

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("情感数据集统计报告\n")
            f.write("="*60 + "\n\n")
            
            # 总体统计
            total = len(all_data)
            f.write(f"总数据量: {total}\n")
            f.write(f"训练集: {len(train_data)} ({len(train_data)/total*100:.1f}%)\n")
            f.write(f"验证集: {len(val_data)} ({len(val_data)/total*100:.1f}%)\n")
            f.write(f"测试集: {len(test_data)} ({len(test_data)/total*100:.1f}%)\n\n")
            
            # 各数据集的情绪分布
            for name, data in [('训练集', train_data), ('验证集', val_data), ('测试集', test_data)]:
                if not data:
                    continue
                f.write(f"\n{name}情绪分布:\n")
                f.write("-"*40 + "\n")
                
                emotion_counts = Counter([d['emotion'] for d in data])
                for emotion, count in emotion_counts.most_common():
                    percentage = count / len(data) * 100
                    f.write(f"  {emotion:15s}: {count:5d} ({percentage:5.1f}%)\n")
            
            # 分数分布统计
            f.write("\n\n分数分布统计:\n")
            f.write("-"*40 + "\n")
            
            for name, data in [('训练集', train_data), ('验证集', val_data), ('测试集', test_data)]:
                if not data:
                    continue
                
                # 过滤掉 None 的分数 (以防万一)
                scores = [d['score'] for d in data if d.get('score') is not None]
                if not scores:
                    f.write(f"\n{name}:\n  (无有效分数)\n")
                    continue

                f.write(f"\n{name}:\n")
                f.write(f"  平均分: {np.mean(scores):.2f}\n")
                f.write(f"  中位数: {np.median(scores):.2f}\n")
                f.write(f"  标准差: {np.std(scores):.2f}\n")
                f.write(f"  最小值: {min(scores):.2f}\n")
                f.write(f"  最大值: {max(scores):.2f}\n")
                
                # 风险等级分布
                risk_levels = {
                    '无风险(0-2)': sum(1 for s in scores if 0 <= s <= 2),
                    '低风险(3-4)': sum(1 for s in scores if 3 <= s <= 4),
                    '中风险(5-6)': sum(1 for s in scores if 5 <= s <= 6),
                    '中高风险(7-8)': sum(1 for s in scores if 7 <= s <= 8),
                    '高风险(9-10)': sum(1 for s in scores if 9 <= s <= 10)
                }
                
                f.write(f"\n  风险等级分布:\n")
                for level, count in risk_levels.items():
                    percentage = count / len(scores) * 100
                    f.write(f"    {level}: {count:5d} ({percentage:5.1f}%)\n")
            
            # 数据来源统计
            f.write("\n\n数据来源统计:\n")
            f.write("-"*40 + "\n")
            
            source_counts = Counter([d['source'] for d in all_data])
            for source, count in source_counts.most_common():
                percentage = count / len(all_data) * 100
                f.write(f"  {source:20s}: {count:5d} ({percentage:5.1f}%)\n")
        
        logger.info(f"统计报告已保存到: {report_path}")


# ==================== 主函数 ====================
def main():
    """主函数"""
    # 创建配置
    config = PipelineConfig(
        base_dir="./emotion_data",
        raw_dir="./emotion_data/raw",
        output_dir="./emotion_data/output",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        bert_model="bert-base-chinese",
        use_gpu=True,
        random_seed=42
    )
    
    # 创建Pipeline
    pipeline = EmotionDataPipeline(config)
    pipeline.run(
        datasets=['nlpcc2013', 'nlpcc'],  
        skip_download=False  # 如果已下载,设为True跳过下载步骤
    )
    
    logger.info("\n处理完成!")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info("\n生成的文件:")
    logger.info("  - train.json / train.jsonl / train.tsv")
    logger.info("  - val.json / val.jsonl / val.tsv")
    logger.info("  - test.json / test.jsonl / test.tsv")
    logger.info("  - statistics_report.txt")


# ==================== 命令行接口 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="情感数据处理Pipeline")
    
    parser.add_argument('--base-dir', type=str, default='./emotion_data',
                       help='基础目录路径')
    parser.add_argument('--bert-model', type=str, default='bert-base-chinese',
                       help='BERT模型名称')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--skip-download', action='store_true',
                       help='跳过下载步骤')
    parser.add_argument('--no-gpu', action='store_true',
                       help='不使用GPU')
    parser.add_argument('--datasets', nargs='+',
                       default=['nlpcc2013', 'nlpcc2020'], # 更新: 删除了你代码中没有的 nlpcc2014
                       help='要处理的数据集列表')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 创建配置
    config = PipelineConfig(
        base_dir=args.base_dir,
        raw_dir=f"{args.base_dir}/raw",
        output_dir=f"{args.base_dir}/output",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        bert_model=args.bert_model,
        use_gpu=not args.no_gpu,
        random_seed=args.seed
    )
    
    # 验证比例和
    if abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) > 0.001:
        logger.error("数据集划分比例之和必须等于1.0")
        exit(1)
    
    # 创建并运行Pipeline
    pipeline = EmotionDataPipeline(config)
    pipeline.run(datasets=args.datasets, skip_download=args.skip_download)
    
    logger.info("\n所有处理完成!")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info("\n 生成的文件:")
    logger.info("   - train.json / train.jsonl / train.tsv  (训练集)")
    logger.info("   - val.json / val.jsonl / val.tsv        (验证集)")
    logger.info("   - test.json / test.jsonl / test.tsv     (测试集)")
    logger.info("   - statistics_report.txt                 (统计报告)")
    logger.info("\n💡 使用提示:")
    logger.info("   python script.py --help                 (查看帮助)")
    logger.info("   python script.py --skip-download        (跳过下载)")
    logger.info("   python script.py --no-gpu               (不使用GPU)")
    logger.info("   python script.py --datasets nlpcc2013 nlpcc2020  (只处理指定数据集)")
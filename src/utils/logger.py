"""
日志管理模块
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """统一日志管理器"""
    
    def __init__(
        self, 
        name: str = "EmotionMonitoring",
        log_dir: Optional[str] = None,
        level: int = logging.INFO
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # 清空已有的handlers
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件输出
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """调试信息"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """一般信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """警告信息"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """错误信息"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """严重错误"""
        self.logger.critical(message)


def get_logger(name: str = "EmotionMonitoring", log_dir: Optional[str] = None) -> Logger:
    """获取日志器实例"""
    return Logger(name=name, log_dir=log_dir)


if __name__ == '__main__':
    # 测试日志
    logger = get_logger(log_dir="../../logs")
    logger.info("日志系统初始化成功")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")


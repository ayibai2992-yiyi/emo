import torch
import numpy as np
from pathlib import Path

from src.monitoring.lightweight_filter import LightweightEmotionFilter
from src.monitoring.deep_analyzer import PersonalizedDeepAnalyzer, UserBaseline
from src.monitoring.alert_system import ExtremeEmotionAlertSystem
from src.evaluation.visualization import EmotionMonitoringVisualizer
from src.utils.config import get_default_config
from src.utils.logger import get_logger


class EmotionMonitoringSystem:
    """
    完整的情绪监测系统
    整合三层架构
    """
    
    def __init__(self, config=None):
        """初始化系统"""
        self.config = config or get_default_config()
        self.logger = get_logger("EmotionMonitoring", log_dir="logs")
        
        self.logger.info("正在初始化情绪监测系统...")
        
        # 第一层：轻量级过滤器
        self.filter = LightweightEmotionFilter(
            model_name=self.config.monitoring.layer1_model,
            device=self.config.device
        )
        self.logger.info("✓ 第一层轻量级过滤器已加载")
        
        # 第二层：深度分析器
        self.analyzer = PersonalizedDeepAnalyzer(device=self.config.device)
        self._load_user_baselines()
        self.logger.info("✓ 第二层深度分析器已加载")
        
        # 第三层：预警系统
        alert_config = {
            'smtp_server': self.config.alert.smtp_server,
            'smtp_port': self.config.alert.smtp_port,
            'sender_email': self.config.alert.sender_email,
            'sender_password': self.config.alert.sender_password,
            'admin_emails': self.config.alert.admin_emails,
            'cooldown_minutes': self.config.alert.alert_cooldown_minutes
        }
        self.alert_system = ExtremeEmotionAlertSystem(alert_config)
        self.logger.info("✓ 预警系统已加载")
        
        # 可视化工具
        self.visualizer = EmotionMonitoringVisualizer(output_dir="results/figures")
        
        self.logger.info("=" * 60)
        self.logger.info("情绪监测系统初始化完成！")
        self.logger.info("=" * 60)
    
    def _load_user_baselines(self):
        """加载用户基线数据（模拟）"""
        # 模拟一些用户基线
        self.analyzer.user_baselines['user001'] = UserBaseline(
            user_id='user001',
            mean_emotion=np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]),
            std_emotion=np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02]),
            expression_style='reserved',
            history_count=50,
            last_update='2024-01-01',
            risk_threshold=7.0
        )
        
        self.analyzer.user_baselines['user002'] = UserBaseline(
            user_id='user002',
            mean_emotion=np.array([0.2, 0.15, 0.1, 0.15, 0.15, 0.1, 0.1, 0.05]),
            std_emotion=np.array([0.15, 0.15, 0.1, 0.1, 0.1, 0.08, 0.05, 0.03]),
            expression_style='expressive',
            history_count=80,
            last_update='2024-01-01',
            risk_threshold=8.5
        )
    
    async def process_message(
        self,
        text: str,
        user_id: str,
        conversation_history: list = None
    ) -> dict:
        """
        处理单条消息
        
        Args:
            text: 消息文本
            user_id: 用户ID
            conversation_history: 对话历史
            
        Returns:
            处理结果
        """
        if conversation_history is None:
            conversation_history = []
        
        self.logger.info(f"\n处理消息 - 用户: {user_id}, 文本: {text}")
        
        # 第一层：轻量级筛选
        layer1_result = self.filter.predict(text, user_id)
        
        result = {
            'user_id': user_id,
            'text': text,
            'layer1_result': layer1_result,
            'need_deep_analysis': layer1_result['need_deep_analysis'],
            'final_result': None,
            'alert_triggered': False
        }
        
        self.logger.info(f"第一层筛选 - 分数: {layer1_result['score']:.2f}, "
                        f"级别: {layer1_result['level']}, "
                        f"延迟: {layer1_result['latency_ms']:.2f}ms")
        
        # 第二层：深度分析（如果需要）
        if layer1_result['need_deep_analysis']:
            self.logger.info("触发第二层深度分析...")
            
            layer2_result = self.analyzer.analyze(
                text, user_id, conversation_history
            )
            
            result['layer2_result'] = layer2_result
            result['final_result'] = layer2_result
            
            self.logger.info(f"第二层分析 - 风险评分: {layer2_result['risk_score']:.2f}, "
                           f"风险等级: {layer2_result['risk_label']}, "
                           f"延迟: {layer2_result['latency_ms']:.2f}ms")
            
            # 第三层：极端情绪预警（如果需要）
            if layer2_result['need_alert']:
                self.logger.warning("⚠️ 触发极端情绪预警！")
                
                alert_record = await self.alert_system.trigger_alert(
                    layer2_result,
                    conversation_history
                )
                
                result['alert_triggered'] = True
                result['alert_record'] = alert_record
                
                self.logger.warning(f"预警记录: {alert_record.get('alert_id', 'N/A')}")
        else:
            # 不需要深度分析
            result['final_result'] = {
                'risk_score': layer1_result['score'],
                'risk_level': 'green' if layer1_result['score'] < 5 else 'blue',
                'risk_label': '正常' if layer1_result['score'] < 5 else '关注'
            }
            self.logger.info("风险较低，跳过深度分析")
        
        # 记录结果用于可视化
        vis_result = {
            'user_id': user_id,
            'text': text,
            'layer1_latency': layer1_result['latency_ms'],
            'need_deep_analysis': result['need_deep_analysis'],
            'risk_level': result['final_result']['risk_level'],
            'risk_score': result['final_result']['risk_score']
        }
        
        if result.get('layer2_result'):
            vis_result['layer2_latency'] = result['layer2_result']['latency_ms']
        
        self.visualizer.add_result(vis_result)
        
        return result
    
    def generate_report(self):
        """生成系统性能报告"""
        self.logger.info("\n生成系统性能报告...")
        
        # 生成可视化图表
        self.visualizer.plot_latency_distribution()
        
        # 生成文本报告
        report = self.visualizer.generate_summary_report()
        print(report)
        
        self.logger.info("报告生成完成！")


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("情绪监测系统 - 完整演示")
    print("=" * 60 + "\n")
    
    # 初始化系统
    system = EmotionMonitoringSystem()
    
    # 测试案例
    test_cases = [
        {
            'user_id': 'user001',
            'text': '今天天气不错，心情还可以',
            'history': []
        },
        {
            'user_id': 'user001',
            'text': '作业有点多，感觉有点累',
            'history': ['今天天气不错']
        },
        {
            'user_id': 'user001',
            'text': '压力好大，真的很难受',
            'history': ['今天天气不错', '作业有点多，感觉有点累']
        },
        {
            'user_id': 'user002',
            'text': '今天上课很无聊',
            'history': []
        },
        {
            'user_id': 'user002',
            'text': '我真的撑不住了，太痛苦了',
            'history': ['今天上课很无聊', '不想学习了']
        },
        {
            'user_id': 'user002',
            'text': '活不下去了，想自杀',
            'history': ['今天上课很无聊', '不想学习了', '我真的撑不住了']
        }
    ]
    
    print("\n开始处理测试消息...\n")
    print("=" * 60)
    
    # 处理每条消息
    for i, case in enumerate(test_cases, 1):
        print(f"\n【案例 {i}】")
        print(f"用户: {case['user_id']}")
        print(f"消息: {case['text']}")
        if case['history']:
            print(f"历史: {case['history'][-2:]}")  # 显示最近2条
        
        result = await system.process_message(
            text=case['text'],
            user_id=case['user_id'],
            conversation_history=case['history']
        )
        
        # 显示结果
        final_result = result['final_result']
        print(f"\n处理结果:")
        print(f"  风险评分: {final_result['risk_score']:.2f}")
        print(f"  风险等级: {final_result['risk_level']} - {final_result['risk_label']}")
        print(f"  是否预警: {'是 ⚠️' if result['alert_triggered'] else '否'}")
        
        print("-" * 60)
    
    # 生成报告
    print("\n" + "=" * 60)
    print("生成系统性能报告...")
    print("=" * 60)
    system.generate_report()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())


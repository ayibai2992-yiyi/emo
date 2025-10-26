"""
可视化工具模块
用于测试阶段展示系统性能
问题
1. 绘制延迟分布图时，如果第二层没有结果，则不绘制第二层延迟
2. 混淆矩阵的标签需要支持中文显示
3. 绘制混淆矩阵时，如果标签数量小于5，则不绘制
"""

import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import platform

class EmotionMonitoringVisualizer:
    """
    情绪监测可视化工具
    """
    
    def __init__(self, output_dir: str = "results/figures"):
        """
        初始化
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        # 更好的跨平台方案
        if platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        elif platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 设置样式
        sns.set_style("whitegrid")
        
        self.results = []
    
    def add_result(self, result: Dict):
        """添加一条监测结果"""
        self.results.append(result)
    
    def plot_latency_distribution(self, save_path: Optional[str] = None):
        """
        绘制延迟分布图
        
        Args:
            save_path: 保存路径
        """
        if not self.results:
            print("没有结果数据")
            return
        #第一层
        layer1_latencies = [r.get('layer1_latency', 0) for r in self.results]
        #第二层
        layer2_results = [r for r in self.results if r.get('need_deep_analysis')]
        layer2_latencies = [r.get('layer2_latency', 0) for r in layer2_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 第一层延迟
        axes[0].hist(layer1_latencies, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(layer1_latencies), color='red', linestyle='--', 
                       label=f'均值: {np.mean(layer1_latencies):.1f}ms')
        axes[0].axvline(np.percentile(layer1_latencies, 95), color='orange', linestyle='--',
                       label=f'P95: {np.percentile(layer1_latencies, 95):.1f}ms')
        axes[0].set_xlabel('延迟 (ms)', fontsize=12)
        axes[0].set_ylabel('频数', fontsize=12)
        axes[0].set_title('第一层轻量级筛选延迟分布', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 第二层延迟
        if layer2_latencies:
            axes[1].hist(layer2_latencies, bins=20, color='salmon', edgecolor='black', alpha=0.7)
            axes[1].axvline(np.mean(layer2_latencies), color='red', linestyle='--',
                           label=f'均值: {np.mean(layer2_latencies):.1f}ms')
            axes[1].set_xlabel('延迟 (ms)', fontsize=12)
            axes[1].set_ylabel('频数', fontsize=12)
            axes[1].set_title('第二层深度分析延迟分布', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, '无深度分析数据', ha='center', va='center', fontsize=14)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'latency_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"延迟分布图已保存: {save_path}")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str],
        save_path: Optional[str] = None
    ):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签名称列表
            save_path: 保存路径
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': '样本数'}
        )
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.title('风险等级分类混淆矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"混淆矩阵已保存: {save_path}")
    
    def plot_emotion_radar(
        self,
        emotion_vector: np.ndarray,
        emotion_labels: List[str],
        title: str = "情绪分布雷达图",
        save_path: Optional[str] = None
    ):
        """
        绘制情绪雷达图
        
        Args:
            emotion_vector: 情绪向量
            emotion_labels: 情绪标签
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(emotion_labels), endpoint=False).tolist()
        values = emotion_vector.tolist()
        
        # 闭合
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='red', label='情绪强度')
        ax.fill(angles, values, alpha=0.25, color='red')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(emotion_labels)
        ax.set_ylim(0, 1)
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
        ax.grid(True)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'emotion_radar.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 情绪雷达图已保存: {save_path}")
    
    def plot_few_shot_performance(
        self,
        shot_sizes: List[int],
        f1_scores: List[float],
        save_path: Optional[str] = None
    ):
        """
        绘制少样本性能曲线
        
        Args:
            shot_sizes: Shot数列表
            f1_scores: F1分数列表
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))
        plt.plot(shot_sizes, f1_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
        plt.xlabel('支持集大小 (k-shot)', fontsize=12)
        plt.ylabel('Macro F1 分数', fontsize=12)
        plt.title('少样本学习性能曲线', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # 标注数值
        for x, y in zip(shot_sizes, f1_scores):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'few_shot_performance.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 少样本性能曲线已保存: {save_path}")
    
    def generate_summary_report(self, save_path: Optional[str] = None) -> str:
        """
        生成汇总报告
        
        Args:
            save_path: 保存路径
            
        Returns:
            报告文本
        """
        total = len(self.results)
        if total == 0:
            return "无测试数据"
        
        layer1_latencies = [r.get('layer1_latency', 0) for r in self.results]
        layer2_count = sum(1 for r in self.results if r.get('need_deep_analysis'))
        layer2_results = [r for r in self.results if r.get('need_deep_analysis')]
        layer2_latencies = [r.get('layer2_latency', 0) for r in layer2_results]
        
        alerts = sum(1 for r in self.results if r.get('risk_level') == 'red')
        
        report = f"""
╔══════════════════════════════════════════════════════╗
║          情绪监测系统性能汇总报告                      ║
╚══════════════════════════════════════════════════════╝

【测试统计】
  总对话数: {total}
  触发深度分析: {layer2_count} ({layer2_count/total*100:.1f}%)
  触发极端预警: {alerts} ({alerts/total*100:.1f}%)

【延迟性能】
  第一层 (轻量级筛选):
    - 平均延迟: {np.mean(layer1_latencies):.2f} ms
    - P50延迟: {np.percentile(layer1_latencies, 50):.2f} ms
    - P95延迟: {np.percentile(layer1_latencies, 95):.2f} ms
    - P99延迟: {np.percentile(layer1_latencies, 99):.2f} ms
    - 最大延迟: {np.max(layer1_latencies):.2f} ms
  
  第二层 (深度分析):
    - 平均延迟: {np.mean(layer2_latencies) if layer2_latencies else 0:.2f} ms
    - P95延迟: {np.percentile(layer2_latencies, 95) if layer2_latencies else 0:.2f} ms

【风险分布】
  正常 (Green): {sum(1 for r in self.results if r.get('risk_level')=='green')}
  关注 (Blue): {sum(1 for r in self.results if r.get('risk_level')=='blue')}
  警告 (Orange): {sum(1 for r in self.results if r.get('risk_level')=='orange')}
  危机 (Red): {sum(1 for r in self.results if r.get('risk_level')=='red')}

【效率评估】
  ✅ P95延迟 < 100ms: {'是' if np.percentile(layer1_latencies, 95) < 100 else '否'}
  ✅ 深度分析比例 < 15%: {'是' if layer2_count/total < 0.15 else '否'}

报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        if save_path is None:
            save_path = self.output_dir.parent / 'summary_report.txt'
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ 汇总报告已保存: {save_path}")
        
        return report


# 使用示例
if __name__ == '__main__':
    print("测试可视化工具...")
    
    # 创建可视化器
    visualizer = EmotionMonitoringVisualizer(output_dir="../../results/figures")
    
    # 模拟测试结果
    np.random.seed(42)
    for i in range(100):
        result = {
            'user_id': f'user{i % 10}',
            'layer1_latency': np.random.gamma(2, 15),
            'need_deep_analysis': np.random.rand() < 0.1,
            'risk_level': np.random.choice(['green', 'blue', 'orange', 'red'], p=[0.7, 0.15, 0.1, 0.05])
        }
        
        if result['need_deep_analysis']:
            result['layer2_latency'] = np.random.gamma(5, 60)
        
        visualizer.add_result(result)
    
    # 绘制延迟分布
    visualizer.plot_latency_distribution()
    
    # 绘制混淆矩阵
    y_true = np.random.choice([0, 1, 2, 3], 50)
    y_pred = np.random.choice([0, 1, 2, 3], 50)
    visualizer.plot_confusion_matrix(
        y_true, y_pred,
        labels=['正常', '关注', '警告', '危机']
    )
    
    # 绘制情绪雷达图
    emotion_vector = np.array([0.1, 0.05, 0.05, 0.3, 0.1, 0.2, 0.05, 0.15])
    emotion_labels = ['中性', '高兴', '惊讶', '悲伤', '愤怒', '恐惧', '厌恶', '绝望']
    visualizer.plot_emotion_radar(emotion_vector, emotion_labels)
    
    # 绘制少样本性能曲线
    shot_sizes = [1, 3, 5, 10, 20, 50]
    f1_scores = [0.45, 0.58, 0.65, 0.72, 0.78, 0.82]
    visualizer.plot_few_shot_performance(shot_sizes, f1_scores)
    
    # 生成汇总报告
    report = visualizer.generate_summary_report()
    print(report)
    
    print("\n测试完成!")


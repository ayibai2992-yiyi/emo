"""
极端情绪预警系统
- 立即邮件通知
- 记录详细日志
- 生成可视化报告
"""

import smtplib
import asyncio
import json
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class ExtremeEmotionAlertSystem:
    """
    极端情绪预警系统
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化预警系统
        
        Args:
            config: 配置字典，包含SMTP设置等
        """
        if config is None:
            config = {}
        
        self.smtp_server = config.get('smtp_server', 'smtp.example.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.sender_email = config.get('sender_email', 'alert@example.com')
        self.sender_password = config.get('sender_password', '')
        
        # 管理员和教师邮箱映射
        self.teacher_mapping = config.get('teacher_mapping', {})
        self.admin_emails = config.get('admin_emails', [])
        
        # 预警记录
        self.alert_history: List[Dict] = []
        
        # 冷却时间 (分钟) - 避免同一用户频繁预警
        self.cooldown_minutes = config.get('cooldown_minutes', 30)
        self.last_alert_time: Dict[str, datetime] = {}
    
    def check_cooldown(self, user_id: str) -> bool:
        """
        检查是否在冷却时间内
        
        Args:
            user_id: 用户ID
            
        Returns:
            True表示可以发送预警，False表示在冷却期
        """
        if user_id not in self.last_alert_time:
            return True
        
        elapsed = (datetime.now() - self.last_alert_time[user_id]).total_seconds() / 60
        return elapsed >= self.cooldown_minutes
    
    def generate_alert_id(self) -> str:
        """
        生成唯一预警ID
        
        Returns:
            预警ID
        """
        return f"ALERT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    
    async def trigger_alert(
        self, 
        analysis_result: Dict,
        conversation_history: Optional[List[str]] = None
    ) -> Dict:
        """
        触发极端情绪预警
        
        Args:
            analysis_result: 深度分析结果
            conversation_history: 对话历史
            
        Returns:
            预警记录
        """
        user_id = analysis_result['user_id']
        text = analysis_result['text']
        risk_score = analysis_result['risk_score']
        
        # 检查冷却时间
        if not self.check_cooldown(user_id):
            print(f"⏰ 用户 {user_id} 在冷却期内，跳过预警")
            return {'status': 'skipped', 'reason': 'cooldown'}
        
        if conversation_history is None:
            conversation_history = []
        
        # 1. 创建预警记录
        alert_record = {
            'alert_id': self.generate_alert_id(),
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'risk_score': risk_score,
            'risk_level': analysis_result['risk_level'],
            'risk_label': analysis_result['risk_label'],
            'trigger_text': text,
            'conversation_context': conversation_history[-10:],  # 最近10轮
            'emotion_vector': analysis_result['emotion_vector'],
            'emotion_distribution': analysis_result.get('emotion_distribution', {}),
            'status': 'pending',  # pending / handled / false_alarm
            'handler': None,
            'notes': ''
        }
        
        # 2. 保存到历史记录
        self.alert_history.append(alert_record)
        self.last_alert_time[user_id] = datetime.now()
        
        # 3. 保存到文件
        self.save_alert_to_file(alert_record)
        
        # 4. 发送邮件通知 (异步)
        if self.admin_emails or user_id in self.teacher_mapping:
            try:
                await self.send_email_alert(alert_record)
            except Exception as e:
                print(f"❌ 邮件发送失败: {e}")
                self.log_email_error(alert_record, str(e))
        
        # 5. 生成可视化报告
        try:
            report_path = self.generate_visual_report(alert_record)
            alert_record['report_path'] = report_path
        except Exception as e:
            print(f"⚠️ 报告生成失败: {e}")
            alert_record['report_path'] = None
        
        print(f"⚠️ 极端情绪预警已触发: User={user_id}, Score={risk_score:.2f}, ID={alert_record['alert_id']}")
        
        return alert_record
    
    async def send_email_alert(self, alert_record: Dict):
        """
        发送邮件通知
        
        Args:
            alert_record: 预警记录
        """
        user_id = alert_record['user_id']
        
        # 获取负责教师
        teacher_emails = self.teacher_mapping.get(user_id, [])
        recipients = teacher_emails + self.admin_emails
        
        if not recipients:
            print(f"警告: 用户 {user_id} 没有配置负责教师邮箱")
            return
        
        # 构造邮件内容
        subject = f"【紧急】学生心理健康预警 - {user_id}"
        
        html_body = self._generate_email_html(alert_record)
        
        # 发送邮件
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(recipients)
        
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))
        
        # SMTP发送 (在线程池中执行以避免阻塞)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_smtp, msg, recipients)
        
        print(f"✅ 预警邮件已发送至: {', '.join(recipients)}")
    
    def _generate_email_html(self, alert_record: Dict) -> str:
        """
        生成邮件HTML内容
        
        Args:
            alert_record: 预警记录
            
        Returns:
            HTML字符串
        """
        emotion_dist = alert_record.get('emotion_distribution', {})
        emotion_str = '<br>'.join([f"{k}: {v:.2%}" for k, v in emotion_dist.items() if v > 0.1])
        
        html_body = f"""
        <html>
        <head>
            <style>
                .alert-box {{
                    background-color: #fff3cd;
                    border: 2px solid #ff4444;
                    padding: 20px;
                    border-radius: 5px;
                    font-family: Arial, sans-serif;
                }}
                .risk-high {{ color: #cc0000; font-weight: bold; }}
                .info-table {{ border-collapse: collapse; width: 100%; }}
                .info-table td {{ padding: 8px; border: 1px solid #ddd; }}
                .info-table td:first-child {{ background-color: #f5f5f5; font-weight: bold; width: 150px; }}
            </style>
        </head>
        <body>
            <div class="alert-box">
                <h2>🚨 学生心理健康紧急预警</h2>
                
                <table class="info-table">
                    <tr>
                        <td>预警ID</td>
                        <td>{alert_record['alert_id']}</td>
                    </tr>
                    <tr>
                        <td>学生ID</td>
                        <td>{alert_record['user_id']}</td>
                    </tr>
                    <tr>
                        <td>风险等级</td>
                        <td class="risk-high">{alert_record['risk_level'].upper()} - {alert_record['risk_label']}</td>
                    </tr>
                    <tr>
                        <td>风险评分</td>
                        <td>{alert_record['risk_score']:.2f} / 10.0</td>
                    </tr>
                    <tr>
                        <td>触发时间</td>
                        <td>{alert_record['timestamp']}</td>
                    </tr>
                    <tr>
                        <td>触发内容</td>
                        <td style="color: #cc0000;">{alert_record['trigger_text']}</td>
                    </tr>
                </table>
                
                <h3>情绪分布:</h3>
                <div style="background: #f5f5f5; padding: 10px; border-radius: 3px;">
                    {emotion_str or '无'}
                </div>
                
                <h3>近期对话记录 (最近5条):</h3>
                <div style="background: #f5f5f5; padding: 10px; border-radius: 3px;">
                    {'<br>'.join(['- ' + msg for msg in alert_record['conversation_context'][-5:]])}
                </div>
                
                <h3>⚠️ 建议行动:</h3>
                <ol>
                    <li><strong>立即</strong>联系学生，了解情况</li>
                    <li>评估是否需要心理咨询师介入</li>
                    <li>如有自杀风险，立即联系家长和校方</li>
                    <li>在系统中标记处理状态</li>
                </ol>
                
                <hr>
                <p style="font-size: 12px; color: #666;">
                    本邮件由学生心理健康AI监测系统自动发送<br>
                    预警ID: {alert_record['alert_id']}<br>
                    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                    如需帮助，请联系技术支持
                </p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _send_smtp(self, msg: MIMEMultipart, recipients: List[str]):
        """
        同步SMTP发送
        
        Args:
            msg: 邮件对象
            recipients: 收件人列表
        """
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
        except Exception as e:
            print(f"SMTP发送错误: {e}")
            # 在实际环境中，这里应该记录到日志系统
    
    def generate_visual_report(self, alert_record: Dict) -> str:
        """
        生成可视化报告 (HTML)
        
        Args:
            alert_record: 预警记录
            
        Returns:
            报告文件路径
        """
        import matplotlib
        matplotlib.use('Agg')  # 无GUI后端
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制情绪雷达图
        emotion_labels = ['中性', '高兴', '惊讶', '悲伤', '愤怒', '恐惧', '厌恶', '绝望']
        emotion_values = alert_record['emotion_vector']
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        import numpy as np
        angles = np.linspace(0, 2 * np.pi, len(emotion_labels), endpoint=False).tolist()
        emotion_values_plot = emotion_values + emotion_values[:1]
        angles_plot = angles + angles[:1]
        
        ax.plot(angles_plot, emotion_values_plot, 'o-', linewidth=2, color='red')
        ax.fill(angles_plot, emotion_values_plot, alpha=0.25, color='red')
        ax.set_xticks(angles)
        ax.set_xticklabels(emotion_labels)
        ax.set_ylim(0, 1)
        ax.set_title('情绪分布雷达图', pad=20, fontsize=14, fontweight='bold')
        ax.grid(True)
        
        # 转为base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # 生成HTML报告
        report_html = f"""
        <html>
        <head>
            <title>预警报告 - {alert_record['alert_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #cc0000; }}
                .info {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                pre {{ background: #f9f9f9; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>🚨 极端情绪预警详细报告</h1>
            
            <div class="info">
                <h2>基本信息</h2>
                <p><strong>预警ID:</strong> {alert_record['alert_id']}</p>
                <p><strong>用户ID:</strong> {alert_record['user_id']}</p>
                <p><strong>风险等级:</strong> <span style="color: #cc0000;">{alert_record['risk_label']}</span></p>
                <p><strong>风险评分:</strong> {alert_record['risk_score']:.2f} / 10.0</p>
                <p><strong>触发时间:</strong> {alert_record['timestamp']}</p>
                <p><strong>触发内容:</strong> "{alert_record['trigger_text']}"</p>
            </div>
            
            <h2>情绪分布可视化</h2>
            <img src="data:image/png;base64,{image_base64}" style="max-width: 800px;" />
            
            <h2>详细数据</h2>
            <pre>{json.dumps(alert_record, indent=2, ensure_ascii=False)}</pre>
        </body>
        </html>
        """
        
        # 保存到文件
        report_dir = Path("results/alerts")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"alert_{alert_record['alert_id']}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        return str(report_path)
    
    def save_alert_to_file(self, alert_record: Dict):
        """
        保存预警记录到JSON文件
        
        Args:
            alert_record: 预警记录
        """
        alert_dir = Path("results/alerts/json")
        alert_dir.mkdir(parents=True, exist_ok=True)
        
        alert_path = alert_dir / f"{alert_record['alert_id']}.json"
        with open(alert_path, 'w', encoding='utf-8') as f:
            json.dump(alert_record, f, ensure_ascii=False, indent=2)
    
    def log_email_error(self, alert_record: Dict, error: str):
        """
        记录邮件发送失败
        
        Args:
            alert_record: 预警记录
            error: 错误信息
        """
        error_log = {
            'alert_id': alert_record['alert_id'],
            'user_id': alert_record['user_id'],
            'timestamp': datetime.now().isoformat(),
            'error': error
        }
        
        error_dir = Path("results/alerts/errors")
        error_dir.mkdir(parents=True, exist_ok=True)
        
        error_path = error_dir / f"email_error_{alert_record['alert_id']}.json"
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, ensure_ascii=False, indent=2)
    
    def get_alert_history(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        获取预警历史
        
        Args:
            user_id: 用户ID（可选）
            
        Returns:
            预警记录列表
        """
        if user_id is None:
            return self.alert_history
        else:
            return [alert for alert in self.alert_history if alert['user_id'] == user_id]


# 使用示例
if __name__ == '__main__':
    import numpy as np
    
    print("正在初始化预警系统...")
    
    # 配置
    config = {
        'admin_emails': ['admin@example.com'],
        'teacher_mapping': {
            'user001': ['teacher1@example.com'],
        },
        'cooldown_minutes': 30
    }
    
    alert_system = ExtremeEmotionAlertSystem(config)
    
    # 模拟预警
    analysis_result = {
        'user_id': 'user001',
        'text': '我真的不想活了',
        'risk_score': 9.5,
        'risk_level': 'red',
        'risk_label': '危机',
        'emotion_vector': [0.1, 0.0, 0.0, 0.3, 0.1, 0.2, 0.0, 0.3],
        'emotion_distribution': {
            '中性': 0.1,
            '悲伤': 0.3,
            '恐惧': 0.2,
            '绝望': 0.3
        }
    }
    
    conversation_history = [
        "最近学习压力好大",
        "睡不好觉",
        "感觉很痛苦",
        "没人理解我",
        "我真的不想活了"
    ]
    
    print("\n" + "="*60)
    print("触发极端情绪预警...")
    print("="*60)
    
    # 同步方式触发预警
    async def test_alert():
        alert_record = await alert_system.trigger_alert(
            analysis_result,
            conversation_history
        )
        print(f"\n预警记录已创建:")
        print(f"  预警ID: {alert_record.get('alert_id', 'N/A')}")
        print(f"  用户ID: {alert_record.get('user_id', 'N/A')}")
        print(f"  风险等级: {alert_record.get('risk_label', 'N/A')}")
        if alert_record.get('report_path'):
            print(f"  报告路径: {alert_record['report_path']}")
    
    # 运行异步测试
    asyncio.run(test_alert())
    
    print("\n" + "="*60)
    print("测试完成!")


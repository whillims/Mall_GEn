import pandas as pd
import numpy as np
import json
import base64
from typing import Dict, List, Optional
from datetime import datetime
import os

class ReportGenerator:
    def __init__(self, metrics: Dict, symbol: str, model_type: str,
                 start_date: str, end_date: str):
        self.metrics = metrics
        self.symbol = symbol
        self.model_type = model_type
        self.start_date = start_date
        self.end_date = end_date
        self.report_dir = 'reports'
        os.makedirs(self.report_dir, exist_ok=True)

    def _generate_equity_chart_js(self) -> str:
        equity = self.metrics.get('equity_curve', [])
        dates = self.metrics.get('dates', [])
        if not equity or not dates:
            return ""

        dates_str = [str(d)[:10] for d in dates]
        equity_str = [f"{e:.2f}" for e in equity]

        return f"""
        var equityCtx = document.getElementById('equityChart').getContext('2d');
        new Chart(equityCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(dates_str)},
                datasets: [{{
                    label: '权益曲线',
                    data: {json.dumps(equity_str)},
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ display: false }},
                    y: {{
                        grid: {{ color: 'rgba(0,0,0,0.05)' }},
                        ticks: {{ color: '#666' }}
                    }}
                }}
            }}
        }});
        """

    def _generate_drawdown_chart_js(self) -> str:
        equity = self.metrics.get('equity_curve', [])
        if not equity:
            return ""

        equity_series = pd.Series(equity)
        cummax = equity_series.cummax()
        drawdown = ((equity_series - cummax) / cummax * 100).tolist()
        dates_str = [str(d)[:10] for d in self.metrics.get('dates', [])]

        return f"""
        var ddCtx = document.getElementById('drawdownChart').getContext('2d');
        new Chart(ddCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(dates_str)},
                datasets: [{{
                    label: '回撤',
                    data: {json.dumps([round(d, 2) for d in drawdown])},
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ display: false }},
                    y: {{
                        grid: {{ color: 'rgba(0,0,0,0.05)' }},
                        ticks: {{ color: '#666', callback: function(v) {{ return v + '%'; }} }}
                    }}
                }}
            }}
        }});
        """

    def _generate_trade_table(self) -> str:
        trades = self.metrics.get('trades')
        if trades is None or trades.empty:
            return '<p class="no-data">无交易记录</p>'

        rows = ""
        for _, trade in trades.iterrows():
            pnl_class = "positive" if trade['pnl'] > 0 else "negative"
            direction_label = "买入" if trade['direction'] == 1 else "卖出"
            rows += f"""
            <tr>
                <td>{str(trade['entry_date'])[:10]}</td>
                <td>{str(trade['exit_date'])[:10]}</td>
                <td><span class="direction {'buy' if trade['direction']==1 else 'sell'}">{direction_label}</span></td>
                <td>{trade['entry_price']:.2f}</td>
                <td>{trade['exit_price']:.2f}</td>
                <td class="{pnl_class}">{trade['pnl']:+.2f}</td>
                <td class="{pnl_class}">{trade['pnl_pct']*100:+.2f}%</td>
            </tr>
            """

        return f"""
        <table class="trade-table">
            <thead>
                <tr>
                    <th>入场日期</th>
                    <th>出场日期</th>
                    <th>方向</th>
                    <th>入场价</th>
                    <th>出场价</th>
                    <th>盈亏</th>
                    <th>盈亏%</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """

    def _get_metric_card(self, title: str, value: str, subtitle: str = "",
                        color: str = "#3498db") -> str:
        return f"""
        <div class="metric-card" style="border-left-color: {color}">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color: {color}">{value}</div>
            {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """

    def generate_report(self) -> str:
        total_return = self.metrics.get('total_return_pct', 0)
        return_color = '#2ecc71' if total_return >= 0 else '#e74c3c'

        win_rate = self.metrics.get('win_rate_pct', 0)
        win_rate_color = '#2ecc71' if win_rate >= 50 else '#e74c3c'

        sharpe = self.metrics.get('sharpe_ratio', 0)
        sharpe_color = '#2ecc71' if sharpe > 0 else '#e74c3c'

        max_dd = self.metrics.get('max_drawdown_pct', 0)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告 - {self.symbol}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #34495e;
            --accent: #3498db;
            --success: #2ecc71;
            --danger: #e74c3c;
            --warning: #f39c12;
            --bg: #f5f7fa;
            --card-bg: #ffffff;
            --text: #2c3e50;
            --text-light: #7f8c8d;
            --border: #e1e8ed;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
                        'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 40px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .header .subtitle {{
            opacity: 0.85;
            font-size: 1rem;
        }}
        .header .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-top: 12px;
            margin-right: 8px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 4px solid var(--accent);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 0.85rem;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent);
        }}
        .metric-subtitle {{
            font-size: 0.8rem;
            color: var(--text-light);
            margin-top: 4px;
        }}
        .chart-section {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 24px;
        }}
        .chart-section h2 {{
            font-size: 1.2rem;
            margin-bottom: 16px;
            color: var(--primary);
        }}
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        .charts-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 24px;
        }}
        .trade-section {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        .trade-section h2 {{
            font-size: 1.2rem;
            margin-bottom: 16px;
            color: var(--primary);
        }}
        .trade-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        .trade-table th {{
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid var(--border);
            color: var(--text-light);
            font-weight: 600;
            white-space: nowrap;
        }}
        .trade-table td {{
            padding: 12px;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        .trade-table tr:hover {{
            background: var(--bg);
        }}
        .positive {{ color: var(--success); font-weight: 600; }}
        .negative {{ color: var(--danger); font-weight: 600; }}
        .direction {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .direction.buy {{
            background: rgba(46, 204, 113, 0.15);
            color: var(--success);
        }}
        .direction.sell {{
            background: rgba(231, 76, 60, 0.15);
            color: var(--danger);
        }}
        .footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-light);
            font-size: 0.85rem;
        }}
        .no-data {{
            text-align: center;
            padding: 40px;
            color: var(--text-light);
        }}
        @media (max-width: 768px) {{
            .charts-row {{ grid-template-columns: 1fr; }}
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .header {{ padding: 24px; }}
            .header h1 {{ font-size: 1.5rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>回测报告</h1>
            <div class="subtitle">{self.symbol} | {self.model_type} | {self.start_date} ~ {self.end_date}</div>
            <div>
                <span class="badge">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
            </div>
        </div>

        <div class="metrics-grid">
            {self._get_metric_card('总收益率', f"{total_return:+.2f}%", color=return_color)}
            {self._get_metric_card('最终资金', f"{self.metrics.get('final_capital', 0):,.2f}", subtitle=f"初始: {self.metrics.get('initial_capital', 100000):,.0f}")}
            {self._get_metric_card('胜率', f"{win_rate:.1f}%", color=win_rate_color)}
            {self._get_metric_card('总交易次数', str(self.metrics.get('total_trades', 0)),
                subtitle=f"盈利: {self.metrics.get('winning_trades', 0)} | 亏损: {self.metrics.get('losing_trades', 0)}")}
            {self._get_metric_card('夏普比率', f"{sharpe:.2f}", color=sharpe_color)}
            {self._get_metric_card('最大回撤', f"{max_dd:.2f}%", color='#e74c3c')}
            {self._get_metric_card('盈亏比', f"{self.metrics.get('profit_factor', 0):.2f}", color='#9b59b6')}
            {self._get_metric_card('平均盈利', f"{self.metrics.get('avg_win', 0):+.2f}", color='#2ecc71')}
            {self._get_metric_card('平均亏损', f"{self.metrics.get('avg_loss', 0):+.2f}", color='#e74c3c')}
        </div>

        <div class="charts-row">
            <div class="chart-section">
                <h2>权益曲线</h2>
                <div class="chart-container">
                    <canvas id="equityChart"></canvas>
                </div>
            </div>
            <div class="chart-section">
                <h2>回撤曲线</h2>
                <div class="chart-container">
                    <canvas id="drawdownChart"></canvas>
                </div>
            </div>
        </div>

        <div class="trade-section">
            <h2>交易明细</h2>
            {self._generate_trade_table()}
        </div>

        <div class="footer">
            由 ML Quant Strategy Backtest System 生成
        </div>
    </div>

    <script>
        {self._generate_equity_chart_js()}
        {self._generate_drawdown_chart_js()}
    </script>
</body>
</html>"""

        return html

    def save_report(self, filename: str = None) -> str:
        if filename is None:
            filename = f"{self.symbol}_report.html"

        filepath = os.path.join(self.report_dir, filename)
        html = self.generate_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"报告已保存: {filepath}")
        return filepath

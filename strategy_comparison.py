import pandas as pd
import numpy as np
import json
import argparse
import os
from typing import Dict, List
from data_loader import DataLoader
from backtest_engine import BacktestEngine
from classic_strategies import STRATEGIES
from report_generator import ReportGenerator
from visualization import Visualizer

class StrategyComparator:
    def __init__(self, symbol: str, start_date: str, end_date: str,
                 initial_capital: float = 100000.0):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.results: Dict[str, Dict] = {}

        data_loader = DataLoader(symbol, start_date, end_date)
        self.data = data_loader.load_data()

        # 使用80%数据作为回测区间
        split_idx = int(len(self.data) * 0.2)
        self.backtest_data = self.data.iloc[split_idx:].copy()

        print(f"\n回测区间: {self.backtest_data.index[0]} ~ {self.backtest_data.index[-1]}")
        print(f"回测天数: {len(self.backtest_data)}")

    def run_strategy(self, strategy_name: str, **kwargs) -> Dict:
        if strategy_name not in STRATEGIES:
            raise ValueError(f"未知策略: {strategy_name}")

        strategy = STRATEGIES[strategy_name](**kwargs)
        print(f"\n{'='*50}")
        print(f"运行策略: {strategy.name}")
        print(f"{'='*50}")

        predictions = strategy.generate_signals(self.backtest_data)

        buy_count = np.sum(predictions == 1)
        sell_count = np.sum(predictions == -1)
        hold_count = np.sum(predictions == 0)
        print(f"信号分布: 买入={buy_count}, 卖出={sell_count}, 持有={hold_count}")

        backtest = BacktestEngine(
            initial_capital=self.initial_capital,
            position_size=0.3,
            stop_loss=0.05,
            take_profit=0.10
        )

        metrics = backtest.run_backtest(self.backtest_data, predictions)
        metrics['equity_curve'] = backtest.equity_curve
        metrics['dates'] = backtest.dates
        metrics['initial_capital'] = self.initial_capital
        metrics['strategy_name'] = strategy.name

        self.results[strategy_name] = metrics
        return metrics

    def run_all_strategies(self):
        print("\n" + "="*60)
        print("开始策略对比回测")
        print("="*60)

        configs = {
            'dual_ma': {'short_window': 20, 'long_window': 50},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'rsi': {'period': 14, 'oversold': 30, 'overbought': 70},
            'bollinger': {'period': 20, 'std_dev': 2.0},
            'momentum': {'lookback': 20, 'threshold': 0.05},
            'mean_reversion': {'period': 20, 'zscore_threshold': 2.0},
            'turtle': {'entry_period': 20, 'exit_period': 10},
            'atr_breakout': {'atr_period': 14, 'multiplier': 2.0},
            'vwap': {'period': 20},
            'golden_cross': {'short': 50, 'long': 200},
        }

        for name, config in configs.items():
            try:
                self.run_strategy(name, **config)
            except Exception as e:
                print(f"策略 {name} 运行失败: {e}")

    def print_comparison(self):
        print("\n" + "="*80)
        print("策略对比结果")
        print("="*80)

        headers = ['策略名称', '总收益率%', '胜率%', '交易次数', '夏普比率', '最大回撤%', '盈亏比']
        print(f"{'策略名称':<20} {'总收益率%':>10} {'胜率%':>8} {'交易次数':>8} {'夏普比率':>10} {'最大回撤%':>10} {'盈亏比':>8}")
        print("-"*80)

        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get('total_return', -999),
            reverse=True
        )

        for name, metrics in sorted_results:
            if 'error' in metrics:
                continue
            print(f"{metrics['strategy_name']:<20} "
                  f"{metrics.get('total_return_pct', 0):>+10.2f} "
                  f"{metrics.get('win_rate_pct', 0):>8.1f} "
                  f"{metrics.get('total_trades', 0):>8} "
                  f"{metrics.get('sharpe_ratio', 0):>10.2f} "
                  f"{metrics.get('max_drawdown_pct', 0):>10.2f} "
                  f"{metrics.get('profit_factor', 0):>8.2f}")

    def generate_comparison_report(self):
        os.makedirs('reports', exist_ok=True)

        html = self._generate_comparison_html()
        filepath = f'reports/{self.symbol}_strategy_comparison.html'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n对比报告已保存: {filepath}")
        return filepath

    def _generate_comparison_html(self) -> str:
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get('total_return', -999),
            reverse=True
        )

        table_rows = ""
        for i, (name, metrics) in enumerate(sorted_results):
            if 'error' in metrics:
                continue

            return_class = 'positive' if metrics.get('total_return_pct', 0) >= 0 else 'negative'
            win_class = 'positive' if metrics.get('win_rate_pct', 0) >= 50 else 'negative'

            table_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td><strong>{metrics['strategy_name']}</strong></td>
                <td class="{return_class}">{metrics.get('total_return_pct', 0):+.2f}%</td>
                <td class="{win_class}">{metrics.get('win_rate_pct', 0):.1f}%</td>
                <td>{metrics.get('total_trades', 0)}</td>
                <td>{metrics.get('winning_trades', 0)}</td>
                <td>{metrics.get('losing_trades', 0)}</td>
                <td>{metrics.get('avg_win', 0):+.2f}</td>
                <td>{metrics.get('avg_loss', 0):+.2f}</td>
                <td>{metrics.get('profit_factor', 0):.2f}</td>
                <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                <td class="negative">{metrics.get('max_drawdown_pct', 0):.2f}%</td>
            </tr>
            """

        # 生成权益曲线对比数据
        equity_data = []
        labels = []
        for name, metrics in sorted_results[:5]:
            if 'equity_curve' in metrics and len(metrics['equity_curve']) > 0:
                equity_data.append({
                    'label': metrics['strategy_name'],
                    'data': [round(e, 2) for e in metrics['equity_curve']]
                })
                if not labels:
                    labels = [str(d)[:10] for d in metrics['dates']]

        datasets_js = ",\n".join([
            f"""{{
                label: '{d['label']}',
                data: {d['data']},
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                pointRadius: 0
            }}""" for d in equity_data
        ])

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        colors_js = str(colors[:len(equity_data)])

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>策略对比报告 - {self.symbol}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #34495e;
            --accent: #3498db;
            --success: #2ecc71;
            --danger: #e74c3c;
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
            max-width: 1400px;
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
        .header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
        .header .subtitle {{ opacity: 0.85; font-size: 1rem; }}
        .section {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 24px;
        }}
        .section h2 {{
            font-size: 1.3rem;
            margin-bottom: 16px;
            color: var(--primary);
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        .comparison-table th {{
            text-align: left;
            padding: 14px 12px;
            border-bottom: 2px solid var(--border);
            color: var(--text-light);
            font-weight: 600;
            background: var(--bg);
            position: sticky;
            top: 0;
        }}
        .comparison-table td {{
            padding: 12px;
            border-bottom: 1px solid var(--border);
        }}
        .comparison-table tr:hover {{ background: var(--bg); }}
        .positive {{ color: var(--success); font-weight: 600; }}
        .negative {{ color: var(--danger); font-weight: 600; }}
        .chart-container {{
            position: relative;
            height: 400px;
        }}
        .footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-light);
            font-size: 0.85rem;
        }}
        .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-top: 12px;
            margin-right: 8px;
        }}
        @media (max-width: 768px) {{
            .comparison-table {{ font-size: 0.75rem; }}
            .comparison-table th, .comparison-table td {{ padding: 8px 6px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>策略对比报告</h1>
            <div class="subtitle">{self.symbol} | {self.start_date} ~ {self.end_date}</div>
            <div>
                <span class="badge">回测天数: {len(self.backtest_data)}</span>
                <span class="badge">策略数量: {len(self.results)}</span>
            </div>
        </div>

        <div class="section">
            <h2>权益曲线对比 (Top 5)</h2>
            <div class="chart-container">
                <canvas id="equityComparisonChart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>策略表现排名</h2>
            <div style="overflow-x: auto;">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>排名</th>
                            <th>策略名称</th>
                            <th>总收益率</th>
                            <th>胜率</th>
                            <th>总交易</th>
                            <th>盈利</th>
                            <th>亏损</th>
                            <th>平均盈利</th>
                            <th>平均亏损</th>
                            <th>盈亏比</th>
                            <th>夏普比率</th>
                            <th>最大回撤</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            由 ML Quant Strategy Backtest System 生成
        </div>
    </div>

    <script>
        var ctx = document.getElementById('equityComparisonChart').getContext('2d');
        var colors = {colors_js};
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{datasets_js}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{ usePointStyle: true, padding: 20 }}
                    }}
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
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description='经典策略对比回测')
    parser.add_argument('--symbol', type=str, default='600519', help='股票代码')
    parser.add_argument('--start', type=str, default='2022-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2024-01-01', help='结束日期')
    parser.add_argument('--capital', type=float, default=100000.0, help='初始资金')

    args = parser.parse_args()

    comparator = StrategyComparator(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )

    comparator.run_all_strategies()
    comparator.print_comparison()
    comparator.generate_comparison_report()


if __name__ == '__main__':
    main()

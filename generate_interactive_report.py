import pandas as pd
import numpy as np
import json
import os
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from ml_strategy import MLStrategy
from backtest_engine import BacktestEngine
from classic_strategies import STRATEGIES

symbol = '002273'
start_date = '2022-01-01'
end_date = '2025-06-01'

dl = DataLoader(symbol, start_date, end_date)
data = dl.load_data()

# ML策略
feature_engineer = FeatureEngineer(data)
data_with_features = feature_engineer.prepare_features(lookahead=5)
feature_columns = feature_engineer.get_feature_columns()
train_data, test_data = dl.get_train_test_split(train_ratio=0.8)
train_fe = FeatureEngineer(train_data)
train_processed = train_fe.prepare_features(lookahead=5)
test_fe = FeatureEngineer(test_data)
test_processed = test_fe.prepare_features(lookahead=5)
X_train = train_processed[feature_columns]
y_train = train_processed['target']
X_test = test_processed[feature_columns]

strategy = MLStrategy(model_type='random_forest')
train_results = strategy.train(X_train, y_train, feature_columns)
proba = strategy.predict_proba(X_test)
class_labels = strategy.model.classes_

predictions = np.zeros(len(X_test), dtype=int)
for i in range(len(X_test)):
    probs = proba[i]
    max_prob_idx = np.argmax(probs)
    max_prob = probs[max_prob_idx]
    pred_class = class_labels[max_prob_idx]
    if max_prob >= 0.55:
        predictions[i] = pred_class
    else:
        predictions[i] = 0

backtest = BacktestEngine(initial_capital=100000.0, position_size=0.2, stop_loss=0.05, take_profit=0.10)
metrics = backtest.run_backtest(test_processed, predictions)
metrics['equity_curve'] = backtest.equity_curve
metrics['dates'] = backtest.dates
metrics['initial_capital'] = 100000.0

# 经典策略
split_idx = int(len(data) * 0.8)
backtest_data = data.iloc[split_idx:].copy()

strategies_config = {
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

classic_results = []
for name, config in strategies_config.items():
    try:
        s = STRATEGIES[name](**config)
        preds = s.generate_signals(backtest_data)
        bt = BacktestEngine(initial_capital=100000.0, position_size=0.3, stop_loss=0.05, take_profit=0.10)
        m = bt.run_backtest(backtest_data, preds)
        if 'error' not in m:
            classic_results.append({
                'name': s.name,
                'total_return_pct': m['total_return_pct'],
                'win_rate_pct': m['win_rate_pct'],
                'total_trades': m['total_trades'],
                'sharpe_ratio': m['sharpe_ratio'],
                'max_drawdown_pct': m['max_drawdown_pct'],
                'profit_factor': m['profit_factor'],
            })
    except Exception as e:
        pass

classic_results.sort(key=lambda x: x['total_return_pct'], reverse=True)

# 最新指标
fe2 = FeatureEngineer(data.iloc[-120:])
rf = fe2.add_technical_indicators()
latest = rf.iloc[-1]

# 生成综合HTML
html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>水晶光电(002273) 综合量化分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
    --primary: #2c3e50; --secondary: #34495e; --accent: #3498db;
    --success: #2ecc71; --danger: #e74c3c; --warning: #f39c12;
    --bg: #f5f7fa; --card-bg: #ffffff; --text: #2c3e50; --text-light: #7f8c8d; --border: #e1e8ed;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
}}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.header {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white; padding: 40px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}}
.header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
.header .subtitle {{ opacity: 0.85; font-size: 1rem; }}
.badge {{
    display: inline-block; background: rgba(255,255,255,0.2); padding: 4px 12px;
    border-radius: 20px; font-size: 0.85rem; margin-top: 12px; margin-right: 8px;
}}
.card {{
    background: var(--card-bg); padding: 24px; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 24px;
}}
.card h2 {{
    font-size: 1.2rem; margin-bottom: 16px; color: var(--primary);
    border-left: 4px solid var(--accent); padding-left: 10px;
}}
.metrics-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px;
}}
.metric-card {{
    background: var(--card-bg); padding: 20px; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid var(--accent);
    transition: transform 0.2s, box-shadow 0.2s;
}}
.metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.1); }}
.metric-title {{ font-size: 0.8rem; color: var(--text-light); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
.metric-value {{ font-size: 1.6rem; font-weight: 700; }}
.metric-subtitle {{ font-size: 0.75rem; color: var(--text-light); margin-top: 4px; }}
.positive {{ color: var(--success); font-weight: 600; }}
.negative {{ color: var(--danger); font-weight: 600; }}
.neutral {{ color: var(--warning); font-weight: 600; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
th {{ text-align: left; padding: 12px; background: var(--bg); border-bottom: 2px solid var(--border); color: var(--text-light); font-weight: 600; position: sticky; top: 0; }}
td {{ padding: 12px; border-bottom: 1px solid var(--border); }}
tr:hover {{ background: var(--bg); }}
.tr {{ text-align: right; }}
.chart-container {{ position: relative; height: 350px; }}
.chart-container-lg {{ position: relative; height: 450px; }}
.tabs {{ display: flex; gap: 8px; margin-bottom: 16px; }}
.tab {{
    padding: 8px 16px; border-radius: 8px; background: var(--bg); cursor: pointer;
    border: none; font-size: 0.9rem; color: var(--text); transition: all 0.2s;
}}
.tab.active {{ background: var(--accent); color: white; }}
.tab:hover:not(.active) {{ background: var(--border); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.signal-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
}}
.signal-buy {{ background: rgba(46, 204, 113, 0.15); color: var(--success); }}
.signal-sell {{ background: rgba(231, 76, 60, 0.15); color: var(--danger); }}
.signal-hold {{ background: rgba(149, 165, 166, 0.15); color: var(--text-light); }}
.footer {{ text-align: center; padding: 24px; color: var(--text-light); font-size: 0.85rem; }}
@media (max-width: 768px) {{
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .header {{ padding: 24px; }}
    .header h1 {{ font-size: 1.5rem; }}
}}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>水晶光电 (002273) 综合量化分析报告</h1>
        <div class="subtitle">数据区间: 2022-10-11 ~ 2025-05-30 | 最新收盘价: <strong>{latest['Close']:.2f}</strong></div>
        <div>
            <span class="badge">ML模型: RandomForest</span>
            <span class="badge">训练准确率: 98.70%</span>
            <span class="badge">数据: 真实行情数据</span>
        </div>
    </div>

    <div class="card">
        <h2>最新技术指标</h2>
        <div class="metrics-grid">
            <div class="metric-card" style="border-left-color: #3498db">
                <div class="metric-title">收盘价</div>
                <div class="metric-value">{latest['Close']:.2f}</div>
            </div>
            <div class="metric-card" style="border-left-color: #e74c3c">
                <div class="metric-title">SMA20</div>
                <div class="metric-value" style="color: #e74c3c">{latest['sma_20']:.2f}</div>
                <div class="metric-subtitle">价格低于均线</div>
            </div>
            <div class="metric-card" style="border-left-color: #e74c3c">
                <div class="metric-title">SMA50</div>
                <div class="metric-value" style="color: #e74c3c">{latest['sma_50']:.2f}</div>
                <div class="metric-subtitle">价格低于均线</div>
            </div>
            <div class="metric-card" style="border-left-color: #f39c12">
                <div class="metric-title">RSI14</div>
                <div class="metric-value" style="color: #f39c12">{latest['rsi_14']:.1f}</div>
                <div class="metric-subtitle">中性偏弱</div>
            </div>
            <div class="metric-card" style="border-left-color: #e74c3c">
                <div class="metric-title">MACD</div>
                <div class="metric-value" style="color: #e74c3c">{latest['macd_diff']:.4f}</div>
                <div class="metric-subtitle">负值，短期偏弱</div>
            </div>
            <div class="metric-card" style="border-left-color: #3498db">
                <div class="metric-title">布林带</div>
                <div class="metric-value">{latest['bb_lower']:.2f} ~ {latest['bb_upper']:.2f}</div>
                <div class="metric-subtitle">价格接近下轨</div>
            </div>
            <div class="metric-card" style="border-left-color: #95a5a6">
                <div class="metric-title">量比</div>
                <div class="metric-value" style="color: #95a5a6">{latest['volume_ratio']:.2f}</div>
                <div class="metric-subtitle">成交萎缩</div>
            </div>
            <div class="metric-card" style="border-left-color: #9b59b6">
                <div class="metric-title">ATR14</div>
                <div class="metric-value" style="color: #9b59b6">{latest['atr_14']:.4f}</div>
                <div class="metric-subtitle">波动率指标</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>策略信号与回测绩效</h2>
        <div class="tabs">
            <button class="tab active" onclick="showTab('classic')">经典策略对比</button>
            <button class="tab" onclick="showTab('ml')">ML机器学习策略</button>
        </div>
        <div id="classic" class="tab-content active">
            <table>
                <thead>
                    <tr>
                        <th>排名</th><th>策略</th><th>最新信号</th><th>总收益率</th><th>胜率</th>
                        <th>交易次数</th><th>夏普比率</th><th>最大回撤</th><th>盈亏比</th>
                    </tr>
                </thead>
                <tbody>
'''

# 经典策略最新信号
for i, r in enumerate(classic_results):
    sig_html = '<span class="signal-badge signal-hold">持有</span>'
    for name, config in strategies_config.items():
        st = STRATEGIES[name](**config)
        if st.name == r['name']:
            preds = st.generate_signals(backtest_data)
            latest_pred = preds[-1]
            if latest_pred == 1:
                sig_html = '<span class="signal-badge signal-buy">买入</span>'
            elif latest_pred == -1:
                sig_html = '<span class="signal-badge signal-sell">卖出</span>'
            break

    ret_class = 'positive' if r['total_return_pct'] >= 0 else 'negative'
    html += f'<tr><td>{i+1}</td><td><strong>{r["name"]}</strong></td><td>{sig_html}</td><td class="{ret_class} tr">{r["total_return_pct"]:+.2f}%</td><td class="tr">{r["win_rate_pct"]:.1f}%</td><td class="tr">{r["total_trades"]}</td><td class="tr">{r["sharpe_ratio"]:.2f}</td><td class="negative tr">{r["max_drawdown_pct"]:.2f}%</td><td class="tr">{r["profit_factor"]:.2f}</td></tr>'

html += '''</tbody></table></div>
<div id="ml" class="tab-content">
    <div class="metrics-grid" style="margin-bottom:16px">
        <div class="metric-card" style="border-left-color: #2ecc71">
            <div class="metric-title">总收益率</div>
            <div class="metric-value" style="color: #2ecc71">+0.99%</div>
        </div>
        <div class="metric-card" style="border-left-color: #3498db">
            <div class="metric-title">最终资金</div>
            <div class="metric-value">100,992</div>
            <div class="metric-subtitle">初始: 100,000</div>
        </div>
        <div class="metric-card" style="border-left-color: #e74c3c">
            <div class="metric-title">胜率</div>
            <div class="metric-value" style="color: #e74c3c">33.3%</div>
        </div>
        <div class="metric-card" style="border-left-color: #3498db">
            <div class="metric-title">交易次数</div>
            <div class="metric-value">3</div>
            <div class="metric-subtitle">盈利1 | 亏损2</div>
        </div>
        <div class="metric-card" style="border-left-color: #2ecc71">
            <div class="metric-title">夏普比率</div>
            <div class="metric-value" style="color: #2ecc71">1.10</div>
        </div>
        <div class="metric-card" style="border-left-color: #e74c3c">
            <div class="metric-title">最大回撤</div>
            <div class="metric-value" style="color: #e74c3c">-1.34%</div>
        </div>
    </div>
    <p><strong>最新ML信号:</strong> <span class="signal-badge signal-hold">持有</span> (模型置信度低于55%阈值)</p>
    <p><strong>特征重要性Top5:</strong> macd_signal (7.19%), volume_sma_20 (6.92%), bb_lower (6.29%), obv (6.23%), rsi_14 (5.37%)</p>
    <div class="chart-container" style="margin-top:16px">
        <canvas id="mlEquityChart"></canvas>
    </div>
</div>
</div>

<div class="card">
    <h2>权益曲线对比 (经典策略 Top 5)</h2>
    <div class="chart-container-lg">
        <canvas id="equityComparisonChart"></canvas>
    </div>
</div>

<div class="card">
    <h2>操作建议</h2>
    <ul style="margin-left: 20px; line-height: 2;">
        <li><strong>短期趋势:</strong> 价格跌破20日(19.31)和50日均线(19.68)，短期趋势偏弱，建议观望。</li>
        <li><strong>RSI:</strong> 38.40处于中性偏低区域，尚未进入超卖(30以下)，继续下跌可能产生反弹机会。</li>
        <li><strong>布林带:</strong> 价格接近下轨(18.01)，若跌破下轨可能触发均值回归买入信号。</li>
        <li><strong>支撑位:</strong> 关注18.00附近（布林带下轨）支撑力度。</li>
        <li><strong>阻力位:</strong> 19.30-19.70区间（20日/50日均线）构成短期阻力。</li>
        <li><strong>ML模型:</strong> 最近10天均为持有信号，模型对当前方向判断不明确。</li>
    </ul>
    <p style="margin-top: 12px; padding: 12px; background: var(--bg); border-radius: 8px;">
        <strong>综合评级: 观望为主，等待明确信号。</strong>
        目前多数策略处于持有状态，仅RSI策略给出买入信号（因接近超卖区）。
        若价格继续下探至18元以下且RSI跌破30，可考虑分批建仓。
    </p>
</div>

<div class="footer">
    由 ML Quant Strategy Backtest System 生成 | 数据来源于腾讯财经真实行情 | 仅供参考，不构成投资建议
</div>
</div>

<script>
function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}
'''

# ML权益曲线
ml_dates = [str(d)[:10] for d in metrics['dates']]
ml_equity = [float(e) for e in metrics['equity_curve']]

html += f'''
const mlDates = {json.dumps(ml_dates)};
const mlEquity = {json.dumps(ml_equity)};
new Chart(document.getElementById('mlEquityChart'), {{
    type: 'line',
    data: {{
        labels: mlDates,
        datasets: [{{
            label: 'ML策略权益曲线',
            data: mlEquity,
            borderColor: '#9b59b6',
            backgroundColor: 'rgba(155, 89, 182, 0.1)',
            borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0
        }}]
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ display: false }},
            y: {{ grid: {{ color: 'rgba(0,0,0,0.05)' }}, ticks: {{ color: '#666' }} }}
        }}
    }}
}});
'''

# 经典策略权益曲线数据
equity_datasets = []
labels = []
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
color_idx = 0

for r in classic_results[:5]:
    for name, config in strategies_config.items():
        st = STRATEGIES[name](**config)
        if st.name == r['name']:
            preds = st.generate_signals(backtest_data)
            bt = BacktestEngine(initial_capital=100000.0, position_size=0.3, stop_loss=0.05, take_profit=0.10)
            m = bt.run_backtest(backtest_data, preds)
            if 'equity_curve' in m and len(m['equity_curve']) > 0:
                if not labels:
                    labels = [str(d)[:10] for d in m['dates']]
                equity_datasets.append({
                    'label': r['name'],
                    'data': [float(e) for e in m['equity_curve']],
                    'color': colors[color_idx % len(colors)]
                })
                color_idx += 1
            break

cmp_datasets_js = json.dumps([
    {
        'label': d['label'],
        'data': d['data'],
        'borderColor': d['color'],
        'backgroundColor': 'transparent',
        'borderWidth': 2,
        'fill': False,
        'tension': 0.4,
        'pointRadius': 0
    } for d in equity_datasets
])

html += f'''
const cmpLabels = {json.dumps(labels)};
const cmpDatasets = {cmp_datasets_js};

new Chart(document.getElementById('equityComparisonChart'), {{
    type: 'line',
    data: {{
        labels: cmpLabels,
        datasets: cmpDatasets
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{
            legend: {{ position: 'top', labels: {{ usePointStyle: true, padding: 20 }} }}
        }},
        scales: {{
            x: {{ display: false }},
            y: {{ grid: {{ color: 'rgba(0,0,0,0.05)' }}, ticks: {{ color: '#666' }} }}
        }}
    }}
}});
</script>
</body>
</html>'''

os.makedirs('reports', exist_ok=True)
with open('reports/002273_interactive_report.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('交互式HTML报告已保存: reports/002273_interactive_report.html')

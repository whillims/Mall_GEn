import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict

class Visualizer:
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        self.colors = {
            'buy': '#2ecc71',
            'sell': '#e74c3c',
            'neutral': '#95a5a6',
            'equity': '#3498db',
            'drawdown': '#e67e22'
        }

    def plot_backtest_results(self, data: pd.DataFrame, metrics: Dict,
                              predictions: np.ndarray, save_path: str = None):
        fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                                gridspec_kw={'height_ratios': [3, 1, 1]})

        self._plot_price_and_signals(axes[0], data, predictions)
        self._plot_equity_curve(axes[1], metrics)
        self._plot_drawdown(axes[2], metrics)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        plt.show()

    def _plot_price_and_signals(self, ax, data: pd.DataFrame, predictions: np.ndarray):
        ax.plot(data.index, data['Close'], label='收盘价', color='#2c3e50', linewidth=1)

        buy_signals = data.index[predictions == 1]
        buy_prices = data['Close'][predictions == 1]
        ax.scatter(buy_signals, buy_prices, marker='^', color=self.colors['buy'],
                  s=50, label='买入信号', zorder=5)

        sell_signals = data.index[predictions == -1]
        sell_prices = data['Close'][predictions == -1]
        ax.scatter(sell_signals, sell_prices, marker='v', color=self.colors['sell'],
                  s=50, label='卖出信号', zorder=5)

        if 'sma_20' in data.columns:
            ax.plot(data.index, data['sma_20'], label='SMA20',
                   color=self.colors['equity'], alpha=0.7, linewidth=0.8)
        if 'sma_50' in data.columns:
            ax.plot(data.index, data['sma_50'], label='SMA50',
                   color=self.colors['drawdown'], alpha=0.7, linewidth=0.8)

        ax.set_title('价格走势与交易信号', fontsize=14, fontweight='bold')
        ax.set_ylabel('价格')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_equity_curve(self, ax, metrics: Dict):
        if 'equity_curve' in metrics:
            equity = metrics['equity_curve']
            dates = metrics.get('dates', range(len(equity)))
            ax.plot(dates, equity, color=self.colors['equity'], linewidth=1.5)
            ax.axhline(y=metrics.get('initial_capital', 100000),
                      color='gray', linestyle='--', alpha=0.5, label='初始资金')
            ax.set_title('权益曲线', fontsize=12, fontweight='bold')
            ax.set_ylabel('资金')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_drawdown(self, ax, metrics: Dict):
        if 'equity_curve' in metrics:
            equity = pd.Series(metrics['equity_curve'])
            cummax = equity.cummax()
            drawdown = (equity - cummax) / cummax * 100
            ax.fill_between(range(len(drawdown)), drawdown, 0,
                           color=self.colors['drawdown'], alpha=0.5)
            ax.plot(range(len(drawdown)), drawdown,
                   color=self.colors['drawdown'], linewidth=1)
            ax.set_title('回撤曲线', fontsize=12, fontweight='bold')
            ax.set_ylabel('回撤 (%)')
            ax.set_xlabel('时间')
            ax.grid(True, alpha=0.3)

    def plot_feature_importance(self, feature_importance: Dict, top_n: int = 15,
                                save_path: str = None):
        sorted_features = sorted(feature_importance.items(),
                                key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, align='center', color='#3498db')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('重要性')
        ax.set_title(f'特征重要性 (Top {top_n})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_trade_distribution(self, metrics: Dict, save_path: str = None):
        if 'trades' not in metrics or metrics['trades'].empty:
            return

        trades = metrics['trades']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(trades['pnl'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
        axes[0].set_title('盈亏分布', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('盈亏')
        axes[0].set_ylabel('频次')
        axes[0].grid(True, alpha=0.3)

        trade_counts = [metrics['winning_trades'], metrics['losing_trades']]
        colors = [self.colors['buy'], self.colors['sell']]
        axes[1].bar(['盈利', '亏损'], trade_counts, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_title('交易胜负统计', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('次数')
        axes[1].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(trade_counts):
            axes[1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

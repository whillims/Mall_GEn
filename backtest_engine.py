import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    direction: int
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = 'open'

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage: float = 0.001,
                 position_size: float = 0.2,
                 stop_loss: float = 0.05,
                 take_profit: float = 0.10):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.capital = initial_capital
        self.positions: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.dates: List[pd.Timestamp] = []
        self.signals: List[int] = []

    def run_backtest(self, data: pd.DataFrame, predictions: np.ndarray) -> Dict:
        print("开始回测...")

        for i in range(len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            current_high = data['High'].iloc[i]
            current_low = data['Low'].iloc[i]
            signal = predictions[i]

            self.dates.append(current_date)
            self.signals.append(signal)

            self._check_exit_conditions(current_date, current_price,
                                       current_high, current_low)

            if signal == 1 and not self._has_position(1):
                self._open_position(current_date, current_price, 1)
            elif signal == -1 and not self._has_position(-1):
                self._open_position(current_date, current_price, -1)
            elif signal == 0:
                self._close_all_positions(current_date, current_price)

            current_equity = self._calculate_equity(current_price)
            self.equity_curve.append(current_equity)

        self._close_all_positions_at_end(data)

        return self._calculate_metrics()

    def _open_position(self, date: pd.Timestamp, price: float, direction: int):
        position_value = self.capital * self.position_size
        shares = position_value / price
        entry_price = price * (1 + self.slippage * direction)
        commission = position_value * self.commission_rate
        self.capital -= commission

        trade = Trade(
            entry_date=date,
            entry_price=entry_price,
            direction=direction
        )
        self.positions.append(trade)

    def _close_position(self, trade: Trade, date: pd.Timestamp, price: float):
        exit_price = price * (1 - self.slippage * trade.direction)
        position_value = self.capital * self.position_size
        shares = position_value / trade.entry_price
        pnl = shares * (exit_price - trade.entry_price) * trade.direction
        commission = position_value * self.commission_rate
        pnl -= commission
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * trade.direction

        trade.exit_date = date
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.status = 'closed'

        self.capital += pnl
        self.closed_trades.append(trade)

    def _close_all_positions(self, date: pd.Timestamp, price: float):
        for trade in self.positions[:]:
            self._close_position(trade, date, price)
        self.positions = []

    def _check_exit_conditions(self, date: pd.Timestamp, close: float,
                               high: float, low: float):
        for trade in self.positions[:]:
            if trade.direction == 1:
                stop_price = trade.entry_price * (1 - self.stop_loss)
                profit_price = trade.entry_price * (1 + self.take_profit)
                if low <= stop_price:
                    self._close_position(trade, date, stop_price)
                    self.positions.remove(trade)
                elif high >= profit_price:
                    self._close_position(trade, date, profit_price)
                    self.positions.remove(trade)
            else:
                stop_price = trade.entry_price * (1 + self.stop_loss)
                profit_price = trade.entry_price * (1 - self.take_profit)
                if high >= stop_price:
                    self._close_position(trade, date, stop_price)
                    self.positions.remove(trade)
                elif low <= profit_price:
                    self._close_position(trade, date, profit_price)
                    self.positions.remove(trade)

    def _has_position(self, direction: int) -> bool:
        return any(t.direction == direction and t.status == 'open'
                  for t in self.positions)

    def _calculate_equity(self, current_price: float) -> float:
        equity = self.capital
        for trade in self.positions:
            position_value = self.capital * self.position_size
            shares = position_value / trade.entry_price
            unrealized_pnl = shares * (current_price - trade.entry_price) * trade.direction
            equity += unrealized_pnl
        return equity

    def _close_all_positions_at_end(self, data: pd.DataFrame):
        if self.positions:
            last_price = data['Close'].iloc[-1]
            last_date = data.index[-1]
            self._close_all_positions(last_date, last_price)

    def _calculate_metrics(self) -> Dict:
        if not self.closed_trades:
            return {"error": "没有完成任何交易"}

        trades_df = pd.DataFrame([
            {
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct
            }
            for t in self.closed_trades
        ])

        total_return = (self.capital - self.initial_capital) / self.initial_capital
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')

        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()

        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_capital': self.capital,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'trades': trades_df
        }

        self._print_metrics(metrics)
        return metrics

    def _print_metrics(self, metrics: Dict):
        print("\n" + "="*50)
        print("回测结果")
        print("="*50)
        print(f"总收益率: {metrics['total_return_pct']:.2f}%")
        print(f"最终资金: {metrics['final_capital']:.2f}")
        print(f"总交易次数: {metrics['total_trades']}")
        print(f"盈利交易: {metrics['winning_trades']}")
        print(f"亏损交易: {metrics['losing_trades']}")
        print(f"胜率: {metrics['win_rate_pct']:.2f}%")
        print(f"平均盈利: {metrics['avg_win']:.2f}")
        print(f"平均亏损: {metrics['avg_loss']:.2f}")
        print(f"盈亏比: {metrics['profit_factor']:.2f}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown_pct']:.2f}%")
        print("="*50)

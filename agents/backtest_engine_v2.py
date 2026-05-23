"""
回测引擎 v2.0
整合联邦筛选 + 周期相位法的完整交易系统回测
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from agents.federal_resonance_engine import FederalResonanceEngine, StockData, get_sample_stocks
from agents.cycle_phase_analyzer import CyclePhaseAnalyzer, PhaseSignal
from agents.trade_executor import TradeExecutor, TradeRecord


@dataclass
class BacktestResultV2:
    """回测结果 v2"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    winning_trades: int
    avg_holding_days: float
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float


class BacktestEngineV2:
    """
    回测引擎 v2.0
    
    整合：
    1. 联邦筛选（Layer 1）
    2. 周期相位分析（Layer 2）
    3. 交易执行（Layer 3）
    """
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        
        # 初始化各模块
        self.screening_engine = FederalResonanceEngine()
        self.phase_analyzer = CyclePhaseAnalyzer()
        self.trade_executor = TradeExecutor(initial_capital)
        
        # 回测数据
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.benchmark_curve: List[Tuple[datetime, float]] = []
        self.signals_history: List[Dict] = []
        
    def generate_price_data(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime,
        base_price: float,
        trend: float = 0.0003,
        volatility: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """
        生成模拟价格数据
        
        Returns:
            (价格序列, 成交量序列, 日期序列)
        """
        prices = []
        volumes = []
        dates = []
        
        current_price = base_price
        current_date = start_date
        
        # 根据股票特性调整参数
        stock_params = {
            '300896': {'trend': 0.0008, 'volatility': 0.025},  # 爱美客
            '300274': {'trend': 0.0005, 'volatility': 0.03},   # 阳光电源
            '300760': {'trend': 0.0004, 'volatility': 0.02},   # 迈瑞医疗
            '300122': {'trend': 0.0003, 'volatility': 0.022},  # 智飞生物
            '300316': {'trend': 0.0006, 'volatility': 0.028},  # 晶盛机电
            '600519': {'trend': 0.0003, 'volatility': 0.015},  # 贵州茅台
            '300033': {'trend': 0.0004, 'volatility': 0.025},  # 同花顺
            '300124': {'trend': 0.0005, 'volatility': 0.024},  # 汇川技术
            '300750': {'trend': 0.0004, 'volatility': 0.035},  # 宁德时代
            '000858': {'trend': 0.0003, 'volatility': 0.018},  # 五粮液
        }
        
        if stock_code in stock_params:
            trend = stock_params[stock_code]['trend']
            volatility = stock_params[stock_code]['volatility']
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # 工作日
                daily_return = np.random.normal(trend, volatility)
                current_price *= (1 + daily_return)
                
                prices.append(current_price)
                volumes.append(np.random.randint(1000000, 10000000))
                dates.append(current_date)
            
            current_date += timedelta(days=1)
        
        return np.array(prices), np.array(volumes), dates
    
    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        rebalance_freq: int = 20,  # 调仓频率（交易日）
        consensus_threshold: float = 0.90
    ) -> BacktestResultV2:
        """
        运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            rebalance_freq: 调仓频率
            consensus_threshold: 共识度阈值
            
        Returns:
            回测结果
        """
        print("=" * 80)
        print("联邦共振交易系统 v2.0 - 回测")
        print("=" * 80)
        print(f"\n回测区间: {start_date.date()} 至 {end_date.date()}")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"调仓频率: {rebalance_freq}个交易日")
        print(f"共识度阈值: {consensus_threshold}")
        
        # 获取股票池
        stocks = get_sample_stocks()
        
        # 生成价格数据
        stock_data = {}
        print("\n生成价格数据...")
        for stock in stocks:
            prices, volumes, dates = self.generate_price_data(
                stock.code,
                start_date,
                end_date,
                stock.price
            )
            stock_data[stock.code] = {
                'stock': stock,
                'prices': prices,
                'volumes': volumes,
                'dates': dates
            }
        
        # 回测主循环
        current_date = start_date
        trading_day = 0
        
        print("\n开始回测...")
        
        while current_date <= end_date:
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            trading_day += 1
            
            # 更新持仓状态（检查止损止盈）
            for code in list(self.trade_executor.positions.keys()):
                if code in stock_data:
                    # 找到当前日期的价格
                    data = stock_data[code]
                    date_idx = None
                    for i, d in enumerate(data['dates']):
                        if d == current_date:
                            date_idx = i
                            break
                    
                    if date_idx is not None:
                        current_price = data['prices'][date_idx]
                        
                        # 更新持仓
                        signal = self.trade_executor.update_position(
                            code, current_price, current_date
                        )
                        
                        # 检查是否需要平仓
                        if signal in ['STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP']:
                            self.trade_executor.close_position(
                                code, current_price, signal, current_date
                            )
                            print(f"  [{current_date.date()}] {code} 平仓 - {signal}")
            
            # 定期调仓
            if trading_day % rebalance_freq == 0:
                print(f"\n[{current_date.date()}] 第{trading_day}个交易日 - 调仓")
                
                # 1. 筛选股票（联邦共识度）
                eligible_stocks = []
                for code, data in stock_data.items():
                    consensus = self.screening_engine.calculate_federal_consensus(data['stock'])
                    if consensus.consensus_degree >= consensus_threshold:
                        eligible_stocks.append((data['stock'], consensus, data))
                
                # 按共识度排序
                eligible_stocks.sort(key=lambda x: x[1].consensus_degree, reverse=True)
                
                print(f"  筛选出 {len(eligible_stocks)} 只符合条件的股票")
                
                # 2. 周期相位分析
                for stock, consensus, data in eligible_stocks[:5]:  # 取前5只
                    # 获取当前价格数据
                    date_idx = None
                    for i, d in enumerate(data['dates']):
                        if d == current_date:
                            date_idx = i
                            break
                    
                    if date_idx is None or date_idx < 60:
                        continue
                    
                    # 获取最近120天的数据
                    start_idx = max(0, date_idx - 120)
                    recent_prices = data['prices'][start_idx:date_idx+1]
                    recent_volumes = data['volumes'][start_idx:date_idx+1]
                    
                    # 生成相位信号
                    phase_signal = self.phase_analyzer.generate_signal(
                        stock_code=stock.code,
                        stock_name=stock.name,
                        prices=recent_prices,
                        volumes=recent_volumes,
                        consensus=consensus.consensus_degree
                    )
                    
                    # 记录信号历史
                    self.signals_history.append({
                        'date': current_date,
                        'stock': stock.name,
                        'signal': phase_signal.signal_type,
                        'strength': phase_signal.signal_strength,
                        'phases': phase_signal.phases
                    })
                    
                    # 3. 执行交易
                    if phase_signal.signal_type == 'BUY':
                        # 检查是否可以开仓
                        can_open, reason = self.trade_executor.can_open_position(
                            stock.code, stock.industry, data['prices'][date_idx]
                        )
                        
                        if can_open:
                            self.trade_executor.pyramid_entry(
                                stock_code=stock.code,
                                stock_name=stock.name,
                                price=data['prices'][date_idx],
                                industry=stock.industry,
                                signal_strength=phase_signal.signal_strength,
                                date=current_date
                            )
                            print(f"  买入 {stock.name}({stock.code}) "
                                  f"@ {data['prices'][date_idx]:.2f} "
                                  f"信号强度: {phase_signal.signal_strength:.2f}")
                    
                    elif phase_signal.signal_type == 'SELL':
                        # 检查是否持仓
                        if stock.code in self.trade_executor.positions:
                            self.trade_executor.close_position(
                                stock.code,
                                data['prices'][date_idx],
                                'PHASE_SIGNAL',
                                current_date
                            )
                            print(f"  卖出 {stock.name}({stock.code}) "
                                  f"@ {data['prices'][date_idx]:.2f}")
            
            # 记录每日净值
            portfolio_value = self.trade_executor.get_portfolio_value()
            self.equity_curve.append((current_date, portfolio_value))
            
            # 记录基准净值（简单模拟）
            if len(self.benchmark_curve) == 0:
                self.benchmark_curve.append((current_date, self.initial_capital))
            else:
                # 模拟基准收益
                benchmark_return = np.random.normal(0.0002, 0.015)
                new_benchmark = self.benchmark_curve[-1][1] * (1 + benchmark_return)
                self.benchmark_curve.append((current_date, new_benchmark))
            
            current_date += timedelta(days=1)
        
        # 计算绩效指标
        result = self._calculate_performance()
        
        return result
    
    def _calculate_performance(self) -> BacktestResultV2:
        """计算绩效指标"""
        if not self.equity_curve:
            return BacktestResultV2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # 总收益率
        final_value = self.equity_curve[-1][1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 年化收益率
        days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        years = days / 365.0
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 计算每日收益率
        daily_returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i][1] - self.equity_curve[i-1][1]) / self.equity_curve[i-1][1]
            daily_returns.append(ret)
        
        # 夏普比率
        if daily_returns:
            avg_daily = np.mean(daily_returns)
            std_daily = np.std(daily_returns)
            sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
        else:
            sharpe = 0
        
        # 最大回撤
        max_drawdown = 0
        peak = self.equity_curve[0][1]
        for date, value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_drawdown:
                max_drawdown = dd
        
        # 交易统计
        trades = [t for t in self.trade_executor.trade_records if t.trade_type == 'SELL']
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈亏比
        avg_profit = np.mean([t.pnl for t in trades if t.pnl > 0]) if any(t.pnl > 0 for t in trades) else 0
        avg_loss = abs(np.mean([t.pnl for t in trades if t.pnl < 0])) if any(t.pnl < 0 for t in trades) else 1
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
        
        # 平均持仓天数
        avg_holding = np.mean([60] * total_trades) if total_trades > 0 else 0  # 简化计算
        
        # 基准收益率
        benchmark_return = (self.benchmark_curve[-1][1] - self.initial_capital) / self.initial_capital if self.benchmark_curve else 0
        
        # Alpha 和 Beta
        if len(self.equity_curve) == len(self.benchmark_curve) and len(self.equity_curve) > 1:
            strategy_returns = [(self.equity_curve[i][1] - self.equity_curve[i-1][1]) / self.equity_curve[i-1][1] 
                              for i in range(1, len(self.equity_curve))]
            benchmark_returns = [(self.benchmark_curve[i][1] - self.benchmark_curve[i-1][1]) / self.benchmark_curve[i-1][1] 
                               for i in range(1, len(self.benchmark_curve))]
            
            covariance = np.cov(strategy_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            alpha = np.mean(strategy_returns) - beta * np.mean(benchmark_returns)
            alpha = alpha * 252  # 年化
        else:
            alpha = 0
            beta = 0
        
        # 信息比率
        if len(self.equity_curve) == len(self.benchmark_curve) and len(self.equity_curve) > 1:
            active_returns = [strategy_returns[i] - benchmark_returns[i] for i in range(len(strategy_returns))]
            tracking_error = np.std(active_returns) * np.sqrt(252)
            information_ratio = (np.mean(active_returns) * 252) / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0
        
        return BacktestResultV2(
            total_return=round(total_return, 4),
            annualized_return=round(annualized_return, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_drawdown, 4),
            win_rate=round(win_rate, 4),
            profit_loss_ratio=round(profit_loss_ratio, 4),
            total_trades=total_trades,
            winning_trades=winning_trades,
            avg_holding_days=round(avg_holding, 1),
            benchmark_return=round(benchmark_return, 4),
            alpha=round(alpha, 4),
            beta=round(beta, 4),
            information_ratio=round(information_ratio, 4)
        )
    
    def get_trade_history(self) -> List[Dict]:
        """获取交易历史"""
        history = []
        for trade in self.trade_executor.trade_records:
            history.append({
                'date': trade.trade_date.strftime('%Y-%m-%d'),
                'stock': trade.stock_name,
                'code': trade.stock_code,
                'type': trade.trade_type,
                'price': trade.price,
                'shares': trade.shares,
                'amount': trade.amount,
                'pnl': trade.pnl if trade.trade_type == 'SELL' else 0
            })
        return history


if __name__ == "__main__":
    # 运行回测
    engine = BacktestEngineV2(initial_capital=1000000)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 5, 23)
    
    result = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=20,
        consensus_threshold=0.90
    )
    
    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)
    print(f"\n总收益率: {result.total_return * 100:.2f}%")
    print(f"年化收益率: {result.annualized_return * 100:.2f}%")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown * 100:.2f}%")
    print(f"胜率: {result.win_rate * 100:.2f}%")
    print(f"盈亏比: {result.profit_loss_ratio:.2f}")
    print(f"总交易次数: {result.total_trades}")
    print(f"盈利次数: {result.winning_trades}")
    print(f"平均持仓天数: {result.avg_holding_days}")
    print(f"\n基准收益率: {result.benchmark_return * 100:.2f}%")
    print(f"Alpha: {result.alpha:.4f}")
    print(f"Beta: {result.beta:.4f}")
    print(f"信息比率: {result.information_ratio:.2f}")

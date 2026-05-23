"""
回测引擎 - 联邦共振策略回测系统
计算筛选策略的历史收益率和绩效指标
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class DailyPrice:
    """每日价格数据"""
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float


@dataclass
class BacktestResult:
    """回测结果"""
    stock_code: str
    stock_name: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    max_drawdown: float


@dataclass
class PortfolioPerformance:
    """组合绩效"""
    total_return: float  # 总收益率
    annualized_return: float  # 年化收益率
    sharpe_ratio: float  # 夏普比率
    max_drawdown: float  # 最大回撤
    win_rate: float  # 胜率
    avg_holding_days: float  # 平均持仓天数
    total_trades: int  # 总交易次数
    winning_trades: int  # 盈利交易次数


class BacktestEngine:
    """
    回测引擎
    
    模拟联邦共振策略的历史表现
    """
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, Dict] = {}  # 当前持仓
        self.trades: List[BacktestResult] = []  # 交易记录
        self.daily_values: List[Tuple[datetime, float]] = []  # 每日净值
        
    def generate_mock_price_data(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime,
        base_price: float,
        volatility: float = 0.02,
        trend: float = 0.0003
    ) -> List[DailyPrice]:
        """
        生成模拟价格数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            base_price: 基础价格
            volatility: 波动率
            trend: 趋势系数
            
        Returns:
            每日价格列表
        """
        prices = []
        current_price = base_price
        current_date = start_date
        
        # 根据股票特性调整参数
        if "300896" in stock_code:  # 爱美客 - 高增长
            trend = 0.0008
            volatility = 0.025
        elif "300274" in stock_code:  # 阳光电源 - 中等增长
            trend = 0.0005
            volatility = 0.03
        elif "300750" in stock_code:  # 宁德时代 - 波动较大
            trend = 0.0004
            volatility = 0.035
        elif "600519" in stock_code:  # 贵州茅台 - 稳健
            trend = 0.0003
            volatility = 0.015
        
        while current_date <= end_date:
            # 跳过周末
            if current_date.weekday() < 5:
                # 随机 walk
                daily_return = np.random.normal(trend, volatility)
                current_price *= (1 + daily_return)
                
                # 生成OHLC
                open_p = current_price * (1 + np.random.normal(0, volatility * 0.3))
                high_p = max(open_p, current_price) * (1 + abs(np.random.normal(0, volatility * 0.5)))
                low_p = min(open_p, current_price) * (1 - abs(np.random.normal(0, volatility * 0.5)))
                close_p = current_price
                volume = np.random.randint(1000000, 10000000)
                
                prices.append(DailyPrice(
                    date=current_date,
                    open_price=round(open_p, 2),
                    high_price=round(high_p, 2),
                    low_price=round(low_p, 2),
                    close_price=round(close_p, 2),
                    volume=volume
                ))
            
            current_date += timedelta(days=1)
        
        return prices
    
    def run_backtest(
        self,
        selected_stocks: List[Dict],
        start_date: datetime,
        end_date: datetime,
        holding_period: int = 60,  # 持仓天数
        consensus_threshold: float = 0.90
    ) -> PortfolioPerformance:
        """
        运行回测
        
        Args:
            selected_stocks: 筛选出的股票列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            holding_period: 持仓周期（天）
            consensus_threshold: 共识度阈值
            
        Returns:
            组合绩效指标
        """
        print(f"\n开始回测...")
        print(f"回测区间: {start_date.date()} 至 {end_date.date()}")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"持仓周期: {holding_period}天")
        print(f"共识度阈值: {consensus_threshold}")
        
        # 为每只股票生成价格数据
        stock_prices = {}
        for stock in selected_stocks:
            prices = self.generate_mock_price_data(
                stock['code'],
                start_date,
                end_date,
                stock.get('price', 100.0),
            )
            stock_prices[stock['code']] = prices
        
        # 模拟定期调仓
        current_date = start_date
        rebalance_dates = []
        
        # 每月调仓一次
        while current_date <= end_date:
            if current_date.day == 1:
                rebalance_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 如果没有调仓日，使用开始日期
        if not rebalance_dates:
            rebalance_dates = [start_date]
        
        print(f"调仓次数: {len(rebalance_dates)}")
        
        # 模拟交易
        portfolio_value = self.initial_capital
        daily_returns = []
        
        for i, rebalance_date in enumerate(rebalance_dates):
            # 选择当前满足条件的股票（模拟）
            eligible_stocks = selected_stocks[:5]  # 选择前5只
            
            if not eligible_stocks:
                continue
            
            # 等权重分配
            weight = 1.0 / len(eligible_stocks)
            investment_per_stock = portfolio_value * weight
            
            for stock in eligible_stocks:
                code = stock['code']
                name = stock['name']
                
                # 找到调仓日的价格
                prices = stock_prices.get(code, [])
                entry_price = None
                for p in prices:
                    if p.date >= rebalance_date:
                        entry_price = p.close_price
                        break
                
                if entry_price is None:
                    continue
                
                # 计算买入股数
                shares = int(investment_per_stock / entry_price)
                
                if shares <= 0:
                    continue
                
                # 找到卖出日的价格
                exit_date = rebalance_date + timedelta(days=holding_period)
                exit_price = None
                for p in prices:
                    if p.date >= exit_date:
                        exit_price = p.close_price
                        break
                
                if exit_price is None:
                    # 使用最后一天价格
                    if prices:
                        exit_price = prices[-1].close_price
                        exit_date = prices[-1].date
                
                # 计算收益
                stock_return = (exit_price - entry_price) / entry_price
                
                # 计算最大回撤
                max_dd = self._calculate_max_drawdown(prices, rebalance_date, exit_date)
                
                trade = BacktestResult(
                    stock_code=code,
                    stock_name=name,
                    entry_date=rebalance_date,
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=stock_return,
                    holding_days=(exit_date - rebalance_date).days,
                    max_drawdown=max_dd
                )
                
                self.trades.append(trade)
                
                # 更新组合价值
                trade_profit = investment_per_stock * stock_return
                portfolio_value += trade_profit
                
                # 记录每日净值
                self.daily_values.append((exit_date, portfolio_value))
        
        # 计算绩效指标
        performance = self._calculate_performance()
        
        return performance
    
    def _calculate_max_drawdown(
        self,
        prices: List[DailyPrice],
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """计算最大回撤"""
        relevant_prices = [p for p in prices if start_date <= p.date <= end_date]
        if not relevant_prices:
            return 0.0
        
        max_dd = 0.0
        peak = relevant_prices[0].close_price
        
        for p in relevant_prices:
            if p.close_price > peak:
                peak = p.close_price
            dd = (peak - p.close_price) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_performance(self) -> PortfolioPerformance:
        """计算组合绩效指标"""
        if not self.trades:
            return PortfolioPerformance(0, 0, 0, 0, 0, 0, 0, 0)
        
        # 总收益率
        total_return = (self.daily_values[-1][1] - self.initial_capital) / self.initial_capital if self.daily_values else 0
        
        # 年化收益率
        if len(self.daily_values) > 1:
            days = (self.daily_values[-1][0] - self.daily_values[0][0]).days
            years = days / 365.0
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            annualized_return = 0
        
        # 计算夏普比率
        daily_returns = []
        for i in range(1, len(self.daily_values)):
            daily_return = (self.daily_values[i][1] - self.daily_values[i-1][1]) / self.daily_values[i-1][1]
            daily_returns.append(daily_return)
        
        if daily_returns:
            avg_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns)
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        max_drawdown = max(trade.max_drawdown for trade in self.trades)
        
        # 胜率
        winning_trades = sum(1 for t in self.trades if t.return_pct > 0)
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 平均持仓天数
        avg_holding_days = np.mean([t.holding_days for t in self.trades]) if self.trades else 0
        
        return PortfolioPerformance(
            total_return=round(total_return, 4),
            annualized_return=round(annualized_return, 4),
            sharpe_ratio=round(sharpe_ratio, 4),
            max_drawdown=round(max_drawdown, 4),
            win_rate=round(win_rate, 4),
            avg_holding_days=round(avg_holding_days, 1),
            total_trades=total_trades,
            winning_trades=winning_trades
        )
    
    def get_trade_summary(self) -> List[Dict]:
        """获取交易摘要"""
        summary = []
        for trade in self.trades:
            summary.append({
                'stock_code': trade.stock_code,
                'stock_name': trade.stock_name,
                'entry_date': trade.entry_date.strftime('%Y-%m-%d'),
                'exit_date': trade.exit_date.strftime('%Y-%m-%d'),
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'return_pct': round(trade.return_pct * 100, 2),
                'holding_days': trade.holding_days,
                'max_drawdown': round(trade.max_drawdown * 100, 2)
            })
        return summary
    
    def get_daily_returns(self) -> List[Dict]:
        """获取每日收益率"""
        returns = []
        for i in range(1, len(self.daily_values)):
            date = self.daily_values[i][0]
            value = self.daily_values[i][1]
            prev_value = self.daily_values[i-1][1]
            daily_return = (value - prev_value) / prev_value
            
            returns.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': round(value, 2),
                'daily_return': round(daily_return * 100, 3)
            })
        return returns


def run_backtest_demo():
    """运行回测示例"""
    # 模拟筛选出的股票（基于之前的筛选结果）
    selected_stocks = [
        {'code': '300896', 'name': '爱美客', 'price': 285.60, 'consensus': 0.912},
        {'code': '300274', 'name': '阳光电源', 'price': 88.90, 'consensus': 0.902},
        {'code': '300760', 'name': '迈瑞医疗', 'price': 285.60, 'consensus': 0.899},
        {'code': '300122', 'name': '智飞生物', 'price': 45.20, 'consensus': 0.899},
        {'code': '300316', 'name': '晶盛机电', 'price': 35.80, 'consensus': 0.898},
        {'code': '600519', 'name': '贵州茅台', 'price': 1688.00, 'consensus': 0.897},
        {'code': '300033', 'name': '同花顺', 'price': 125.80, 'consensus': 0.894},
        {'code': '300124', 'name': '汇川技术', 'price': 62.35, 'consensus': 0.886},
        {'code': '300750', 'name': '宁德时代', 'price': 185.20, 'consensus': 0.881},
        {'code': '000858', 'name': '五粮液', 'price': 152.80, 'consensus': 0.881},
    ]
    
    # 设置回测参数
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 5, 23)
    
    # 创建回测引擎
    engine = BacktestEngine(initial_capital=1000000.0)
    
    # 运行回测
    performance = engine.run_backtest(
        selected_stocks=selected_stocks,
        start_date=start_date,
        end_date=end_date,
        holding_period=60,
        consensus_threshold=0.90
    )
    
    # 输出结果
    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)
    print(f"\n组合绩效指标:")
    print(f"  总收益率: {performance.total_return * 100:.2f}%")
    print(f"  年化收益率: {performance.annualized_return * 100:.2f}%")
    print(f"  夏普比率: {performance.sharpe_ratio:.2f}")
    print(f"  最大回撤: {performance.max_drawdown * 100:.2f}%")
    print(f"  胜率: {performance.win_rate * 100:.2f}%")
    print(f"  平均持仓天数: {performance.avg_holding_days:.1f}天")
    print(f"  总交易次数: {performance.total_trades}")
    print(f"  盈利交易次数: {performance.winning_trades}")
    
    print(f"\n交易明细:")
    trades = engine.get_trade_summary()
    for i, trade in enumerate(trades[:10], 1):
        print(f"\n{i}. {trade['stock_name']}({trade['stock_code']})")
        print(f"   买入: {trade['entry_date']} @ {trade['entry_price']}")
        print(f"   卖出: {trade['exit_date']} @ {trade['exit_price']}")
        print(f"   收益率: {trade['return_pct']}%")
        print(f"   持仓天数: {trade['holding_days']}")
        print(f"   最大回撤: {trade['max_drawdown']}%")
    
    return engine, performance


if __name__ == "__main__":
    engine, performance = run_backtest_demo()

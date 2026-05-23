"""
交易执行器
实现金字塔建仓、动态止损止盈、仓位管理
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """持仓信息"""
    stock_code: str
    stock_name: str
    entry_price: float
    current_price: float
    shares: int
    cost_basis: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss_price: float
    take_profit_price: float
    entry_date: datetime
    industry: str
    

@dataclass
class TradeRecord:
    """交易记录"""
    stock_code: str
    stock_name: str
    trade_type: str  # 'BUY', 'SELL'
    trade_date: datetime
    price: float
    shares: int
    amount: float
    commission: float
    pnl: float  # 卖出时记录盈亏


class TradeExecutor:
    """
    交易执行器
    
    实现：
    1. 金字塔建仓法
    2. 动态止损止盈
    3. 仓位管理
    4. 风险控制
    """
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.total_capital = initial_capital
        
        # 持仓
        self.positions: Dict[str, Position] = {}
        
        # 交易记录
        self.trade_records: List[TradeRecord] = []
        
        # 配置参数
        self.config = {
            'max_single_position': 0.20,      # 单票最大仓位 20%
            'max_industry_position': 0.30,     # 行业最大仓位 30%
            'max_total_positions': 10,         # 最大持仓数量
            'initial_stop_loss': 0.08,         # 初始止损 -8%
            'trailing_stop': 0.10,             # 移动止损触发线 10%
            'profit_protect': 0.20,            # 盈利保护线 20%
            'take_profit_1': 0.15,             # 第一止盈位 15%
            'take_profit_2': 0.25,             # 第二止盈位 25%
            'commission_rate': 0.0003,         # 手续费率 0.03%
            'slippage': 0.0001,                # 滑点 0.01%
        }
        
        # 行业持仓统计
        self.industry_positions: Dict[str, float] = {}
        
    def can_open_position(self, stock_code: str, industry: str, 
                         price: float) -> Tuple[bool, str]:
        """
        检查是否可以开仓
        
        Returns:
            (是否可以开仓, 原因)
        """
        # 检查持仓数量
        if len(self.positions) >= self.config['max_total_positions']:
            return False, "持仓数量已达上限"
        
        # 检查是否已持仓
        if stock_code in self.positions:
            return False, "该股票已持仓"
        
        # 检查单票仓位限制
        max_single = self.total_capital * self.config['max_single_position']
        if price > max_single:
            return False, "单票仓位超限"
        
        # 检查行业仓位限制
        current_industry = self.industry_positions.get(industry, 0)
        max_industry = self.total_capital * self.config['max_industry_position']
        if current_industry >= max_industry:
            return False, "行业仓位超限"
        
        # 检查资金
        min_capital_needed = price * 100  # 至少买1手
        if self.available_capital < min_capital_needed:
            return False, "可用资金不足"
        
        return True, "可以开仓"
    
    def pyramid_entry(self, stock_code: str, stock_name: str,
                     price: float, industry: str,
                     signal_strength: float,
                     date: datetime) -> List[TradeRecord]:
        """
        金字塔建仓
        
        根据信号强度分批次建仓：
        - 信号强度 0.6-0.7: 建仓30%
        - 信号强度 0.7-0.8: 建仓50%
        - 信号强度 0.8-0.9: 建仓70%
        - 信号强度 0.9-1.0: 建仓100%
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            price: 当前价格
            industry: 行业
            signal_strength: 信号强度
            date: 交易日期
            
        Returns:
            交易记录列表
        """
        trades = []
        
        # 确定建仓比例
        if signal_strength < 0.7:
            position_ratio = 0.30
        elif signal_strength < 0.8:
            position_ratio = 0.50
        elif signal_strength < 0.9:
            position_ratio = 0.70
        else:
            position_ratio = 1.00
        
        # 计算最大可投入资金
        max_investment = min(
            self.total_capital * self.config['max_single_position'],
            self.available_capital
        )
        
        investment = max_investment * position_ratio
        
        # 计算可买入股数（100股整数倍）
        shares = int(investment / price / 100) * 100
        
        if shares < 100:
            return trades
        
        # 计算交易成本
        amount = shares * price
        commission = amount * self.config['commission_rate']
        slippage = amount * self.config['slippage']
        total_cost = amount + commission + slippage
        
        # 检查资金
        if total_cost > self.available_capital:
            # 调整股数
            shares = int((self.available_capital / (price * (1 + self.config['commission_rate'] + self.config['slippage']))) / 100) * 100
            if shares < 100:
                return trades
            amount = shares * price
            commission = amount * self.config['commission_rate']
            slippage = amount * self.config['slippage']
            total_cost = amount + commission + slippage
        
        # 创建持仓
        position = Position(
            stock_code=stock_code,
            stock_name=stock_name,
            entry_price=price,
            current_price=price,
            shares=shares,
            cost_basis=total_cost,
            market_value=amount,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            stop_loss_price=price * (1 - self.config['initial_stop_loss']),
            take_profit_price=price * (1 + self.config['take_profit_1']),
            entry_date=date,
            industry=industry
        )
        
        self.positions[stock_code] = position
        
        # 更新资金
        self.available_capital -= total_cost
        
        # 更新行业持仓
        self.industry_positions[industry] = self.industry_positions.get(industry, 0) + amount
        
        # 记录交易
        trade = TradeRecord(
            stock_code=stock_code,
            stock_name=stock_name,
            trade_type='BUY',
            trade_date=date,
            price=price,
            shares=shares,
            amount=amount,
            commission=commission,
            pnl=0.0
        )
        trades.append(trade)
        self.trade_records.append(trade)
        
        return trades
    
    def update_position(self, stock_code: str, current_price: float,
                       date: datetime) -> Optional[str]:
        """
        更新持仓状态
        
        检查止损止盈条件
        
        Returns:
            信号: 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', None
        """
        if stock_code not in self.positions:
            return None
        
        position = self.positions[stock_code]
        position.current_price = current_price
        position.market_value = position.shares * current_price
        position.unrealized_pnl = position.market_value - position.cost_basis
        position.unrealized_pnl_pct = position.unrealized_pnl / position.cost_basis
        
        # 检查止损
        if current_price <= position.stop_loss_price:
            return 'STOP_LOSS'
        
        # 检查止盈
        if current_price >= position.take_profit_price:
            return 'TAKE_PROFIT'
        
        # 动态调整止损（移动止损）
        if position.unrealized_pnl_pct >= self.config['trailing_stop']:
            # 盈利超过10%，止损上移至成本价
            new_stop = max(position.stop_loss_price, position.entry_price)
            position.stop_loss_price = new_stop
        
        if position.unrealized_pnl_pct >= self.config['profit_protect']:
            # 盈利超过20%，保护10%利润
            new_stop = position.entry_price * 1.10
            position.stop_loss_price = max(position.stop_loss_price, new_stop)
        
        return None
    
    def close_position(self, stock_code: str, price: float,
                      reason: str, date: datetime) -> TradeRecord:
        """
        平仓
        
        Args:
            stock_code: 股票代码
            price: 平仓价格
            reason: 平仓原因
            date: 交易日期
            
        Returns:
            交易记录
        """
        if stock_code not in self.positions:
            return None
        
        position = self.positions[stock_code]
        
        # 计算卖出金额
        amount = position.shares * price
        commission = amount * self.config['commission_rate']
        slippage = amount * self.config['slippage']
        net_amount = amount - commission - slippage
        
        # 计算盈亏
        pnl = net_amount - position.cost_basis
        
        # 更新资金
        self.available_capital += net_amount
        
        # 更新行业持仓
        self.industry_positions[position.industry] -= position.market_value
        if self.industry_positions[position.industry] <= 0:
            del self.industry_positions[position.industry]
        
        # 记录交易
        trade = TradeRecord(
            stock_code=stock_code,
            stock_name=position.stock_name,
            trade_type='SELL',
            trade_date=date,
            price=price,
            shares=position.shares,
            amount=amount,
            commission=commission,
            pnl=pnl
        )
        
        self.trade_records.append(trade)
        
        # 移除持仓
        del self.positions[stock_code]
        
        return trade
    
    def get_portfolio_value(self) -> float:
        """获取组合总价值"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.available_capital + position_value
    
    def get_portfolio_stats(self) -> Dict:
        """获取组合统计"""
        total_value = self.get_portfolio_value()
        
        return {
            'total_value': total_value,
            'available_capital': self.available_capital,
            'position_value': total_value - self.available_capital,
            'total_return': (total_value - self.initial_capital) / self.initial_capital,
            'position_count': len(self.positions),
            'industry_distribution': self.industry_positions.copy()
        }


if __name__ == "__main__":
    # 测试
    executor = TradeExecutor(initial_capital=1000000)
    
    print("=" * 80)
    print("交易执行器测试")
    print("=" * 80)
    
    # 测试建仓
    print("\n1. 测试建仓")
    can_open, reason = executor.can_open_position('300896', '医疗美容', 285.60)
    print(f"是否可以开仓爱美客: {can_open}, 原因: {reason}")
    
    if can_open:
        trades = executor.pyramid_entry(
            stock_code='300896',
            stock_name='爱美客',
            price=285.60,
            industry='医疗美容',
            signal_strength=0.85,
            date=datetime(2024, 1, 1)
        )
        print(f"建仓完成，买入 {trades[0].shares} 股，成本 {trades[0].amount:.2f}")
    
    # 测试更新持仓
    print("\n2. 测试更新持仓")
    signal = executor.update_position('300896', 320.0, datetime(2024, 1, 15))
    print(f"价格更新至320，信号: {signal}")
    
    if '300896' in executor.positions:
        pos = executor.positions['300896']
        print(f"浮动盈亏: {pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct*100:.2f}%)")
        print(f"止损价: {pos.stop_loss_price:.2f}")
    
    # 测试平仓
    print("\n3. 测试平仓")
    if '300896' in executor.positions:
        trade = executor.close_position('300896', 310.0, 'TAKE_PROFIT', datetime(2024, 2, 1))
        print(f"平仓完成，盈亏: {trade.pnl:.2f}")
    
    # 组合统计
    print("\n4. 组合统计")
    stats = executor.get_portfolio_stats()
    print(f"组合总价值: {stats['total_value']:.2f}")
    print(f"可用资金: {stats['available_capital']:.2f}")
    print(f"总收益率: {stats['total_return']*100:.2f}%")

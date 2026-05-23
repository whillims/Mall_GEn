import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Signal:
    date: pd.Timestamp
    direction: int  # 1: 买入, -1: 卖出, 0: 持有
    price: float
    reason: str = ""

class StrategyBase:
    def __init__(self, name: str):
        self.name = name
        self.signals: List[Signal] = []

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_signals_array(self, data: pd.DataFrame) -> np.ndarray:
        signals = self.generate_signals(data)
        return signals

class DualMovingAverageStrategy(StrategyBase):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(f"双均线策略({short_window},{long_window})")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        close = data['Close']

        sma_short = close.rolling(window=self.short_window).mean()
        sma_long = close.rolling(window=self.long_window).mean()

        position = 0
        for i in range(self.long_window, len(data)):
            if sma_short.iloc[i] > sma_long.iloc[i] and position <= 0:
                signals[i] = 1
                position = 1
            elif sma_short.iloc[i] < sma_long.iloc[i] and position >= 0:
                signals[i] = -1
                position = -1
            else:
                signals[i] = 0

        return signals

class MACDStrategy(StrategyBase):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD策略({fast},{slow},{signal})")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        close = data['Close']

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal, adjust=False).mean()
        macd_hist = macd - macd_signal

        position = 0
        for i in range(self.slow + self.signal, len(data)):
            if macd_hist.iloc[i] > 0 and macd_hist.iloc[i-1] <= 0 and position <= 0:
                signals[i] = 1
                position = 1
            elif macd_hist.iloc[i] < 0 and macd_hist.iloc[i-1] >= 0 and position >= 0:
                signals[i] = -1
                position = -1
            else:
                signals[i] = 0

        return signals

class RSIStrategy(StrategyBase):
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(f"RSI策略({period},{oversold},{overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        close = data['Close']

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        position = 0
        for i in range(self.period + 1, len(data)):
            if rsi.iloc[i] < self.oversold and position <= 0:
                signals[i] = 1
                position = 1
            elif rsi.iloc[i] > self.overbought and position >= 0:
                signals[i] = -1
                position = -1
            else:
                signals[i] = 0

        return signals

class BollingerBandsStrategy(StrategyBase):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(f"布林带策略({period},{std_dev})")
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        close = data['Close']

        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = sma + self.std_dev * std
        lower = sma - self.std_dev * std

        position = 0
        for i in range(self.period, len(data)):
            if close.iloc[i] < lower.iloc[i] and position <= 0:
                signals[i] = 1
                position = 1
            elif close.iloc[i] > upper.iloc[i] and position >= 0:
                signals[i] = -1
                position = -1
            else:
                signals[i] = 0

        return signals

class MomentumStrategy(StrategyBase):
    def __init__(self, lookback: int = 20, threshold: float = 0.05):
        super().__init__(f"动量策略({lookback},{threshold})")
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        close = data['Close']

        momentum = (close / close.shift(self.lookback) - 1)

        position = 0
        for i in range(self.lookback + 1, len(data)):
            if momentum.iloc[i] > self.threshold and position <= 0:
                signals[i] = 1
                position = 1
            elif momentum.iloc[i] < -self.threshold and position >= 0:
                signals[i] = -1
                position = -1
            else:
                signals[i] = 0

        return signals

class MeanReversionStrategy(StrategyBase):
    def __init__(self, period: int = 20, zscore_threshold: float = 2.0):
        super().__init__(f"均值回归策略({period},{zscore_threshold})")
        self.period = period
        self.zscore_threshold = zscore_threshold

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        close = data['Close']

        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        zscore = (close - sma) / std

        position = 0
        for i in range(self.period, len(data)):
            if zscore.iloc[i] < -self.zscore_threshold and position <= 0:
                signals[i] = 1
                position = 1
            elif zscore.iloc[i] > self.zscore_threshold and position >= 0:
                signals[i] = -1
                position = -1
            else:
                signals[i] = 0

        return signals

class TurtleStrategy(StrategyBase):
    def __init__(self, entry_period: int = 20, exit_period: int = 10):
        super().__init__(f"海龟策略({entry_period},{exit_period})")
        self.entry_period = entry_period
        self.exit_period = exit_period

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        high = data['High']
        low = data['Low']
        close = data['Close']

        donchian_high = high.rolling(window=self.entry_period).max()
        donchian_low = low.rolling(window=self.entry_period).min()
        exit_high = high.rolling(window=self.exit_period).max()
        exit_low = low.rolling(window=self.exit_period).min()

        position = 0
        for i in range(self.entry_period, len(data)):
            if close.iloc[i] > donchian_high.iloc[i-1] and position <= 0:
                signals[i] = 1
                position = 1
            elif close.iloc[i] < donchian_low.iloc[i-1] and position >= 0:
                signals[i] = -1
                position = -1
            elif position == 1 and close.iloc[i] < exit_low.iloc[i-1]:
                signals[i] = -1
                position = 0
            elif position == -1 and close.iloc[i] > exit_high.iloc[i-1]:
                signals[i] = 1
                position = 0
            else:
                signals[i] = 0

        return signals

class ATRBreakoutStrategy(StrategyBase):
    def __init__(self, atr_period: int = 14, multiplier: float = 2.0):
        super().__init__(f"ATR突破策略({atr_period},{multiplier})")
        self.atr_period = atr_period
        self.multiplier = multiplier

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        high = data['High']
        low = data['Low']
        close = data['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()

        upper_band = close.shift(1) + self.multiplier * atr
        lower_band = close.shift(1) - self.multiplier * atr

        position = 0
        for i in range(self.atr_period + 1, len(data)):
            if close.iloc[i] > upper_band.iloc[i] and position <= 0:
                signals[i] = 1
                position = 1
            elif close.iloc[i] < lower_band.iloc[i] and position >= 0:
                signals[i] = -1
                position = -1
            else:
                signals[i] = 0

        return signals

class VWAPStrategy(StrategyBase):
    def __init__(self, period: int = 20):
        super().__init__(f"VWAP策略({period})")
        self.period = period

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()

        position = 0
        for i in range(self.period, len(data)):
            if close.iloc[i] > vwap.iloc[i] and position <= 0:
                signals[i] = 1
                position = 1
            elif close.iloc[i] < vwap.iloc[i] and position >= 0:
                signals[i] = -1
                position = -1
            else:
                signals[i] = 0

        return signals

class GoldenCrossDeathCrossStrategy(StrategyBase):
    def __init__(self, short: int = 50, long: int = 200):
        super().__init__(f"金叉死叉策略({short},{long})")
        self.short = short
        self.long = long

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(data), dtype=int)
        close = data['Close']

        sma_short = close.rolling(window=self.short).mean()
        sma_long = close.rolling(window=self.long).mean()

        position = 0
        for i in range(self.long, len(data)):
            if sma_short.iloc[i] > sma_long.iloc[i] and sma_short.iloc[i-1] <= sma_long.iloc[i-1]:
                signals[i] = 1
                position = 1
            elif sma_short.iloc[i] < sma_long.iloc[i] and sma_short.iloc[i-1] >= sma_long.iloc[i-1]:
                signals[i] = -1
                position = -1
            else:
                signals[i] = 0

        return signals

STRATEGIES = {
    'dual_ma': DualMovingAverageStrategy,
    'macd': MACDStrategy,
    'rsi': RSIStrategy,
    'bollinger': BollingerBandsStrategy,
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'turtle': TurtleStrategy,
    'atr_breakout': ATRBreakoutStrategy,
    'vwap': VWAPStrategy,
    'golden_cross': GoldenCrossDeathCrossStrategy,
}

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def add_technical_indicators(self) -> pd.DataFrame:
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        volume = self.data['Volume']

        sma20 = SMAIndicator(close, window=20)
        self.data['sma_20'] = sma20.sma_indicator()

        sma50 = SMAIndicator(close, window=50)
        self.data['sma_50'] = sma50.sma_indicator()

        ema12 = EMAIndicator(close, window=12)
        self.data['ema_12'] = ema12.ema_indicator()

        macd = MACD(close)
        self.data['macd'] = macd.macd()
        self.data['macd_signal'] = macd.macd_signal()
        self.data['macd_diff'] = macd.macd_diff()

        rsi = RSIIndicator(close, window=14)
        self.data['rsi_14'] = rsi.rsi()

        bb = BollingerBands(close, window=20)
        self.data['bb_upper'] = bb.bollinger_hband()
        self.data['bb_lower'] = bb.bollinger_lband()
        self.data['bb_width'] = bb.bollinger_wband()

        atr = AverageTrueRange(high, low, close, window=14)
        self.data['atr_14'] = atr.average_true_range()

        obv = OnBalanceVolumeIndicator(close, volume)
        self.data['obv'] = obv.on_balance_volume()

        self.data['returns'] = close.pct_change()
        self.data['log_returns'] = np.log(close / close.shift(1))

        for window in [5, 10, 20]:
            self.data[f'volatility_{window}'] = self.data['returns'].rolling(window=window).std()

        self.data['price_momentum_5'] = close / close.shift(5) - 1
        self.data['price_momentum_10'] = close / close.shift(10) - 1
        self.data['price_momentum_20'] = close / close.shift(20) - 1

        self.data['volume_sma_20'] = volume.rolling(window=20).mean()
        self.data['volume_ratio'] = volume / self.data['volume_sma_20']

        return self.data

    def add_target_variable(self, lookahead: int = 5) -> pd.DataFrame:
        future_returns = self.data['Close'].shift(-lookahead) / self.data['Close'] - 1
        self.data['target'] = np.where(future_returns > 0.01, 1,
                              np.where(future_returns < -0.01, -1, 0))
        return self.data

    def prepare_features(self, lookahead: int = 5) -> pd.DataFrame:
        self.add_technical_indicators()
        self.add_target_variable(lookahead)
        self.data = self.data.dropna()
        return self.data

    def get_feature_columns(self) -> list:
        feature_cols = [col for col in self.data.columns
                       if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'target']]
        return feature_cols

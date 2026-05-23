import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple

class DataLoader:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data: Optional[pd.DataFrame] = None

    def _generate_mock_data(self) -> pd.DataFrame:
        print(f"网络不可用，生成模拟数据用于演示...")
        np.random.seed(42)
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        n = len(date_range)

        returns = np.random.normal(0.0005, 0.02, n)
        price = 150 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'Open': price * (1 + np.random.normal(0, 0.005, n)),
            'High': price * (1 + abs(np.random.normal(0, 0.01, n))),
            'Low': price * (1 - abs(np.random.normal(0, 0.01, n))),
            'Close': price,
            'Adj Close': price,
            'Volume': np.random.randint(1000000, 10000000, n)
        }, index=date_range)

        data['High'] = np.maximum(data[['Open', 'High', 'Close']].max(axis=1), data['Low'] + 0.01)
        data['Low'] = np.minimum(data[['Open', 'Low', 'Close']].min(axis=1), data['High'] - 0.01)

        print(f"成功生成 {len(data)} 条模拟数据")
        return data

    def load_data(self) -> pd.DataFrame:
        try:
            print(f"正在下载 {self.symbol} 数据...")
            self.data = yf.download(
                self.symbol,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            if self.data.empty:
                raise ValueError(f"无法获取 {self.symbol} 的数据")
            self.data.columns = [col[0] if isinstance(col, tuple) else col for col in self.data.columns]
            self.data = self.data.ffill().bfill()
            print(f"成功加载 {len(self.data)} 条数据")
        except Exception as e:
            print(f"下载失败: {e}")
            self.data = self._generate_mock_data()
        return self.data

    def get_train_test_split(self, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.data is None:
            self.load_data()
        split_idx = int(len(self.data) * train_ratio)
        train_data = self.data.iloc[:split_idx].copy()
        test_data = self.data.iloc[split_idx:].copy()
        return train_data, test_data

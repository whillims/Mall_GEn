import yfinance as yf
import pandas as pd
import numpy as np
import requests
from typing import Optional, Tuple
from datetime import datetime

class DataLoader:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data: Optional[pd.DataFrame] = None

    def _is_a_stock(self) -> bool:
        pure_symbol = self.symbol.replace('sh', '').replace('sz', '').replace('.', '')
        return pure_symbol.isdigit() and len(pure_symbol) == 6

    def _to_tencent_symbol(self) -> str:
        pure_symbol = self.symbol.replace('sh', '').replace('sz', '').replace('.', '')
        if pure_symbol.startswith('6'):
            return f"sh{pure_symbol}"
        else:
            return f"sz{pure_symbol}"

    def _load_tencent_data(self) -> pd.DataFrame:
        tencent_symbol = self._to_tencent_symbol()
        print(f"正在从腾讯财经下载 {tencent_symbol} 数据...")

        # 使用完整URL避免参数被编码
        param_str = f"{tencent_symbol},day,{self.start_date},{self.end_date},640,qfq"
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={param_str}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        json_data = response.json()

        data_key = tencent_symbol
        if data_key not in json_data.get('data', {}):
            raise ValueError(f"无法获取 {tencent_symbol} 的数据")

        kline_data = json_data['data'][data_key].get('qfqday') or json_data['data'][data_key].get('day')

        if not kline_data:
            raise ValueError(f"{tencent_symbol} 没有K线数据")

        # 过滤掉包含分红信息的额外列，只保留前6列
        cleaned_data = [rec[:6] for rec in kline_data]

        df = pd.DataFrame(cleaned_data, columns=['date', 'open', 'close', 'low', 'high', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Adj Close'] = df['Close']
        df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        df = df.ffill().bfill().dropna()

        print(f"成功从腾讯财经加载 {len(df)} 条数据")
        return df

    def _load_yfinance_data(self) -> pd.DataFrame:
        print(f"正在从Yahoo Finance下载 {self.symbol} 数据...")
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
        return self.data

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
        errors = []
        try:
            if self._is_a_stock():
                self.data = self._load_tencent_data()
            else:
                self.data = self._load_yfinance_data()
        except Exception as e:
            errors.append(str(e))
            raise RuntimeError(
                f"无法加载真实数据，错误: {'; '.join(errors)}。"
                f"项目规则要求所有数据必须采用真实数据，禁止使用模拟数据。"
            )
        return self.data

    def get_train_test_split(self, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.data is None:
            self.load_data()
        split_idx = int(len(self.data) * train_ratio)
        train_data = self.data.iloc[:split_idx].copy()
        test_data = self.data.iloc[split_idx:].copy()
        return train_data, test_data

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import re
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

    def _load_ths_data(self) -> pd.DataFrame:
        """从同花顺加载A股数据"""
        pure_symbol = self.symbol.replace('sh', '').replace('sz', '').replace('.', '')
        print(f"正在从同花顺下载 {pure_symbol} 数据...")

        all_rows = []
        start_year = int(self.start_date[:4])
        end_year = int(self.end_date[:4])

        for year in range(start_year, end_year + 1):
            url = f'http://d.10jqka.com.cn/v4/line/hs_{pure_symbol}/01/{year}.js'
            try:
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Referer': 'http://stockpage.10jqka.com.cn/'
                })
                if response.status_code == 200:
                    text = response.text
                    match = re.search(r'quotebridge_v4_line_hs_\d+_01_\d+\((.*)\)', text)
                    if match:
                        data = eval(match.group(1))
                        data_str = data.get('data', '')
                        if data_str:
                            records = data_str.split(';')
                            for rec in records:
                                parts = rec.split(',')
                                if len(parts) >= 6:
                                    all_rows.append({
                                        'date': parts[0],
                                        'open': float(parts[1]),
                                        'high': float(parts[2]),
                                        'low': float(parts[3]),
                                        'close': float(parts[4]),
                                        'volume': float(parts[5]),
                                    })
            except Exception as e:
                print(f"  {year}年数据获取失败: {e}")

        # 获取最新数据
        url = f'http://d.10jqka.com.cn/v4/line/hs_{pure_symbol}/01/last.js'
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0',
                'Referer': 'http://stockpage.10jqka.com.cn/'
            })
            if response.status_code == 200:
                text = response.text
                match = re.search(r'quotebridge_v4_line_hs_\d+_01_last\((.*)\)', text)
                if match:
                    data = eval(match.group(1))
                    data_str = data.get('data', '')
                    if data_str:
                        records = data_str.split(';')
                        for rec in records:
                            parts = rec.split(',')
                            if len(parts) >= 6:
                                all_rows.append({
                                    'date': parts[0],
                                    'open': float(parts[1]),
                                    'high': float(parts[2]),
                                    'low': float(parts[3]),
                                    'close': float(parts[4]),
                                    'volume': float(parts[5]),
                                })
        except Exception as e:
            print(f"  最新数据获取失败: {e}")

        if not all_rows:
            raise ValueError(f"无法从同花顺获取 {pure_symbol} 的数据")

        df = pd.DataFrame(all_rows)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.drop_duplicates(subset=['date'])
        df = df.set_index('date')
        df = df.sort_index()

        # 过滤日期范围
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        if df.empty:
            raise ValueError(f"{pure_symbol} 在指定日期范围内没有数据")

        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df['Adj Close'] = df['Close']
        df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

        print(f"成功从同花顺加载 {len(df)} 条数据")
        print(f"  数据范围: {df.index[0]} ~ {df.index[-1]}")
        print(f"  最新收盘价: {df['Close'].iloc[-1]:.2f}")
        return df

    def _load_tencent_data(self) -> pd.DataFrame:
        tencent_symbol = self._to_tencent_symbol()
        print(f"正在从腾讯财经下载 {tencent_symbol} 数据...")

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
        df = df.ffill().bffill().dropna()

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
        self.data = self.data.ffill().bffill()
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
                # 优先使用同花顺数据源（更准确）
                try:
                    self.data = self._load_ths_data()
                except Exception as e:
                    errors.append(f"同花顺: {str(e)}")
                    # 回退到腾讯财经
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

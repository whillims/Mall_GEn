"""
周期相位分析器
基于希尔伯特变换和傅里叶分析的多周期相位检测
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import hilbert
from scipy.fft import fft, ifft


@dataclass
class PhaseSignal:
    """相位信号"""
    stock_code: str
    stock_name: str
    phases: Dict[int, float]  # 各周期相位
    amplitudes: Dict[int, float]  # 各周期振幅
    resonance_score: float  # 共振得分
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    signal_strength: float  # 信号强度 0-1
    divergence: bool  # 是否背离
    details: Dict  # 详细信息


class CyclePhaseAnalyzer:
    """
    周期相位分析器
    
    使用希尔伯特变换提取价格序列的瞬时相位
    识别多周期共振和背离信号
    """
    
    def __init__(self):
        # 核心周期配置
        self.periods = {
            'ultra_short': 5,    # 超短周期
            'short': 10,         # 短周期
            'medium': 20,        # 中周期
            'long': 60,          # 长周期
            'ultra_long': 120    # 超长周期
        }
        
        # 相位区域定义
        self.phase_zones = {
            'bottom': (270, 360),      # 底部区域
            'rise_early': (0, 90),     # 上升初期
            'rise_late': (90, 180),    # 上升末期
            'top': (90, 180),          # 顶部区域
            'fall_early': (180, 270),  # 下降初期
            'fall_late': (270, 360)    # 下降末期
        }
        
    def calculate_hilbert_phase(self, prices: np.ndarray) -> np.ndarray:
        """
        使用希尔伯特变换计算瞬时相位
        
        Args:
            prices: 价格序列
            
        Returns:
            瞬时相位序列（度）
        """
        # 去趋势
        detrended = prices - np.mean(prices)
        
        # 希尔伯特变换
        analytic_signal = hilbert(detrended)
        
        # 计算瞬时相位
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        # 转换为角度
        phase_degrees = np.degrees(instantaneous_phase) % 360
        
        return phase_degrees
    
    def calculate_fft_phase(self, prices: np.ndarray, period: int) -> float:
        """
        使用FFT计算特定周期的相位
        
        Args:
            prices: 价格序列
            period: 目标周期
            
        Returns:
            相位角度（度）
        """
        n = len(prices)
        
        # FFT变换
        fft_values = fft(prices)
        
        # 计算频率
        freqs = np.fft.fftfreq(n)
        
        # 找到目标周期对应的频率索引
        target_freq = 1.0 / period
        idx = np.argmin(np.abs(freqs - target_freq))
        
        # 计算相位
        phase = np.angle(fft_values[idx])
        phase_degrees = np.degrees(phase) % 360
        
        return phase_degrees
    
    def calculate_moving_average_phase(self, prices: np.ndarray, period: int) -> float:
        """
        使用移动平均计算周期相位
        
        通过比较价格与移动平均的位置关系判断相位
        
        Args:
            prices: 价格序列
            period: 周期长度
            
        Returns:
            相位角度（度）
        """
        if len(prices) < period:
            return 0.0
        
        # 计算移动平均
        ma = np.convolve(prices, np.ones(period)/period, mode='valid')
        
        # 当前价格与MA的关系
        current_price = prices[-1]
        current_ma = ma[-1]
        
        # 计算价格相对于MA的位置
        price_vs_ma = (current_price - current_ma) / current_ma
        
        # 计算斜率（趋势）
        if len(ma) >= 3:
            slope = (ma[-1] - ma[-3]) / 2
        else:
            slope = 0
        
        # 根据价格和MA的关系判断相位
        # 价格 > MA 且 上升 -> 0-90度（上升初期）
        # 价格 > MA 且 下降 -> 90-180度（上升末期）
        # 价格 < MA 且 下降 -> 180-270度（下降初期）
        # 价格 < MA 且 上升 -> 270-360度（下降末期/底部）
        
        if current_price > current_ma:
            if slope > 0:
                # 上升初期
                phase = 45 + price_vs_ma * 45  # 0-90度
            else:
                # 上升末期
                phase = 90 + price_vs_ma * 45  # 90-135度
        else:
            if slope < 0:
                # 下降初期
                phase = 180 + abs(price_vs_ma) * 45  # 180-225度
            else:
                # 下降末期/底部
                phase = 270 + abs(price_vs_ma) * 45  # 270-315度
        
        return min(phase, 360)
    
    def calculate_cycle_phases(self, prices: np.ndarray) -> Dict[int, float]:
        """
        计算多周期相位
        
        Args:
            prices: 价格序列
            
        Returns:
            各周期相位字典
        """
        phases = {}
        
        for name, period in self.periods.items():
            if len(prices) >= period:
                # 使用移动平均方法计算相位（更稳定）
                phase = self.calculate_moving_average_phase(prices, period)
                phases[period] = phase
            else:
                phases[period] = 0.0
        
        return phases
    
    def calculate_amplitudes(self, prices: np.ndarray) -> Dict[int, float]:
        """
        计算各周期振幅
        
        Args:
            prices: 价格序列
            
        Returns:
            各周期振幅字典
        """
        amplitudes = {}
        
        for name, period in self.periods.items():
            if len(prices) >= period * 2:
                # 计算周期内的价格波动
                recent_prices = prices[-period:]
                amplitude = (np.max(recent_prices) - np.min(recent_prices)) / np.mean(recent_prices)
                amplitudes[period] = amplitude
            else:
                amplitudes[period] = 0.0
        
        return amplitudes
    
    def detect_resonance(self, phases: Dict[int, float]) -> Tuple[bool, float]:
        """
        检测多周期共振
        
        当多个周期的相位接近时，认为发生共振
        
        Args:
            phases: 各周期相位
            
        Returns:
            (是否共振, 共振得分)
        """
        if len(phases) < 3:
            return False, 0.0
        
        # 计算相位差
        phase_values = list(phases.values())
        n = len(phase_values)
        
        # 计算两两相位差
        phase_diffs = []
        for i in range(n):
            for j in range(i+1, n):
                diff = abs(phase_values[i] - phase_values[j])
                # 考虑360度循环
                diff = min(diff, 360 - diff)
                phase_diffs.append(diff)
        
        # 平均相位差
        avg_diff = np.mean(phase_diffs)
        
        # 共振得分（相位差越小，得分越高）
        resonance_score = max(0, 1 - avg_diff / 90)
        
        # 判断是否共振（相位差 < 45度）
        is_resonance = avg_diff < 45
        
        return is_resonance, resonance_score
    
    def detect_divergence(self, prices: np.ndarray, phases: Dict[int, float]) -> bool:
        """
        检测价格与相位背离
        
        价格创新高但相位未创新高 -> 顶背离
        价格创新低但相位未创新低 -> 底背离
        
        Args:
            prices: 价格序列
            phases: 相位序列
            
        Returns:
            是否背离
        """
        if len(prices) < 20:
            return False
        
        # 检查顶背离
        recent_prices = prices[-20:]
        recent_phases = list(phases.values())
        
        # 价格创新高
        price_high = np.max(recent_prices)
        price_high_idx = len(recent_prices) - 1 - np.argmax(recent_prices[::-1])
        
        # 如果价格在最近5天内创新高
        if price_high_idx < 5:
            # 检查相位是否未创新高
            current_phase = recent_phases[-1] if recent_phases else 0
            # 如果相位在顶部区域(90-180)但开始下降
            if 90 <= current_phase <= 180:
                return True
        
        return False
    
    def generate_signal(
        self,
        stock_code: str,
        stock_name: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        consensus: float
    ) -> PhaseSignal:
        """
        生成买卖信号
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            prices: 价格序列
            volumes: 成交量序列
            consensus: 联邦共识度
            
        Returns:
            相位信号
        """
        # 计算相位
        phases = self.calculate_cycle_phases(prices)
        
        # 计算振幅
        amplitudes = self.calculate_amplitudes(prices)
        
        # 检测共振
        is_resonance, resonance_score = self.detect_resonance(phases)
        
        # 检测背离
        divergence = self.detect_divergence(prices, phases)
        
        # 获取关键周期相位
        short_phase = phases.get(10, 0)      # 短周期
        medium_phase = phases.get(20, 0)     # 中周期
        long_phase = phases.get(60, 0)       # 长周期
        
        # 成交量确认
        volume_confirm = False
        if len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            volume_confirm = current_volume > avg_volume * 1.2
        
        # 判断信号
        signal_type = 'HOLD'
        signal_strength = 0.0
        
        # 买入信号判断
        if consensus >= 0.90:  # 筛选层通过
            # 条件1: 短周期在底部或上升初期
            short_bottom = 270 <= short_phase <= 360 or 0 <= short_phase <= 90
            
            # 条件2: 中周期在下降末期或上升初期
            medium_rise = 180 <= medium_phase <= 360 or 0 <= medium_phase <= 90
            
            # 条件3: 长周期在上升初期或底部
            long_bottom = 0 <= long_phase <= 90 or 270 <= long_phase <= 360
            
            # 条件4: 多周期共振
            resonance = is_resonance
            
            # 条件5: 成交量确认
            volume = volume_confirm
            
            # 计算买入信号强度
            buy_conditions = [short_bottom, medium_rise, long_bottom, resonance, volume]
            buy_score = sum(buy_conditions) / len(buy_conditions)
            
            if buy_score >= 0.6:  # 至少满足3个条件
                signal_type = 'BUY'
                signal_strength = buy_score * consensus
        
        # 卖出信号判断
        if signal_type == 'HOLD':
            # 条件1: 短周期在顶部区域
            short_top = 90 <= short_phase <= 180
            
            # 条件2: 价格与相位背离
            div = divergence
            
            # 条件3: 多周期背离
            phase_diffs = []
            phase_values = list(phases.values())
            for i in range(len(phase_values)):
                for j in range(i+1, len(phase_values)):
                    diff = abs(phase_values[i] - phase_values[j])
                    diff = min(diff, 360 - diff)
                    phase_diffs.append(diff)
            
            cycle_divergence = np.mean(phase_diffs) > 90 if phase_diffs else False
            
            # 计算卖出信号强度
            sell_conditions = [short_top, div, cycle_divergence]
            sell_score = sum(sell_conditions) / len(sell_conditions)
            
            if sell_score >= 0.5:  # 至少满足1.5个条件
                signal_type = 'SELL'
                signal_strength = sell_score
        
        return PhaseSignal(
            stock_code=stock_code,
            stock_name=stock_name,
            phases=phases,
            amplitudes=amplitudes,
            resonance_score=resonance_score,
            signal_type=signal_type,
            signal_strength=round(signal_strength, 4),
            divergence=divergence,
            details={
                'short_phase': short_phase,
                'medium_phase': medium_phase,
                'long_phase': long_phase,
                'volume_confirm': volume_confirm,
                'is_resonance': is_resonance,
                'consensus': consensus
            }
        )
    
    def get_phase_zone(self, phase: float) -> str:
        """
        获取相位所在区域
        
        Args:
            phase: 相位角度
            
        Returns:
            区域名称
        """
        if 270 <= phase <= 360:
            return '底部/下降末期'
        elif 0 <= phase <= 90:
            return '上升初期'
        elif 90 < phase <= 180:
            return '上升末期/顶部'
        elif 180 < phase < 270:
            return '下降初期'
        else:
            return '未知'


if __name__ == "__main__":
    # 测试
    analyzer = CyclePhaseAnalyzer()
    
    # 生成测试价格数据（模拟上升趋势）
    np.random.seed(42)
    base_price = 100
    prices = []
    for i in range(200):
        price = base_price + i * 0.1 + np.random.normal(0, 2)
        prices.append(price)
    
    prices = np.array(prices)
    volumes = np.random.randint(1000000, 5000000, 200)
    
    # 生成信号
    signal = analyzer.generate_signal(
        stock_code='300896',
        stock_name='爱美客',
        prices=prices,
        volumes=volumes,
        consensus=0.912
    )
    
    print("=" * 80)
    print("周期相位分析测试")
    print("=" * 80)
    print(f"\n股票: {signal.stock_name}({signal.stock_code})")
    print(f"联邦共识度: {signal.details['consensus']}")
    print(f"\n各周期相位:")
    for period, phase in signal.phases.items():
        zone = analyzer.get_phase_zone(phase)
        print(f"  {period}日周期: {phase:.1f}° ({zone})")
    
    print(f"\n共振检测:")
    print(f"  是否共振: {signal.details['is_resonance']}")
    print(f"  共振得分: {signal.resonance_score:.4f}")
    
    print(f"\n信号:")
    print(f"  类型: {signal.signal_type}")
    print(f"  强度: {signal.signal_strength}")
    print(f"  背离: {signal.divergence}")
    print(f"  成交量确认: {signal.details['volume_confirm']}")

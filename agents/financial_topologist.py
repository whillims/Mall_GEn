"""
智能体 F：财务拓扑师 (Financial Topologist)
职责边界：将财务报表重构为元空间拓扑结构，识别数据背后的能量-物质-熵流动
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FinancialTopology:
    """财务拓扑结构"""
    expansion: float  # 扩张性 λ1
    robustness: float  # 稳健性 λ2
    entropy_efficiency: float  # 熵排效率 λ3
    
    def to_vector(self) -> np.ndarray:
        """转换为numpy向量"""
        return np.array([
            self.expansion,
            self.robustness,
            self.entropy_efficiency
        ])


@dataclass
class FinancialStatements:
    """财务报表数据"""
    operating_cash_flow: float  # 经营现金流
    investing_cash_flow: float  # 投资现金流
    financing_cash_flow: float  # 融资现金流
    revenue: float  # 营收
    gross_profit: float  # 毛利润
    net_profit: float  # 净利润
    total_assets: float  # 总资产
    total_liabilities: float  # 总负债
    roic: float  # ROIC
    wacc: float  # WACC
    inventory_turnover: float  # 存货周转率
    asset_impairment: float  # 资产减值


class FinancialTopologist:
    """财务拓扑师智能体"""
    
    def __init__(self, agent_id: str = "F", kappa_threshold: float = 0.40):
        self.agent_id = agent_id
        self.kappa_threshold = kappa_threshold  # 相变阈值
        self.history: List[Tuple[datetime, FinancialTopology, FinancialStatements]] = []
        self.confidence = 0.88  # 初始置信度
        
    def build_cash_flow_topology(
        self,
        statements: FinancialStatements
    ) -> Dict[str, float]:
        """
        构建资金流的拓扑网络
        
        Args:
            statements: 财务报表数据
            
        Returns:
            资金流拓扑指标字典
        """
        # 计算资金转化率
        cash_conversion_rate = abs(statements.investing_cash_flow) / max(
            abs(statements.operating_cash_flow), 1
        )
        
        # 现金流稳定性
        total_cash = abs(statements.operating_cash_flow) + \
                    abs(statements.investing_cash_flow) + \
                    abs(statements.financing_cash_flow)
        
        operating_ratio = abs(statements.operating_cash_flow) / max(total_cash, 1)
        
        return {
            "cash_conversion_rate": cash_conversion_rate,
            "operating_ratio": operating_ratio
        }
    
    def detect_phase_transition(
        self,
        current_metrics: Dict[str, float],
        historical_metrics: List[Dict[str, float]]
    ) -> Tuple[bool, float]:
        """
        识别收入/成本/费用的相变点
        
        Args:
            current_metrics: 当前指标
            historical_metrics: 历史指标列表
            
        Returns:
            (是否发生相变, 相变强度)
        """
        if len(historical_metrics) < 2:
            return False, 0.0
        
        # 计算当前与历史的偏差
        recent_avg = {
            key: np.mean([m.get(key, 0) for m in historical_metrics[-3:]])
            for key in current_metrics
        }
        
        max_deviation = 0.0
        for key in current_metrics:
            if key in recent_avg and recent_avg[key] != 0:
                deviation = abs(current_metrics[key] - recent_avg[key]) / recent_avg[key]
                max_deviation = max(max_deviation, deviation)
        
        # 偏差超过30%视为相变
        phase_transition = max_deviation > 0.30
        
        return phase_transition, max_deviation
    
    def monitor_entropy_increase(
        self,
        statements: FinancialStatements
    ) -> float:
        """
        监测物质占位的熵增信号
        
        Args:
            statements: 财务报表数据
            
        Returns:
            熵增分数 (0-1)，越高表示熵增越快
        """
        entropy_score = 0.0
        
        # 存货周转率下降 -> 熵增
        if statements.inventory_turnover < 3.0:
            entropy_score += 0.3
        elif statements.inventory_turnover < 5.0:
            entropy_score += 0.15
        
        # 资产减值增加 -> 熵增
        if statements.asset_impairment > 0.1 * statements.total_assets:
            entropy_score += 0.4
        elif statements.asset_impairment > 0.05 * statements.total_assets:
            entropy_score += 0.2
        
        return min(entropy_score, 1.0)
    
    def calculate_energy_flux(
        self,
        statements: FinancialStatements
    ) -> float:
        """
        计算价值创造的净能量通量 (ROIC-WACC spread)
        
        Args:
            statements: 财务报表数据
            
        Returns:
            净能量通量
        """
        return statements.roic - statements.wacc
    
    def financial_metacurvature(
        self,
        topology: FinancialTopology,
        prev_topology: FinancialTopology = None
    ) -> float:
        """
        财务元空间曲率
        
        κ = |∇²F| / (1 + |∇F|²)^(3/2)
        
        Args:
            topology: 当前财务拓扑
            prev_topology: 前一时刻财务拓扑
            
        Returns:
            曲率 κ
        """
        vec_current = topology.to_vector()
        
        if prev_topology is None:
            # 没有历史数据时，计算相对于基准的曲率
            vec_prev = np.array([0.5, 0.5, 0.5])  # 基准向量
        else:
            vec_prev = prev_topology.to_vector()
        
        # 计算一阶导数（梯度）
        grad = vec_current - vec_prev
        
        # 计算二阶导数（拉普拉斯）
        if len(self.history) >= 2:
            _, prev2_topology, _ = self.history[-2]
            vec_prev2 = prev2_topology.to_vector()
            laplacian = vec_current - 2 * vec_prev + vec_prev2
        else:
            laplacian = grad  # 数据不足时用一阶导数近似
        
        # 计算曲率
        norm_grad = np.linalg.norm(grad)
        norm_laplacian = np.linalg.norm(laplacian)
        
        denominator = (1 + norm_grad ** 2) ** (3/2)
        
        if denominator == 0:
            return 0.0
            
        kappa = norm_laplacian / denominator
        
        return kappa
    
    def generate_financial_topology(
        self,
        statements: FinancialStatements,
        cash_flow_topology: Dict[str, float],
        entropy_score: float,
        energy_flux: float
    ) -> FinancialTopology:
        """
        生成财务拓扑结构
        
        Args:
            statements: 财务报表数据
            cash_flow_topology: 资金流拓扑指标
            entropy_score: 熵增分数
            energy_flux: 净能量通量
            
        Returns:
            财务拓扑结构
        """
        # 扩张性 λ1：基于现金转化率和营收增长
        expansion = cash_flow_topology.get("cash_conversion_rate", 0.5)
        # 调整到0-1范围
        expansion = min(expansion, 1.0)
        if energy_flux > 0:
            expansion = max(expansion, 0.5 + energy_flux / 2)
        expansion = min(expansion, 1.0)
        
        # 稳健性 λ2：基于经营现金流比例和资产负债率
        operating_ratio = cash_flow_topology.get("operating_ratio", 0.5)
        debt_ratio = statements.total_liabilities / max(statements.total_assets, 1)
        robustness = operating_ratio * 0.6 + (1 - debt_ratio) * 0.4
        robustness = max(0, min(robustness, 1.0))
        
        # 熵排效率 λ3：基于熵增分数和存货周转率
        entropy_efficiency = 1 - entropy_score
        if statements.inventory_turnover > 5:
            entropy_efficiency += 0.1
        entropy_efficiency = max(0, min(entropy_efficiency, 1.0))
        
        return FinancialTopology(
            expansion=expansion,
            robustness=robustness,
            entropy_efficiency=entropy_efficiency
        )
    
    def analyze(
        self,
        statements: FinancialStatements,
        historical_statements: List[FinancialStatements] = None,
        timestamp: datetime = None
    ) -> Dict:
        """
        完整分析流程
        
        Args:
            statements: 当前财务报表数据
            historical_statements: 历史财务报表数据
            timestamp: 时间戳
            
        Returns:
            分析结果字典
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 1. 构建资金流拓扑
        cash_flow_topology = self.build_cash_flow_topology(statements)
        
        # 2. 检测相变（需要历史数据）
        phase_transition = False
        phase_transition_strength = 0.0
        if historical_statements:
            # 简单起见，用营收、毛利率等构建指标
            current_metrics = {
                "revenue": statements.revenue,
                "gross_margin": statements.gross_profit / max(statements.revenue, 1)
            }
            historical_metrics = [
                {
                    "revenue": hs.revenue,
                    "gross_margin": hs.gross_profit / max(hs.revenue, 1)
                }
                for hs in historical_statements
            ]
            phase_transition, phase_transition_strength = self.detect_phase_transition(
                current_metrics, historical_metrics
            )
        
        # 3. 监测熵增
        entropy_score = self.monitor_entropy_increase(statements)
        
        # 4. 计算净能量通量
        energy_flux = self.calculate_energy_flux(statements)
        
        # 5. 生成财务拓扑
        topology = self.generate_financial_topology(
            statements, cash_flow_topology, entropy_score, energy_flux
        )
        
        # 6. 计算元空间曲率
        prev_topology = None
        if self.history:
            _, prev_topology, _ = self.history[-1]
        kappa = self.financial_metacurvature(topology, prev_topology)
        
        # 7. 判断是否超过相变阈值
        phase_transition_warning = kappa > self.kappa_threshold
        
        # 保存历史
        self.history.append((timestamp, topology, statements))
        
        # 构建结果
        result = {
            "agent_id": self.agent_id,
            "timestamp": timestamp.isoformat(),
            "excitation_mode": "季报激发",
            "vector_payload": topology.to_vector().tolist(),
            "confidence": self.confidence,
            "entropy_flag": entropy_score > 0.5,
            "details": {
                "financial_topology": topology,
                "cash_flow_topology": cash_flow_topology,
                "entropy_score": entropy_score,
                "energy_flux": energy_flux,
                "metacurvature": kappa,
                "kappa_threshold": self.kappa_threshold,
                "phase_transition_warning": phase_transition_warning,
                "phase_transition_strength": phase_transition_strength
            }
        }
        
        return result

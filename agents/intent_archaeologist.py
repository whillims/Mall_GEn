"""
智能体 I：意图考古学家 (Intent Archaeologist)
职责边界：挖掘企业投资意图的历史轨迹，识别意图的连续性与断裂点
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class IntentVector:
    """意图矢量"""
    product_innovation: float  # 产品革新强度 i1
    operational_efficiency: float  # 运营效率导向 i2
    capacity_expansion: float  # 产能扩张系数 i3
    market_investment: float  # 市场投入强度 i4
    intent_coherence: float  # 意图自洽指数 i5 (0-1)
    
    def to_vector(self) -> np.ndarray:
        """转换为numpy向量"""
        return np.array([
            self.product_innovation,
            self.operational_efficiency,
            self.capacity_expansion,
            self.market_investment,
            self.intent_coherence
        ])


class IntentArchaeologist:
    """意图考古学家智能体"""
    
    def __init__(self, agent_id: str = "I"):
        self.agent_id = agent_id
        self.intent_history: List[Tuple[datetime, IntentVector]] = []
        self.confidence = 0.91  # 初始置信度
        
    def extract_keyword_frequency(self, mda_text: str, keywords: List[str]) -> Dict[str, float]:
        """
        从MD&A文本中提取关键词频率
        
        Args:
            mda_text: 管理层讨论与分析文本
            keywords: 关键词列表
            
        Returns:
            关键词频率字典
        """
        word_counts = {}
        total_words = len(mda_text.split())
        
        for keyword in keywords:
            count = mda_text.lower().count(keyword.lower())
            frequency = count / max(total_words, 1)
            word_counts[keyword] = frequency
            
        return word_counts
    
    def calculate_capital_expenditure_deviation(
        self,
        promised_investment: float,
        actual_investment: float
    ) -> float:
        """
        计算资本开支承诺 vs 实际执行偏差
        
        Args:
            promised_investment: 承诺投资额
            actual_investment: 实际投资额
            
        Returns:
            偏差率 (实际/承诺)
        """
        if promised_investment <= 0:
            return 0.0
        return actual_investment / promised_investment
    
    def detect_strategic_shift(self, 
                             historical_language_modes: List[str],
                             current_language_mode: str
                            ) -> bool:
        """
        识别战略转向的元空间跃迁
        
        Args:
            historical_language_modes: 历史语言模式列表
            current_language_mode: 当前语言模式
            
        Returns:
            是否发生战略转向
        """
        if not historical_language_modes:
            return False
        
        # 简单判断：如果当前模式与最近3个历史模式都不同，视为战略转向
        recent_modes = historical_language_modes[-3:]
        return all(mode != current_language_mode for mode in recent_modes)
    
    def generate_intent_vector(
        self,
        keyword_frequencies: Dict[str, float],
        capex_deviation: float,
        strategic_shift_flag: bool,
        language_coherence_score: float
    ) -> IntentVector:
        """
        生成意图矢量
        
        Args:
            keyword_frequencies: 关键词频率字典
            capex_deviation: 资本开支偏差率
            strategic_shift_flag: 是否战略转向
            language_coherence_score: 语言自洽分数 (0-1)
            
        Returns:
            意图矢量
        """
        # 基于关键词频率计算各维度
        product_innovation = keyword_frequencies.get("创新", 0.0) * 5 + \
                            keyword_frequencies.get("研发", 0.0) * 5 + \
                            keyword_frequencies.get("技术", 0.0) * 3
        product_innovation = min(product_innovation, 1.0)
        
        operational_efficiency = keyword_frequencies.get("效率", 0.0) * 5 + \
                               keyword_frequencies.get("成本", 0.0) * 3
        operational_efficiency = min(operational_efficiency, 1.0)
        
        capacity_expansion = keyword_frequencies.get("产能", 0.0) * 5 + \
                            keyword_frequencies.get("扩张", 0.0) * 4
        capacity_expansion = min(capacity_expansion, 1.0)
        
        market_investment = keyword_frequencies.get("市场", 0.0) * 4 + \
                           keyword_frequencies.get("销售", 0.0) * 3
        market_investment = min(market_investment, 1.0)
        
        # 计算意图自洽指数
        intent_coherence = 0.7  # 基础分
        if capex_deviation > 0.8:
            intent_coherence += 0.2
        if not strategic_shift_flag:
            intent_coherence += 0.1
        intent_coherence *= language_coherence_score
        intent_coherence = min(intent_coherence, 1.0)
        
        return IntentVector(
            product_innovation=product_innovation,
            operational_efficiency=operational_efficiency,
            capacity_expansion=capacity_expansion,
            market_investment=market_investment,
            intent_coherence=intent_coherence
        )
    
    def intent_continuity_function(self, 
                                  vec_current: np.ndarray, 
                                  vec_prev: np.ndarray
                                 ) -> float:
        """
        意图连续性函数
        
        C(t) = (vec_current · vec_prev) / (|vec_current| |vec_prev|)
        
        Args:
            vec_current: 当前意图矢量
            vec_prev: 前一时刻意图矢量
            
        Returns:
            连续性分数 (0-1)，越接近1表示越连续
        """
        dot_product = np.dot(vec_current, vec_prev)
        norm_current = np.linalg.norm(vec_current)
        norm_prev = np.linalg.norm(vec_prev)
        
        if norm_current == 0 or norm_prev == 0:
            return 0.0
            
        return dot_product / (norm_current * norm_prev)
    
    def analyze(
        self,
        mda_text: str,
        promised_investment: float,
        actual_investment: float,
        historical_language_modes: List[str],
        current_language_mode: str,
        language_coherence_score: float = 0.95,
        timestamp: datetime = None
    ) -> Dict:
        """
        完整分析流程
        
        Args:
            mda_text: 管理层讨论与分析文本
            promised_investment: 承诺投资额
            actual_investment: 实际投资额
            historical_language_modes: 历史语言模式列表
            current_language_mode: 当前语言模式
            language_coherence_score: 语言自洽分数
            timestamp: 时间戳
            
        Returns:
            分析结果字典
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 1. 提取关键词频率
        keywords = ["创新", "研发", "技术", "效率", "成本", "产能", "扩张", "市场", "销售"]
        keyword_frequencies = self.extract_keyword_frequency(mda_text, keywords)
        
        # 2. 计算资本开支偏差
        capex_deviation = self.calculate_capital_expenditure_deviation(
            promised_investment, actual_investment
        )
        
        # 3. 检测战略转向
        strategic_shift = self.detect_strategic_shift(
            historical_language_modes, current_language_mode
        )
        
        # 4. 生成意图矢量
        intent_vector = self.generate_intent_vector(
            keyword_frequencies,
            capex_deviation,
            strategic_shift,
            language_coherence_score
        )
        
        # 5. 计算连续性（如果有历史数据）
        continuity_score = None
        if self.intent_history:
            _, prev_vector = self.intent_history[-1]
            continuity_score = self.intent_continuity_function(
                intent_vector.to_vector(),
                prev_vector.to_vector()
            )
        
        # 保存历史
        self.intent_history.append((timestamp, intent_vector))
        
        # 构建结果
        result = {
            "agent_id": self.agent_id,
            "timestamp": timestamp.isoformat(),
            "excitation_mode": "季报激发",
            "vector_payload": intent_vector.to_vector().tolist(),
            "confidence": self.confidence,
            "entropy_flag": strategic_shift,
            "details": {
                "keyword_frequencies": keyword_frequencies,
                "capex_deviation": capex_deviation,
                "strategic_shift": strategic_shift,
                "intent_vector": intent_vector,
                "continuity_score": continuity_score
            }
        }
        
        return result

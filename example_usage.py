"""
差异化联邦沉思引擎 · 使用示例
演示如何使用意图考古学家和财务拓扑师智能体
"""

from datetime import datetime
from agents import IntentArchaeologist, FinancialTopologist
from agents.financial_topologist import FinancialStatements


def example_intent_archaeologist():
    """意图考古学家使用示例"""
    print("=" * 60)
    print("智能体 I：意图考古学家 (Intent Archaeologist) 示例")
    print("=" * 60)
    
    # 创建智能体
    agent = IntentArchaeologist()
    
    # 示例数据：量子智芯2026年Q1季报
    mda_text_2025q4 = """
    2025年第四季度，公司继续夯实基础，稳步推进各项业务。
    我们注重效率提升，优化成本结构，为未来发展奠定坚实基础。
    """
    
    mda_text_2026q1 = """
    2026年第一季度，公司进入全面领跑阶段，大力投入创新和研发。
    我们积极扩张产能，拓展市场，推动技术升级。
    研发费用同比增长67%，占营收比例提升至28%。
    同时公告拟投资15亿元建设第三代AI芯片生产线。
    """
    
    # 第一次分析（2025Q4）
    print("\n[1] 分析2025年Q4数据：")
    result_2025q4 = agent.analyze(
        mda_text=mda_text_2025q4,
        promised_investment=10.0,
        actual_investment=9.5,
        historical_language_modes=["稳健", "稳健"],
        current_language_mode="夯实基础",
        language_coherence_score=0.95,
        timestamp=datetime(2025, 10, 30)
    )
    
    intent_vec_2025q4 = result_2025q4["details"]["intent_vector"]
    print(f"  意图矢量: [{intent_vec_2025q4.product_innovation:.2f}, "
          f"{intent_vec_2025q4.operational_efficiency:.2f}, "
          f"{intent_vec_2025q4.capacity_expansion:.2f}, "
          f"{intent_vec_2025q4.market_investment:.2f}, "
          f"{intent_vec_2025q4.intent_coherence:.2f}]")
    print(f"  战略转向: {result_2025q4['details']['strategic_shift']}")
    
    # 第二次分析（2026Q1）
    print("\n[2] 分析2026年Q1数据：")
    result_2026q1 = agent.analyze(
        mda_text=mda_text_2026q1,
        promised_investment=15.0,
        actual_investment=14.8,
        historical_language_modes=["稳健", "稳健", "夯实基础"],
        current_language_mode="全面领跑",
        language_coherence_score=0.92,
        timestamp=datetime(2026, 4, 28)
    )
    
    intent_vec_2026q1 = result_2026q1["details"]["intent_vector"]
    print(f"  意图矢量: [{intent_vec_2026q1.product_innovation:.2f}, "
          f"{intent_vec_2026q1.operational_efficiency:.2f}, "
          f"{intent_vec_2026q1.capacity_expansion:.2f}, "
          f"{intent_vec_2026q1.market_investment:.2f}, "
          f"{intent_vec_2026q1.intent_coherence:.2f}]")
    print(f"  连续性分数: {result_2026q1['details']['continuity_score']:.4f}")
    print(f"  战略转向: {result_2026q1['details']['strategic_shift']}")
    
    # 广播格式输出
    print("\n[3] 联邦广播格式：")
    broadcast_data = {
        "agent_id": result_2026q1["agent_id"],
        "timestamp": result_2026q1["timestamp"],
        "excitation_mode": result_2026q1["excitation_mode"],
        "vector_payload": result_2026q1["vector_payload"],
        "confidence": result_2026q1["confidence"],
        "entropy_flag": result_2026q1["entropy_flag"]
    }
    print(f"  {broadcast_data}")


def example_financial_topologist():
    """财务拓扑师使用示例"""
    print("\n" + "=" * 60)
    print("智能体 F：财务拓扑师 (Financial Topologist) 示例")
    print("=" * 60)
    
    # 创建智能体
    agent = FinancialTopologist(kappa_threshold=0.40)
    
    # 示例数据：量子智芯历史财报
    statements_2025q4 = FinancialStatements(
        operating_cash_flow=8.5,
        investing_cash_flow=-7.2,
        financing_cash_flow=1.2,
        revenue=28.5,
        gross_profit=15.2,
        net_profit=4.8,
        total_assets=85.0,
        total_liabilities=32.0,
        roic=0.115,
        wacc=0.065,
        inventory_turnover=6.2,
        asset_impairment=0.2
    )
    
    statements_2026q1 = FinancialStatements(
        operating_cash_flow=10.2,
        investing_cash_flow=-12.3,
        financing_cash_flow=3.5,
        revenue=40.5,
        gross_profit=23.0,
        net_profit=7.2,
        total_assets=98.0,
        total_liabilities=38.0,
        roic=0.128,
        wacc=0.062,
        inventory_turnover=5.8,
        asset_impairment=0.3
    )
    
    # 第一次分析（2025Q4）
    print("\n[1] 分析2025年Q4财务数据：")
    result_2025q4 = agent.analyze(
        statements=statements_2025q4,
        historical_statements=None,
        timestamp=datetime(2025, 10, 30)
    )
    
    topology_2025q4 = result_2025q4["details"]["financial_topology"]
    print(f"  财务拓扑: λ1={topology_2025q4.expansion:.2f}, "
          f"λ2={topology_2025q4.robustness:.2f}, "
          f"λ3={topology_2025q4.entropy_efficiency:.2f}")
    print(f"  元空间曲率: κ={result_2025q4['details']['metacurvature']:.4f}")
    print(f"  净能量通量: {result_2025q4['details']['energy_flux']:.4f}")
    
    # 第二次分析（2026Q1）
    print("\n[2] 分析2026年Q1财务数据：")
    result_2026q1 = agent.analyze(
        statements=statements_2026q1,
        historical_statements=[statements_2025q4],
        timestamp=datetime(2026, 4, 28)
    )
    
    topology_2026q1 = result_2026q1["details"]["financial_topology"]
    print(f"  财务拓扑: λ1={topology_2026q1.expansion:.2f}, "
          f"λ2={topology_2026q1.robustness:.2f}, "
          f"λ3={topology_2026q1.entropy_efficiency:.2f}")
    print(f"  元空间曲率: κ={result_2026q1['details']['metacurvature']:.4f}")
    print(f"  相变预警: {result_2026q1['details']['phase_transition_warning']}")
    print(f"  净能量通量: {result_2026q1['details']['energy_flux']:.4f}")
    
    # 广播格式输出
    print("\n[3] 联邦广播格式：")
    broadcast_data = {
        "agent_id": result_2026q1["agent_id"],
        "timestamp": result_2026q1["timestamp"],
        "excitation_mode": result_2026q1["excitation_mode"],
        "vector_payload": result_2026q1["vector_payload"],
        "confidence": result_2026q1["confidence"],
        "entropy_flag": result_2026q1["entropy_flag"]
    }
    print(f"  {broadcast_data}")


def main():
    """主函数"""
    example_intent_archaeologist()
    example_financial_topologist()
    print("\n" + "=" * 60)
    print("示例执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

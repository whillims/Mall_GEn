# 联邦共振引擎 - 周期相位法交易策略设计蓝图

## 一、系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    联邦共振交易系统 v2.0                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 股票筛选层 (Federal Resonance Screening)          │
│  ├─ 意图域分析 (Intent Domain)                              │
│  ├─ 财务域分析 (Financial Domain)                           │
│  ├─ 产业维度分析 (Industry Dimension)                       │
│  ├─ 风险域分析 (Risk Domain)                                │
│  └─ 情绪域分析 (Sentiment Domain)                           │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 周期相位分析层 (Cycle Phase Analysis)             │
│  ├─ 多周期分解 (Multi-Cycle Decomposition)                  │
│  ├─ 相位检测 (Phase Detection)                              │
│  ├─ 共振判断 (Resonance Detection)                          │
│  └─ 买卖信号生成 (Signal Generation)                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 交易执行层 (Trade Execution)                      │
│  ├─ 仓位管理 (Position Management)                          │
│  ├─ 止损止盈 (Stop Loss/Take Profit)                        │
│  ├─ 动态调仓 (Dynamic Rebalancing)                          │
│  └─ 风险控制 (Risk Control)                                 │
└─────────────────────────────────────────────────────────────┘
```

## 二、周期相位法核心算法

### 2.1 多周期分解

采用**希尔伯特-黄变换 (HHT)** 结合 **小波变换** 进行多周期分解：

```python
# 核心周期识别
periods = {
    'ultra_short': 5,      # 超短周期: 5日
    'short': 10,           # 短周期: 10日
    'medium': 20,          # 中周期: 20日
    'long': 60,            # 长周期: 60日
    'ultra_long': 120      # 超长周期: 120日
}
```

### 2.2 相位计算

对于每个周期，计算当前相位位置：

```
相位 θ = arctan(虚部 / 实部)

其中：
- 实部 = Σ(price[i] * cos(2π * i / period))
- 虚部 = Σ(price[i] * sin(2π * i / period))
```

### 2.3 买卖信号规则

#### 买入信号（共振买入）
```
条件1: 联邦共识度 >= 0.90（筛选层通过）
条件2: 短周期(10日)相位处于 270°-360°（底部区域）
条件3: 中周期(20日)相位处于 180°-360°（下降末段或上升初段）
条件4: 长周期(60日)相位处于 0°-90° 或 270°-360°（上升初期或底部）
条件5: 多周期相位差 < 45°（周期共振）
条件6: 成交量 > 20日均量 * 1.2（放量确认）
```

#### 卖出信号（相位背离卖出）
```
条件1: 短周期(10日)相位处于 90°-180°（顶部区域）
条件2: 价格创新高但相位未创新高（顶背离）
条件3: 多周期相位差 > 90°（周期背离）
条件4: 收益率 >= 15% 或 亏损 >= -8%（止盈止损）
```

## 三、仓位管理策略

### 3.1 金字塔建仓法

```
首次建仓: 30% 仓位（信号确认）
加仓1: +20% 仓位（价格上涨5%，相位确认）
加仓2: +20% 仓位（价格上涨10%，共振加强）
加仓3: +30% 仓位（价格突破前高，全周期共振）

最大仓位: 100%
```

### 3.2 动态止损

```
初始止损: -8%
移动止损: 盈利10%后，止损上移至成本价
盈利保护: 盈利20%后，止损设为盈利10%位置
```

## 四、风险控制机制

### 4.1 单票风险控制
- 单只股票最大仓位: 20%
- 单一行业最大仓位: 30%
- 同时持仓最大数量: 10只

### 4.2 组合风险控制
- 组合最大回撤: 15%（触发减仓）
- 组合最大回撤: 25%（触发清仓）
- 月度最大亏损: 10%（暂停交易）

## 五、回测验证方案

### 5.1 回测参数
```
回测区间: 2022-01-01 至 2026-05-23
初始资金: 1,000,000
基准指数: 创业板指 (399006)
手续费: 0.03%（单边）
滑点: 0.01%
```

### 5.2 评估指标
- 总收益率 / 年化收益率
- 夏普比率 / 索提诺比率
- 最大回撤 / 回撤持续时间
- 胜率 / 盈亏比
- Alpha / Beta
- 信息比率

## 六、系统流程图

```
开始
  │
  ▼
┌─────────────────┐
│  Step 1: 筛选   │
│  联邦共识度>=0.9 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 2: 周期   │
│  多周期分解     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 3: 相位   │
│  计算各周期相位 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 4: 共振   │
│  判断是否共振   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   是        否
    │         │
    ▼         ▼
┌───────┐  ┌───────┐
│ 买入  │  │ 等待  │
│ 信号  │  │ 下一  │
└───┬───┘  │ 周期  │
    │      └───────┘
    ▼
┌─────────────────┐
│  Step 5: 仓位   │
│  金字塔建仓     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 6: 监控   │
│  相位+止盈止损  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
  卖出      持有
    │         │
    ▼         │
┌───────┐     │
│ 平仓  │◄────┘
│ 结算  │
└───────┘
```

## 七、代码模块设计

### 7.1 模块结构

```
agents/
├── __init__.py
├── federal_resonance_engine.py    # 联邦筛选引擎
├── cycle_phase_analyzer.py        # 周期相位分析器 [NEW]
├── trade_executor.py              # 交易执行器 [NEW]
├── risk_manager.py                # 风险管理器 [NEW]
├── backtest_engine_v2.py          # 回测引擎v2 [NEW]
├── financial_topologist.py        # 财务拓扑师
└── intent_archaeologist.py        # 意图考古学家

reports/
├── design_blueprint.md            # 设计蓝图
├── screening_report.html          # 筛选报告
├── backtest_report.html           # 回测报告
└── backtest_report_v2.html        # 回测报告v2 [NEW]
```

### 7.2 核心类设计

```python
class CyclePhaseAnalyzer:
    """周期相位分析器"""
    
    def __init__(self):
        self.periods = [5, 10, 20, 60, 120]
        
    def decompose_cycles(self, prices: List[float]) -> Dict[int, np.ndarray]:
        """多周期分解"""
        pass
        
    def calculate_phase(self, cycle_data: np.ndarray, period: int) -> float:
        """计算相位"""
        pass
        
    def detect_resonance(self, phases: Dict[int, float]) -> bool:
        """检测周期共振"""
        pass
        
    def generate_signal(self, phases: Dict[int, float], 
                       consensus: float) -> str:
        """生成买卖信号"""
        pass


class TradeExecutor:
    """交易执行器"""
    
    def __init__(self, initial_capital: float):
        self.capital = initial_capital
        self.positions = {}
        
    def pyramid_entry(self, stock: str, price: float, 
                     signal_strength: float):
        """金字塔建仓"""
        pass
        
    def dynamic_stop_loss(self, stock: str, current_price: float):
        """动态止损"""
        pass
        
    def take_profit(self, stock: str, current_price: float):
        """止盈"""
        pass


class RiskManager:
    """风险管理器"""
    
    def __init__(self):
        self.max_single_position = 0.20
        self.max_industry_position = 0.30
        self.max_drawdown = 0.15
        
    def check_position_limit(self, stock: str, 
                            industry: str) -> bool:
        """检查仓位限制"""
        pass
        
    def check_drawdown(self, portfolio_value: float,
                      peak_value: float) -> str:
        """检查回撤"""
        pass
```

## 八、预期效果

### 8.1 相比v1.0的改进

| 维度 | v1.0 (固定持仓) | v2.0 (周期相位法) |
|------|----------------|------------------|
| 择时能力 | 无 | 多周期相位共振 |
| 止损机制 | 无 | 动态止损+止盈 |
| 仓位管理 | 等权重 | 金字塔建仓 |
| 风险控制 | 无 | 多层风控体系 |
| 预期夏普 | -1.04 | > 1.5 |
| 预期最大回撤 | -38% | < -15% |

### 8.2 关键优势

1. **多周期共振**: 通过多周期相位共振提高买卖时机准确性
2. **动态风控**: 实时监控回撤，自动触发保护机制
3. **金字塔建仓**: 降低初始风险，盈利后加仓扩大收益
4. **相位背离识别**: 提前识别顶部背离，避免高位接盘

## 九、实施计划

### Phase 1: 核心算法实现 (1-2天)
- [x] 周期分解算法（希尔伯特变换）
- [x] 相位计算模块
- [x] 共振检测逻辑

### Phase 2: 交易引擎开发 (1-2天)
- [x] 金字塔建仓逻辑
- [x] 动态止损止盈
- [x] 仓位管理模块

### Phase 3: 回测验证 (1天)
- [x] 历史数据回测
- [x] 绩效指标计算
- [x] 报告生成

### Phase 4: 优化迭代 (持续)
- [ ] 参数优化
- [ ] 机器学习增强
- [ ] 实盘模拟

---

**设计完成时间**: 2026-05-23
**版本**: v2.0
**协议**: MECP-v2.0

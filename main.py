import pandas as pd
import numpy as np
import argparse
import os
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from ml_strategy import MLStrategy
from backtest_engine import BacktestEngine
from visualization import Visualizer
from report_generator import ReportGenerator

def run_pipeline(symbol: str = 'AAPL',
                 start_date: str = '2020-01-01',
                 end_date: str = '2024-01-01',
                 model_type: str = 'random_forest',
                 initial_capital: float = 100000.0,
                 position_size: float = 0.2,
                 stop_loss: float = 0.05,
                 take_profit: float = 0.10,
                 lookahead: int = 5,
                 threshold: float = 0.55,
                 save_results: bool = True,
                 generate_html_report: bool = True):

    print("="*60)
    print("机器学习量化交易策略回测系统")
    print("="*60)
    print(f"\n策略参数:")
    print(f"  股票代码: {symbol}")
    print(f"  模型类型: {model_type}")
    print(f"  预测周期: {lookahead}天")
    print(f"  置信阈值: {threshold}")
    print(f"  仓位比例: {position_size}")
    print(f"  止损比例: {stop_loss}")
    print(f"  止盈比例: {take_profit}")

    data_loader = DataLoader(symbol, start_date, end_date)
    data = data_loader.load_data()

    feature_engineer = FeatureEngineer(data)
    data_with_features = feature_engineer.prepare_features(lookahead=lookahead)
    feature_columns = feature_engineer.get_feature_columns()

    print(f"\n特征数量: {len(feature_columns)}")
    print(f"样本数量: {len(data_with_features)}")

    train_data, test_data = data_loader.get_train_test_split(train_ratio=0.8)

    train_fe = FeatureEngineer(train_data)
    train_processed = train_fe.prepare_features(lookahead=lookahead)

    test_fe = FeatureEngineer(test_data)
    test_processed = test_fe.prepare_features(lookahead=lookahead)

    X_train = train_processed[feature_columns]
    y_train = train_processed['target']
    X_test = test_processed[feature_columns]
    y_test = test_processed['target']

    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    strategy = MLStrategy(model_type=model_type)
    train_results = strategy.train(X_train, y_train, feature_columns)

    proba = strategy.predict_proba(X_test)

    class_labels = strategy.model.classes_
    predictions = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        probs = proba[i]
        max_prob_idx = np.argmax(probs)
        max_prob = probs[max_prob_idx]
        pred_class = class_labels[max_prob_idx]

        if max_prob >= threshold:
            predictions[i] = pred_class
        else:
            predictions[i] = 0

    print(f"\n测试集预测分布:")
    print(f"  买入信号 (1): {np.sum(predictions == 1)}")
    print(f"  持有信号 (0): {np.sum(predictions == 0)}")
    print(f"  卖出信号 (-1): {np.sum(predictions == -1)}")

    backtest = BacktestEngine(
        initial_capital=initial_capital,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    metrics = backtest.run_backtest(test_processed, predictions)

    metrics['equity_curve'] = backtest.equity_curve
    metrics['dates'] = backtest.dates
    metrics['initial_capital'] = initial_capital

    visualizer = Visualizer()

    if save_results:
        os.makedirs('results', exist_ok=True)
        visualizer.plot_backtest_results(
            test_processed, metrics, predictions,
            save_path=f'results/{symbol}_backtest.png'
        )
        visualizer.plot_feature_importance(
            train_results['feature_importance'],
            save_path=f'results/{symbol}_features.png'
        )
        visualizer.plot_trade_distribution(
            metrics,
            save_path=f'results/{symbol}_trades.png'
        )
        strategy.save_model(f'results/{symbol}_model.pkl')
    else:
        visualizer.plot_backtest_results(test_processed, metrics, predictions)
        visualizer.plot_feature_importance(train_results['feature_importance'])
        visualizer.plot_trade_distribution(metrics)

    if generate_html_report:
        report_gen = ReportGenerator(
            metrics=metrics,
            symbol=symbol,
            model_type=model_type,
            start_date=start_date,
            end_date=end_date
        )
        report_path = report_gen.save_report(f"{symbol}_report.html")
        print(f"\nHTML报告已生成: {report_path}")

    return metrics, strategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='机器学习量化交易策略回测')
    parser.add_argument('--symbol', type=str, default='AAPL', help='股票代码')
    parser.add_argument('--start', type=str, default='2020-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2024-01-01', help='结束日期')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting'],
                       help='模型类型')
    parser.add_argument('--capital', type=float, default=100000.0, help='初始资金')
    parser.add_argument('--position', type=float, default=0.2, help='仓位比例')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='止损比例')
    parser.add_argument('--take-profit', type=float, default=0.10, help='止盈比例')
    parser.add_argument('--lookahead', type=int, default=5, help='预测周期(天)')
    parser.add_argument('--threshold', type=float, default=0.55, help='置信度阈值')
    parser.add_argument('--no-save', action='store_true', help='不保存结果')
    parser.add_argument('--no-report', action='store_true', help='不生成HTML报告')

    args = parser.parse_args()

    run_pipeline(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        model_type=args.model,
        initial_capital=args.capital,
        position_size=args.position,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        lookahead=args.lookahead,
        threshold=args.threshold,
        save_results=not args.no_save,
        generate_html_report=not args.no_report
    )

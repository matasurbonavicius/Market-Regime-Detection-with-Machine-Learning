"""
Simple backtester with proper timing.

Key: Prediction at T -> Position entered at T+1 -> P&L realized at T+2
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_values: npt.NDArray[np.floating]
    benchmark_values: npt.NDArray[np.floating]
    dates: npt.NDArray[np.datetime64]

    strategy_return: float
    benchmark_return: float
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_max_dd: float
    benchmark_max_dd: float


def calculate_max_drawdown(values: npt.NDArray[np.floating]) -> float:
    """Calculate maximum drawdown from equity curve."""
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return float(np.max(drawdown))


def calculate_sharpe(values: npt.NDArray[np.floating], periods_per_year: float = 52) -> float:
    """Calculate annualized Sharpe ratio (assuming 0 risk-free rate)."""
    returns = np.diff(values) / values[:-1]
    if np.std(returns) == 0:
        return 0.0
    sharpe = np.mean(returns) / np.std(returns)
    return float(sharpe * np.sqrt(periods_per_year))


def run_backtest(dates: npt.NDArray[np.datetime64],
                 prices: npt.NDArray[np.floating],
                 predictions: npt.NDArray[np.integer],
                 position_map: Optional[Dict[int, float]] = None,
                 initial_capital: float = 10000) -> BacktestResult:
    """
    Run backtest with proper timing.

    Timing:
    - At close of period T, we have features and make prediction
    - Position is entered at close of T (or equivalently, open of T+1)
    - P&L is realized based on return from T to T+1

    So: predictions[i] determines exposure to returns[i+1]

    Args:
        dates: Array of dates
        prices: Array of close prices
        predictions: Array of predicted states (0, 1, 2)
        position_map: Dict mapping state -> position size (-1 to 1)
        initial_capital: Starting capital

    Returns:
        BacktestResult with equity curves and metrics
    """

    if position_map is None:
        # Default: 0=bull(long), 1=neutral(flat), 2=bear(short)
        position_map = {0: 2.0, 1: 1, 2: 0}

    # Calculate returns
    returns = np.diff(prices) / prices[:-1]

    # Strategy equity curve
    # predictions[i] -> position for return from i to i+1
    # So we use predictions[:-1] aligned with returns
    strategy_value = initial_capital
    strategy_values_list: list[float] = [strategy_value]

    for i in range(len(returns)):
        if i < len(predictions):
            position = position_map.get(int(predictions[i]), 0)
        else:
            position = 0

        strategy_value *= (1 + position * returns[i])
        strategy_values_list.append(strategy_value)

    strategy_values = np.array(strategy_values_list)

    # Benchmark (buy and hold)
    benchmark_values = initial_capital * prices / prices[0]

    # Calculate metrics
    result = BacktestResult(
        strategy_values=strategy_values,
        benchmark_values=benchmark_values,
        dates=dates,

        strategy_return=float((strategy_values[-1] / strategy_values[0]) - 1),
        benchmark_return=float((benchmark_values[-1] / benchmark_values[0]) - 1),

        strategy_sharpe=calculate_sharpe(strategy_values),
        benchmark_sharpe=calculate_sharpe(benchmark_values),

        strategy_max_dd=calculate_max_drawdown(strategy_values),
        benchmark_max_dd=calculate_max_drawdown(benchmark_values),
    )

    return result


def print_results(result: BacktestResult) -> None:
    """Print backtest results."""

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    print(f"\n{'Metric':<25} {'Strategy':>12} {'Benchmark':>12}")
    print("-" * 50)
    print(f"{'Total Return':<25} {result.strategy_return:>11.2%} {result.benchmark_return:>11.2%}")
    print(f"{'Sharpe Ratio (ann.)':<25} {result.strategy_sharpe:>12.2f} {result.benchmark_sharpe:>12.2f}")
    print(f"{'Max Drawdown':<25} {result.strategy_max_dd:>11.2%} {result.benchmark_max_dd:>11.2%}")
    print("=" * 50)


def plot_results(result: BacktestResult, title: str = "Backtest Results") -> None:
    """Plot equity curves."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # Equity curves
    ax1 = axes[0]
    ax1.plot(result.dates, result.strategy_values, label='Strategy', linewidth=1.5)
    ax1.plot(result.dates, result.benchmark_values, label='Buy & Hold', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    strategy_dd = (np.maximum.accumulate(result.strategy_values) - result.strategy_values) / np.maximum.accumulate(result.strategy_values)
    benchmark_dd = (np.maximum.accumulate(result.benchmark_values) - result.benchmark_values) / np.maximum.accumulate(result.benchmark_values)

    ax2.fill_between(result.dates, strategy_dd, alpha=0.5, label='Strategy DD')
    ax2.fill_between(result.dates, benchmark_dd, alpha=0.5, label='Benchmark DD')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    n = 100
    dates = np.arange(n, dtype=np.datetime64)
    prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    predictions = np.random.randint(0, 3, n)

    result = run_backtest(dates, prices, predictions)
    print_results(result)

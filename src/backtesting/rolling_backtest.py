"""
Rolling backtest driver
-------------------------
Phase 5's in-sample breach check was tautological (VaR estimated on the
same data it was tested against). This is the real, out-of-sample test:

For each day t (after an initial training window):
    1. Estimate VaR using only data from [t-window, t-1]  (no lookahead)
    2. Compare that forecast against the ACTUAL return on day t
    3. Record a violation if actual loss > forecasted VaR

The resulting violation sequence is genuinely out-of-sample and is what
Kupiec/Christoffersen should be run against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_var_backtest(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns:
        actual_return, forecast_var (positive loss fraction), violation (bool)
    covering every day from `window` onward.
    """
    returns = portfolio_returns.values
    dates = portfolio_returns.index
    n = len(returns)

    records = []
    for t in range(window, n):
        train = returns[t - window:t]  # strictly past data only
        alpha = 1 - confidence
        var_forecast = max(-np.percentile(train, alpha * 100), 0.0)

        actual = returns[t]
        violation = actual < -var_forecast

        records.append({
            "date": dates[t],
            "actual_return": actual,
            "forecast_var": var_forecast,
            "violation": violation,
        })

    return pd.DataFrame(records).set_index("date")


if __name__ == "__main__":
    from src.data.data_loader import MarketDataLoader
    from src.data.returns_processor import ReturnsProcessor
    from src.portfolio.portfolio import PortfolioManager
    from src.backtesting.kupiec import KupiecTest
    from src.backtesting.christoffersen import ChristoffersenTest

    tickers = ["SPY", "QQQ", "GLD", "TLT", "USO"]
    weights_dict = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}

    prices = MarketDataLoader(tickers, start="2019-01-01").load(mode="synthetic")
    returns = ReturnsProcessor(prices).returns
    pm = PortfolioManager(returns, weights_dict, initial_value=1_000_000)

    print("Running rolling out-of-sample backtest (252-day window, 95% VaR)...")
    backtest = rolling_var_backtest(pm.portfolio_returns, confidence=0.95, window=252)
    print(f"Out-of-sample days tested: {len(backtest)}")
    print(f"Violations: {backtest['violation'].sum()} "
          f"({backtest['violation'].mean()*100:.2f}% -- target is 5%)")

    print("\n--- Kupiec POF test (out-of-sample) ---")
    kupiec_result = KupiecTest(backtest["violation"].values, confidence=0.95).result()
    for k, v in kupiec_result.items():
        print(f"  {k}: {v}")

    print("\n--- Christoffersen independence test (out-of-sample) ---")
    christoffersen_result = ChristoffersenTest(backtest["violation"].values).result()
    for k, v in christoffersen_result.items():
        print(f"  {k}: {v}")

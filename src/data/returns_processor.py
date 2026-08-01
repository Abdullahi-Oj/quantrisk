"""
ReturnsProcessor
-----------------
Converts a price DataFrame into log returns (preferred for risk work,
since they're additive through time) and produces diagnostic statistics
per asset: annualized mean, annualized vol, skewness, kurtosis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ReturnsProcessor:
    def __init__(self, prices: pd.DataFrame, method: str = "log"):
        if method not in ("log", "arithmetic"):
            raise ValueError("method must be 'log' or 'arithmetic'")
        self.prices = prices
        self.method = method
        self.returns = self._compute_returns()

    def _compute_returns(self) -> pd.DataFrame:
        if self.prices.empty:
            raise ValueError(
                "Price data is empty -- cannot compute returns. "
                "Check the data source/date range/ticker symbols."
            )

        if self.method == "log":
            returns = np.log(self.prices / self.prices.shift(1))
        else:
            returns = self.prices.pct_change()

        returns = returns.dropna(how="any")

        if returns.empty:
            nan_counts = self.prices.isna().sum()
            problem_cols = nan_counts[nan_counts > 0].to_dict()
            raise ValueError(
                "Computed returns are empty after dropping rows with missing "
                "values. This usually means at least one ticker has no usable "
                f"price data: {problem_cols if problem_cols else 'check input prices'}."
            )

        return returns

    def diagnostics(self, trading_days: int = 252) -> pd.DataFrame:
        """Per-asset annualized mean/vol plus skew and excess kurtosis."""
        r = self.returns
        stats = pd.DataFrame({
            "ann_mean": r.mean() * trading_days,
            "ann_vol": r.std() * np.sqrt(trading_days),
            "skew": r.skew(),
            "kurtosis": r.kurtosis(),  # pandas reports EXCESS kurtosis (normal = 0)
            "min": r.min(),
            "max": r.max(),
        })
        return stats.round(4)

    def correlation_matrix(self) -> pd.DataFrame:
        return self.returns.corr().round(3)

    def covariance_matrix(self, annualize: bool = True, trading_days: int = 252) -> pd.DataFrame:
        cov = self.returns.cov()
        if annualize:
            cov = cov * trading_days
        return cov


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.data_loader import MarketDataLoader

    tickers = ["SPY", "QQQ", "GLD", "TLT", "USO"]
    prices = MarketDataLoader(tickers, start="2019-01-01").load(mode="synthetic")

    rp = ReturnsProcessor(prices, method="log")
    print("Diagnostics:\n", rp.diagnostics())
    print("\nCorrelation matrix:\n", rp.correlation_matrix())

"""
PortfolioManager
-----------------
Takes per-asset returns and a weight vector, and produces:
  - portfolio_returns: weighted daily return series
  - portfolio_value: simulated value path given an initial investment
  - basic risk/return summary for the combined portfolio

Weights are validated to sum to 1.0 (within tolerance) and must align
1:1 with the tickers in the returns DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioManager:
    def __init__(self, returns: pd.DataFrame, weights: dict[str, float], initial_value: float = 1_000_000):
        unknown = set(weights) - set(returns.columns)
        if unknown:
            raise ValueError(f"Weights given for tickers not in returns data: {unknown}")

        unweighted = set(returns.columns) - set(weights)
        if unweighted:
            raise ValueError(
                f"No weight specified for: {unweighted}. "
                f"Every ticker in the returns data needs an explicit weight (use 0.0 to exclude it)."
            )

        # Align weight vector to the column order of `returns`
        self.tickers = list(returns.columns)
        self.weights = np.array([weights[t] for t in self.tickers])

        if not np.isclose(self.weights.sum(), 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1.0, got {self.weights.sum():.4f}")

        self.returns = returns
        self.initial_value = initial_value

        if returns.empty:
            raise ValueError(
                "Returns data passed to PortfolioManager is empty -- cannot "
                "build a portfolio. This is almost always caused upstream "
                "(failed data load or all rows dropped during return calc)."
            )

        self.portfolio_returns = self._compute_portfolio_returns()
        self.portfolio_value = self._compute_portfolio_value()

    def _compute_portfolio_returns(self) -> pd.Series:
        # R_p = sum(w_i * R_i), assuming log returns (additive across assets
        # at each point in time, even though log returns aren't additive
        # across TIME -- that's fine, it's the standard approximation used
        # for portfolio VaR and is accurate for daily horizons).
        port_ret = self.returns.values @ self.weights
        return pd.Series(port_ret, index=self.returns.index, name="portfolio_return")

    def _compute_portfolio_value(self) -> pd.Series:
        cum_growth = np.exp(np.cumsum(self.portfolio_returns))
        value = self.initial_value * cum_growth
        return value.rename("portfolio_value")

    def summary(self, trading_days: int = 252) -> dict:
        r = self.portfolio_returns
        return {
            "initial_value": self.initial_value,
            "final_value": round(self.portfolio_value.iloc[-1], 2),
            "total_return_pct": round((self.portfolio_value.iloc[-1] / self.initial_value - 1) * 100, 2),
            "annualized_return_pct": round(r.mean() * trading_days * 100, 2),
            "annualized_vol_pct": round(r.std() * np.sqrt(trading_days) * 100, 2),
            "sharpe_ratio_approx": round((r.mean() * trading_days) / (r.std() * np.sqrt(trading_days)), 3),
            "max_drawdown_pct": round(self._max_drawdown() * 100, 2),
        }

    def _max_drawdown(self) -> float:
        cum_max = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - cum_max) / cum_max
        return drawdown.min()

    def weights_table(self) -> pd.DataFrame:
        return pd.DataFrame({"ticker": self.tickers, "weight": self.weights}).set_index("ticker")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.data_loader import MarketDataLoader
    from data.returns_processor import ReturnsProcessor

    tickers = ["SPY", "QQQ", "GLD", "TLT", "USO"]
    weights = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}
    # Note: balanced portfolio per the recommended Portfolio 2 design (USO excluded, weight 0)

    prices = MarketDataLoader(tickers, start="2019-01-01").load(mode="synthetic")
    returns = ReturnsProcessor(prices, method="log").returns

    pm = PortfolioManager(returns, weights, initial_value=1_000_000)

    print("Weights:\n", pm.weights_table())
    print("\nPortfolio summary:")
    for k, v in pm.summary().items():
        print(f"  {k}: {v}")
    print("\nPortfolio returns (tail):\n", pm.portfolio_returns.tail())
    print("\nPortfolio value (tail):\n", pm.portfolio_value.tail())

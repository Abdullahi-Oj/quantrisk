"""
ParametricVaR (Variance-Covariance method)
-------------------------------------------
Assumes portfolio returns are normally distributed. Fast to compute,
but known to understate tail risk during crises (fat tails, skew aren't
captured) -- this is exactly the kind of limitation that's worth
contrasting against HistoricalVaR and MonteCarloVaR in the final report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class ParametricVaR:
    Z_SCORES = {0.90: norm.ppf(0.10), 0.95: norm.ppf(0.05), 0.99: norm.ppf(0.01)}

    def __init__(self, portfolio_returns: pd.Series, portfolio_value: float):
        if len(portfolio_returns) == 0:
            raise ValueError("portfolio_returns is empty -- cannot compute VaR.")
        self.returns = portfolio_returns
        self.portfolio_value = portfolio_value
        self.mu = portfolio_returns.mean()
        self.sigma = portfolio_returns.std()

    def _z(self, confidence: float) -> float:
        return self.Z_SCORES.get(confidence, norm.ppf(1 - confidence))

    def var(self, confidence: float = 0.95) -> float:
        z = self._z(confidence)
        var_return = self.mu + z * self.sigma  # z is negative, so this is a loss
        return max(-var_return, 0.0)

    def var_amount(self, confidence: float = 0.95) -> float:
        return self.var(confidence) * self.portfolio_value

    def summary(self, confidences: list[float] = [0.95, 0.99]) -> pd.DataFrame:
        rows = []
        for c in confidences:
            v = self.var(c)
            rows.append({
                "confidence": c,
                "var_pct": round(v * 100, 3),
                "var_amount": round(v * self.portfolio_value, 2),
            })
        return pd.DataFrame(rows).set_index("confidence")


if __name__ == "__main__":
    from src.data.data_loader import MarketDataLoader
    from src.data.returns_processor import ReturnsProcessor
    from src.portfolio.portfolio import PortfolioManager
    from src.risk.historical_var import HistoricalVaR

    tickers = ["SPY", "QQQ", "GLD", "TLT", "USO"]
    weights = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}

    prices = MarketDataLoader(tickers, start="2019-01-01").load(mode="synthetic")
    returns = ReturnsProcessor(prices).returns
    pm = PortfolioManager(returns, weights, initial_value=1_000_000)
    final_value = pm.portfolio_value.iloc[-1]

    pvar = ParametricVaR(pm.portfolio_returns, final_value)
    hvar = HistoricalVaR(pm.portfolio_returns, final_value)

    print("Parametric VaR:\n", pvar.summary())
    print("\nHistorical VaR (for comparison):\n", hvar.summary()[["var_pct", "var_amount"]])

    # Compare skew/kurtosis to show WHY they diverge
    print(f"\nReturn skew: {pm.portfolio_returns.skew():.3f}, "
          f"excess kurtosis: {pm.portfolio_returns.kurtosis():.3f}")
    print("(Normal distribution has skew=0, excess kurtosis=0 -- "
          "deviations from these explain the gap between methods.)")

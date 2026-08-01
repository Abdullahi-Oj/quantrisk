"""
HistoricalVaR
-------------
Estimates Value-at-Risk directly from the empirical distribution of
historical portfolio returns -- no distributional assumption required.

VaR is reported as a POSITIVE loss number (the convention used across
this engine): e.g. var_95 = 0.025 means "5% chance of losing more than
2.5% of portfolio value in one day."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HistoricalVaR:
    def __init__(self, portfolio_returns: pd.Series, portfolio_value: float):
        if len(portfolio_returns) == 0:
            raise ValueError("portfolio_returns is empty -- cannot compute VaR.")
        self.returns = portfolio_returns
        self.portfolio_value = portfolio_value

    def var(self, confidence: float = 0.95) -> float:
        """
        Returns VaR as a positive fraction of portfolio value.
        confidence=0.95 -> uses the 5th percentile of the return distribution.
        """
        alpha = 1 - confidence
        percentile_return = np.percentile(self.returns, alpha * 100)
        return max(-percentile_return, 0.0)  # VaR reported as a positive loss

    def var_amount(self, confidence: float = 0.95) -> float:
        """VaR in currency terms (same currency as portfolio_value)."""
        return self.var(confidence) * self.portfolio_value

    def summary(self, confidences: list[float] = [0.95, 0.99]) -> pd.DataFrame:
        rows = []
        for c in confidences:
            v = self.var(c)
            rows.append({
                "confidence": c,
                "var_pct": round(v * 100, 3),
                "var_amount": round(v * self.portfolio_value, 2),
                "interpretation": (
                    f"{int((1-c)*100)}% chance of losing more than "
                    f"{v*100:.2f}% (~{v*self.portfolio_value:,.0f}) in one day"
                ),
            })
        return pd.DataFrame(rows).set_index("confidence")

    def breaches(self, confidence: float = 0.95) -> pd.Series:
        """Boolean series: True where actual loss exceeded the static historical VaR."""
        var_threshold = self.var(confidence)
        return self.returns < -var_threshold


if __name__ == "__main__":
    from src.data.data_loader import MarketDataLoader
    from src.data.returns_processor import ReturnsProcessor
    from src.portfolio.portfolio import PortfolioManager

    tickers = ["SPY", "QQQ", "GLD", "TLT", "USO"]
    weights = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}

    prices = MarketDataLoader(tickers, start="2019-01-01").load(mode="synthetic")
    returns = ReturnsProcessor(prices).returns
    pm = PortfolioManager(returns, weights, initial_value=1_000_000)

    hvar = HistoricalVaR(pm.portfolio_returns, pm.portfolio_value.iloc[-1])
    print(hvar.summary())

    breach95 = hvar.breaches(0.95)
    print(f"\nNumber of 95% VaR breaches: {breach95.sum()} out of {len(breach95)} days "
          f"({breach95.mean()*100:.2f}% -- expect ~5% if model is well-calibrated)")

"""
ExpectedShortfall (CVaR)
-------------------------
VaR answers "where does the tail begin?" ES answers "given that we're
in the tail, how bad is it on average?" -- this is the metric Basel
now requires banks to use instead of VaR for internal models, precisely
because VaR ignores everything beyond the threshold.

Two estimation methods are provided:

1. Historical ES: empirical mean of returns beyond the historical VaR
   threshold. Simple, non-parametric, but noisy in the tail (few data
   points beyond a 99% threshold on a few years of daily data).

2. Parametric ES: closed-form expression under the normal assumption.
   ES_alpha = mu - sigma * phi(z_alpha) / alpha
   where phi is the standard normal PDF and z_alpha = norm.ppf(alpha).
   Smooth and stable, but inherits the same fat-tail blind spot as
   Parametric VaR.

(Monte Carlo ES is already available directly on MonteCarloVaR --
 included here too via `from_simulated` for a single consistent interface.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class ExpectedShortfall:
    def __init__(self, portfolio_returns: pd.Series, portfolio_value: float):
        self.returns = portfolio_returns
        self.portfolio_value = portfolio_value
        self.mu = portfolio_returns.mean()
        self.sigma = portfolio_returns.std()

    def historical(self, confidence: float = 0.95) -> float:
        alpha = 1 - confidence
        var_threshold_return = np.percentile(self.returns, alpha * 100)
        tail = self.returns[self.returns <= var_threshold_return]
        if len(tail) == 0:
            return max(-var_threshold_return, 0.0)
        return max(-tail.mean(), 0.0)

    def parametric(self, confidence: float = 0.95) -> float:
        alpha = 1 - confidence
        z = norm.ppf(alpha)
        es_return = self.mu - self.sigma * norm.pdf(z) / alpha
        return max(-es_return, 0.0)

    @staticmethod
    def from_simulated(simulated_returns: np.ndarray, confidence: float = 0.95) -> float:
        alpha = 1 - confidence
        var_threshold = np.percentile(simulated_returns, alpha * 100)
        tail = simulated_returns[simulated_returns <= var_threshold]
        if len(tail) == 0:
            return max(-var_threshold, 0.0)
        return max(-tail.mean(), 0.0)

    def summary(self, confidences: list[float] = [0.95, 0.99]) -> pd.DataFrame:
        rows = []
        for c in confidences:
            hist = self.historical(c)
            param = self.parametric(c)
            rows.append({
                "confidence": c,
                "es_historical_pct": round(hist * 100, 3),
                "es_historical_amount": round(hist * self.portfolio_value, 2),
                "es_parametric_pct": round(param * 100, 3),
                "es_parametric_amount": round(param * self.portfolio_value, 2),
            })
        return pd.DataFrame(rows).set_index("confidence")


if __name__ == "__main__":
    from src.data.data_loader import MarketDataLoader
    from src.data.returns_processor import ReturnsProcessor
    from src.portfolio.portfolio import PortfolioManager
    from src.risk.historical_var import HistoricalVaR
    from src.risk.monte_carlo_var import MonteCarloVaR

    tickers = ["SPY", "QQQ", "GLD", "TLT", "USO"]
    weights_dict = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}

    prices = MarketDataLoader(tickers, start="2019-01-01").load(mode="synthetic")
    returns = ReturnsProcessor(prices).returns
    pm = PortfolioManager(returns, weights_dict, initial_value=1_000_000)
    final_value = pm.portfolio_value.iloc[-1]

    es = ExpectedShortfall(pm.portfolio_returns, final_value)
    print("ES summary (Historical vs Parametric):\n", es.summary())

    # Cross-check against Monte Carlo ES from Phase 7
    mc = MonteCarloVaR(returns, pm.weights, final_value, n_sims=10_000)
    print("\nMonte Carlo ES (from Phase 7 module):")
    for c in [0.95, 0.99]:
        print(f"  confidence={c}: MC ES = {mc.expected_shortfall(c)*100:.3f}%")

    # Sanity check: ES must always be >= VaR at the same confidence level
    hvar = HistoricalVaR(pm.portfolio_returns, final_value)
    for c in [0.95, 0.99]:
        v, e = hvar.var(c), es.historical(c)
        assert e >= v, f"ES ({e}) should never be less than VaR ({v}) at same confidence"
        print(f"Sanity check passed at {c}: ES ({e*100:.3f}%) >= VaR ({v*100:.3f}%)")

"""
MonteCarloVaR
-------------
Estimates portfolio mean/covariance from historical asset returns, then
simulates a large number of joint future return scenarios to build an
empirical distribution of simulated portfolio returns. VaR is read off
that simulated distribution.

This is the "professional" VaR method in the sense that it generalizes
cleanly to non-linear portfolios (options, structured products) where
parametric VaR breaks down -- though for a linear equity/ETF portfolio
like ours, it should converge close to Parametric VaR as n_sims grows,
since we're still sampling from the same multivariate normal assumption.
The real payoff of Monte Carlo shows up once you swap in fat-tailed or
t-distributed innovations, which is a natural "future work" extension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MonteCarloVaR:
    def __init__(
        self,
        asset_returns: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float,
        n_sims: int = 10_000,
        seed: int = 7,
    ):
        self.asset_returns = asset_returns
        self.weights = np.asarray(weights)
        self.portfolio_value = portfolio_value
        self.n_sims = n_sims
        self.rng = np.random.default_rng(seed)

        if self.weights.shape[0] != asset_returns.shape[1]:
            raise ValueError(
                f"weights length ({self.weights.shape[0]}) doesn't match the "
                f"number of assets in asset_returns ({asset_returns.shape[1]}). "
                f"Make sure weights are ordered the same as asset_returns.columns."
            )
        if n_sims <= 0:
            raise ValueError(f"n_sims must be positive, got {n_sims}")

        self.mean = asset_returns.mean().values
        self.cov = asset_returns.cov().values

        self.simulated_asset_returns = self._simulate()
        self.simulated_portfolio_returns = self.simulated_asset_returns @ self.weights

    def _simulate(self) -> np.ndarray:
        return self.rng.multivariate_normal(self.mean, self.cov, size=self.n_sims)

    def var(self, confidence: float = 0.95) -> float:
        alpha = 1 - confidence
        percentile_return = np.percentile(self.simulated_portfolio_returns, alpha * 100)
        return max(-percentile_return, 0.0)

    def expected_shortfall(self, confidence: float = 0.95) -> float:
        var_threshold = self.var(confidence)
        tail = self.simulated_portfolio_returns[self.simulated_portfolio_returns < -var_threshold]
        if len(tail) == 0:
            return var_threshold
        return -tail.mean()

    def var_amount(self, confidence: float = 0.95) -> float:
        return self.var(confidence) * self.portfolio_value

    def summary(self, confidences: list[float] = [0.95, 0.99]) -> pd.DataFrame:
        rows = []
        for c in confidences:
            v = self.var(c)
            es = self.expected_shortfall(c)
            rows.append({
                "confidence": c,
                "var_pct": round(v * 100, 3),
                "var_amount": round(v * self.portfolio_value, 2),
                "es_pct": round(es * 100, 3),
                "es_amount": round(es * self.portfolio_value, 2),
            })
        return pd.DataFrame(rows).set_index("confidence")


if __name__ == "__main__":
    from src.data.data_loader import MarketDataLoader
    from src.data.returns_processor import ReturnsProcessor
    from src.portfolio.portfolio import PortfolioManager
    from src.risk.historical_var import HistoricalVaR
    from src.risk.parametric_var import ParametricVaR

    tickers = ["SPY", "QQQ", "GLD", "TLT", "USO"]
    weights_dict = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}

    prices = MarketDataLoader(tickers, start="2019-01-01").load(mode="synthetic")
    returns = ReturnsProcessor(prices).returns
    pm = PortfolioManager(returns, weights_dict, initial_value=1_000_000)
    final_value = pm.portfolio_value.iloc[-1]

    weights_arr = pm.weights  # aligned to returns.columns order

    mc = MonteCarloVaR(returns, weights_arr, final_value, n_sims=10_000)
    hvar = HistoricalVaR(pm.portfolio_returns, final_value)
    pvar = ParametricVaR(pm.portfolio_returns, final_value)

    print("Monte Carlo VaR + ES:\n", mc.summary())
    print("\n--- Method comparison at 95% / 99% ---")
    for c in [0.95, 0.99]:
        print(f"confidence={c}: Historical={hvar.var(c)*100:.3f}%  "
              f"Parametric={pvar.var(c)*100:.3f}%  MonteCarlo={mc.var(c)*100:.3f}%")

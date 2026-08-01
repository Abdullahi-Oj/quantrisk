"""
RiskAttribution
-----------------
Decomposes total portfolio VaR into per-asset contributions using the
analytical (covariance-based) marginal VaR approach. This is the
standard method banks use for risk attribution because it has a clean
property: component VaRs sum EXACTLY to total portfolio VaR (unlike
naive weight-based or standalone-VaR approaches).

Math:
    sigma_p^2 = w' Cov w                         (portfolio variance)
    sigma_p   = sqrt(w' Cov w)                   (portfolio vol)

    Marginal VaR_i = z * (Cov @ w)_i / sigma_p    (sensitivity of portfolio
                                                    VaR to a small change in
                                                    asset i's weight)

    Component VaR_i = w_i * Marginal VaR_i        (asset i's slice of total VaR)

    sum(Component VaR_i) = z * sigma_p = Portfolio VaR   <- exact decomposition

This assumes normally distributed returns (same assumption as Parametric
VaR) -- it's the price of getting an exact, additive decomposition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class RiskAttribution:
    def __init__(self, asset_returns: pd.DataFrame, weights: np.ndarray):
        self.tickers = list(asset_returns.columns)
        self.weights = np.asarray(weights)

        if self.weights.shape[0] != len(self.tickers):
            raise ValueError(
                f"weights length ({self.weights.shape[0]}) doesn't match the "
                f"number of assets in asset_returns ({len(self.tickers)}). "
                f"Make sure weights are ordered the same as asset_returns.columns."
            )

        self.cov = asset_returns.cov().values

        self.portfolio_variance = self.weights @ self.cov @ self.weights
        self.portfolio_vol = np.sqrt(self.portfolio_variance)

    def marginal_var(self, confidence: float = 0.95) -> np.ndarray:
        z = abs(norm.ppf(1 - confidence))
        # Cov @ w gives each asset's covariance contribution to portfolio variance
        return z * (self.cov @ self.weights) / self.portfolio_vol

    def component_var(self, confidence: float = 0.95) -> np.ndarray:
        return self.weights * self.marginal_var(confidence)

    def portfolio_var(self, confidence: float = 0.95) -> float:
        z = abs(norm.ppf(1 - confidence))
        return z * self.portfolio_vol

    def attribution_table(self, confidence: float = 0.95) -> pd.DataFrame:
        mvar = self.marginal_var(confidence)
        cvar = self.component_var(confidence)
        total_var = cvar.sum()  # should equal self.portfolio_var(confidence)

        df = pd.DataFrame({
            "ticker": self.tickers,
            "weight": self.weights,
            "marginal_var_pct": mvar * 100,
            "component_var_pct": cvar * 100,
            "pct_of_total_risk": (cvar / total_var) * 100,
        }).set_index("ticker")
        return df.round(3).sort_values("pct_of_total_risk", ascending=False)

    def verify_decomposition(self, confidence: float = 0.95) -> tuple[float, float]:
        """Returns (sum of component VaRs, portfolio VaR) -- should match."""
        component_sum = self.component_var(confidence).sum()
        total = self.portfolio_var(confidence)
        return component_sum, total


if __name__ == "__main__":
    from src.data.data_loader import MarketDataLoader
    from src.data.returns_processor import ReturnsProcessor
    from src.portfolio.portfolio import PortfolioManager

    tickers = ["SPY", "QQQ", "GLD", "TLT", "USO"]
    weights_dict = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}

    prices = MarketDataLoader(tickers, start="2019-01-01").load(mode="synthetic")
    returns = ReturnsProcessor(prices).returns
    pm = PortfolioManager(returns, weights_dict, initial_value=1_000_000)

    ra = RiskAttribution(returns, pm.weights)

    print("Risk Attribution (95% confidence):\n")
    print(ra.attribution_table(0.95))

    comp_sum, total = ra.verify_decomposition(0.95)
    print(f"\nSum of component VaRs: {comp_sum*100:.4f}%")
    print(f"Total portfolio VaR:   {total*100:.4f}%")
    print(f"Decomposition exact?   {np.isclose(comp_sum, total)}")

    # Highlight the "weight vs risk contribution" gap the doc calls out
    print("\nWeight vs. Risk Contribution gap (the whole point of this phase):")
    table = ra.attribution_table(0.95)
    for ticker in table.index:
        w = table.loc[ticker, "weight"] * 100
        r = table.loc[ticker, "pct_of_total_risk"]
        print(f"  {ticker}: {w:.1f}% of capital, but {r:.1f}% of portfolio risk")

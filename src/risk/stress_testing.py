"""
StressTester
-------------
VaR/ES are statistical estimates from a fitted distribution -- they're
only as good as the historical data feeding them. Stress testing is a
deliberately different approach: apply documented, REAL historical
crisis shocks directly to the current portfolio weights, bypassing the
statistical model entirely. This is the standard complement to VaR in
both regulatory frameworks and risk-management practice, precisely
because it captures regime shifts and tail events that a stationary
statistical model (like the one underlying our backtested VaR) won't
generate on its own -- which is exactly the gap the Phase 10 backtest
caveat flagged.

IMPORTANT CALIBRATION NOTE:
The shock magnitudes below are approximate, illustrative figures based
on widely-documented peak-to-trough moves during each crisis (e.g. SPY's
~34% COVID drawdown, oil's collapse in 2020, the flight-to-safety bond
rally in 2008). They are good enough to demonstrate the methodology and
get directionally sound results, but for a rigorous report you should
replace them with shocks computed directly from real historical price
data once you're running this with live yfinance data locally:
e.g. actual peak-to-trough log return for each ticker over the exact
crisis window you choose to define.
"""

from __future__ import annotations

import pandas as pd

# Approximate documented peak-to-trough (or peak-to-peak for rallying
# assets) returns during each historical scenario, by ticker. Extend
# this dict with more tickers/scenarios as needed -- any ticker not
# listed in a scenario is assumed unaffected (shock = 0.0).
SCENARIOS: dict[str, dict[str, float]] = {
    "covid_crash_2020": {
        # Feb 19 - Mar 23, 2020
        "SPY": -0.34, "QQQ": -0.28, "GLD": -0.10, "TLT": 0.18, "USO": -0.65,
        "EEM": -0.33, "BTC-USD": -0.50,
    },
    "gfc_2008": {
        # Sep - Nov 2008, post-Lehman collapse
        "SPY": -0.45, "QQQ": -0.42, "GLD": -0.05, "TLT": 0.20, "USO": -0.55,
        "EEM": -0.55, "BTC-USD": 0.0,  # didn't exist yet
    },
    "oil_price_shock": {
        # Illustrative standalone oil-collapse scenario (e.g. 2014-16 style)
        "SPY": -0.05, "QQQ": -0.02, "GLD": 0.05, "TLT": 0.05, "USO": -0.75,
        "EEM": -0.10, "BTC-USD": 0.0,
    },
}


class StressTester:
    def __init__(self, weights: dict[str, float], portfolio_value: float):
        if not weights:
            raise ValueError("weights dict cannot be empty.")
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"weights must sum to ~1.0 to represent a real portfolio, got {total:.4f}. "
                f"Did you pass partial weights or forget an asset?"
            )
        if portfolio_value <= 0:
            raise ValueError(f"portfolio_value must be positive, got {portfolio_value}")

        self.weights = weights
        self.portfolio_value = portfolio_value

    def apply_scenario(self, scenario_name: str) -> dict:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIOS)}")

        shocks = SCENARIOS[scenario_name]
        per_asset_loss = {}
        total_return = 0.0

        for ticker, w in self.weights.items():
            shock = shocks.get(ticker, 0.0)  # unlisted ticker assumed unaffected
            contribution = w * shock
            total_return += contribution
            per_asset_loss[ticker] = {
                "weight": w,
                "shock_pct": round(shock * 100, 2),
                "contribution_to_portfolio_pct": round(contribution * 100, 3),
            }

        return {
            "scenario": scenario_name,
            "portfolio_return_pct": round(total_return * 100, 2),
            "portfolio_loss_amount": round(-total_return * self.portfolio_value, 2),
            "stressed_value": round(self.portfolio_value * (1 + total_return), 2),
            "per_asset": per_asset_loss,
        }

    def run_all(self) -> pd.DataFrame:
        rows = []
        for name in SCENARIOS:
            r = self.apply_scenario(name)
            rows.append({
                "scenario": name,
                "portfolio_return_pct": r["portfolio_return_pct"],
                "loss_amount": r["portfolio_loss_amount"],
                "stressed_value": r["stressed_value"],
            })
        return pd.DataFrame(rows).set_index("scenario")


if __name__ == "__main__":
    weights = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}
    portfolio_value = 1_000_000

    st = StressTester(weights, portfolio_value)

    print("=== Stress Test Summary ===\n")
    print(st.run_all())

    print("\n=== Detail: COVID Crash 2020 ===")
    detail = st.apply_scenario("covid_crash_2020")
    for ticker, info in detail["per_asset"].items():
        print(f"  {ticker}: weight={info['weight']:.0%}, "
              f"shock={info['shock_pct']}%, "
              f"contribution={info['contribution_to_portfolio_pct']}%")
    print(f"  -> Total portfolio return: {detail['portfolio_return_pct']}%  "
          f"(loss of ~{detail['portfolio_loss_amount']:,.0f})")

    # Compare stress loss against the VaR/ES figures from earlier phases
    print("\n=== Why this matters: stress vs. statistical VaR ===")
    print("Recall 95% historical VaR ~= 1.02% daily, 99% ES ~= 1.85% daily (Phases 5 & 8).")
    covid_loss_pct = abs(detail['portfolio_return_pct'])
    print(f"COVID crash scenario implies a {covid_loss_pct}% loss -- "
          f"roughly {covid_loss_pct/1.02:.0f}x the 95% daily VaR figure, "
          "underscoring that VaR describes 'normal' bad days, not crisis events.")

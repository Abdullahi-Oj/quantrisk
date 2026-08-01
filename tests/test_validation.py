"""
Validation suite -- one test per specific question, each with an
explicit manual cross-check rather than just trusting the module's own
internal logic.

Run with: pytest tests/test_validation.py -v -s
(-s shows the printed evidence for each check, not just pass/fail)
"""

import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import MarketDataLoader
from src.data.returns_processor import ReturnsProcessor
from src.portfolio.portfolio import PortfolioManager
from src.risk.historical_var import HistoricalVaR
from src.risk.parametric_var import ParametricVaR
from src.risk.monte_carlo_var import MonteCarloVaR
from src.risk.expected_shortfall import ExpectedShortfall
from src.risk.stress_testing import StressTester, SCENARIOS
from src.attribution.risk_attribution import RiskAttribution

TICKERS = ["SPY", "QQQ", "GLD", "TLT", "USO"]

# Three portfolios spanning the risk spectrum, matching the original
# project brief's "conservative / balanced / aggressive" design.
PORTFOLIOS = {
    "conservative": {"SPY": 0.20, "QQQ": 0.0, "GLD": 0.20, "TLT": 0.60, "USO": 0.0},
    "balanced":     {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0},
    "aggressive":   {"SPY": 0.20, "QQQ": 0.70, "GLD": 0.0, "TLT": 0.0, "USO": 0.10},
}


@pytest.fixture(scope="module")
def returns_data():
    prices = MarketDataLoader(TICKERS, start="2019-01-01").load(mode="synthetic")
    return ReturnsProcessor(prices).returns


def build(name, returns_data):
    pm = PortfolioManager(returns_data, PORTFOLIOS[name], initial_value=1_000_000)
    return pm


# --------------------------------------------------------------------- #
# Q: Does Historical VaR match manual calculation?
# --------------------------------------------------------------------- #
def test_historical_var_matches_manual_percentile(returns_data):
    pm = build("balanced", returns_data)
    hvar = HistoricalVaR(pm.portfolio_returns, pm.portfolio_value.iloc[-1])

    for c in [0.90, 0.95, 0.99]:
        manual = max(-np.percentile(pm.portfolio_returns, (1 - c) * 100), 0.0)
        module = hvar.var(c)
        print(f"[Historical VaR {c}] manual={manual:.6f}  module={module:.6f}")
        assert module == pytest.approx(manual, abs=1e-9)


# --------------------------------------------------------------------- #
# Q: Does Parametric VaR behave as expected at 95% and 99%?
# --------------------------------------------------------------------- #
def test_parametric_var_matches_manual_formula(returns_data):
    from scipy.stats import norm
    pm = build("balanced", returns_data)
    pvar = ParametricVaR(pm.portfolio_returns, pm.portfolio_value.iloc[-1])

    mu, sigma = pm.portfolio_returns.mean(), pm.portfolio_returns.std()
    for c, z in [(0.95, norm.ppf(0.05)), (0.99, norm.ppf(0.01))]:
        manual = max(-(mu + z * sigma), 0.0)
        module = pvar.var(c)
        print(f"[Parametric VaR {c}] manual={manual:.6f}  module={module:.6f}")
        assert module == pytest.approx(manual, abs=1e-9)


def test_parametric_var_99_always_exceeds_95(returns_data):
    """A more extreme confidence level must always imply a larger loss
    threshold -- true by construction of the normal distribution, but
    worth confirming the implementation doesn't break that."""
    for name in PORTFOLIOS:
        pm = build(name, returns_data)
        pvar = ParametricVaR(pm.portfolio_returns, pm.portfolio_value.iloc[-1])
        v95, v99 = pvar.var(0.95), pvar.var(0.99)
        print(f"[{name}] 95% VaR={v95*100:.3f}%  99% VaR={v99*100:.3f}%")
        assert v99 >= v95


# --------------------------------------------------------------------- #
# Q: Does Monte Carlo VaR converge as n_sims increases?
# --------------------------------------------------------------------- #
def test_monte_carlo_converges_with_more_simulations(returns_data):
    """As n_sims grows, the MC estimate's variance across different random
    seeds should shrink (standard Monte Carlo error ~ 1/sqrt(n))."""
    pm = build("balanced", returns_data)
    final_value = pm.portfolio_value.iloc[-1]
    weights = pm.weights

    def estimate_spread(n_sims, n_trials=8):
        estimates = [
            MonteCarloVaR(returns_data, weights, final_value, n_sims=n_sims, seed=s).var(0.95)
            for s in range(n_trials)
        ]
        return np.std(estimates)

    spread_small = estimate_spread(500)
    spread_large = estimate_spread(20_000)
    print(f"[MC convergence] std across seeds: n=500 -> {spread_small:.6f}, "
          f"n=20000 -> {spread_large:.6f}")
    assert spread_large < spread_small, "More simulations should reduce estimate variance"


def test_monte_carlo_converges_to_parametric_value(returns_data):
    pm = build("balanced", returns_data)
    final_value = pm.portfolio_value.iloc[-1]
    pvar = ParametricVaR(pm.portfolio_returns, final_value).var(0.95)

    mc_small = MonteCarloVaR(returns_data, pm.weights, final_value, n_sims=200, seed=1).var(0.95)
    mc_large = MonteCarloVaR(returns_data, pm.weights, final_value, n_sims=50_000, seed=1).var(0.95)

    diff_small = abs(mc_small - pvar)
    diff_large = abs(mc_large - pvar)
    print(f"[MC vs Parametric] n=200 diff={diff_small:.6f}, n=50000 diff={diff_large:.6f}")
    assert diff_large < diff_small


# --------------------------------------------------------------------- #
# Q: Does Expected Shortfall always exceed VaR in magnitude?
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("portfolio_name", list(PORTFOLIOS))
def test_es_exceeds_var_across_portfolios(returns_data, portfolio_name):
    pm = build(portfolio_name, returns_data)
    final_value = pm.portfolio_value.iloc[-1]
    hvar = HistoricalVaR(pm.portfolio_returns, final_value)
    pvar = ParametricVaR(pm.portfolio_returns, final_value)
    es = ExpectedShortfall(pm.portfolio_returns, final_value)

    for c in [0.95, 0.99]:
        print(f"[{portfolio_name} @ {c}] VaR(hist)={hvar.var(c)*100:.3f}%  "
              f"ES(hist)={es.historical(c)*100:.3f}%  "
              f"VaR(param)={pvar.var(c)*100:.3f}%  ES(param)={es.parametric(c)*100:.3f}%")
        assert es.historical(c) >= hvar.var(c) - 1e-9
        assert es.parametric(c) >= pvar.var(c) - 1e-9


# --------------------------------------------------------------------- #
# Q: Do component VaRs sum exactly to total portfolio VaR? (multi-portfolio)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("portfolio_name", list(PORTFOLIOS))
def test_decomposition_exact_across_portfolios(returns_data, portfolio_name):
    pm = build(portfolio_name, returns_data)
    ra = RiskAttribution(returns_data, pm.weights)
    for c in [0.90, 0.95, 0.99]:
        component_sum, total = ra.verify_decomposition(c)
        print(f"[{portfolio_name} @ {c}] sum(component)={component_sum:.6f}  total={total:.6f}")
        assert component_sum == pytest.approx(total, rel=1e-9)


# --------------------------------------------------------------------- #
# Q: Are stress-test losses consistent with the defined shocks?
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("portfolio_name", list(PORTFOLIOS))
@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_stress_loss_matches_manual_weighted_shock(portfolio_name, scenario_name):
    weights = PORTFOLIOS[portfolio_name]
    value = 1_000_000
    st_tester = StressTester(weights, value)
    result = st_tester.apply_scenario(scenario_name)

    shocks = SCENARIOS[scenario_name]
    manual_return = sum(w * shocks.get(t, 0.0) for t, w in weights.items())
    manual_loss = -manual_return * value

    print(f"[{portfolio_name} / {scenario_name}] manual_loss={manual_loss:,.2f}  "
          f"module_loss={result['portfolio_loss_amount']:,.2f}")
    assert result["portfolio_loss_amount"] == pytest.approx(manual_loss, abs=0.01)


# --------------------------------------------------------------------- #
# Multi-portfolio sanity: every portfolio should produce sane, finite stats
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("portfolio_name", list(PORTFOLIOS))
def test_portfolio_summary_is_sane(returns_data, portfolio_name):
    pm = build(portfolio_name, returns_data)
    summary = pm.summary()
    print(f"[{portfolio_name}] {summary}")

    assert np.isfinite(summary["annualized_vol_pct"])
    assert summary["annualized_vol_pct"] > 0
    assert -100 <= summary["max_drawdown_pct"] <= 0
    assert summary["final_value"] > 0

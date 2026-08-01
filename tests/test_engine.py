"""
Engine invariant tests.

These check properties that MUST hold regardless of input data -- if any
of these ever fail, something is genuinely broken (not just "numbers
look different on real data").

Run with: pytest tests/test_engine.py -v
"""

import numpy as np
import pytest

from src.data.data_loader import MarketDataLoader
from src.data.returns_processor import ReturnsProcessor
from src.portfolio.portfolio import PortfolioManager
from src.risk.historical_var import HistoricalVaR
from src.risk.parametric_var import ParametricVaR
from src.risk.monte_carlo_var import MonteCarloVaR
from src.risk.expected_shortfall import ExpectedShortfall
from src.attribution.risk_attribution import RiskAttribution
from src.backtesting.kupiec import KupiecTest
from src.backtesting.christoffersen import ChristoffersenTest

TICKERS = ["SPY", "QQQ", "GLD", "TLT", "USO"]
WEIGHTS = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}


@pytest.fixture(scope="module")
def engine():
    prices = MarketDataLoader(TICKERS, start="2019-01-01").load(mode="synthetic")
    returns = ReturnsProcessor(prices).returns
    pm = PortfolioManager(returns, WEIGHTS, initial_value=1_000_000)
    return {"prices": prices, "returns": returns, "pm": pm}


# --------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------- #
def test_synthetic_loader_produces_data():
    prices = MarketDataLoader(TICKERS, start="2019-01-01").load(mode="synthetic")
    assert not prices.empty
    assert list(prices.columns) == TICKERS
    assert prices.shape[0] > 100


def test_returns_processor_rejects_empty_data():
    import pandas as pd
    with pytest.raises(ValueError):
        ReturnsProcessor(pd.DataFrame())


def test_returns_processor_rejects_all_nan_column():
    import pandas as pd
    dates = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"SPY": np.linspace(400, 410, 10), "BAD": [np.nan] * 10}, index=dates)
    with pytest.raises(ValueError):
        ReturnsProcessor(prices)


# --------------------------------------------------------------------- #
# Portfolio engine
# --------------------------------------------------------------------- #
def test_portfolio_weights_must_sum_to_one(engine):
    with pytest.raises(ValueError):
        PortfolioManager(engine["returns"], {"SPY": 0.5, "QQQ": 0.3})  # missing tickers


def test_portfolio_rejects_unknown_ticker(engine):
    with pytest.raises(ValueError):
        PortfolioManager(engine["returns"], {**WEIGHTS, "FAKE": 0.0})


def test_portfolio_value_starts_at_initial_value(engine):
    pm = engine["pm"]
    # value series starts AFTER day 1 return is applied, so just check it's
    # in a sane neighborhood of the initial value, not wildly off
    assert pm.portfolio_value.iloc[0] / pm.initial_value == pytest.approx(1.0, abs=0.1)


# --------------------------------------------------------------------- #
# Risk metrics invariants
# --------------------------------------------------------------------- #
def test_var_is_nonnegative(engine):
    pm = engine["pm"]
    final_value = pm.portfolio_value.iloc[-1]
    for cls in [HistoricalVaR, ParametricVaR]:
        model = cls(pm.portfolio_returns, final_value)
        for c in [0.90, 0.95, 0.99]:
            assert model.var(c) >= 0


def test_var_increases_with_confidence(engine):
    """99% VaR must always be >= 95% VaR -- a higher confidence level
    means a more extreme (larger) loss threshold."""
    pm = engine["pm"]
    final_value = pm.portfolio_value.iloc[-1]
    for cls in [HistoricalVaR, ParametricVaR]:
        model = cls(pm.portfolio_returns, final_value)
        assert model.var(0.99) >= model.var(0.95)


def test_expected_shortfall_never_less_than_var(engine):
    """ES is the average loss BEYOND VaR -- mathematically it can never
    be smaller than VaR at the same confidence level."""
    pm = engine["pm"]
    final_value = pm.portfolio_value.iloc[-1]
    hvar = HistoricalVaR(pm.portfolio_returns, final_value)
    es = ExpectedShortfall(pm.portfolio_returns, final_value)
    for c in [0.95, 0.99]:
        assert es.historical(c) >= hvar.var(c) - 1e-9
        assert es.parametric(c) >= ParametricVaR(pm.portfolio_returns, final_value).var(c) - 1e-9


def test_monte_carlo_var_converges_to_parametric(engine):
    """With enough simulations, Monte Carlo VaR should land close to
    Parametric VaR, since both sample the same multivariate normal."""
    pm, returns = engine["pm"], engine["returns"]
    final_value = pm.portfolio_value.iloc[-1]
    mc = MonteCarloVaR(returns, pm.weights, final_value, n_sims=20_000, seed=1)
    pvar = ParametricVaR(pm.portfolio_returns, final_value)
    assert abs(mc.var(0.95) - pvar.var(0.95)) < 0.005  # within 0.5 percentage points


# --------------------------------------------------------------------- #
# Risk attribution
# --------------------------------------------------------------------- #
def test_component_var_sums_exactly_to_portfolio_var(engine):
    pm, returns = engine["pm"], engine["returns"]
    ra = RiskAttribution(returns, pm.weights)
    for c in [0.90, 0.95, 0.99]:
        component_sum, total = ra.verify_decomposition(c)
        assert component_sum == pytest.approx(total, rel=1e-9)


def test_risk_contributions_sum_to_100_percent(engine):
    pm, returns = engine["pm"], engine["returns"]
    ra = RiskAttribution(returns, pm.weights)
    table = ra.attribution_table(0.95)
    assert table["pct_of_total_risk"].sum() == pytest.approx(100.0, abs=0.01)


# --------------------------------------------------------------------- #
# Backtesting statistics
# --------------------------------------------------------------------- #
def test_kupiec_passes_on_well_calibrated_sequence():
    rng = np.random.default_rng(0)
    violations = rng.random(2000) < 0.05  # exactly the target rate, by construction
    result = KupiecTest(violations, confidence=0.95).result()
    assert result["verdict"] == "PASS"


def test_kupiec_fails_on_excessive_breaches():
    rng = np.random.default_rng(0)
    violations = rng.random(2000) < 0.20  # way more than the 5% target
    result = KupiecTest(violations, confidence=0.95).result()
    assert result["verdict"] == "FAIL"


def test_christoffersen_detects_clustering():
    """Same total violation count, independent vs clustered -- only
    Christoffersen should tell them apart."""
    independent = np.zeros(1000, dtype=int)
    rng = np.random.default_rng(0)
    idx = rng.choice(1000, 50, replace=False)
    independent[idx] = 1

    clustered = np.zeros(1000, dtype=int)
    for start in [100, 300, 500, 700, 900]:
        clustered[start:start + 10] = 1

    assert independent.sum() == clustered.sum() == 50  # same count

    indep_result = ChristoffersenTest(independent).result()
    clustered_result = ChristoffersenTest(clustered).result()

    assert indep_result["verdict"] == "PASS"
    assert clustered_result["verdict"] == "FAIL"


# --------------------------------------------------------------------- #
# New input-validation error messages
# --------------------------------------------------------------------- #
def test_monte_carlo_rejects_mismatched_weights(engine):
    with pytest.raises(ValueError, match="doesn't match"):
        MonteCarloVaR(engine["returns"], np.array([0.5, 0.5]), 1_000_000, n_sims=100)


def test_monte_carlo_rejects_invalid_n_sims(engine):
    with pytest.raises(ValueError, match="n_sims must be positive"):
        MonteCarloVaR(engine["returns"], engine["pm"].weights, 1_000_000, n_sims=0)


def test_risk_attribution_rejects_mismatched_weights(engine):
    with pytest.raises(ValueError, match="doesn't match"):
        RiskAttribution(engine["returns"], np.array([0.5, 0.5]))


def test_stress_tester_rejects_bad_weight_sum():
    from src.risk.stress_testing import StressTester
    with pytest.raises(ValueError, match="sum to ~1.0"):
        StressTester({"SPY": 0.5, "QQQ": 0.3}, 1_000_000)


def test_stress_tester_rejects_empty_weights():
    from src.risk.stress_testing import StressTester
    with pytest.raises(ValueError, match="cannot be empty"):
        StressTester({}, 1_000_000)


def test_historical_var_rejects_empty_returns():
    import pandas as pd
    with pytest.raises(ValueError, match="empty"):
        HistoricalVaR(pd.Series([], dtype=float), 1_000_000)


def test_parametric_var_rejects_empty_returns():
    import pandas as pd
    with pytest.raises(ValueError, match="empty"):
        ParametricVaR(pd.Series([], dtype=float), 1_000_000)

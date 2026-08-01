"""
QuantRisk Dashboard
---------------------
Streamlit front-end tying together every module built in Phases 2-11.

Run with:
    streamlit run app/dashboard.py
(run from the quantrisk/ project root so the `src` package import resolves)
"""

import sys
from pathlib import Path

# Make `src` importable regardless of where streamlit is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.data_loader import MarketDataLoader
from src.data.returns_processor import ReturnsProcessor
from src.portfolio.portfolio import PortfolioManager
from src.risk.historical_var import HistoricalVaR
from src.risk.parametric_var import ParametricVaR
from src.risk.monte_carlo_var import MonteCarloVaR
from src.risk.expected_shortfall import ExpectedShortfall
from src.risk.stress_testing import StressTester, SCENARIOS
from src.attribution.risk_attribution import RiskAttribution
from src.backtesting.kupiec import KupiecTest
from src.backtesting.christoffersen import ChristoffersenTest
from src.backtesting.rolling_backtest import rolling_var_backtest

st.set_page_config(page_title="QuantRisk", layout="wide", page_icon="📉")

DEFAULT_TICKERS = ["SPY", "QQQ", "GLD", "TLT", "USO"]
DEFAULT_WEIGHTS = {"SPY": 0.50, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15, "USO": 0.0}
CURRENCY_SYMBOLS = {"₦ Naira": "₦", "$ US Dollar": "$", "€ Euro": "€", "£ Pound": "£"}


def fmt_money(value: float) -> str:
    """Format a number with the session's selected currency symbol."""
    symbol = st.session_state.get("currency_symbol", "₦")
    return f"{symbol}{value:,.0f}"


# ---------------------------------------------------------------------- #
# Cached data/engine construction so re-running pages doesn't re-simulate
# ---------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading market data...")
def load_prices(tickers: tuple, start: str, mode: str) -> pd.DataFrame:
    return MarketDataLoader(list(tickers), start=start).load(mode=mode)


def build_engine(tickers: list, weights: dict, start: str, mode: str, initial_value: float):
    prices = load_prices(tuple(tickers), start, mode)
    returns = ReturnsProcessor(prices, method="log").returns
    pm = PortfolioManager(returns, weights, initial_value=initial_value)
    return prices, returns, pm


# ---------------------------------------------------------------------- #
# Sidebar: global portfolio config (shared across all pages via session_state)
# ---------------------------------------------------------------------- #
def sidebar_config():
    st.sidebar.title("📉 QuantRisk")
    st.sidebar.caption("Portfolio VaR & Expected Shortfall Analytics")
    page = st.sidebar.radio(
        "Navigate",
        ["1. Portfolio Builder", "2. Risk Analytics", "3. Backtesting",
         "4. Risk Attribution", "5. Stress Testing"],

    )

    st.sidebar.divider()
    st.sidebar.caption(
        "💡 'synthetic' mode generates realistic-looking data offline -- "
        "good for testing the app itself. Switch to 'live' for real "
        "yfinance data (requires internet access to Yahoo Finance)."
    )
    data_mode = st.sidebar.selectbox("Data source", ["synthetic", "live"], index=0)
    start_date = st.sidebar.text_input("History start date", value="2019-01-01")

    currency_label = st.sidebar.selectbox("Currency", list(CURRENCY_SYMBOLS), index=0)
    st.session_state.currency_symbol = CURRENCY_SYMBOLS[currency_label]
    initial_value = st.sidebar.number_input(
        "Initial portfolio value", min_value=1_000.0, value=1_000_000.0, step=10_000.0
    )

    if "weights" not in st.session_state:
        st.session_state.weights = dict(DEFAULT_WEIGHTS)

    st.sidebar.divider()
    with st.sidebar.expander("ℹ️ About QuantRisk"):
        st.markdown(
            "An end-to-end portfolio risk engine: Historical/Parametric/Monte "
            "Carlo VaR, Expected Shortfall, risk attribution, out-of-sample "
            "backtesting (Kupiec + Christoffersen), and historical-scenario "
            "stress testing.\n\n"
            "Built by **Abdullahi Aliyu Ojonimi** — "
            "[GitHub](https://github.com/Abdullahi-Oj)"
        )

    return page, data_mode, start_date, initial_value


# ---------------------------------------------------------------------- #
# Page 1: Portfolio Builder
# ---------------------------------------------------------------------- #
def page_portfolio_builder(data_mode, start_date, initial_value):
    st.header("1. Portfolio Builder")
    st.write("Select assets and set weights. Weights must sum to 100%.")

    tickers = st.multiselect(
        "Asset universe", options=list(set(DEFAULT_TICKERS) | set(st.session_state.weights)),
        default=DEFAULT_TICKERS,
    )
    custom_ticker = st.text_input("Add a custom ticker (optional)", value="")
    if custom_ticker:
        custom_ticker = custom_ticker.strip().upper()
        if custom_ticker not in tickers:
            tickers.append(custom_ticker)

    if not tickers:
        st.warning("Select at least one ticker to continue.")
        return

    # Detect a change in the ticker SET (added or removed an asset) and
    # auto-rebalance to equal weights when that happens. This has to
    # directly overwrite each widget's session_state value (not just the
    # `weights` dict / `value=` default) -- once a number_input with a
    # given key has rendered once, Streamlit remembers its last value and
    # ignores the `value=` parameter on subsequent reruns. Setting
    # st.session_state[key] BEFORE the widget is (re)created is the only
    # way to actually change what's displayed.
    prev_tickers = st.session_state.get("prev_tickers")
    if prev_tickers != tickers:
        equal_weight = round(1 / len(tickers), 4)
        for ticker in tickers:
            st.session_state[f"w_{ticker}"] = equal_weight
        st.session_state.prev_tickers = list(tickers)
        if prev_tickers is not None:  # don't toast on the very first render
            st.toast(f"Asset list changed -- weights reset to equal ({equal_weight*100:.1f}% each).")

    st.subheader("Weights")
    rebalance_col, _ = st.columns([1, 3])
    if rebalance_col.button("⚖️ Reset to equal weights"):
        equal_weight = round(1 / len(tickers), 4)
        for ticker in tickers:
            st.session_state[f"w_{ticker}"] = equal_weight
        st.rerun()

    cols = st.columns(len(tickers))
    weights = {}
    for col, ticker in zip(cols, tickers):
        weights[ticker] = col.number_input(
            ticker, min_value=0.0, max_value=1.0, step=0.05, key=f"w_{ticker}"
        )

    total = sum(weights.values())
    st.metric("Total weight", f"{total*100:.1f}%", delta=f"{(total-1)*100:+.1f}% from 100%")

    if not np.isclose(total, 1.0, atol=1e-3):
        st.error("Weights must sum to 100% before you can build the portfolio.")
        return

    st.session_state.weights = weights
    st.session_state.tickers = tickers

    if st.button("Build Portfolio", type="primary"):
        try:
            with st.spinner("Loading data and computing portfolio returns..."):
                prices, returns, pm = build_engine(tickers, weights, start_date, data_mode, initial_value)
                st.session_state.engine = {
                    "prices": prices, "returns": returns, "pm": pm,
                    "tickers": tickers, "weights": weights, "initial_value": initial_value,
                }
            st.success("Portfolio built. Head to the other pages to see risk analytics.")
        except Exception as e:
            st.error(
                f"Couldn't build the portfolio: {e}\n\n"
                f"If you're using 'live' mode, this is usually a yfinance/network "
                f"issue (try upgrading yfinance, or check the ticker symbols are "
                f"valid). Switch the sidebar data source to 'synthetic' to verify "
                f"the rest of the app works while you debug the live data issue."
            )
            st.session_state.pop("engine", None)

    if "engine" in st.session_state:
        pm = st.session_state.engine["pm"]
        st.subheader("Portfolio Summary")
        summary = pm.summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Value", fmt_money(summary['final_value']))
        c2.metric("Total Return", f"{summary['total_return_pct']:.1f}%")
        c3.metric("Annualized Vol", f"{summary['annualized_vol_pct']:.1f}%")
        c4.metric("Max Drawdown", f"{summary['max_drawdown_pct']:.1f}%")

        fig = px.line(pm.portfolio_value, title="Portfolio Value Over Time")
        symbol = st.session_state.get("currency_symbol", "₦")
        fig.update_layout(showlegend=False, yaxis_title=f"Value ({symbol})", xaxis_title="Date")
        st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------- #
# Page 2: Risk Analytics
# ---------------------------------------------------------------------- #
def page_risk_analytics():
    st.header("2. Risk Analytics")
    if "engine" not in st.session_state:
        st.info("Build a portfolio on Page 1 first.")
        return

    eng = st.session_state.engine
    pm, returns = eng["pm"], eng["returns"]
    final_value = pm.portfolio_value.iloc[-1]

    confidences = st.multiselect("Confidence levels", [0.90, 0.95, 0.99], default=[0.95, 0.99])
    if not confidences:
        st.warning("Select at least one confidence level.")
        return

    hvar = HistoricalVaR(pm.portfolio_returns, final_value)
    pvar = ParametricVaR(pm.portfolio_returns, final_value)
    mc = MonteCarloVaR(returns, pm.weights, final_value, n_sims=10_000)
    es = ExpectedShortfall(pm.portfolio_returns, final_value)

    rows = []
    for c in confidences:
        rows.append({
            "Confidence": f"{int(c*100)}%",
            "Historical VaR": hvar.var(c) * 100,
            "Parametric VaR": pvar.var(c) * 100,
            "Monte Carlo VaR": mc.var(c) * 100,
            "Historical ES": es.historical(c) * 100,
            "Parametric ES": es.parametric(c) * 100,
        })
    df = pd.DataFrame(rows).set_index("Confidence")

    st.subheader("VaR & Expected Shortfall (% of portfolio, daily)")
    st.dataframe(df.round(3).style.format("{:.3f}%"), width='stretch')

    fig = px.bar(df.reset_index().melt(id_vars="Confidence", var_name="Method", value_name="Pct"),
                 x="Confidence", y="Pct", color="Method", barmode="group",
                 title="VaR / ES Method Comparison")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Return Distribution")
    fig2 = px.histogram(pm.portfolio_returns, nbins=80, title="Daily Portfolio Return Distribution")
    var95 = hvar.var(0.95)
    fig2.add_vline(x=-var95, line_dash="dash", line_color="red",
                    annotation_text="95% VaR threshold")
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, width='stretch')

    st.caption(
        "Note: methods converge closely here because the data feeding this "
        "engine is currently synthetic and approximately normal. Expect "
        "more divergence (especially at 99%) once running on real market data."
    )


# ---------------------------------------------------------------------- #
# Page 3: Backtesting
# ---------------------------------------------------------------------- #
def page_backtesting():
    st.header("3. Backtesting")
    if "engine" not in st.session_state:
        st.info("Build a portfolio on Page 1 first.")
        return

    pm = st.session_state.engine["pm"]
    confidence = st.select_slider("VaR confidence level to backtest", [0.90, 0.95, 0.99], value=0.95)
    window = st.slider("Rolling estimation window (trading days)", 60, 504, 252, step=21)

    if len(pm.portfolio_returns) <= window:
        st.error("Not enough history for this window length. Reduce the window or extend the start date.")
        return

    with st.spinner("Running rolling out-of-sample backtest..."):
        backtest = rolling_var_backtest(pm.portfolio_returns, confidence=confidence, window=window)

    n_violations = int(backtest["violation"].sum())
    obs_rate = backtest["violation"].mean() * 100
    expected_rate = (1 - confidence) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Out-of-sample days", len(backtest))
    c2.metric("Violations", n_violations, delta=f"{obs_rate - expected_rate:+.2f}pp vs target")
    c3.metric("Target breach rate", f"{expected_rate:.1f}%")

    kupiec = KupiecTest(backtest["violation"].values, confidence=confidence).result()
    christoffersen = ChristoffersenTest(backtest["violation"].values).result()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Kupiec (Proportion of Failures)")
        st.write(f"**Verdict: {kupiec['verdict']}**")
        st.json(kupiec, expanded=False)
    with col2:
        st.subheader("Christoffersen (Independence)")
        st.write(f"**Verdict: {christoffersen['verdict']}**")
        st.json(christoffersen, expanded=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest.index, y=backtest["actual_return"] * 100,
                              mode="lines", name="Actual return", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=backtest.index, y=-backtest["forecast_var"] * 100,
                              mode="lines", name="VaR threshold (forecast)",
                              line=dict(color="red", dash="dash")))
    breaches = backtest[backtest["violation"]]
    fig.add_trace(go.Scatter(x=breaches.index, y=breaches["actual_return"] * 100,
                              mode="markers", name="Violation", marker=dict(color="red", size=7)))
    fig.update_layout(title="Out-of-sample VaR vs Actual Returns", yaxis_title="Return (%)")
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------- #
# Page 4: Risk Attribution
# ---------------------------------------------------------------------- #
def page_risk_attribution():
    st.header("4. Risk Attribution")
    if "engine" not in st.session_state:
        st.info("Build a portfolio on Page 1 first.")
        return

    eng = st.session_state.engine
    returns, pm = eng["returns"], eng["pm"]
    confidence = st.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95)

    ra = RiskAttribution(returns, pm.weights)
    table = ra.attribution_table(confidence)

    comp_sum, total = ra.verify_decomposition(confidence)
    st.caption(f"Decomposition check: sum of component VaRs = {comp_sum*100:.4f}%, "
               f"total portfolio VaR = {total*100:.4f}% "
               f"({'✓ exact match' if np.isclose(comp_sum, total) else '✗ mismatch -- check inputs'})")

    st.dataframe(table, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(table.reset_index(), names="ticker", values="pct_of_total_risk",
                          title="Risk Contribution by Asset")
        st.plotly_chart(fig_pie, width='stretch')
    with c2:
        comp = table.reset_index()[["ticker", "weight", "pct_of_total_risk"]].copy()
        comp["weight"] = comp["weight"] * 100
        comp = comp.melt(id_vars="ticker", var_name="Metric", value_name="Pct")
        fig_bar = px.bar(comp, x="ticker", y="Pct", color="Metric", barmode="group",
                          title="Capital Weight vs. Risk Contribution")
        st.plotly_chart(fig_bar, width='stretch')

    st.caption(
        "The point of this page: an asset can hold a small share of capital "
        "but a large share of risk (concentration via correlation), or vice "
        "versa (diversification benefit). Compare the two bars per asset above."
    )


# ---------------------------------------------------------------------- #
# Page 5: Stress Testing
# ---------------------------------------------------------------------- #
def page_stress_testing():
    st.header("5. Stress Testing")
    if "engine" not in st.session_state:
        st.info("Build a portfolio on Page 1 first.")
        return

    eng = st.session_state.engine
    weights, initial_value = eng["weights"], eng["initial_value"]
    pm = eng["pm"]
    current_value = pm.portfolio_value.iloc[-1]

    st.caption(
        "Shock magnitudes are documented historical approximations (see "
        "src/risk/stress_testing.py for details and caveats), not statistically "
        "estimated -- this deliberately bypasses the VaR model to show what "
        "happens in genuine regime-shift events."
    )

    st_tester = StressTester(weights, current_value)
    summary = st_tester.run_all()
    st.dataframe(summary, width='stretch')

    fig = px.bar(summary.reset_index(), x="scenario", y="portfolio_return_pct",
                 title="Portfolio Return Under Each Stress Scenario", color="scenario")
    fig.update_layout(showlegend=False, yaxis_title="Return (%)")
    st.plotly_chart(fig, width='stretch')

    scenario_choice = st.selectbox("Scenario detail", list(SCENARIOS.keys()))
    detail = st_tester.apply_scenario(scenario_choice)
    detail_df = pd.DataFrame(detail["per_asset"]).T
    st.subheader(f"Breakdown: {scenario_choice}")
    st.dataframe(detail_df, width='stretch')
    st.metric("Total stressed portfolio loss",
              fmt_money(detail['portfolio_loss_amount']),
              delta=f"{detail['portfolio_return_pct']}%")


# ---------------------------------------------------------------------- #
def main():
    page, data_mode, start_date, initial_value = sidebar_config()

    if page == "1. Portfolio Builder":
        page_portfolio_builder(data_mode, start_date, initial_value)
    elif page == "2. Risk Analytics":
        page_risk_analytics()
    elif page == "3. Backtesting":
        page_backtesting()
    elif page == "4. Risk Attribution":
        page_risk_attribution()
    elif page == "5. Stress Testing":
        page_stress_testing()


if __name__ == "__main__":
    main()

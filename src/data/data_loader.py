"""
MarketDataLoader
-----------------
Loads daily adjusted close prices for a portfolio of tickers.

Two modes:
1. LIVE  - uses yfinance to pull real historical prices (requires open internet).
2. SYNTH - generates realistic synthetic multi-asset price paths via a
           correlated geometric Brownian motion model. Used for offline
           development/testing of the risk engine, and as a safety net if
           a live data pull fails (missing ticker, API hiccup, no internet).

The rest of the QuantRisk engine (portfolio, VaR, ES, backtesting,
attribution) only ever sees a clean DataFrame of adjusted close prices,
so it doesn't care which mode produced it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketDataLoader:
    def __init__(self, tickers: list[str], start: str, end: str | None = None, seed: int = 42):
        self.tickers = tickers
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.seed = seed

    # ------------------------------------------------------------------ #
    # LIVE DATA (run this locally / wherever yfinance has internet access)
    # ------------------------------------------------------------------ #
    def load_live(self) -> pd.DataFrame:
        """Fetch real adjusted close prices via yfinance."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from exc

        data = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )

        if data is None or data.empty:
            raise ValueError(
                f"yfinance returned no data at all for {self.tickers} "
                f"({self.start} to {self.end}). This usually means: no internet "
                f"access, Yahoo Finance is rate-limiting/blocking this network, "
                f"or yfinance needs upgrading (`pip install -U yfinance`)."
            )

        # yfinance returns a MultiIndex column structure for multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            # Single ticker case
            prices = data[["Close"]]
            prices.columns = self.tickers

        prices = prices[self.tickers]  # enforce consistent column order
        prices = prices.dropna(how="all")

        # Validate every ticker actually has usable data -- catch the case
        # where the overall download "succeeded" but one ticker silently
        # came back empty/all-NaN (a real yfinance failure mode).
        bad_tickers = [t for t in self.tickers if prices[t].notna().sum() < 2]
        if bad_tickers:
            raise ValueError(
                f"No usable price data returned for: {bad_tickers}. "
                f"Check the ticker symbols are correct, or try again "
                f"(Yahoo Finance occasionally rate-limits/blocks requests)."
            )

        return prices

    # ------------------------------------------------------------------ #
    # SYNTHETIC DATA (works anywhere, no network needed)
    # ------------------------------------------------------------------ #
    def load_synthetic(self) -> pd.DataFrame:
        """
        Generate correlated synthetic price paths.

        Each asset class gets a plausible annualized drift/vol, and assets
        are correlated via a realistic-ish correlation structure so the
        risk engine has non-trivial diversification effects to chew on.
        """
        rng = np.random.default_rng(self.seed)

        dates = pd.bdate_range(start=self.start, end=self.end)
        n_days = len(dates)
        n_assets = len(self.tickers)

        # Rough annualized (drift, vol) priors by common ticker, fallback to generic equity
        priors = {
            "SPY": (0.09, 0.16), "QQQ": (0.13, 0.22), "GLD": (0.05, 0.14),
            "TLT": (0.02, 0.13), "USO": (0.00, 0.35), "BTC-USD": (0.30, 0.65),
            "EEM": (0.06, 0.20),
        }
        default_prior = (0.08, 0.20)

        mu = np.array([priors.get(t, default_prior)[0] for t in self.tickers])
        sigma = np.array([priors.get(t, default_prior)[1] for t in self.tickers])

        # Build a plausible correlation matrix: equities co-move, gold/bonds
        # are defensive (mild negative/low corr with equities), crypto/oil are noisy.
        corr = np.full((n_assets, n_assets), 0.25)
        np.fill_diagonal(corr, 1.0)
        defensive = {"GLD", "TLT"}
        for i, ti in enumerate(self.tickers):
            for j, tj in enumerate(self.tickers):
                if i == j:
                    continue
                if ti in defensive or tj in defensive:
                    corr[i, j] = -0.10 if (ti in defensive) != (tj in defensive) else 0.30
                if ti == "BTC-USD" or tj == "BTC-USD":
                    corr[i, j] = 0.05

        # Ensure symmetric + PSD-ish (clip eigenvalues if needed)
        corr = (corr + corr.T) / 2
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 1e-4, None)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

        daily_mu = mu / 252
        daily_sigma = sigma / np.sqrt(252)
        cov = np.outer(daily_sigma, daily_sigma) * corr

        log_returns = rng.multivariate_normal(daily_mu, cov, size=n_days)

        prices_start = 100.0
        log_prices = np.cumsum(log_returns, axis=0) + np.log(prices_start)
        prices = np.exp(log_prices)

        df = pd.DataFrame(prices, index=dates, columns=self.tickers)
        return df

    # ------------------------------------------------------------------ #
    def load(self, mode: str = "auto") -> pd.DataFrame:
        """
        mode: 'live' | 'synthetic' | 'auto'
        'auto' tries live first, falls back to synthetic on any failure.
        """
        if mode == "synthetic":
            return self.load_synthetic()
        if mode == "live":
            return self.load_live()
        # auto
        try:
            return self.load_live()
        except Exception:
            return self.load_synthetic()


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "GLD", "TLT", "USO"]
    loader = MarketDataLoader(tickers, start="2019-01-01")
    prices = loader.load(mode="synthetic")
    print(prices.head())
    print(prices.tail())
    print(f"\nShape: {prices.shape}")

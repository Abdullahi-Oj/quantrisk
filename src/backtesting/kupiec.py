"""
KupiecTest (Proportion of Failures / POF test)
-------------------------------------------------
Tests whether a VaR model's breach rate matches its target rate.
e.g. a 95% VaR model SHOULD be breached ~5% of the time -- not
materially more (model understates risk) and not materially less
(model is overly conservative, wasting capital).

Likelihood ratio statistic:
    LR_pof = -2 * ln[ (1-p)^(n-x) * p^x / ((1 - x/n)^(n-x) * (x/n)^x) ]

where:
    p = expected violation rate = 1 - confidence
    n = number of observations
    x = number of observed violations (breaches)

Under H0 (model is correctly calibrated), LR_pof ~ chi-squared(1 df).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


class KupiecTest:
    def __init__(self, violations: np.ndarray, confidence: float = 0.95):
        """
        violations: boolean array/series, True where actual loss breached VaR
        confidence: the VaR confidence level being tested (e.g. 0.95)
        """
        self.violations = np.asarray(violations, dtype=bool)
        self.confidence = confidence
        self.p = 1 - confidence  # expected violation rate
        self.n = len(self.violations)
        self.x = int(self.violations.sum())  # observed violations

    def likelihood_ratio(self) -> float:
        n, x, p = self.n, self.x, self.p
        if n == 0:
            return np.nan

        x_rate = x / n

        # Handle edge cases (0 or all violations) where log(0) would blow up
        if x == 0:
            log_likelihood_h0 = n * np.log(1 - p)
            log_likelihood_h1 = n * np.log(1 - x_rate) if x_rate < 1 else 0.0
        elif x == n:
            log_likelihood_h0 = n * np.log(p)
            log_likelihood_h1 = n * np.log(x_rate)
        else:
            log_likelihood_h0 = (n - x) * np.log(1 - p) + x * np.log(p)
            log_likelihood_h1 = (n - x) * np.log(1 - x_rate) + x * np.log(x_rate)

        lr = -2 * (log_likelihood_h0 - log_likelihood_h1)
        return max(lr, 0.0)  # numerical guard against tiny negative values

    def p_value(self) -> float:
        lr = self.likelihood_ratio()
        if np.isnan(lr):
            return np.nan
        return 1 - chi2.cdf(lr, df=1)

    def result(self, significance: float = 0.05) -> dict:
        lr = self.likelihood_ratio()
        pval = self.p_value()
        passed = pval > significance if not np.isnan(pval) else None

        return {
            "n_obs": self.n,
            "n_violations": self.x,
            "expected_violations": round(self.n * self.p, 1),
            "observed_rate_pct": round((self.x / self.n) * 100, 2) if self.n else None,
            "expected_rate_pct": round(self.p * 100, 2),
            "lr_statistic": round(lr, 4) if not np.isnan(lr) else None,
            "p_value": round(pval, 4) if not np.isnan(pval) else None,
            "verdict": "PASS" if passed else ("FAIL" if passed is not None else "N/A"),
            "interpretation": self._interpret(passed),
        }

    def _interpret(self, passed: bool | None) -> str:
        if passed is None:
            return "Insufficient data"
        if passed:
            return "Breach rate is statistically consistent with the model's target confidence level"
        observed = self.x / self.n if self.n else 0
            
        if observed > self.p:
            return "Model UNDERSTATES risk -- too many breaches (dangerous: real losses exceed predicted VaR more often than expected)"
        return "Model OVERSTATES risk -- too few breaches (conservative: ties up more capital than necessary)"


if __name__ == "__main__":
    # Sanity check with synthetic violation sequences
    rng = np.random.default_rng(0)

    print("--- Well-calibrated model (5% true breach rate, 95% VaR) ---")
    well_calibrated = rng.random(1000) < 0.05
    print(KupiecTest(well_calibrated, confidence=0.95).result())

    print("\n--- Bad model: too many breaches (15% actual vs 5% expected) ---")
    too_many = rng.random(1000) < 0.15
    print(KupiecTest(too_many, confidence=0.95).result())

    print("\n--- Overly conservative model: too few breaches (1% actual vs 5% expected) ---")
    too_few = rng.random(1000) < 0.01
    print(KupiecTest(too_few, confidence=0.95).result())

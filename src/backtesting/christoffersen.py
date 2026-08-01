"""
ChristoffersenTest (Independence test)
----------------------------------------
Kupiec's test only checks the TOTAL breach count. It can't tell the
difference between:
    - 50 breaches spread randomly across 1000 days  (healthy model)
    - 50 breaches all clustered in one crisis month (model fails
      precisely when it matters most -- during stress)

Christoffersen's test checks whether breaches are independent over
time by counting state transitions in the violation sequence:
    n00 = no-violation -> no-violation
    n01 = no-violation -> violation
    n10 = violation    -> no-violation
    n11 = violation    -> violation

If violations are independent, the probability of a violation tomorrow
shouldn't depend on whether there was one today (pi0 should equal pi1).
Clustering shows up as pi1 >> pi0.

LR_ind ~ chi-squared(1 df) under H0 (independence holds).

The combined Conditional Coverage test:
    LR_cc = LR_pof (Kupiec) + LR_ind (Christoffersen) ~ chi-squared(2 df)
tests BOTH correct unconditional coverage AND independence at once --
this is the test regulators actually rely on.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .kupiec import KupiecTest


class ChristoffersenTest:
    def __init__(self, violations: np.ndarray):
        self.violations = np.asarray(violations, dtype=int)

    def _transition_counts(self) -> dict:
        v = self.violations
        n00 = n01 = n10 = n11 = 0
        for t in range(len(v) - 1):
            cur, nxt = v[t], v[t + 1]
            if cur == 0 and nxt == 0:
                n00 += 1
            elif cur == 0 and nxt == 1:
                n01 += 1
            elif cur == 1 and nxt == 0:
                n10 += 1
            else:
                n11 += 1
        return {"n00": n00, "n01": n01, "n10": n10, "n11": n11}

    def likelihood_ratio(self) -> float:
        c = self._transition_counts()
        n00, n01, n10, n11 = c["n00"], c["n01"], c["n10"], c["n11"]

        n0_total = n00 + n01  # days following a "no violation"
        n1_total = n10 + n11  # days following a "violation"
        n_total = n0_total + n1_total

        if n0_total == 0 or n1_total == 0 or n_total == 0:
            return 0.0  # not enough transitions of one type to test independence

        pi0 = n01 / n0_total if n0_total else 0  # P(violation | no violation yesterday)
        pi1 = n11 / n1_total if n1_total else 0  # P(violation | violation yesterday)
        pi = (n01 + n11) / n_total                # unconditional P(violation)

        def _safe_log_term(count, prob):
            if count == 0:
                return 0.0
            return count * np.log(prob) if prob > 0 else 0.0

        log_l_h0 = (
            _safe_log_term(n00, 1 - pi) + _safe_log_term(n01, pi)
            + _safe_log_term(n10, 1 - pi) + _safe_log_term(n11, pi)
        )
        log_l_h1 = (
            _safe_log_term(n00, 1 - pi0) + _safe_log_term(n01, pi0)
            + _safe_log_term(n10, 1 - pi1) + _safe_log_term(n11, pi1)
        )

        lr = -2 * (log_l_h0 - log_l_h1)
        return max(lr, 0.0)

    def p_value(self) -> float:
        return 1 - chi2.cdf(self.likelihood_ratio(), df=1)

    def result(self, significance: float = 0.05) -> dict:
        lr = self.likelihood_ratio()
        pval = self.p_value()
        passed = pval > significance
        c = self._transition_counts()

        return {
            **c,
            "lr_statistic": round(lr, 4),
            "p_value": round(pval, 4),
            "verdict": "PASS" if passed else "FAIL",
            "interpretation": (
                "Violations appear independent over time"
                if passed else
                "Violations are CLUSTERED -- model tends to fail in streaks, "
                "likely missing volatility regime shifts"
            ),
        }

    @staticmethod
    def conditional_coverage(violations: np.ndarray, confidence: float = 0.95, significance: float = 0.05) -> dict:
        """Combined test: LR_pof + LR_ind ~ chi2(2 df)."""
        kupiec = KupiecTest(violations, confidence)
        christoffersen = ChristoffersenTest(violations)

        lr_cc = kupiec.likelihood_ratio() + christoffersen.likelihood_ratio()
        pval = 1 - chi2.cdf(lr_cc, df=2)
        passed = pval > significance

        return {
            "lr_pof": round(kupiec.likelihood_ratio(), 4),
            "lr_ind": round(christoffersen.likelihood_ratio(), 4),
            "lr_cc": round(lr_cc, 4),
            "p_value": round(pval, 4),
            "verdict": "PASS" if passed else "FAIL",
        }


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("--- Independent violations (random 5% breach rate) ---")
    independent = (rng.random(1000) < 0.05).astype(int)
    print(ChristoffersenTest(independent).result())

    print("\n--- Clustered violations (same total count, but bunched) ---")
    clustered = np.zeros(1000, dtype=int)
    # Same total violations (~50) but concentrated into 5 ten-day crisis streaks
    for start in [100, 300, 500, 700, 900]:
        clustered[start:start + 10] = 1
    print(f"(total violations: {clustered.sum()})")
    print(ChristoffersenTest(clustered).result())

    print("\n--- Combined Conditional Coverage test on the clustered case ---")
    print(ChristoffersenTest.conditional_coverage(clustered, confidence=0.95))

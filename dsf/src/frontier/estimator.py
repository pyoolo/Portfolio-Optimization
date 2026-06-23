"""
Chapter 1: The Duration–Spread Frontier as a Quantile Regression Object
========================================================================

The frontier is estimated as the conditional upper quantile of spread given
duration. Unlike mean regression, which estimates E[spread | duration], we
estimate Q_τ[spread | duration] for τ ∈ {0.85, 0.90, 0.95}.

Functional form (motivated by Merton 1974):
    log(spread) = α + β·log(duration⁺) + ε
    → spread = exp(α) · duration^β

where duration⁺ = max(duration, ε) to handle near-zero/negative durations.

Key result: β < 0 (higher duration → lower spread on the frontier),
and the frontier is well-identified across τ ∈ [0.85, 0.95].
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from scipy.stats import kstest
from typing import Tuple, Dict, Optional
import warnings


# ── Functional forms ──────────────────────────────────────────────────────────

def hyperbolic(duration: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Frontier functional form: spread = a / duration^b + c"""
    d = np.clip(duration, 0.25, None)
    return a / (d ** b) + c


def log_linear(log_dur: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Log-linearized form: log(spread) = alpha + beta * log(duration)"""
    return alpha + beta * log_dur


# ── Quantile regression estimator ─────────────────────────────────────────────

class FrontierEstimator:
    """
    Estimates the duration–spread frontier via quantile regression.

    Parameters
    ----------
    tau : float
        Quantile level. 0.90 recommended (90th percentile of spread | duration).
    min_duration : float
        Bonds with duration below this threshold are excluded from estimation
        (distressed/floaters distort the frontier for IG/HY).
    """

    def __init__(self, tau: float = 0.90, min_duration: float = 0.25):
        self.tau = tau
        self.min_duration = min_duration
        self.params_: Optional[Dict] = None
        self.qr_result_ = None

    def fit(self, df: pd.DataFrame) -> "FrontierEstimator":
        """
        Fit the frontier to a bond cross-section.

        Parameters
        ----------
        df : DataFrame with columns 'duration' and 'spread'
        """
        mask = df['duration'] > self.min_duration
        data = df[mask].copy()
        data['log_dur'] = np.log(data['duration'])
        data['log_spread'] = np.log(data['spread'].clip(lower=1))

        model = smf.quantreg('log_spread ~ log_dur', data=data)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.fit(q=self.tau, max_iter=2000)

        self.qr_result_ = result
        alpha = result.params['Intercept']
        beta = result.params['log_dur']

        self.params_ = {
            'alpha': alpha,
            'beta': beta,
            'a': np.exp(alpha),
            'tau': self.tau,
        }
        return self

    def predict(self, duration: np.ndarray) -> np.ndarray:
        """
        Evaluate the frontier at given duration values.

        Returns spread in basis points.
        """
        if self.params_ is None:
            raise RuntimeError("Call .fit() first")
        dur = np.clip(duration, self.min_duration, None)
        return np.exp(self.params_['alpha'] + self.params_['beta'] * np.log(dur))

    def frontier_summary(self) -> pd.DataFrame:
        """Return a compact summary of the estimated frontier."""
        if self.params_ is None:
            raise RuntimeError("Call .fit() first")
        p = self.params_
        return pd.DataFrame([{
            'tau': p['tau'],
            'alpha': round(p['alpha'], 4),
            'beta (slope in log-log)': round(p['beta'], 4),
            'a = exp(alpha)': round(p['a'], 2),
            'interpretation': f"spread ∝ duration^{p['beta']:.3f}",
        }])

    def estimate_all_quantiles(
        self, df: pd.DataFrame, taus: Tuple[float, ...] = (0.75, 0.85, 0.90, 0.95)
    ) -> pd.DataFrame:
        """
        Fit frontier at multiple quantile levels and return parameter table.
        Useful to show frontier robustness.
        """
        rows = []
        mask = df['duration'] > self.min_duration
        data = df[mask].copy()
        data['log_dur'] = np.log(data['duration'])
        data['log_spread'] = np.log(data['spread'].clip(lower=1))

        for tau in taus:
            model = smf.quantreg('log_spread ~ log_dur', data=data)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = model.fit(q=tau, max_iter=2000)
            rows.append({
                'tau': tau,
                'alpha': res.params['Intercept'],
                'beta': res.params['log_dur'],
                'a': np.exp(res.params['Intercept']),
                'pseudo_r2': res.prsquared,
            })

        return pd.DataFrame(rows).round(4)


# ── Isocurve analysis ─────────────────────────────────────────────────────────

def dts_isocurves(dts_levels: np.ndarray, duration_grid: np.ndarray) -> pd.DataFrame:
    """
    Compute DTS isocurves in the (duration, spread) plane.

    DTS = spread × spread_duration ≈ spread × duration (for fixed-rate bonds)
    → spread = DTS / duration  (rectangular hyperbola)

    These isocurves are the natural coordinates of the Ben Dor et al. (2007) framework.
    """
    rows = []
    for dts in dts_levels:
        spreads = dts / np.clip(duration_grid, 0.1, None)
        for d, s in zip(duration_grid, spreads):
            rows.append({'dts': dts, 'duration': d, 'spread': s})
    return pd.DataFrame(rows)


# ── Frontier distance ─────────────────────────────────────────────────────────

def frontier_distance(df: pd.DataFrame, estimator: FrontierEstimator) -> pd.Series:
    """
    Compute signed distance of each bond from the frontier (in log-spread space).

        δᵢ = log(spread_i) - log(frontier(duration_i))

    δ > 0 → bond is above the frontier (unusual; potential data error or extreme stress)
    δ < 0 → bond is below the frontier (normal)
    δ ≈ 0 → bond is on the frontier

    Note: this is NOT a mispricing signal. It is a geometric characterization
    of where each bond sits relative to the envelope of its peer universe.
    """
    frontier_spread = estimator.predict(df['duration'].values)
    return np.log(df['spread'].values) - np.log(frontier_spread)


# ── Temporal frontier ─────────────────────────────────────────────────────────

def estimate_panel_frontier(
    panel: pd.DataFrame,
    tau: float = 0.90,
) -> pd.DataFrame:
    """
    Estimate frontier parameters for each period in a panel dataset.

    Returns a DataFrame with columns:
        period, alpha, beta, a, stress_index
    """
    rows = []
    for period, grp in panel.groupby('period'):
        est = FrontierEstimator(tau=tau)
        try:
            est.fit(grp)
            p = est.params_
            stress = grp['stress_index'].iloc[0] if 'stress_index' in grp.columns else np.nan
            rows.append({
                'period': period,
                'alpha': p['alpha'],
                'beta': p['beta'],
                'a': p['a'],
                'stress_index': stress,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


if __name__ == '__main__':
    from utils.synthetic import generate_universe, generate_panel

    print("=" * 60)
    print("Chapter 1: Frontier Estimation")
    print("=" * 60)

    df = generate_universe()

    est = FrontierEstimator(tau=0.90)
    est.fit(df)
    print("\nFrontier summary (τ = 0.90):")
    print(est.frontier_summary().to_string(index=False))

    print("\nRobustness across quantile levels:")
    print(est.estimate_all_quantiles(df).to_string(index=False))

    # Panel dynamics
    print("\nEstimating panel frontier across 60 periods...")
    panel = generate_panel(n_periods=60)
    panel_params = estimate_panel_frontier(panel)
    corr = panel_params['beta'].corr(panel_params['stress_index'])
    print(f"Correlation(β, stress_index) = {corr:.3f}")
    print("  → Negative correlation expected: in stress, frontier steepens")
    print("    (short-duration bonds widen more than long-duration bonds)")

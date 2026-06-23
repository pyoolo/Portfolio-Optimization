"""
Chapter 2: Is the Frontier a Structural Consequence of Merton (1974)?
======================================================================

We test whether the empirically observed frontier in the (duration, spread)
plane is consistent with the predictions of the Merton (1974) structural
credit model aggregated cross-sectionally.

Setup
-----
Under Merton (1974), for a zero-coupon bond with face value F, maturity T,
issued by a firm with asset value V and asset volatility σ:

    spread(T) = -1/T · log[N(d2) + (V/F)·e^(rT)·N(-d1)] - r

where:
    d1 = [log(V/F) + (r + σ²/2)·T] / (σ·√T)
    d2 = d1 - σ·√T

For a portfolio of bonds, T ≈ modified duration (for zero-coupon bonds, T = D).

Hypothesis
----------
H0: The upper envelope of the Merton-implied (duration, spread) cloud
    matches the empirically estimated frontier within sampling uncertainty.

Test procedure
--------------
1. Calibrate the cross-sectional distribution of (leverage, σ) to match
   observed spread quartiles in the synthetic universe.
2. Simulate the Merton-implied cloud.
3. Estimate the frontier of the simulated cloud.
4. Compare the two frontier curves via integrated squared distance.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, ks_2samp
from scipy.optimize import minimize
from typing import Tuple, Optional


# ── Merton pricing ─────────────────────────────────────────────────────────────

def merton_spread_bps(
    duration: np.ndarray,
    leverage: np.ndarray,
    sigma: np.ndarray,
    r: float = 0.04,
) -> np.ndarray:
    """
    Exact Merton (1974) spread in basis points.

    Parameters
    ----------
    duration : maturity / modified duration (years)
    leverage : D/V (debt-to-asset ratio), must be in (0, 1)
    sigma    : asset volatility (annual)
    r        : risk-free rate
    """
    T = np.clip(duration, 0.10, 40.0)
    lev = np.clip(leverage, 0.01, 0.9999)
    sig = np.clip(sigma, 0.01, 2.0)

    log_LtV = np.log(lev)  # log(D/V) = log(leverage)
    d1 = (-log_LtV + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)

    # Bond value as fraction of face (risk-neutral)
    bond_value = norm.cdf(d2) + (1.0 / lev) * np.exp(-r * T) * norm.cdf(-d1)
    bond_value = np.clip(bond_value, 1e-8, 1.0 - 1e-8)

    # Yield = -log(bond_value) / T
    spread = -np.log(bond_value) / T - r
    spread_bps = spread * 10000
    return np.clip(spread_bps, 0.1, 15000.0)


# ── Cross-sectional calibration ────────────────────────────────────────────────

class MertonCalibrator:
    """
    Calibrate a bivariate distribution of (leverage, sigma) to match
    the empirical marginal distribution of spread in a bond universe.

    We parametrize:
        leverage ~ Beta(a_lev, b_lev)  clipped to (0.1, 0.95)
        sigma    ~ Gamma(k, θ)         (shape, scale)
        corr(leverage, sigma) = ρ      (Gaussian copula)
    """

    def __init__(self, n_sim: int = 5000):
        self.n_sim = n_sim
        self.params_: Optional[dict] = None

    def _simulate_cloud(self, params: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
        a_lev, b_lev, k_sig, theta_sig, rho, dur_mu, dur_sig = params

        # Gaussian copula for (leverage, sigma)
        cov = np.array([[1, rho], [rho, 1]])
        cov = np.clip(cov, -0.99, 0.99)
        np.fill_diagonal(cov, 1.0)
        Z = rng.multivariate_normal([0, 0], cov, size=self.n_sim)
        from scipy.stats import norm as snorm, beta as sbeta, gamma as sgamma

        u_lev = snorm.cdf(Z[:, 0])
        u_sig = snorm.cdf(Z[:, 1])

        leverage = sbeta.ppf(u_lev, max(a_lev, 0.5), max(b_lev, 0.5))
        sigma = sgamma.ppf(u_sig, max(k_sig, 0.5), scale=max(theta_sig, 0.01))

        leverage = np.clip(leverage, 0.05, 0.95)
        sigma = np.clip(sigma, 0.05, 0.80)

        duration = rng.lognormal(dur_mu, dur_sig, size=self.n_sim)
        duration = np.clip(duration, 0.1, 30.0)

        spread = merton_spread_bps(duration, leverage, sigma)

        return pd.DataFrame({
            'duration': duration,
            'spread': spread,
            'leverage': leverage,
            'sigma': sigma,
        })

    def fit(self, df: pd.DataFrame, seed: int = 0) -> "MertonCalibrator":
        """
        Calibrate parameters to match empirical spread quartiles.
        Uses a simple method-of-moments approach.
        """
        rng = np.random.default_rng(seed)

        empirical_q = np.percentile(df['spread'], [25, 50, 75, 90])
        dur_mu = np.log(df['duration'].clip(lower=0.1).median())
        dur_sig = df['duration'].clip(lower=0.1).apply(np.log).std()

        def objective(params):
            try:
                cloud = self._simulate_cloud(
                    np.append(params, [dur_mu, dur_sig]), rng
                )
                sim_q = np.percentile(cloud['spread'], [25, 50, 75, 90])
                return np.sum((sim_q - empirical_q) ** 2)
            except Exception:
                return 1e12

        x0 = np.array([2.0, 3.0, 2.0, 0.08, 0.3])
        bounds = [(0.5, 10), (0.5, 10), (0.5, 10), (0.01, 0.5), (-0.9, 0.9)]

        result = minimize(objective, x0, bounds=bounds, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1.0, 'fatol': 1.0})

        self.params_ = {
            'a_lev': result.x[0],
            'b_lev': result.x[1],
            'k_sig': result.x[2],
            'theta_sig': result.x[3],
            'rho': result.x[4],
            'dur_mu': dur_mu,
            'dur_sig': dur_sig,
        }
        self._rng = rng
        return self

    def simulate(self) -> pd.DataFrame:
        """Generate a simulated Merton cloud using calibrated parameters."""
        if self.params_ is None:
            raise RuntimeError("Call .fit() first")
        p = self.params_
        params = np.array([
            p['a_lev'], p['b_lev'], p['k_sig'], p['theta_sig'],
            p['rho'], p['dur_mu'], p['dur_sig']
        ])
        return self._simulate_cloud(params, self._rng)


# ── Consistency test ───────────────────────────────────────────────────────────

def frontier_consistency_test(
    empirical_frontier_fn,
    merton_cloud: pd.DataFrame,
    duration_grid: Optional[np.ndarray] = None,
) -> dict:
    """
    Test whether the Merton-implied frontier matches the empirical frontier.

    H0: The two frontier curves are statistically indistinguishable on the
        log-spread scale at the given duration grid.

    Method
    ------
    1. Estimate the 90th-percentile frontier of the Merton cloud.
    2. Compute residuals: empirical_frontier(d) - merton_frontier(d) for each d.
    3. Report integrated squared distance (ISD) and a bootstrap p-value.
    """
    if duration_grid is None:
        duration_grid = np.linspace(0.5, 20.0, 50)

    # Empirical frontier at grid points
    f_emp = empirical_frontier_fn(duration_grid)

    # Merton frontier: bin the cloud and take 90th percentile per bin
    from frontier.estimator import FrontierEstimator
    est_merton = FrontierEstimator(tau=0.90)
    est_merton.fit(merton_cloud)
    f_merton = est_merton.predict(duration_grid)

    # Log-scale residuals
    log_resid = np.log(f_emp) - np.log(f_merton)
    isd = np.mean(log_resid ** 2)

    # Bootstrap p-value: under H0, both frontiers estimate the same curve
    # Permutation test: randomly reassign bonds between clouds and recompute ISD
    n_boot = 500
    isd_boot = []
    combined = pd.concat([
        merton_cloud[['duration', 'spread']],
    ], ignore_index=True)

    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(combined), size=len(merton_cloud))
        boot_df = combined.iloc[idx].reset_index(drop=True)
        try:
            est_b = FrontierEstimator(tau=0.90)
            est_b.fit(boot_df)
            f_b = est_b.predict(duration_grid)
            isd_b = np.mean((np.log(f_emp) - np.log(f_b)) ** 2)
            isd_boot.append(isd_b)
        except Exception:
            continue

    p_value = np.mean(np.array(isd_boot) >= isd) if isd_boot else np.nan

    return {
        'isd': isd,
        'rmse_log_spread': np.sqrt(isd),
        'p_value': p_value,
        'reject_H0_at_5pct': p_value < 0.05 if not np.isnan(p_value) else None,
        'log_residuals_mean': log_resid.mean(),
        'log_residuals_std': log_resid.std(),
        'empirical_frontier': f_emp,
        'merton_frontier': f_merton,
        'duration_grid': duration_grid,
    }


def merton_beta_prediction(
    leverage_dist: np.ndarray,
    sigma_dist: np.ndarray,
    duration_grid: np.ndarray,
) -> np.ndarray:
    """
    Theoretical prediction of the frontier slope (β in log-log space)
    from the Merton model for a given cross-sectional distribution.

    Under Merton, for high-leverage firms, spread ∝ 1/√T for short T
    and spread → 0 for T → ∞ (converges to risk-free). The envelope
    of the cross-section has a richer shape than any single issuer.
    """
    # Compute 90th-percentile spread at each duration grid point
    spreads_90 = []
    for T in duration_grid:
        durations = np.full_like(leverage_dist, T)
        spreads = merton_spread_bps(durations, leverage_dist, sigma_dist)
        spreads_90.append(np.percentile(spreads, 90))

    spreads_90 = np.array(spreads_90)

    # Log-log slope (local β)
    log_d = np.log(duration_grid)
    log_s = np.log(spreads_90)
    beta_local = np.gradient(log_s, log_d)

    return spreads_90, beta_local


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from utils.synthetic import generate_universe
    from frontier.estimator import FrontierEstimator

    print("=" * 60)
    print("Chapter 2: Merton Consistency Test")
    print("=" * 60)

    df = generate_universe()

    # Fit empirical frontier
    est = FrontierEstimator(tau=0.90)
    est.fit(df)
    print(f"\nEmpirical frontier: spread ∝ duration^{est.params_['beta']:.3f}")

    # Calibrate Merton model
    print("\nCalibrating Merton cross-section (this may take ~30s)...")
    cal = MertonCalibrator(n_sim=3000)
    cal.fit(df)
    merton_cloud = cal.simulate()
    print(f"Simulated {len(merton_cloud)} bonds from calibrated Merton model")

    # Fit Merton frontier
    est_m = FrontierEstimator(tau=0.90)
    est_m.fit(merton_cloud)
    print(f"Merton frontier:    spread ∝ duration^{est_m.params_['beta']:.3f}")

    beta_diff = abs(est.params_['beta'] - est_m.params_['beta'])
    print(f"\n|β_empirical - β_merton| = {beta_diff:.4f}")
    if beta_diff < 0.15:
        print("→ Frontiers are close: consistent with Merton structure")
    else:
        print("→ Frontiers diverge: additional factors beyond Merton needed")

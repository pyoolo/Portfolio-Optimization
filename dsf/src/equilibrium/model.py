"""
Chapter 3: The Frontier as an Issuance Equilibrium
====================================================

Why is the upper-left region of the (duration, spread) plane structurally
empty? This module formalizes the answer as a separating equilibrium.

Intuition
---------
A distressed issuer (high credit risk) wants to minimize borrowing costs.
It could issue long-dated debt, but the market demands a term premium for
long-dated distressed debt that grows faster than the issuer's benefit
from locking in funding. At some duration threshold D*(quality), the
issuer is indifferent between issuing and not issuing. The locus of
these thresholds across the cross-section of issuers traces the frontier.

Formal setup (Diamond 1991 style)
----------------------------------
- Issuer quality θ ∈ [0,1] (1 = best)
- Issuer benefit of long-term debt: B(T, θ) = b₀ · T · θ
  (refinancing risk avoided, strategic commitment)
- Market required spread for maturity T, quality θ:
  s(T, θ) = κ · (1-θ)^α / T^β + c
- Issuer cost: C(T, θ) = T · s(T, θ) · notional
- Issuer issues at maturity T if B(T, θ) ≥ C(T, θ)

Equilibrium condition (indifference):
    b₀ · T · θ = T · s(T*, θ)
    → s(T*, θ) = b₀ · θ
    → κ · (1-θ)^α / T*^β = b₀ · θ - c
    → T* = [κ · (1-θ)^α / (b₀ · θ - c)]^(1/β)

The frontier is: spread = b₀ · θ for T = T*(θ), parametrized by θ.

Key prediction: the frontier slope in log-log is β (same as Merton β),
and the frontier intercept is determined by the distribution of θ.
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq
from typing import Tuple, Optional, Callable


# ── Primitives ─────────────────────────────────────────────────────────────────

def market_spread(
    duration: np.ndarray,
    quality: np.ndarray,
    kappa: float = 500.0,
    alpha: float = 1.5,
    beta: float = 0.6,
    c: float = 20.0,
) -> np.ndarray:
    """
    Market-required spread for debt of maturity T issued by firm of quality θ.

    s(T, θ) = κ · (1-θ)^α / T^β + c

    Parameters
    ----------
    duration : maturity T (years)
    quality  : θ ∈ (0, 1], 1 = best quality
    kappa    : spread level parameter (controls the level of the frontier)
    alpha    : quality sensitivity (how fast spread rises with distress)
    beta     : duration sensitivity (how fast spread falls with duration)
    c        : floor spread (risk-free premium, transaction costs)
    """
    T = np.clip(duration, 0.10, 40.0)
    q = np.clip(quality, 0.001, 0.999)
    return kappa * (1 - q) ** alpha / T ** beta + c


def issuer_benefit(
    duration: np.ndarray,
    quality: np.ndarray,
    b0: float = 30.0,
) -> np.ndarray:
    """
    Issuer benefit per unit notional of issuing debt with maturity T.

    B(T, θ) = b₀ · T · θ

    Interpretation: better-quality firms benefit more from long-term
    commitment (strategic value, avoidance of rollover risk).
    """
    return b0 * duration * quality


def net_benefit(
    duration: float,
    quality: float,
    kappa: float = 500.0,
    alpha: float = 1.5,
    beta: float = 0.6,
    c: float = 20.0,
    b0: float = 30.0,
) -> float:
    """
    Net benefit to issuer: B(T, θ) - C(T, θ) where C = T · s(T, θ).

    Issuer participates iff net_benefit ≥ 0.
    """
    s = market_spread(np.array([duration]), np.array([quality]), kappa, alpha, beta, c)[0]
    b = issuer_benefit(np.array([duration]), np.array([quality]), b0)[0]
    cost = duration * s  # total interest cost per unit notional
    return b - cost


# ── Equilibrium frontier ───────────────────────────────────────────────────────

class IssuanceEquilibrium:
    """
    Computes the separating equilibrium frontier in the (duration, spread) plane.

    The frontier is the locus of (T*(θ), s(T*(θ), θ)) for θ ∈ (0, 1).

    Parameters
    ----------
    kappa, alpha, beta, c : market spread parameters
    b0 : issuer benefit rate
    """

    def __init__(
        self,
        kappa: float = 500.0,
        alpha: float = 1.5,
        beta: float = 0.6,
        c: float = 20.0,
        b0: float = 120.0,
    ):
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.b0 = b0

    def optimal_duration(self, quality: float) -> Optional[float]:
        """
        Find the maximum maturity T*(θ) an issuer of quality θ will issue.

        T*(θ) is the root of net_benefit(T, θ) = 0.
        If no root exists, the issuer cannot issue (return None).
        """
        if net_benefit(0.1, quality, self.kappa, self.alpha, self.beta, self.c, self.b0) < 0:
            return None  # Cannot even issue short-term

        # Check if net benefit is positive at some point
        # Net benefit at T → 0: b0·T·θ - T·(κ/T^β + c) = T·(b0·θ - c) - κ·T^(1-β)
        # For small T and β < 1: dominated by -κ·T^(1-β) → -∞ (bad for issuer)
        # For β > 1: at T → 0, net benefit → +∞ (issuer prefers short)

        try:
            T_star = brentq(
                lambda T: net_benefit(T, quality, self.kappa, self.alpha, self.beta, self.c, self.b0),
                0.1, 40.0,
                xtol=0.01, maxiter=100,
            )
            return T_star
        except ValueError:
            # No zero crossing: either always positive (issue at max T) or always negative
            if net_benefit(40.0, quality, self.kappa, self.alpha, self.beta, self.c, self.b0) > 0:
                return 40.0
            return None

    def frontier_curve(
        self,
        n_points: int = 100,
        quality_grid: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Trace the equilibrium frontier.

        Returns DataFrame with (quality, duration, spread, feasible).
        """
        if quality_grid is None:
            quality_grid = np.linspace(0.01, 0.99, n_points)

        rows = []
        for theta in quality_grid:
            T_star = self.optimal_duration(theta)
            if T_star is None or T_star < 0.1:
                rows.append({'quality': theta, 'duration': np.nan, 'spread': np.nan, 'feasible': False})
                continue

            s_star = market_spread(
                np.array([T_star]), np.array([theta]),
                self.kappa, self.alpha, self.beta, self.c
            )[0]
            rows.append({'quality': theta, 'duration': T_star, 'spread': s_star, 'feasible': True})

        return pd.DataFrame(rows)

    def simulate_bonds(
        self,
        n: int = 500,
        quality_dist: str = 'beta',
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Simulate a bond universe under the equilibrium model.

        Each issuer of quality θ issues debt at some T ≤ T*(θ).
        Bonds cluster below the frontier by construction.
        """
        rng = np.random.default_rng(seed)

        if quality_dist == 'beta':
            quality = rng.beta(2, 3, size=n)  # skewed toward lower quality
        else:
            quality = rng.uniform(0.01, 0.99, size=n)

        records = []
        for theta in quality:
            T_star = self.optimal_duration(theta)
            if T_star is None:
                continue
            # Issuer chooses T ≤ T* (uniformly for simplicity)
            T = rng.uniform(0.1, T_star)
            s = market_spread(
                np.array([T]), np.array([theta]),
                self.kappa, self.alpha, self.beta, self.c
            )[0]
            s += rng.normal(0, 10)  # noise
            records.append({'quality': theta, 'duration': T, 'spread': max(s, 1.0)})

        return pd.DataFrame(records)

    def comparative_statics(
        self,
        param: str,
        values: np.ndarray,
        n_quality: int = 50,
    ) -> pd.DataFrame:
        """
        Compute how the frontier shifts as a model parameter changes.

        Parameters
        ----------
        param  : one of 'kappa', 'alpha', 'beta', 'b0'
        values : array of parameter values to evaluate
        """
        frames = []
        for val in values:
            params = dict(kappa=self.kappa, alpha=self.alpha,
                          beta=self.beta, c=self.c, b0=self.b0)
            params[param] = val
            eq = IssuanceEquilibrium(**params)
            curve = eq.frontier_curve(n_points=n_quality)
            curve['param_value'] = val
            curve['param_name'] = param
            frames.append(curve)
        return pd.concat(frames, ignore_index=True)


# ── Welfare analysis ───────────────────────────────────────────────────────────

def aggregate_issuance_volume(
    quality_grid: np.ndarray,
    quality_pdf: Callable,
    equilibrium: IssuanceEquilibrium,
) -> float:
    """
    Compute total expected issuance duration (proxy for market depth).

    Integral of T*(θ) · f(θ) dθ over feasible θ.
    """
    total = 0.0
    for theta in quality_grid:
        T_star = equilibrium.optimal_duration(theta)
        if T_star is not None:
            total += T_star * quality_pdf(theta)
    return total / len(quality_grid)


# ── Prediction: how the frontier slope depends on b0 ──────────────────────────

def frontier_slope_vs_benefit(
    b0_values: np.ndarray,
    base_kappa: float = 500.0,
) -> pd.DataFrame:
    """
    Key theoretical result: the frontier slope (β in log-log) should decrease
    (steepen) as b0 increases (higher benefit of long-term issuance).

    This is because high-b0 environments allow even distressed issuers to
    place long-dated debt, pulling the frontier upward at long durations.
    """
    rows = []
    for b0 in b0_values:
        eq = IssuanceEquilibrium(kappa=base_kappa, b0=b0)
        curve = eq.frontier_curve(n_points=60).dropna()
        if len(curve) < 10:
            continue
        # Log-log regression on the frontier curve
        log_d = np.log(curve['duration'].clip(lower=0.1))
        log_s = np.log(curve['spread'].clip(lower=1))
        beta = np.polyfit(log_d, log_s, 1)[0]
        rows.append({'b0': b0, 'frontier_beta': beta, 'n_feasible': len(curve)})

    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("=" * 60)
    print("Chapter 3: Issuance Equilibrium")
    print("=" * 60)

    eq = IssuanceEquilibrium(kappa=500, alpha=1.5, beta=0.6, b0=30)

    # Frontier curve
    frontier = eq.frontier_curve(n_points=80).dropna()
    print(f"\nFrontier computed for {len(frontier)} quality levels")
    print(frontier[['quality', 'duration', 'spread']].describe().round(2))

    # Comparative statics: b0 effect
    print("\nComparative statics: frontier slope vs. b0")
    b0_values = np.linspace(10, 60, 8)
    slope_df = frontier_slope_vs_benefit(b0_values)
    print(slope_df.round(3))

    # Simulate bond universe under equilibrium
    bonds = eq.simulate_bonds(n=500)
    print(f"\nSimulated {len(bonds)} bonds under equilibrium model")
    print(bonds[['duration', 'spread', 'quality']].describe().round(2))

    # Key prediction
    corr = slope_df['frontier_beta'].corr(slope_df['b0'])
    print(f"\nCorr(β_frontier, b₀) = {corr:.3f}")
    print("Expected: negative (higher b0 → flatter frontier = less steep)")

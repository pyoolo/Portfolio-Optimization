"""
Synthetic bond universe generator.

Generates a realistic cross-section of bonds in the (duration, spread) plane,
calibrated to reproduce the empirical structure observed in multi-fund portfolios.

Three populations:
  - Investment Grade (IG): low spread, broad duration range
  - High Yield (HY): high spread, short–medium duration
  - Distressed: very high spread, near-zero or negative duration
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class UniverseConfig:
    n_ig: int = 600
    n_hy: int = 250
    n_distressed: int = 80
    seed: int = 42


def _merton_spread(duration: np.ndarray, leverage: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Approximate Merton (1974) spread for an array of bonds.

    Closed-form approximation:
        spread ≈ -1/T * log[N(d2) + (1/L)*N(-d1)*exp(r*T)]

    where L = leverage ratio (D/V), sigma = asset volatility, T = maturity ≈ duration.

    We use a simplified version that preserves the qualitative structure.
    """
    from scipy.stats import norm

    T = np.clip(duration, 0.1, 30.0)
    r = 0.04  # risk-free rate

    d1 = (np.log(1.0 / leverage) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    bond_value = norm.cdf(d2) + (1.0 / leverage) * norm.cdf(-d1) * np.exp(r * T)
    bond_value = np.clip(bond_value, 1e-6, 1.0)

    spread = -np.log(bond_value) / T - r
    spread = np.clip(spread, 1.0, 8000.0)
    return spread * 10000  # in basis points


def generate_universe(cfg: Optional[UniverseConfig] = None) -> pd.DataFrame:
    """
    Generate a synthetic bond universe with three credit segments.

    Returns a DataFrame with columns:
        duration    : modified duration (years), can be negative for floaters
        spread      : Z-spread (basis points)
        segment     : 'IG' | 'HY' | 'Distressed'
        leverage    : synthetic firm leverage ratio
        asset_vol   : synthetic asset volatility
        rating      : synthetic letter rating
        mtm         : mark-to-market value (bubble size proxy)
        merton_spread: spread predicted by Merton model
    """
    if cfg is None:
        cfg = UniverseConfig()

    rng = np.random.default_rng(cfg.seed)

    records = []

    # ── Investment Grade ──────────────────────────────────────────────────────
    # Duration: broad range 0–25y, spread: 20–600 bp
    # Leverage: low (0.2–0.6), asset vol: low (0.10–0.20)
    n = cfg.n_ig
    duration_ig = rng.uniform(-1.0, 25.0, n)
    leverage_ig = rng.uniform(0.20, 0.60, n)
    sigma_ig = rng.uniform(0.08, 0.20, n)

    spread_ig = _merton_spread(duration_ig, leverage_ig, sigma_ig)
    # Add noise and floor
    spread_ig += rng.normal(0, 15, n)
    spread_ig = np.clip(spread_ig, 5, 700)

    ratings_ig = rng.choice(['AAA', 'AA', 'A', 'BBB'], n, p=[0.05, 0.20, 0.40, 0.35])
    mtm_ig = rng.lognormal(mean=14, sigma=1.5, size=n)

    for i in range(n):
        records.append(dict(
            duration=duration_ig[i],
            spread=spread_ig[i],
            segment='IG',
            leverage=leverage_ig[i],
            asset_vol=sigma_ig[i],
            rating=ratings_ig[i],
            mtm=mtm_ig[i],
            merton_spread=_merton_spread(
                np.array([duration_ig[i]]),
                np.array([leverage_ig[i]]),
                np.array([sigma_ig[i]])
            )[0],
        ))

    # ── High Yield ────────────────────────────────────────────────────────────
    # Duration: short–medium (0–8y), spread: 200–2500 bp
    # Leverage: medium–high (0.55–0.85), asset vol: medium (0.20–0.35)
    n = cfg.n_hy
    duration_hy = rng.uniform(0.0, 8.0, n)
    leverage_hy = rng.uniform(0.55, 0.85, n)
    sigma_hy = rng.uniform(0.18, 0.35, n)

    spread_hy = _merton_spread(duration_hy, leverage_hy, sigma_hy)
    spread_hy += rng.normal(0, 50, n)
    spread_hy = np.clip(spread_hy, 150, 3000)

    ratings_hy = rng.choice(['BB', 'B', 'CCC'], n, p=[0.40, 0.45, 0.15])
    mtm_hy = rng.lognormal(mean=12, sigma=1.8, size=n)

    for i in range(n):
        records.append(dict(
            duration=duration_hy[i],
            spread=spread_hy[i],
            segment='HY',
            leverage=leverage_hy[i],
            asset_vol=sigma_hy[i],
            rating=ratings_hy[i],
            mtm=mtm_hy[i],
            merton_spread=_merton_spread(
                np.array([duration_hy[i]]),
                np.array([leverage_hy[i]]),
                np.array([sigma_hy[i]])
            )[0],
        ))

    # ── Distressed ────────────────────────────────────────────────────────────
    # Duration: near-zero or negative (floaters, restructured), spread: 1000–5000+ bp
    # Leverage: high–very high (0.80–0.99), asset vol: high (0.30–0.60)
    n = cfg.n_distressed
    duration_d = rng.uniform(-2.0, 3.0, n)
    leverage_d = rng.uniform(0.80, 0.99, n)
    sigma_d = rng.uniform(0.30, 0.60, n)

    spread_d = _merton_spread(duration_d, leverage_d, sigma_d)
    spread_d += rng.normal(0, 200, n)
    spread_d = np.clip(spread_d, 800, 6000)

    mtm_d = rng.lognormal(mean=10, sigma=2.0, size=n)

    for i in range(n):
        records.append(dict(
            duration=duration_d[i],
            spread=spread_d[i],
            segment='Distressed',
            leverage=leverage_d[i],
            asset_vol=sigma_d[i],
            rating='D',
            mtm=mtm_d[i],
            merton_spread=_merton_spread(
                np.array([duration_d[i]]),
                np.array([leverage_d[i]]),
                np.array([sigma_d[i]])
            )[0],
        ))

    df = pd.DataFrame(records).reset_index(drop=True)

    # DTS = spread_duration × spread  (Ben Dor et al. 2007)
    # For simplicity: spread_duration ≈ modified_duration for non-callable fixed rate
    df['dts'] = df['duration'].clip(lower=0) * df['spread']

    # Spread per unit duration (ratio)
    df['spread_per_dur'] = df['spread'] / df['duration'].clip(lower=0.1)

    return df


def generate_panel(n_periods: int = 60, base_cfg: Optional[UniverseConfig] = None) -> pd.DataFrame:
    """
    Generate a time-series panel of cross-sections (monthly snapshots).

    Stress regimes are simulated by shifting leverage and volatility parameters.
    Returns a DataFrame with an additional 'period' column (0 = most recent).
    """
    if base_cfg is None:
        base_cfg = UniverseConfig()

    rng = np.random.default_rng(base_cfg.seed + 999)
    frames = []

    # Macro state: random walk with mean reversion (stress index)
    stress = np.zeros(n_periods)
    for t in range(1, n_periods):
        stress[t] = 0.85 * stress[t - 1] + rng.normal(0, 0.15)
    stress = (stress - stress.min()) / (stress.max() - stress.min())  # 0–1

    for t in range(n_periods):
        cfg_t = UniverseConfig(
            n_ig=base_cfg.n_ig,
            n_hy=base_cfg.n_hy,
            n_distressed=base_cfg.n_distressed,
            seed=base_cfg.seed + t,
        )
        df_t = generate_universe(cfg_t)

        # In stress periods: widen spreads, compress duration (issuers shorten)
        s = stress[t]
        df_t['spread'] *= (1 + 1.5 * s * rng.uniform(0.8, 1.2, len(df_t)))
        df_t['duration'] *= (1 - 0.3 * s)
        df_t['spread'] = df_t['spread'].clip(lower=1)
        df_t['period'] = t
        df_t['stress_index'] = s
        frames.append(df_t)

    return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    df = generate_universe()
    print(df.groupby('segment')[['duration', 'spread', 'dts']].describe().round(1))
    df.to_csv('/tmp/synthetic_bonds.csv', index=False)
    print(f"\nSaved {len(df)} bonds to /tmp/synthetic_bonds.csv")

"""
Tests for the Duration–Spread Frontier project.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ── Synthetic data ─────────────────────────────────────────────────────────────

def test_generate_universe():
    from utils.synthetic import generate_universe, UniverseConfig
    cfg = UniverseConfig(n_ig=100, n_hy=50, n_distressed=20)
    df = generate_universe(cfg)
    assert len(df) == 170
    assert set(df['segment']) == {'IG', 'HY', 'Distressed'}
    assert df['spread'].min() > 0
    assert df['dts'].min() >= 0


def test_panel_generation():
    from utils.synthetic import generate_panel, UniverseConfig
    cfg = UniverseConfig(n_ig=50, n_hy=30, n_distressed=10)
    panel = generate_panel(n_periods=5, base_cfg=cfg)
    assert 'period' in panel.columns
    assert 'stress_index' in panel.columns
    assert panel['period'].nunique() == 5


# ── Frontier estimation ────────────────────────────────────────────────────────

def test_frontier_fit_and_predict():
    from utils.synthetic import generate_universe, UniverseConfig
    from frontier.estimator import FrontierEstimator

    df = generate_universe(UniverseConfig(n_ig=200, n_hy=80, n_distressed=30))
    est = FrontierEstimator(tau=0.90)
    est.fit(df)

    assert est.params_ is not None
    assert est.params_['beta'] < 0, "Frontier slope should be negative in log-log"
    assert est.params_['a'] > 0

    d_test = np.array([1.0, 5.0, 10.0, 20.0])
    s_pred = est.predict(d_test)
    assert (np.diff(s_pred) < 0).all(), "Frontier should be decreasing in duration"


def test_frontier_robustness():
    from utils.synthetic import generate_universe, UniverseConfig
    from frontier.estimator import FrontierEstimator

    df = generate_universe(UniverseConfig(n_ig=300, n_hy=100, n_distressed=40))
    est = FrontierEstimator()
    robust = est.estimate_all_quantiles(df, taus=(0.80, 0.90, 0.95))

    assert len(robust) == 3
    # Higher quantile → higher intercept (frontier shifts up)
    assert robust.loc[2, 'alpha'] > robust.loc[0, 'alpha']
    # All slopes negative
    assert (robust['beta'] < 0).all()


def test_dts_isocurves():
    from frontier.estimator import dts_isocurves
    d_grid = np.array([1.0, 2.0, 5.0, 10.0])
    df = dts_isocurves(np.array([500, 2000]), d_grid)
    # DTS = spread × duration → spread = DTS / duration → check monotonicity
    sub = df[df['dts'] == 500].sort_values('duration')
    assert (np.diff(sub['spread'].values) < 0).all()


# ── Merton model ──────────────────────────────────────────────────────────────

def test_merton_spread_monotonicity():
    from merton.consistency import merton_spread_bps
    durations = np.array([0.5, 1.0, 3.0, 7.0, 15.0])
    # For high leverage/sigma (distressed), spread is highest at short duration
    leverage = np.full_like(durations, 0.92)
    sigma = np.full_like(durations, 0.45)
    spreads = merton_spread_bps(durations, leverage, sigma)
    assert spreads[0] > spreads[-1], "Distressed Merton spread should be higher at short than long duration"


def test_merton_calibrator():
    from utils.synthetic import generate_universe, UniverseConfig
    from merton.consistency import MertonCalibrator

    df = generate_universe(UniverseConfig(n_ig=150, n_hy=60, n_distressed=20))
    cal = MertonCalibrator(n_sim=500)
    cal.fit(df, seed=0)
    cloud = cal.simulate()

    assert len(cloud) == 500
    assert cloud['spread'].min() > 0
    assert cloud['leverage'].between(0.05, 0.95).all()


# ── Equilibrium model ─────────────────────────────────────────────────────────

def test_equilibrium_frontier():
    from equilibrium.model import IssuanceEquilibrium

    eq = IssuanceEquilibrium(kappa=500, alpha=1.5, beta=0.6, b0=120)
    frontier = eq.frontier_curve(n_points=30).dropna()

    assert len(frontier) > 0, "Frontier should have at least some feasible points"
    # Better quality issuers should have higher or equal optimal duration
    if len(frontier) > 3:
        high_q = frontier[frontier['quality'] > frontier['quality'].median()]['duration'].mean()
        low_q  = frontier[frontier['quality'] <= frontier['quality'].median()]['duration'].mean()
        assert high_q >= low_q, "Better quality → longer optimal duration"


def test_equilibrium_comparative_statics():
    from equilibrium.model import IssuanceEquilibrium

    eq = IssuanceEquilibrium()
    b0_vals = np.array([15.0, 30.0, 50.0])
    cs = eq.comparative_statics('b0', b0_vals, n_quality=20)
    assert 'param_value' in cs.columns
    assert cs['param_name'].unique()[0] == 'b0'


def test_net_benefit_sign():
    from equilibrium.model import net_benefit

    # High quality, short duration: should be positive (issuer participates)
    nb_good = net_benefit(5.0, 0.9, kappa=500, alpha=1.5, beta=0.6, c=20, b0=30)
    # Low quality, very long duration: should be negative (issuer won't issue)
    nb_bad  = net_benefit(30.0, 0.05, kappa=500, alpha=1.5, beta=0.6, c=20, b0=30)
    assert nb_good > nb_bad

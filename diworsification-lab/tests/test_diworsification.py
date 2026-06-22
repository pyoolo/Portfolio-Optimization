"""Tests for diworsification-lab. Run with: pytest"""

import numpy as np
import pandas as pd
import pytest

from diworsification import (
    effective_number_of_bets,
    diversification_ratio,
    average_pairwise_correlation,
    herfindahl_index,
    marginal_diversification_curve,
    Backtester,
)
from diworsification.data import parse_locale_float, build_weight_matrix, load_compositions


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("12,5%", 12.5),
    ("1.234,56", 1234.56),
    ("3.5", 3.5),
    (" 7 ", 7.0),
    ("", np.nan),
    (None, np.nan),
    (42, 42.0),
])
def test_parse_locale_float(raw, expected):
    got = parse_locale_float(raw)
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_weight_matrix_normalizes():
    comp = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01"] * 2),
        "ticker": ["A", "B"],
        "weight": [0.3, 0.3],
    })
    wm = build_weight_matrix(comp, normalize=True)
    assert wm.sum(axis=1).iloc[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# metrics: known-answer checks
# --------------------------------------------------------------------------- #
def test_enb_perfectly_correlated_is_one():
    # Two assets with identical returns -> single factor -> ENB ~ 1.
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=["A", "B"], columns=["A", "B"])
    enb = effective_number_of_bets([0.5, 0.5], cov)
    assert enb == pytest.approx(1.0, abs=1e-6)


def test_enb_independent_equal_is_n():
    # Two uncorrelated equal-variance assets, equal weights -> ENB ~ 2.
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["A", "B"], columns=["A", "B"])
    enb = effective_number_of_bets([0.5, 0.5], cov)
    assert enb == pytest.approx(2.0, abs=1e-6)


def test_diversification_ratio_correlated_is_one():
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=["A", "B"], columns=["A", "B"])
    dr = diversification_ratio([0.5, 0.5], cov)
    assert dr == pytest.approx(1.0, abs=1e-6)


def test_herfindahl_concentration():
    assert herfindahl_index([1.0, 0.0]) == pytest.approx(1.0)        # fully concentrated
    assert herfindahl_index([0.25] * 4) == pytest.approx(0.25)       # 1/N for equal weights


def test_avg_pairwise_correlation_range():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("ABCD"))
    c = average_pairwise_correlation(rets)
    assert -1.0 <= c <= 1.0


# --------------------------------------------------------------------------- #
# marginal curve monotonic-ish behavior
# --------------------------------------------------------------------------- #
def test_marginal_curve_reduces_vol_for_independent_assets():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(scale=0.01, size=(500, 6)), columns=list("ABCDEF"))
    curve = marginal_diversification_curve(rets)
    # adding independent assets should reduce equal-weight vol
    assert curve["portfolio_vol"].iloc[-1] < curve["portfolio_vol"].iloc[0]
    # ENB should grow toward the number of assets
    assert curve["enb"].iloc[-1] > curve["enb"].iloc[0]


# --------------------------------------------------------------------------- #
# backtest sanity
# --------------------------------------------------------------------------- #
def test_backtester_nav_starts_near_100():
    dates = pd.bdate_range("2021-01-01", periods=300)
    rng = np.random.default_rng(2)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, size=(300, 3)), axis=0),
        index=dates, columns=["A", "B", "C"],
    )
    weights = pd.DataFrame(
        [[0.4, 0.3, 0.3]],
        index=[dates[0]], columns=["A", "B", "C"],
    )
    bt = Backtester(prices, weights)
    res = bt.run()
    assert 95 < res.nav.iloc[0] < 105
    assert "sharpe" in res.stats

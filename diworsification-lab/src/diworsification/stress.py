"""
stress.py
=========

Tools to test the second half of the over-diversification thesis: that in
stress regimes correlations rise toward 1, so the diversification you paid for
in calm markets evaporates exactly when you need it.

This is well documented in the literature: cross-market correlations rise
sharply during crises (Longin & Solnik 2001; Forbes & Rigobon 2002 caution that
part of this is a volatility artifact, which is why we report both raw and
volatility-conditioned views).

Three functions:

* rolling_average_correlation -- time series of mean pairwise correlation.
* correlation_by_volatility_regime -- split days into calm/normal/stress by
  portfolio volatility and compare average correlation and ENB in each bucket.
* stress_correlation_uplift -- single number: how much higher correlation is in
  the stress bucket vs the calm bucket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import effective_number_of_bets, average_pairwise_correlation

_EPS = 1e-12


def rolling_average_correlation(
    returns: pd.DataFrame,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling mean off-diagonal correlation across all assets.

    Computed on a rolling window of daily returns. Spikes in this series mark
    episodes where holdings move together regardless of their nominal variety.
    """
    rets = returns.dropna(axis=1, how="all")
    if min_periods is None:
        min_periods = window // 2
    n = rets.shape[1]
    if n < 2:
        raise ValueError("need at least 2 assets")

    out = {}
    arr = rets.to_numpy()
    idx = rets.index
    offdiag = ~np.eye(n, dtype=bool)
    for end in range(window, len(rets) + 1):
        chunk = arr[end - window : end]
        valid = ~np.isnan(chunk).all(axis=0)
        sub = chunk[:, valid]
        if sub.shape[1] < 2 or (~np.isnan(sub)).sum() < min_periods:
            continue
        c = np.corrcoef(np.ma.masked_invalid(sub), rowvar=False)
        c = np.asarray(c)
        m = ~np.eye(c.shape[0], dtype=bool)
        out[idx[end - 1]] = float(np.nanmean(c[m]))
    return pd.Series(out, name=f"avg_corr_{window}d")


def correlation_by_volatility_regime(
    returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    vol_window: int = 21,
    calm_q: float = 0.33,
    stress_q: float = 0.67,
) -> pd.DataFrame:
    """Bucket days into calm / normal / stress by trailing portfolio vol, then
    compare average pairwise correlation and ENB within each bucket.

    The signature of fragile diversification: avg_corr in 'stress' is markedly
    higher than in 'calm', and ENB in 'stress' is markedly lower.

    Returns
    -------
    DataFrame indexed by regime with columns [n_days, avg_corr, enb, mean_vol].
    """
    rets = returns.dropna(axis=1, how="all")
    trailing_vol = portfolio_returns.rolling(vol_window).std()
    aligned = trailing_vol.dropna()
    if aligned.empty:
        raise ValueError("not enough data to compute trailing volatility")

    lo, hi = aligned.quantile(calm_q), aligned.quantile(stress_q)

    def label(v):
        if v <= lo:
            return "calm"
        if v >= hi:
            return "stress"
        return "normal"

    regimes = aligned.map(label)

    rows = []
    n = rets.shape[1]
    eye = np.eye(n, dtype=bool)
    for name in ["calm", "normal", "stress"]:
        days = regimes.index[regimes == name]
        days = rets.index.intersection(days)
        if len(days) < 5:
            rows.append({"regime": name, "n_days": len(days),
                         "avg_corr": np.nan, "enb": np.nan, "mean_vol": np.nan})
            continue
        sub = rets.loc[days]
        c = sub.corr().to_numpy()
        avg_c = float(c[~eye[: c.shape[0], : c.shape[0]]].mean())
        cov = sub.cov()
        w = np.repeat(1.0 / cov.shape[0], cov.shape[0])
        enb = effective_number_of_bets(w, cov)
        rows.append({
            "regime": name,
            "n_days": len(days),
            "avg_corr": avg_c,
            "enb": enb,
            "mean_vol": float(aligned.loc[days].mean()),
        })
    return pd.DataFrame(rows).set_index("regime")


def stress_correlation_uplift(regime_table: pd.DataFrame) -> dict:
    """Summarize the calm->stress shift from correlation_by_volatility_regime.

    Returns a dict with the absolute correlation uplift and the ENB collapse,
    i.e. the quantitative cost of fragile diversification.
    """
    calm = regime_table.loc["calm"]
    stress = regime_table.loc["stress"]
    return {
        "calm_avg_corr": float(calm["avg_corr"]),
        "stress_avg_corr": float(stress["avg_corr"]),
        "corr_uplift": float(stress["avg_corr"] - calm["avg_corr"]),
        "calm_enb": float(calm["enb"]),
        "stress_enb": float(stress["enb"]),
        "enb_collapse": float(calm["enb"] - stress["enb"]),
    }

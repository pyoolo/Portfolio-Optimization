"""
metrics.py
==========

Diversification diagnostics used to test the over-diversification thesis.

The central claim being tested is: beyond a modest number of funds, additional
holdings stop adding *independent* risk-return sources. Three complementary
lenses:

* Effective Number of Bets (ENB) -- Meucci (2009). Runs PCA on the covariance
  matrix; the portfolio's variance is decomposed onto uncorrelated principal
  components, and the entropy of that distribution gives the effective number
  of *independent* bets. 31 funds can still be ~1-2 bets if they all load on
  the same factor.

* Diversification Ratio -- Choueifaty & Coignard (2008). Weighted average of
  asset vols divided by portfolio vol. Equals 1 when assets are perfectly
  correlated; rises as genuine diversification appears. Its square approximates
  the number of independent risk factors.

* Average pairwise correlation & Herfindahl -- simple, robust sanity checks.

And the headline chart: marginal_diversification_curve, which adds funds one at
a time (by a chosen ordering) and tracks how fast portfolio volatility stops
falling -- the empirical "diminishing returns" curve of Evans & Archer (1968).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def _as_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Covariance matrix from a returns frame, dropping all-NaN columns."""
    r = returns.dropna(axis=1, how="all")
    return r.cov()


def effective_number_of_bets(
    weights: np.ndarray | pd.Series,
    cov: pd.DataFrame,
) -> float:
    r"""Meucci's Effective Number of Bets via principal-component torsion.

    Decompose portfolio variance onto the principal components of ``cov``.
    Each PC is an uncorrelated risk source; its share of portfolio variance is
    ``p_i``. The ENB is the exponential of the Shannon entropy of ``{p_i}``:

        ENB = exp( -sum_i p_i * ln p_i )

    ENB = 1 means the whole portfolio rides a single factor; ENB = N means
    variance is spread evenly across N independent factors.

    Parameters
    ----------
    weights
        Portfolio weights aligned to the columns/index of ``cov``.
    cov
        Asset covariance matrix (DataFrame, symmetric).
    """
    if isinstance(weights, pd.Series):
        weights = weights.reindex(cov.index).fillna(0.0).to_numpy()
    weights = np.asarray(weights, dtype=float)

    sigma = cov.to_numpy()
    # Eigendecomposition: sigma = E diag(lam) E^T, columns of E are PCs.
    lam, eigvecs = np.linalg.eigh(sigma)
    lam = np.clip(lam, 0.0, None)

    # Exposure of the portfolio to each principal component.
    # Factor loadings: y = E^T w ; variance on PC i = lam_i * y_i^2.
    y = eigvecs.T @ weights
    var_contrib = lam * (y ** 2)
    total = var_contrib.sum()
    if total <= _EPS:
        return float("nan")

    p = var_contrib / total
    p = p[p > _EPS]
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def diversification_ratio(
    weights: np.ndarray | pd.Series,
    cov: pd.DataFrame,
) -> float:
    r"""Choueifaty-Coignard diversification ratio.

        DR = (sum_i w_i * sigma_i) / sqrt(w^T Sigma w)

    DR = 1 for a single asset or perfectly correlated holdings; larger means
    more diversification. DR^2 approximates the effective number of independent
    risk factors.
    """
    if isinstance(weights, pd.Series):
        weights = weights.reindex(cov.index).fillna(0.0).to_numpy()
    weights = np.asarray(weights, dtype=float)

    sigma = cov.to_numpy()
    vols = np.sqrt(np.clip(np.diag(sigma), 0.0, None))
    weighted_avg_vol = float(weights @ vols)
    port_var = float(weights @ sigma @ weights)
    if port_var <= _EPS:
        return float("nan")
    return weighted_avg_vol / np.sqrt(port_var)


def average_pairwise_correlation(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.Series | None = None,
) -> float:
    """Mean off-diagonal correlation.

    If ``weights`` is given, pairs are weighted by ``w_i * w_j`` so the figure
    reflects the correlation the portfolio actually experiences, not a naive
    equal-weight average across the whole universe.
    """
    corr = returns.corr()
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    c = corr.to_numpy()
    mask = ~np.eye(n, dtype=bool)

    if weights is None:
        return float(c[mask].mean())

    if isinstance(weights, pd.Series):
        weights = weights.reindex(corr.index).fillna(0.0).to_numpy()
    weights = np.asarray(weights, dtype=float)
    ww = np.outer(weights, weights)
    num = (c * ww)[mask].sum()
    den = ww[mask].sum()
    return float(num / den) if den > _EPS else float("nan")


def herfindahl_index(weights: np.ndarray | pd.Series) -> float:
    """Herfindahl concentration index, sum of squared weights.

    1/HHI is the "effective number of holdings" by weight (ignores correlation).
    Contrast with ENB: a low ENB but high 1/HHI is the diworsification signature
    -- many positions by weight, few independent bets.
    """
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy()
    weights = np.asarray(weights, dtype=float)
    s = weights.sum()
    if s <= _EPS:
        return float("nan")
    w = weights / s
    return float((w ** 2).sum())


def marginal_diversification_curve(
    returns: pd.DataFrame,
    ordering: list[str] | None = None,
    equal_weight: bool = True,
) -> pd.DataFrame:
    """Add assets one at a time and track how fast risk reduction decays.

    This is the empirical Evans-Archer curve. For k = 1..N we form a portfolio
    of the first k assets (equal-weighted by default) and record its annualized
    volatility, ENB, and diversification ratio. The volatility typically drops
    fast for the first handful of names and then flattens -- visual proof of
    diminishing diversification benefit.

    Returns
    -------
    DataFrame indexed by n_assets with columns
    [portfolio_vol, enb, div_ratio, avg_corr, marginal_vol_reduction].
    """
    rets = returns.dropna(axis=1, how="all")
    if ordering is None:
        ordering = list(rets.columns)
    ordering = [c for c in ordering if c in rets.columns]

    rows = []
    prev_vol = None
    for k in range(1, len(ordering) + 1):
        cols = ordering[:k]
        sub = rets[cols].dropna(how="all")
        cov = sub.cov()
        if equal_weight:
            w = np.repeat(1.0 / k, k)
        else:
            w = np.repeat(1.0 / k, k)  # placeholder hook for custom schemes

        port_var = float(w @ cov.to_numpy() @ w)
        vol_ann = np.sqrt(max(port_var, 0.0)) * np.sqrt(252)

        enb = effective_number_of_bets(w, cov) if k >= 2 else 1.0
        dr = diversification_ratio(w, cov) if k >= 2 else 1.0
        avg_c = average_pairwise_correlation(sub) if k >= 2 else np.nan
        marg = (prev_vol - vol_ann) if prev_vol is not None else np.nan
        prev_vol = vol_ann

        rows.append(
            {
                "n_assets": k,
                "portfolio_vol": vol_ann,
                "enb": enb,
                "div_ratio": dr,
                "avg_corr": avg_c,
                "marginal_vol_reduction": marg,
            }
        )

    return pd.DataFrame(rows).set_index("n_assets")

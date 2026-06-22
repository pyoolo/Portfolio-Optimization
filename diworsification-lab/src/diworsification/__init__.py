"""
diworsification-lab
===================

Empirical toolkit to test whether adding more funds/ETFs to a portfolio
actually improves diversification, or whether it collapses into
"diworsification": marginal diversification benefit decays, holdings become
redundant, and correlations converge toward 1 in stress regimes.

The library is organized around four questions:

1. data    -> load messy real-world composition / NAV / ISIN spreadsheets
2. backtest-> reconstruct a returns-based, self-financing NAV from weights+prices
3. metrics -> quantify diversification: Effective Number of Bets (Meucci 2009),
              diversification ratio, average pairwise correlation, marginal
              risk reduction as N grows
4. stress  -> measure how correlations behave in high-volatility regimes
              (the "correlations go to 1" effect)

See examples/run_analysis.py for an end-to-end run.
"""

from .data import (
    load_compositions,
    load_nav_long,
    load_isin_map,
    build_weight_matrix,
    align_prices,
)
from .backtest import Backtester, BacktestResult
from .metrics import (
    effective_number_of_bets,
    diversification_ratio,
    average_pairwise_correlation,
    herfindahl_index,
    marginal_diversification_curve,
)
from .stress import (
    rolling_average_correlation,
    correlation_by_volatility_regime,
    stress_correlation_uplift,
)

__version__ = "0.1.0"

__all__ = [
    "load_compositions",
    "load_nav_long",
    "load_isin_map",
    "build_weight_matrix",
    "align_prices",
    "Backtester",
    "BacktestResult",
    "effective_number_of_bets",
    "diversification_ratio",
    "average_pairwise_correlation",
    "herfindahl_index",
    "marginal_diversification_curve",
    "rolling_average_correlation",
    "correlation_by_volatility_regime",
    "stress_correlation_uplift",
]

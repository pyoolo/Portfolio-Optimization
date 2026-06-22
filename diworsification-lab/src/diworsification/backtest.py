"""
backtest.py
===========

Returns-based, self-financing NAV reconstruction from a rebalance-date weight
matrix plus an aligned price panel, refactored from the original analysis
scripts into one tested class.

Conventions
-----------
* Weights are expanded to a daily "as-of" schedule (forward-filled from each
  rebalance date) and lagged by one day before being applied to returns, so
  there is no look-ahead: the weight set on day t-1 earns the return of day t.
* NAV is base 100.
* Performance stats annualize with 252 trading days.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_EPS = 1e-12
_TRADING_DAYS = 252


@dataclass
class BacktestResult:
    nav: pd.Series
    portfolio_returns: pd.Series
    weights_daily: pd.DataFrame
    prices: pd.DataFrame
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        return (
            "=== PERFORMANCE ===\n"
            f"- Period:               {s['start'].date()} -> {s['end'].date()}\n"
            f"- NAV:                  {s['nav_start']:.2f} -> {s['nav_end']:.2f}\n"
            f"- Total return:         {s['total_return']*100:.2f}%\n"
            f"- Annualized return:    {s['ann_return']*100:.2f}%\n"
            f"- Annualized vol:       {s['ann_vol']*100:.2f}%\n"
            f"- Sharpe (rf=0):        {s['sharpe']:.2f}\n"
            f"- Max drawdown:         {s['max_drawdown']*100:.2f}%"
        )


class Backtester:
    """Reconstruct portfolio NAV from weights + prices."""

    def __init__(self, prices: pd.DataFrame, weights_reb: pd.DataFrame):
        common = prices.columns.intersection(weights_reb.columns)
        if len(common) == 0:
            raise ValueError("no common tickers between prices and weights")
        self.prices = prices[common].sort_index()
        self.weights_reb = weights_reb[common].fillna(0.0).sort_index()
        self.common = list(common)

    # ------------------------------------------------------------------ #
    def _daily_weights(self, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        wm = self.weights_reb.reindex(calendar).ffill().fillna(0.0)
        # renormalize each row defensively
        rs = wm.sum(axis=1).replace(0, np.nan)
        return wm.div(rs, axis=0).fillna(0.0)

    def run(self, start: str | pd.Timestamp | None = None) -> BacktestResult:
        prices = self.prices
        if start is not None:
            prices = prices.loc[prices.index >= pd.to_datetime(start)]
        calendar = prices.index

        weights_daily = self._daily_weights(calendar)

        asset_returns = prices.pct_change()
        weights_lag = weights_daily.shift(1)

        mask = weights_lag.sum(axis=1) > 0
        asset_returns = asset_returns.loc[mask]
        weights_lag = weights_lag.loc[mask]

        port_ret = (weights_lag * asset_returns).sum(axis=1)
        nav = (1 + port_ret.fillna(0)).cumprod() * 100

        stats = self._compute_stats(nav, port_ret)
        return BacktestResult(
            nav=nav,
            portfolio_returns=port_ret,
            weights_daily=weights_daily,
            prices=prices,
            stats=stats,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_stats(nav: pd.Series, port_ret: pd.Series) -> dict:
        rets = nav.pct_change().dropna()
        n = len(rets)
        dd = nav / nav.cummax() - 1
        total_ret = nav.iloc[-1] / nav.iloc[0] - 1
        ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (_TRADING_DAYS / n) - 1 if n else np.nan
        ann_vol = rets.std() * np.sqrt(_TRADING_DAYS)
        sharpe = (rets.mean() / rets.std()) * np.sqrt(_TRADING_DAYS) if rets.std() > _EPS else np.nan
        return {
            "start": nav.index.min(),
            "end": nav.index.max(),
            "nav_start": float(nav.iloc[0]),
            "nav_end": float(nav.iloc[-1]),
            "total_return": float(total_ret),
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(dd.min()),
        }

    # ------------------------------------------------------------------ #
    def attribution(self, result: BacktestResult, names: dict | None = None) -> pd.DataFrame:
        """Total contribution to return per security (lagged weight * return)."""
        asset_ret = result.prices.pct_change()
        w_lag = result.weights_daily.shift(1)
        m = w_lag.sum(axis=1) > 0
        contrib = (w_lag.loc[m] * asset_ret.loc[m]).sum() * 100
        df = contrib.sort_values(ascending=False).to_frame("contribution_pct")
        if names:
            df.insert(0, "name", [names.get(t, "") for t in df.index])
        return df

    def evaluate_rebalances(self, start: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """For each rebalance, compare realized return (new weights) against the
        counterfactual of keeping the previous weights, over the window until the
        next rebalance. Positive value_added means the rebalance helped.
        """
        prices = self.prices
        if start is not None:
            prices = prices.loc[prices.index >= pd.to_datetime(start)]
        asset_ret = prices.pct_change()
        reb_dates = pd.to_datetime(self.weights_reb.index).sort_values()
        w = self.weights_reb

        def compound(ret_df, w_vec):
            return (1.0 + (ret_df * w_vec).sum(axis=1)).prod() - 1.0

        rows = []
        for k in range(1, len(reb_dates) - 1):
            d_prev, d_new, d_next = reb_dates[k - 1], reb_dates[k], reb_dates[k + 1]
            idx = asset_ret.index
            start_dates = idx[idx > d_new]
            if len(start_dates) == 0:
                continue
            period = idx[(idx >= start_dates.min()) & (idx <= d_next)]
            if len(period) < 5:
                continue
            ret_p = asset_ret.loc[period].dropna(how="all")

            w_old = w.loc[d_prev]
            w_new = w.loc[d_new]
            w_old = w_old / w_old.sum() if w_old.sum() != 0 else w_old
            w_new = w_new / w_new.sum() if w_new.sum() != 0 else w_new

            r_real = compound(ret_p, w_new)
            r_hold = compound(ret_p, w_old)
            rows.append({
                "rebalance_date": d_new,
                "period_start": period.min(),
                "period_end": period.max(),
                "realized_return": r_real,
                "no_rebalance_return": r_hold,
                "value_added": r_real - r_hold,
            })
        return pd.DataFrame(rows)

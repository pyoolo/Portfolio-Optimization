"""
make_sample_data.py
===================

Generate a synthetic but realistic dataset so the repo is runnable end-to-end
without any proprietary spreadsheet. It deliberately builds a universe that
*looks* diversified (many funds) but is driven by only a few common factors,
plus a stress regime in which the factor loadings tighten -- exactly the
situation the library is meant to detect.

Run:
    python -m diworsification.make_sample_data
which writes data/sample_portfolio.xlsx with sheets Composizioni / NAV / ISIN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def simulate_prices(n_funds=24, n_days=1000, n_factors=3, start="2021-01-04"):
    dates = pd.bdate_range(start=start, periods=n_days)

    # Common factors: a market factor plus a couple of style/region factors.
    # In calm times the style factors carry real, distinct risk -> genuine
    # diversification is available.
    factor_vol = np.array([0.007, 0.006, 0.005])[:n_factors]
    factors = RNG.normal(0, 1, size=(n_days, n_factors)) * factor_vol

    # Stress regime: a contiguous block where the market factor's vol jumps and
    # the style factors get partially absorbed into it (correlations -> 1).
    stress = slice(int(n_days * 0.45), int(n_days * 0.55))
    factors[stress, 0] *= 3.5
    for f in range(1, n_factors):
        # style factors become highly loaded on the market shock in stress
        factors[stress, f] = 0.3 * factors[stress, f] + 0.7 * factors[stress, 0]

    # Funds spread their loadings across factors -> decent ENB in calm markets.
    loadings = RNG.uniform(0.3, 1.0, size=(n_funds, n_factors))
    loadings[:, 0] += 0.3  # moderate shared market exposure

    idio_vol = RNG.uniform(0.004, 0.008, size=n_funds)
    idio = RNG.normal(0, 1, size=(n_days, n_funds)) * idio_vol
    idio[stress] *= 0.3  # idiosyncratic risk shrinks in stress (co-movement up)

    daily_ret = factors @ loadings.T + idio
    prices = 100 * np.cumprod(1 + daily_ret, axis=0)

    tickers = [f"FUND{ i:02d} LX Equity" for i in range(n_funds)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def build_compositions(prices: pd.DataFrame, n_rebalances=12):
    """Random long-only weights at a set of rebalance dates."""
    tickers = list(prices.columns)
    dates = prices.index
    reb_idx = np.linspace(20, len(dates) - 20, n_rebalances).astype(int)
    reb_dates = dates[reb_idx]

    rows = []
    for d in reb_dates:
        k = RNG.integers(len(tickers) - 4, len(tickers) + 1)  # most funds held
        chosen = RNG.choice(tickers, size=k, replace=False)
        w = RNG.uniform(0.5, 1.5, size=k)
        w = w / w.sum() * 100
        for t, wi in zip(chosen, w):
            # store with Italian formatting to exercise the parser
            rows.append({
                "Data Di Ribilanciamento": d.strftime("%d/%m/%Y"),
                "Ticker": t,
                "Peso": f"{wi:.2f}".replace(".", ","),
            })
    return pd.DataFrame(rows)


def build_nav_wide(prices: pd.DataFrame):
    """Reshape into the repeated Date|TICKER wide layout the loader expects."""
    blocks = {}
    for j, t in enumerate(prices.columns):
        # introduce slight asynchronicity: drop a few random days per fund
        s = prices[t].copy()
        drop = RNG.choice(len(s), size=int(len(s) * 0.03), replace=False)
        s.iloc[drop] = np.nan
        s = s.dropna()
        blocks[f"Data_{j}"] = s.index.strftime("%d/%m/%Y").to_series(index=range(len(s))).reset_index(drop=True)
        blocks[t] = pd.Series([f"{v:.4f}".replace(".", ",") for v in s.values])
    wide = pd.DataFrame(dict(blocks))
    # rename Data_j columns to literal "Data" so pandas dedups to Data, Data.1...
    wide.columns = ["Data" if c.startswith("Data_") else c for c in wide.columns]
    return wide


def build_isin(prices: pd.DataFrame):
    return pd.DataFrame({
        "Ticker": list(prices.columns),
        "Nome": [f"Synthetic Fund {i}" for i in range(len(prices.columns))],
        "Codice ISIN": [f"LU000000{i:04d}" for i in range(len(prices.columns))],
    })


def main(out_path="data/sample_portfolio.xlsx"):
    prices = simulate_prices()
    comp = build_compositions(prices)
    nav = build_nav_wide(prices)
    isin = build_isin(prices)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        comp.to_excel(xl, sheet_name="Composizioni", index=False)
        nav.to_excel(xl, sheet_name="NAV", index=False)
        isin.to_excel(xl, sheet_name="ISIN", index=False)
    print(f"wrote {out} ({len(prices.columns)} funds, {len(prices)} days)")
    return out


if __name__ == "__main__":
    main()

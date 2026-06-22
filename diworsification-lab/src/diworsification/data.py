"""
data.py
=======

Loaders for the kind of real-world, slightly messy spreadsheets that managed
portfolios are delivered in:

* a "Composizioni" / compositions sheet: long format, one row per
  (rebalance_date, ticker, weight), with Italian-style numbers ("12,5%")
* a "NAV" sheet: wide format with repeated (Date | TICKER) column pairs and
  asynchronous date columns (pandas reads duplicate headers as Data, Data.1, ...)
* an optional "ISIN" sheet mapping ticker -> human-readable name

All parsing is locale-tolerant: it strips "%", converts decimal commas, and
coerces bad cells to NaN rather than throwing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Numeric parsing
# --------------------------------------------------------------------------- #
def parse_locale_float(x) -> float:
    """Parse a possibly Italian-formatted number ('1.234,56', '12,5%', ' 3 ').

    Returns np.nan for anything unparseable instead of raising.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("%", "").replace(" ", "")
    # Heuristic: if both separators present, the last one is the decimal sep.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- #
# Compositions
# --------------------------------------------------------------------------- #
def load_compositions(
    source,
    sheet_name: str = "Composizioni",
    col_date: str = "Data Di Ribilanciamento",
    col_ticker: str = "Ticker",
    col_weight: str = "Peso",
    weights_are_percent: bool = True,
) -> pd.DataFrame:
    """Load the long-format compositions sheet.

    Parameters
    ----------
    source
        Path to an .xlsx file, or a pre-loaded DataFrame (useful for testing).
    weights_are_percent
        If True, a weight value of 12.5 is interpreted as 12.5% -> 0.125.

    Returns
    -------
    DataFrame with columns [date, ticker, weight], cleaned and NaN-dropped.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_excel(source, sheet_name=sheet_name)

    missing = {col_date, col_ticker, col_weight} - set(df.columns)
    if missing:
        raise KeyError(f"Compositions sheet missing columns: {sorted(missing)}")

    df = df[[col_date, col_ticker, col_weight]].copy()
    df.columns = ["date", "ticker", "weight"]

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["weight"] = df["weight"].apply(parse_locale_float)

    before = len(df)
    df = df.dropna(subset=["date", "ticker", "weight"])
    df = df[df["ticker"] != ""]
    logger.info("compositions: dropped %d invalid rows, kept %d", before - len(df), len(df))

    if weights_are_percent:
        df["weight"] = df["weight"] / 100.0

    return df.reset_index(drop=True)


def build_weight_matrix(comp: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Pivot long compositions into a (rebalance_date x ticker) weight matrix.

    With normalize=True each row is rescaled to sum to 1, so rounding noise in
    the source weights does not bias downstream return calculations.
    """
    wm = comp.pivot_table(
        index="date",
        columns="ticker",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    if normalize:
        row_sums = wm.sum(axis=1).replace(0, np.nan)
        wm = wm.div(row_sums, axis=0).fillna(0.0)

    return wm


# --------------------------------------------------------------------------- #
# NAV (wide, repeated Date|Ticker pairs)
# --------------------------------------------------------------------------- #
def load_nav_long(source, sheet_name: str = "NAV") -> pd.DataFrame:
    """Parse a wide NAV sheet of repeated (Date | TICKER) column pairs.

    pandas reads duplicate "Data" headers as Data, Data.1, Data.2, ... so we
    walk the columns, treating any header starting with 'data' as a date column
    paired with the next column (the price series, named by its ticker).

    Returns
    -------
    Long DataFrame with columns [date, ticker, price].
    """
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        raw = pd.read_excel(source, sheet_name=sheet_name)

    cols = list(raw.columns)
    records = []
    i = 0
    while i < len(cols) - 1:
        if str(cols[i]).strip().lower().startswith("data"):
            date_col, price_col = cols[i], cols[i + 1]
            tmp = raw[[date_col, price_col]].copy()
            tmp.columns = ["date", "price"]
            tmp["ticker"] = str(price_col).strip()
            tmp["date"] = pd.to_datetime(tmp["date"], dayfirst=True, errors="coerce")
            tmp["price"] = tmp["price"].apply(parse_locale_float)
            tmp = tmp.dropna(subset=["date", "price"])
            if len(tmp):
                records.append(tmp)
            i += 2
        else:
            i += 1

    if not records:
        raise ValueError(
            "NAV parsing failed: expected repeated pairs like "
            "Data | TICKER | Data | TICKER | ..."
        )
    return pd.concat(records, ignore_index=True)


def align_prices(
    prices_long: pd.DataFrame,
    freq: str = "B",
    forward_fill: bool = True,
) -> pd.DataFrame:
    """Pivot long prices to wide and align on a common calendar.

    Funds publish NAV on different (asynchronous) days. We build one regular
    calendar and forward-fill, which is standard but does dampen measured
    volatility for funds with stale NAVs -- noted in the docs.
    """
    wide = prices_long.pivot_table(
        index="date", columns="ticker", values="price", aggfunc="last"
    )
    calendar = pd.date_range(wide.index.min(), wide.index.max(), freq=freq)
    wide = wide.reindex(calendar)
    if forward_fill:
        wide = wide.ffill()
    return wide


# --------------------------------------------------------------------------- #
# ISIN / name map
# --------------------------------------------------------------------------- #
def load_isin_map(
    source,
    sheet_name: str = "ISIN",
    col_ticker: str = "Ticker",
    col_name: str = "Nome",
) -> dict:
    """Return a {ticker: human_readable_name} dict. Empty dict if unavailable."""
    try:
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            df = pd.read_excel(source, sheet_name=sheet_name)
    except Exception as exc:  # sheet absent or unreadable -> degrade gracefully
        logger.warning("could not load ISIN map: %s", exc)
        return {}

    df.columns = df.columns.str.strip()
    if col_ticker not in df.columns or col_name not in df.columns:
        return {}
    df[col_ticker] = df[col_ticker].astype(str).str.strip()
    df[col_name] = df[col_name].astype(str).str.strip()
    return dict(zip(df[col_ticker], df[col_name]))


@dataclass
class PortfolioData:
    """Bundle of everything an analysis needs, all aligned to common tickers."""

    weights_reb: pd.DataFrame   # rebalance_date x ticker
    prices: pd.DataFrame        # calendar x ticker (aligned, ffilled)
    names: dict                 # ticker -> name

    @property
    def common_tickers(self):
        return self.prices.columns.intersection(self.weights_reb.columns)

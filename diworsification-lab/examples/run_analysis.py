"""
run_analysis.py
===============

End-to-end demonstration of the over-diversification thesis on the bundled
sample dataset (or your own spreadsheet via --file).

It produces, in examples/output/:
  1. marginal_diversification_curve.png  -- vol & ENB vs number of funds
  2. rolling_correlation.png             -- avg pairwise correlation over time
  3. correlation_regimes.png             -- avg correlation & ENB by vol regime
  4. nav.png                             -- reconstructed portfolio NAV
  5. summary.txt                         -- the numerical headline results

Usage:
    python examples/run_analysis.py
    python examples/run_analysis.py --file path/to/your.xlsx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

from diworsification import (
    load_compositions, load_nav_long, load_isin_map,
    build_weight_matrix, align_prices,
    Backtester,
    effective_number_of_bets, diversification_ratio, herfindahl_index,
    marginal_diversification_curve,
    rolling_average_correlation, correlation_by_volatility_regime,
    stress_correlation_uplift,
)
from diworsification.make_sample_data import main as make_sample

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def main(file: str | None):
    if file is None:
        file = Path(__file__).parents[1] / "data" / "sample_portfolio.xlsx"
        if not Path(file).exists():
            make_sample(out_path=str(file))

    # ---- load -------------------------------------------------------------
    comp = load_compositions(file)
    weights_reb = build_weight_matrix(comp)
    prices = align_prices(load_nav_long(file))
    names = load_isin_map(file)

    common = prices.columns.intersection(weights_reb.columns)
    prices = prices[common]
    weights_reb = weights_reb[common]
    returns = prices.pct_change().dropna(how="all")

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    # ---- backtest ---------------------------------------------------------
    bt = Backtester(prices, weights_reb)
    result = bt.run()
    log(result.summary())
    log()

    # ---- whole-portfolio diversification snapshot -------------------------
    cov = returns.cov()
    w_avg = weights_reb.mean()  # average allocation across rebalances
    enb = effective_number_of_bets(w_avg, cov)
    dr = diversification_ratio(w_avg, cov)
    hhi = herfindahl_index(w_avg)
    log("=== DIVERSIFICATION SNAPSHOT (avg weights) ===")
    log(f"- Funds held:                 {int((w_avg > 0).sum())}")
    log(f"- Effective number of holdings (1/HHI): {1/hhi:.1f}")
    log(f"- Effective Number of Bets (ENB):       {enb:.2f}")
    log(f"- Diversification Ratio:                {dr:.2f}")
    log(f"- DR^2 (approx independent factors):    {dr**2:.2f}")
    log("  -> Many funds by weight, far fewer independent bets = diworsification.")
    log()

    # ---- marginal curve ---------------------------------------------------
    # order funds by descending average weight (how a real PM would add them)
    ordering = list(w_avg.sort_values(ascending=False).index)
    curve = marginal_diversification_curve(returns, ordering=ordering)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(curve.index, curve["portfolio_vol"] * 100, "o-", color="#c0392b",
             label="Annualized vol (%)")
    ax1.set_xlabel("Number of funds in portfolio")
    ax1.set_ylabel("Annualized volatility (%)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(curve.index, curve["enb"], "s--", color="#2c3e50",
             label="Effective Number of Bets")
    ax2.set_ylabel("Effective Number of Bets", color="#2c3e50")
    ax2.tick_params(axis="y", labelcolor="#2c3e50")
    plt.title("Diminishing diversification: volatility flattens, ENB plateaus")
    fig.tight_layout()
    fig.savefig(OUT / "marginal_diversification_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # where does marginal vol reduction effectively die?
    marg = curve["marginal_vol_reduction"].dropna() * 100
    knee = marg[marg < 0.10]
    knee_n = int(knee.index[0]) if len(knee) else curve.index[-1]
    log("=== MARGINAL BENEFIT (Evans-Archer curve) ===")
    log(f"- Vol at 1 fund:   {curve['portfolio_vol'].iloc[0]*100:.2f}%")
    log(f"- Vol at {curve.index[-1]} funds: {curve['portfolio_vol'].iloc[-1]*100:.2f}%")
    log(f"- Beyond ~{knee_n} funds each new fund cuts vol by < 0.10pp.")
    log(f"- ENB saturates at {curve['enb'].iloc[-1]:.2f} despite "
        f"{curve.index[-1]} funds.")
    log()

    # ---- rolling correlation ---------------------------------------------
    roll = rolling_average_correlation(returns, window=60)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(roll.index, roll.values, color="#2980b9")
    ax.axhline(roll.mean(), ls="--", color="grey", lw=1,
               label=f"mean = {roll.mean():.2f}")
    ax.set_title("Rolling 60d average pairwise correlation (spikes = co-movement)")
    ax.set_ylabel("Average pairwise correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "rolling_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- regime analysis: correlations go to 1 in stress ------------------
    regimes = correlation_by_volatility_regime(returns, result.portfolio_returns)
    uplift = stress_correlation_uplift(regimes)
    log("=== CORRELATION BY VOLATILITY REGIME ===")
    log(regimes.round(3).to_string())
    log(f"- Calm avg corr:   {uplift['calm_avg_corr']:.3f}")
    log(f"- Stress avg corr: {uplift['stress_avg_corr']:.3f}")
    log(f"- Correlation uplift in stress: +{uplift['corr_uplift']:.3f}")
    log(f"- ENB collapse calm->stress: {uplift['calm_enb']:.2f} -> "
        f"{uplift['stress_enb']:.2f}")
    log("  -> Diversification is weakest exactly when it is needed most.")
    log()

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))
    order = ["calm", "normal", "stress"]
    rg = regimes.reindex(order)
    axa.bar(order, rg["avg_corr"], color=["#27ae60", "#f39c12", "#c0392b"])
    axa.set_title("Average pairwise correlation by regime")
    axa.set_ylabel("Avg correlation")
    axb.bar(order, rg["enb"], color=["#27ae60", "#f39c12", "#c0392b"])
    axb.set_title("Effective Number of Bets by regime")
    axb.set_ylabel("ENB")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_regimes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- NAV plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(result.nav.index, result.nav.values, color="#2c3e50")
    for d in weights_reb.index:
        if result.nav.index.min() <= d <= result.nav.index.max():
            ax.axvline(d, ls="--", lw=0.7, alpha=0.3, color="grey")
    ax.set_title("Reconstructed portfolio NAV (base 100), dashed = rebalances")
    ax.set_ylabel("NAV")
    fig.tight_layout()
    fig.savefig(OUT / "nav.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"[saved 5 plots + summary.txt to {OUT}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=None, help="Path to portfolio .xlsx")
    args = p.parse_args()
    main(args.file)

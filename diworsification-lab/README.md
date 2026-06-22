# diworsification-lab

**An empirical toolkit for detecting over-diversification ("diworsification") in fund/ETF portfolios.**

Adding more funds to a portfolio feels safe. This library tests, on real or
synthetic data, whether it actually *is* — or whether past a modest number of
holdings you are simply paying more fees for the same single bet, with
correlations that snap toward 1 in exactly the stress regimes where you wanted
the protection.

The thesis, made measurable:

1. **Marginal diversification benefit decays fast.** Past a handful of funds,
   each new one barely moves portfolio volatility (the Evans–Archer curve).
2. **Many holdings ≠ many bets.** A portfolio of 30 funds can have an
   *Effective Number of Bets* near 1 if they all load on the same factor.
3. **Correlations rise in stress.** Average pairwise correlation measured in
   high-volatility regimes is markedly higher than in calm ones, so
   diversification is weakest precisely when it matters most.

---

## What it computes

| Module | What it answers |
|---|---|
| `data` | Load messy real-world composition / NAV / ISIN spreadsheets (Italian number formats, repeated `Data\|TICKER` columns, async dates). |
| `backtest` | Reconstruct a returns-based, self-financing NAV from weights + prices; performance stats; per-rebalance value-added; security attribution. |
| `metrics` | Effective Number of Bets (Meucci 2009), Diversification Ratio (Choueifaty & Coignard 2008), average pairwise correlation, Herfindahl, and the marginal-diversification curve. |
| `stress` | Rolling average correlation, correlation & ENB bucketed by volatility regime, and the calm→stress correlation uplift. |

## Quick start

```bash
git clone <your-repo-url> diworsification-lab
cd diworsification-lab
pip install -e .

# generate a synthetic but realistic dataset, then run the full analysis
python -m diworsification.make_sample_data
python examples/run_analysis.py
```

Outputs land in `examples/output/`: five plots plus `summary.txt`.

To run on your own portfolio spreadsheet (sheets `Composizioni`, `NAV`, and
optionally `ISIN`, in the format described in [`docs/data_format.md`](docs/data_format.md)):

```bash
python examples/run_analysis.py --file path/to/your_portfolio.xlsx
```

## Example: measuring the bets, not the holdings

```python
from diworsification import (
    load_compositions, build_weight_matrix, load_nav_long, align_prices,
    effective_number_of_bets, diversification_ratio,
)

comp    = load_compositions("portfolio.xlsx")
weights = build_weight_matrix(comp)
prices  = align_prices(load_nav_long("portfolio.xlsx"))

common  = prices.columns.intersection(weights.columns)
cov     = prices[common].pct_change().dropna().cov()
w_avg   = weights[common].mean()

print("Funds held:", int((w_avg > 0).sum()))
print("Effective Number of Bets:", round(effective_number_of_bets(w_avg, cov), 2))
print("Diversification Ratio:", round(diversification_ratio(w_avg, cov), 2))
```

A typical "diworsified" result: 30 funds held, ENB ≈ 1.2.

## The metrics, briefly

**Effective Number of Bets (ENB).** Decompose portfolio variance onto the
principal components of the covariance matrix. Each PC is an *uncorrelated* risk
source; the entropy of how variance spreads across them gives an effective count
of independent bets: `ENB = exp(-Σ pᵢ ln pᵢ)`. ENB = 1 means a single factor
drives everything; ENB = N means N evenly-weighted independent sources.

**Diversification Ratio (DR).** Weighted-average asset volatility divided by
portfolio volatility. Equals 1 when holdings are perfectly correlated and rises
with genuine diversification; `DR²` approximates the number of independent risk
factors.

**Marginal diversification curve.** Add funds one at a time and watch portfolio
volatility, ENB, and average correlation. The volatility curve typically drops
steeply for the first few names and then flattens — the classic diminishing
returns of diversification.

See [`docs/methodology.md`](docs/methodology.md) for the full derivations and the
look-ahead-free backtest conventions.

## Caveats

- Forward-filling asynchronous NAVs dampens measured volatility for funds that
  report infrequently; treat absolute vol/correlation levels as indicative.
- Correlation uplift in stress is partly a heteroskedasticity artifact
  (Forbes & Rigobon 2002); the regime tool reports raw levels, so read it as a
  practitioner's diagnostic rather than a formal contagion test.
- This is research/educational tooling, not investment advice.

## References

See [`docs/references.md`](docs/references.md). Key sources: Markowitz (1952);
Evans & Archer (1968); Choueifaty & Coignard (2008); Meucci (2009), *Managing
Diversification*; Longin & Solnik (2001); Forbes & Rigobon (2002).

## License

MIT.

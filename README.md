# Portfolio-Optimization

A collection of quantitative portfolio optimization models and algorithms, including Modern Portfolio Theory, diversification techniques, risk analysis, and optimization methods implemented in Python.

This repository gathers self-contained research projects. Each lives in its own folder with its own package, demo script, tests, and documentation, and each is built around a single empirical or theoretical thesis tested on synthetic data.

## Projects

### [`diworsification-lab/`](./diworsification-lab)

An empirical toolkit that tests whether adding more funds/ETFs to a portfolio actually improves diversification, or whether it collapses into *diworsification* — marginal diversification benefit decays, holdings become redundant, and correlations converge toward 1 in stress regimes.

The library is organized around four stages:

- **data** — load messy real-world composition / NAV / ISIN spreadsheets
- **backtest** — reconstruct a returns-based, self-financing NAV from weights + prices
- **metrics** — quantify diversification: Effective Number of Bets (Meucci 2009), diversification ratio, average pairwise correlation, and marginal risk reduction as N grows
- **stress** — measure how correlations behave in high-volatility regimes (the "correlations go to 1" effect)

See `diworsification-lab/examples/run_analysis.py` for an end-to-end run.

### [`dsf/`](./dsf) — Duration–Spread Frontier

Treats the hyperbolic boundary in the (modified duration, Z-spread) plane — the envelope above which no bonds exist — as a first-class mathematical object, and asks three questions:

1. **Geometry** — the statistical shape of the frontier and how to estimate it rigorously (quantile regression on the envelope)
2. **Theory I** — whether the frontier is a structural consequence of the Merton (1974) credit model aggregated cross-sectionally (consistency test)
3. **Theory II** — whether the frontier reflects an issuance equilibrium: distressed issuers cannot place long-dated debt, so the boundary emerges from supply and demand rather than arbitrage

See `dsf/demo.py` for an end-to-end run.

## Getting started

Each project is an installable Python package. To work with one:

```bash
cd diworsification-lab    # or: cd dsf
pip install -e .
```

Both projects ship with synthetic data generators and runnable demos, so no external datasets are required.

| | diworsification-lab | dsf |
|---|---|---|
| Python | ≥ 3.9 | ≥ 3.10 |
| Core deps | numpy, pandas, matplotlib, openpyxl | numpy, pandas, scipy, statsmodels, matplotlib |
| Entry point | `examples/run_analysis.py` | `demo.py` |
| Tests | `pytest` | `pytest` |

## License

MIT (per-project; see each project's `pyproject.toml`).

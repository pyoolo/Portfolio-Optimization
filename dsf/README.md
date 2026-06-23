# Duration–Spread Frontier (DSF)

> **The duration–spread plane has a natural efficient frontier. This project formalizes it.**

Every fixed income portfolio manager sees it in their scatter plots: an hyperbolic boundary in the (modified duration, Z-spread) plane above which no bonds exist. This project treats that boundary as a first-class mathematical object and asks three questions:

1. **Geometry** — What is the statistical shape of the frontier, and how do we estimate it rigorously? *(quantile regression on the envelope)*
2. **Theory I** — Is the frontier a structural consequence of the Merton (1974) credit model aggregated cross-sectionally? *(consistency test)*
3. **Theory II** — Why is the region above the frontier *infeasible*? We argue it reflects an issuance equilibrium: distressed issuers cannot place long-dated debt, not because of arbitrage, but because of market microstructure and information asymmetry.

---

## Structure

```
dsf/
├── src/
│   ├── frontier/      # Chapter 1 — Quantile regression frontier estimation
│   ├── merton/        # Chapter 2 — Structural credit model consistency
│   ├── equilibrium/   # Chapter 3 — Issuance equilibrium model
│   └── utils/         # Synthetic data generation, plotting
├── notebooks/
│   ├── 01_frontier_estimation.ipynb
│   ├── 02_merton_consistency.ipynb
│   └── 03_equilibrium_model.ipynb
├── data/              # Synthetic datasets (see utils/synthetic.py)
└── tests/
```

## Quickstart

```bash
pip install -e ".[dev]"
python -m dsf.demo          # generates all three figures
```

Or run each notebook independently.

---

## The three contributions

### 1. Frontier as a quantile regression object

The frontier is estimated as the conditional upper quantile of spread given duration:

```
Q_τ(spread | duration) = exp(α + β·log(duration) + γ·log(duration)²)
```

We use `statsmodels` quantile regression on the log-linearized form. The key insight: the frontier is **not** a mean regression — it is an envelope, and must be estimated as such. We show the frontier is well-identified at τ = 0.90–0.95 and robust to outliers.

### 2. Merton consistency

Under Merton (1974), for a zero-coupon bond:

```
spread(T) = -1/T · log[N(d2) + (V/F)·N(-d1)·e^(rT)]
```

We aggregate this formula across a synthetic cross-section of firms with heterogeneous leverage and asset volatility. The prediction: the *envelope* of the resulting (duration, spread) cloud should match the empirically estimated frontier. We test goodness-of-fit with a Kolmogorov–Smirnov statistic on the frontier curve.

### 3. Issuance equilibrium

Borrowing from the information asymmetry literature (Myers–Majluf 1984, Diamond 1991), we model why the upper-left region (high spread, long duration) is structurally empty. Distressed issuers face a separating equilibrium: the market demands a premium for long-dated distressed debt that exceeds the issuer's willingness to pay. The frontier is the locus of *indifference* between issuing and not issuing. We show this implies a hyperbolic boundary with slope determined by the cross-sectional distribution of firm quality.

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Merton, R.C. (1974). *On the pricing of corporate debt*. Journal of Finance.
- Fong, H.G. & Vasicek, O. (1983). *The tradeoff between return and risk in immunized portfolios*. Financial Analysts Journal.
- Ben Dor, A., Dynkin, L., et al. (2007). *DTS (Duration Times Spread)*. Journal of Portfolio Management.
- Martin, R.J. (2020). *Fixed income portfolio optimisation: Interest rates, credit, and the efficient frontier*. arXiv:2004.02312.
- Diamond, D.W. (1991). *Debt maturity structure and liquidity risk*. Quarterly Journal of Economics.
- Myers, S.C. & Majluf, N.S. (1984). *Corporate financing and investment decisions*. Journal of Financial Economics.

---

## License

MIT. If you use this in a paper, a citation would be appreciated.

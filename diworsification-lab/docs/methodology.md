# Methodology

This document records the precise definitions and conventions used, so results
are reproducible and auditable.

## 1. NAV reconstruction (returns-based, self-financing)

Given a rebalance-date weight matrix `W_reb` (rows = rebalance dates, columns =
tickers) and an aligned daily price panel `P`:

1. **Daily as-of weights.** Reindex `W_reb` onto the price calendar and forward-
   fill, so the weight set chosen at a rebalance persists until the next one.
   Each row is renormalized to sum to 1.
2. **No look-ahead.** Lag the daily weights by one day before multiplying by
   returns: the weights known at the close of day *t−1* earn the return of day
   *t*.

   ```
   r_p(t) = Σ_i  w_i(t−1) · r_i(t)
   NAV(t) = 100 · Π_{s≤t} (1 + r_p(s))
   ```

3. This is *self-financing*: it tracks the drift of held weights between
   rebalances implicitly through the returns, and assumes costless rebalancing
   (transaction costs are out of scope and would only strengthen the
   over-diversification case).

Performance statistics annualize with 252 trading days. Sharpe assumes a zero
risk-free rate.

## 2. Effective Number of Bets (Meucci, 2009)

The number of *holdings* says nothing about the number of independent *risk
sources*. ENB measures the latter.

Let `Σ` be the asset covariance matrix and `w` the portfolio weights.
Eigendecompose `Σ = E Λ Eᵀ`, where the columns of `E` are principal components
(uncorrelated by construction) and `Λ = diag(λ₁,…,λₙ)`.

Project the portfolio onto the components, `y = Eᵀ w`. The variance contributed
by component *i* is `λᵢ yᵢ²`, and its normalized share is

```
pᵢ = λᵢ yᵢ² / Σⱼ λⱼ yⱼ²
```

The `{pᵢ}` form the *diversification distribution*. Its exponential Shannon
entropy is the effective number of bets:

```
ENB = exp( − Σᵢ pᵢ ln pᵢ )
```

- ENB = 1: all variance loads on one component → a single bet.
- ENB = N: variance spread evenly across N independent components.

We use the principal-components flavour. Minimum-torsion bets
(Meucci–Santangelo–Deguest, 2013) are a refinement that ties the uncorrelated
factors to the investment process; a future extension.

## 3. Diversification Ratio (Choueifaty & Coignard, 2008)

```
DR(w) = ( Σᵢ wᵢ σᵢ ) / sqrt( wᵀ Σ w )
```

The numerator is the weighted-average standalone volatility; the denominator is
the realized portfolio volatility. With perfectly correlated assets the two are
equal and DR = 1. `DR²` approximates the number of independent risk factors and
behaves similarly to the ENB in practice.

## 4. Marginal diversification curve (Evans & Archer, 1968)

For `k = 1 … N`, form an equal-weighted portfolio of the first `k` assets under
a chosen ordering and record annualized volatility, ENB, DR, and average
pairwise correlation. The *marginal volatility reduction* is the drop from `k−1`
to `k`. Empirically this collapses toward zero after a small number of assets;
the "knee" is reported as the point where each new fund cuts annualized vol by
less than 0.10 percentage points.

Ordering matters: by descending weight mimics how a manager actually layers
positions; alphabetical or random orderings test robustness.

## 5. Correlations in stress regimes

1. Compute trailing portfolio volatility over a 21-day window.
2. Split days into **calm / normal / stress** by the lower and upper terciles of
   that volatility series.
3. Within each bucket, compute the average off-diagonal correlation and the
   equal-weight ENB.

The diworsification signature is `avg_corr(stress) ≫ avg_corr(calm)` together
with `ENB(stress) < ENB(calm)`: diversification thins out exactly when drawdowns
cluster.

**Caveat (Forbes & Rigobon, 2002).** Measured correlation mechanically rises
with volatility even if the underlying dependence is unchanged. The tool reports
raw, unconditioned levels; it is a practitioner's diagnostic, not a formal,
heteroskedasticity-corrected contagion test. The qualitative conclusion — that
realized co-movement is higher in drawdowns — is robust and is what matters for
an investor experiencing the loss.

## 6. Per-rebalance value-added

For each rebalance at date `d`, compare the compounded return over the window
until the next rebalance using the *new* weights (what was actually done) versus
the *previous* weights (the counterfactual of not rebalancing). The difference
is the value added by that single rebalancing decision.

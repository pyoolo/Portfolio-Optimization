#!/usr/bin/env python3
"""
Duration–Spread Frontier (DSF) — End-to-end demo
=================================================

Runs all three chapters sequentially and saves publication figures.

Usage:
    python -m dsf.demo
    python -m dsf.demo --output ./figures --no-panel
"""

import sys
import os
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_demo(output_dir: str = '.', run_panel: bool = True):
    from utils.synthetic import generate_universe, generate_panel
    from frontier.estimator import FrontierEstimator, estimate_panel_frontier
    from merton.consistency import MertonCalibrator, frontier_consistency_test
    from equilibrium.model import IssuanceEquilibrium
    from utils.plots import (
        plot_frontier_estimation,
        plot_merton_consistency,
        plot_equilibrium,
        plot_panel_dynamics,
    )

    os.makedirs(output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Generating synthetic bond universe...")
    df = generate_universe()
    print(f"  {len(df)} bonds across {df['segment'].nunique()} segments")

    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Chapter 1: Frontier Estimation ──────────────────────────────────")
    est = FrontierEstimator(tau=0.90)
    est.fit(df)
    p = est.params_
    print(f"  Frontier: spread = {p['a']:.1f} × duration^{p['beta']:.3f}")
    print(f"  Interpretation: 10% increase in duration → {100*(10**p['beta']-1):.1f}% change in spread")

    print("  Quantile robustness:")
    robust = est.estimate_all_quantiles(df)
    for _, row in robust.iterrows():
        print(f"    τ={row['tau']:.2f}  β={row['beta']:.4f}  pseudo-R²={row['pseudo_r2']:.4f}")

    fig1 = plot_frontier_estimation(df, est, save_path=f'{output_dir}/fig1_frontier.png')
    print(f"  Saved → {output_dir}/fig1_frontier.png")

    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Chapter 2: Merton Consistency ───────────────────────────────────")
    print("  Calibrating Merton cross-section (n=2000 simulations)...")
    cal = MertonCalibrator(n_sim=2000)
    cal.fit(df)
    merton_cloud = cal.simulate()
    print(f"  Simulated cloud: {len(merton_cloud)} bonds")

    est_m = FrontierEstimator(tau=0.90)
    est_m.fit(merton_cloud)
    print(f"  Empirical β  = {p['beta']:.4f}")
    print(f"  Merton β     = {est_m.params_['beta']:.4f}")

    test = frontier_consistency_test(
        empirical_frontier_fn=est.predict,
        merton_cloud=merton_cloud,
        duration_grid=np.linspace(0.5, 20, 40),
    )
    print(f"  RMSE(log) = {test['rmse_log_spread']:.4f}")
    print(f"  p-value   = {test['p_value']:.3f}" if not np.isnan(test['p_value']) else "  p-value   = N/A")

    fig2 = plot_merton_consistency(df, merton_cloud, test, save_path=f'{output_dir}/fig2_merton.png')
    print(f"  Saved → {output_dir}/fig2_merton.png")

    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Chapter 3: Issuance Equilibrium ─────────────────────────────────")
    eq = IssuanceEquilibrium(kappa=500, alpha=1.5, beta=0.6, b0=120)
    frontier_curve = eq.frontier_curve(n_points=100).dropna()
    print(f"  Frontier computed for {len(frontier_curve)} quality levels")
    print(f"  Duration range: [{frontier_curve['duration'].min():.1f}, {frontier_curve['duration'].max():.1f}] years")
    print(f"  Spread range:   [{frontier_curve['spread'].min():.0f}, {frontier_curve['spread'].max():.0f}] bp")

    print("  Computing comparative statics (b0 effect)...")
    b0_values = np.linspace(60, 250, 10)
    cs_df = eq.comparative_statics('b0', b0_values, n_quality=40)

    fig3 = plot_equilibrium(eq, cs_df, empirical_df=df, save_path=f'{output_dir}/fig3_equilibrium.png')
    print(f"  Saved → {output_dir}/fig3_equilibrium.png")

    # ═══════════════════════════════════════════════════════════════════════════
    if run_panel:
        print("\n── Panel dynamics (60 periods) ─────────────────────────────────────")
        print("  Generating panel (this takes ~20s)...")
        panel = generate_panel(n_periods=60)
        panel_params = estimate_panel_frontier(panel)
        corr_alpha = panel_params['alpha'].corr(panel_params['stress_index'])
        corr_beta  = panel_params['beta'].corr(panel_params['stress_index'])
        print(f"  corr(α, stress) = {corr_alpha:.3f}")
        print(f"  corr(β, stress) = {corr_beta:.3f}")
        print(f"  → In stress: frontier {'steepens (β↓)' if corr_beta < 0 else 'flattens (β↑)'}")

        fig4 = plot_panel_dynamics(panel_params, save_path=f'{output_dir}/fig4_panel.png')
        print(f"  Saved → {output_dir}/fig4_panel.png")

    print("\n✓ Done. All figures saved to:", output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSF end-to-end demo')
    parser.add_argument('--output', default='.', help='Output directory for figures')
    parser.add_argument('--no-panel', action='store_true', help='Skip panel estimation (slow)')
    args = parser.parse_args()
    run_demo(output_dir=args.output, run_panel=not args.no_panel)

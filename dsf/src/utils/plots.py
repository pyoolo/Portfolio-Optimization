"""
Plotting utilities for the Duration–Spread Frontier project.

Produces publication-quality figures for all three chapters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import warnings

# ── Style ──────────────────────────────────────────────────────────────────────

PALETTE = {
    'IG':          '#185FA5',
    'HY':          '#BA7517',
    'Distressed':  '#A32D2D',
    'frontier':    '#2C2C2A',
    'merton':      '#0F6E56',
    'equilibrium': '#533AB7',
    'dts_iso':     '#B4B2A9',
    'bg':          '#FAFAF8',
    'grid':        '#E8E6DF',
}

SEGMENT_LABELS = {
    'IG': 'Investment Grade',
    'HY': 'High Yield',
    'Distressed': 'Distressed',
}


def _setup_axes(ax, xlabel='Modified Duration (years)', ylabel='Z-Spread (bp)'):
    ax.set_facecolor(PALETTE['bg'])
    ax.grid(True, color=PALETTE['grid'], linewidth=0.6, zorder=0)
    ax.set_xlabel(xlabel, fontsize=10, color='#444441')
    ax.set_ylabel(ylabel, fontsize=10, color='#444441')
    ax.tick_params(colors='#888780', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE['grid'])
    return ax


# ── Figure 1: Frontier estimation ──────────────────────────────────────────────

def plot_frontier_estimation(
    df: pd.DataFrame,
    estimator,
    taus=(0.85, 0.90, 0.95),
    save_path: str = None,
):
    """
    Figure 1: Scatter plot of bond universe with estimated frontier at
    multiple quantile levels, and DTS isocurves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle(
        'Duration–Spread Frontier: Quantile Regression Estimation',
        fontsize=13, fontweight='500', color='#2C2C2A', y=1.01
    )

    # ── Left panel: scatter + frontier ────────────────────────────────────────
    ax = _setup_axes(axes[0])

    for seg, color in PALETTE.items():
        if seg not in ['IG', 'HY', 'Distressed']:
            continue
        sub = df[df['segment'] == seg]
        ax.scatter(
            sub['duration'], sub['spread'],
            c=color, alpha=0.35, s=sub['mtm'] / sub['mtm'].max() * 60 + 8,
            label=SEGMENT_LABELS[seg], zorder=2, linewidths=0,
        )

    # DTS isocurves
    d_grid = np.linspace(0.3, 26, 200)
    for dts_val, lw in [(500, 0.6), (1500, 0.8), (4000, 0.8), (10000, 0.8)]:
        s_iso = dts_val / np.clip(d_grid, 0.1, None)
        ax.plot(d_grid, s_iso, '--', color=PALETTE['dts_iso'],
                linewidth=lw, alpha=0.7, zorder=1)
        idx = np.searchsorted(d_grid, 18)
        if s_iso[idx] < 4500:
            ax.text(d_grid[idx], s_iso[idx], f'DTS={dts_val}',
                    fontsize=7, color='#888780', va='bottom')

    # Frontier curves at multiple quantiles
    tau_styles = {0.85: ('--', 0.8), 0.90: ('-', 2.0), 0.95: (':', 1.2)}
    from frontier.estimator import FrontierEstimator
    for tau in taus:
        est_t = FrontierEstimator(tau=tau)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est_t.fit(df)
        f_vals = est_t.predict(d_grid)
        ls, lw = tau_styles.get(tau, ('-', 1.0))
        ax.plot(d_grid, f_vals, ls, color=PALETTE['frontier'],
                linewidth=lw, label=f'Frontier τ={tau}', zorder=5)

    ax.set_xlim(-3, 27)
    ax.set_ylim(-50, 5500)
    ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
    ax.set_title('Bond universe with estimated frontiers', fontsize=10, pad=8)

    # ── Right panel: log-log with linear fit ──────────────────────────────────
    ax2 = _setup_axes(axes[1], xlabel='log(Duration)', ylabel='log(Z-Spread)')

    for seg, color in PALETTE.items():
        if seg not in ['IG', 'HY', 'Distressed']:
            continue
        sub = df[(df['segment'] == seg) & (df['duration'] > 0.25)]
        ax2.scatter(
            np.log(sub['duration']), np.log(sub['spread'].clip(lower=1)),
            c=color, alpha=0.30, s=12, zorder=2, linewidths=0,
        )

    # Linear quantile regression fits
    log_d = np.linspace(-0.5, 3.5, 200)
    for tau in taus:
        est_t = FrontierEstimator(tau=tau)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            est_t.fit(df)
        p = est_t.params_
        log_s = p['alpha'] + p['beta'] * log_d
        ls, lw = tau_styles.get(tau, ('-', 1.0))
        ax2.plot(log_d, log_s, ls, color=PALETTE['frontier'],
                 linewidth=lw, label=f'τ={tau}  β={p["beta"]:.3f}', zorder=5)

    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.set_title('Log–log space: linear quantile regression', fontsize=10, pad=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ── Figure 2: Merton consistency ──────────────────────────────────────────────

def plot_merton_consistency(
    empirical_df: pd.DataFrame,
    merton_cloud: pd.DataFrame,
    test_result: dict,
    save_path: str = None,
):
    """
    Figure 2: Overlay empirical and Merton-simulated clouds, compare frontiers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle(
        'Merton (1974) Consistency: Is the Frontier Structural?',
        fontsize=13, fontweight='500', color='#2C2C2A', y=1.01
    )

    # ── Left: overlay scatter ─────────────────────────────────────────────────
    ax = _setup_axes(axes[0])
    ax.scatter(empirical_df['duration'], empirical_df['spread'],
               c=PALETTE['IG'], alpha=0.25, s=8, label='Empirical bonds', zorder=2, linewidths=0)
    ax.scatter(merton_cloud['duration'], merton_cloud['spread'],
               c=PALETTE['merton'], alpha=0.25, s=8, label='Merton-simulated bonds', zorder=2, linewidths=0)

    d_grid = test_result['duration_grid']
    ax.plot(d_grid, test_result['empirical_frontier'], '-',
            color=PALETTE['frontier'], linewidth=2.0, label='Empirical frontier (τ=0.90)', zorder=5)
    ax.plot(d_grid, test_result['merton_frontier'], '--',
            color=PALETTE['merton'], linewidth=2.0, label='Merton frontier (τ=0.90)', zorder=5)

    ax.set_xlim(0, 22)
    ax.set_ylim(0, 5000)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_title('Empirical vs. Merton-implied cloud and frontier', fontsize=10, pad=8)

    # ── Right: residual analysis ──────────────────────────────────────────────
    ax2 = _setup_axes(axes[1], xlabel='Duration (years)', ylabel='log(emp. frontier) − log(Merton frontier)')

    log_resid = np.log(test_result['empirical_frontier']) - np.log(test_result['merton_frontier'])
    ax2.axhline(0, color=PALETTE['frontier'], linewidth=1.0, linestyle='--')
    ax2.fill_between(d_grid, log_resid, 0, alpha=0.3,
                     color=PALETTE['merton'] if log_resid.mean() > 0 else PALETTE['Distressed'])
    ax2.plot(d_grid, log_resid, '-', color='#2C2C2A', linewidth=1.5)

    isd = test_result['isd']
    rmse = test_result['rmse_log_spread']
    pval = test_result.get('p_value', np.nan)

    stats_text = (
        f"RMSE(log) = {rmse:.4f}\n"
        f"ISD = {isd:.6f}\n"
        f"p-value = {pval:.3f}" if not np.isnan(pval) else f"RMSE(log) = {rmse:.4f}\nISD = {isd:.6f}"
    )
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    ax2.set_title('Frontier residuals (log scale)', fontsize=10, pad=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ── Figure 3: Equilibrium frontier ────────────────────────────────────────────

def plot_equilibrium(
    equilibrium,
    comparative_statics_df: pd.DataFrame,
    empirical_df: pd.DataFrame = None,
    save_path: str = None,
):
    """
    Figure 3: Equilibrium frontier and comparative statics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle(
        'Issuance Equilibrium: Why the Upper-Left Region is Empty',
        fontsize=13, fontweight='500', color='#2C2C2A', y=1.01
    )

    # ── Left: frontier + feasibility regions ─────────────────────────────────
    ax = _setup_axes(axes[0])

    if empirical_df is not None:
        ax.scatter(empirical_df['duration'], empirical_df['spread'],
                   c='#D3D1C7', alpha=0.2, s=8, zorder=1, linewidths=0, label='Empirical bonds')

    # Equilibrium bonds
    bonds = equilibrium.simulate_bonds(n=600)
    ax.scatter(bonds['duration'], bonds['spread'],
               c=bonds['quality'], cmap='RdYlGn', alpha=0.5, s=15,
               vmin=0, vmax=1, zorder=2, linewidths=0)

    # Frontier curve
    frontier = equilibrium.frontier_curve(n_points=120).dropna()
    ax.plot(frontier['duration'], frontier['spread'], '-',
            color=PALETTE['equilibrium'], linewidth=2.5, zorder=5,
            label='Equilibrium frontier')

    # Feasibility regions
    ax.fill_betweenx(
        [0, 6000], [-3, -3], [0, 0],
        color='#EEEDFE', alpha=0.4, zorder=0,
    )
    ax.text(12, 4500, 'INFEASIBLE\n(no issuer willing\nto issue here)',
            fontsize=9, color=PALETTE['equilibrium'], ha='center',
            style='italic', alpha=0.8)
    ax.text(8, 800, 'FEASIBLE\n(bonds issued)',
            fontsize=9, color='#5F5E5A', ha='center', style='italic')

    ax.set_xlim(-1, 25)
    ax.set_ylim(0, 5500)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_title('Equilibrium frontier separates feasible from infeasible', fontsize=10, pad=8)

    # ── Right: comparative statics (b0 effect) ────────────────────────────────
    ax2 = _setup_axes(axes[1])

    b0_vals = comparative_statics_df['param_value'].unique()
    cmap = plt.cm.viridis
    colors = [cmap(i / len(b0_vals)) for i in range(len(b0_vals))]

    for i, b0_val in enumerate(b0_vals):
        sub = comparative_statics_df[comparative_statics_df['param_value'] == b0_val].dropna()
        if len(sub) < 5:
            continue
        ax2.plot(sub['duration'], sub['spread'], '-',
                 color=colors[i], linewidth=1.2, alpha=0.8)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(b0_vals.min(), b0_vals.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.04, pad=0.02)
    cbar.set_label('b₀ (issuer benefit rate)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax2.set_xlim(0, 25)
    ax2.set_ylim(0, 4500)
    ax2.set_title('Comparative statics: frontier shifts with b₀', fontsize=10, pad=8)
    ax2.text(0.05, 0.95,
             'Higher b₀ → distressed issuers\ncan access longer maturities\n→ frontier shifts up-right',
             transform=ax2.transAxes, fontsize=8.5, va='top', color='#444441',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ── Figure 4: Panel dynamics ───────────────────────────────────────────────────

def plot_panel_dynamics(panel_params: pd.DataFrame, save_path: str = None):
    """
    Figure 4: Time evolution of frontier parameters (alpha, beta) vs. stress.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), facecolor='white')
    fig.suptitle('Frontier Dynamics: Parameter Evolution Over Time',
                 fontsize=13, fontweight='500', color='#2C2C2A', y=1.01)

    for ax, (col, label, color) in zip(axes, [
        ('alpha', 'α (log-intercept)', PALETTE['IG']),
        ('beta',  'β (log-slope, duration sensitivity)', PALETTE['Distressed']),
    ]):
        _setup_axes(ax, xlabel='Period', ylabel=label)
        ax.plot(panel_params['period'], panel_params[col], '-',
                color=color, linewidth=1.5, label=label)
        ax2_twin = ax.twinx()
        ax2_twin.plot(panel_params['period'], panel_params['stress_index'], '--',
                      color='#888780', linewidth=1.0, alpha=0.7, label='Stress index')
        ax2_twin.set_ylabel('Stress index', fontsize=9, color='#888780')
        ax2_twin.tick_params(axis='y', colors='#888780', labelsize=8)

        corr = panel_params[col].corr(panel_params['stress_index'])
        ax.set_title(f'corr({col}, stress) = {corr:.3f}', fontsize=10, pad=8)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

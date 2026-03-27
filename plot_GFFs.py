"""
plot_GFFs.py — Reusable Plotting Helpers for GFF Reconstructions
================================================================
Authors: H. Alharazin, J. Yu. Panteleeva

Provides two plotting functions for visualising DDPM-generated
gravitational form factor curves:

  - plot_on_ax : draws a reconstruction onto an existing Axes object
                 (for multi-panel figures)
  - plot       : creates a standalone single-panel figure

Both functions load a .pt results file produced by sample_GFF() and
overlay the median/mean curves, 68% CI band, known data points
(with optional error bars), optional EMT constraints, and optional
lattice reference curves (dipole and z-expansion fits).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
#  Multi-panel helper: plot a single GFF reconstruction on a given Axes
# ═══════════════════════════════════════════════════════════════════

def plot_on_ax(ax, path=None, FF=None, title=None,
               LatticeFF_dipole=None, LatticeFF_zexp=None,
               EMT_Condition=False, y_lim=(0, 1.5), legend=True,
               lattice_errors=None, chPlabel = False, DFF = False):
    """
    Draw a GFF reconstruction onto an existing matplotlib Axes.

    Parameters
    ----------
    ax              : matplotlib Axes to draw on
    path            : path to a .pt file produced by sample_GFF()
    FF              : LaTeX string for the form factor name (e.g. "A", "J", "D")
    title           : optional plot title
    LatticeFF_dipole: array of lattice dipole-fit reference values on t_grid
    LatticeFF_zexp  : array of lattice z-expansion reference values on t_grid
    EMT_Condition   : if True, treat the first data point as an EMT constraint
                      (plotted in green, excluded from the main scatter)
    y_lim           : y-axis limits
    legend          : whether to show a legend
    lattice_errors  : dict {index: error} for error bars on conditioning points
    chPlabel        : if True, label data as "ChPT data" instead of "Lattice data"
    DFF             : if True, skip EMT-constraint special handling for D(t)
    """
    # Choose label based on data source
    if chPlabel:
        labellc = "ChPT data"
    else:
        labellc = "Lattice data [Hackett et al.]"

    # ── Load saved results ──────────────────────────────────
    orig = torch.load(path, weights_only=False)
    mean      = orig['mean_curve'].numpy()
    median    = orig['median_curve'].numpy()
    q16       = orig['q16'].numpy()
    q84       = orig['q84'].numpy()
    t_grid    = -orig['t_grid'].numpy()        # Plot as −t (positive x-axis)
    t_known   = -orig['t_known'].numpy()
    gff_known_full = orig['gff_known'].numpy()

    # For form factors with an EMT constraint (A(0)=1, J(0)=1/2),
    # the first point is the constraint — exclude it from the main scatter
    if EMT_Condition:
        if not DFF:
            gff_known = gff_known_full[1:]
    else:
        gff_known = gff_known_full

    # ── Main result: median + mean + 68% CI band ───────────
    ax.fill_between(t_grid, q16, q84,
                    alpha=0.25, color="steelblue",
                    label=r"$68\%$ CI (16th–84th percentile)")#[1:]
    ax.plot(t_grid, median, color="darkblue", lw=2.5, label="Median")
    ax.plot(t_grid, mean, color="darkblue", lw=1.5, ls="--", alpha=0.5, label="Mean")

    # ── Lattice / ChPT conditioning points ──────────────────
    if EMT_Condition:
        t_pts = t_known[1:]      # Exclude the EMT constraint point
    else:
        t_pts = t_known

    if lattice_errors is not None:
        # Plot with error bars when experimental uncertainties are provided
        yerr = np.array([lattice_errors.get(i, 0.0) for i in range(len(gff_known))])
        ax.errorbar(t_pts, gff_known, yerr=yerr,
                    fmt='o', color="red", ms=6, zorder=5,
                    ecolor="darkred", elinewidth=1.5, capsize=3,
                    markeredgecolor="darkred", markeredgewidth=1,
                    label=labellc)
    else:
        # Simple scatter when no errors are available
        ax.scatter(t_pts, gff_known,
                   color="red", s=33, zorder=5,
                   edgecolors="darkred", lw=3, label=labellc)

    # Plot the EMT constraint point separately (in green)
    if EMT_Condition and not DFF:
        ax.scatter(t_known[0], gff_known_full[0], marker='o', s=33,
                   color="green", zorder=5, linewidths=3,
                   label="EMT constraint")

    # ── Reference curves from lattice fits ──────────────────
    # These are visually subdued (gray, thin) to avoid dominating the plot
    ref_color = "gray"
    if LatticeFF_dipole is not None:
        ax.plot(t_grid, LatticeFF_dipole,
                color=ref_color, lw=1.5, ls="--", alpha=0.7,
                label="Lattice dipole fit [Hackett et al.]")
    if LatticeFF_zexp is not None:
        ax.plot(t_grid, LatticeFF_zexp,
                color=ref_color, lw=1.5, ls=":", alpha=0.7,
                label="Lattice $z$-expansion [Hackett et al.]")

    # ── Styling ─────────────────────────────────────────────
    ax.set_ylim(*y_lim)
    ax.grid(True, alpha=0.3)

    if legend:
        ax.legend(fontsize=10, framealpha=0.9)

    # Only the bottom panel in a multi-panel figure should have an x-label
    if legend:
        ax.set_xlabel(r"$-t\;[\mathrm{GeV}^2]$", fontsize=13)

    ax.set_ylabel(rf"${FF}(t)$", fontsize=13)


# ═══════════════════════════════════════════════════════════════════
#  Standalone helper: create a single-panel figure for one GFF
# ═══════════════════════════════════════════════════════════════════

def plot(path=None, FF=None, title=None, LatticeFF_dipole=None, LatticeFF_zexp=None, EMT_Condition=False, y_lim=(0, 1.5), legend=True,
         lattice_errors=None, chPlabel = False, DFF = False):
    """
    Create a standalone figure for a single GFF reconstruction.

    Parameters are identical to plot_on_ax(); this function creates
    its own Figure and Axes, then calls plt.show().
    """
    if chPlabel:
        labellc = "ChPT data"
    else:
        labellc = "lattice data"

    # ── Load saved results ──────────────────────────────────
    orig = torch.load(path, weights_only=False)
    mean      = orig['mean_curve'].numpy()
    median    = orig['median_curve'].numpy()
    q16       = orig['q16'].numpy()
    q84       = orig['q84'].numpy()
    t_grid    = -orig['t_grid'].numpy()
    t_known   = -orig['t_known'].numpy()
    gff_known_full = orig['gff_known'].numpy()

    if EMT_Condition and not DFF:
        gff_known = gff_known_full[1:]
    else:
        gff_known = gff_known_full

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    # ── 1. Our result: median + mean + 68% CI band ─────────
    ax.fill_between(t_grid, q16, q84,
                    alpha=0.25, color="steelblue",
                    label=r"$68\%$ CI (16th–84th percentile)")
    ax.plot(t_grid, median, color="darkblue", lw=2.5, label="Median")
    ax.plot(t_grid, mean, color="darkblue", lw=1.5, ls="--", alpha=0.5, label="Mean")

    # ── 2. Lattice / ChPT conditioning points ──────────────
    t_pts = t_known[1:] if (EMT_Condition and not DFF) else t_known

    if lattice_errors is not None:
        yerr = np.array([lattice_errors.get(i, 0.0) for i in range(len(gff_known))])
        ax.errorbar(t_pts, gff_known, yerr=yerr,
                    fmt='o', color="red", ms=7, zorder=5,
                    ecolor="darkred", elinewidth=1.5, capsize=3,
                    markeredgecolor="black", markeredgewidth=1,
                    label=labellc)
    else:
        ax.scatter(t_pts, gff_known,
                   color="red", s=50, zorder=5,
                   edgecolors="darkred", lw=1, label=labellc)

    if EMT_Condition and not DFF:
        ax.scatter(t_known[0], gff_known_full[0], marker='o', s=50,
                   color="green", zorder=5, linewidths=2, label="EMT constraint")

    # ── 3. Reference curves: visually subdued ──────────────
    ref_color = "gray"
    if LatticeFF_dipole is not None:
        ax.plot(t_grid, LatticeFF_dipole,
                color=ref_color, lw=1.5, ls="--", alpha=0.7,
                label="Lattice dipole fit [Hackett et al.]")
    if LatticeFF_zexp is not None:
        ax.plot(t_grid, LatticeFF_zexp,
                color=ref_color, lw=1.5, ls=":", alpha=0.7,
                label="Lattice $z$-expansion [Hackett et al.]")

    # ── Axes and styling ─────────────────────────────────────
    ax.set_xlabel(r"$-t\;[\mathrm{GeV}^2]$", fontsize=13)
    ax.set_ylabel(rf"${FF}(t)$", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(*y_lim)
    if legend:
        ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.show()

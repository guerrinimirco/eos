#!/usr/bin/env python
"""PSR J0614-3329: Mauviard et al. 2025 vs Miller et al. 2026 M-R contours.

Both sets live in data/contours/ as `# R_km  M_sun` polygons:
  J0614_{68,95}.txt         Mauviard+25 (Amsterdam, X-PSI ST+PDT, NICER+XMM)
  J0614_Miller_{68,95}.txt  Miller+26 (Maryland-Illinois, 3 circles, NICER only)

    python plot_j0614_comparison.py   # -> j0614_comparison.png + printed stats
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from eos.general.figure_style import (STANDARD_COLORS, set_global_style,
                                      setup_scientific_figure, apply_style,
                                      save_figure)

HERE = Path(__file__).resolve().parent
C = HERE / "data" / "contours"

SETS = {
    "Mauviard+2025 (NICER+XMM, ST+PDT)":  ("J0614",        STANDARD_COLORS['Green']),
    "Miller+2026 (NICER only, 3 circles)": ("J0614_Miller", STANDARD_COLORS['Red']),
}


def load(stem, lvl):
    return np.loadtxt(C / f"{stem}_{lvl}.txt").T          # -> (R[], M[])


def area(R, M):
    """Shoelace area of the closed polygon, in km*Msun."""
    return 0.5 * abs(np.dot(R, np.roll(M, 1)) - np.dot(M, np.roll(R, 1)))


def main():
    set_global_style()
    fig, ax = setup_scientific_figure()
    for label, (stem, col) in SETS.items():
        R95, M95 = load(stem, 95)
        R68, M68 = load(stem, 68)
        ax.fill(R95, M95, color=col, alpha=0.20, lw=0)
        ax.fill(R68, M68, color=col, alpha=0.45, lw=0)
        ax.plot(np.append(R95, R95[0]), np.append(M95, M95[0]), color=col, lw=1.0)
        ax.plot(np.append(R68, R68[0]), np.append(M68, M68[0]), color=col, lw=1.6)
        ax.plot([], [], color=col, lw=6, alpha=0.5, label=label)

        print(f"{label}")
        for lvl, (R, M) in (("68%", (R68, M68)), ("95%", (R95, M95))):
            print(f"   {lvl}: R in [{R.min():5.2f},{R.max():5.2f}] km "
                  f"(extent {R.ptp():4.2f}), M in [{M.min():.3f},{M.max():.3f}], "
                  f"area {area(R, M):.3f} km Msun")

    ax.set_xlabel(r"equatorial radius $R_{\rm eq}$ [km]")
    ax.set_ylabel(r"gravitational mass $M$ [$M_\odot$]")
    ax.set_title("PSR J0614$-$3329: 68% / 95% credible regions")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    apply_style(ax, legend=False)
    fig.tight_layout()
    save_figure(fig, str(HERE / "j0614_comparison"))
    print(f"\nwrote {HERE / 'j0614_comparison'}.png/.pdf")


if __name__ == "__main__":
    main()

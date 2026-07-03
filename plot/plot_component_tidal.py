#!/usr/bin/env python
"""Per-component mass-tidal contours for GW170817 & GW190425.

Two panels: primary (m1, Λ1) and secondary (m2, Λ2).  Contours are the 68%/95%
enclosed-probability regions of the raw posterior samples, built on the fly with
the same KDE machinery as compute_contours.py (nothing precomputed here).

    python plot_component_tidal.py       # writes component_tidal_contours.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from compute_contours import extract_2d_contour, SAMPLES

# source -> (label, colour, sample file, (cM1,cM2,cΛ1,cΛ2)).  Both are the
# low-spin source-frame extracts (m1,m2,Λ1,Λ2 → cols 0..3) from fetch_gw1708NN.py.
SOURCES = {
    "GW170817": ("GW170817", (0.24, 0.6, 0.8),  "gw170817_extracted_table.txt", (0, 1, 2, 3)),
    "GW190425": ("GW190425", (0.8, 0.25, 0.33), "gw190425_extracted_table.txt", (0, 1, 2, 3)),
}


def _fill(ax, m, lam, colour):
    """Draw the 68% (dark) + 95% (light) KDE contours of (m, Λ) on ax."""
    paths = extract_2d_contour(m, lam)
    ax.fill(*paths[0.95].T, color=colour, alpha=0.30, lw=0)
    ax.fill(*paths[0.68].T, color=colour, alpha=0.55, lw=0)


def main():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    # both components of both events on one axis: colour = event, primary and
    # secondary blobs separate by their mass position.
    for key, (label, colour, fname, (cM1, cM2, cL1, cL2)) in SOURCES.items():
        a = np.loadtxt(SAMPLES / fname)
        _fill(ax, a[:, cM1], a[:, cL1], colour)           # primary: m1, Λ1
        _fill(ax, a[:, cM2], a[:, cL2], colour)           # secondary: m2, Λ2
        ax.plot([], [], color=colour, lw=6, alpha=0.55, label=label)  # legend proxy

    ax.set_xlabel(r"mass  $m\ [M_\odot]$")
    ax.set_ylabel(r"tidal deformability  $\Lambda$")
    ax.set_ylim(bottom=0)                                 # Λ >= 0; clip KDE-pad tail
    ax.legend(frameon=False)
    ax.set_title("Component mass–tidal posteriors (68% / 95%)")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "component_tidal_contours.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

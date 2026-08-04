#!/usr/bin/env python
"""Plot the GW170817 & GW190425 mass-tidal (chirp mass – Λ̃) 68%/95% contours.

Reads the precomputed CSVs from data/contours/ (built by compute_contours.py)
and draws each source as two soft filled credible regions: 95% light, 68% darker.

    python plot_tidal_contours.py        # writes tidal_contours.png (+ shows if GUI)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# Contours are shipped inside the package -- see compute_contours.py.
CONTOURS = HERE.parent / "eos" / "general" / "data" / "contours"

# source key -> (label, RGB colour).  Keys match the CSV basenames.
SOURCES = {
    "GW170817": ("GW170817", (0.24, 0.6, 0.8)),   # blue
    "GW190425": ("GW190425", (0.8, 0.25, 0.33)),  # red
}


def _load(key, lvl):
    """(Mc, Λ̃) columns of one contour, or None if the CSV is absent."""
    f = CONTOURS / f"{key}_{lvl}.csv"
    if not f.exists():
        return None
    return np.loadtxt(f, delimiter=",", skiprows=1).T   # -> (Mc[], Λt[])


def main():
    fig, ax = plt.subplots(figsize=(6, 5))
    for key, (label, colour) in SOURCES.items():
        c95, c68 = _load(key, 95), _load(key, 68)
        if c95 is None:
            print(f"  {key}: no 95% contour — skipping")
            continue
        ax.fill(*c95, color=colour, alpha=0.30, lw=0)                 # 2σ light
        if c68 is not None:
            ax.fill(*c68, color=colour, alpha=0.55, lw=0)            # 1σ darker
        ax.plot([], [], color=colour, lw=6, alpha=0.55, label=label)  # legend proxy

    ax.set_xlabel(r"chirp mass  $\mathcal{M}_c\ [M_\odot]$")
    ax.set_ylabel(r"effective tidal deformability  $\tilde\Lambda$")
    ax.set_ylim(bottom=0)                          # Λ̃ >= 0; clip the KDE-pad tail
    ax.legend(frameon=False)
    ax.set_title("GW170817 & GW190425 mass–tidal posteriors (68% / 95%)")
    fig.tight_layout()
    out = HERE / "tidal_contours.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.show()


if __name__ == "__main__":
    main()

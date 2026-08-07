"""
observational_constraints.py
============================
Observational constraint overlays for the two planes an equation of state is
usually judged in:

* **mass-radius** -- NICER / HESS credible regions plus mass-only bands, via
  ``add_observational_constraints``;
* **mass-tidal deformability** -- the per-component (m, Lambda) posteriors of
  GW170817 and GW190425, via ``add_tidal_constraints``.

The 68% / 95% credible contours are precomputed OFFLINE by
``plot/compute_contours.py`` (which rebuilds the KDEs from the raw posterior
samples in ``plot/data/samples/``) and shipped inside this package as CSV files
under ``eos/general/data/contours/``. Plotting therefore never rebuilds a KDE,
and both functions work straight from an installed wheel with no path argument.

Usage::

    from eos.general.observational_constraints import (
        add_observational_constraints, add_tidal_constraints)
    add_observational_constraints(ax)                    # packaged contours
    add_observational_constraints(ax, contour_dir=other) # a custom set
    add_tidal_constraints(ax_lambda)                     # M on x, Lambda on y
"""
import csv
from pathlib import Path

import numpy as np

from eos.general.figure_style import STANDARD_COLORS

# Contours shipped with the package. Passing contour_dir=None (the default)
# resolves here, so the caller needs no knowledge of the install layout.
DEFAULT_CONTOUR_DIR = Path(__file__).resolve().parent / "data" / "contours"

# 2D M-R posteriors: CSV basename -> (full label, colour, label-anchor).
# Soft fills (no border) so they read as background; anchor = (dx, dy, ha, va)
# offsets the inline label from the blob centroid in DATA units (dx in km,
# dy in M_sun); ha/va are its text alignment. dy>0 + 'bottom' = above the blob,
# dy<0 + 'top' = below it. Tune the anchors when axis limits change.
MR_LABEL_FS = 8.5         # source labels (PSR/HESS + mass band); < body text on purpose
MR_CONSTRAINTS = {
    "J0030": ("PSR J0030+0451", STANDARD_COLORS['Orange'],  (0.0,  -0.1, 'center', 'center')),
    "J0740": ("PSR J0740+6620", STANDARD_COLORS['Blue'],    (0.0,   0.0, 'center', 'center')),
    "J0614": ("PSR J0614-3329", STANDARD_COLORS['Green'],   (-0.3,  0.0, 'center', 'center')),
    "HESS":  ("HESS J1731-347", STANDARD_COLORS['Magenta'], (0.0,  -0.2, 'center', 'center')),
}
# 1D mass-only measurements -> horizontal bands. Only J0952-0607 is shown as a
# band (J0740 is already an M-R contour). Gold fill + darker brown line/label.
MASS_LABELS = {"J0952-0607": "PSR J0952-0607"}
MASS_COLORS = {"J0952-0607": STANDARD_COLORS['Yellow']}
MASS_TEXT = {"J0952-0607": STANDARD_COLORS['Brown']}

# 2D per-component (m, Lambda) posteriors of the two BNS detections: event ->
# (label, colour, csv stem). Each event contributes TWO blobs, one per star,
# and they share a colour and a single legend entry because the two components
# of one binary share one equation of state -- the Lambda(M) curve has to pass
# through both. The stems resolve to <stem>_1_{68,95}.csv (primary) and
# <stem>_2_{68,95}.csv (secondary).
#
# These are NOT the GW170817_*.csv / GW190425_*.csv files, which hold Lambda-
# tilde against chirp mass: that is a property of the binary as a whole and
# cannot be drawn against a single sequence without first choosing a mass
# ratio. Both sets are built by plot/compute_contours.py.
TIDAL_CONSTRAINTS = {
    "GW170817": ("GW170817", STANDARD_COLORS['Blue'],    "GW170817ML"),
    "GW190425": ("GW190425", STANDARD_COLORS['Magenta'], "GW190425ML"),
}


def _smooth_closed(x, y, frac=0.04):
    """Low-pass a *closed* contour so a jagged posterior boundary reads as a
    smooth blob (like published M-R figures). Periodic moving average: wrap-pad
    the loop, box-filter x(t) and y(t), unwrap. ``frac`` = window as a fraction
    of the point count (bigger = smoother, but shrinks convex bulges more)."""
    n = len(x)
    if n < 8:
        return x, y                                   # too few points to smooth
    w = max(3, int(n * frac) | 1)                     # odd window
    k = np.ones(w) / w
    xp, yp = np.r_[x[-w:], x, x[:w]], np.r_[y[-w:], y, y[:w]]   # periodic pad
    return np.convolve(xp, k, 'same')[w:-w], np.convolve(yp, k, 'same')[w:-w]


def add_observational_constraints(ax, contour_dir=None, show_mass_bands=True,
                                  inline_labels=False):
    """Overlay precomputed observational constraints *behind* the model curves.

    - 2D M-R posteriors: 95% + 68% credible regions as soft pastel fills (no
      border). With inline_labels=False a boundary line carries a legend entry;
      with inline_labels=True the full source name is written near the blob.
    - 1D mass-only measurements: horizontal ``axhspan`` bands (68% + 95%),
      annotated at the left edge.
    Everything is drawn at low zorder (0-1) so the model curves stay on top.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes with radius [km] on x and gravitational mass [M_sun] on y.
    contour_dir : str, Path or None
        Directory holding the ``<key>_68.csv`` / ``<key>_95.csv`` contour files
        and ``mass_bounds.csv``. None (default) uses the contours shipped with
        the package. Missing files are silently skipped, so a partial contour
        set degrades gracefully instead of raising.
    show_mass_bands : bool
        Draw the 1D mass-only horizontal bands.
    inline_labels : bool
        True  -> write the source name near each blob (no legend entry).
        False -> draw the 68% boundary as a labelled line for the legend.
    """
    contour_dir = Path(DEFAULT_CONTOUR_DIR if contour_dir is None else contour_dir)
    for key, (label, colour, anchor) in MR_CONSTRAINTS.items():
        f95, f68 = contour_dir / f"{key}_95.csv", contour_dir / f"{key}_68.csv"
        if not f95.exists():
            continue
        R95, M95 = _smooth_closed(*np.loadtxt(f95, delimiter=",", skiprows=1).T)
        ax.fill(R95, M95, color=colour, alpha=0.20, lw=0, zorder=0)      # 2 sigma light
        if f68.exists():
            R68, M68 = _smooth_closed(*np.loadtxt(f68, delimiter=",", skiprows=1).T)
            ax.fill(R68, M68, color=colour, alpha=0.40, lw=0, zorder=1)  # 1 sigma darker
            Rc, Mc = R68, M68
        else:
            Rc, Mc = R95, M95
        if inline_labels:
            if anchor is not None:                    # label once per source
                dx, dy, ha, va = anchor
                ax.text(Rc.mean() + dx, Mc.mean() + dy, label, color=colour,
                        fontsize=MR_LABEL_FS, fontweight='bold', ha=ha, va=va,
                        zorder=3, clip_on=True)
        else:                                         # legend path: draw boundary
            ax.plot(Rc, Mc, color=colour, lw=1.2, zorder=1, label=label)

    if show_mass_bands and (contour_dir / "mass_bounds.csv").exists():
        with open(contour_dir / "mass_bounds.csv") as fh:
            rows = list(csv.DictReader(fh))
        for name, disp in MASS_LABELS.items():
            by_lvl = {r["level"]: r for r in rows if r["name"] == name}
            if not by_lvl:
                continue
            colour = MASS_COLORS.get(name, "gray")
            txt_c = MASS_TEXT.get(name, colour)       # darker tone for line/label
            for lvl, a in (("95", 0.10), ("68", 0.18)):   # 2 sigma lighter
                r = by_lvl[lvl]
                ax.axhspan(float(r["lower"]), float(r["upper"]),
                           color=colour, alpha=a, lw=0, zorder=0)
            # x in axes fraction (blended transform) -> pinned to the left edge,
            # independent of the current xlim, so it never escapes the panel.
            ax.text(0.02, float(by_lvl["68"]["upper"]), " " + disp,
                    transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                    fontsize=MR_LABEL_FS, fontweight='bold', color=txt_c,
                    zorder=3, clip_on=True)


def add_tidal_constraints(ax, contour_dir=None, inline_labels=False):
    """Overlay the GW170817 / GW190425 per-component (m, Lambda) posteriors
    *behind* the model curves -- the tidal counterpart of
    :func:`add_observational_constraints`.

    Each event draws two blobs, one per neutron star, in one colour with one
    legend entry: both components of a binary share an equation of state, so a
    single Lambda(M) sequence has to thread through both. Fills and zorder
    follow the M-R overlay exactly, so the two panels of a figure read the same.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes with gravitational mass [M_sun] on x and the dimensionless tidal
        deformability Lambda on y.
    contour_dir : str, Path or None
        Directory holding the ``<stem>_<component>_68.csv`` / ``_95.csv``
        contour files. None (default) uses the contours shipped with the
        package. Missing files are silently skipped.
    inline_labels : bool
        True  -> write the event name near its primary blob (no legend entry).
        False -> draw the 68% boundary of the primary as a labelled line.
    """
    contour_dir = Path(DEFAULT_CONTOUR_DIR if contour_dir is None else contour_dir)
    for label, colour, stem in TIDAL_CONSTRAINTS.values():
        for comp in ("1", "2"):
            f95 = contour_dir / f"{stem}_{comp}_95.csv"
            f68 = contour_dir / f"{stem}_{comp}_68.csv"
            if not f95.exists():
                continue
            M95, L95 = _smooth_closed(*np.loadtxt(f95, delimiter=",",
                                                  skiprows=1).T)
            ax.fill(M95, L95, color=colour, alpha=0.20, lw=0, zorder=0)
            Mc, Lc = M95, L95
            if f68.exists():
                M68, L68 = _smooth_closed(*np.loadtxt(f68, delimiter=",",
                                                      skiprows=1).T)
                ax.fill(M68, L68, color=colour, alpha=0.40, lw=0, zorder=1)
                Mc, Lc = M68, L68
            # Only the primary carries the annotation, so one event gives one
            # legend entry rather than two identical ones.
            if comp != "1":
                continue
            if inline_labels:
                ax.text(Mc.mean(), Lc.max(), label, color=colour,
                        fontsize=MR_LABEL_FS, fontweight='bold',
                        ha='center', va='bottom', zorder=3, clip_on=True)
            else:
                ax.plot(Mc, Lc, color=colour, lw=1.2, zorder=1, label=label)


if __name__ == '__main__':
    # ponytail: smallest check that the packaged contours are actually shipped
    # and parse -- catches a broken package-data config, which an editable
    # install would otherwise hide.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert DEFAULT_CONTOUR_DIR.is_dir(), f"missing packaged contours: {DEFAULT_CONTOUR_DIR}"
    _found = sorted(p.name for p in DEFAULT_CONTOUR_DIR.glob("*.csv"))
    for _k in MR_CONSTRAINTS:
        assert f"{_k}_95.csv" in _found, f"no 95% contour for {_k}"
    assert "mass_bounds.csv" in _found
    for _, _, _stem in TIDAL_CONSTRAINTS.values():
        for _c in ("1", "2"):
            assert f"{_stem}_{_c}_95.csv" in _found, f"no contour for {_stem}_{_c}"
    _fig, _ax = plt.subplots()
    add_observational_constraints(_ax)
    assert _ax.patches or _ax.lines, "nothing was drawn"
    plt.close(_fig)
    _fig, _ax = plt.subplots()
    add_tidal_constraints(_ax)
    # Two events x two components x two credible levels, Lambda never negative.
    assert len(_ax.patches) == 8, f"expected 8 tidal fills, got {len(_ax.patches)}"
    assert min(p.get_xy()[:, 1].min() for p in _ax.patches) >= 0.0
    plt.close(_fig)
    print(f"observational_constraints self-check ok ({len(_found)} CSVs)")

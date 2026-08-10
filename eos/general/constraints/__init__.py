"""Observational and experimental constraints, as one-call overlays.

An equation of state is judged in several different planes, and a constraint
belongs to exactly one of them: NICER and HESS give credible regions in
mass-radius, the BNS detections give them in mass-tidal-deformability, the
heavy-ion flow analyses give a pressure band against density, and chiral EFT
gives an energy band. They cannot be mixed, so the API is keyed by plane:

    from eos.general.constraints import overlay, list_available

    overlay(ax, "M-R")                     # NICER + HESS + the mass band
    overlay(ax, "M-Lambda")                # GW170817 and GW190425 components
    overlay(ax, "P-n")                     # FOPI and Danielewicz flow bands
    overlay(ax, "M-R", style="gradient")   # posterior density, not 68/95 rings

    list_available()                       # every plane and what is in it
    list_available("P-n")                  # just that plane

`overlay` draws at low zorder so model curves stay on top, and returns the
handles it made.

ADDING A CONSTRAINT is a data entry, not a new code path: drop the file in
``data/``, add one `Constraint(...)` to `REGISTRY` naming its plane and kind,
and it appears in `overlay` and `list_available` at once. The three kinds
below cover everything the repository has:

  contour_2d   a closed credible region, from <key>_68.csv / <key>_95.csv,
               each two columns of the plane's own coordinates. Optionally
               accompanied by <key>_density.npz for gradient rendering.
  band         a filled band against baryon density, from one CSV with
               columns rho_fm3, lower, upper.
  mass_band    a horizontal band at a measured mass, from mass_bounds.csv.

The CSVs are produced offline by :mod:`eos.general.constraints.build` from the
raw posterior samples, and ship inside the package, so plotting never rebuilds
a KDE and everything works from an installed wheel with no path argument.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eos.general.figure_style import STANDARD_COLORS

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Source labels sit below body text on purpose -- they annotate, not narrate.
LABEL_FS = 8.5

#: Fill opacity for the 95% and 68% credible regions. The 2-sigma region is
#: fainter so the nesting reads without needing an outline.
ALPHA_95 = 0.20
ALPHA_68 = 0.40

#: How a credible region may be drawn. See `overlay`.
STYLES = ("contours", "gradient", "gradient+contours")


@dataclass(frozen=True)
class Constraint:
    """One constraint: which plane it lives in and which files carry it."""

    key: str                     # file stem, or the row name in mass_bounds.csv
    plane: str                   # which axes it may be drawn on
    kind: str                    # contour_2d | band | mass_band
    label: str                   # what appears in the legend
    colour: str
    reference: str               # so a figure caption can be written from here
    components: tuple = ("",)    # contour_2d: sub-blobs sharing one legend entry
    anchor: tuple = None         # contour_2d: (dx, dy, ha, va) for inline label
    text_colour: str = None      # mass_band: darker tone for the annotation


# --------------------------------------------------------------------------
# The registry. One entry per constraint; the plane string is the only thing
# that decides where it may be drawn.
# --------------------------------------------------------------------------

REGISTRY = [
    # -- mass-radius: x = R [km], y = M [M_sun] ----------------------------
    Constraint("J0030", "M-R", "contour_2d", "PSR J0030+0451",
               STANDARD_COLORS['Orange'], "Miller et al. 2019, ApJL 887 L24",
               anchor=(0.0, -0.1, 'center', 'center')),
    Constraint("J0740", "M-R", "contour_2d", "PSR J0740+6620",
               STANDARD_COLORS['Blue'], "Miller et al. 2021, ApJL 918 L28",
               anchor=(0.0, 0.0, 'center', 'center')),
    Constraint("J0614", "M-R", "contour_2d", "PSR J0614-3329",
               STANDARD_COLORS['Green'], "Mauviard et al. 2025, ApJ",
               anchor=(-0.3, 0.0, 'center', 'center')),
    Constraint("HESS", "M-R", "contour_2d", "HESS J1731-347",
               STANDARD_COLORS['Magenta'],
               "Doroshenko et al. 2022, Nat. Astron. 6 1444",
               anchor=(0.0, -0.2, 'center', 'center')),
    Constraint("J0952-0607", "M-R", "mass_band", "PSR J0952-0607",
               STANDARD_COLORS['Yellow'], "Romani et al. 2022, ApJL 934 L17",
               text_colour=STANDARD_COLORS['Brown']),

    # -- mass-tidal deformability: x = M [M_sun], y = Lambda ---------------
    # Two blobs per event, one per star, sharing a colour and a single legend
    # entry: both components of a binary obey the same equation of state, so
    # one Lambda(M) curve has to thread through both.
    Constraint("GW170817ML", "M-Lambda", "contour_2d", "GW170817",
               STANDARD_COLORS['Blue'], "Abbott et al. 2019, PRX 9 031040",
               components=("1", "2")),
    Constraint("GW190425ML", "M-Lambda", "contour_2d", "GW190425",
               STANDARD_COLORS['Magenta'], "Abbott et al. 2020, ApJL 892 L3",
               components=("1", "2")),

    # -- chirp mass vs effective tidal deformability -----------------------
    # A property of the binary as a whole: it cannot be drawn against a single
    # M(R) sequence without first choosing a mass ratio, which is why it is a
    # plane of its own rather than part of "M-Lambda".
    Constraint("GW170817", "Mchirp-Lambdatilde", "contour_2d", "GW170817",
               STANDARD_COLORS['Blue'], "Abbott et al. 2019, PRX 9 031040"),
    Constraint("GW190425", "Mchirp-Lambdatilde", "contour_2d", "GW190425",
               STANDARD_COLORS['Magenta'], "Abbott et al. 2020, ApJL 892 L3"),

    # -- pressure of symmetric matter vs density ---------------------------
    Constraint("DLL_2002_PSM", "P-n", "band", "Danielewicz et al. 2002",
               STANDARD_COLORS['Gray'],
               "Danielewicz, Lacey & Lynch 2002, Science 298 1592"),
    Constraint("FOPI_2016_PSM", "P-n", "band", "FOPI (IQMD) 2016",
               STANDARD_COLORS['Green'],
               "Le Fevre et al. 2016, Nucl. Phys. A 945 112"),

    # -- energy per baryon vs density --------------------------------------
    Constraint("chiral_eft", "E-n", "band", "chiral EFT (PNM)",
               STANDARD_COLORS['Orange'],
               "Hebeler, Lattimer, Pethick & Schwenk 2013, ApJ 773 11"),
    Constraint("chiEFT_PNM_E", "E-n", "band", "chiral EFT PNM (nucleardatapy)",
               STANDARD_COLORS['Orange'], "Drischler et al., via nucleardatapy"),
    Constraint("chiEFT_SNM_E", "E-n", "band", "chiral EFT SNM (nucleardatapy)",
               STANDARD_COLORS['Blue'], "Drischler et al., via nucleardatapy"),
    Constraint("FOPI_2016_eSM", "E-n", "band", "FOPI (IQMD) 2016",
               STANDARD_COLORS['Green'],
               "Le Fevre et al. 2016, Nucl. Phys. A 945 112"),

    # -- symmetry energy vs density ----------------------------------------
    Constraint("ASYEOS_2016_Esym", "Esym-n", "band", "ASY-EOS 2016",
               STANDARD_COLORS['Magenta'],
               "Russotto et al. 2016, PRC 94 034608"),
]

#: What each plane's axes must be, so a caller can check they match.
PLANES = {
    "M-R": ("R [km]", "M [M_sun]"),
    "M-Lambda": ("M [M_sun]", "Lambda"),
    "Mchirp-Lambdatilde": ("M_chirp [M_sun]", "Lambda-tilde"),
    "P-n": ("n_B [fm^-3]", "P [MeV/fm^3]"),
    "E-n": ("n_B [fm^-3]", "E/A [MeV]"),
    "Esym-n": ("n_B [fm^-3]", "E_sym [MeV]"),
}


class MissingConstraintData(FileNotFoundError):
    """Raised when a constraint's files are absent, saying how to get them."""


def _require(path):
    """Read a data file, or explain how to produce it."""
    if not path.exists():
        raise MissingConstraintData(
            f"{path.name} is missing from {DATA_DIR}.\n"
            f"These CSVs ship with the package; if you are working from a "
            f"source tree they are rebuilt with\n"
            f"    python plot/fetch_samples.py          # raw posteriors, ~160 MB\n"
            f"    python -m eos.general.constraints.build\n"
            f"The raw samples are not tracked in git -- see "
            f"plot/data/samples/SOURCES.md for what each one is.")
    return path


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


def available(plane=None):
    """The constraints whose data is actually present, optionally one plane."""
    found = []
    for c in REGISTRY:
        if plane is not None and c.plane != plane:
            continue
        if _has_data(c):
            found.append(c)
    return found


def _has_data(c):
    if c.kind == "band":
        return (DATA_DIR / f"{c.key}.csv").exists()
    if c.kind == "mass_band":
        return (DATA_DIR / "mass_bounds.csv").exists()
    stem = c.key if c.components == ("",) else f"{c.key}_{c.components[0]}"
    return (DATA_DIR / f"{stem}_95.csv").exists()


def list_available(plane=None):
    """Print what can be overlaid, by plane. Returns the Constraint list."""
    found = available(plane)
    planes = [plane] if plane else list(PLANES)
    for p in planes:
        xlabel, ylabel = PLANES[p]
        here = [c for c in found if c.plane == p]
        print(f"{p}   ({xlabel} vs {ylabel})")
        if not here:
            print("    (no data present)")
        for c in here:
            gradient = " +density" if _density_path(c).exists() else ""
            print(f"    {c.key:18s} {c.kind:10s}{gradient:9s} "
                  f"{c.label}  [{c.reference}]")
    return found


def _density_path(c):
    return DATA_DIR / f"{c.key}_density.npz"


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _label_region(ax, c, xc, yc, zorder):
    """Write the source name next to its region, at the registered anchor."""
    if c.anchor is not None:
        dx, dy, ha, va = c.anchor
        ax.text(xc.mean() + dx, yc.mean() + dy, c.label, color=c.colour,
                fontsize=LABEL_FS, fontweight='bold', ha=ha, va=va,
                zorder=zorder + 3, clip_on=True)
    else:
        ax.text(xc.mean(), yc.max(), c.label, color=c.colour,
                fontsize=LABEL_FS, fontweight='bold', ha='center', va='bottom',
                zorder=zorder + 3, clip_on=True)


def _draw_gradient(ax, c, inline_labels, zorder):
    """Shade the posterior density continuously instead of drawing 68/95 rings.

    The source's own colour at constant hue, with the density carried entirely
    by the alpha channel, so several overlapping posteriors stay individually
    readable instead of blending into a third colour.

    Drawn with `imshow` rather than `pcolormesh`: a mesh emits one quad per
    cell, and with a partly transparent colour the antialiased seams between
    them show up as a fine grid over the blob. `imshow` composites a single
    image and interpolates it, which is also what the field deserves -- the
    KDE smoothing is far wider than a grid cell, so the 120x120 storage is
    already finer than the information in it.
    """
    from matplotlib.colors import to_rgb

    grid = np.load(_density_path(c))
    x, y, z = grid["x"], grid["y"], grid["z"]
    rgba = np.zeros(z.shape + (4,), dtype=float)
    rgba[..., :3] = to_rgb(c.colour)
    rgba[..., 3] = np.clip(z, 0.0, 1.0) * 0.85
    handle = ax.imshow(rgba, origin="lower", aspect="auto",
                       extent=(x[0], x[-1], y[0], y[-1]),
                       interpolation="bilinear", zorder=zorder,
                       rasterized=True)
    if inline_labels:
        # Intensity-weighted centroid, so the label lands where it does in the
        # contour style: the mean of a contour path and the centre of mass of
        # the density agree for these unimodal posteriors, while the density
        # PEAK can sit noticeably off-centre in a skewed one.
        weight = z / z.sum()
        x_c = float((weight.sum(axis=0) * x).sum())
        y_c = float((weight.sum(axis=1) * y).sum())
        _label_region(ax, c, np.array([x_c]), np.array([y_c]), zorder)
    return [handle]


def _legend_proxy(ax, c):
    """A legend swatch that draws nothing on the axes.

    The credible regions are soft fills with no border. Giving one of them an
    outline purely so the legend has something to point at looked like a
    deliberate distinction that isn't there -- most visibly for the GW events,
    where the primary component came out ringed and the secondary bare. A fill
    at NaN coordinates carries the label and the colour into the legend while
    leaving every blob drawn the same way.
    """
    return ax.fill([np.nan], [np.nan], color=c.colour, alpha=ALPHA_68,
                   lw=0, label=c.label)


def _draw_levels(ax, c, zorder, lw=0.9, alpha=0.9):
    """The 68% and 95% boundaries as bare lines, with nothing filled."""
    handles = []
    for component in c.components:
        stem = c.key if component == "" else f"{c.key}_{component}"
        for level in ("68", "95"):
            path = DATA_DIR / f"{stem}_{level}.csv"
            if not path.exists():
                continue
            x, y = _smooth_closed(*np.loadtxt(path, delimiter=",",
                                              skiprows=1).T)
            handles.append(ax.plot(np.r_[x, x[:1]], np.r_[y, y[:1]],
                                   color=c.colour, lw=lw, alpha=alpha,
                                   zorder=zorder + 2)[0])
    return handles


def _draw_contour_2d(ax, c, style, inline_labels, zorder):
    """Credible regions: nested fills, a density shade, or both."""
    handles = []
    if style.startswith("gradient") and _density_path(c).exists():
        handles += _draw_gradient(ax, c, inline_labels, zorder)
        if style == "gradient+contours":
            handles += _draw_levels(ax, c, zorder)
        if not inline_labels:
            handles += _legend_proxy(ax, c)
        return handles

    for component in c.components:
        stem = c.key if component == "" else f"{c.key}_{component}"
        path_95 = DATA_DIR / f"{stem}_95.csv"
        path_68 = DATA_DIR / f"{stem}_68.csv"
        if not path_95.exists():
            continue
        x95, y95 = _smooth_closed(*np.loadtxt(path_95, delimiter=",",
                                              skiprows=1).T)
        handles.append(ax.fill(x95, y95, color=c.colour, alpha=ALPHA_95,
                               lw=0, zorder=zorder))
        xc, yc = x95, y95
        if path_68.exists():
            x68, y68 = _smooth_closed(*np.loadtxt(path_68, delimiter=",",
                                                  skiprows=1).T)
            handles.append(ax.fill(x68, y68, color=c.colour, alpha=ALPHA_68,
                                   lw=0, zorder=zorder + 1))
            xc, yc = x68, y68

        # Only the first component is annotated, so one event gives one label
        # rather than two identical ones.
        if component == c.components[0] and inline_labels:
            _label_region(ax, c, xc, yc, zorder)

    if not inline_labels and handles:
        handles += _legend_proxy(ax, c)
    return handles


def _draw_band(ax, c, inline_labels, zorder):
    """A filled band against baryon density (flow constraints, chiral EFT).

    The two stored curves are the band's EDGES, not a sorted (min, max) pair.
    They are allowed to cross: the ASY-EOS symmetry-energy constraint bounds
    the SLOPE of E_sym, so its band is pinned where E_sym is already known,
    near saturation, and fans out on both sides -- the soft edge runs above the
    stiff one below n_0 and below it above n_0. `fill_between` fills between
    two curves in whichever order they come, so this needs no special case,
    but sorting them would be wrong.
    """
    data = np.loadtxt(_require(DATA_DIR / f"{c.key}.csv"), delimiter=",",
                      skiprows=1)
    n_B, edge_a, edge_b = data[:, 0], data[:, 1], data[:, 2]
    label = None if inline_labels else c.label
    handle = ax.fill_between(n_B, edge_a, edge_b, color=c.colour,
                             alpha=ALPHA_95, lw=0, zorder=zorder, label=label)
    ax.plot(n_B, edge_a, color=c.colour, lw=0.8, alpha=0.7, zorder=zorder)
    ax.plot(n_B, edge_b, color=c.colour, lw=0.8, alpha=0.7, zorder=zorder)
    if inline_labels:
        mid = len(n_B) // 2
        ax.text(n_B[mid], max(edge_a[mid], edge_b[mid]), c.label, color=c.colour,
                fontsize=LABEL_FS, fontweight='bold', ha='center', va='bottom',
                zorder=zorder + 3, clip_on=True)
    return [handle]


def _draw_mass_band(ax, c, zorder):
    """A horizontal band at a mass measured without a radius."""
    rows = list(csv.DictReader(open(_require(DATA_DIR / "mass_bounds.csv"))))
    by_level = {r["level"]: r for r in rows if r["name"] == c.key}
    if not by_level:
        return []
    handles = []
    for level, alpha in (("95", 0.10), ("68", 0.18)):
        row = by_level[level]
        handles.append(ax.axhspan(float(row["lower"]), float(row["upper"]),
                                  color=c.colour, alpha=alpha, lw=0,
                                  zorder=zorder))
    # x in axes fraction (blended transform) so the label is pinned to the left
    # edge and never escapes the panel when xlim changes.
    ax.text(0.02, float(by_level["68"]["upper"]), " " + c.label,
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=LABEL_FS, fontweight='bold',
            color=c.text_colour or c.colour, zorder=zorder + 3, clip_on=True)
    return handles


def overlay(ax, plane, *, only=None, style="contours", inline_labels=False,
            show_mass_bands=True, zorder=0):
    """Draw every available constraint of one plane onto `ax`.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes whose coordinates are those of `plane` -- see `PLANES`.
    plane : str
        One of the keys of `PLANES`: "M-R", "M-Lambda", "Mchirp-Lambdatilde",
        "P-n", "E-n", "Esym-n".
    only : sequence of str, optional
        Restrict to these constraint keys.
    style : {"contours", "gradient", "gradient+contours"}
        "contours" nests the 95% and 68% credible regions as soft fills.
        "gradient" shades the posterior density continuously.
        "gradient+contours" shades the density AND draws the 68% and 95%
        boundaries as bare lines over it, so the credible levels stay readable
        on a continuous field.
        Both gradient styles need a density grid; a source without one falls
        back to nested fills, since a level set cannot be interpolated back
        into a density.
    inline_labels : bool
        True writes the source name next to its region and adds no legend
        entry; False draws a labelled boundary line for the legend.
    show_mass_bands : bool
        Include mass-only measurements (M-R plane only).
    zorder : int
        Base zorder. Everything is drawn at or just above it, well below the
        model curves.

    Returns
    -------
    list of the artists created.
    """
    if plane not in PLANES:
        raise KeyError(f"unknown plane {plane!r}; expected one of "
                       f"{sorted(PLANES)}")
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")

    handles = []
    for c in available(plane):
        if only is not None and c.key not in only:
            continue
        if c.kind == "mass_band":
            if show_mass_bands:
                handles += _draw_mass_band(ax, c, zorder)
        elif c.kind == "band":
            handles += _draw_band(ax, c, inline_labels, zorder)
        else:
            handles += _draw_contour_2d(ax, c, style, inline_labels, zorder)
    return handles

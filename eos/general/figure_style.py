"""
figure_style.py
===============
The single home for publication figure styling shared by every project that
builds on ``eos`` (nucleation, DD2, mixed-phase, ...).

Two style families live here, and they are NOT interchangeable:

  * **Paper style** -- ``set_paper_style`` / ``paper_grid`` / ``panel_label``.
    Built for a PRD / REVTeX two-column article: 10 pt text, 300 dpi, inward
    ticks, no grid, and figures constructed at the exact page width they will
    occupy so no LaTeX rescaling shrinks the fonts. Use this for anything going
    into a manuscript.

  * **Notebook style** -- ``set_global_style`` / ``setup_scientific_figure`` /
    ``add_panel_labels`` / ``apply_style``. Bigger text (12-14 pt), 150 dpi,
    gridlines on, arbitrary n x m panel grids. Use this for exploratory work
    where readability on screen beats page fidelity.

Calling one after the other overwrites the other's rcParams -- pick one per
figure and stick to it.

Usage (paper)::

    from eos.general.figure_style import paper_grid, panel_label
    fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double',
                                               placeholder=False)
    panel_label(axA, '(a)')

Usage (notebook)::

    from eos.general.figure_style import set_global_style, setup_scientific_figure
    set_global_style()                       # once at notebook start
    fig, axes = setup_scientific_figure(nrows=2, ncols=2)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# =============================================================================
# PALETTES
# =============================================================================
# Mathematica-style StandardColor scheme: muted tones that survive screen AND
# print. This is THE palette -- do not re-declare these RGB triples elsewhere.
STANDARD_COLORS = {
    'Red':     (0.8,  0.25, 0.33),
    'Green':   (0.24, 0.6,  0.44),
    'Blue':    (0.24, 0.6,  0.8),
    'Gray':    (0.4,  0.4,  0.4),
    'Cyan':    (0.1,  0.6,  0.6),
    'Magenta': (0.6,  0.24, 0.6),
    'Yellow':  (0.95, 0.75, 0.1),
    'Brown':   (0.6,  0.4,  0.2),
    'Orange':  (0.9,  0.4,  0.0),
    'Pink':    (0.9,  0.4,  0.6),
    'Purple':  (0.5,  0.35, 0.65),
}

# Okabe-Ito colourblind-safe set, for CATEGORICAL groups (a few discrete lines).
# Convention: categorical -> OKAB; continuous/sequential field -> viridis;
# diverging (signed difference) -> RdBu_r.
OKAB = dict(orange='#E69F00', sky='#56B4E9', green='#009E73', blue='#0072B2',
            vermillion='#D55E00', purple='#CC79A7', grey='#9a9a9a')
# Ordered picks for "colour = category" line groups: OKAB_CAT[:n] gives n
# well-separated hues (blue -> orange -> green -> vermillion -> purple -> sky).
# No yellow (poor contrast on white) and no black/grey (reserved for reference
# lines), so a legend of up to six categories stays unambiguous.
OKAB_CAT = ('#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9')

# Quark-phase line weights and opacity -- ONE source of truth so every figure
# that overlays the three droplet phases reads the same way: unpCFL is the
# emphasised composite (thick, opaque, drawn on top); CFL and unpaired are its
# two limiting cases (thinner, equal weight, one shared fainter alpha).
PHASE_LW = {'unpCFL': 2.1, 'cfl': 1.4, 'unpaired': 1.4}
PHASE_ALPHA = {'unpCFL': 1.0, 'cfl': 0.6, 'unpaired': 0.6}

# Temperature -> colour, for T-labelled curve families.
T_COLORS = {
    0.1: STANDARD_COLORS['Gray'],
    10.0: STANDARD_COLORS['Purple'],
    50.0: STANDARD_COLORS['Orange'],
    100.0: STANDARD_COLORS['Red'],
}

# Generic ordered sequence when the category has no natural colour.
COLORS_SEQ = [
    STANDARD_COLORS['Gray'],
    STANDARD_COLORS['Purple'],
    STANDARD_COLORS['Orange'],
    STANDARD_COLORS['Red'],
    STANDARD_COLORS['Blue'],
    STANDARD_COLORS['Green'],
    STANDARD_COLORS['Cyan'],
]

# Baryon species: colour by particle, linestyle by multiplet (nucleons solid,
# hyperons dashed, Deltas dash-dot, quarks densely dashed, leptons dotted) so a
# composition plot is readable in b/w too. Quarks and Deltas are here because a
# hybrid hadron-quark figure has to draw all of them on one axis; keeping them
# in this one mapping is what stops two figures colouring the same species
# differently. Electrons answer to both 'e-' and 'e' — eos.dd2 and eos.mixed
# key their composition rows with the bare 'e'.
PARTICLE_COLORS = {
    'p': STANDARD_COLORS['Blue'],
    'n': STANDARD_COLORS['Orange'],
    'Lambda': STANDARD_COLORS['Green'],
    'Sigma+': STANDARD_COLORS['Red'],
    'Sigma0': STANDARD_COLORS['Purple'],
    'Sigma-': STANDARD_COLORS['Brown'],
    'Xi0': STANDARD_COLORS['Cyan'],
    'Xi-': STANDARD_COLORS['Yellow'],
    'Delta++': STANDARD_COLORS['Pink'],
    'Delta+': STANDARD_COLORS['Magenta'],
    'Delta0': STANDARD_COLORS['Purple'],
    'Delta-': STANDARD_COLORS['Red'],
    'u': STANDARD_COLORS['Blue'],
    'd': STANDARD_COLORS['Orange'],
    's': STANDARD_COLORS['Green'],
    'e-': STANDARD_COLORS['Gray'],
    'e': STANDARD_COLORS['Gray'],
    'mu-': STANDARD_COLORS['Brown'],
}
PARTICLE_STYLES = {
    'p': '-', 'n': '-',
    'Lambda': '--', 'Sigma+': '--', 'Sigma0': '--', 'Sigma-': '--',
    'Xi0': '--', 'Xi-': '--',
    'Delta++': '-.', 'Delta+': '-.', 'Delta0': '-.', 'Delta-': '-.',
    'u': (0, (4, 1)), 'd': (0, (4, 1)), 's': (0, (4, 1)),
    'e-': ':', 'e': ':', 'mu-': ':',
}


def particle_style(name, default_color=None):
    """(colour, linestyle) for a species name, with a graceful fallback.

    A composition figure should never crash or silently drop a curve because a
    species is not in the table — an unknown name comes back grey and solid.
    """
    return (PARTICLE_COLORS.get(name, default_color or STANDARD_COLORS['Gray']),
            PARTICLE_STYLES.get(name, '-'))

# =============================================================================
# AXIS LABEL TEMPLATES
# =============================================================================
# LaTeX-ready axis names with units, so the same quantity is never labelled two
# different ways across figures.
LABELS = {
    # Density
    'nB': r'$n_B$ [fm$^{-3}$]',
    'nB_n0': r'$n_B/n_0$',
    'n0': 0.16,                       # saturation density VALUE, not a label
    # Temperature
    'T': r'$T$ [MeV]',
    # Thermodynamics
    'P': r'$P$ [MeV fm$^{-3}$]',
    'epsilon': r'$\varepsilon$ [MeV fm$^{-3}$]',
    's': r'$s$ [fm$^{-3}$]',
    's_nB': r'$s/n_B$',
    'S': r'$S = s/n_B$',
    # Chemical potentials
    'mu_B': r'$\mu_B$ [MeV]',
    'mu_C': r'$\mu_C$ [MeV]',
    'mu_Q': r'$\mu_Q$ [MeV]',
    'mu_S': r'$\mu_S$ [MeV]',
    # Meson mean fields
    'sigma': r'$\sigma$ [MeV]',
    'omega': r'$\omega$ [MeV]',
    'rho': r'$\rho$ [MeV]',
    'phi': r'$\phi$ [MeV]',
    # Fractions
    'Y_C': r'$Y_C$',
    'Y_i': r'$Y_i$',
}

# =============================================================================
# PAPER STYLE -- PRD / REVTeX two-column
# =============================================================================
# PRD / REVTeX geometry [inches]. A figure MUST be built at the width it will
# occupy on the page and included at 1:1 (no \includegraphics width= rescaling,
# which would shrink the fonts). Then the pt sizes below land as-is on paper.
# Single column = \columnwidth, full width = \textwidth span.
PRD_COL_W = 3.375
PRD_FULL_W = 7.0
# Height cap for a full-width four-panel (2x2) figure: keep it within ~half the
# PRD text height (~9 in) so two figures fit per page. Panels go landscape at
# this height; fonts stay 10 pt (PRD body). 1x2 figures use half of this.
PRD_2X2_H = 4.5
# Mid-size square figure: wider than one column but kept under half the PRD text
# height (~9 in) so two stack on a page. Used by paper_grid(mode='centered').
PRD_CENTERED_W = 4.75


def set_paper_style(fontsize=10, labelsize=None, legendsize=None, rc=None):
    """Publication rcParams: CMU Serif + Computer-Modern math (matches a LaTeX
    two-column layout), inward ticks on all four sides, frame-less legends,
    300 dpi tight saves. All text defaults to 10 pt = PRD body-text size, so
    labels, ticks and legends match the surrounding paper when the figure is
    placed at its true width (see PRD_COL_W / PRD_FULL_W).

      fontsize  : base text size in pt (title + anything not overridden below).
                  Keep 10 for a figure placed at true column width; only raise it
                  if the figure will be down-scaled in \\includegraphics.
      labelsize : size of the axis names AND the x/y tick numbers. Defaults to
                  `fontsize`; set separately to shrink/grow axis text alone.
      legendsize: size of the legend text. Defaults to `fontsize`; set separately
                  (journals often run legends a touch smaller than axis labels).
      rc        : optional dict of extra rcParams to override on top, e.g.
                  rc={'axes.linewidth': 0.5}. Escape hatch for one-off tweaks.
    """
    labelsize = fontsize if labelsize is None else labelsize
    legendsize = fontsize if legendsize is None else legendsize
    base = {
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',                 # CM math to match CMU Serif
        'font.size': fontsize, 'axes.titlesize': fontsize,
        'axes.labelsize': labelsize,              # axis names
        'xtick.labelsize': labelsize, 'ytick.labelsize': labelsize,  # tick numbers
        'legend.fontsize': legendsize, 'legend.frameon': False,
        # Shorter legend handle lines + tighter text gap so keys read compact.
        'legend.handlelength': 1.4, 'legend.handletextpad': 0.5,
        # Thin box + ticks: spines 0.7, ticks 0.6/0.5 (lighter than mpl default).
        'lines.linewidth': 1.8, 'axes.linewidth': 0.7,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.unicode_minus': False,              # CMU Serif lacks U+2212 minus
    }
    if rc:
        base.update(rc)
    mpl.rcParams.update(base)


def paper_grid(layout='2x2', mode='single', placeholder=True, square=True,
               aspect=1.2, fontsize=10, labelsize=None, legendsize=None, rc=None):
    """Build a PRD panel grid with the publication style applied.

    Physics: each panel is forced to a fixed box aspect (W/H = `aspect`) so a
    plane you must read geometrically -- an M-R plane, a phase diagram -- isn't
    silently stretched. Line-vs-parameter panels read better landscape (aspect
    ~1.2, the default); plane plots want `aspect=1.0` (square). The figure is
    built at the true page width it will occupy (no LaTeX rescaling), so the
    `fontsize`-pt fonts from set_paper_style() land as PRD body text.

    Mechanics:
      layout   : '2x2' (four panels) or '1x2' (a two-panel row = the 2x2 top row).
      mode     : sets the figure width W -
                   'single'   -> W = PRD_COL_W      (3.375", one column)
                   'centered' -> W = PRD_CENTERED_W (4.75",  mid-size)
                   'double'   -> W = PRD_FULL_W     (7.0",   two columns)
                 The height is derived from `aspect` so panels come out at that
                 W/H with minimal slack: 2x2 is (W, W/aspect); 1x2 is
                 (W, (W/2)/aspect) so its panels match a 2x2 top row.
      aspect   : panel width/height. 1.2 (default) = mild landscape; 1.0 = square.
      fontsize : base text size in pt (default 10 = PRD body; title + fallback).
      labelsize: axis-name + tick-number size (default = fontsize).
      legendsize: legend text size (default = fontsize).
      rc       : optional dict of extra rcParams (e.g. line/axes widths),
                 forwarded to set_paper_style for per-figure tweaks.
                 All four are forwarded to set_paper_style.
      square   : True (default) pins every panel to the exact `aspect` box --
                 because the figsize is fixed the slack dimension leaves a
                 centring margin (harmless: savefig(bbox_inches='tight') crops
                 it). False drops the box constraint so panels stretch to fill
                 the figure (zero centring whitespace) -- use it when the inline
                 gap bothers you and near-`aspect` is fine.

    Panels do NOT share axes: every one carries its own x/y axis. With
    placeholder=True (default) each panel gets dummy x/y labels and a title so
    the bare layout reads as a template; pass placeholder=False for a real
    figure that sets its own labels (and tags panels with panel_label instead of
    a title). Returns (fig, axes) with axes a 2-D ndarray; unpack as
    ((axA, axB), (axC, axD)) or (axA, axB).
    """
    widths = {'single': PRD_COL_W, 'centered': PRD_CENTERED_W, 'double': PRD_FULL_W}
    if mode not in widths:
        raise ValueError(f"mode must be one of {sorted(widths)}, got {mode!r}")
    if layout not in ('2x2', '1x2'):
        raise ValueError(f"layout must be '2x2' or '1x2', got {layout!r}")
    if aspect <= 0:
        raise ValueError(f"aspect must be > 0, got {aspect!r}")

    set_paper_style(fontsize=fontsize, labelsize=labelsize,
                    legendsize=legendsize, rc=rc)
    w = widths[mode]
    nrows = 2 if layout == '2x2' else 1
    # Height follows the panel aspect: one panel is w/2 wide, so w/2/aspect tall;
    # 2x2 stacks two of those (~ w/aspect), 1x2 is a single row.
    fig_h = w / aspect if layout == '2x2' else (w / 2) / aspect
    figsize = (w, fig_h)

    # squeeze=False -> axes always 2-D, so callers unpack uniformly.
    # sharex/sharey default False: independent axes per panel.
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, layout='constrained',
                             squeeze=False)
    # Tighten the constrained-layout padding so panels sit closer together and to
    # the figure edge (less whitespace); edit here to rescale for all figures.
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    for ax in axes.flat:
        if square:
            ax.set_box_aspect(1 / aspect)    # fixed W/H box, independent of labels
        if placeholder:
            # Dummy labels/title so the empty template looks complete - real
            # figures pass placeholder=False and set their own.
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_title('panel')
    return fig, axes


def panel_label(ax, lab, corner='upper left'):
    """Put an (a)/(b)/... tag in ONE panel corner. `corner` is a vertical word
    ('upper'/'lower') optionally followed by a horizontal one ('left'/'right');
    horizontal defaults to 'left'. e.g. 'upper', 'lower right', 'upper right'.

    See add_panel_labels() for the notebook-style variant that tags a whole
    axes array with auto-generated letters.
    """
    words = corner.split()
    y, va = (0.945, 'top') if words[0] == 'upper' else (0.055, 'bottom')
    right = len(words) > 1 and words[1] == 'right'
    x, ha = (0.955, 'right') if right else (0.045, 'left')
    ax.text(x, y, lab, transform=ax.transAxes,
            fontweight='bold', va=va, ha=ha)


# =============================================================================
# NOTEBOOK STYLE -- on-screen exploratory figures
# =============================================================================
# Bigger text and gridlines: readable in a notebook cell, NOT page-faithful.
FONTS = {
    'family': 'serif',
    'serif': ['CMU Serif'],       # Computer Modern Unicode - matches LaTeX
    'mathtext': 'cm',             # Computer Modern for math
    'title': 14,
    'label': 12,
    'tick': 11,
    'legend': 10,
    'annotation': 10,
    'panel_label': 14,            # For (a), (b), (c), (d) labels
}

STYLE = {
    # Figure sizes (width, height) in inches, chosen by panel-grid shape.
    'figsize_square': (6, 6),           # Single square plot
    'figsize_2x2': (8, 8),              # 2x2 grid, almost square
    'figsize_1x2': (10, 4.5),           # 1 row, 2 columns
    'figsize_2x1': (5, 9),              # 2 rows, 1 column
    'figsize_1x4': (14, 3.5),           # 1 row, 4 columns
    'figsize_wide': (12, 5),            # Wide format
    'figsize_default': (8, 6),          # Default

    # Resolution and line properties
    'dpi': 150,
    'linewidth': 1.8,
    'linewidth_thin': 1.2,
    'markersize': 5,

    # Grid properties
    'grid_alpha': 0.5,
    'grid_style': '-',                  # Solid gridlines
    'grid_linewidth': 0.5,

    # Minor ticks and grid
    'minor_ticks': True,
    'minor_grid': True,
    'minor_grid_alpha': 0.25,           # Lighter than major grid
    'minor_grid_linewidth': 0.3,
    'n_minor_ticks': 4,                 # Minor ticks between major ticks
}


def set_global_style():
    """Notebook rcParams: CMU Serif at 12-14 pt, 150 dpi, ASCII minus.

    This is the ON-SCREEN style. For a manuscript figure use set_paper_style()
    instead -- it is 10 pt at 300 dpi with no grid, sized to the PRD page.
    Calling both in one session means the last one wins.
    """
    plt.rcParams['font.family'] = FONTS['family']
    plt.rcParams['font.serif'] = FONTS['serif']

    # Math text - Computer Modern to match serif
    plt.rcParams['mathtext.fontset'] = FONTS['mathtext']
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['mathtext.it'] = 'serif:italic'
    plt.rcParams['mathtext.bf'] = 'serif:bold'

    # Font sizes
    plt.rcParams['font.size'] = FONTS['label']
    plt.rcParams['axes.labelsize'] = FONTS['label']
    plt.rcParams['axes.titlesize'] = FONTS['title']
    plt.rcParams['xtick.labelsize'] = FONTS['tick']
    plt.rcParams['ytick.labelsize'] = FONTS['tick']
    plt.rcParams['legend.fontsize'] = FONTS['legend']

    # ASCII minus: CMU Serif has no U+2212 glyph, so the Unicode minus renders
    # as a missing-glyph box on every negative tick label.
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.formatter.use_mathtext'] = True

    # Figure defaults
    plt.rcParams['figure.dpi'] = STYLE['dpi']
    plt.rcParams['savefig.dpi'] = STYLE['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'


def setup_scientific_figure(nrows=1, ncols=1, figsize=None,
                            gray_background=False, sharex=False, sharey=False):
    """Create an arbitrary nrows x ncols figure, sizing it from the grid shape.

    The notebook-style counterpart of paper_grid(): any panel count, size looked
    up from STYLE, no forced box aspect and no page-width constraint. Returns
    (fig, axes) with axes shaped exactly as plt.subplots would give it.
    """
    if figsize is None:
        if nrows == 2 and ncols == 2:
            figsize = STYLE['figsize_2x2']
        elif nrows == 1 and ncols == 2:
            figsize = STYLE['figsize_1x2']
        elif nrows == 2 and ncols == 1:
            figsize = STYLE['figsize_2x1']
        elif nrows == 1 and ncols == 4:
            figsize = STYLE['figsize_1x4']
        elif nrows == 1 and ncols == 1:
            figsize = STYLE['figsize_square']
        else:
            figsize = STYLE['figsize_default']

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=sharex, sharey=sharey, dpi=STYLE['dpi'])

    if gray_background:
        facecolor = '#f0f0f0'
        if nrows * ncols == 1:
            axes.set_facecolor(facecolor)
        else:
            for ax in np.atleast_1d(axes).flat:
                ax.set_facecolor(facecolor)

    return fig, axes


def add_panel_labels(axes, labels=None, loc='upper left', fontsize=None):
    """Tag a whole axes ARRAY with (a), (b), (c), ... in one call.

    See panel_label() for the paper-style single-axis variant with an explicit
    label string.
    """
    if fontsize is None:
        fontsize = FONTS['panel_label']

    axes_flat = np.atleast_1d(axes).flat
    n_panels = len(list(axes_flat))

    if labels is None:
        labels = [f'({chr(ord("a") + i)})' for i in range(n_panels)]

    pos_map = {
        'upper left': (0.05, 0.95),
        'upper right': (0.95, 0.95),
        'lower left': (0.05, 0.05),
        'lower right': (0.95, 0.05),
    }
    x, y = pos_map.get(loc, (0.05, 0.95))
    ha = 'left' if 'left' in loc else 'right'
    va = 'top' if 'upper' in loc else 'bottom'

    for ax, label in zip(np.atleast_1d(axes).flat, labels):
        ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
                fontweight='bold', ha=ha, va=va)


def apply_style(ax, grid=True, legend=True, minor_ticks=None, minor_grid=None):
    """Apply notebook-style per-axis dressing: ticks, major/minor grid, legend.

    Only for the notebook family -- set_paper_style() already handles ticks via
    rcParams and deliberately leaves the grid off for manuscript figures.

    Parameters:
        ax: matplotlib axis
        grid: show major grid (default True)
        legend: show legend if handles exist (default True)
        minor_ticks: show minor ticks (default from STYLE['minor_ticks'])
        minor_grid: show minor grid (default from STYLE['minor_grid'])
    """
    ax.tick_params(labelsize=FONTS['tick'])

    if grid:
        ax.grid(True, which='major', alpha=STYLE['grid_alpha'],
                linestyle=STYLE['grid_style'],
                linewidth=STYLE['grid_linewidth'])

    if minor_ticks is None:
        minor_ticks = STYLE['minor_ticks']
    if minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(STYLE['n_minor_ticks']))
        ax.yaxis.set_minor_locator(AutoMinorLocator(STYLE['n_minor_ticks']))
        ax.tick_params(which='minor', length=3, width=0.5)

    if minor_grid is None:
        minor_grid = STYLE['minor_grid']
    if minor_grid and minor_ticks:
        ax.grid(True, which='minor', alpha=STYLE['minor_grid_alpha'],
                linestyle=STYLE['grid_style'],
                linewidth=STYLE['minor_grid_linewidth'])

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=FONTS['legend'], frameon=False)


# =============================================================================
# SHARED HELPERS
# =============================================================================
def get_T_color(T):
    """Colour for a temperature [MeV]; nearest key in T_COLORS if not exact."""
    if T in T_COLORS:
        return T_COLORS[T]
    temps = list(T_COLORS.keys())
    closest = min(temps, key=lambda t: abs(t - T))
    return T_COLORS[closest]


def save_figure(fig, filename, formats=('png', 'pdf')):
    """Save one figure under several extensions. `filename` carries NO suffix."""
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', dpi=STYLE['dpi'],
                    bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}.{{{', '.join(formats)}}}")


if __name__ == '__main__':
    # ponytail: smallest check that the paper_grid sizing/squareness contract
    # holds -- figure width equals the declared page width, panel box aspect
    # equals 1/aspect, and the three font sizes land where asked.
    mpl.use('Agg')
    _W = {'single': PRD_COL_W, 'centered': PRD_CENTERED_W, 'double': PRD_FULL_W}
    for _asp in (1.0, 1.2):
        for _lay in ('2x2', '1x2'):
            for _m, _w in _W.items():
                _fig, _ax = paper_grid(_lay, _m, aspect=_asp, fontsize=11,
                                       labelsize=9, legendsize=8)
                _fw, _fh = _fig.get_size_inches()
                _exp_h = _w / _asp if _lay == '2x2' else (_w / 2) / _asp
                assert abs(_fw - _w) < 1e-9
                assert abs(_fh - _exp_h) < 1e-9
                assert all(abs(a.get_box_aspect() - 1 / _asp) < 1e-9 for a in _ax.flat)
                assert abs(mpl.rcParams['font.size'] - 11) < 1e-9
                assert abs(mpl.rcParams['axes.labelsize'] - 9) < 1e-9
                assert abs(mpl.rcParams['legend.fontsize'] - 8) < 1e-9
                plt.close(_fig)
    # The two palettes must not have drifted apart into near-duplicates.
    assert len(set(map(tuple, STANDARD_COLORS.values()))) == len(STANDARD_COLORS)
    assert set(PHASE_LW) == set(PHASE_ALPHA)
    print('figure_style self-check ok')

"""
plotting_info.py  --  DEPRECATED, kept as a compatibility shim.

Everything that used to live here now lives in
:mod:`eos.general.figure_style`, which merges it with the PRD/REVTeX paper
style (``set_paper_style`` / ``paper_grid`` / ``panel_label``) so there is ONE
home for figure styling across every project built on ``eos``.

Migrate::

    from eos.general.plotting_info import set_global_style, STANDARD_COLORS
    ->
    from eos.general.figure_style import set_global_style, STANDARD_COLORS

Nothing has changed behaviourally: ``set_global_style`` is the same 12-14 pt /
150 dpi / gridlines-on notebook style it always was. The paper style is a
SEPARATE function -- see the module docstring of ``figure_style``.

Note: this module never exported a name ``COLORS``. The palette dict is
``STANDARD_COLORS`` and the ordered list is ``COLORS_SEQ``.
"""
import warnings

from eos.general.figure_style import (  # noqa: F401  (re-export surface)
    FONTS, STYLE,
    STANDARD_COLORS, T_COLORS, COLORS_SEQ,
    PARTICLE_COLORS, PARTICLE_STYLES, LABELS,
    set_global_style, setup_scientific_figure, add_panel_labels, apply_style,
    get_T_color, save_figure,
)

warnings.warn(
    "eos.general.plotting_info is deprecated; import from "
    "eos.general.figure_style instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FONTS", "STYLE",
    "STANDARD_COLORS", "T_COLORS", "COLORS_SEQ",
    "PARTICLE_COLORS", "PARTICLE_STYLES", "LABELS",
    "set_global_style", "setup_scientific_figure", "add_panel_labels",
    "apply_style", "get_T_color", "save_figure",
]

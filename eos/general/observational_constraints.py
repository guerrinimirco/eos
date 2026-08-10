"""Compatibility surface for the two constraint overlays that predate
:mod:`eos.general.constraints`.

The constraints now live in one module family with their data and their
builder — `eos/general/constraints/` — and are reached through a single
plane-keyed call::

    from eos.general.constraints import overlay, list_available

    overlay(ax, "M-R")          # was add_observational_constraints(ax)
    overlay(ax, "M-Lambda")     # was add_tidal_constraints(ax)

`overlay` also reaches the planes these two functions never covered (pressure
and energy against density, the symmetry energy), and can shade a posterior
continuously instead of drawing 68%/95% rings.

The two names below are kept because `nucleation` imports them; they are thin
wrappers with the same signature and the same output, and they go away once
that import is updated.
"""
from eos.general.constraints import (            # noqa: F401  (re-exported)
    DATA_DIR as DEFAULT_CONTOUR_DIR,
    LABEL_FS as MR_LABEL_FS,
    REGISTRY,
    _smooth_closed,
    available,
    list_available,
    overlay,
)

#: The mass-radius sources as ``{key: (label, colour, anchor)}``, the shape
#: this module exposed before the registry existed. Derived from REGISTRY
#: rather than restated, so a colour or label is still defined in exactly one
#: place. `nucleation` reads it to colour its own M-R annotations.
MR_CONSTRAINTS = {
    c.key: (c.label, c.colour, c.anchor)
    for c in REGISTRY if c.plane == "M-R" and c.kind == "contour_2d"
}


def add_observational_constraints(ax, contour_dir=None, show_mass_bands=True,
                                  inline_labels=False):
    """Mass-radius constraints. Superseded by ``overlay(ax, "M-R")``."""
    _reject_custom_dir(contour_dir)
    return overlay(ax, "M-R", inline_labels=inline_labels,
                   show_mass_bands=show_mass_bands)


def add_tidal_constraints(ax, contour_dir=None, inline_labels=False):
    """Mass-tidal constraints. Superseded by ``overlay(ax, "M-Lambda")``."""
    _reject_custom_dir(contour_dir)
    return overlay(ax, "M-Lambda", inline_labels=inline_labels)


def _reject_custom_dir(contour_dir):
    """The old functions took a contour_dir; the registry resolves its own.

    Naming the packaged directory is a no-op and is allowed, since callers
    that passed ``DEFAULT_CONTOUR_DIR`` explicitly were asking for exactly
    what they now get by default. Naming any OTHER directory raises: silently
    ignoring it would draw the packaged contours while the caller believed it
    had substituted another set, and that is a figure which is wrong in a way
    nothing on the page reveals.
    """
    if contour_dir is None:
        return
    from pathlib import Path
    if Path(contour_dir).resolve() == Path(DEFAULT_CONTOUR_DIR).resolve():
        return
    raise TypeError(
        f"contour_dir={contour_dir!r} is no longer supported: "
        f"eos.general.constraints resolves its own data directory so an "
        f"installed wheel works with no path argument. To draw a different "
        f"contour set, point eos.general.constraints.DATA_DIR at it, or add a "
        f"Constraint entry to eos.general.constraints.REGISTRY.")

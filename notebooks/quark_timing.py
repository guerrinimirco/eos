# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "eos").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from eos import abpr, alphabag, ccdm, njl, vmit
from eos.ccdm.table import rows_from_result as ccdm_rows_from_result
from eos.general.figure_style import (LABELS, OKAB_CAT, log_decades,
                                      panel_label, paper_grid,
                                      particle_style)
from eos.general import pqcd
from eos.general.physics_constants import hc3
from eos.general.table_io import matrix_from_rows
from eos.ccdm.couplings import vector_coupling
from scipy.optimize import root

from eos.ccdm.thermodynamics import (dielectric, glue_potential,
                                     state_at, thermo_from_mu)
from eos.njl import vector_self_energy
from eos.njl.table import rows_from_result as njl_rows_from_result

# Name -> package. Every model exposes the same entry points under the same
# names — `Parameters`, `SpeciesFlags`, `eos_point`, `eos_table` — so one loop
# body serves all four. `rows_from_result` is the one that is not uniform:
# `vmit` and `alphabag` re-export it from the package, `njl` and `ccdm` leave it
# in their `table` module, so it is looked up here once rather than branched on
# inside the loop.
MODELS = {"vmit": vmit, "alphabag": alphabag, "njl": njl, "ccdm": ccdm}

# The two that carry a `backends/`, and so accept `backend='fast'`.
BACKEND_MODELS = {"njl": njl, "ccdm": ccdm}
ROWS_FROM_RESULT = {"vmit": vmit.rows_from_result,
                    "alphabag": alphabag.rows_from_result,
                    "njl": njl_rows_from_result,
                    "ccdm": ccdm_rows_from_result}


# %% [markdown]
# ## One T = 0 beta-equilibrium line, per model
#
# The quark counterpart of `hadronic_timing`: four models of deconfined matter
# driven through the same public entry points, in the same mode, on the same
# density grid, and timed the same way.
#
# The sectors are the six names of CLAUDE.md section 4, spelled identically in
# every model, plus `csc` — the pairing sector, which exists only where the
# functional has it. `vmit` and `alphabag` carry no such field, so passing it
# would be a `TypeError`, which is this notebook's own bug and not a model
# refusing anything; the flag is therefore added per model rather than shared.
#
# `muons` is off for all four, and that is a gap rather than a choice: `njl` and
# `ccdm` wire the muon family, `vmit` and `alphabag` do not and raise if asked.
# One shared value is what keeps all four models in the same figures.

# %%
# The six section-4 names, identical in every model.
SPECIES = dict(muons=False, thermal_mesons=False, thermal_neutrinos=False, photons=False)

# `csc` only where the pairing sector is in the functional. False everywhere:
# this notebook is unpaired quark matter.
CSC = {"njl": False, "ccdm": False}

# The published set each model is driven at; None = Parameters.default().
#   vmit, alphabag  ship exactly one set each, so there is nothing to name.
#   njl             "rkh" is the shipped default — the set every verified number
#                   in the model's documentation was produced at and the one
#                   test/baseline is frozen at. 
#   ccdm            "baseline" is the specification's own baseline set, likewise
#                   the shipped default. Its deconfinement onset sits just below
#                   1 fm^-3, well inside the grid below, which is the point:
#                   the densities underneath it come back unsolved and that is
#                   physics about the model, not a solver failure.
PARAMETER_SET = {"vmit": None, "alphabag": None,
                 "njl": "rkh", "ccdm": "baseline"}

# 0.1 to 1.6 fm^-3. The top is a few times saturation, which is where a hybrid
# star's quark core lives; the bottom is deliberately below any of these phases
# being the physical state, so `ccdm`'s onset shows up as refusals in the count
# rather than being hidden by a grid chosen to flatter it.
N_B = np.linspace(0.1, 1.6, 60)
T = 0.0

TABLES = {}          # {model: rows}, read by every figure below
PARAMS = {}          # {model: par}, read by the effective-mass figure

print(f"=== one line, beta_eq_neutrinoless, T = {T} MeV, "
      f"{len(N_B)} points from {N_B[0]:.3f} to {N_B[-1]:.3f} fm^-3 ===")

for name, module in MODELS.items():
    published = PARAMETER_SET[name]
    par = (module.Parameters.default() if published is None
           else module.Parameters.named(published))

    # A model that does not have one of the selected sectors refuses — at flag
    # construction where the sector is absent from the functional, inside
    # `eos_point` where the mode is not one this phase has. Both are the
    # library's contract working, so they are printed and skipped, never dressed
    # up as a result. `TypeError` is deliberately not caught: an unexpected
    # keyword is this notebook's own bug.
    # `backend` exists only where the model HAS a second flavour of its
    # integrals: `njl` and `ccdm` carry a `backends/`, the two bag models do
    # not and would raise TypeError. Added per model for the same reason `csc`
    # is, rather than passed to all four.
    fast = {"backend": "fast"} if name in BACKEND_MODELS else {}

    try:
        species = module.SpeciesFlags(
            **SPECIES, **({"csc": CSC[name]} if name in CSC else {}))

        # The first 'fast' call in a session pays numba's compile time, so
        # the jitted models run once untimed. Without this the compile lands
        # on the "first point" number and reads as a slow model.
        if fast:
            module.eos_point(par, "beta_eq_neutrinoless", species,
                             n_B=float(N_B[0]), T=T, **fast)

        start = time.perf_counter()
        point = module.eos_point(par, "beta_eq_neutrinoless", species,
                                 n_B=float(N_B[0]), T=T, **fast)
        first_s = time.perf_counter() - start

        start = time.perf_counter()
        table = module.eos_table(par, "beta_eq_neutrinoless", species,
                                 {"nB": N_B, "T": np.array([T])}, **fast)
        line_s = time.perf_counter() - start
    except (NotImplementedError, ValueError) as err:
        print(f"  [{name}] not supported: {err}")
        continue

    rows = ROWS_FROM_RESULT[name](table)
    TABLES[name] = rows
    PARAMS[name] = par

    # Non-convergence is a return value, not an exception: it is found by
    # testing `.ok`, which no `except` clause would ever see. Unlike the
    # hadronic notebook this does NOT drop the model — the first density of this
    # grid is below `ccdm`'s deconfinement onset, so a failure there is a
    # statement about where the phase begins and the rest of the line is still
    # a result.
    note = "" if point.ok else f"   [first point: {point.message}]"
    print(f"  [{name:9s} {published or 'default':9s}] "
          f"first point {1e3 * first_s:8.2f} ms   "
          f"line {line_s:7.3f} s   "
          f"{1e3 * line_s / max(len(rows), 1):7.2f} ms/pt   "
          f"{len(rows)}/{len(N_B)} points{note}")


# %% [markdown]
# ### The paired enumeration, full against restricted
#
# With `csc=True` every point solves several candidates and keeps the lowest
# free energy. The asymmetric `free` seed exists to let a CFL-layout solve
# fall to a state that is not CFL (a uSC/dSC-like state); where no such state
# exists it burns its whole retry ladder discovering so — measured at 90% of
# one njl point's cost. `patterns=("unpaired", "2SC", "CFL")` is the
# documented fast restriction: an explicit declaration that those states are
# not being hunted, made by the caller rather than by a default.
#
# A tiny grid, because even restricted, a paired point diagonalises the BdG
# problem at every quadrature node of every residual evaluation.

# %%
N_B_GRID = np.linspace(0.16, 2, 300)
SIMPLE_CSC = ("unpaired", "2SC", "CFL")

par = njl.Parameters.named(PARAMETER_SET[name])
species = module.SpeciesFlags(**SPECIES, csc=True)

timing = {}
for label, restriction in (("full", None), ("restricted", SIMPLE_CSC)):
        kwargs = {} if restriction is None else {"patterns": restriction}
        module.eos_table(par, "beta_eq_neutrinoless", species, axes,
                         backend="fast", **kwargs)       # warm up, not timed
        start = time.perf_counter()
        table = module.eos_table(par, "beta_eq_neutrinoless", species, axes,
                                 backend="fast", **kwargs)
        timing[label] = time.perf_counter() - start
        n_solved = len(ROWS_FROM_RESULT[name](table))

    n_pts = max(n_solved, 1)
    print(f"  [{name:5s}] full enumeration {1e3 * timing['full'] / n_pts:9.1f}"
          f" ms/pt   ('unpaired','2SC','CFL') "
          f"{1e3 * timing['restricted'] / n_pts:9.1f} ms/pt   "
          f"{timing['full'] / timing['restricted']:5.2f}x")


# %% [markdown]
# ## The same scan, paired
#
# The `p` sweep above is unpaired matter: `csc=False`, no gaps, and the only
# thing the dielectric moves is the chiral sector. Turning `csc=True` on makes
# the three gaps unknowns of the same solve, so the pairing sector is now a
# knob of its own — and the one that IS the pairing sector is the diquark
# coupling `G_D`, which is what this cell sweeps with the dielectric held.
#
# `G_D` is dressed by the dielectric exactly as the specification asks,
# `G_D -> G_D/chi^q`, so the number in `Parameters` is the BARE coupling and
# the gap it buys grows as the glue melts. The shipped `5e-6` MeV^-2 with
# `Lambda = 600` MeV is the calibrated point (a gap of 20-150 MeV at
# `mu_q ~ 450` MeV); the sweep brackets it.
#
# `patterns=CSC_PATTERNS` is the documented restriction of the timing cell
# above and is a declaration, not a default: uSC/dSC-like states are not
# hunted here. The pattern that wins is printed per set, because on this axis
# it is not constant — the enumeration is what decides, at every density.
#
# The density axis is short and the grid coarse compared with the unpaired
# scan: a paired point diagonalises the BdG problem at every quadrature node
# of every residual evaluation, and this is ~0.8 s/point against the unpaired
# scan's milliseconds.

# %%
# Restated from the sections above so this one runs on its own, straight after
# the import cell: every name here is either a declaration (which pairing
# candidates are hunted, which linestyle is which flavour) or four lines of
# matplotlib, and the cells that first define them solve tables of hundreds of
# points to get there. Same values, no second meaning.
CSC_PATTERNS = ("unpaired", "2SC", "CFL")
FLAVOUR_LS = {"u": "-", "d": "--", "s": "-."}
CURRENT = {"u": "m_u", "d": "m_d", "s": "m_s"}


def scan_figure(ylabel):
    fig, axes = paper_grid("1x1", mode="single", placeholder=False, aspect=1.25)
    ax = axes.ravel()[0]
    ax.set_xlabel(LABELS["nB"])
    ax.set_ylabel(ylabel)
    return fig, ax


def njl_style_legend(axis, entries, **kw):
    """A legend naming what the LINESTYLES mean, in neutral grey."""
    return axis.legend(handles=[plt.Line2D([], [], color="0.3", ls=s, label=n)
                                for n, s in entries], **kw)

CSC_SWEEP = ("G_D", (3.0e-6, 5.0e-6, 7.0e-6))

# The dielectric and bag sector held at the `p` sweep's own values, so the only
# thing moving between curves is the pairing.
CSC_HELD = dict(B_g_quarter=165.,
                g_q=3.0,
                g_s=3.0,
                m_sigma=550.0,
                gbar_omega=2.0,
                n_c=float("inf"),
                p=2)

CSC_SCAN_NB = np.linspace(0.8, 2.0, 20)
CSC_SCAN_FLAGS = ccdm.SpeciesFlags(csc=True)

csc_name, csc_values = CSC_SWEEP
csc_scan = {}
print("=== ccdm, beta_eq_neutrinoless, T = 0, csc=True ===")
print(f"sweeping {csc_name} over {list(csc_values)}; held: {CSC_HELD}\n")
for value in csc_values:
    held = {name: held_value for name, held_value in CSC_HELD.items()
            if name != csc_name}
    par = replace(ccdm.Parameters.default(), **held, **{csc_name: value})
    rows = ccdm_rows_from_result(ccdm.eos_table(
        par, "beta_eq_neutrinoless", CSC_SCAN_FLAGS,
        {"nB": CSC_SCAN_NB, "T": np.array([0.0])}, backend="fast",
        patterns=CSC_PATTERNS))
    label = f"{csc_name} = {value:.1e}"
    if not rows:
        print(f"  [{label:22s}]   0/{len(CSC_SCAN_NB)} points: no deconfined "
              f"phase on this axis")
        continue
    csc_scan[label] = (par, *matrix_from_rows(rows))
    # Which pattern won, and where: the enumeration decides per point, so a
    # single name for the whole curve would be a claim the table does not make.
    won = {}
    for row in rows:
        won[row["pattern"]] = won.get(row["pattern"], 0) + 1
    print(f"  [{label:22s}] {len(rows):3d}/{len(CSC_SCAN_NB)} points, onset at "
          f"n_B = {rows[0]['n_B']:.3f} fm^-3, "
          f"Delta_3 = {rows[-1]['Delta_3']:6.1f} MeV at the top; "
          f"patterns {won}")


def csc_curves(store):
    """(label, par, matrix, n_B, colour) for every set in a paired scan."""
    for (label, (par, matrix, axs)), colour in zip(store.items(), OKAB_CAT):
        yield label, par, matrix, axs["nB"], colour


# %%
# One figure, four planes, three colour-superconducting quark models. Every
# number that is a choice is named in this block; nothing below the divider is
# edited to change the physics.
from eos.general.sound_speeds import sound_speed_eq

CFL_NB = np.linspace(0.4, 1.8, 40)          # fm^-3
CFL_T = np.array([0.0])                     # MeV

# njl: eta_D is the pairing sector, eta_V and vector_form the vector one
# ("constant", "power_law", "gluon_exchange" -- njl.VECTOR_FORMS). The pattern
# is HELD at CFL rather than enumerated, so all three models are the same phase
# and panel (d) differentiates along ONE branch: the free enumeration puts a
# first-order jump in the middle of dP/deps, and the number there means nothing.
NJL_SET = "rkh"
NJL_HELD = dict(eta_D=1.45, eta_V=0.5, vector_form="constant",
                alpha=2.0 / 3.0, n_ref=0.48, G_V0_over_GS=0.5, M_g=500.0)
NJL_PATTERNS = ("CFL",)

# alphaBag: Delta0 is a per-call CONDITION of the cfl mode, not a parameter.
ALPHABAG_HELD = dict(m_s=150.0, alpha=0.3, B4=165.0)
ALPHABAG_DELTA0 = 100.0                     # MeV

# abpr: Delta0 IS one of its four parameters, and cfl is its only mode.
ABPR_HELD = dict(m_s=150.0, Delta0=80.0, a4=0.7, B4=135.0)

PQCD_MU_B = np.linspace(pqcd.MU_B_MIN, 3600.0, 40)   # MeV
PQCD_DELTA = ALPHABAG_DELTA0                # the gap the CFL band is drawn at
# ---------------------------------------------------------------------------

njl_par = replace(njl.Parameters.named(NJL_SET), **NJL_HELD)
alphabag_par = replace(alphabag.Parameters.default(), **ALPHABAG_HELD)
abpr_par = replace(abpr.Parameters.default(), **ABPR_HELD)

cfl_rows = {}
cfl_rows[rf"njl  $\eta_D$={njl_par.eta_D:g}, $\eta_V$={njl_par.eta_V:g}"] = \
    njl_rows_from_result(njl.eos_table(
        njl_par, "beta_eq_neutrinoless", njl.SpeciesFlags(csc=True),
        {"nB": CFL_NB, "T": CFL_T}, backend="fast", patterns=NJL_PATTERNS))
cfl_rows[rf"$\alpha$Bag CFL  $\Delta_0$={ALPHABAG_DELTA0:.0f} MeV"] = \
    alphabag.rows_from_result(alphabag.eos_table(
        alphabag_par, "cfl", alphabag.SpeciesFlags(),
        {"nB": CFL_NB, "T": CFL_T}, fixed={"Delta0": ALPHABAG_DELTA0}))
cfl_rows[rf"abpr CFL  $\Delta_0$={abpr_par.Delta0:.0f} MeV"] = \
    abpr.eos_table(abpr_par, axes={"nB": CFL_NB, "T": CFL_T}, rows=True)

print(f"=== CFL comparison, T = 0, {len(CFL_NB)} points, "
      f"n_B = {CFL_NB[0]:.2f} to {CFL_NB[-1]:.2f} fm^-3 ===")

cfl_curves = {}          # label -> (matrix, axs); the four panels read this
for label, rows in cfl_rows.items():
    # A density the phase does not exist at is ABSENT from the rows rather than
    # a NaN in them: the njl CFL branch does not reach the bottom of the grid.
    if not rows:
        print(f"  [{label:36s}]  0/{len(CFL_NB)} points")
        continue
    matrix, axs = matrix_from_rows(rows)
    cfl_curves[label] = (matrix, axs)
    gap = matrix["Delta"][0] if "Delta" in matrix else matrix["Delta_3"][0]
    print(f"  [{label:36s}] {len(axs['nB']):3d}/{len(CFL_NB)} points, "
          f"gap {gap[-1]:6.1f} MeV, P {matrix['P'][0][-1]:7.1f} MeV/fm^3 "
          f"at the top")

fig, axes = paper_grid("2x2", mode="double", placeholder=False, aspect=1.2)
(ax_P, ax_gap), (ax_mu, ax_cs2) = axes

# The bands first, so every model curve is drawn over them.
unpaired_lo, unpaired_hi = pqcd.band(PQCD_MU_B)
cfl_lo, cfl_hi = pqcd.band(PQCD_MU_B, Delta=PQCD_DELTA)
ax_mu.fill_between(PQCD_MU_B, unpaired_lo, unpaired_hi, color="0.55",
                   alpha=0.35, lw=0, zorder=0, label="pQCD N3LO, unpaired")
ax_mu.fill_between(PQCD_MU_B, cfl_lo, cfl_hi, facecolor="none",
                   edgecolor="0.35", hatch="///", lw=0.8, zorder=1,
                   label=rf"pQCD NLO, CFL $\Delta$={PQCD_DELTA:.0f} MeV")

for (label, (matrix, axs)), colour in zip(cfl_curves.items(), OKAB_CAT):
    n_B = axs["nB"]
    ax_P.plot(n_B, matrix["P"][0], color=colour, label=label)
    ax_mu.plot(matrix["mu_B"][0], matrix["P"][0], color=colour, label=label)
    ax_cs2.plot(n_B, sound_speed_eq(matrix["P"][0], matrix["eps"][0]),
                color=colour)
    if "Delta" in matrix:                     # alphaBag, abpr: one gap
        ax_gap.plot(n_B, matrix["Delta"][0], color=colour, ls="-")
        continue
    for key, style in (("Delta_1", "-"), ("Delta_3", "-.")):    # njl: three
        ax_gap.plot(n_B, matrix[key][0], color=colour, ls=style)

ax_P.set_xlabel(LABELS["nB"])
ax_P.set_ylabel(LABELS["P"])
ax_P.legend(loc="upper left", fontsize="xx-small")
panel_label(ax_P, "(a)")

ax_gap.set_xlabel(LABELS["nB"])
ax_gap.set_ylabel(r"$\Delta_\eta$ [MeV]")
njl_style_legend(ax_gap, [(r"$\Delta_1$ ($ds$)", "-"),
                          (r"$\Delta_3$ ($ud$)", "-.")],
                 loc="lower right", fontsize="xx-small")
panel_label(ax_gap, "(b)")

ax_mu.set_xlabel(LABELS["mu_B"])
ax_mu.set_ylabel(LABELS["P"])
ax_mu.set_yscale("log")
ax_mu.set_ylim(50.0, 3e4)
log_decades(ax_mu, axis="y")
ax_mu.legend(loc="lower right", fontsize="xx-small")
panel_label(ax_mu, "(c)")

ax_cs2.set_xlabel(LABELS["nB"])
ax_cs2.set_ylabel(r"$c_e^2 = \mathrm{d}P/\mathrm{d}\varepsilon$")
ax_cs2.axhline(1.0 / 3.0, color="0.5", ls="--", lw=0.8)   # the conformal value
panel_label(ax_cs2, "(d)")
plt.show()


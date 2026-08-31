# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "eos").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from eos import alphabag, ccdm, njl, vmit
from eos.ccdm.table import rows_from_result as ccdm_rows_from_result
from eos.general.figure_style import (LABELS, OKAB_CAT, log_decades,
                                      panel_label, paper_grid,
                                      particle_style)
from eos.general.physics_constants import hc3
from eos.general.table_io import matrix_from_rows
from eos.ccdm.couplings import vector_coupling
from eos.ccdm.thermodynamics import thermo_from_mu
from eos.njl import vector_self_energy
from eos.njl.table import rows_from_result as njl_rows_from_result

# Name -> package. Every model exposes the same entry points under the same
# names — `Parameters`, `SpeciesFlags`, `eos_point`, `eos_table` — so one loop
# body serves all four. `rows_from_result` is the one that is not uniform:
# `vmit` and `alphabag` re-export it from the package, `njl` and `ccdm` leave it
# in their `table` module, so it is looked up here once rather than branched on
# inside the loop.
MODELS = {"vmit": vmit, "alphabag": alphabag, "njl": njl, "ccdm": ccdm}
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
SPECIES = dict(hyperons=False, deltas=False, muons=False,
               thermal_mesons=False, thermal_neutrinos=False, photons=False)

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
    try:
        species = module.SpeciesFlags(
            **SPECIES, **({"csc": CSC[name]} if name in CSC else {}))

        start = time.perf_counter()
        point = module.eos_point(par, "beta_eq_neutrinoless", species,
                                 n_B=float(N_B[0]), T=T)
        first_s = time.perf_counter() - start

        start = time.perf_counter()
        table = module.eos_table(par, "beta_eq_neutrinoless", species,
                                 {"nB": N_B, "T": np.array([T])})
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
# ### The two backends, on the same line
#
# `njl` and `ccdm` carry the reference/fast split of CLAUDE.md section 9: the
# reference flavour is plain NumPy and is what correctness is judged against,
# and `backend='fast'` is the jitted kernel of the model's own `backends/`
# (each model has its own — `ccdm`'s recomputes an integration ceiling per mode,
# which `njl`'s cut theory has no notion of), plus the block-diagonalised BdG
# problem in the pairing sector.
#
# The default is `'reference'` everywhere, so a table is the same table it has
# always been and `test/baseline` keeps measuring the path it is frozen against.
# The two agree to round-off rather than bit for bit, because they sum the nine
# modes in different orders — which is why this cell reports the largest
# relative column difference beside the speedup rather than asserting equality.
#
# **The first `'fast'` call in a session pays numba's compile time**, so each
# backend is run once untimed before the clock starts.

# %%
BACKEND_MODELS = {"njl": njl, "ccdm": ccdm}

print(f"=== reference vs fast, beta_eq_neutrinoless, T = {T} MeV, "
      f"{len(N_B)} points ===")

for name, module in BACKEND_MODELS.items():
    published = PARAMETER_SET[name]
    par = module.Parameters.named(published)
    species = module.SpeciesFlags(**SPECIES, csc=CSC[name])
    axes = {"nB": N_B, "T": np.array([T])}

    timing, tables = {}, {}
    for backend in ("reference", "fast"):
        module.eos_table(par, "beta_eq_neutrinoless", species, axes,
                         backend=backend)          # warm up: compile, not time
        start = time.perf_counter()
        tables[backend] = module.eos_table(par, "beta_eq_neutrinoless",
                                           species, axes, backend=backend)
        timing[backend] = time.perf_counter() - start

    solved = {b: ROWS_FROM_RESULT[name](t) for b, t in tables.items()}
    n_solved = len(solved["reference"])

    # The largest relative difference over every numeric column, skipping the
    # ones that are identically zero: Y_C and Y_e vanish in a CFL phase, so a
    # ratio of one round-off to another there is a cancellation and not a
    # disagreement between the two flavours.
    worst, worst_key = 0.0, ""
    for row_ref, row_fast in zip(solved["reference"], solved["fast"]):
        for key, value in row_ref.items():
            other = row_fast[key]
            if not isinstance(value, float) or value != value:
                continue
            if max(abs(value), abs(other)) < 1.0e-10:
                continue
            relative = abs(value - other) / max(abs(value), abs(other))
            if relative > worst:
                worst, worst_key = relative, key

    per_point = {b: 1e3 * s / max(n_solved, 1) for b, s in timing.items()}
    print(f"  [{name:5s}] reference {per_point['reference']:8.2f} ms/pt   "
          f"fast {per_point['fast']:8.2f} ms/pt   "
          f"{per_point['reference'] / per_point['fast']:5.2f}x   "
          f"{n_solved}/{len(N_B)} points   "
          f"worst column {worst_key or '-'} {worst:.1e}")


# %% [markdown]
# ### From rows to arrays
#
# `rows_from_result` gives the long format — a plain list of dicts, one per
# **converged** point, with the same column names in every model.
# `matrix_from_rows` lays those rows back on the grid they were asked for and
# returns `({name: array[i_temp, i_nB]}, axes)`, so `matrix["P"][0]` is the whole
# T = 0 isotherm.
#
# The axes are left to default here rather than passed in. The solved densities
# come back carrying float noise off the requested grid — `njl` returns
# 0.10000000000000006 where `np.linspace` gave 0.1 — and `matrix_from_rows`
# matches exactly and raises rather than dropping a row that is not on the axes
# it was handed. Defaulting takes the densities the rows actually carry, which
# for one temperature and one line is what a curve is. The trade is that a
# refused density is absent instead of nan, which is exactly what the counts
# printed above already report.

# %%
MATRIX = {}
for name, rows in TABLES.items():
    matrix, axes = matrix_from_rows(rows)
    MATRIX[name] = (matrix, axes["nB"])
    print(f"  [{name:9s}] {len(axes['nB'])} densities, "
          f"{axes['nB'][0]:.3f} -> {axes['nB'][-1]:.3f} fm^-3   "
          f"columns: {len(matrix)}")


# %% [markdown]
# ### Pressure
#
# One curve per model, colour from `OKAB_CAT`. Linear in both axes: over 0.1 to
# 1.6 fm^-3 the pressure spans zero to several hundred MeV/fm^3 and passes
# through zero on the way, which a log axis cannot draw. Where a curve starts
# well inside the panel, that model has no deconfined phase below the density it
# starts at.

# %%
fig, axes = paper_grid("1x1", mode="centered", placeholder=False, aspect=1.3)
ax = axes.ravel()[0]

for (name, (matrix, n_B)), colour in zip(MATRIX.items(), OKAB_CAT):
    ax.plot(n_B, matrix["P"][0], color=colour, label=name)

ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(LABELS["P"])
ax.set_xlim(0.0, 1.65)
ax.legend(loc="lower right")
panel_label(ax, "(a)")

plt.show()


# %% [markdown]
# ### Composition
#
# The flavour fractions and the leptons, all four models on one panel. Colour is
# the SPECIES and comes from `particle_style` — u blue, d orange, s green,
# electrons grey — and linestyle is the MODEL, so the two legends name four
# things each instead of one legend naming sixteen. `particle_style` also
# returns a linestyle, the one that marks a multiplet in a hadronic composition
# figure; here the multiplet is the same for every curve and the linestyle is
# needed to say which functional produced it, so only the colour is taken.
#
# Y_e is the one that separates these models: it is the electron fraction beta
# equilibrium needs to neutralize whatever charge the flavour composition leaves
# over, so a phase that sits close to Y_u = Y_d = Y_s barely needs electrons at
# all and one far from it needs many. Muons would appear here as a `Y_mu-`
# column in `njl` and `ccdm`; the flag is off (above), so the column is zero and
# is dropped by the same threshold that drops any other empty curve.

# %%
# Linestyle is the model. Four of them, so the legend that names the models is
# read off this dict rather than from the curves.
MODEL_STYLE = {"vmit": "-", "alphabag": "--", "njl": "-.", "ccdm": ":"}

# The row column name -> the species name `particle_style` knows.
FRACTIONS = {"Y_u": "u", "Y_d": "d", "Y_s": "s",
             "Y_e": "e", "Y_mu-": "mu-"}

# Full page width: sixteen curves and two legends do not fit a single column.
fig, axes = paper_grid("1x1", mode="double", placeholder=False, aspect=1.7)
ax = axes.ravel()[0]

for name, (matrix, n_B) in MATRIX.items():
    for column, species_name in FRACTIONS.items():
        if column not in matrix:
            continue
        fraction = matrix[column][0]
        if np.nanmax(fraction) < 1e-4:      # below the panel, not worth a colour
            continue
        colour, _ = particle_style(species_name)
        ax.plot(n_B, fraction, color=colour, ls=MODEL_STYLE[name])

ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(LABELS["Y_i"])
ax.set_xlim(0.0, 1.65)
ax.set_yscale("log")
ax.set_ylim(1e-4, 3.0)
# CMU Serif has no U+2212, so matplotlib's own mathtext log labels come out as
# hollow boxes below 1. `log_decades` is figure_style's protection for exactly
# that axis and is never removed (CLAUDE.md section 10).
log_decades(ax, axis="y")

# Two legends, built from proxy lines: one says which colour is which species,
# the other which linestyle is which model.
species_keys = [plt.Line2D([], [], color=particle_style(s)[0], label=s)
                for s in ("u", "d", "s", "e")]
model_keys = [plt.Line2D([], [], color="0.3", ls=style, label=name)
              for name, style in MODEL_STYLE.items() if name in MATRIX]
ax.add_artist(ax.legend(handles=species_keys, loc="lower center", ncol=4,
                        fontsize="x-small"))
ax.legend(handles=model_keys, loc="center left", fontsize="x-small")
panel_label(ax, "(b)")

plt.show()


# %% [markdown]
# ### Effective quark masses
#
# The same colour-is-species, linestyle-is-model reading, and the panel shows two
# genuinely different things rather than one quantity computed four ways.
#
# * **`njl` and `ccdm` return a mass.** Their constituent masses come out of a
#   gap equation solved at every point, so `M_u`, `M_d`, `M_s` are columns of the
#   table and fall with density as chiral symmetry is restored. That falling
#   curve is most of what those two models cost — it is why their per-point
#   timings above are tens to hundreds of times the bag models'.
# * **`vmit` and `alphabag` return no mass column at all.** They have no gap
#   equation: the quark masses are current masses carried on the parameter
#   object and are the same number at every density. What is drawn for them is
#   therefore `par.m_u`, `par.m_d`, `par.m_s`, read from the parameters, and the
#   flat lines are shown rather than omitted because a flat line IS the physical
#   statement — alphaBag's u and d sit at exactly zero and are off the bottom of
#   a log axis, which is the same statement again.
#
# The two are not interchangeable and the panel does not pretend they are: one
# set of curves is a solved quantity and the other is an input.

# %%
fig, axes = paper_grid("1x1", mode="double", placeholder=False, aspect=1.7)
ax = axes.ravel()[0]

for name, (matrix, n_B) in MATRIX.items():
    for flavour in ("u", "d", "s"):
        column = f"M_{flavour}"
        if column in matrix:
            mass = matrix[column][0]                 # solved: the gap equation
        else:
            # No gap equation in this functional; the mass is the parameter.
            mass = np.full(n_B.shape,
                           getattr(PARAMS[name], f"m_{flavour}"))
        colour, _ = particle_style(flavour)
        ax.plot(n_B, mass, color=colour, ls=MODEL_STYLE[name])

ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(r"$m^*_i$ [MeV]")
ax.set_xlim(0.0, 1.65)
ax.set_yscale("log")
ax.set_ylim(1.0, 1e3)
log_decades(ax, axis="y")

flavour_keys = [plt.Line2D([], [], color=particle_style(f)[0], label=f)
                for f in ("u", "d", "s")]
ax.add_artist(ax.legend(handles=flavour_keys, loc="lower left", ncol=3,
                        fontsize="x-small"))
ax.legend(handles=model_keys, loc="lower right", fontsize="x-small")
panel_label(ax, "(c)")

plt.show()

# %% [markdown]
# ## The njl parametrizations, side by side
#
# `eos.njl.PUBLISHED_SETS` ships three, and at `csc=False` they are **not**
# three independent points: the diquark coupling `eta_D` only enters the
# pairing sector, so with the gaps off `kunkel` differs from `rkh` by its
# vector coupling and by nothing else. What separates the three here is
# therefore one sector — how the model carries vector repulsion:
#
# * **rkh** — the RKH vacuum fit (Rehberg, Klevansky, Huefner, PRC 53, 410),
#   `eta_V = 0`: no vector repulsion at all. `Sigma_V` is identically zero.
# * **kunkel** — `eta_D = 1.45, eta_V = 0.7`, the constant-coupling form, so
#   `Sigma_V` grows linearly with quark density. `eta_D` is inert at
#   `csc=False`; it is what makes this set a strong-coupling point once
#   `csc=True`.
# * **gluon_exchange** — `G_V = G_V0/[1 + 8 k_F^2/(9 M_g^2)]`, a coupling that
#   is a FUNCTION of the state, so `Sigma_V` saturates instead of running away.
#
# The tier-1 vacuum numbers (Lambda, G_S Lambda^2, K Lambda^5, the current
# masses) are the same in all three, so the vacuum masses and the whole chiral
# sector start identical and separate only where the density does.
#
# NJL has no meson mean fields — it is a contact interaction, and the only
# field-like quantity it carries is `Sigma_V`, the vector self-energy that
# shifts `mu*_j = mu_j - Sigma_V`. The condensates phi_f are the model's other
# "field", and they are the constituent masses of the mass figure restated,
# since `M_f = m_f - 4 G_S phi_f + 2 K phi_g phi_h`.
#
# Colour is the SET in every figure, because comparing parametrizations is what
# these are for; the species or field is the linestyle.
#
# **The mass and composition figures come out degenerate, and that is the
# result.** At fixed n_B in beta equilibrium the three sets agree on the
# constituent masses, on every Y_i and on mu_C to 1e-14, and disagree on P, eps
# and mu_B. Sigma_V is FLAVOUR-BLIND -- it shifts every mode by the same
# amount -- so it cancels out of mu_C = mu_d - mu_u, which is what beta
# equilibrium and neutrality close the composition with; and the condensates
# that set M_f are functions of the densities, which the density axis has
# already fixed. A vector coupling in this model buys stiffness and nothing
# else, which is exactly why it is the knob a hybrid-star study turns.

# %%
NJL_SETS = ("rkh", "kunkel", "gluon_exchange")
NJL_NB = np.linspace(0.2, 1.8, 80)      # above the chiral transition, into the
                                        # density a hybrid star's core reaches
NJL_FLAGS = njl.SpeciesFlags(csc=False)  # unpaired: no gaps, so eta_D is inert

njl_lines = {}
print("=== njl published sets, beta_eq_neutrinoless, T = 0, unpaired ===")
for name in NJL_SETS:
    par = njl.Parameters.named(name)
    table = njl.eos_table(par, "beta_eq_neutrinoless", NJL_FLAGS,
                          {"nB": NJL_NB, "T": np.array([0.0])})
    rows = njl_rows_from_result(table)
    njl_lines[name] = (par, rows)
    print(f"  [{name:15s}] {len(rows):3d}/{len(NJL_NB)} points   "
          f"eta_D={par.eta_D:4.2f} eta_V={par.eta_V:4.2f} "
          f"form={par.vector_form}")

# The matrices once, so each figure below is a plot and not another solve.
njl_matrix = {name: matrix_from_rows(rows)
              for name, (par, rows) in njl_lines.items()}

FLAVOUR_LS = {"u": "-", "d": "--", "s": "-."}

# The sets lie exactly on top of each other in the mass and composition
# figures, so they are drawn thick to thin. A coincidence has to be VISIBLE as
# a coincidence; drawn at one width the last curve hides the others and the
# figure reads as a missing result.
NJL_WIDTHS = (3.0, 1.8, 0.9)

# One figure per quantity. `single` is the one-column PRD width, which is what
# a figure carrying a single panel is for.
def njl_figure():
    fig, axes = paper_grid("1x1", mode="single", placeholder=False, aspect=1.25)
    return fig, axes.ravel()[0]


def njl_style_legend(axis, entries, **kw):
    """A legend naming what the LINESTYLES mean, in neutral grey."""
    return axis.legend(handles=[plt.Line2D([], [], color="0.3", ls=s, label=n)
                                for n, s in entries], **kw)


# %% [markdown]
# ### njl — pressure

# %%
fig, ax = njl_figure()
for (name, (matrix, axs)), colour, width in zip(njl_matrix.items(), OKAB_CAT,
                                                NJL_WIDTHS):
    ax.plot(axs["nB"], matrix["P"][0], color=colour, lw=width, label=name)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(LABELS["P"])
ax.legend(loc="upper left", fontsize="x-small")
plt.show()

# %% [markdown]
# ### njl — constituent masses
#
# From the gap equation at every point, so these are solved quantities. All
# three sets coincide; the widths are what makes that legible.

# %%
fig, ax = njl_figure()
for (name, (matrix, axs)), colour, width in zip(njl_matrix.items(), OKAB_CAT,
                                                NJL_WIDTHS):
    for flavour, style in FLAVOUR_LS.items():
        ax.plot(axs["nB"], matrix[f"M_{flavour}"][0], color=colour, ls=style,
                lw=width)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(r"$M_i$ [MeV]")
ax.set_yscale("log")
log_decades(ax, axis="y")
njl_style_legend(ax, FLAVOUR_LS.items(), loc="lower left", ncol=3,
                 fontsize="x-small")
plt.show()

# %% [markdown]
# ### njl — composition

# %%
fig, ax = njl_figure()
for (name, (matrix, axs)), colour, width in zip(njl_matrix.items(), OKAB_CAT,
                                                NJL_WIDTHS):
    for flavour, style in FLAVOUR_LS.items():
        ax.plot(axs["nB"], matrix[f"Y_{flavour}"][0], color=colour, ls=style,
                lw=width)
    ax.plot(axs["nB"], matrix["Y_e"][0], color=colour, ls=":", lw=width)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(LABELS["Y_i"])
ax.set_yscale("log")
ax.set_ylim(1e-5, 3.0)
log_decades(ax, axis="y")
njl_style_legend(ax, list(FLAVOUR_LS.items()) + [("e", ":")],
                 loc="lower left", ncol=4, fontsize="x-small")
plt.show()

# %% [markdown]
# ### njl — the vector self-energy
#
# The one field-like quantity NJL carries. `n_q` is the QUARK density in
# natural units, three times n_B converted through hc^3; `vector_self_energy`
# is a coupling evaluated on the state rather than a stored number, so it is
# recomputed here instead of read from a column.

# %%
fig, ax = njl_figure()
for (name, (par, rows)), colour, width in zip(njl_lines.items(), OKAB_CAT,
                                              NJL_WIDTHS):
    n_B = njl_matrix[name][1]["nB"]
    sigma_V = np.array([vector_self_energy(par, 3.0 * float(n) * hc3)
                        for n in n_B])
    ax.plot(n_B, sigma_V, color=colour, lw=width, label=name)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(r"$\Sigma_V$ [MeV]")
ax.legend(loc="upper left", fontsize="x-small")
plt.show()

# %% [markdown]
# ## The ccdm parametrizations, side by side
#
# `eos.ccdm.PUBLISHED_SETS` ships four, and unlike njl's they are not one knob
# seen three ways — each moves a different part of the functional:
#
# * **baseline** — the specification's own set, `B_g^(1/4) = 150 MeV`,
#   `m_sigma = 550 MeV`, `g_q = 3.0`. The shipped default.
# * **novector** — the same with `gbar_omega = 0`: the L1 -> L0 reduction, the
#   vector meson switched off at its coupling rather than by a flag (CLAUDE.md
#   section 4 — a sector the model already carries a coupling for gets no
#   flag).
# * **dressed** — `q = 1`, which dresses the DIQUARK coupling,
#   `G_D -> G_D/chi^q`. With `csc=False` there is no diquark condensate, so
#   `G_D` never enters and this set is **bit-identical to baseline here** —
#   every column agrees to exactly 0.0. That is not a defect and not a
#   duplicate entry: it is the same structural fact as `njl`'s `eta_D`, and it
#   means an unpaired comparison of "all four sets" is really a comparison of
#   three. Turn `csc=True` and it separates.
# * **stiff** — `B_g^(1/4) = 190 MeV`, a larger gluon-condensate bag. A bigger
#   bag costs more to open, so this set's **deconfinement onset moves up** and
#   its line starts well inside the axis. That is the model speaking, not a
#   solver failure, and the counts printed below say where each one begins.
#
# `novector` separates from `baseline` in P, `omega_0` and `Sigma_R` and in
# NOTHING else — the masses, every Y_i and the two confining fields agree to
# 1e-12. The vector field is flavour-blind, so it cancels out of the
# beta-equilibrium condition that fixes the composition, and at fixed n_B the
# scalar sector is already determined. Only `stiff`, which moves the bag
# itself, changes the composition and the fields.
#
# **This model does have mean fields**, which is the difference from njl, and
# they are columns of the table rather than something to recompute:
#
# * `sigma`, `zeta` [MeV] — the non-strange and strange scalar condensates.
#   `M*_(u,d) = (g_q sigma + m_q)/chi` and `M*_s = (g_s zeta + m_s)/chi`, so
#   those two and the dielectric together ARE the effective-mass figure.
# * `omega_0` [MeV] — the vector field, and `Sigma_R` [MeV] the rearrangement
#   self-energy that rides with it (it enters mu and P, never eps — section 8).
# * `phi_bar` — the confining scalar, dimensionless.
# * `chi_diel` = (1 - phi_bar^4)^p — the dielectric function. It is named with
#   a suffix because `chi` is already the shared column name for the
#   mixed-phase quark volume fraction, which here is identically 1.
#
# `branch` is reported per point: the enumeration chooses between the
# `restored` and `partial` chiral branches by free energy, and a set can change
# branch part-way up the density axis.

# %%
CCDM_SETS = ("baseline", "novector", "dressed", "stiff")
CCDM_NB = np.linspace(0.8, 2.0, 40)      # starts below every onset, so the
                                         # onsets themselves are in the figures
CCDM_FLAGS = ccdm.SpeciesFlags(csc=False)

ccdm_lines = {}
print("=== ccdm published sets, beta_eq_neutrinoless, T = 0, unpaired ===")
for name in CCDM_SETS:
    par = ccdm.Parameters.named(name)
    table = ccdm.eos_table(par, "beta_eq_neutrinoless", CCDM_FLAGS,
                           {"nB": CCDM_NB, "T": np.array([0.0])})
    rows = ccdm_rows_from_result(table)
    ccdm_lines[name] = (par, rows)
    if rows:
        branches = sorted({row["branch"] for row in rows})
        print(f"  [{name:9s}] {len(rows):3d}/{len(CCDM_NB)} points, onset at "
              f"n_B = {rows[0]['n_B']:.3f} fm^-3   branches: {branches}")
    else:
        # A set with no deconfined phase anywhere on this axis is a statement
        # about the set, and is printed rather than silently absent.
        print(f"  [{name:9s}]   0/{len(CCDM_NB)} points: no deconfined phase "
              f"on this density axis")

ccdm_matrix = {name: matrix_from_rows(rows)
               for name, (par, rows) in ccdm_lines.items() if rows}

# Thick to thin, so a set lying exactly on another is VISIBLE as a coincidence:
# `dressed` sits on `baseline` everywhere, `novector` on it in all but the
# pressure and vector figures.
CCDM_WIDTHS = (3.2, 2.2, 1.4, 0.8)


def ccdm_figure():
    fig, axes = paper_grid("1x1", mode="single", placeholder=False, aspect=1.25)
    return fig, axes.ravel()[0]


def ccdm_curves():
    """(name, matrix, n_B, colour, width) for every set that solved."""
    for (name, (matrix, axs)), colour, width in zip(ccdm_matrix.items(),
                                                    OKAB_CAT, CCDM_WIDTHS):
        yield name, matrix, axs["nB"], colour, width


# %% [markdown]
# ### ccdm — pressure

# %%
fig, ax = ccdm_figure()
for name, matrix, n_B, colour, width in ccdm_curves():
    ax.plot(n_B, matrix["P"][0], color=colour, lw=width, label=name)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(LABELS["P"])
ax.legend(loc="upper left", fontsize="x-small")
plt.show()

# %% [markdown]
# ### ccdm — effective masses

# %%
fig, ax = ccdm_figure()
for name, matrix, n_B, colour, width in ccdm_curves():
    for flavour, style in FLAVOUR_LS.items():
        ax.plot(n_B, matrix[f"M_{flavour}"][0], color=colour, ls=style, lw=width)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(r"$M^*_i$ [MeV]")
ax.set_yscale("log")
log_decades(ax, axis="y")
njl_style_legend(ax, FLAVOUR_LS.items(), loc="lower left", ncol=3,
                 fontsize="x-small")
plt.show()

# %% [markdown]
# ### ccdm — composition

# %%
fig, ax = ccdm_figure()
for name, matrix, n_B, colour, width in ccdm_curves():
    for flavour, style in FLAVOUR_LS.items():
        ax.plot(n_B, matrix[f"Y_{flavour}"][0], color=colour, ls=style, lw=width)
    ax.plot(n_B, matrix["Y_e"][0], color=colour, ls=":", lw=width)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(LABELS["Y_i"])
ax.set_yscale("log")
ax.set_ylim(1e-5, 3.0)
log_decades(ax, axis="y")
njl_style_legend(ax, list(FLAVOUR_LS.items()) + [("e", ":")],
                 loc="lower left", ncol=4, fontsize="x-small")
plt.show()

# %% [markdown]
# ### ccdm — the confining fields
#
# `phi_bar` is the confining scalar and `chi = (1 - phi_bar^4)^p` the
# dielectric it drives. Both are dimensionless, which is why they share a
# figure: chi -> 0 is the confined end, where a quark's effective mass diverges.

# %%
fig, ax = ccdm_figure()
for name, matrix, n_B, colour, width in ccdm_curves():
    ax.plot(n_B, matrix["phi_bar"][0], color=colour, ls="-", lw=width, label=name)
    ax.plot(n_B, matrix["chi_diel"][0], color=colour, ls="--", lw=width)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(r"$\bar{\Phi}$,  $\chi$")
njl_style_legend(ax, [(r"$\bar{\Phi}$", "-"), (r"$\chi$", "--")],
                 loc="center left", fontsize="x-small")
plt.show()

# %% [markdown]
# ### ccdm — the scalar fields

# %%
fig, ax = ccdm_figure()
for name, matrix, n_B, colour, width in ccdm_curves():
    ax.plot(n_B, matrix["sigma"][0], color=colour, ls="-", lw=width, label=name)
    ax.plot(n_B, matrix["zeta"][0], color=colour, ls="--", lw=width)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(r"$\sigma$,  $\zeta$ [MeV]")
njl_style_legend(ax, [(r"$\sigma$", "-"), (r"$\zeta$", "--")],
                 loc="center left", fontsize="x-small")
plt.show()

# %% [markdown]
# ### ccdm — the vector field and its rearrangement

# %%
fig, ax = ccdm_figure()
for name, matrix, n_B, colour, width in ccdm_curves():
    ax.plot(n_B, matrix["omega_0"][0], color=colour, ls="-", lw=width, label=name)
    ax.plot(n_B, matrix["Sigma_R"][0], color=colour, ls="--", lw=width)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(r"$\omega_0$,  $\Sigma_R$ [MeV]")
njl_style_legend(ax, [(r"$\omega_0$", "-"), (r"$\Sigma_R$", "--")],
                 loc="center left", fontsize="x-small")
plt.show()

# %% [markdown]
# ### Why switching the vector OFF makes ccdm stiffer
#
# The pressure figure looks backwards: `novector` sits several hundred
# MeV/fm^3 ABOVE `baseline`, so removing a repulsive vector meson appears to
# stiffen the matter. It does, and the reason is the rearrangement term rather
# than the field.
#
# `eos.ccdm.couplings` makes the vector coupling a FUNCTION of the density,
#
#     g_omega(n_B) = gbar_omega / [1 + (n_B/n_c)^2] ,
#
# a repulsion that dies off at high density — which is what keeps the sound
# speed off the causal limit without a hand-placed ceiling. With `n_c = 1`
# fm^-3 it falls steeply: g_omega = 3.20 at n_B = 0.5 and 0.80 at 2.0. A
# coupling that FALLS has `dg_omega/dn_B < 0`, and the rearrangement
# self-energy is built from exactly that derivative,
#
#     Sigma_R = (dg_omega/dn_B) omega_0 n_B  <  0 .
#
# So the vector sector contributes to the two thermodynamic quantities
# differently, which is the whole content of "Sigma_R enters mu and P, never
# eps" (CLAUDE.md section 8):
#
#     eps :  + W                    with W = (1/2) m_omega^2 omega_0^2
#     P   :  + W  +  n_q Sigma_R
#
# and at these densities the rearrangement piece is about 2.5x the field
# energy and negative, so the sector's NET contribution to the pressure is
# negative. Deleting it therefore raises P. The cell below checks that
# arithmetic against the two solved tables rather than asserting it: the
# predicted differences reproduce the observed ones to the last printed digit.
#
# None of this is a consistency failure — `eos/ccdm/verify` finds Euler at
# 7e-16 and n = dP/dmu_B at 1.5e-5 on these same states. It is what a strongly
# density-dependent coupling does, and it is why `novector` is the reduction
# the verify suite uses: the two differ by a term that is easy to get wrong and
# whose sign is not the one intuition supplies.

# %%
BASE = ccdm.Parameters.named("baseline")

print(f"g_omega(n_B) = {BASE.gbar_omega} / [1 + (n_B/{BASE.n_c})^2],  "
      f"m_omega = {BASE.m_omega} MeV\n")
print(f"{'n_B':>5} {'g_omega':>8} {'W':>9} {'n_q*Sig_R':>10} "
      f"{'dP obs':>9} {'dP pred':>9} {'deps obs':>9} {'deps pred':>10}")

base_rows = {round(row["n_B"], 6): row for row in ccdm_lines["baseline"][1]}
novec_rows = {round(row["n_B"], 6): row for row in ccdm_lines["novector"][1]}

for key in sorted(base_rows):
    if key not in novec_rows:
        continue
    base, novec = base_rows[key], novec_rows[key]
    # W is the vector field energy, which eps and P both carry; n_q Sigma_R is
    # the rearrangement, which only P carries. Removing the sector removes W
    # from eps, and W + n_q Sigma_R from P.
    W = 0.5 * BASE.m_omega ** 2 * base["omega_0"] ** 2 / hc3
    rearrangement = 3.0 * key * base["Sigma_R"]
    if abs(key - round(key, 1)) > 1e-9:      # print a readable subset
        continue
    print(f"{key:5.2f} {vector_coupling(BASE, key * hc3):8.4f} {W:9.2f} "
          f"{rearrangement:10.2f} {novec['P'] - base['P']:9.2f} "
          f"{-(W + rearrangement):9.2f} {novec['eps'] - base['eps']:9.2f} "
          f"{-W:10.2f}")

# %% [markdown]
# ## The two transitions, seen at fixed mu_B
#
# Everything above this point was solved at fixed **density**, and a
# fixed-density sweep cannot show a first-order transition: it walks straight
# through the region where the branch exists but is not the favoured state,
# which is why `ccdm`'s table reports an "onset" near 1 fm^-3 that is not where
# quark matter actually takes over. A transition is a statement about which
# branch has the LOWER free energy, and at T = 0 that is a comparison of
# pressures at equal potentials — so it has to be made at fixed mu_B.
#
# `eos.ccdm.thermodynamics.thermo_from_mu` is the surface for it. It closes the
# model's internal system (the four fields, the gaps, colour neutrality) at
# declared potentials in ONE declared branch, and the branch list at fixed
# potential carries one entry the density list cannot:
#
#     POTENTIAL_BRANCHES = ("confined", "restored", "partial")
#     DENSITY_BRANCHES   =             ("restored", "partial")
#
# `confined` is absent from the density list on purpose — with the dielectric
# closed the quarks are not in the medium at all, so n_B = 0 identically and no
# nonzero-density row can be met. At fixed mu_B it is exactly the branch that
# describes the vacuum, and without it there is no transition to see.
#
# **The two order parameters.**
#
# * **chiral** — `sigma`, the non-strange scalar condensate. It sits at its
#   vacuum value sigma_0 = f_pi = 93 MeV in the broken phase and falls to
#   ~0 when chiral symmetry is restored. `zeta` is its strange partner and does
#   NOT restore, which is the 't Hooft-like statement that the strange
#   condensate survives: it moves from 94.0 to ~64 MeV over the whole scan.
# * **deconfinement** — the dilaton. `Phi = phi_bar^4` is the gluon condensate
#   in units of its vacuum value and `chi = (1 - Phi)^p` is the dielectric, the
#   medium's transparency to colour. Confined is `phi_bar = 1`, `chi = 0`,
#   where `M*_i = (g sigma + m)/chi` diverges and no quark is in the medium;
#   deconfined is `phi_bar < 1`, `chi -> 1`.
#
# The slice is **mu_C = 0**, T = 0, unpaired. That is the standard plane for a
# chiral/deconfinement phase diagram and it is NOT beta equilibrium, which is
# why the transition density here differs from the beta-equilibrium onset the
# tables above report: `thermo_from_mu` takes mu_C as an input rather than
# solving for it, so imposing neutrality would be an outer solve on top.

# %%
MU_B = np.arange(1150.0, 1701.0, 10.0)      # MeV, across the transition
MU_C = 0.0                                   # the slice; NOT beta equilibrium

# One record per branch per potential. Non-convergence is a return value, so a
# branch that has no solution at this mu_B is simply absent from its list --
# below the transition the deconfined branch genuinely does not exist.
transition_scan = {"confined": [], "restored": []}
for branch in transition_scan:
    for mu in MU_B:
        state, ok, err = thermo_from_mu(ccdm.Parameters.default(), float(mu),
                                        MU_C, 0.0, 0.0, branch=branch)
        if ok:
            transition_scan[branch].append(
                dict(mu_B=float(mu), P=state.P, n_B=state.n_B,
                     sigma=state.sigma, zeta=state.zeta,
                     phi_bar=state.Phi ** 0.25, chi=state.chi))

# The transition is where the deconfined pressure overtakes the confined one.
# Both are read from the scan rather than assumed: the confined branch is the
# vacuum and its pressure is zero, but saying so is the check, not the input.
confined = {r["mu_B"]: r for r in transition_scan["confined"]}
deconfined = {r["mu_B"]: r for r in transition_scan["restored"]}
mu_transition = None
for mu in sorted(deconfined):
    if mu in confined and deconfined[mu]["P"] > confined[mu]["P"]:
        mu_transition = mu
        break

print(f"=== ccdm at mu_C = {MU_C} MeV, T = 0, unpaired ===")
print(f"confined branch solved at {len(confined)}/{len(MU_B)} potentials, "
      f"deconfined at {len(deconfined)}/{len(MU_B)}")
if mu_transition is None:
    print("  no crossing on this mu_B range")
else:
    lo = confined[mu_transition]
    hi = deconfined[mu_transition]
    print(f"  transition at mu_B = {mu_transition:.0f} MeV")
    print(f"    n_B      {lo['n_B']:8.4f} -> {hi['n_B']:8.4f} fm^-3   "
          f"({hi['n_B'] / 0.16:.2f} n_sat)")
    print(f"    sigma    {lo['sigma']:8.3f} -> {hi['sigma']:8.3f} MeV   "
          f"(chiral)")
    print(f"    zeta     {lo['zeta']:8.3f} -> {hi['zeta']:8.3f} MeV   "
          f"(strange, does NOT restore)")
    print(f"    phi_bar  {lo['phi_bar']:8.4f} -> {hi['phi_bar']:8.4f}       "
          f"(deconfinement)")
    print(f"    chi      {lo['chi']:8.4f} -> {hi['chi']:8.4f}")
    print("  both order parameters jump at the SAME mu_B: in this model "
          "chiral restoration\n  and deconfinement are one first-order "
          "transition, not two.")


def transition_axes(ylabel):
    """One panel with the transition marked, shared by the two figures."""
    fig, axes = paper_grid("1x1", mode="single", placeholder=False, aspect=1.25)
    ax = axes.ravel()[0]
    if mu_transition is not None:
        ax.axvline(mu_transition, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel(r"$\mu_B$ [MeV]")
    ax.set_ylabel(ylabel)
    return fig, ax


def branch_curve(branch, key):
    rows = transition_scan[branch]
    return (np.array([r["mu_B"] for r in rows]),
            np.array([r[key] for r in rows]))


# %% [markdown]
# ### The chiral order parameter
#
# `sigma` holds at f_pi = 93 MeV on the confined branch and collapses to below
# 1 MeV the moment the deconfined branch takes over — a jump, not a crossover.
# `zeta` drops but stays large: the strange condensate does not restore here.

# %%
fig, ax = transition_axes(r"$\sigma$,  $\zeta$ [MeV]")
for branch, style, width in (("confined", "-", 2.4), ("restored", "-", 2.4)):
    mu, sigma = branch_curve(branch, "sigma")
    mu, zeta = branch_curve(branch, "zeta")
    ax.plot(mu, sigma, color=OKAB_CAT[0], ls=style, lw=width)
    ax.plot(mu, zeta, color=OKAB_CAT[2], ls="--", lw=width)
ax.set_ylim(0.0, 105.0)
njl_style_legend(ax, [(r"$\sigma$", "-"), (r"$\zeta$", "--")],
                 loc="center left", fontsize="x-small")
panel_label(ax, "chiral")
plt.show()

# %% [markdown]
# ### The deconfinement order parameter
#
# The dilaton `phi_bar` sits at 1 on the confined branch — the medium is opaque,
# `chi = 0`, and `M*_i` diverges, which is what "no quark in the medium" means
# here — and drops to below 0.5 across the transition, taking `chi` from 0 to
# ~0.96. The two curves are one statement seen twice, since `chi = 1 -
# phi_bar^4`; both are drawn because it is `phi_bar` that the field equations
# solve for and `chi` that the masses divide by.

# %%
fig, ax = transition_axes(r"$\bar{\phi}$,  $\chi$")
for branch in ("confined", "restored"):
    mu, phi = branch_curve(branch, "phi_bar")
    mu, chi = branch_curve(branch, "chi")
    ax.plot(mu, phi, color=OKAB_CAT[3], ls="-", lw=2.4)
    ax.plot(mu, chi, color=OKAB_CAT[1], ls="--", lw=2.4)
ax.set_ylim(-0.05, 1.15)
njl_style_legend(ax, [(r"$\bar{\phi}$", "-"), (r"$\chi$", "--")],
                 loc="center left", fontsize="x-small")
panel_label(ax, "deconfinement")
plt.show()

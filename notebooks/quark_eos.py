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

# %% [markdown]
# # Quark equations of state — vMIT, alphaBag, NJL, CCDM (and ABPR alongside)
#
# Four models of deconfined quark matter, driven through the public API and
# nothing else: `eos_point`, `eos_table`, `eos_response` and the model's
# parameter and species objects. No solver internal is touched and no helper
# module sits beside this notebook — everything the notebook needs is either in
# the library or in the cells below. It is the quark counterpart of
# `hadronic_eos` and carries the same spine: the same knobs cell, the same
# three-way gap reporting, the same table-naming convention.
#
# **Why `abpr` is here but is not a fifth model in the knobs cell.** ABPR is the
# closed-form T = 0 parametrisation of the colour-flavour-locked phase that
# `alphabag` also carries: it is `cfl`-only and T = 0-only, so as a peer in the
# knobs cell it would refuse every mode and every temperature the other four
# accept, and the output would be a column of refusals rather than physics. It
# appears instead as a **companion panel** (section 7), driven against
# `alphabag`'s CFL phase as the matched pair the two were written to be —
# `alpha_s = pi/2 (1 - a4)` is one knob spelled two ways. That is where its
# narrowness is the subject rather than the noise.
#
# What is here:
#
# 1. **The knobs** — every choice this notebook makes, in one cell.
# 2. **Reporting a gap** — the three distinct things that can happen when a
#    model is asked for something, and why they must stay three.
# 3. **Saving a table** — the automatic name every generated table gets.
# 4. **A section per mode** — the equilibrium conditions of the library,
#    including `cfl`, each exercised through `eos_point` and `eos_table`.
# 5. **The published parameter sets** — parameters are arguments, and two of
#    these four ship named sets.
# 6. **Figures** — the pure-quark planes: `P` vs `n_B`, `eps` vs `P`, the sound
#    speed under the name that says which one it is, and the flavour
#    composition.
# 7. **The ABPR companion panel.**
# 8. **Does a bare quark model give a star?** — answered with a TOV sequence,
#    not with an empty mass–radius panel.
# 9. **Benchmarks** — what each model costs, where its lines do not converge,
#    and what the profile says dominates.
#
# Units are the ones every public boundary uses: densities in fm^-3,
# temperatures and chemical potentials in MeV, pressure and energy density in
# MeV/fm^3.

# %%
import importlib
import sys
from pathlib import Path

import numpy as np

# `eos` is imported from this checkout rather than from site-packages: the
# package is not installed, so the repository root goes on the path first. This
# works whether the notebook is run from `notebooks/` or from the root.
ROOT = Path.cwd()
if not (ROOT / "eos").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from eos.general.table_io import save_table, standard_name, table_path

# %% [markdown]
# ## 1. The knobs
#
# Everything selectable is selectable here and nowhere else; no cell below
# reaches past this one for a number.
#
# `mode` and the fractions are the library's equilibrium conditions. `leptons`
# is orthogonal to them — it says whether neutralizing electrons (and muons, if
# that family is on) are added to a fixed-fraction solve — so it is a field of
# its own and never an entry in `conditions()`. The six species booleans are the
# named degrees of freedom, spelled the same way in every model.
#
# `conditions(mode)` returns only the fractions *that* mode takes: set `Y_S`
# while asking for `fixed_YC` and it is dropped rather than quietly accepted.
#
# Two things are quark-specific. `cfl` is in the mode list, and it is the one
# mode that is not a choice of equilibrium condition but a statement about which
# phase the model describes — only `alphabag` (and `abpr`, in section 7) has it,
# and the other three refuse it, which is section 2's pattern doing its job.
# And the density grid starts well above saturation: these are deconfined
# phases, and one of them (`ccdm`) has a deconfinement onset above `n_B = 1`
# fm^-3 and reports every density below it as unsolved.

# %%
from dataclasses import dataclass, field


@dataclass
class Knobs:
    """Every choice this notebook makes, in one place."""

    # --- which models ---------------------------------------------------
    # `abpr` is deliberately NOT here; see the intro and section 7.
    models: tuple = ("vmit", "alphabag", "njl", "ccdm")

    # --- the equilibria to exercise, and the fractions they take ---------
    modes: tuple = ("beta_eq_neutrinoless", "beta_eq_neutrino_trapped",
                    "fixed_YC", "fixed_YC_YS", "cfl")
    Y_C: float = 0.0                   # fixed_YC, fixed_YC_YS
    Y_S: float = 1.0                   # fixed_YC_YS: one s quark per baryon
    Y_Le: float = 0.3                  # beta_eq_neutrino_trapped
    Y_Lmu: float = None                # beta_eq_neutrino_trapped, optional
    Delta0: float = 100.0              # cfl: the T = 0 pairing gap, MeV
    leptons: bool = True               # orthogonal to the mode

    # --- the grid -------------------------------------------------------
    n_B: tuple = (0.40, 1.60, 13)      # (lo, hi, count), fm^-3
    thermal: str = "T"                 # "T" or "SnB"
    thermal_grid: tuple = (0.0, 10.0, 2)   # MeV, or k_B per baryon

    # --- the sectors ----------------------------------------------------
    # Every one is explicit: a sector that is off is off because its flag is
    # False, never because a coupling happens to vanish. Turning one on that a
    # model has not wired raises, and section 2 below reports that as a refusal.
    species: dict = field(default_factory=lambda: dict(
        hyperons=False, deltas=False, muons=False,
        thermal_mesons=False, thermal_neutrinos=False, photons=True))

    # --- the parameters (they are arguments, never module state) --------
    #   "default"              -> Parameters.default()
    #   ("named", "kunkel")    -> Parameters.named("kunkel")
    parameters: dict = field(default_factory=dict)   # per model; missing = default

    def n_B_grid(self):
        lo, hi, count = self.n_B
        return np.linspace(lo, hi, count)

    def thermal_values(self):
        lo, hi, count = self.thermal_grid
        return np.linspace(lo, hi, count)

    def conditions(self, mode):
        """Only the fractions THIS mode takes, under the library's names."""
        taken = {"beta_eq_neutrinoless": (),
                 "beta_eq_neutrino_trapped": ("Y_Le", "Y_Lmu"),
                 "fixed_YC": ("Y_C",),
                 "fixed_YC_YS": ("Y_C", "Y_S"),
                 "cfl": ("Delta0",)}[mode]
        return {name: getattr(self, name) for name in taken
                if getattr(self, name) is not None}

    def axes(self, mode):
        """The grid as `eos_table`'s `axes` argument, fraction axes included."""
        axes = {"nB": self.n_B_grid(), self.thermal: self.thermal_values()}
        for name, value in self.conditions(mode).items():
            axes[name] = np.array([value])
        return axes


KNOBS = Knobs()
KNOBS

# %% [markdown]
# `Delta0` sits with the fractions because `alphabag`'s `cfl` mode takes it
# there — its `MODE_FRACTIONS["cfl"]` is `("Delta0",)`, so the gap is a
# condition of the mode and may be swept as an axis. That is a divergence from
# the library's own description of `cfl`, which says the locking leaves no free
# fraction to name; in `abpr` the same gap is a **parameter** on the parameter
# object instead. Section 7 drives ABPR through its parameters and alphaBag
# through this knob, which is the only place in the notebook the two spellings
# have to be crossed.

# %% [markdown]
# ## 2. Reporting a gap without presenting it as a result
#
# Asking a model for something can end three different ways, and collapsing them
# would be the single most misleading thing this notebook could do.
#
# * **not supported** — the model refuses the mode, the flag or the
#   parametrisation, and says which. A refusal is the library's contract
#   working, not a defect, and it is never dressed up as a result.
# * **did not converge** — the solve ran and failed to converge. That is a
#   *return value*, not an exception, so no `except` clause ever sees it; it is
#   found by testing `.ok`. Calling it "not supported" would be a lie about the
#   physics.
# * **ok** — there is a number.
#
# `TypeError` is deliberately not caught. An unexpected keyword argument is this
# notebook's own bug, and a broad `except` would file it under "the model does
# not support that", where nobody would ever find it.

# %%
def run(name, call, *args, **kwargs):
    """Call one model's public entry point; report which of three happened.

    Returns `(status, payload)` with status in {"ok", "unsupported",
    "unconverged"}. `NotImplementedError` and `ValueError` are the two a
    refusal uses; anything else is left to propagate.
    """
    try:
        result = call(*args, **kwargs)
    except (NotImplementedError, ValueError) as err:
        print(f"  [{name}] not supported: {err}")
        return "unsupported", None

    if not getattr(result, "ok", True):
        print(f"  [{name}] did not converge: {getattr(result, 'message', '')}")
        return "unconverged", result
    return "ok", result


def header(title, mode=None):
    """One printed header, so a skipped model is visible in the output."""
    if mode is None:
        print(f"\n=== {title} ===")
    else:
        print(f"\n=== {title} — mode={mode} {KNOBS.conditions(mode)} "
              f"leptons={KNOBS.leptons} ===")


def model(name):
    """The model package, by the name used in the knobs cell."""
    return importlib.import_module(f"eos.{name}")


def parameters_for(name):
    """The parameter object the knobs ask for. Parameters are arguments."""
    choice = KNOBS.parameters.get(name, "default")
    if choice == "default":
        return model(name).Parameters.default()
    kind, published = choice
    assert kind == "named", f"unknown parameter choice {choice!r}"
    return model(name).Parameters.named(published)


def flags_for(name):
    """The species flags the knobs ask for; raises where a model has not
    wired one of them, which is the refusal `run` reports."""
    return model(name).SpeciesFlags(**KNOBS.species)


def thermo(point):
    """`(P, eps, s)` from a solved point.

    All four quark models spell the totals `P_total`, `e_total`, `s_total` on
    the point object — the `zl` spelling rather than the `dd2` one. Nothing
    else about the point objects is read by this notebook: the grids below go
    through table rows, whose column names are uniform.
    """
    if hasattr(point, "P"):
        return point.P, point.eps, point.s
    return point.P_total, point.e_total, point.s_total


def lepton_kwargs(mode):
    """`leptons=` where it means something, nothing where it does not.

    The flag says whether neutralizing leptons are added to a fixed-fraction
    solve. Beta equilibrium is *defined* by the leptons, and the locked phase
    is electrically neutral by construction with no electrons at all, so in
    neither case is the flag a choice to be named.
    """
    return {"leptons": KNOBS.leptons} if mode.startswith("fixed_") else {}


# %% [markdown]
# ### What the selected models accept
#
# Each named sector, offered to each model on its own. A model that has not
# wired one refuses at flag construction, before any physics runs; a model that
# has wired it may still fail to converge at the probe density, which is a
# different statement and is printed differently.
#
# The second table below is the one this notebook cannot express in its knobs
# cell: each model's **own** flags, beyond the six the library names, with the
# value they are left at. `alphabag`'s thermal gluons and the pairing sector of
# `njl` and `ccdm` are real sectors that the six-flag knob does not reach, so
# they run at their own defaults throughout — stated here rather than left for a
# reader to discover from a number that does not add up. The pairing sector is
# the subject of the step-by-step NJL/CCDM section, which is a separate ticket.

# %%
PROBE_N_B = 1.0
PROBE_T = 10.0
SIX_FLAGS = tuple(KNOBS.species)

header("species flags")
for name in KNOBS.models:
    print(f" {name}:")
    for flag in SIX_FLAGS:
        one_on = {key: (key == flag) for key in SIX_FLAGS}

        def probe(flag_values=one_on, model_name=name):
            module = model(model_name)
            species = module.SpeciesFlags(**flag_values)
            return module.eos_point(parameters_for(model_name),
                                    "beta_eq_neutrinoless", species,
                                    n_B=PROBE_N_B, T=PROBE_T)

        status, _ = run(f"{name} {flag}", probe)
        if status == "ok":
            print(f"  [{name} {flag}] ok")

# %%
import dataclasses

header("flags each model has beyond the library's six")
for name in KNOBS.models:
    own = [f for f in dataclasses.fields(model(name).SpeciesFlags)
           if f.name not in SIX_FLAGS]
    if not own:
        print(f"  [{name}] none")
        continue
    for f in own:
        print(f"  [{name}] {f.name:20s} left at {f.default}")

# %% [markdown]
# ## 3. Saving a table
#
# Every table generated below can be written to `output/tables/<model>/` under a
# name built from the run itself. Every choice that changes a number is in the
# name — the model, the mode, the mode's fractions, the thermal axis, the
# density axis, the sectors that are on, and `nolep` when the neutralizing
# leptons are off — so two runs cannot collide silently and a folder listing
# says months later how each file was made. The complete metadata still goes
# *inside* the file, through `save_table(meta=...)`.

# %%
example = standard_name("vmit", "fixed_YC", KNOBS.conditions("fixed_YC"),
                        KNOBS.axes("fixed_YC"), KNOBS.species,
                        leptons=KNOBS.leptons)
print(example)
print(table_path("vmit", example))

# %% [markdown]
# ## 4. A section per mode
#
# The equilibrium conditions the library defines, plus the one entry that is not
# an equilibrium condition at all:
#
# | mode | independent variables | meaning |
# |---|---|---|
# | `beta_eq_neutrinoless` | (n_B, T) | beta equilibrium, free-streaming neutrinos, charge neutral |
# | `beta_eq_neutrino_trapped` | (n_B, Y_Le, T) | beta equilibrium with trapped neutrinos |
# | `fixed_YC` | (n_B, Y_C, T) | fixed non-leptonic charge fraction — the simulation-table mode |
# | `fixed_YC_YS` | (n_B, Y_C, Y_S, T) | fixed charge and strangeness |
# | `cfl` | (n_B, T) | colour-flavour-locked quark matter |
#
# `Y_C` is the charge fraction of the strongly-interacting matter only — here
# the quarks — and the leptons are excluded from it; total electric neutrality
# is the separate, additional condition that `leptons=True` imposes. `Y_S` is
# `n_s/n_B` with **S = +1 per s quark**, this repository's sign convention and
# the opposite of the PDG one, so `Y_S = 1` is one strange quark per baryon.
#
# `cfl` is the one mode not available to every model, because it is not a choice
# of equilibrium condition but a statement about which phase the model
# describes: a locked phase has no free charge or strangeness fraction to name.
# The three models that are not that phase refuse it below.
#
# ### One point per model

# %%
POINT_N_B = float(np.median(KNOBS.n_B_grid()))
POINT_T = float(KNOBS.thermal_values()[-1])

for mode in KNOBS.modes:
    header("one point", mode)
    conditions = KNOBS.conditions(mode)
    for name in KNOBS.models:

        def solve(model_name=name, mode=mode, conditions=conditions):
            module = model(model_name)
            return module.eos_point(parameters_for(model_name), mode,
                                    flags_for(model_name),
                                    n_B=POINT_N_B, T=POINT_T,
                                    **conditions, **lepton_kwargs(mode))

        status, result = run(name, solve)
        if status == "ok":
            P, eps, s = thermo(result.point)
            print(f"  [{name}] n_B={POINT_N_B:.3f}  T={POINT_T:.1f}  "
                  f"P={P:9.3f}  eps={eps:9.3f}  s={s:7.4f}")

# %% [markdown]
# ### A table per model
#
# The same modes over the density and temperature grid of the knobs cell. The
# density axis is warm-started inside the library — each solved point seeds the
# next — so a table is not a loop over `eos_point` that the caller could have
# written. `eos_table(..., rows=True)` returns the long format the table writer
# and the structure solver both read, and its column names are the same in every
# model: `n_B, T, P, eps, s, mu_B, mu_C, mu_S, mu_e, Y_C, Y_S, Y_u, Y_d, Y_s,
# Y_e, S_per_B, chi, phase`, with each model adding its own columns beside them.
#
# The knobs cell's `leptons` reaches this entry point under the same rule as the
# single points: named for the fixed-fraction modes, left unsaid for beta
# equilibrium and for the locked phase, where the leptons are not a choice.

# %%
tables = {}
for mode in KNOBS.modes:
    header("a grid", mode)
    axes = KNOBS.axes(mode)
    for name in KNOBS.models:

        def build(model_name=name, mode=mode, axes=axes):
            return model(model_name).eos_table(
                parameters_for(model_name), mode, flags_for(model_name),
                axes, rows=True, **lepton_kwargs(mode))

        status, rows = run(name, build)
        if status != "ok" or not rows:
            continue
        tables[(name, mode)] = rows
        requested = len(KNOBS.n_B_grid()) * len(KNOBS.thermal_values())
        first, last = rows[0], rows[-1]
        print(f"  [{name}] {len(rows):3d}/{requested} rows   "
              f"P {first['P']:8.3f} -> {last['P']:8.3f}   "
              f"eps {first['eps']:8.3f} -> {last['eps']:8.3f}")

# %% [markdown]
# Non-converged points are dropped from their line rather than aborting the
# table, so a row count below the requested count is the table saying which
# points it could not solve — not a silent truncation. `ccdm` is the model where
# that matters most: below its deconfinement onset there is no deconfined phase
# to find, and the rows are missing for a stated physical reason.
#
# One of these written out, under its automatic name:

# %%
SAVE = ("vmit", "fixed_YC")

if SAVE in tables:
    name, mode = SAVE
    filename = standard_name(name, mode, KNOBS.conditions(mode),
                             KNOBS.axes(mode), KNOBS.species,
                             leptons=KNOBS.leptons)
    path = save_table(tables[SAVE], table_path(name, filename),
                      meta={"model": name, "mode": mode,
                            "parameters": parameters_for(name),
                            "species": flags_for(name),
                            **KNOBS.conditions(mode)})
    print("wrote", path)

# %% [markdown]
# ## 5. The published parameter sets
#
# Model parameters are arguments, never module-level constants: every call above
# took a parameter object, and the published sets below are named defaults
# rather than hardcoded values. That is what makes an inference run over the
# couplings possible at all.
#
# The four split two ways. `njl` and `ccdm` ship several published sets and a
# `Parameters.named(...)` to reach them; `vmit` and `alphabag` ship one set each
# and no `named` — their whole parameter content is a handful of numbers (a bag
# constant, a vector coupling or a QCD coupling, the quark masses), so a
# published *set* is not the unit those two are varied in. Both are reached by
# the knobs cell's `parameters` field, and a model with no `named` simply has
# nothing to put in it.

# %%
header("published parameter sets")
for name in KNOBS.models:
    module = model(name)
    sets = list(getattr(module, "PUBLISHED_SETS", {})) or ["default"]
    for published in sets:

        def solve(model_name=name, published=published):
            module = model(model_name)
            par = (module.Parameters.default() if published == "default"
                   else module.Parameters.named(published))
            return module.eos_point(par, "beta_eq_neutrinoless",
                                    flags_for(model_name),
                                    n_B=PROBE_N_B, T=PROBE_T)

        status, result = run(f"{name} {published}", solve)
        if status == "ok":
            P, eps, _ = thermo(result.point)
            print(f"  [{name} {published:15s}] n_B={PROBE_N_B:.3f}  "
                  f"P={P:9.3f}  eps={eps:9.3f}")

# %% [markdown]
# ## 6. Figures
#
# The pure-quark planes. All styling comes from `eos.general.figure_style` and
# nothing else, and every figure is written to `output/quark/`.
#
# The lines are the tables of section 4, re-read rather than re-solved, so a
# figure cannot silently disagree with the numbers printed above it.

# %%
import matplotlib.pyplot as plt

from eos.general import figure_style as fs

fs.set_paper_style(fontsize=10, labelsize=9, legendsize=8)

FIG_DIR = ROOT / "output" / "quark"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLOR = dict(zip(KNOBS.models, fs.OKAB_CAT))
FIG_MODE = "beta_eq_neutrinoless"
COLD = float(KNOBS.thermal_values()[0])
HOT = float(KNOBS.thermal_values()[-1])


def line(name, mode, T, *columns):
    """Columns of one solved line, as arrays. Empty where nothing solved."""
    rows = [r for r in tables.get((name, mode), []) if abs(r["T"] - T) < 1e-9]
    if not rows:
        return tuple(np.empty(0) for _ in columns)
    return tuple(np.array([r[c] for r in rows], dtype=float) for c in columns)


print(f"figures from mode={FIG_MODE}, T = {COLD} and {HOT} MeV, into {FIG_DIR}")

# %% [markdown]
# ### 6.1 Pressure against baryon density
#
# Solid at the cold end of the temperature grid, dashed at the warm end. Where a
# curve starts well inside the axis, that model has no deconfined phase below
# the density it starts at.

# %%
fig, axes = fs.paper_grid("1x2", "double", aspect=1.2, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_P, ax_eP = axes.ravel()

# Colour is the model, linestyle is the temperature, so the legend names four
# things rather than eight: at these temperatures the two lines of one model
# lie almost on top of each other and eight keys would say nothing.
for name in KNOBS.models:
    for T, style in ((COLD, "-"), (HOT, "--")):
        n, P = line(name, FIG_MODE, T, "n_B", "P")
        if n.size:
            ax_P.plot(n, P, style, color=MODEL_COLOR[name],
                      label=(name if T == COLD else None))
ax_P.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_P.set_ylabel(r"$P$ [MeV/fm$^3$]")
ax_P.axhline(0.0, color="0.6", lw=0.6, zorder=0)
ax_P.legend(loc="upper left", bbox_to_anchor=(0.06, 0.98))
fs.apply_style(ax_P, legend=False)
fs.panel_label(ax_P, "(a)", corner='lower right')

# %% [markdown]
# ### 6.2 Energy density against pressure
#
# The plane the structure solver actually integrates: `eps(P)` is the whole of
# what a TOV run reads from a table, and a stiffer equation of state is the one
# that reaches a given pressure at a lower energy density.

# %%
for name in KNOBS.models:
    for T, style in ((COLD, "-"), (HOT, "--")):
        P, eps = line(name, FIG_MODE, T, "P", "eps")
        if P.size:
            ax_eP.plot(P, eps, style, color=MODEL_COLOR[name],
                       label=(f"{name}, T = {T:.0f} MeV" if name == "vmit"
                              else None))
ax_eP.set_xlabel(r"$P$ [MeV/fm$^3$]")
ax_eP.set_ylabel(r"$\epsilon$ [MeV/fm$^3$]")
ax_eP.legend(loc="lower right", title="solid / dashed:")
fs.apply_style(ax_eP, legend=False)
fs.panel_label(ax_eP, "(b)", corner='upper left')

fs.save_figure(fig, str(FIG_DIR / "quark_P_nB_and_eps_P"))
plt.show()

# %% [markdown]
# ### 6.3 The speed of sound, under the name that says which one it is
#
# A second derivative is only defined once one says what is held fixed, so the
# library never returns a bare `cs2`. The four models do not, however, spell the
# answer the same way, and the notebook has to read three keys:
#
# * `njl` and `ccdm` return `cs2_isothermal` and `cs2_adiabatic` — named for the
#   **thermal** variable held, which is the axis the library's own description
#   names.
# * `vmit` and `alphabag` return `cs2_eq` — named for the **composition** axis
#   instead (everything re-equilibrates), leaving the thermal variable unsaid
#   although the derivative is in fact taken at fixed `T`.
#
# The curve below is drawn at the cold end of the temperature grid, `T = 0`,
# where the isothermal and adiabatic sound speeds coincide — which is why the
# three keys can share one axis here and only here. The panel is labelled for
# what was computed, not for the shortest name.
#
# The response is asked for only at the densities where that model's own table
# converged, so a model is never asked to differentiate through a state it could
# not find.

# %%
CS2_KEYS = ("cs2_isothermal", "cs2_eq")

fig, axes = fs.paper_grid("1x3", "double", aspect=1.1, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_cs, ax_frac, ax_lep = axes.ravel()

header("sound speed at T = 0")
for name in KNOBS.models:
    n_grid, = line(name, FIG_MODE, COLD, "n_B")
    n_ok, cs2, key = [], [], None
    for n_B in n_grid:

        def respond(model_name=name, n_B=float(n_B)):
            return model(model_name).eos_response(
                parameters_for(model_name), FIG_MODE, flags_for(model_name),
                frozen="equilibrium", n_B=n_B, T=COLD)

        status, out = run(f"{name} n_B={n_B:.2f}", respond)
        if status != "ok":
            continue
        if not out.get("converged", False):
            # `eos_response` reports non-convergence as a return value too, and
            # its dict carries no `.ok` for `run` to test, so it is read here.
            print(f"  [{name} n_B={n_B:.2f}] did not converge: "
                  f"{out.get('reason', '')}")
            continue
        key = next((k for k in CS2_KEYS if k in out), None)
        if key is None:
            print(f"  [{name}] no sound speed in {sorted(out)}")
            continue
        n_ok.append(float(n_B))
        cs2.append(float(out[key]))
    print(f"  [{name}] {len(n_ok)}/{n_grid.size} responses, key={key!r}")
    if n_ok:
        ax_cs.plot(n_ok, cs2, "-", color=MODEL_COLOR[name], label=name)

ax_cs.axhline(1.0 / 3.0, color="0.6", lw=0.6, ls=":", zorder=0)
ax_cs.text(0.02, 1.0 / 3.0, r"$1/3$", va="bottom", ha="left", fontsize=8,
           color="0.4", transform=ax_cs.get_yaxis_transform())
ax_cs.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_cs.set_ylabel(r"$c_s^2$ (isothermal; $=$ adiabatic at $T=0$)")
ax_cs.set_ylim(0.0, 0.7)
ax_cs.legend(loc="upper left")
fs.apply_style(ax_cs, legend=False)
fs.panel_label(ax_cs, "(a)", corner='lower right')

# %% [markdown]
# ### 6.4 Flavour composition
#
# `Y_u`, `Y_d`, `Y_s` and `Y_e`, each `n_i/n_B`, at the cold end of the
# temperature grid, in two panels because they do not share a scale: the three
# quark fractions each sit near 1 and the electron fraction is five orders
# smaller. Panel (b) is the quark content of one model, since the four together
# would be twelve curves; panel (c) is `Y_e` for all four, which is where they
# genuinely differ. Species colours and linestyles come from
# `figure_style.particle_style`, so a quark curve is the same colour here as in
# any hybrid figure that draws quarks and baryons on one axis, and the printed
# table below is all four models at one density.

# %%
QUARKS = (("u", "Y_u"), ("d", "Y_d"), ("s", "Y_s"))
COMP_MODEL = "vmit"

# The three quark fractions each sit near 1 — they sum to 3 by construction,
# since every baryon is three quarks — so what a reader wants to see is their
# SPREAD, and a decade axis would crush all three onto one line. The electron
# fraction is five orders smaller and cannot share that axis, so it gets its
# own logarithmic panel, where it is worth showing for all four models at once:
# it is the one composition number the models genuinely disagree about.
for species, column in QUARKS:
    n, Y = line(COMP_MODEL, FIG_MODE, COLD, "n_B", column)
    color, dash = fs.particle_style(species)
    ax_frac.plot(n, Y, ls=dash, color=color, label=f"${species}$")
ax_frac.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_frac.set_ylabel(r"$Y_i = n_i/n_B$")
ax_frac.set_ylim(0.80, 1.25)
ax_frac.set_title(f"{COMP_MODEL}, T = {COLD:.0f} MeV", fontsize=9)
ax_frac.legend(loc="lower center", ncol=3)
fs.apply_style(ax_frac, legend=False)
fs.panel_label(ax_frac, "(b)", corner='upper left')

for name in KNOBS.models:
    n, Y_e = line(name, FIG_MODE, COLD, "n_B", "Y_e")
    if n.size:
        ax_lep.plot(n, Y_e, "-", color=MODEL_COLOR[name], label=name)
ax_lep.set_yscale("log")
ax_lep.set_ylim(1e-6, 1e-1)
# The decade labels are written as mathtext rather than left to the log
# formatter: that one emits a U+2212 minus, which the paper style's serif face
# does not carry, and the exponent renders as a hollow box. This is the same
# glyph hazard `figure_style` guards on a linear axis with axes.unicode_minus.
ax_lep.set_yticks([10.0 ** k for k in range(-6, 0)])
ax_lep.set_yticklabels([rf"$10^{{{k}}}$" for k in range(-6, 0)])
ax_lep.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_lep.set_ylabel(r"$Y_e = n_e/n_B$")
ax_lep.legend(loc="lower left")
fs.apply_style(ax_lep, legend=False, minor_ticks=False)
fs.panel_label(ax_lep, "(c)", corner='upper left')

fs.save_figure(fig, str(FIG_DIR / "quark_cs2_and_composition"))
plt.show()

header(f"flavour composition at n_B = {POINT_N_B:.2f} fm^-3, T = {COLD:.0f} MeV")
COMPOSITION = QUARKS + (("e", "Y_e"),)
print(f"  {'model':10s} " + " ".join(f"{s:>9s}" for s, _ in COMPOSITION))
for name in KNOBS.models:
    n, *fractions = line(name, FIG_MODE, COLD, "n_B",
                         *[c for _, c in COMPOSITION])
    if not n.size:
        print(f"  {name:10s} (no solved line)")
        continue
    i = int(np.argmin(np.abs(n - POINT_N_B)))
    print(f"  {name:10s} " + " ".join(f"{f[i]:9.5f}" for f in fractions))

# %% [markdown]
# ## 7. The ABPR companion panel
#
# ABPR is the analytic T = 0 parametrisation of the colour-flavour-locked phase
# that `alphabag` also carries. The two are one physical statement written two
# ways, and they are driven as a **matched pair**: ABPR's pQCD factor `a4` and
# alphaBag's coupling are the same knob under `alpha_s = pi/2 (1 - a4)`, and the
# bag constant, the strange quark mass and the gap are set to the same numbers
# on both sides. What is then left between them is the treatment of the strange
# quark mass: ABPR expands it to `O(m_s^2)`, alphaBag carries it exactly through
# the Fermi integrals, so their difference is the next term of that expansion.
#
# The two spellings that must be crossed here — the gap is a *parameter* of ABPR
# and a *condition* of alphaBag's `cfl` mode, and the coupling is `a4` on one
# side and `alpha` on the other — are the whole of the adaptation. Everything
# else goes through `eos_table`.
#
# **The panel compares at equal density, which is what a table gives.** The
# closed statement — that the whole difference is the `O(m_s^4)` term
# `-m_s^4/(8 pi^2 (hbar c)^3) [9/4 + 3 ln(2 mu/m_s)]` — holds at equal *quark
# chemical potentials*, and equal density is not equal potential: the two models
# reach a given `n_B` at slightly different `mu`, and the pressure difference
# picks up `n dmu` on top of the expansion term. That comparison is made where
# it belongs, in `eos/abpr/verify/run_full_check.py`, which measures the ratio
# to better than 1% over `mu = 350`–`800` MeV. Here the panel shows what a
# density-driven table shows: how far apart the two curves are, which is the
# number a reader choosing between them cares about.

# %%
import eos.abpr as abpr
import eos.alphabag as alphabag

abpr_par = abpr.Parameters.default()
matched = alphabag.Parameters(name="matched_to_abpr", m_u=0.0, m_d=0.0,
                              m_s=abpr_par.m_s, alpha=abpr_par.alpha,
                              B4=abpr_par.B4)
print(f"ABPR:     m_s={abpr_par.m_s} MeV  a4={abpr_par.a4}  "
      f"Delta0={abpr_par.Delta0} MeV  B^(1/4)={abpr_par.B4} MeV")
print(f"alphaBag: m_s={matched.m_s} MeV  alpha_s={matched.alpha:.4f} "
      f"= pi/2 (1 - a4)  B^(1/4)={matched.B4} MeV")

cfl_axes = {"nB": KNOBS.n_B_grid(), "T": np.array([0.0])}

rows_abpr = abpr.eos_table(abpr_par, "cfl", None,
                           dict(cfl_axes, T=[0.0]), rows=True)
rows_alpha = alphabag.eos_table(
    matched, "cfl", alphabag.SpeciesFlags(**KNOBS.species),
    dict(cfl_axes, Delta0=np.array([abpr_par.Delta0])), rows=True)

n_a = np.array([r["n_B"] for r in rows_abpr])
P_a = np.array([r["P"] for r in rows_abpr])
n_b = np.array([r["n_B"] for r in rows_alpha])
P_b = np.array([r["P"] for r in rows_alpha])
assert np.allclose(n_a, n_b), "the two CFL tables are on different densities"

# %%
fig, axes = fs.paper_grid("1x2", "double", aspect=1.2, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_cfl, ax_diff = axes.ravel()

ax_cfl.plot(n_a, P_a, "-", color=fs.OKAB_CAT[0], lw=fs.PHASE_LW["cfl"],
            label=r"abpr (CFL, $m_s^2$ expansion)")
ax_cfl.plot(n_b, P_b, "--", color=fs.OKAB_CAT[1], lw=fs.PHASE_LW["cfl"],
            label=r"alphabag (CFL, $m_s$ exact)")
ax_cfl.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_cfl.set_ylabel(r"$P$ [MeV/fm$^3$]")
ax_cfl.set_title(r"matched pair, $T=0$, $\Delta_0 = %.0f$ MeV"
                 % abpr_par.Delta0)
ax_cfl.legend(loc="upper left")
fs.apply_style(ax_cfl, legend=False)
fs.panel_label(ax_cfl, "(a)", corner='lower right')

ax_diff.plot(n_a, 100.0 * (P_a - P_b) / P_b, "-", color=fs.OKAB_CAT[2])
ax_diff.axhline(0.0, color="0.6", lw=0.6, zorder=0)
ax_diff.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_diff.set_ylabel(r"$(P_{\rm abpr} - P_{\rm alphabag})/P_{\rm alphabag}$ [%]")
fs.apply_style(ax_diff)
fs.panel_label(ax_diff, "(b)", corner='upper right')

fs.save_figure(fig, str(FIG_DIR / "abpr_vs_alphabag_cfl"))
plt.show()

print(f"  P_abpr - P_alphabag runs from {P_a[0] - P_b[0]:+.3f} to "
      f"{P_a[-1] - P_b[-1]:+.3f} MeV/fm^3 over the grid "
      f"({100.0 * (P_a[0] - P_b[0]) / P_b[0]:+.2f}% to "
      f"{100.0 * (P_a[-1] - P_b[-1]) / P_b[-1]:+.2f}%); ABPR is the softer of "
      f"the two at every density, which is the sign the O(m_s^4) term carries.")

# %% [markdown]
# ## 8. Does a bare quark model give a star?
#
# Every model above is a **bare** deconfined phase: no crust, no hadronic
# branch, no mixed phase. Such a phase is self-bound — the pressure reaches zero
# at a finite density, which is the star's surface — so a mass–radius sequence
# can be integrated from the table directly, and there is no need to guess
# whether a panel would be empty. It is cheaper to ask.
#
# The check runs the beta-equilibrium, `T = 0` table through `eos.astro.tov` on
# the rows where `P > 0`, and reports three things: whether the sequence turns
# over at all (a maximum mass is what makes a branch stable up to it), what that
# maximum is, and the radius there. A sequence still rising at the last density
# the model can reach has **no located maximum**, and the honest report is that
# sentence rather than a curve.
#
# The table handed to the solver is checked first for monotone `P` and
# `0 <= c_s^2 <= 1`, which is the gate that belongs before integration rather
# than after: a table that fails it produces a number, not a mass.

# %%
from eos.general.state import EOSTable_for_TOV
from eos.astro.tov import (compute_tov_sequence, find_mmax_precise,
                           generate_ec_logspace)

TOV_MODE = "beta_eq_neutrinoless"
N_CENTRAL = 25

header("bare quark stars")
for name in KNOBS.models:
    n, P, eps = line(name, TOV_MODE, COLD, "n_B", "P", "eps")
    if n.size == 0:
        print(f"  [{name}] no solved T = {COLD:.0f} MeV line to integrate")
        continue

    inside = P > 0.0
    if inside.sum() < 4:
        print(f"  [{name}] only {inside.sum()} rows above P = 0 on this grid: "
              f"no surface is bracketed, so no sequence is integrated")
        continue
    n, P, eps = n[inside], P[inside], eps[inside]

    # The gate of CLAUDE.md section 8, run BEFORE integration.
    cs2 = np.gradient(P, eps)
    if not np.all(np.diff(P) > 0):
        print(f"  [{name}] P is not monotone in n_B on the delivered rows: "
              f"not integrated")
        continue
    if not np.all((cs2 >= 0.0) & (cs2 <= 1.0)):
        print(f"  [{name}] c_s^2 leaves [0, 1] on the delivered rows "
              f"(min {cs2.min():.3f}, max {cs2.max():.3f}): not integrated")
        continue

    table = EOSTable_for_TOV(P=P, epsilon=eps, nB=n)
    e_c = generate_ec_logspace(eps[0] * 1.02, eps[-1] * 0.98, N_CENTRAL)
    sequence = compute_tov_sequence(table, e_c)
    index, _, M_max = find_mmax_precise(sequence)

    if index >= len(sequence) - 1:
        print(f"  [{name}] the sequence is still rising at the last density "
              f"of the knobs cell's grid (M = {M_max:.3f} M_sun, "
              f"R = {sequence[index, 3]:.2f} km): NO maximum mass is located, "
              f"so there is no stable branch to plot. Widen `n_B` in the knobs "
              f"cell to find out whether this model has one further up")
    else:
        print(f"  [{name}] M_max = {M_max:.3f} M_sun at "
              f"R = {sequence[index, 3]:.2f} km "
              f"({index + 1} of {len(sequence)} models on the stable branch)")

# %% [markdown]
# What the numbers say. Where a **bare** quark phase does give a star, it is a
# self-bound object of about eight kilometres with a maximum mass well under
# the 2 M_sun that pulsar timing requires; and the models that do not give one
# on this grid fail in two different ways, which the report keeps apart. One
# runs out of grid before the sequence turns over — widening `n_B` is the
# answer there, not a conclusion about the model. The other has almost no
# positive-pressure range at all on the knobs cell's densities, because its
# deconfinement onset sits so high that the surface is barely bracketed.
#
# None of that is a defect of these four models: it is what a bare deconfined
# phase does, and it is why the models in this notebook are the quark *half* of
# a construction. The other half, and the mass–radius figure that is worth
# drawing, live in the hybrid notebook, where one of these phases is coupled to
# a hadronic one through `eos/mixed`.

# %% [markdown]
# ## 9. Benchmarks
#
# What each of these four models costs, per model and per configuration. Every
# timing here comes from `time`/`timeit` around a **public** call, or out of the
# `progress` callback the table builders already carry — **no timing hook is
# added to library code**, and nothing below reads a solver internal.
#
# Four numbers, and they are four because they answer four different questions:
#
# * **cold point** — one `eos_point` with no warm start, which is what an
#   inference sampler pays per proposal. Best of `BENCH_REPEAT` runs, so it is
#   the cost of the call rather than the cost of the first-ever call in a
#   process (imports, first-touch caches).
# * **warm point** — the per-point cost *inside* a sweep, where each solved
#   point seeds the next: the line's `elapsed_s` divided by its `n_solved`. It
#   is the number that matters for building a table, and it is not the cold
#   number. Where a line has non-converged points their cost is in `elapsed_s`
#   but not in `n_solved`, which inflates this figure — the honest reading,
#   since a table pays for the attempts too.
# * **line wall time** — one full `n_B` line at one temperature and one
#   combination of fractions, straight from the callback's `elapsed_s`.
# * **non-converged** — the count, and the `n_B` where they fall.
#   Non-convergence is a *return value*, so the benchmark counts these and keeps
#   going; it never crashes on them and never reports them as time saved.
#
# The benchmark line spans 0.05 to 3.0 fm^-3, both ends outside where a
# deconfined phase is the physical state. That is deliberate: it is where the
# non-convergence counter reports the real thing rather than a column of zeros,
# and for these models the low end is not an artificial stress — `ccdm` has a
# deconfinement onset and simply has no phase to find below it.
#
# The rows come from `eos_table(..., rows=True)` rather than from
# `rows_from_result`, which `njl`, `ccdm` and `abpr` do not export at package
# level. That is the better spelling anyway, and it is the one all five accept.
#
# **This section is the expensive part of the notebook — about ten minutes, and
# essentially all of it is `ccdm`.** That is not an accident of the grid: it is
# the measurement. The four models span three orders of magnitude in cost, and
# the cell below is where that becomes a number instead of an impression.

# %%
import cProfile
import io
import pstats
import timeit

BENCH_N_B = np.linspace(0.05, 3.0, 24)
BENCH_REPEAT = 3

# (mode, T, the mode's fractions). `leptons` is not in here: it is the knobs
# cell's flag, applied through `lepton_kwargs` exactly as everywhere above.
BENCH_CONFIGS = (("beta_eq_neutrinoless", 0.0, {}),
                 ("fixed_YC", 10.0, {"Y_C": KNOBS.Y_C}))


def bench_line(name, mode, T, conditions):
    """One benchmark row for one model in one configuration.

    Returns None when the model refuses the configuration or the whole line
    fails — a refusal is not a slow result and does not belong in a timing
    table. Everything measured is a public call.
    """
    module = model(name)
    par, species = parameters_for(name), flags_for(name)
    extra = lepton_kwargs(mode)
    n_B_probe = float(np.median(BENCH_N_B))

    def one_point():
        return module.eos_point(par, mode, species, n_B=n_B_probe, T=T,
                                **conditions, **extra)

    status, _ = run(name, one_point)
    if status != "ok":
        return None
    cold_s = min(timeit.repeat(one_point, repeat=BENCH_REPEAT, number=1))

    axes = {"nB": BENCH_N_B, "T": np.array([T])}
    for key, value in conditions.items():
        axes[key] = np.array([value])

    lines = []
    status, rows = run(name, module.eos_table, par, mode, species, axes,
                       rows=True, progress=lines.append, **extra)
    if status != "ok" or not lines:
        return None
    info = lines[-1]          # one temperature, one fraction combination

    solved = {round(float(row["n_B"]), 9) for row in rows}
    missed = [float(x) for x in BENCH_N_B if round(float(x), 9) not in solved]
    return dict(model=name, mode=mode, T=T,
                cold_ms=1e3 * cold_s,
                warm_ms=1e3 * info["elapsed_s"] / max(info["n_solved"], 1),
                line_s=info["elapsed_s"],
                n_solved=info["n_solved"],
                n_requested=info["n_requested"],
                missed=missed)


benchmarks = []
for mode, T, conditions in BENCH_CONFIGS:
    header("benchmark", mode)
    for name in KNOBS.models:
        row = bench_line(name, mode, T, conditions)
        if row is not None:
            benchmarks.append(row)
            print(f"  [{name:8s}] cold {row['cold_ms']:8.3f} ms   "
                  f"warm {row['warm_ms']:8.3f} ms/pt   "
                  f"line {row['line_s']:7.3f} s   "
                  f"{row['n_solved']}/{row['n_requested']} points")

# %% [markdown]
# **What the spread says.** These four are not four variations on one cost. A
# closed-form bag model (`alphabag`) and a bag model with a vector coupling
# (`vmit`) solve a point in a fraction of a millisecond; `njl` pays tens of
# milliseconds for the same point, because its constituent masses come out of a
# gap equation that has to be re-solved at every residual evaluation; and `ccdm`
# pays more again, because on top of the gap equation it *enumerates*
# candidates — the chiral/dielectric branch and the pairing pattern — and solves
# each.
#
# The `warm` column for `ccdm` is the one to read carefully, and it is why cold
# and warm are reported separately rather than as one number. Its cold point is
# around a tenth of a second, but its warm figure is ten seconds and more per
# point — of order a hundred times its own cold cost, where every other model's
# warm and cold numbers sit within a factor of three of each other. Nothing
# about the solved points got slower.
# The line simply spends most of its wall clock on the points it never solves —
# each miss is retried through up to `MAX_BISECT = 6` halved steps back towards
# the last solved point, and every one of those retries enumerates the
# candidates again. Those attempts are in `elapsed_s` and not in `n_solved`.
# That is the honest arithmetic for anyone budgeting a table, since a table
# does pay for the attempts; it is not a per-solved-point cost.

# %% [markdown]
# ### 9.1 Where the line did not converge
#
# The count and the densities, per model and configuration. A model that solved
# every requested point says so; nothing is inferred from a row count alone.

# %%
header("non-converged points")
for row in benchmarks:
    missed = row["missed"]
    label = f"  [{row['model']:8s} {row['mode']:20s} T={row['T']:4.1f}]"
    if not missed:
        print(f"{label} 0 of {row['n_requested']}")
        continue
    shown = ", ".join(f"{x:.3f}" for x in missed[:8])
    more = "" if len(missed) <= 8 else f", ... (+{len(missed) - 8})"
    print(f"{label} {len(missed)} of {row['n_requested']}  "
          f"at n_B = {shown}{more} fm^-3")

# %% [markdown]
# The two models that miss nothing miss nothing at either end, which is worth
# stating rather than leaving to be inferred: `vmit` and `alphabag` return a
# solved point at every one of the 24 requested densities in both
# configurations, including 0.05 fm^-3, where a deconfined phase is not the
# physical state. A bag model has a root there; whether that root means anything
# is a question for the construction that uses it, not for the solver.
#
# The two that do miss, miss in different places, and the distinction is the
# reason the densities are printed and not just the count:
#
# * **`njl` misses one point, at the top** — n_B = 3.0 fm^-3, the last density
#   of the grid, in both configurations. That is the far end of a
#   cutoff-regularized model's domain, not a hole in the middle of it.
# * **`ccdm` misses seven, in an interior band** — 0.178 to 0.948 fm^-3,
#   identically in both configurations. Below its deconfinement onset there is
#   no deconfined phase to find, which its own `table.py` states as physics
#   rather than as a solver limit, and the table reports those points as missing
#   instead of inventing them. Note that the first density, 0.05 fm^-3, *does*
#   solve while the band above it does not — the band is interior on both sides.
#   Whatever the solver finds down there is worth a look before anyone leans on
#   it; the benchmark's job here is to report it, which it does.

# %% [markdown]
# ### 9.2 Bottlenecks
#
# `cProfile` over one representative line — one model, one mode, the same grid
# the timings above used. Top 15 by cumulative time.
#
# It is profiled *after* the benchmark cells, which for these four is a matter
# of first-touch caches rather than of compilation: none of the quark models
# ships a jitted kernel, so a cold profile here does not report a compiler the
# way a `T = 0` hadronic line does.

# %%
PROFILE = ("njl", "beta_eq_neutrinoless", 0.0, {})

profile_model, profile_mode, profile_T, profile_conditions = PROFILE
profile_axes = {"nB": BENCH_N_B, "T": np.array([profile_T])}
for key, value in profile_conditions.items():
    profile_axes[key] = np.array([value])

profiler = cProfile.Profile()
profiler.enable()
model(profile_model).eos_table(parameters_for(profile_model), profile_mode,
                               flags_for(profile_model), profile_axes,
                               **lepton_kwargs(profile_mode))
profiler.disable()

report = io.StringIO()
pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(15)
print(f"=== cProfile — {profile_model} {profile_mode} T={profile_T} ===")
print(report.getvalue())

# %% [markdown]
# **Reading it** (for the default selection, `njl` in beta equilibrium at
# T = 0): the line is a root find whose residual is expensive, and the profile
# says so twice over. Almost the whole cumulative time sits under
# `general/solve.py:solve_system`, and what `solve_system` spends it on is
# `njl/solver.py:residual` -> `_state` -> `njl/thermodynamics.py:state_at` —
# the NJL state itself, the constituent masses and the cutoff-regularized
# integrals that go with them, rebuilt from scratch on every residual call.
#
# The call counts are the other half of the reading: roughly 1600 residual
# evaluations for about 30 attempted points, some 50 per point. That is what a
# **finite-difference** Jacobian over this model's unknown vector costs, and it
# is not a defect of the profile — section 9.3 below confirms that no quark
# model in this repository ships an analytic one. The gap between `njl`/`ccdm`
# and the two bag models is therefore two compounding factors, not one: a
# residual that is itself far more expensive, evaluated many more times per
# solve.

# %% [markdown]
# ### 9.3 Reference against fast backend
#
# Section 9 of the repository's conventions asks for the two flavours side by
# side **where a model ships one**. None of these four does, and that is checked
# below rather than asserted: no `eos/<model>/backends/`, and no backend switch
# on `eos_point`. `vmit` and `alphabag` say so in their own `eos_response`
# docstrings — "no analytic Jacobian in this repository" — and the reference
# NumPy/SciPy path is therefore the only path, which is why every number above
# is a reference number and needs no second column.

# %%
import inspect

header("backends — is there a fast flavour to compare against?")
for name in KNOBS.models + ("abpr",):
    has_dir = (ROOT / "eos" / name / "backends").is_dir()
    switches = [p for p in inspect.signature(model(name).eos_point).parameters
                if p in ("analytic_jac", "backend", "fast", "jit")]
    print(f"  [{name:8s}] backends/ {'present' if has_dir else 'absent':7s}  "
          f"backend switch on eos_point: {', '.join(switches) or 'none'}")

# %% [markdown]
# The nearest thing the quark side does ship is a **pair of models**, not a pair
# of backends, and the distinction matters: `abpr` evaluates the CFL phase in
# closed form where `alphabag` root-finds it through the Fermi integrals, so the
# two differ in physics (the `O(m_s^4)` term of section 7) as well as in cost.
# It is timed here because it is the one place a reader can see what the
# closed-form path buys — labelled for what it is, and never counted as a
# backend-parity check.

# %%
header("closed form against root-finding, at CFL and T = 0")
cfl_bench_axes = {"nB": BENCH_N_B, "T": np.array([0.0])}

for label, call in (
        ("abpr (closed form)",
         lambda: abpr.eos_table(abpr_par, "cfl", None, cfl_bench_axes,
                                rows=True)),
        ("alphabag (root-found)",
         lambda: alphabag.eos_table(
             matched, "cfl", alphabag.SpeciesFlags(**KNOBS.species),
             dict(cfl_bench_axes,
                  Delta0=np.array([abpr_par.Delta0])), rows=True))):
    status, rows = run(label, call)
    if status != "ok":
        continue
    best = min(timeit.repeat(call, repeat=BENCH_REPEAT, number=1))
    print(f"  [{label:22s}] {best * 1e3:8.3f} ms for a {len(BENCH_N_B)}-point "
          f"line, {len(rows)} rows solved")

# %% [markdown]
# About twenty times, over the same 24 densities — and `abpr` solves all 24
# where `alphabag`'s CFL sweep solves 23. That is the shape of the trade the
# two make: the closed form is faster and has no point it can fail to bracket,
# and what it costs is the `O(m_s^4)` term of section 7. Neither number is a
# backend-parity check, and calling it one would be the misreading this cell
# exists to prevent.

# %% [markdown]
# ### 9.4 The summary table
#
# One row per model and configuration, and the same rows written out under the
# naming convention of section 3. The model slot of the name carries the study
# (`quark`) rather than a model, because the table spans all four — the model of
# each row is a column inside it. `missed` is a list and does not survive as a
# table column, so the file keeps the count and the first density; the densities
# themselves are printed in 9.1.
#
# **The `root=` argument is not decoration.** `table_path`'s default root is the
# *relative* path `output/tables`, so a kernel started in `notebooks/` — which is
# what `jupytext --execute` does — writes to `notebooks/output/tables/` instead
# of to the repository's `output/`. That is why `notebooks/output/` exists in
# this tree, and it is why the save in section 4 above lands there. The fix
# belongs in `eos/general/table_io.py`, uniformly for every notebook rather than
# in one of them, so it is reported here and not patched: this cell passes the
# absolute root built from the bootstrap `ROOT`, and prints where it wrote.

# %%
header("summary")
print(f"  {'model':8s} {'mode':22s} {'T':>5s} {'cold ms':>9s} "
      f"{'warm ms/pt':>11s} {'line s':>8s} {'solved':>10s}")
for row in benchmarks:
    print(f"  {row['model']:8s} {row['mode']:22s} {row['T']:5.1f} "
          f"{row['cold_ms']:9.3f} {row['warm_ms']:11.3f} "
          f"{row['line_s']:8.3f} "
          f"{row['n_solved']:4d}/{row['n_requested']:<5d}")

# %%
TABLE_ROOT = str(ROOT / "output" / "tables")

bench_rows = [dict(model=row["model"], mode=row["mode"], T=row["T"],
                   cold_ms=row["cold_ms"], warm_ms=row["warm_ms"],
                   line_s=row["line_s"], n_solved=row["n_solved"],
                   n_requested=row["n_requested"],
                   n_missed=len(row["missed"]),
                   n_B_first_missed=(row["missed"][0] if row["missed"]
                                     else float("nan")))
              for row in benchmarks]

if bench_rows:
    bench_name = standard_name(
        "quark", "benchmark", {},
        {"nB": BENCH_N_B,
         "T": np.array([T for _, T, _ in BENCH_CONFIGS])},
        KNOBS.species, leptons=KNOBS.leptons)
    bench_path = save_table(bench_rows,
                            table_path("quark", bench_name, root=TABLE_ROOT),
                            meta={"study": "quark benchmark",
                                  "models": ",".join(KNOBS.models),
                                  "modes": ",".join(m for m, _, _
                                                    in BENCH_CONFIGS),
                                  "species": KNOBS.species,
                                  "leptons": KNOBS.leptons,
                                  "repeat": BENCH_REPEAT})
    print("wrote", bench_path)

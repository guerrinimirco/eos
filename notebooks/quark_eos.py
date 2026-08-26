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
# 10. **The pairing sector, step by step** — `njl` and `ccdm` with colour
#    superconductivity on: the gaps, the patterns, and which phase wins where.
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
# reader to discover from a number that does not add up. `csc` is the flag
# section 10 turns on; the rest are left where the model puts them throughout.

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
# library never returns a bare `cs2`. All four models spell the key
# `cs2_isothermal`, which names the **thermal** variable the derivative was
# taken at; `njl` and `ccdm` return `cs2_adiabatic` beside it. The composition
# axis is not part of the key — it is the `frozen='equilibrium'` these calls
# pass, under which nothing is held and the composition re-equilibrates.
#
# The curve below is drawn at the cold end of the temperature grid, `T = 0`,
# where the isothermal and adiabatic sound speeds coincide — which is why one
# axis carries both here and only here. The panel is labelled for what was
# computed, not for the shortest name.
#
# The response is asked for only at the densities where that model's own table
# converged, so a model is never asked to differentiate through a state it could
# not find.

# %%
fig, axes = fs.paper_grid("1x3", "double", aspect=1.1, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_cs, ax_frac, ax_lep = axes.ravel()

header("sound speed at T = 0")
for name in KNOBS.models:
    n_grid, = line(name, FIG_MODE, COLD, "n_B")
    n_ok, cs2 = [], []
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
        n_ok.append(float(n_B))
        cs2.append(float(out["cs2_isothermal"]))
    print(f"  [{name}] {len(n_ok)}/{n_grid.size} responses")
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

# %% [markdown]
# ## 10. The pairing sector, step by step: `njl` and `ccdm`
#
# The two models above that condense diquarks, taken apart one step at a time.
# Section 2 printed their seventh flag and left it at its default; this section
# is that flag turned on.
#
# **`csc` is one flag that changes the equations rather than adding a sector.**
# With it off, `njl` is ordinary three-flavour NJL and `ccdm` is the unpaired
# colour-dielectric model: no gap matrix, no Bogoliubov-de Gennes problem, and
# the two colour chemical potentials `mu_3`, `mu_8` are zero identically. With
# it on, the three gaps `Delta_1, Delta_2, Delta_3` become unknowns, colour
# neutrality becomes two more rows of the residual, and the pairing correction
# enters `Omega`, `eps`, `s` and every density.
#
# **A pairing pattern is not a mode.** Which condensates survive is an outcome
# decided by free energy among the enumerated candidates, not a condition a
# caller declares — so it is not in the knobs cell's `modes` and it is not in
# `conditions()`. `patterns=('2SC',)` is a *restriction on the enumeration*,
# which is what draws a branch that is not the ground state; every cell below
# that names one pattern is doing exactly that, and the cells that do not name
# one let the model choose.
#
# The gap matrix, the BdG problem, the pairing correction and the
# Hellmann-Feynman gap kernels are `eos.general.pairing`, shared by both
# models, so the three patterns below mean the same thing in each:
#
# | pattern | free gaps | what pairs |
# |---|---|---|
# | `unpaired` | none | nothing; `Delta = 0` and `mu_3 = mu_8 = 0` |
# | `2SC` | `Delta_3` | u with d, in two colours; the s quark and the third colour stay unpaired |
# | `CFL` | all three | every flavour with every colour, `Delta_1 = Delta_2 ~ Delta_3` |
#
# The five steps: the unpaired model at one point, the same point one pattern
# at a time, the three phases compared, the gap over the `(n_B, T)` plane, and
# the composition, sound speed and phase boundary that go with them.
#
# **Cost.** This section re-solves both models a few hundred times: about five
# minutes on the machine it was written on, against section 9's ten, and most
# of both is `ccdm` — the same three orders of magnitude section 9 measured,
# now paid with a pairing sector on top. Every sweep below is warm-started for
# that reason, which for `njl`'s 2SC line at T = 0 is the difference between
# 0.9 s and 9 s. The two slowest cells are the neutrality map of 10.3a and the
# sweep of 10.3b, in that order.

# %%
# This section's own knobs. The knobs cell above is not touched: `csc` is the
# seventh flag, which section 2 reported that the library's six cannot reach,
# and the grids here are the pairing sector's own — a gap needs densities well
# above the knobs cell's grid before it opens at all.
from eos.general.physics_constants import hc3   # MeV^3 fm^3, the unit bridge

CSC_MODELS = ("njl", "ccdm")
CSC_MODE = "beta_eq_neutrinoless"
CSC_PATTERNS = ("unpaired", "2SC", "CFL")
GAP_PATTERNS = ("2SC", "CFL")   # the two that have a gap to map

CSC_N_B_GRID = np.linspace(1.0, 2.4, 8)         # fm^-3
CSC_T_GRID = np.array([0.0, 20.0, 40.0, 60.0])  # MeV
CSC_MU_B_GRID = np.linspace(1450.0, 2050.0, 7)  # MeV
CSC_CS2_N_B = (1.4, 1.8, 2.2)                   # fm^-3, where the response is taken

# The one-point probe is TAKEN FROM the grid rather than written again: every
# lookup below is keyed by a float, and a probe written as 1.6 beside a grid
# point that is 1.5999999999999999 misses silently.
CSC_N_B = float(CSC_N_B_GRID[3])                # 1.6 fm^-3
CSC_T = float(CSC_T_GRID[0])                    # 0 MeV

PATTERN_COLOR = dict(zip(CSC_PATTERNS, fs.OKAB_CAT))
MODEL_DASH = {"njl": "-", "ccdm": "--"}         # linestyle says which model


def csc_flags(name, csc):
    """The knobs cell's six sectors, plus this section's own `csc`.

    The six named sectors still come from the knobs cell, so nothing below
    silently runs a different physical system from the sections above it.
    """
    return model(name).SpeciesFlags(csc=csc, **KNOBS.species)


def show(title, rows):
    """One block of labelled quantities: symbol, value, unit, what it is."""
    print(f"  {title}")
    for symbol, value, unit, meaning in rows:
        cell = (f"{value:>13.6g}" if isinstance(value, (int, float))
                else f"{value:>13s}")
        print(f"    {symbol:16s} = {cell}  {unit:10s} {meaning}")


def csc_point(name, csc=True, n_B=CSC_N_B, T=CSC_T, pattern=None, x0=None,
              label=None):
    """One `eos_point`, optionally restricted to a single pairing pattern."""
    def solve(name=name):
        return model(name).eos_point(
            parameters_for(name), CSC_MODE, csc_flags(name, csc),
            n_B=float(n_B), T=float(T),
            patterns=(pattern,) if pattern is not None else None, x0=x0)

    return run(label or f"{name} {pattern or 'enumerated'}", solve)


print(f"probe point: n_B = {CSC_N_B} fm^-3, T = {CSC_T} MeV, "
      f"mode = {CSC_MODE}")
print(f"density grid  {CSC_N_B_GRID[0]:.2f} to {CSC_N_B_GRID[-1]:.2f} fm^-3, "
      f"{CSC_N_B_GRID.size} points")
print(f"thermal grid  {[float(t) for t in CSC_T_GRID]} MeV")
print(f"mu_B grid     {CSC_MU_B_GRID[0]:.0f} to {CSC_MU_B_GRID[-1]:.0f} MeV, "
      f"{CSC_MU_B_GRID.size} points")
for name in CSC_MODELS:
    print(f"  [{name}] flags with pairing on: {csc_flags(name, True)}")

# %% [markdown]
# ### 10.1 The model without colour superconductivity
#
# `csc=False`: the parameters, the equations that close the model, and the
# thermodynamics at one `(n_B, T)` point, each quantity labelled with its
# symbol and its units.
#
# **NJL.** The parameters are a cutoff and three dimensionless couplings; the
# constituent masses are *not* parameters but come out of three coupled gap
# equations,
#
#     M_f = m_f - 4 G_S phi_f + 2 K phi_g phi_h        (f, g, h) all different
#
# with `phi_f = <q-bar_f q_f>` the condensate of flavour `f`. The 't Hooft term
# is what ties the three flavours together: `M_u` depends on the strange
# condensate. The residual printed below is what the solver drove to zero, so a
# reader can see the equations are satisfied rather than take it on trust. The
# effective bag constant is likewise derived — a vacuum pressure difference —
# and not an input.

# %%
name = "njl"
par = parameters_for(name)

header(f"{name}, csc=False, at n_B = {CSC_N_B} fm^-3, T = {CSC_T} MeV")
show("parameters (arguments, never module state)", [
    ("Lambda", par.Lambda, "MeV", "three-momentum cutoff"),
    ("G_S Lambda^2", par.GS_Lambda2, "-", "scalar four-fermion coupling"),
    ("K Lambda^5", par.K_Lambda5, "-", "'t Hooft determinant coupling"),
    ("m_u = m_d", par.m_u, "MeV", "current light quark mass"),
    ("m_s", par.m_s, "MeV", "current strange quark mass"),
    ("eta_D = G_D/G_S", par.eta_D, "-", "diquark coupling; used only at csc=True"),
    ("eta_V = G_V/G_S", par.eta_V, "-", "vector coupling"),
    ("vector_form", par.vector_form, "", "how G_V depends on the state"),
])

status, result = csc_point(name, csc=False)
if status == "ok":
    point = result.point
    state = point.state
    show("the gap equations  M_f = m_f - 4 G_S phi_f + 2 K phi_g phi_h", [
        ("phi_u", float(state.phi[0]), "MeV^3", "condensate <u-bar u>"),
        ("phi_d", float(state.phi[1]), "MeV^3", "condensate <d-bar d>"),
        ("phi_s", float(state.phi[2]), "MeV^3", "condensate <s-bar s>"),
        ("M_u", float(state.M[0]), "MeV", "constituent mass"),
        ("M_d", float(state.M[1]), "MeV", "constituent mass"),
        ("M_s", float(state.M[2]), "MeV", "constituent mass"),
        ("max|R_mass|", float(np.max(np.abs(state.mass_residual))), "MeV",
         "largest gap-equation residual at the solution"),
    ])
    show("the grand potential and the thermodynamics", [
        ("Omega", state.Omega / hc3, "MeV/fm^3",
         "= -P of the matter, vacuum-subtracted"),
        ("P (matter)", state.P_fm, "MeV/fm^3", "quarks only, no leptons"),
        ("P", point.P_total, "MeV/fm^3", "matter + leptons + thermal sectors"),
        ("eps", point.e_total, "MeV/fm^3", "energy density"),
        ("s", point.s_total, "fm^-3", "entropy density"),
        ("f = eps - T s", point.f_total, "MeV/fm^3", "free-energy density"),
        ("mu_B", point.mu_B, "MeV", "baryon chemical potential"),
        ("mu_C", point.mu_C, "MeV", "= mu_p - mu_n; beta equilibrium is mu_C + mu_e = 0"),
        ("mu_e", point.mu_e, "MeV", "electron chemical potential"),
        ("n_u", point.n_u, "fm^-3", "flavour densities"),
        ("n_d", point.n_d, "fm^-3", ""),
        ("n_s", point.n_s, "fm^-3", ""),
        ("n_e", point.n_e, "fm^-3", ""),
        ("Y_u", point.Y_u, "-", "= n_u/n_B"),
        ("Y_d", point.Y_d, "-", ""),
        ("Y_s", point.Y_s, "-", ""),
        ("Y_e", point.Y_e, "-", ""),
        ("B_eff^(1/4)", model(name).bag_constant(par) ** 0.25, "MeV",
         "derived vacuum pressure difference, not an input"),
        ("Euler residual", state.euler_residual(), "-",
         "(eps + P - T s - sum_i mu_i n_i)/eps"),
        ("pattern", point.pattern, "", "no pairing sector exists at csc=False"),
        ("Delta", str(tuple(round(float(g), 3) for g in point.Delta)), "MeV",
         "zero by construction here"),
    ])

# %% [markdown]
# **CCDM.** A different closure: four coupled mean-field equations for the
# dilaton `Phi = phi-bar^4`, the two chiral condensates `sigma`, `zeta` and the
# vector field `omega_0`. The dielectric `chi = (1 - Phi)^p` sits in the
# *denominator* of the effective masses,
#
#     M*_u,d = (g_q sigma + m_u,d)/chi ,      M*_s = (g_s zeta + m_s)/chi
#
# so as the condensate reaches its vacuum value the medium goes opaque, the
# masses diverge and the quarks leave it entirely. That is this model's
# confinement, and it is why the chiral/dielectric *branch* is a second thing
# the solver has to enumerate, printed below beside the fields.

# %%
name = "ccdm"
par = parameters_for(name)
derived = par.derived

header(f"{name}, csc=False, at n_B = {CSC_N_B} fm^-3, T = {CSC_T} MeV")
show("parameters (arguments, never module state)", [
    ("f_pi", par.f_pi, "MeV", "pion decay constant; fixes sigma_0"),
    ("f_K", par.f_K, "MeV", "kaon decay constant; fixes zeta_0"),
    ("m_sigma", par.m_sigma, "MeV", "light scalar mass"),
    ("m_zeta", par.m_zeta, "MeV", "strange scalar mass"),
    ("m_phi", par.m_phi, "MeV", "scalar glueball mass (lattice)"),
    ("m_omega", par.m_omega, "MeV", "vector meson mass"),
    ("B_g^(1/4)", par.B_g_quarter, "MeV", "glue bag scale"),
    ("g_q", par.g_q, "-", "quark-sigma coupling"),
    ("g_s", par.g_s, "-", "quark-zeta coupling"),
    ("gbar_omega", par.gbar_omega, "-", "vector coupling"),
    ("G_D", par.G_D, "MeV^-2", "diquark coupling; used only at csc=True"),
    ("Lambda", par.Lambda, "MeV", "pairing cutoff"),
    ("p", par.p, "-", "dielectric exponent, chi = (1 - phi-bar^4)^p"),
])
show("derived from the vacuum, not input", [
    ("sigma_0", derived.sigma_0, "MeV", "light condensate in vacuum"),
    ("zeta_0", derived.zeta_0, "MeV", "strange condensate in vacuum"),
    ("phi_0", derived.phi_0, "MeV", "vacuum dilaton"),
    ("B_eff^(1/4)", model(name).bag_constant(par) ** 0.25, "MeV",
     "= B_g + B_chi; the chiral part is the larger one"),
])

status, result = csc_point(name, csc=False)
if status == "ok":
    point = result.point
    state = point.state
    show("the field equations R_1..R_4 (dilaton, sigma, zeta, omega_0)", [
        ("branch", point.branch, "", "the chiral/dielectric root that won"),
        ("Phi", state.Phi, "-", "dilaton solve variable, = phi-bar^4"),
        ("phi-bar", state.phi_bar, "-", "dilaton, in units of its vacuum value"),
        ("chi", state.chi, "-", "the dielectric, (1 - Phi)^p"),
        ("sigma", state.sigma, "MeV", "light chiral condensate"),
        ("zeta", state.zeta, "MeV", "strange chiral condensate"),
        ("omega_0", state.omega_0, "MeV", "vector mean field"),
        ("Sigma_R", state.Sigma_R, "MeV", "rearrangement self-energy"),
        ("M*_u", float(state.M_star[0]), "MeV", "= (g_q sigma + m_u)/chi"),
        ("M*_d", float(state.M_star[1]), "MeV", ""),
        ("M*_s", float(state.M_star[2]), "MeV", "= (g_s zeta + m_s)/chi"),
        ("max|R_field|", float(np.max(np.abs(state.field_residual))), "-",
         "largest field-equation residual at the solution"),
    ])
    show("the grand potential and the thermodynamics", [
        ("U", state.U / hc3, "MeV/fm^3", "glue potential"),
        ("V", state.V / hc3, "MeV/fm^3", "chiral potential"),
        ("Omega", state.Omega / hc3, "MeV/fm^3", "= -P of the matter"),
        ("P (matter)", state.P_fm, "MeV/fm^3", "quarks and fields, no leptons"),
        ("P", point.P_total, "MeV/fm^3", "matter + leptons + thermal sectors"),
        ("eps", point.e_total, "MeV/fm^3", ""),
        ("s", point.s_total, "fm^-3", ""),
        ("f = eps - T s", point.f_total, "MeV/fm^3", ""),
        ("mu_B", point.mu_B, "MeV", ""),
        ("mu_C", point.mu_C, "MeV", "= mu_p - mu_n"),
        ("mu_e", point.mu_e, "MeV", ""),
        ("n_u", point.n_u, "fm^-3", ""),
        ("n_d", point.n_d, "fm^-3", ""),
        ("n_s", point.n_s, "fm^-3", ""),
        ("n_e", point.n_e, "fm^-3", ""),
        ("Y_u", point.Y_u, "-", "= n_u/n_B"),
        ("Y_d", point.Y_d, "-", ""),
        ("Y_s", point.Y_s, "-", ""),
        ("Y_e", point.Y_e, "-", ""),
        ("Euler residual", state.euler_residual(), "-",
         "(eps + P - T s - sum_i mu_i n_i)/eps"),
        ("pattern", point.pattern, "", "no pairing sector exists at csc=False"),
        ("beyond_cutoff", str(point.beyond_cutoff), "",
         "whether the largest mode potential passed the pairing cutoff"),
    ])

# %% [markdown]
# ### 10.2 The same point with pairing on, one pattern at a time
#
# `csc=True`, and the enumeration restricted to a single candidate so that each
# pattern can be looked at on its own. Three quantities are new and belong to
# the answer rather than to diagnostics:
#
# * **`Delta_1, Delta_2, Delta_3`** — the gaps [MeV]. `Delta_eta` pairs the two
#   flavours and the two colours that `eta` is *not*, so `Delta_3` is the 2SC
#   gap (u with d) and CFL is all three nonzero.
# * **`mu_3, mu_8`** — the colour chemical potentials that colour neutrality
#   fixes. They exist only where pairing does: in an unpaired region `n_3` and
#   `n_8` vanish identically at `mu_3 = mu_8 = 0`, so they are pinned there and
#   never solved for. The generator normalisation is `T_8 = diag(1, 1, -2)/3`;
#   two other normalisations are in circulation and a comparison with a paper
#   has to convert (`eos.general.pairing` documents the factors).
# * **`gapless`** — whether a quasiparticle branch has reached zero. A gapless
#   state is perfectly physical, but *comparing candidates by `Omega` across
#   one is not*, so it is reported rather than silently ranked.
#
# The pairing pieces of the potential are printed beside them: `delta_Omega`,
# the correction that the condensate makes to the quasiparticle spectrum, and
# the condensation cost `sum_eta Delta_eta^2/(4 G_D)`. The correction form is
# used deliberately — it vanishes identically, not merely to quadrature
# accuracy, when `Delta = 0`, which is why the `unpaired` row below reads
# exactly zero.

# %%
for name in CSC_MODELS:
    header(f"{name}, csc=True, one pattern at a time, "
           f"n_B = {CSC_N_B} fm^-3, T = {CSC_T} MeV")
    for pattern in CSC_PATTERNS:
        status, result = csc_point(name, csc=True, pattern=pattern)
        if status != "ok":
            continue
        point = result.point
        state = point.state
        show(f"pattern = {pattern!r}", [
            ("Delta_1", float(point.Delta[0]), "MeV", "pairs d with s"),
            ("Delta_2", float(point.Delta[1]), "MeV", "pairs u with s"),
            ("Delta_3", float(point.Delta[2]), "MeV", "pairs u with d (the 2SC gap)"),
            ("mu_3", point.mu_3, "MeV", "colour potential, T_3 = diag(1,-1,0)/2"),
            ("mu_8", point.mu_8, "MeV", "colour potential, T_8 = diag(1,1,-2)/3"),
            ("gapless", str(point.gapless), "",
             "a quasiparticle branch has reached zero"),
            ("delta_Omega", state.delta_omega / hc3, "MeV/fm^3",
             "the pairing correction alone; exactly 0 when Delta = 0"),
            ("pair cost", state.pair_cost / hc3, "MeV/fm^3",
             "sum_eta Delta_eta^2/(4 G_D)"),
            ("Omega", -state.P_fm, "MeV/fm^3", "= -P of the matter"),
            ("P", point.P_total, "MeV/fm^3", ""),
            ("eps", point.e_total, "MeV/fm^3", ""),
            ("s", point.s_total, "fm^-3", ""),
            ("f = eps - T s", point.f_total, "MeV/fm^3",
             "what the enumeration compares at fixed n_B"),
            ("mu_B", point.mu_B, "MeV", ""),
            ("mu_C", point.mu_C, "MeV", ""),
            ("mu_e", point.mu_e, "MeV", ""),
            ("Y_e", point.Y_e, "-", "= n_e/n_B"),
            ("Euler residual", state.euler_residual(), "-",
             "holds paired as well as unpaired"),
        ])

# %% [markdown]
# **What the three rows say at this point.** The gaps are of order a hundred
# MeV, which is the scale the diquark coupling sets; the colour potentials are
# small but not zero wherever a pattern pairs, and are exactly zero where it
# does not; and `Y_e` collapses towards zero as the pairing gets more
# symmetric, because a phase that pairs every flavour with every colour is
# electrically neutral *without* electrons. The Euler relation holds to machine
# precision in every row — the pairing correction enters `Omega`, `eps` and `s`
# consistently, which is the one check that catches an assembly error in this
# sector.
#
# `f` is the number the enumeration ranks at fixed density, and it is the one
# to compare across the three rows: the mode fixes `n_B`, so the candidates are
# compared by free energy and not by pressure. At fixed `mu_B` the comparison
# is by pressure instead, and the two agree — which is what 10.3 shows.

# %% [markdown]
# ### 10.3 Unpaired against 2SC against CFL
#
# Two comparisons, because there are two ways to hold the state and they answer
# different questions.
#
# #### 10.3a The grand potential at fixed `(mu_B, T)`
#
# At fixed chemical potential the favoured phase is the one with the lowest
# `Omega`, equivalently the highest pressure. Getting this comparison right
# needs one more condition than the pattern: **the matter has to be electrically
# neutral**, or the answer is decided by a charge nobody is paying for. So for
# each pattern the electron potential is tied to the charge potential by beta
# equilibrium, `mu_e = -mu_C`, and `mu_C` is solved from
#
#     n_C(mu_B, mu_C, T) = n_e(-mu_C, T)
#
# with `n_C` the charge of the strongly-interacting matter only — the leptons
# are excluded from it, exactly as in the knobs cell's `Y_C`. `mu_S = 0`
# throughout: these are free-streaming-neutrino beta equilibria, where
# strangeness is not conserved.
#
# The phase surface used here is `thermo_from_mu`, the phase-adapter contract
# both models expose and the surface `eos.mixed` consumes: it maps
# `(mu_B, mu_C, mu_S, T)` and one declared pattern to a block, solving the
# model's own internal self-consistency at those potentials. It is the right
# entry point for a fixed-potential question, and the only one in this section
# that is not `eos_point` / `eos_table` / `eos_response`.
#
# **Two declared restrictions**, both stated rather than left to be noticed.
# `ccdm` also enumerates a chiral/dielectric branch, and the map below declares
# `'restored'` rather than enumerating: above `mu_B ~ 1550` MeV the `restored`
# and `partial` seeds converge to the same root to every printed digit, and
# below it neither converges, so on this grid the branch enumeration has
# nothing to decide. And the confined branch carries no quarks and has
# `Omega = 0` identically, so **a positive `Omega` in the table below means the
# confining vacuum wins there and there is no deconfined phase at all** — that
# is this model's deconfinement transition, not a solver failure.

# %%
from scipy.optimize import brentq

from eos.general.thermodynamics_leptons import electron_thermo

MU_C_BRACKET = (-400.0, 0.0)     # MeV; n_C - n_e is increasing in mu_C

# The neutrality root is located to half an MeV and no further, which is not a
# loosened tolerance but the stationarity of the potential being used: at
# neutrality dOmega/dmu_C = -n_C + n_e = 0, so an error in mu_C costs Omega
# only at second order. Tightening it to 1e-3 MeV changes no printed digit of
# the table below and roughly doubles the cost of this cell.
MU_C_XTOL = 0.5                  # MeV


def phase_block(name, par, mu_B, mu_C, T, pattern):
    """One phase at fixed potentials, through the phase-adapter surface.

    Returns `(state, converged)`. The two models spell the extra argument of
    that surface differently — `njl` takes a cached vacuum solution, `ccdm` a
    declared chiral branch — which is the whole of the adaptation.
    """
    module = model(name)
    if name == "njl":
        state, ok, _ = module.thermo_from_mu(par, mu_B, mu_C, 0.0, T,
                                             pattern=pattern)
    else:
        state, ok, _ = module.thermo_from_mu(par, mu_B, mu_C, 0.0, T,
                                             branch="restored",
                                             pattern=pattern)
    return state, bool(ok)


def neutral_phase(name, par, mu_B, T, pattern):
    """`Omega` [MeV/fm^3] of one pattern at fixed (mu_B, T), made neutral.

    The neutrality root is bracketed rather than iterated from a guess, so a
    failure is a *reported* failure and never a wrong root. CFL is the case
    that needs saying: it is neutral without electrons, so its root sits at
    `mu_C = 0` and no bracket contains a sign change — that is checked first
    and is a physics statement, not a special case for the solver's comfort.
    """
    def charge_excess(mu_C):
        state, _ = phase_block(name, par, mu_B, float(mu_C), T, pattern)
        return state.n_C_fm - electron_thermo(-float(mu_C), T).n

    try:
        if abs(charge_excess(0.0)) < 1.0e-8:
            mu_C = 0.0
        else:
            mu_C = float(brentq(charge_excess, *MU_C_BRACKET,
                                xtol=MU_C_XTOL))
    except (ValueError, RuntimeError) as err:
        return dict(ok=False, reason=str(err)[:60], pattern=pattern)

    state, ok = phase_block(name, par, mu_B, mu_C, T, pattern)
    electrons = electron_thermo(-mu_C, T)
    return dict(ok=ok, pattern=pattern, mu_C=mu_C, state=state,
                Omega=-(state.P_fm + electrons.P),
                Delta=tuple(abs(float(g)) for g in state.Delta),
                gapless=bool(state.gapless), n_B=state.n_B_fm,
                n_e=electrons.n, reason="converged")


# The (mu_B, T) map is built ONCE here: 10.3a reads its T = 0 row and 10.5
# reads the whole of it for the phase boundary.
csc_omega = {}
for name in CSC_MODELS:
    header(f"{name}: Omega per pattern at fixed (mu_B, T), neutral")
    par = parameters_for(name)
    for T in CSC_T_GRID:
        for mu_B in CSC_MU_B_GRID:
            for pattern in CSC_PATTERNS:
                csc_omega[(name, float(T), float(mu_B), pattern)] = (
                    neutral_phase(name, par, float(mu_B), float(T), pattern))
    solved = sum(1 for key, value in csc_omega.items()
                 if key[0] == name and value["ok"])
    total = CSC_T_GRID.size * CSC_MU_B_GRID.size * len(CSC_PATTERNS)
    print(f"  [{name}] {solved}/{total} (mu_B, T, pattern) blocks converged")

# %%
GAP_FLOOR = 1.0e-3               # MeV; below this a gap has closed


def favoured(name, T, mu_B):
    """The pattern with the lowest Omega, among those that converged.

    Where the gaps have closed the three candidates ARE one state and their
    potentials agree to every digit, so which name `min` returns is decided by
    float noise. Such a cell is reported as `unpaired`, which is the statement
    that there is no condensate rather than an arbitrary label for one.
    """
    have = [csc_omega[(name, T, mu_B, p)] for p in CSC_PATTERNS]
    have = [r for r in have if r["ok"]]
    if not have:
        return None
    best = min(have, key=lambda r: r["Omega"])
    if max(best["Delta"]) < GAP_FLOOR:
        return next((r for r in have if r["pattern"] == "unpaired"), best)
    return best


for name in CSC_MODELS:
    header(f"{name}: Omega [MeV/fm^3] at T = {CSC_T:.0f} MeV, neutral matter")
    print(f"  {'mu_B [MeV]':>11s} " +
          " ".join(f"{p:>12s}" for p in CSC_PATTERNS) +
          f" {'favoured':>10s} {'n_B [fm^-3]':>12s} {'max Delta':>10s}")
    for mu_B in CSC_MU_B_GRID:
        cells = []
        for pattern in CSC_PATTERNS:
            entry = csc_omega[(name, CSC_T, float(mu_B), pattern)]
            cells.append(f"{entry['Omega']:12.2f}" if entry["ok"]
                         else f"{'--':>12s}")
        best = favoured(name, CSC_T, float(mu_B))
        if best is None:
            print(f"  {mu_B:11.0f} " + " ".join(cells) + f" {'none':>10s}")
            continue
        tail = ("  (confined vacuum wins: Omega > 0)" if best["Omega"] > 0.0
                else "")
        print(f"  {mu_B:11.0f} " + " ".join(cells) +
              f" {best['pattern']:>10s} {best['n_B']:12.3f} "
              f"{max(best['Delta']):10.2f}" + tail)

# %% [markdown]
# #### 10.3c The same solve against the reference document's own table
#
# `docs/njl_csc_implementation.md` section 6 prints a neutral solve at
# `mu_B = 1500` MeV, `T = 0`, `eta_D = 0.75` and no vector coupling — the
# shipped default parameter set — for the unpaired and 2SC patterns. It is the
# one number-for-number check available to a notebook here, so it is made
# rather than described, and the disagreements are printed beside the
# agreements.
#
# The document is a specification and the code is what runs; where the two
# differ this cell reports the difference and takes the code's number.

# %%
DOC_TABLE = {
    # pattern: (M_u, M_d, M_s, mu_C, mu_8, Delta_3, n_B, P)
    "unpaired": (9.84, 8.55, 265.59, -34.20, 0.00, 0.0, 1.4319, 302.12),
    "2SC": (11.96, 7.65, 243.13, -62.27, -2.46, 95.50, 1.4887, 324.75),
}
DOC_MU_B = 1500.0
DOC_FIELDS = ("M_u", "M_d", "M_s", "mu_C", "mu_8", "Delta_3", "n_B", "P")

header("njl against docs/njl_csc_implementation.md section 6, "
       "mu_B = 1500 MeV, T = 0")
par = parameters_for("njl")
print(f"  parameter set: eta_D = {par.eta_D}, eta_V = {par.eta_V}, "
      f"lambda_UV = {par.lambda_UV} (the document's conditions)")
print(f"  {'pattern':9s} {'quantity':9s} {'document':>10s} {'code':>10s} "
      f"{'difference':>12s}")
for pattern, expected in DOC_TABLE.items():
    entry = neutral_phase("njl", par, DOC_MU_B, 0.0, pattern)
    if not entry["ok"]:
        print(f"  {pattern:9s} did not converge: {entry['reason']}")
        continue
    state = entry["state"]
    got = (float(state.M[0]), float(state.M[1]), float(state.M[2]),
           entry["mu_C"], state.mu_8, float(state.Delta[2]),
           state.n_B_fm, state.P_fm)
    for field, doc_value, code_value in zip(DOC_FIELDS, expected, got):
        print(f"  {pattern:9s} {field:9s} {doc_value:10.4f} "
              f"{code_value:10.4f} {code_value - doc_value:12.4f}")
    print(f"  {pattern:9s} {'residual':9s} {'':>10s} "
          f"{max(abs(float(r)) for r in state.mass_residual):10.2e} "
          f"  gapless={state.gapless}, "
          f"Euler={state.euler_residual():.1e}")

# %% [markdown]
# **What the table says.** In neutral matter CFL has the lowest `Omega` at
# every `mu_B` where a gap survives — the condensation energy of pairing all
# three flavours beats the price of forcing equal flavour densities, and it
# beats it by more than 2SC does. Where the three columns read the same number
# to every digit the gaps have closed: the pattern is a *restriction on the
# enumeration*, so asking for `CFL` above the gap's endpoint returns the
# unpaired state, correctly, rather than failing.
#
# This is the comparison that needs neutrality to be meaningful. Without it the
# question is decided by a charge the phase is not paying for: CFL is neutral
# with no electrons at all, so it starts from a different place than the other
# two, and comparing them at `mu_C = 0` compares two systems rather than two
# phases of one.
#
# #### 10.3b `P` and `eps` against `n_B`, per phase
#
# The other holding: at fixed density the candidates are ranked by free energy
# `f = eps - T s`, which is what `eos_point` does internally, and each phase
# gets its own `P(n_B)` and `eps(n_B)` branch. Restricting the enumeration is
# how a branch that is not the ground state gets drawn at all.
#
# The sweep is warm-started in **both** directions, and that is not only about
# speed. The gap equation has three roots at any Fermi-surface mismatch — zero,
# a barrier maximum, and the physical BCS root — so a Newton solve returns
# whichever one its seed was nearest, silently. Seeding each point from its
# neighbour is what keeps a line on one root:
#
# * along density, each solved point seeds the next. For `njl`'s 2SC line at
#   T = 0 that is the difference between 8 solved points and 6, and between one
#   second and ten;
# * along temperature, each point is seeded from the *same density at the
#   temperature below* in preference to its density neighbour. Without it the
#   `ccdm` 2SC row at T = 40 MeV starts cold, lands on the trivial root at the
#   first density and — being warm-started from there — carries the zero across
#   the whole row, printing a gapless band between two gapped ones. With it the
#   row reads 125 to 133 MeV and the melting is monotone in T, which is what a
#   gap does.

# %%
def csc_sweep(name, patterns=CSC_PATTERNS, n_B_grid=CSC_N_B_GRID,
              T_grid=CSC_T_GRID):
    """`eos_point` over (pattern, T, n_B), continued in both directions.

    Each point is seeded from the same density one temperature down where that
    exists, and from the previous density otherwise. Returns
    {(pattern, T, n_B): point}. A pattern whose gap has closed simply returns
    the unpaired state at that point, which is the correct answer and not a
    failure; a point the solver cannot reach is reported by `run` and left out
    of the dictionary.
    """
    solved = {}
    for pattern in patterns:
        cooler = {}                      # n_B -> the seed one temperature down
        for T in T_grid:
            warmer, x0 = {}, None
            for n_B in n_B_grid:
                seed = cooler.get(float(n_B), x0)
                status, result = csc_point(
                    name, csc=True, n_B=n_B, T=T, pattern=pattern, x0=seed,
                    label=f"{name} {pattern} T={T:.0f} n_B={n_B:.2f}")
                if status == "ok":
                    solved[(pattern, float(T), float(n_B))] = result.point
                    warmer[float(n_B)] = x0 = result.point.x
                else:
                    x0 = None
            cooler = warmer
    return solved


csc_points = {}
for name in CSC_MODELS:
    header(f"{name}: one line per (pattern, T), warm-started along n_B")
    csc_points[name] = csc_sweep(name)
    for pattern in CSC_PATTERNS:
        for T in CSC_T_GRID:
            got = [csc_points[name][(pattern, float(T), float(n))]
                   for n in CSC_N_B_GRID
                   if (pattern, float(T), float(n)) in csc_points[name]]
            if not got:
                print(f"  [{name} {pattern:8s} T={T:4.0f}] no solved point")
                continue
            gaps = [max(abs(float(g)) for g in p.Delta) for p in got]
            print(f"  [{name} {pattern:8s} T={T:4.0f}] "
                  f"{len(got)}/{CSC_N_B_GRID.size} points, "
                  f"max|Delta| {min(gaps):6.2f} to {max(gaps):6.2f} MeV")

# %%
def sweep_line(name, pattern, T, *fields):
    """One (pattern, T) line as arrays: n_B first, then the named fields.

    `Delta` is reported as `max_eta |Delta_eta|`: the sign of a gap is a phase
    convention the gap equation is odd in, and `thermo_from_mu` returns either
    sign, so a magnitude is the only thing that compares across patterns.
    """
    n_B, columns = [], {field: [] for field in fields}
    for n in CSC_N_B_GRID:
        point = csc_points[name].get((pattern, float(T), float(n)))
        if point is None:
            continue
        n_B.append(float(n))
        for field in fields:
            if field == "Delta":
                columns[field].append(max(abs(float(g)) for g in point.Delta))
            else:
                columns[field].append(float(getattr(point, field)))
    return (np.array(n_B),) + tuple(np.array(columns[f]) for f in fields)


header(f"P, eps and f per phase at T = {CSC_T:.0f} MeV, "
       f"n_B = {CSC_N_B:.2f} fm^-3")
print(f"  {'model':6s} {'pattern':9s} {'P':>10s} {'eps':>10s} {'f':>10s} "
      f"{'f - f_unp':>10s} {'mu_B':>9s} {'max Delta':>10s}")
for name in CSC_MODELS:
    reference = csc_points[name].get(("unpaired", CSC_T, CSC_N_B))
    for pattern in CSC_PATTERNS:
        point = csc_points[name].get((pattern, CSC_T, CSC_N_B))
        if point is None:
            print(f"  {name:6s} {pattern:9s} (not solved here)")
            continue
        shift = (point.f_total - reference.f_total) if reference else float("nan")
        print(f"  {name:6s} {pattern:9s} {point.P_total:10.3f} "
              f"{point.e_total:10.3f} {point.f_total:10.3f} {shift:10.3f} "
              f"{point.mu_B:9.2f} "
              f"{max(abs(float(g)) for g in point.Delta):10.2f}")
print("  units: P, eps, f in MeV/fm^3; mu_B and Delta in MeV. A NEGATIVE "
      "f - f_unp is\n  a favoured phase, which is the fixed-density "
      "statement of 10.3a's fixed-mu_B one.")

# %%
fig, axes = fs.paper_grid("1x3", "double", aspect=1.1, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_f, ax_P, ax_eps = axes.ravel()

for name in CSC_MODELS:
    reference = dict()
    n_ref, f_ref = sweep_line(name, "unpaired", CSC_T, "f_total")
    for i, n in enumerate(n_ref):
        reference[float(n)] = f_ref[i]
    for pattern in CSC_PATTERNS:
        n, f_total, P, eps = sweep_line(name, pattern, CSC_T,
                                        "f_total", "P_total", "e_total")
        if not n.size:
            continue
        shift = np.array([f_total[i] - reference.get(float(n[i]), np.nan)
                          for i in range(n.size)])
        ax_f.plot(n, shift, MODEL_DASH[name], color=PATTERN_COLOR[pattern],
                  label=(pattern if name == "njl" else None))
        ax_P.plot(n, P, MODEL_DASH[name], color=PATTERN_COLOR[pattern])
        ax_eps.plot(n, eps, MODEL_DASH[name], color=PATTERN_COLOR[pattern],
                    label=(f"{name}" if pattern == "unpaired" else None))

ax_f.axhline(0.0, color="0.6", lw=0.6, zorder=0)
ax_f.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_f.set_ylabel(r"$f - f_{\rm unpaired}$ [MeV/fm$^3$]")
ax_f.set_title(f"T = {CSC_T:.0f} MeV; below zero is favoured", fontsize=9)
ax_f.legend(loc="lower left", title="pattern:")
fs.apply_style(ax_f, legend=False)
fs.panel_label(ax_f, "(a)", corner='upper right')

ax_P.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_P.set_ylabel(r"$P$ [MeV/fm$^3$]")
fs.apply_style(ax_P)
fs.panel_label(ax_P, "(b)", corner='upper left')

ax_eps.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_eps.set_ylabel(r"$\epsilon$ [MeV/fm$^3$]")
ax_eps.legend(loc="upper left", title="solid / dashed:")
fs.apply_style(ax_eps, legend=False)
fs.panel_label(ax_eps, "(c)", corner='lower right')

for ax in (ax_f, ax_P, ax_eps):
    ax.set_xticks([1.0, 1.5, 2.0])

fs.save_figure(fig, str(FIG_DIR / "csc_phases_vs_nB"))
plt.show()

# %% [markdown]
# Panel (a) is the comparison the enumeration makes, drawn: the free energy of
# each restricted branch measured from the unpaired one at the same density.
# Where a curve is below zero that pattern is the ground state; where it lies
# exactly on zero the gap has not opened and the restricted solve has returned
# the unpaired state.
#
# Panels (b) and (c) are what it does to the equation of state, and the
# direction is worth stating because it is the opposite of the one "pairing
# costs energy" suggests: at a given density a paired branch has a **higher**
# `P` and a **lower** `eps` than the unpaired one, in both models. Condensation
# lowers the free energy, and it does so by lowering `eps` faster than it
# lowers `mu_B n_B`.
#
# **`P` does not rank the phases here, and the table above shows it doing so
# wrongly.** At `n_B = 1.6` fm^-3 `njl`'s 2SC branch carries the higher
# pressure of the two paired ones while CFL carries the lower free energy —
# and CFL is the ground state. The two orderings are allowed to differ because
# the branches sit at different `mu_B` at the same density; ranking by pressure
# is the fixed-`mu_B` statement of 10.3a, not the fixed-`n_B` one.

# %% [markdown]
# ### 10.4 The gap over the `(n_B, T)` plane
#
# `Delta` as a function of both independent variables of the mode, one map per
# pairing pattern per model, from the same sweep as 10.3b. What is mapped is
# `max_eta |Delta_eta|`: the gap equation is odd in `Delta_eta`, so its sign is
# a phase convention and only the magnitude compares across patterns. For 2SC
# that magnitude *is* `Delta_3`; for CFL the three are nearly degenerate and it
# is the largest of them.
#
# The `unpaired` pattern is not mapped, since `Delta = 0` there by
# construction — which the printed table below shows rather than assumes.

# %%
header("max_eta |Delta_eta| [MeV] over the (n_B, T) plane")
for name in CSC_MODELS:
    for pattern in CSC_PATTERNS:
        print(f"  [{name} {pattern}]")
        print(f"    {'T | n_B':>10s} " +
              " ".join(f"{n:7.2f}" for n in CSC_N_B_GRID))
        for T in CSC_T_GRID:
            cells = []
            for n_B in CSC_N_B_GRID:
                point = csc_points[name].get((pattern, float(T), float(n_B)))
                cells.append(f"{'--':>7s}" if point is None else
                             f"{max(abs(float(g)) for g in point.Delta):7.2f}")
            print(f"    {T:10.0f} " + " ".join(cells))

# %%
from matplotlib.colors import LinearSegmentedColormap

# The one colormap this notebook builds, and it is built OUT OF the shared
# palette rather than picked from matplotlib's: `figure_style` is the only
# module that decides colours (CLAUDE.md section 10), and a ramp between two of
# its own colours keeps the choice there rather than here.
GAP_CMAP = LinearSegmentedColormap.from_list(
    "gap", ["#ffffff", fs.OKAB["sky"], fs.OKAB["blue"]])
# White is a CLOSED gap and grey is a point the solver could not reach. They
# are different statements and the map must not print them the same colour.
GAP_CMAP.set_bad(fs.STANDARD_COLORS["Gray"])


def gap_grid(name, pattern):
    """max|Delta| on the full (T, n_B) rectangle, nan where nothing solved."""
    grid = np.full((CSC_T_GRID.size, CSC_N_B_GRID.size), np.nan)
    for i, T in enumerate(CSC_T_GRID):
        for j, n_B in enumerate(CSC_N_B_GRID):
            point = csc_points[name].get((pattern, float(T), float(n_B)))
            if point is not None:
                grid[i, j] = max(abs(float(g)) for g in point.Delta)
    return grid


# One colour scale across all four panels, so a reader comparing `ccdm`'s gaps
# with `njl`'s is comparing lengths and not two differently-stretched rulers.
GAP_MAX = max(np.nanmax(gap_grid(name, pattern))
              for name in CSC_MODELS for pattern in GAP_PATTERNS)

fig, axes = fs.paper_grid("2x2", "double", aspect=1.0, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
panels = axes.ravel()
print(f"  colour scale shared across all four panels: 0 to {GAP_MAX:.2f} MeV")
labels = ("(a)", "(b)", "(c)", "(d)")
index = 0
for name in CSC_MODELS:
    for pattern in GAP_PATTERNS:
        ax = panels[index]
        grid = gap_grid(name, pattern)
        finite = grid[np.isfinite(grid)]
        mesh = ax.pcolormesh(CSC_N_B_GRID, CSC_T_GRID,
                             np.ma.masked_invalid(grid), cmap=GAP_CMAP,
                             shading="nearest", vmin=0.0, vmax=GAP_MAX)
        bar = fig.colorbar(mesh, ax=ax)
        bar.set_label(r"$\max_\eta |\Delta_\eta|$ [MeV]", fontsize=8)
        bar.ax.tick_params(labelsize=8)
        ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
        ax.set_ylabel(r"$T$ [MeV]")
        ax.set_title(f"{name}, {pattern}", fontsize=9)
        fs.apply_style(ax, grid=False, legend=False, minor_ticks=False)
        fs.panel_label(ax, labels[index], corner='upper left')
        print(f"  [{name} {pattern}] mapped "
              f"{int(np.isfinite(grid).sum())}/{grid.size} cells, "
              f"max|Delta| up to "
              f"{(finite.max() if finite.size else float('nan')):.2f} MeV")
        index += 1

fs.save_figure(fig, str(FIG_DIR / "csc_gap_maps"))
plt.show()

# %% [markdown]
# #### The two cuts through those maps
#
# `Delta` against `n_B` at the cold end of the thermal grid, and `Delta` against
# `T` at the probe density. They are the two directions the map is read in, and
# they carry the two statements a map at this resolution cannot make sharply:
# where along the density axis a gap opens, and how far up in temperature it
# survives.

# %%
fig, axes = fs.paper_grid("1x2", "double", aspect=1.2, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_n, ax_T = axes.ravel()

header(f"Delta vs T at n_B = {CSC_N_B:.2f} fm^-3")
for name in CSC_MODELS:
    for pattern in GAP_PATTERNS:
        n, gaps = sweep_line(name, pattern, CSC_T, "Delta")
        if n.size:
            ax_n.plot(n, gaps, MODEL_DASH[name], color=PATTERN_COLOR[pattern],
                      label=(pattern if name == "njl" else None))

        T_ok, gaps_T = [], []
        for T in CSC_T_GRID:
            point = csc_points[name].get((pattern, float(T), CSC_N_B))
            if point is not None:
                T_ok.append(float(T))
                gaps_T.append(max(abs(float(g)) for g in point.Delta))
        if T_ok:
            ax_T.plot(T_ok, gaps_T, MODEL_DASH[name],
                      color=PATTERN_COLOR[pattern],
                      label=(name if pattern == "2SC" else None))
            print(f"  [{name} {pattern:4s}] " +
                  "  ".join(f"T={t:4.0f}: {g:6.2f} MeV"
                            for t, g in zip(T_ok, gaps_T)))

ax_n.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_n.set_ylabel(r"$\max_\eta |\Delta_\eta|$ [MeV]")
ax_n.set_title(f"T = {CSC_T:.0f} MeV", fontsize=9)
ax_n.legend(loc="lower right", title="pattern:")
fs.apply_style(ax_n, legend=False)
fs.panel_label(ax_n, "(a)", corner='upper left')

ax_T.set_xlabel(r"$T$ [MeV]")
ax_T.set_ylabel(r"$\max_\eta |\Delta_\eta|$ [MeV]")
ax_T.set_title(rf"$n_B$ = {CSC_N_B:.2f} fm$^{{-3}}$", fontsize=9)
ax_T.legend(loc="lower left", title="solid / dashed:")
fs.apply_style(ax_T, legend=False)
fs.panel_label(ax_T, "(b)", corner='upper right')

fs.save_figure(fig, str(FIG_DIR / "csc_gap_cuts"))
plt.show()

# %% [markdown]
# ### 10.5 What goes with the gap: composition, sound speed, phase boundary
#
# #### 10.5a Quark and electron fractions, per phase
#
# `Y_i = n_i/n_B` for the three flavours and for the electrons, at the cold end
# of the thermal grid. The three quark fractions sum to 3 by construction — a
# baryon is three quarks — so what distinguishes the phases is how the sum is
# split, and the electron fraction is what pays for the imbalance.
#
# The statement to watch for is CFL's: locking every flavour to every colour
# forces `n_u = n_d = n_s`, so the matter is electrically neutral with **no
# electrons at all**, and `Y_e` drops by orders of magnitude rather than by a
# factor. That is the same fact 10.3a's neutrality solve met from the other
# side, where CFL's root sat at `mu_C = 0`.

# %%
header(f"fractions at T = {CSC_T:.0f} MeV, n_B = {CSC_N_B:.2f} fm^-3")
print(f"  {'model':6s} {'pattern':9s} {'Y_u':>10s} {'Y_d':>10s} {'Y_s':>10s} "
      f"{'Y_u+Y_d+Y_s':>12s} {'Y_e':>12s} {'mu_e [MeV]':>11s}")
for name in CSC_MODELS:
    for pattern in CSC_PATTERNS:
        point = csc_points[name].get((pattern, CSC_T, CSC_N_B))
        if point is None:
            print(f"  {name:6s} {pattern:9s} (not solved here)")
            continue
        print(f"  {name:6s} {pattern:9s} {point.Y_u:10.5f} {point.Y_d:10.5f} "
              f"{point.Y_s:10.5f} {point.Y_u + point.Y_d + point.Y_s:12.5f} "
              f"{point.Y_e:12.3e} {point.mu_e:11.4f}")

# %% [markdown]
# #### 10.5b The sound speed, per phase
#
# `eos_response` at `frozen='equilibrium'` — nothing held, the composition and
# (unless a pattern is named) the pairing pattern both re-equilibrating under
# the perturbation — restricted to one pattern at a time, so the derivative is
# taken *within* a branch instead of across the enumeration. Taking it across
# would give a chord over a first-order jump rather than a tangent, which is a
# number with no meaning.
#
# The key is read the same way section 6.3 reads it: `cs2_isothermal`, with the
# panel labelled for what was computed. At the cold end of the grid it coincides
# with `cs2_adiabatic`, which both models here return beside it.
#
# This is the slowest cell of the section: each response is a re-solved finite
# difference, so it is several full solves per number.

# %%
csc_cs2 = {}
header(f"sound speed per phase at T = {CSC_T:.0f} MeV, frozen='equilibrium'")
for name in CSC_MODELS:
    for pattern in CSC_PATTERNS:
        for n_B in CSC_CS2_N_B:

            def respond(name=name, pattern=pattern, n_B=n_B):
                return model(name).eos_response(
                    parameters_for(name), CSC_MODE, csc_flags(name, True),
                    frozen="equilibrium", n_B=n_B, T=CSC_T,
                    patterns=(pattern,))

            status, out = run(f"{name} {pattern} n_B={n_B:.2f}", respond)
            if status != "ok":
                continue
            if not out.get("converged", False):
                print(f"  [{name} {pattern} n_B={n_B:.2f}] did not converge: "
                      f"{out.get('reason', '')}")
                continue
            csc_cs2[(name, pattern, n_B)] = float(out["cs2_isothermal"])
            print(f"  [{name:5s} {pattern:8s} n_B={n_B:.2f}] "
                  f"cs2_isothermal = {out['cs2_isothermal']:.5f}"
                  + ("   (stencil crossed a branch or pattern change)"
                     if out.get("branch_changed") else ""))

# %%
fig, axes = fs.paper_grid("1x3", "double", aspect=1.1, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_quark, ax_lepton, ax_sound = axes.ravel()

COMP_CSC_MODEL = "njl"
for species, column in QUARKS:
    color, dash = fs.particle_style(species)
    for pattern in CSC_PATTERNS:
        n, Y = sweep_line(COMP_CSC_MODEL, pattern, CSC_T, column)
        if n.size:
            ax_quark.plot(n, Y, ls=dash, color=color,
                          alpha=(1.0 if pattern == "CFL" else 0.45),
                          label=(f"${species}$" if pattern == "CFL" else None))
ax_quark.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_quark.set_ylabel(r"$Y_i = n_i/n_B$")
ax_quark.set_title(f"{COMP_CSC_MODEL}, T = {CSC_T:.0f} MeV; "
                   f"CFL solid, others faded", fontsize=8)
ax_quark.legend(loc="upper right", ncol=3)
fs.apply_style(ax_quark, legend=False)
fs.panel_label(ax_quark, "(a)", corner='upper left')

for name in CSC_MODELS:
    for pattern in CSC_PATTERNS:
        n, Y_e = sweep_line(name, pattern, CSC_T, "Y_e")
        keep = n[Y_e > 0.0], Y_e[Y_e > 0.0]
        if keep[0].size:
            ax_lepton.plot(*keep, MODEL_DASH[name],
                           color=PATTERN_COLOR[pattern],
                           label=(pattern if name == "njl" else None))
ax_lepton.set_yscale("log")
# The floor is set rather than left to the data: CFL's electron fraction is
# numerically zero (order 1e-16, the residue of a cancellation), and an axis
# that reaches down to it crushes the two phases that do carry electrons into
# one line. The curve is off the bottom, which is the statement.
ax_lepton.set_ylim(1.0e-9, 1.0e-1)
ax_lepton.text(0.97, 0.03, r"CFL below $10^{-9}$ is off scale",
               transform=ax_lepton.transAxes, fontsize=7, color="0.35",
               ha="right")
ax_lepton.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_lepton.set_ylabel(r"$Y_e = n_e/n_B$")
fs.log_decades(ax_lepton, axis="y")
ax_lepton.legend(loc="lower left", title="pattern:")
fs.apply_style(ax_lepton, legend=False, minor_ticks=False)
fs.panel_label(ax_lepton, "(b)", corner='upper right')

for name in CSC_MODELS:
    for pattern in CSC_PATTERNS:
        got = [(n, csc_cs2[(name, pattern, n)]) for n in CSC_CS2_N_B
               if (name, pattern, n) in csc_cs2]
        if not got:
            continue
        ax_sound.plot([g[0] for g in got], [g[1] for g in got],
                      MODEL_DASH[name] + "o", color=PATTERN_COLOR[pattern],
                      ms=3, label=(name if pattern == "unpaired" else None))
ax_sound.axhline(1.0 / 3.0, color="0.6", lw=0.6, ls=":", zorder=0)
ax_sound.text(0.02, 1.0 / 3.0, r"$1/3$", va="bottom", ha="left", fontsize=8,
              color="0.4", transform=ax_sound.get_yaxis_transform())
ax_sound.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_sound.set_ylabel(r"$c_s^2$ (isothermal)")
ax_sound.set_ylim(0.0, 0.7)
ax_sound.legend(loc="lower right", title="solid / dashed:")
fs.apply_style(ax_sound, legend=False)
fs.panel_label(ax_sound, "(c)", corner='upper left')

fs.save_figure(fig, str(FIG_DIR / "csc_composition_cs2"))
plt.show()

# %% [markdown]
# #### 10.5c The phase boundary in the `(mu_B, T)` plane
#
# The map of 10.3a, read for its winner instead of its numbers: at every
# `(mu_B, T)` the pattern with the lowest neutral `Omega`. The boundary between
# two colours is the phase boundary, and a cell where the favoured `Omega` is
# **positive** is one where the confining vacuum beats every deconfined
# candidate — no quark phase at all, which for `ccdm` is a statement its own
# branch enumeration makes and for `njl` is a statement about the vacuum
# pressure `B_eff`.
#
# The resolution is the grid of 10.3a and no finer: each cell is a neutrality
# solve for each of three patterns, so the map is coarse on purpose. What it is
# for is the *shape*, and the shape has two edges. The paired region is bounded
# **above in `mu_B`**, where the gap closes because the Fermi surface has run
# out towards the cutoff, and **above in `T`**, where it melts; and the two
# edges are not independent — as `T` rises the high-`mu_B` edge moves down.
#
# The melting edge is checkable against the one number BCS theory gives for
# free, `T_c ~ 0.57 Delta(T = 0)`, so the cell below prints that estimate from
# the gaps of 10.4 beside the temperature at which the map actually loses its
# condensate.

# %%
from matplotlib.colors import ListedColormap

BOUNDARY_COLORS = ListedColormap(
    [PATTERN_COLOR[p] for p in CSC_PATTERNS] + [fs.STANDARD_COLORS["Gray"]])
CONFINED_INDEX = len(CSC_PATTERNS)

fig, axes = fs.paper_grid("1x2", "double", aspect=1.0, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)

header("the favoured pattern over the (mu_B, T) plane")
for ax, name, label in zip(axes.ravel(), CSC_MODELS, ("(a)", "(b)")):
    grid = np.full((CSC_T_GRID.size, CSC_MU_B_GRID.size), np.nan)
    print(f"  [{name}]")
    print(f"    {'T | mu_B':>10s} " +
          " ".join(f"{mu:>10.0f}" for mu in CSC_MU_B_GRID))
    for i, T in enumerate(CSC_T_GRID):
        cells = []
        for j, mu_B in enumerate(CSC_MU_B_GRID):
            best = favoured(name, float(T), float(mu_B))
            if best is None:
                cells.append(f"{'--':>10s}")
                continue
            if best["Omega"] > 0.0:
                grid[i, j] = CONFINED_INDEX
                cells.append(f"{'confined':>10s}")
                continue
            grid[i, j] = CSC_PATTERNS.index(best["pattern"])
            cells.append(f"{best['pattern']:>10s}")
        print(f"    {T:10.0f} " + " ".join(cells))

    ax.pcolormesh(CSC_MU_B_GRID, CSC_T_GRID, grid, cmap=BOUNDARY_COLORS,
                  shading="nearest", vmin=-0.5,
                  vmax=CONFINED_INDEX + 0.5)
    ax.set_xlabel(r"$\mu_B$ [MeV]")
    ax.set_ylabel(r"$T$ [MeV]")
    ax.set_ylim(float(CSC_T_GRID[0]), float(CSC_T_GRID[-1]))
    ax.set_title(f"{name}, neutral matter", fontsize=9)
    fs.apply_style(ax, grid=False, legend=False, minor_ticks=False)
    fs.panel_label(ax, label, corner='upper left')

handles = [plt.Line2D([], [], color=PATTERN_COLOR[p], lw=6, label=p)
           for p in CSC_PATTERNS]
handles.append(plt.Line2D([], [], color=fs.STANDARD_COLORS["Gray"], lw=6,
                          label="confining vacuum"))
# The legend sits on top of a filled panel, so it carries its own opaque
# background: the paper style draws legends unframed, which is unreadable here.
axes.ravel()[-1].legend(handles=handles, loc="upper right", fontsize=7,
                        frameon=True, facecolor="white", framealpha=0.92)

fs.save_figure(fig, str(FIG_DIR / "csc_phase_boundary"))
plt.show()

# %% [markdown]
# ### What section 10 established
#
# * The pairing sector is one flag, and turning it on changes the equations
#   rather than adding a term: three gaps become unknowns, colour neutrality
#   becomes two rows, and `mu_3`, `mu_8` exist only where a pattern pairs.
# * The pattern is an **outcome**, ranked by free energy at fixed density and
#   by pressure at fixed potential; restricting the enumeration is what draws a
#   branch that is not the ground state, and every restricted branch above is a
#   branch, not a mode.
# * In neutral matter CFL is favoured over 2SC and over unpaired wherever a gap
#   survives, in both models, and its signature in the composition is the
#   collapse of `Y_e` — a locked phase is neutral without electrons.
# * Pairing **softens** both models at fixed density: `P` and `eps` are both
#   lower in a paired branch than in the unpaired one at the same `n_B`.
# * The gaps are of order 100 MeV and close from the top of the density range,
#   which is where a cutoff-regularized model runs out of the Fermi surface it
#   is cutting.
#
# Every number above is a bare deconfined phase, with the caveat section 8
# already made: what such a phase is *for* is the quark half of a construction,
# and the hybrid notebook is where it is coupled to a hadronic one.

# %%
# The BCS ratio is not fitted here and not tuned: it is 2 Delta(0)/(k_B T_c) =
# 3.52, the weak-coupling value, quoted as the sanity check it is. A gap of a
# hundred MeV melting at a few tens of MeV is the expected order; agreement to
# a factor of two is all this coarse a grid can claim, and it is all that is
# claimed.
BCS_RATIO = 0.5669           # T_c/Delta(0) at weak coupling

header("melting temperature: BCS estimate against the map")
for name in CSC_MODELS:
    for pattern in GAP_PATTERNS:
        cold = csc_points[name].get((pattern, CSC_T, CSC_N_B))
        if cold is None:
            continue
        gap = max(abs(float(g)) for g in cold.Delta)

        gapped = []
        for T in CSC_T_GRID:
            point = csc_points[name].get((pattern, float(T), CSC_N_B))
            if point is None:
                continue
            if max(abs(float(g)) for g in point.Delta) > GAP_FLOOR:
                gapped.append(float(T))
        if not gapped:
            print(f"  [{name:5s} {pattern:4s}] no gap anywhere on the "
                  f"thermal grid at n_B = {CSC_N_B:.2f} fm^-3")
            continue

        melted = [float(T) for T in CSC_T_GRID if float(T) > max(gapped)]
        where = (f"and none at {min(melted):.0f} MeV" if melted
                 else "(the grid ends before it melts)")
        print(f"  [{name:5s} {pattern:4s}] Delta(T=0) = {gap:6.2f} MeV  ->  "
              f"T_c ~ {BCS_RATIO * gap:6.2f} MeV;  the map still has a gap at "
              f"T = {max(gapped):4.0f} MeV {where}")

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
# # Hadronic equations of state — ZL, SFHo, DD2, DID
#
# Four hadronic models of `eos`, driven through the public API and nothing else:
# `eos_point`, `eos_table` and the model's parameter and species objects. No
# solver internal is touched and no helper module sits beside this notebook —
# everything the notebook needs is either in the library or in the cells below.
#
# What is here:
#
# 1. **The knobs** — every choice this notebook makes, in one cell.
# 2. **Reporting a gap** — the three distinct things that can happen when a
#    model is asked for something, and why they must stay three.
# 3. **Saving a table** — the automatic name every generated table gets.
# 4. **A section per mode** — the equilibrium conditions of the library, each
#    exercised through `eos_point` and `eos_table` across the selected models.
# 5. **Parametrisation** — the published sets, the forward nuclear-matter-
#    parameter map, and the inverse where a model has one.
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

# %%
from dataclasses import dataclass, field


@dataclass
class Knobs:
    """Every choice this notebook makes, in one place."""

    # --- which models ---------------------------------------------------
    models: tuple = ("zl", "sfho", "dd2", "did")

    # --- the equilibria to exercise, and the fractions they take ---------
    modes: tuple = ("beta_eq_neutrinoless", "beta_eq_neutrino_trapped",
                    "fixed_YC", "fixed_YC_YS")
    Y_C: float = 0.1                   # fixed_YC, fixed_YC_YS
    Y_S: float = 0.0                   # fixed_YC_YS
    Y_Le: float = 0.3                  # beta_eq_neutrino_trapped
    Y_Lmu: float = None                # beta_eq_neutrino_trapped, optional
    leptons: bool = True               # orthogonal to the mode

    # --- the grid -------------------------------------------------------
    n_B: tuple = (0.10, 0.80, 12)      # (lo, hi, count), fm^-3
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
    #   ("named", "DD2Y")      -> Parameters.named("DD2Y")
    parameters: dict = field(default_factory=dict)   # per model; missing = default

    # The inverse nuclear-matter map ships OFF. It is a solve of its own with
    # its own failure modes, and section 5 shows what each model does with it
    # when it is on.
    use_nmp_inversion: bool = False

    # The target the inversion is asked for when it is on.
    target_nmp: dict = field(default_factory=lambda: dict(
        n_sat=0.15, E_sat=-16.0, m_eff_ratio=0.60,
        K_sat=240.0, E_sym=32.0, L_sym=50.0))

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
                 "cfl": ()}[mode]
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

    One naming divergence has to be crossed here: `zl` spells the totals
    `P_total`, `e_total`, `s_total` and the other three spell them `P`, `eps`,
    `s`. Nothing else about the point objects is read by this notebook — the
    grids below go through `rows_from_result`, whose column names are uniform.
    """
    if hasattr(point, "P"):
        return point.P, point.eps, point.s
    return point.P_total, point.e_total, point.s_total


# %% [markdown]
# ### What the selected models accept
#
# Each named sector, offered to each model on its own. A model that has not
# wired one refuses at flag construction, before any physics runs; a model that
# has wired it may still fail to converge at the probe density, which is a
# different statement and is printed differently.

# %%
PROBE_N_B = 0.4
PROBE_T = 10.0

header("species flags")
for name in KNOBS.models:
    print(f" {name}:")
    for flag in KNOBS.species:
        one_on = {key: (key == flag) for key in KNOBS.species}

        def probe(flag_values=one_on, model_name=name):
            module = model(model_name)
            species = module.SpeciesFlags(**flag_values)
            return module.eos_point(parameters_for(model_name),
                                    "beta_eq_neutrinoless", species,
                                    n_B=PROBE_N_B, T=PROBE_T)

        status, _ = run(f"{name} {flag}", probe)
        if status == "ok":
            print(f"  [{name} {flag}] ok")

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
example = standard_name("dd2", "fixed_YC", KNOBS.conditions("fixed_YC"),
                        KNOBS.axes("fixed_YC"), KNOBS.species,
                        leptons=KNOBS.leptons)
print(example)
print(table_path("dd2", example))

# %% [markdown]
# ## 4. A section per mode
#
# The equilibrium conditions the library defines for hadronic matter. Each fixes
# which variables are independent:
#
# | mode | independent variables | meaning |
# |---|---|---|
# | `beta_eq_neutrinoless` | (n_B, T) | beta equilibrium, free-streaming neutrinos, charge neutral |
# | `beta_eq_neutrino_trapped` | (n_B, Y_Le, T) | beta equilibrium with trapped neutrinos |
# | `fixed_YC` | (n_B, Y_C, T) | fixed non-leptonic charge fraction — the simulation-table mode |
# | `fixed_YC_YS` | (n_B, Y_C, Y_S, T) | fixed charge and strangeness |
#
# `Y_C` is the charge fraction of the strongly-interacting matter only; the
# leptons are excluded from it, and total electric neutrality is the separate,
# additional condition that `leptons=True` imposes.
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
            extra = dict(conditions)
            # The neutralizing-lepton flag applies to the fixed-fraction modes
            # and to those only: beta equilibrium is defined by the leptons, so
            # naming the flag there is a contradiction rather than a choice.
            if mode.startswith("fixed_"):
                extra["leptons"] = KNOBS.leptons
            return module.eos_point(parameters_for(model_name), mode,
                                    flags_for(model_name),
                                    n_B=POINT_N_B, T=POINT_T, **extra)

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
# written. `rows_from_result` flattens the result into the long format the table
# writer and the structure solver both read, and its column names are the same
# in every model.
#
# The knobs cell's `leptons` reaches this entry point too, under the same rule
# as the single points: it is named for the fixed-fraction modes and left unsaid
# for beta equilibrium, where the leptons are what the equilibrium is about.

# %%
tables = {}
for mode in KNOBS.modes:
    header("a grid", mode)
    axes = KNOBS.axes(mode)
    for name in KNOBS.models:

        def build(model_name=name, mode=mode, axes=axes):
            module = model(model_name)
            extra = ({"leptons": KNOBS.leptons} if mode.startswith("fixed_")
                     else {})
            return module.eos_table(parameters_for(model_name), mode,
                                    flags_for(model_name), axes, **extra)

        status, result = run(name, build)
        if status != "ok":
            continue
        rows = model(name).rows_from_result(result)
        tables[(name, mode)] = rows
        requested = len(KNOBS.n_B_grid()) * len(KNOBS.thermal_values())
        first, last = rows[0], rows[-1]
        print(f"  [{name}] {len(rows):3d}/{requested} rows   "
              f"P {first['P']:8.3f} -> {last['P']:8.3f}   "
              f"eps {first['eps']:8.3f} -> {last['eps']:8.3f}")

# %% [markdown]
# Non-converged points are dropped from their line rather than aborting the
# table, so a row count below the requested count is the table saying which
# points it could not solve — not a silent truncation.
#
# One of these written out, under its automatic name:

# %%
SAVE = ("dd2", "fixed_YC")

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
# ## 5. Parametrisation
#
# Model parameters are arguments, never module-level constants: every call above
# took a parameter object, and the published sets below are named defaults
# rather than hardcoded values. That is what makes an inference run over the
# couplings possible at all.
#
# ### The published sets

# %%
# There is no uniform way to ask a model what it ships: `sfho` exposes a
# PUBLISHED_SETS mapping, `dd2` and `did` keep theirs inside `Parameters.named`,
# and `zl` has one set and no `named` at all. So the sets to show are a knob.
PUBLISHED = {"zl": ("default",),
             "sfho": ("SFHo_Nucleonic", "SFHoY_Fortin", "SFHo_2fam"),
             "dd2": ("DD2", "DD2Y"),
             "did": ("DID", "DIDY")}

header("published parameter sets")
for name in KNOBS.models:
    module = model(name)
    for published in PUBLISHED[name]:

        def solve(model_name=name, published=published):
            module = model(model_name)
            par = (module.Parameters.default() if published == "default"
                   else module.Parameters.named(published))
            return module.eos_point(par, "beta_eq_neutrinoless",
                                    flags_for(model_name),
                                    n_B=0.6, T=PROBE_T)

        status, result = run(f"{name} {published}", solve)
        if status == "ok":
            P, eps, _ = thermo(result.point)
            print(f"  [{name} {published:15s}] n_B=0.600  "
                  f"P={P:9.3f}  eps={eps:9.3f}")

# %% [markdown]
# Within a model the sets agree to every digit printed — because the sets of one
# model differ only in the couplings of sectors that the knobs cell has switched
# off. A parametrisation is not a knob that changes the nucleonic answer; it is
# the set of couplings the sectors are read through, and with the hyperons and
# the deltas off there is nothing for the extra couplings to act on. Turn a
# sector on and the sets part company, which is the next cell.
#
# The parameters and the species flags are therefore not independent, and the
# four models split two ways on how. In `sfho` and `dd2` a hyperonic sector
# needs a parametrisation whose hyperon couplings were fitted, and asking a
# nucleonic set for hyperons is refused rather than answered with couplings
# nobody published. In `did` the hyperon couplings were fitted with the rest, so
# `DID` and `DIDY` are the same numbers and the flag alone selects the sector.
# `zl` has no hyperons at all: the functional is written in the neutron and
# proton densities, so the sector is absent from the model rather than
# unimplemented.

# %%
header("hyperons and the parameter set")
HYPERONIC = {"sfho": "SFHoY_Fortin", "dd2": "DD2Y"}
with_hyperons = dict(KNOBS.species, hyperons=True)

for name in KNOBS.models:
    sets = ["default"]
    if name in HYPERONIC:
        sets.append(HYPERONIC[name])
    for published in sets:

        def solve(model_name=name, published=published):
            module = model(model_name)
            par = (module.Parameters.default() if published == "default"
                   else module.Parameters.named(published))
            species = module.SpeciesFlags(**with_hyperons)
            return module.eos_point(par, "beta_eq_neutrinoless", species,
                                    n_B=0.6, T=PROBE_T)

        status, result = run(f"{name} {published}", solve)
        if status == "ok":
            P, eps, _ = thermo(result.point)
            print(f"  [{name} {published:15s}] hyperons on  n_B=0.600  "
                  f"P={P:9.3f}  eps={eps:9.3f}")

# %% [markdown]
# ### The nuclear-matter parameters, forward
#
# `compute_nmp` maps couplings to the properties of nuclear matter at
# saturation. The higher derivatives a model does not impose — `Q_sat` and
# `K_sym` — come back as **predictions**: they are what the parametrisation
# happens to give, not targets it was fitted to.

# %%
header("compute_nmp")
for name in KNOBS.models:
    nmp_module = importlib.import_module(f"eos.{name}.nmp")
    status, values = run(name, nmp_module.compute_nmp, parameters_for(name))
    if status != "ok":
        continue
    print(f" [{name}]")
    for key, value in values.items():
        print(f"   {key:14s} {float(value):12.5f}")

# %% [markdown]
# The keys are not spelled alike in all four: three models return the standard
# list (`n_sat`, `E_sat`, `K_sat`, `Q_sat`, `E_sym`, `L_sym`, `K_sym`) and `did`
# returns its own names (`n_0`, `B`, `K`, `Q`, `M`, `S_2`, `L_2`, `K_sym2`, and
# a full-step `S`, `L`, `K_sym` beside the quadratic ones). Read `did`'s row
# against its own document before comparing it with the others.
#
# ### The nuclear-matter parameters, inverse
#
# The inverse map builds a parametrisation *from* a set of nuclear-matter
# parameters. It is not available everywhere, and where it is missing the reason
# is physics rather than an oversight:
#
# * **`dd2`** inverts, closing the isoscalar sector with the model's own
#   structural conditions.
# * **`sfho`** inverts, and returns a status alongside the parameters: a target
#   the functional form cannot represent comes back as a failure to score, not
#   as an exception.
# * **`zl`** refuses. Six parameters against the five nuclear-matter parameters
#   of the standard list leaves a one-parameter family, and nothing published
#   singles out a member of it. So `zl` shows its nuclear-matter parameters as
#   computed predictions, above, and cannot be built *from* a set of them.
# * **`did`** carries the forward map only.
#
# Turn `use_nmp_inversion` on in the knobs cell to run it.

# %%
header("invert_nmp")
print(f" target: {KNOBS.target_nmp}")

if not KNOBS.use_nmp_inversion:
    print(" use_nmp_inversion is off; the calls below are skipped")
else:
    # The two inversions do not share a calling convention — dd2 takes the
    # nuclear-matter parameters as one dictionary, sfho expands them as keyword
    # arguments — so the two calls are written out rather than looped over.
    dd2_nmp = importlib.import_module("eos.dd2.nmp")
    status, out = run("dd2", dd2_nmp.invert_nmp, KNOBS.target_nmp)
    if status == "ok":
        par, inversion = out
        print(f"  [dd2] {'recovered' if par is not None else 'no parameters'}"
              f" — {getattr(inversion, 'message', inversion)}")
        if par is not None:
            print("   predicted:", {k: round(float(v), 4) for k, v in
                                    dd2_nmp.compute_nmp(par).items()
                                    if k in ("Q_sat", "K_sym")})

    sfho_nmp = importlib.import_module("eos.sfho.nmp")
    status, out = run("sfho", sfho_nmp.invert_nmp, **KNOBS.target_nmp)
    if status == "ok":
        par, inversion = out
        print(f"  [sfho] {'recovered' if par is not None else 'no parameters'}"
              f" — {getattr(inversion, 'message', inversion)}")
        if par is not None:
            print("   predicted:", {k: round(float(v), 4) for k, v in
                                    sfho_nmp.compute_nmp(par).items()
                                    if k in ("Q_sat", "K_sym")})

    zl_nmp = importlib.import_module("eos.zl.nmp")
    run("zl", zl_nmp.invert_nmp, **KNOBS.target_nmp)

    print("  [did] no inverse map: the forward map only")

# %% [markdown]
# ## 6. Benchmarks
#
# What a model costs, per model and per configuration. Every timing here comes
# from `time`/`timeit` around a public call, or out of the `progress` callback
# the table builders already carry — **no timing hook is added to library
# code**, and nothing below reads a solver internal.
#
# Four numbers, and they are four because they answer different questions:
#
# * **cold point** — one `eos_point` with no warm start, which is what an
#   inference sampler pays per proposal. Best of `BENCH_REPEAT` runs, so it is
#   the cost of the call and not the cost of the first-ever call in a process
#   (imports, JIT, first-touch caches).
# * **warm point** — the per-point cost *inside* a sweep, where each solved
#   point seeds the next: the line's `elapsed_s` divided by its `n_solved`.
#   It is the number that matters for building a table, and it is not the cold
#   number. Where a line has non-converged points their cost is in `elapsed_s`
#   but not in `n_solved`, which inflates this figure — the honest reading,
#   since a table pays for the attempts too.
# * **line wall time** — one full `n_B` line at one temperature and one
#   combination of fractions, straight from the callback.
# * **non-converged** — the count, and the `n_B` where they fall.
#   Non-convergence is a *return value*, so the benchmark counts these and
#   keeps going; it never crashes on them and never reports them as time saved.
#
# The benchmark line is deliberately wider than a production table — down to
# 0.002 and up to 3.0 fm^-3, both ends outside where uniform matter is the
# physical state — so that the non-convergence counter reports the real thing
# rather than a column of zeros.

# %%
import cProfile
import io
import pstats
import timeit

BENCH_N_B = np.linspace(0.002, 3.0, 64)
BENCH_REPEAT = 3

# (mode, T, the mode's fractions). `leptons` is not in here: it is the knobs
# cell's flag and is applied to the fixed-fraction modes exactly as elsewhere.
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
    extra = {"leptons": KNOBS.leptons} if mode.startswith("fixed_") else {}
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
    status, result = run(name, module.eos_table, par, mode, species, axes,
                         progress=lines.append, **extra)
    if status != "ok":
        return None
    info = lines[-1]          # one temperature, one fraction combination
    rows = module.rows_from_result(result)

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
            print(f"  [{name}] cold {row['cold_ms']:7.3f} ms   "
                  f"warm {row['warm_ms']:7.3f} ms/pt   "
                  f"line {row['line_s']:6.3f} s   "
                  f"{row['n_solved']}/{row['n_requested']} points")

# %% [markdown]
# ### Where the line did not converge
#
# The count and the densities, per model and configuration. A model that
# solved every requested point says so; nothing is inferred from a row count
# alone.

# %%
header("non-converged points")
for row in benchmarks:
    missed = row["missed"]
    label = f"  [{row['model']:5s} {row['mode']:20s} T={row['T']:4.1f}]"
    if not missed:
        print(f"{label} 0 of {row['n_requested']}")
        continue
    shown = ", ".join(f"{x:.3f}" for x in missed[:8])
    more = "" if len(missed) <= 8 else f", ... (+{len(missed) - 8})"
    print(f"{label} {len(missed)} of {row['n_requested']}  "
          f"at n_B = {shown}{more} fm^-3")

# %% [markdown]
# ### Bottlenecks
#
# `cProfile` over one representative line — one model, one mode, the same grid
# the timings above used. Top 15 by cumulative time.
#
# The default selection is a **finite-T** line, and it is profiled *after* the
# benchmark cells above have run. Both matter. A T = 0 line takes the model's
# jitted T = 0 kernel, and profiling it in a fresh process reports the Numba
# compilation — `llvmlite`, `install_registry`, `marshal.loads` at the top of
# the list — rather than any physics. Run cold and the profile describes the
# compiler; run warm and it describes the solve.

# %%
PROFILE = ("dd2", "beta_eq_neutrinoless", 10.0, {})

profile_model, profile_mode, profile_T, profile_conditions = PROFILE
profile_axes = {"nB": BENCH_N_B, "T": np.array([profile_T])}
for key, value in profile_conditions.items():
    profile_axes[key] = np.array([value])
profile_extra = ({"leptons": KNOBS.leptons}
                 if profile_mode.startswith("fixed_") else {})

profiler = cProfile.Profile()
profiler.enable()
model(profile_model).eos_table(parameters_for(profile_model), profile_mode,
                               flags_for(profile_model), profile_axes,
                               **profile_extra)
profiler.disable()

report = io.StringIO()
pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(15)
print(f"=== cProfile — {profile_model} {profile_mode} T={profile_T} ===")
print(report.getvalue())

# %% [markdown]
# **Reading it** (for the default selection, `dd2` in beta equilibrium at
# T = 10 MeV): the line is root-finding around integral evaluation. Most of the
# cumulative time sits under MINPACK's `hybrj`, and what `hybrj` spends it on is
# `residual` — whose own cost is `kinetic_thermo` and, below that,
# `solve_fermi_jel`, the JEL Fermi integrals of `eos.general.fermi_integrals`,
# which is also the largest single entry by *internal* time. The analytic
# Jacobian of `backends/` is a comparable per-call cost to the residual it
# differentiates, which is why it buys less here than the evaluation count
# suggests, and why `sfho` leaves its own off. Profiling another model or mode
# moves the balance; the top of the list says which.
#
# ### Reference against fast backend
#
# `dd2` is the only one of these four whose fast backend is reachable from the
# public API: `eos_point` takes `analytic_jac`, and `False` selects the
# finite-difference reference. Deleting `backends/` changes no number, only
# speed — which is what the two columns below measure.
#
# The other three: `sfho` ships an analytic Jacobian but leaves it off, and its
# own docstring says why — it cuts residual evaluations per point but each
# finite-T kinetic derivative costs four JEL evaluations, so the wall clock
# moves the wrong way; what it is for is the second-derivative quantities,
# which need `dR/dx` itself rather than a faster root. `zl` and `did` ship no
# `backends/` at all, so there is one path and nothing to compare.

# %%
header("backends — dd2, reference vs fast")
BACKEND_N_B = 0.4
BACKEND_T = 10.0

for label, analytic_jac in (("reference (finite difference)", False),
                            ("fast (analytic Jacobian)", True)):

    def one_point(analytic_jac=analytic_jac):
        return model("dd2").eos_point(parameters_for("dd2"),
                                      "beta_eq_neutrinoless", flags_for("dd2"),
                                      n_B=BACKEND_N_B, T=BACKEND_T,
                                      analytic_jac=analytic_jac)

    status, _ = run(f"dd2 {label}", one_point)
    if status == "ok":
        best = min(timeit.repeat(one_point, repeat=BENCH_REPEAT, number=1))
        print(f"  [dd2 {label:28s}] {best * 1e3:7.3f} ms per cold point")

# %% [markdown]
# The table path is not affected by that choice: `dd2`'s warm-started sweep
# takes the analytic Jacobian by default, so the `warm` column of the summary
# is already the fast backend.
#
# ### The summary table
#
# One row per model and configuration, and the same rows written out under the
# naming convention of section 3. The model slot of the name carries the study
# (`hadronic`) rather than a model, because the table spans all four — the
# model of each row is a column inside it. `missed` is a list and does not
# survive as a table column, so the file keeps the count and the first density;
# the densities themselves are printed above.

# %%
header("summary")
print(f"  {'model':6s} {'mode':22s} {'T':>5s} {'cold ms':>9s} "
      f"{'warm ms/pt':>11s} {'line s':>8s} {'solved':>10s}")
for row in benchmarks:
    print(f"  {row['model']:6s} {row['mode']:22s} {row['T']:5.1f} "
          f"{row['cold_ms']:9.3f} {row['warm_ms']:11.3f} "
          f"{row['line_s']:8.3f} "
          f"{row['n_solved']:4d}/{row['n_requested']:<5d}")

# %%
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
        "hadronic", "benchmark", {},
        {"nB": BENCH_N_B,
         "T": np.array([T for _, T, _ in BENCH_CONFIGS])},
        KNOBS.species, leptons=KNOBS.leptons)
    bench_path = save_table(bench_rows, table_path("hadronic", bench_name),
                            meta={"study": "hadronic benchmark",
                                  "models": ",".join(KNOBS.models),
                                  "modes": ",".join(m for m, _, _
                                                    in BENCH_CONFIGS),
                                  "species": KNOBS.species,
                                  "leptons": KNOBS.leptons,
                                  "repeat": BENCH_REPEAT})
    print("wrote", bench_path)

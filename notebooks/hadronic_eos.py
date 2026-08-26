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
# 6. **Benchmarks** — what a model costs, and where a line does not converge.
# 7. **Figures** — six families, every panel selectable for hyperons and
#    Deltas, and the stellar-structure pass behind two of them.
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
# `table_path`'s root is relative, so it is anchored to the ROOT found in the
# first cell: a kernel started in `notebooks/` would otherwise write the table
# into `notebooks/output/`.
print(table_path("dd2", example, root=str(ROOT / "output" / "tables")))

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
    path = save_table(tables[SAVE],
                      table_path(name, filename,
                                 root=str(ROOT / "output" / "tables")),
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
    bench_path = save_table(bench_rows,
                            table_path("hadronic", bench_name,
                                       root=str(ROOT / "output" / "tables")),
                            meta={"study": "hadronic benchmark",
                                  "models": ",".join(KNOBS.models),
                                  "modes": ",".join(m for m, _, _
                                                    in BENCH_CONFIGS),
                                  "species": KNOBS.species,
                                  "leptons": KNOBS.leptons,
                                  "repeat": BENCH_REPEAT})
    print("wrote", bench_path)

# %% [markdown]
# ## 7. Figures
#
# Six families, every one of them drawn for the sector combinations selected in
# the cell below — the panels of a figure ARE that selection, so a figure with
# hyperons on and off is one file with two panels rather than two files to line
# up by eye.
#
# All styling comes from `eos.general.figure_style` and nothing else: no
# rcParams are set here, no colour is re-declared, and the observational bands
# come from `eos.general.constraints.overlay`, keyed by the plane they live in.
# Everything is written under the repository root computed in the first cell,
# not under the working directory — a notebook executed from `notebooks/` would
# otherwise scatter its output into `notebooks/output/`.

# %%
import matplotlib.pyplot as plt

from eos.astro.tov import compute_tov_sequence, find_mmax_precise
from eos.general.constraints import overlay
from eos.general.figure_style import (LABELS, OKAB_CAT, PARTICLE_STYLES,
                                      log_decades, paper_grid, panel_label,
                                      particle_style, save_figure)
from eos.general.state import EOSTable_for_TOV

# Anchored to ROOT, not to the working directory.
FIG_DIR = ROOT / "output" / "hadronic"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_ROOT = str(ROOT / "output" / "tables")
print("figures  ->", FIG_DIR)
print("tables   ->", TABLE_ROOT)

# --- what every panel is selectable over -------------------------------
# Each entry is one panel: a name and the two sector flags it overrides on the
# knobs cell's species dict. Everything else about the species stays as the
# knobs cell set it.
FIG_SECTORS = (("nucleonic", dict(hyperons=False, deltas=False)),
               ("hyperons", dict(hyperons=True, deltas=False)))
# The other two, ready to be selected:
#   ("deltas",          dict(hyperons=False, deltas=True)),
#   ("hyperons+deltas", dict(hyperons=True,  deltas=True)),

FIG_N_B = np.geomspace(0.05, 1.2, 60)     # fm^-3, the curve grid
FIG_T = 0.0                               # MeV; the figures are the cold EoS
FIG_CS2_N_B = np.geomspace(0.10, 1.2, 25)  # coarser: one solve per stencil point
FIG_N_STARS = 20                          # central densities per TOV sequence

# Colour = model, so the same model is the same colour in all six families.
MODEL_COLOR = dict(zip(KNOBS.models, OKAB_CAT))

# %% [markdown]
# ### The sector is not free of the parametrisation
#
# Section 5 showed the coupling: `sfho` and `dd2` refuse a hyperonic sector on a
# nucleonic parameter set, because nobody published those couplings, and
# `sfho`'s Deltas need `SFHo_2fam`. So a sector selection carries a parameter
# set with it, and where it does not the knobs cell's choice stands. `did`
# appears nowhere below: its hyperon and Delta couplings were fitted with the
# rest, so the flag alone selects the sector.
#
# Symmetric nuclear matter is `Y_C = 0.5` with `Y_S = 0`. Three of the four take
# that as `fixed_YC_YS`. `zl` refuses the mode — and the refusal is the reason
# it does not need it: the functional is written in `n_p` and `n_n` alone, so
# `n_S = 0` identically and `fixed_YC` at `Y_C = 0.5` **is** symmetric matter
# there. One mode name per model, each for a stated reason.

# %%
SECTOR_SETS = {("sfho", "hyperons"): "SFHoY_Fortin",
               ("sfho", "deltas"): "SFHo_2fam",
               ("sfho", "hyperons+deltas"): "SFHo_2fam",
               ("dd2", "hyperons"): "DD2Y",
               ("dd2", "hyperons+deltas"): "DD2Y"}

SNM_MODES = {"zl": "fixed_YC", "sfho": "fixed_YC_YS",
             "dd2": "fixed_YC_YS", "did": "fixed_YC_YS"}


def figure_rows(name, sector, flags, mode, leptons=None, **conditions):
    """One model's rows over `FIG_N_B` at `FIG_T`, or None if it refuses.

    `leptons` is named only by the caller that means it — the fixed-fraction
    modes — and left unsaid for beta equilibrium, exactly as in section 4.
    """
    module = model(name)
    axes = {"nB": FIG_N_B, "T": np.array([FIG_T])}
    for key, value in conditions.items():
        axes[key] = np.array([value])

    def build():
        published = SECTOR_SETS.get((name, sector))
        par = (module.Parameters.named(published) if published
               else parameters_for(name))
        species = module.SpeciesFlags(**dict(KNOBS.species, **flags))
        extra = {} if leptons is None else {"leptons": leptons}
        return module.eos_table(par, mode, species, axes, **extra)

    status, result = run(f"{name} {sector}", build)
    return None if status != "ok" else module.rows_from_result(result)


def column(rows, key):
    """One column of a row list as an array."""
    return np.array([row[key] for row in rows])


header("beta equilibrium at T = 0")
beta_rows = {}
for sector, flags in FIG_SECTORS:
    for name in KNOBS.models:
        rows = figure_rows(name, sector, flags, "beta_eq_neutrinoless")
        if rows is None:
            continue
        beta_rows[(name, sector)] = rows
        print(f"  [{name:5s} {sector:16s}] {len(rows):3d}/{len(FIG_N_B)} rows   "
              f"P {rows[0]['P']:7.3f} -> {rows[-1]['P']:8.3f} MeV/fm^3")

header("symmetric nuclear matter — Y_C = 0.5, Y_S = 0, leptons=False")
snm_rows = {}
for sector, flags in FIG_SECTORS:
    for name in KNOBS.models:
        mode = SNM_MODES[name]
        conditions = ({"Y_C": 0.5} if mode == "fixed_YC"
                      else {"Y_C": 0.5, "Y_S": 0.0})
        rows = figure_rows(name, sector, flags, mode, leptons=False,
                           **conditions)
        if rows is None:
            continue
        snm_rows[(name, sector)] = rows
        print(f"  [{name:5s} {sector:16s}] {mode:12s} {len(rows):3d} rows   "
              f"P {rows[0]['P']:7.3f} -> {rows[-1]['P']:8.3f} MeV/fm^3")

# %% [markdown]
# ### Family 1 — pressure in beta equilibrium
#
# All selected models on one axis, one panel per sector selection.

# %%
def sector_grid(aspect=1.0, width=None):
    """A panel per selected sector, in the paper style. Nothing else in this
    notebook builds a figure, so the geometry is stated once, here."""
    fig, axes = paper_grid(f"1x{len(FIG_SECTORS)}", mode="double",
                           placeholder=False, aspect=aspect, width=width)
    return fig, axes[0]


fig, axes = sector_grid()
for ax, (sector, _) in zip(axes, FIG_SECTORS):
    for name in KNOBS.models:
        rows = beta_rows.get((name, sector))
        if rows is None:
            continue
        ax.plot(column(rows, "n_B"), column(rows, "P"),
                color=MODEL_COLOR[name], label=name)
    ax.set_xlabel(LABELS['nB'])
    ax.set_ylabel(LABELS['P'])
    ax.set_yscale("log")
    log_decades(ax)
    ax.set_title(sector)
axes[0].legend(loc="upper left")
for ax, tag in zip(axes, "abcd"):
    panel_label(ax, f"({tag})", corner="lower right")
save_figure(fig, str(FIG_DIR / "pressure_beta_eq"))

# %% [markdown]
# ### Family 2 — pressure of symmetric nuclear matter, against the flow data
#
# The heavy-ion constraints live in the `P-n` plane and are drawn by
# `overlay`: the Danielewicz, Lacey and Lynch (2002) analysis and the FOPI
# (IQMD) 2016 re-analysis, both bands on the pressure of SYMMETRIC matter, which
# is why this panel and not the beta-equilibrium one carries them.
#
# Below saturation the pressure of symmetric matter is negative — the binding —
# and a log axis cannot show it, so the panel starts where the bands do.

# %%
fig, axes = sector_grid()
for ax, (sector, _) in zip(axes, FIG_SECTORS):
    overlay(ax, "P-n")
    for name in KNOBS.models:
        rows = snm_rows.get((name, sector))
        if rows is None:
            continue
        ax.plot(column(rows, "n_B"), column(rows, "P"),
                color=MODEL_COLOR[name], label=name)
    ax.set_xlabel(LABELS['nB'])
    ax.set_ylabel(LABELS['P'])
    ax.set_xlim(0.15, 0.85)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 400.0)
    log_decades(ax)
    ax.set_title(f"{sector} — symmetric matter")
axes[0].legend(loc="lower right")
for ax, tag in zip(axes, "abcd"):
    panel_label(ax, f"({tag})", corner="upper left")
save_figure(fig, str(FIG_DIR / "pressure_snm"))

# %% [markdown]
# ### The gate that runs before any structure integration
#
# A table handed to a structure solver must have `P` non-decreasing in `n_B` and
# `0 <= c_s^2 <= 1`; a raw model branch may legitimately violate both inside a
# first-order transition, where mechanical instability is real physics rather
# than a bug (CLAUDE.md section 8). So the check runs **before** the TOV
# integration and returns a status: a branch that fails it is reported and left
# alone, never quietly repaired and never integrated into a mass that would mean
# nothing.
#
# `c_s^2` here is the finite-difference `dP/deps` of the delivered table itself,
# which is the quantity the solver will interpolate — not the model's
# `eos_response`, which is the next family and answers a different question.

# %%
def deliverable(core):
    """CLAUDE.md section 8's gate on a table about to be integrated.

    Returns (ok, message, cs2), with `cs2 = dP/deps` on the mid-points of the
    table. Nothing is modified: a failing table comes back as a status.
    """
    dP = np.diff(core.P)
    d_eps = np.diff(core.epsilon)
    cs2 = np.divide(dP, d_eps, out=np.full(dP.shape, np.nan), where=d_eps != 0)
    falling = np.flatnonzero(dP <= 0.0)
    acausal = np.flatnonzero(~((cs2 >= 0.0) & (cs2 <= 1.0)))
    mid = 0.5 * (core.nB[:-1] + core.nB[1:])
    parts = []
    if falling.size:
        parts.append(f"P falls at {falling.size} of {dP.size} steps, first at "
                     f"n_B = {mid[falling[0]]:.3f} fm^-3")
    if acausal.size:
        parts.append(f"c_s^2 outside [0, 1] at {acausal.size} steps, first at "
                     f"n_B = {mid[acausal[0]]:.3f} fm^-3")
    ok = not parts
    message = ("P non-decreasing and 0 <= c_s^2 <= 1 over "
               f"{dP.size} steps, max c_s^2 = {np.nanmax(cs2):.3f}"
               if ok else "; ".join(parts))
    return ok, message, cs2


header("the section 8 gate, before integration")
cores = {}
for key, rows in beta_rows.items():
    core = EOSTable_for_TOV(P=column(rows, "P"), epsilon=column(rows, "eps"),
                            nB=column(rows, "n_B"))
    ok, message, _ = deliverable(core)
    name, sector = key
    print(f"  [{name:5s} {sector:16s}] {'PASS' if ok else 'HOLD'}  {message}")
    if ok:
        cores[key] = core

# %% [markdown]
# ### Families 3 and 4 — mass–radius and mass–tidal deformability
#
# One TOV sequence per model and sector, over the gated tables only, with the
# BPS crust attached at `n_B = 0.08` fm^-3 — the density where that table tops
# out. `compute_tov_sequence` returns `(e_c, n_c, P_c, R, M, M_b, k2, Lambda)`,
# and `find_mmax_precise` gives the index of the maximum-mass star, so the slice
# up to it is the stable branch — everything beyond is unstable and belongs on
# neither plane. The library's own `truncate_to_stable_branch` is not used here:
# it re-orders to six columns and drops `Lambda`, which family 4 needs.

# %%
header("TOV sequences")
sequences = {}
for key, core in cores.items():
    name, sector = key
    e_c = np.geomspace(250.0, 0.95 * float(core.epsilon.max()), FIG_N_STARS)
    sequence = compute_tov_sequence(core, e_c, add_crust_table="BPS",
                                    n_transition=0.08, verbose=False)
    index, _, m_max = find_mmax_precise(sequence)
    sequences[key] = sequence[:index + 1]
    print(f"  [{name:5s} {sector:16s}] M_max = {m_max:.3f} M_sun at "
          f"R = {sequence[index, 3]:5.2f} km, {len(sequences[key]):2d} stable "
          f"of {len(sequence)} stars")

# %%
fig, axes = sector_grid()
for ax, (sector, _) in zip(axes, FIG_SECTORS):
    overlay(ax, "M-R")
    for name in KNOBS.models:
        sequence = sequences.get((name, sector))
        if sequence is None:
            continue
        ax.plot(sequence[:, 3], sequence[:, 4],
                color=MODEL_COLOR[name], label=name)
    ax.set_xlabel("$R$ [km]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_xlim(8.5, 16.0)
    ax.set_ylim(0.5, 2.7)
    ax.set_title(sector)
axes[0].legend(loc="lower left")
for ax, tag in zip(axes, "abcd"):
    panel_label(ax, f"({tag})", corner="upper right")
save_figure(fig, str(FIG_DIR / "mass_radius"))

# %%
fig, axes = sector_grid()
for ax, (sector, _) in zip(axes, FIG_SECTORS):
    overlay(ax, "M-Lambda")
    for name in KNOBS.models:
        sequence = sequences.get((name, sector))
        if sequence is None:
            continue
        ax.plot(sequence[:, 4], sequence[:, 7],
                color=MODEL_COLOR[name], label=name)
    ax.set_xlabel(r"$M$ [$M_\odot$]")
    ax.set_ylabel(r"$\Lambda$")
    ax.set_xlim(0.9, 2.1)
    ax.set_yscale("log")
    ax.set_ylim(5.0, 5e3)
    log_decades(ax)
    ax.set_title(sector)
axes[0].legend(loc="lower left")
for ax, tag in zip(axes, "abcd"):
    panel_label(ax, f"({tag})", corner="upper right")
save_figure(fig, str(FIG_DIR / "mass_lambda"))

# %% [markdown]
# ### Family 5 — the sound speed, named for what it holds
#
# `eos_response(frozen='equilibrium')` at each density: nothing is held, so the
# composition re-equilibrates under the perturbation and the derivative is taken
# along the mode's own sequence. These curves are at `T = 0`, where the
# isothermal and the adiabatic sound speed coincide — but the label still says
# which one was computed, because at `T > 0` they are different numbers and a
# bare `c_s^2` would mean whichever the arguments happened to select.
#
# All four models spell the key `cs2_isothermal`, naming the thermal variable
# the derivative was taken at. The composition axis is not part of the key: it
# is the `frozen='equilibrium'` these calls pass, under which nothing is held
# and the composition re-equilibrates. `dd2`, `did` and `sfho` return
# `cs2_adiabatic` beside it — the same number here at `T = 0`.

# %%
header("sound speed")
cs2_curves = {}
for sector, flags in FIG_SECTORS:
    for name in KNOBS.models:
        module = model(name)
        published = SECTOR_SETS.get((name, sector))

        # The parameter set and the species flags are built once, and inside
        # `run`: constructing the flags is itself a refusal site, and a model
        # that has not wired the sector must say so once rather than once per
        # density.
        def prepare():
            par = (module.Parameters.named(published) if published
                   else parameters_for(name))
            return par, module.SpeciesFlags(**dict(KNOBS.species, **flags))

        status, prepared = run(f"{name} {sector}", prepare)
        if status != "ok":
            continue
        par, species = prepared

        densities, values = [], []
        for n_B in FIG_CS2_N_B:

            def respond(n_B=float(n_B)):
                return module.eos_response(par, "beta_eq_neutrinoless",
                                           species, n_B=n_B, T=FIG_T)

            status, out = run(f"{name} {sector}", respond)
            if status != "ok" or not out.get("converged", True):
                continue
            densities.append(float(n_B))
            values.append(float(out["cs2_isothermal"]))
        if not densities:
            continue
        cs2_curves[(name, sector)] = (np.array(densities), np.array(values))
        print(f"  [{name:5s} {sector:16s}] {len(densities):2d} points, "
              f"max {max(values):.3f}")

# %%
fig, axes = sector_grid()
for ax, (sector, _) in zip(axes, FIG_SECTORS):
    for name in KNOBS.models:
        curve = cs2_curves.get((name, sector))
        if curve is None:
            continue
        ax.plot(curve[0], curve[1], color=MODEL_COLOR[name], label=name)
    ax.axhline(1.0 / 3.0, color='0.5', lw=0.8, ls=':')
    ax.set_xlabel(LABELS['nB'])
    ax.set_ylabel(r"$c_{s,\,\mathrm{isothermal}}^{2}$  $[c^2]$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{sector} — $T = 0$")
axes[0].legend(loc="upper left")
for ax, tag in zip(axes, "abcd"):
    panel_label(ax, f"({tag})", corner="lower right")
save_figure(fig, str(FIG_DIR / "cs2_isothermal"))

# %% [markdown]
# ### Family 6 — composition
#
# One panel per model, one row of panels per sector selection, every species
# coloured by `figure_style.particle_style`: colour by particle, linestyle by
# multiplet — nucleons solid, hyperons dashed, Deltas dash-dot, leptons dotted —
# so the panel reads in black and white too. A dozen species in a panel is what
# the `width` override of `paper_grid` is for.

# %%
def species_of(rows):
    """The species columns of a row list, in the order the style table names
    them, so the same particle sits in the same place in every panel."""
    present = {key[2:] for key in rows[-1] if key.startswith("Y_")}
    return [name for name in PARTICLE_STYLES if name in present]


fig, axes = paper_grid(f"{len(FIG_SECTORS)}x{len(KNOBS.models)}", mode="double",
                       placeholder=False, aspect=1.0, width=11.0)
for row, (sector, _) in zip(axes, FIG_SECTORS):
    for ax, name in zip(row, KNOBS.models):
        ax.set_title(f"{name} — {sector}")
        ax.set_xlabel(LABELS['nB'])
        ax.set_ylabel(LABELS['Y_i'])
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        log_decades(ax)
        rows = beta_rows.get((name, sector))
        if rows is None:
            ax.text(0.5, 0.5, "sector absent\nfrom the model", ha="center",
                    va="center", transform=ax.transAxes)
            continue
        n_B = column(rows, "n_B")
        for particle in species_of(rows):
            fraction = column(rows, f"Y_{particle}")
            if np.nanmax(fraction) < 1e-4:
                continue
            colour, linestyle = particle_style(particle)
            ax.plot(n_B, fraction, color=colour, ls=linestyle, label=particle)

# One legend for the whole figure: the species colours are the same table in
# every panel, so twelve panel legends would say the same thing twelve times
# and cover the onsets while doing it.
found = {}
for ax in axes.flat:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        found.setdefault(label, handle)
order = [name for name in PARTICLE_STYLES if name in found]
fig.legend([found[name] for name in order], order,
           loc="outside lower center", ncol=min(len(order), 9))
save_figure(fig, str(FIG_DIR / "composition"))

# %% [markdown]
# ### What was written
#
# Every path below is anchored to the repository root found in the first cell.

# %%
header("figures written")
for path in sorted(FIG_DIR.iterdir()):
    print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} kB)")

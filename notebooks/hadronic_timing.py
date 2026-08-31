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
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "eos").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from eos import dd2, did, sfho, zl
from eos.astro.tov import compute_tov_sequence, find_mmax_precise
from eos.dd2 import nmp as dd2_nmp
from eos.sfho import nmp as sfho_nmp
from eos.general.constraints import overlay
from eos.general.figure_style import (LABELS, OKAB_CAT, log_decades,
                                      panel_label, paper_grid,
                                      particle_style)
from eos.general.state import EOSTable_for_TOV
from eos.general.table_io import matrix_from_rows

# Name -> package. Every model exposes the same entry points under the same
# names — `Parameters`, `SpeciesFlags`, `eos_point`, `eos_table`,
# `rows_from_result` — so one loop body serves all four.
MODELS = {"zl": zl, "sfho": sfho, "dd2": dd2, "did": did}


# %% [markdown]
# ## One T = 0 beta-equilibrium line, per model
#

# %%
SPECIES = dict(hyperons=True, deltas=True, muons=False,
               thermal_mesons=False, thermal_neutrinos=False, photons=False)

# The published set these sectors' couplings come from; None = Parameters.default().
PARAMETER_SET = {"zl": None, "sfho": "SFHo_2fam", "dd2": "DD2Y", "did": None}

N_B = np.linspace(0.016, 1.6, 300)
T = 0.0

print(f"=== one line, beta_eq_neutrinoless, T = {T} MeV, "
      f"{len(N_B)} points from {N_B[0]:.3f} to {N_B[-1]:.3f} fm^-3 ===")

for name, module in MODELS.items():
    published = PARAMETER_SET[name]
    par = (module.Parameters.default() if published is None
           else module.Parameters.named(published))

    # A model that does not have one of the selected sectors refuses — at flag
    # construction where the sector is absent from the functional, inside
    # `eos_point` where the parameter set carries no couplings for it. Both are
    # the library's contract working, so they are printed and skipped, never
    # dressed up as a result. `TypeError` is deliberately not caught: an
    # unexpected keyword is this notebook's own bug.
    try:
        species = module.SpeciesFlags(**SPECIES)

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

    # Non-convergence is a return value, not an exception: it is found by
    # testing `.ok`, which no `except` clause would ever see.
    if not point.ok:
        print(f"  [{name}] first point did not converge: {point.message}")
        continue

    rows = module.rows_from_result(table)
    print(f"  [{name:5s} {published or 'default':13s}] "
          f"first point {1e3 * first_s:8.2f} ms   "
          f"line {line_s:6.3f} s   "
          f"{1e3 * line_s / max(len(rows), 1):6.2f} ms/pt   "
          f"{len(rows)}/{len(N_B)} points")


# %% [markdown]
# ## Reading a table
#
# `eos_table` returns a `TableResult` — the solver's own structure, one line per
# temperature and per combination of the fractions the mode fixes. It is not the
# shape you plot from; it is the shape the solver filled in, and it carries the
# axes it was asked for:
#
# * `table.nB`, `table.temp_key` ("T" or "SnB"), `table.temp_values` — the axes.
# * `table.combos` — `(temperature, fractions)` per line, in order.
# * `table.points` — the lines themselves, `table.points[i]` pairing with
#   `table.combos[i]`. Each entry is an `EoSPoint` carrying `n_B`, `T`, `P`,
#   `eps`, `s`, `converged`, `euler_residual` and the `matter` composition.
#
# `rows_from_result` flattens that into the long format everything downstream
# reads — a plain list of dicts, one per **converged** point, with the same
# column names in every model. That is what to plot, save and hand to TOV; reach
# into `table.points` only for something a row does not carry, such as the Euler
# residual of a single point.

# %%
par = dd2.Parameters.named("DD2Y")
species = dd2.SpeciesFlags(**SPECIES)
n_B_read = np.linspace(0.1, 1.2, 60)

table = dd2.eos_table(par, "beta_eq_neutrinoless", species,
                      {"nB": n_B_read, "T": np.array([0.0, 20.0])})

print("nB axis      ", table.nB[:3], "...", table.nB[-1])
print("temp_key     ", table.temp_key)
print("temp_values  ", table.temp_values)
print("combos       ", table.combos)
print("points       ", len(table.points), "lines of",
      [len(line) for line in table.points], "points")

point = table.points[0][0]
print("\none EoSPoint:", type(point).__name__)
print(f"  n_B={point.n_B:.3f}  T={point.T:.1f}  P={point.P:.3f}  "
      f"eps={point.eps:.3f}  s={point.s:.4f}")
# `euler_residual()` is a method, not a field: eps + P - T s - sum_i mu_i n_i
# over eps, the identity of CLAUDE.md section 8 as a number to test.
print(f"  converged={point.converged}  euler_residual={point.euler_residual():.2e}")

rows = dd2.rows_from_result(table)
print(f"\n{len(rows)} rows of {len(n_B_read) * len(table.temp_values)} requested")
print("columns:", sorted(rows[0]))

# %% [markdown]
# ### From rows to arrays, and a figure
#
# A row list holds every temperature at once, so a curve is a filter and then a
# column. Both are one line each — the filter picks the line, the list
# comprehension picks the column.
#
# All styling comes from `eos.general.figure_style`, the one home for it: the
# axis labels, the categorical palette, and `particle_style`, which gives every
# species its colour and the linestyle of its multiplet — nucleons solid,
# hyperons dashed, Deltas dash-dot, leptons dotted — so a composition panel
# reads in black and white too.

# %%
cold = [row for row in rows if row["T"] == 0.0]
n_B = np.array([row["n_B"] for row in cold])
P = np.array([row["P"] for row in cold])
eps = np.array([row["eps"] for row in cold])

print(f"T = 0 line: {len(cold)} points, "
      f"P {P[0]:.3f} -> {P[-1]:.3f} MeV/fm^3")

fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)

# Pressure against density, one curve per temperature.
for temperature, colour in zip(table.temp_values, OKAB_CAT):
    line = [row for row in rows if row["T"] == temperature]
    left.plot([row["n_B"] for row in line], [row["P"] for row in line],
              color=colour, label=f"T = {temperature:.0f} MeV")
left.set_xlabel(LABELS["nB"])
left.set_ylabel(LABELS["P"])
left.set_yscale("log")
left.legend()

# The composition of the T = 0 line: every Y_i column the rows carry.
for column in sorted(cold[0]):
    if not column.startswith("Y_") or column in ("Y_C", "Y_S"):
        continue
    fraction = np.array([row[column] for row in cold])
    if np.nanmax(fraction) < 1e-4:      # below the panel, not worth a colour
        continue
    colour, linestyle = particle_style(column[2:])
    right.plot(n_B, fraction, color=colour, ls=linestyle, label=column[2:])
right.set_xlabel(LABELS["nB"])
right.set_ylabel(LABELS["Y_i"])
right.set_yscale("log")
right.set_ylim(1e-4, 1.5)
right.legend(ncol=2, fontsize="small")

plt.show()


# %% [markdown]
# ## Particles, then parameters, then a star
#
# Three steps, timed one at a time, in dd2 and then in sfho:
#
# 1. **the particles** — a `SpeciesFlags`. Microseconds, and not a computation:
#    it is where a model refuses a sector it does not have.
# 2. **the parameters** — built rather than looked up. The nucleon sector comes
#    from a nuclear-matter-parameter target through the INVERSE map; the
#    hyperons from their single-particle potentials U_Y at saturation; the
#    Deltas from coupling ratios x_iDelta. Each sector is inverted ON the
#    inverted nucleon base, so it adapts to the couplings step 2 has just
#    moved instead of assuming the published ones.
# 3. **the star** — a T = 0 beta-equilibrium table, wrapped as the
#    `EOSTable_for_TOV` that is the model's side of the contract with
#    `eos.astro`, then a TOV sequence over it. `backend="fast"` is the
#    numba Dormand-Prince kernel, ~100x the scipy driver on this sequence and
#    the one to use for a sweep; `backend="scipy"` is the default and the
#    reference — adaptive DOP853, the flavour correctness is judged against
#    (CLAUDE.md section 9), and the one to keep for a strong phase transition
#    or an edge case. On this EoS they agree to 2e-4 M_sun in M_max and 0.01 km
#    in R. `tov_parallel=False` turns off the prange over the sequence, for a
#    call that is already inside a parallel map.
#
# The two models do not spell step 2 the same way, and the difference is
# physics, not style:
#
# * **dd2** has `build_parametrization`, one call that runs `invert_nmp`, the
#   SU(6) rescaling, `from_hyperon_potentials` and `from_delta_potential` in
#   that order. Its Delta sector may be given either way round -- the DEPTH
#   `U_Delta`, which it inverts, or the ratio `x_Delta_sigma`, which needs no
#   solve; `delta_potential` reads the other back. The vector ratios
#   x_Delta_omega and x_Delta_rho stay ratios either way.
# * **sfho** has no such wrapper, so the stages are written out: `invert_nmp`
#   for the nucleons, `replace` for the SU(6) factors, then
#   `from_potential_depths` for the hyperon depths and the three Delta ratios.
#   This model inverts no Delta depth, so x_Delta_sigma is named directly and
#   there is no depth form to choose.
#
# The nine SU(6) factors go in BEFORE the hyperon depths are closed, in both
# models: they scale the vector couplings and the scalar ones are inverted on
# top of them, so first means the depths hold and x_sigma re-fits, after means
# x_sigma holds and the depths move.
#
# The nuclear-matter parameters are CHOSEN, in `NMP` below, not looked up: both
# models are asked for the same six numbers, so the two curves at the end differ
# by the functional and by nothing else. Each model also ships its own published
# set — `dd2.PUBLISHED_NMP`, `sfho_nmp.PUBLISHED_NMP` — if you want to aim at one
# of those instead.
#
# What is legitimate to ask for is bounded by the functional. m*/m is the one to
# watch: DD2 sits at 0.5625 and SFHo at 0.76, and a target far outside a model's
# neighbourhood either fails to invert (a return value, scored and reported) or
# inverts into a parametrisation whose scalar mass collapses before the top of
# the density grid, which shows up as a short EoS table in step 3. m*/m = 0.70
# below is inside both.
#
# The inverse map is checked by the forward one: `compute_nmp` on the recovered
# parameters is printed against the target, together with Q_sat and K_sym, which
# this closure does not impose and therefore PREDICTS.

# %%
STARS = {}
SELECTED = {}          # {model: (par, flags)}, read by the grid cell
NBgrid = 200
N_B_TOV = np.linspace(0.05, 1.6, NBgrid)     # baryon density grid
N_STARS = 50                              # central densities per sequence

# --- 1. parameters and particles -----------------------------------------------------
flags = dd2.SpeciesFlags(hyperons=True, deltas=True)

NMP = dict(n_sat=0.153,          # fm^-3
           E_sat=-16.0,          # MeV
           m_eff_ratio=0.70,     # m*/m at saturation
           K_sat=222.0,          # MeV, incompressibility
           E_sym=30.0,           # MeV, symmetry energy
           L_sym=45.0)           # MeV, slope of the symmetry energy

# The hyperon depths at saturation in symmetric matter, MeV. Also read by both.
U_HYPERON = dict(U_Lambda=-27.0, U_Sigma=20.0, U_Xi=-15.0)

# SU(6)-breaking factors [y = 1 is SU(6)]
SU6 = dict(y_omega_Lambda=1.5, y_rho_Lambda=1.0, y_phi_Lambda=1.5,
           y_omega_Sigma=1.5,  y_rho_Sigma=1.0,  y_phi_Sigma=1.5,
           y_omega_Xi=1.875,   y_rho_Xi=1.0,     y_phi_Xi=1.875)

# The Delta scalar sector `U_Delta` is a DEPTH

#DELTA = dict(x_Delta_sigma=1.15)     # DELTA = dict(U_Delta=-100.0) # the other way round

DELTA_VECTOR = dict(x_Delta_sigma=1.15, x_Delta_omega=1.0, x_Delta_rho=1.0)

# The two nucleon shape coefficients the NMP closure HOLDS rather than fits.
# Among the imposed rows only P(n_sat) = 0 and K_sat carry shape information,
# so four shape coefficients answer to two rows and two of them have to be
# pinned; at WHICH values is the caller's choice, not the fit's.
# `dd2_nmp.PINNED_DEFAULT` names them, and the published DD2 numbers are the
# default -- read off the published set here rather than copied, so there is
# one home for them.
#
# They are not a cosmetic knob. The six NMPs come back to ~1e-12 whatever
# b_sigma is, because the shape acts above saturation and the imposed rows do
# not: it is a direction the nuclear-matter parameters cannot see and a
# neutron star can. Over +-30% b_sigma moves M_max by ~0.18 M_sun and R_1.4 by
# ~0.44 km, while c_omega over the same range moves them by ~0.02 M_sun and
# ~0.05 km. Leaving them pinned is defensible; leaving them pinned SILENTLY,
# in an inference that reports a width on M_max, is not.
_published = dd2.Parameters.default()
SHAPE = dict(b_sigma=_published.b_sigma, c_omega=_published.c_omega)

# One dict carries the NMPs, the SU(6) factors and the Delta sector, because
# that is how a sampler declares them: every key here is an axis it can vary,
# and `dd2_nmp.SECTOR_KEYS` is the list of which ones may ride alongside the
# nuclear-matter parameters. The held shape goes in as its own argument below;
# it would be accepted here too, and a key named in both places is taken from
# the sample.
target = dict(NMP, **SU6, **U_HYPERON, **DELTA_VECTOR)

start = time.perf_counter()

par, stage, message = dd2_nmp.build_parametrization(target, flags, pinned=SHAPE)

time_parametrization = time.perf_counter() - start

print("=== dd2 ===")
print(f"  1. parameters  {1e3 * time_parametrization:8.1f} ms   stage={stage} {message}")
print(par)


if stage == "ok":
    # --- the couplings the inversion produced ------------------------------
    # Only x_sigma is stored per hyperon; the three vector ratios are SU(6)
    # times the `SU6` factors, computed on demand, which is why breaking SU(6)
    # needs no new fields.
    Gs, Gw, Gr, _, _, _ = par.couplings_at(par.n_sat)
    print(f"  couplings at n_sat = {par.n_sat:.6f} fm^-3:")
    print(f"     Gamma_sigmaN {Gs:8.4f}   Gamma_omegaN {Gw:8.4f}   "
          f"Gamma_rhoN {Gr:8.4f}")
    # What the closure solved for, against what it was told to hold.
    print(f"     fitted   c_sigma {par.c_sigma:8.4f}  b_omega {par.b_omega:8.4f}"
          f"     held  b_sigma {par.b_sigma:8.4f}  c_omega {par.c_omega:8.4f}")
    print(f"     {'species':10s} {'m [MeV]':>9s} {'x_sigma':>9s} {'x_omega':>9s}"
          f" {'x_rho':>9s} {'x_phi':>9s}")
    for name, (mass, x_s, x_w, x_r, x_p) in par.hyperon_coupling_map.items():
        print(f"     {name:10s} {mass:9.2f} {x_s:9.4f} {x_w:9.4f}"
              f" {x_r:9.4f} {x_p:9.4f}")
    print(f"     {'Delta':10s} {1232.00:9.2f} {par.x_Delta_sigma:9.4f}"
          f" {par.x_Delta_omega:9.4f} {par.x_Delta_rho:9.4f} {0.0:9.4f}")

    # --- and the forward maps, read back off those couplings ---------------
    # `forward_maps` is every quantity `build_parametrization` takes, computed
    # back from the couplings it returned: the nuclear-matter parameters, the
    # hyperon depths and the Delta depth, in one call and one vocabulary.
    # Handing it straight back to `build_parametrization` reproduces these
    # couplings, which is the round trip that says the map inverted something.
    got = dd2_nmp.forward_maps(par)
    for key, wanted in {**NMP, **U_HYPERON}.items():
        print(f"     {key:12s} asked {wanted:9.4f}   got {got[key]:9.4f}")
    # Not imposed by this closure, so predictions -- U_Delta among them here,
    # because `DELTA` named the ratio and left the depth to come out.
    for key in ("Q_sat", "K_sym", "U_Delta"):
        print(f"     {key:12s} {'predicted':>9s}   {got[key]:9.4f}")

    # --- 2. eos table -------------------------------------------------------
    start = time.perf_counter()

    table = dd2.eos_table(par, "beta_eq_neutrinoless", flags, {"nB": N_B_TOV, "T": np.array([0.0])})

    time_eostable = time.perf_counter() - start

    print(f"  2. table eos: tot {1e3 * time_eostable:8.1f} ms, per point {1e3 * time_eostable/NBgrid:8.1f} ms ")

    # --- 3.tov -------------------------------------------------------
    start = time.perf_counter()

    rows = dd2.rows_from_result(table)
    core = EOSTable_for_TOV(P=np.array([row["P"] for row in rows]),
                            epsilon=np.array([row["eps"] for row in rows]),
                            nB=np.array([row["n_B"] for row in rows]))

    e_c = np.geomspace(250.0, 1 * float(core.epsilon.max()), N_STARS)
    sequence = compute_tov_sequence(core, e_c, add_crust_table="BPS",
                                    n_transition=0.08, 
                                    add_crust_mode='interpolate',delta_n=0.01,
                                    verbose=False,
                                    backend="fast", 
                                    compute_tidal=True, compute_baryonic_mass=False)
    index, ec_max, m_max = find_mmax_precise(sequence)
    STARS["dd2"] = dict(core=core, sequence=sequence[:index + 1]) #why +1?
    time_tov = time.perf_counter() - start

    print(f"  3. tov    {time_tov:8.2f} s "
          f"M_max = {m_max:.3f} M_sun at R = {sequence[index, 3]:.2f} km ; --  {sequence[index]}")

    # Every model's selection, so the grid cell below can loop over all three.
    SELECTED["dd2"] = (par, flags)


# %% [markdown]
# ### The same three steps in sfho
#
# Same six nuclear-matter parameters, same hyperon depths, same SU(6) factors,
# so the two curves differ by the functional and by nothing else. Two
# differences are the models', not the notebook's:
#
# * **sfho has no `build_parametrization`**, so the stages are written out:
#   `invert_nmp` for the nucleons, `replace` for the SU(6) factors, then
#   `from_potential_depths` for the hyperon depths and the three Delta ratios.
# * **sfho inverts no Delta depth.** `x_Delta_sigma` is named directly and
#   there is no `U_Delta` form to choose, so there is no `delta_potential` to
#   read one back with either.
#
# **What the NMP closure does NOT fix here.** Comparing a parametrization
# before and after `invert_nmp` field by field: it writes SIX numbers —
# g_sigmaN, g_omegaN, g_rhoN, g2, g3 and b_coeffs[1] — and inherits FORTY from
# `par_base`. The continuous ones among them are the four meson masses,
# `g_phi_N`, `c3` (the omega^4 self-coupling), `c4` (the rho^4 one), and eight
# of the nine shape coefficients of the A(sigma, omega) function that carries
# the symmetry energy above saturation. `sfho_nmp.PINNED_DEFAULT` names the
# seven scalars; the arrays and the baryon masses are moved through `par_base`
# itself, which is the general hook.
#
# Re-inverting the SAME six nuclear-matter parameters on a rescaled base
# reproduces them to ~1e-11 while moving the star, so every row below is a
# direction a nuclear-matter likelihood cannot see:
#
# | knob | ΔM_max | ΔR_1.4 | note |
# |------|--------|--------|------|
# | sfho `g_phi_N` (0 → 6) | 0.149 M_sun | **2.17 km** | the phi sector switched on |
# | sfho `m_omega` (±30%) | 0.209 M_sun | 0.12 km | and the branch dies at 80/150 |
# | dd2 `b_sigma` (±30%) | 0.176 M_sun | 0.44 km | |
# | sfho `c3` (±30%) | 0.040 M_sun | 0.01 km | |
# | dd2 `c_omega` (±30%) | 0.02 M_sun | 0.05 km | |
# | sfho `m_sigma`, `m_phi`, `c4`, a/b coeffs | ≤0.02 M_sun | ≤0.03 km | |
#
# `m_sigma` is nearly inert because m*/m is an imposed row and g_sigmaN is
# fitted, so the scalar sector is pinned by the targets themselves. `g_phi_N`
# is the largest lever in either model and the easiest to miss: published SFHo
# sets it to ZERO, which IS the statement that the sector is off — there is no
# flag to find it with (section 4) — so a multiplicative scan never moves it.
# For inference this makes SFHo the harder of the two models, not the easier:
# two directions the NMPs cannot see, one worth 0.2 M_sun and one worth 2 km,
# both pinned silently by `Parameters.default()` unless named.

# %%
# --- 1. parameters and particles -------------------------------------------
flags = sfho.SpeciesFlags(hyperons=True, deltas=True)

# The seven couplings the NMP closure holds rather than fits, read off the
# published set rather than copied. `sfho_nmp.PINNED_DEFAULT` names them; the
# shape arrays a_coeffs and b_coeffs[2:], the baryon masses and anything else
# stay reachable through `par_base`, which is the general hook.
_published_sfho = sfho.Parameters.default()
SHAPE_SFHO = {name: getattr(_published_sfho, name)
              for name in sfho_nmp.PINNED_DEFAULT}

start = time.perf_counter()

# The same call dd2 has, taking the same six nuclear-matter parameters, so one
# sample dict drives either model. It runs the stages this model spells out
# separately -- held couplings onto the base, `invert_nmp`, the SU(6) factors,
# then `from_potential_depths` -- in the order that holds the depths.
par, stage, message = sfho_nmp.build_parametrization(
    target, flags, pinned=SHAPE_SFHO)

time_parametrization = time.perf_counter() - start

print("=== sfho ===")
print(f"  1. parameters  {1e3 * time_parametrization:8.1f} ms   stage={stage} "
      f"{message}")

if stage == "ok":
    # --- the couplings the inversion produced ------------------------------
    print(f"  couplings at n_sat = {NMP['n_sat']:.6f} fm^-3:")
    print(f"     g_sigmaN {par.g_sigma_N:8.4f}   g_omegaN {par.g_omega_N:8.4f}   "
          f"g_rhoN {par.g_rho_N:8.4f}")
    # What the closure solved for, against what it inherited from par_base.
    # g2 and g3 are the scalar self-couplings the Boguta-Bodmer arm fits; c3
    # and c4 are the omega^4 and rho^4 terms it does not touch.
    print(f"     fitted   g2 {par.g2:10.6f}  g3 {par.g3:10.6f}"
          f"     held  c3 {par.c3:9.4f}  c4 {par.c4:9.4f}")
    print(f"     {'species':10s} {'x_sigma':>9s} {'x_omega':>9s} {'x_rho':>9s}"
          f" {'x_phi':>9s}")
    for name in ("lambda", "sigma+", "xi0", "delta++"):
        print(f"     {name:10s} "
              f"{par.get_coupling(name, 'sigma') / par.g_sigma_N:9.4f} "
              f"{par.get_coupling(name, 'omega') / par.g_omega_N:9.4f} "
              f"{par.get_coupling(name, 'rho') / par.g_rho_N:9.4f} "
              f"{par.get_coupling(name, 'phi') / par.g_omega_N:9.4f}")

    # --- and the forward maps, read back off those couplings ---------------
    # sfho has no single `forward_maps`: the NMPs and the hyperon depths are
    # two calls, and there is no third for the Delta because there is no
    # Delta depth in this model to invert or to read back.
    achieved = sfho_nmp.compute_nmp(par)
    depths = sfho_nmp.compute_hyperon_potentials(par)
    for key, wanted in NMP.items():
        print(f"     {key:12s} asked {wanted:9.4f}   got {achieved[key]:9.4f}")
    for key in ("Q_sat", "K_sym"):          # not imposed: predictions
        print(f"     {key:12s} {'predicted':>9s}   {achieved[key]:9.4f}")
    for key, U in depths.items():
        print(f"     {key:12s} asked {U_HYPERON[key]:9.4f}   got {U:9.4f}")

    # --- 2. eos table -------------------------------------------------------
    start = time.perf_counter()

    table = sfho.eos_table(par, "beta_eq_neutrinoless", flags,
                           {"nB": N_B_TOV, "T": np.array([0.0])})

    time_eostable = time.perf_counter() - start

    print(f"  2. table eos: tot {1e3 * time_eostable:8.1f} ms, "
          f"per point {1e3 * time_eostable / NBgrid:8.1f} ms ")

    # --- 3. tov -------------------------------------------------------------
    start = time.perf_counter()

    rows = sfho.rows_from_result(table)
    core = EOSTable_for_TOV(P=np.array([row["P"] for row in rows]),
                            epsilon=np.array([row["eps"] for row in rows]),
                            nB=np.array([row["n_B"] for row in rows]))

    e_c = np.geomspace(250.0, 1 * float(core.epsilon.max()), N_STARS)
    sequence = compute_tov_sequence(core, e_c, add_crust_table="BPS",
                                    n_transition=0.08,
                                    add_crust_mode='interpolate', delta_n=0.01,
                                    verbose=False,
                                    backend="fast",
                                    compute_tidal=True,
                                    compute_baryonic_mass=False)
    index, ec_max, m_max = find_mmax_precise(sequence)
    STARS["sfho"] = dict(core=core, sequence=sequence[:index + 1])
    time_tov = time.perf_counter() - start

    print(f"  3. tov    {time_tov:8.2f} s "
          f"M_max = {m_max:.3f} M_sun at R = {sequence[index, 3]:.2f} km")

    SELECTED["sfho"] = (par, flags)


# %% [markdown]
# ### One grid, four modes, three models
#
# The same `(n_B, T)` grid solved in every mode the three models share, timed
# and scored. `leptons` is section 3's orthogonal flag and is an explicit
# argument, never smuggled through the axes: on `fixed_YC` it chooses whether
# electrons are added to enforce total neutrality, and on a beta-equilibrium
# mode it is not a choice at all, so it is not passed there.
#
# **Counting the misses.** A line comes back shorter than the density axis for
# two different reasons and they are not the same failure. Past the
# scalar-collapse boundary (m* -> 0) there is nothing left to solve and EVERY
# remaining density fails, so those misses are a property of the model, not of
# the solver. A density that misses with solved points ABOVE it is a hole: the
# continuation lost the basin and that is a genuine non-convergence. Laying the
# rows back on the grid with `matrix_from_rows` separates them by construction
# — a trailing run of nan against an interior one — and `m_eff_i` at the last
# solved point is read to confirm the trailing run really is collapse.

# %%
SELECTED["did"] = (did.Parameters.default(),
                   did.SpeciesFlags(hyperons=True, deltas=True))

GRID_NB = np.linspace(0.05, 1.2, 60)
GRID_T = np.array([0.0, 10.0, 30.0])

# (mode, the fraction axes it fixes, the leptons flag). None means the mode
# does not take one.
RUNS = [("beta_eq_neutrinoless", {}, None),
        ("fixed_YC", {"Y_C": np.array([0.1])}, True),
        ("fixed_YC_YS", {"Y_C": np.array([0.5]), "Y_S": np.array([0.0])}, False),
        ("fixed_YC_YS", {"Y_C": np.array([0.0]), "Y_S": np.array([0.0])}, False)]

print(f"grid: {len(GRID_NB)} densities {GRID_NB[0]:.2f}-{GRID_NB[-1]:.2f} fm^-3 "
      f"x {len(GRID_T)} temperatures {list(GRID_T)} MeV")
print(f"{'model':6s} {'mode':20s} {'fixed':18s} {'lep':4s} {'tot s':>8s} "
      f"{'ms/pt':>7s} {'solved':>9s} {'m*->0':>6s} {'nonconv':>8s}")

for model_name, (model_par, model_flags) in SELECTED.items():
    module = MODELS[model_name]
    for mode, fracs, leptons in RUNS:
        axes = {"nB": GRID_NB, "T": GRID_T, **fracs}
        lepton_kw = {} if leptons is None else {"leptons": leptons}

        start = time.perf_counter()
        table = module.eos_table(model_par, mode, model_flags, axes, **lepton_kw)
        elapsed = time.perf_counter() - start

        rows = module.rows_from_result(table)

        # The grid as ASKED FOR, with nan where the solver returned nothing.
        # The axes come from this cell rather than from the result, because
        # the three TableResults do not carry the same attributes: dd2 and
        # sfho have temp_key/temp_values/combos and did has lines/points.
        matrix, _ = matrix_from_rows(rows, GRID_NB, GRID_T, "T")
        missing = np.isnan(matrix["P"])
        requested = missing.size
        collapsed, unconverged = 0, 0
        for i, line in enumerate(missing):
            solved = np.flatnonzero(~line)
            if len(solved) == 0:            # nothing at all: all of it is a miss
                unconverged += len(line)
                continue
            last = solved[-1]
            tail = int(line[last + 1:].sum())   # above the last solved density
            unconverged += int(line[:last].sum())          # holes underneath it
            # The tail is the branch ending. Where the model's point exposes
            # the effective masses, confirm that rather than assume it: a tail
            # left by a solver that stalled with m* still healthy is a
            # non-convergence wearing the boundary's clothes. did's EoSPoint
            # carries no `matter` block, so there it stays an inference.
            matter = getattr(table.points[i][-1], "matter", None)
            if tail and matter is not None and min(matter.m_eff_i.values()) > 100.0:
                unconverged += tail
            else:
                collapsed += tail

        fixed_label = " ".join(f"{k}={v[0]:g}" for k, v in fracs.items()) or "-"
        print(f"{model_name:6s} {mode:20s} {fixed_label:18s} "
              f"{str(leptons):4s} {elapsed:8.3f} {1e3 * elapsed / max(len(rows), 1):7.3f} "
              f"{len(rows):4d}/{requested:<4d} {collapsed:6d} {unconverged:8d}")


# %%

# %% [markdown]
# ### The two figures
#
# `compute_tov_sequence` returns one row per star, eight columns:
# `(eps_c, n_c, P_c, R, M, M_b, k2, Lambda)` — so radius is column 3 and
# gravitational mass column 4. The sequences stored above are already cut at
# the maximum-mass star.
#
# Left: mass–radius, with the observational bands from
# `eos.general.constraints`, drawn by the one `overlay` call that knows which
# constraints live in that plane. Right: the EoS these stars were integrated
# from, P against eps — the same `EOSTable_for_TOV` the solver read, so the two
# panels cannot drift apart.

# %%
fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                  constrained_layout=True)

overlay(left, "M-R")
for (name, star), colour in zip(STARS.items(), OKAB_CAT):
    left.plot(star["sequence"][:, 3], star["sequence"][:, 4],
              color=colour, label=name)
    right.plot(star["core"].epsilon, star["core"].P, color=colour, label=name)

left.set_xlabel("$R$ [km]")
left.set_ylabel(r"$M$ [$M_\odot$]")
left.set_xlim(8.5, 16.0)
left.set_ylim(0.5, 2.7)
left.legend(loc="lower left")

right.set_xlabel(LABELS["epsilon"])
right.set_ylabel(LABELS["P"])
right.set_xscale("log")
right.set_yscale("log")
right.legend(loc="upper left")

plt.show()

# %% [markdown]
# ### The three models against the data
#
# Four planes, one panel each, every one of them a plane
# `eos.general.constraints` knows: `overlay(ax, plane)` draws whatever lives
# there and nothing else has to know which constraints those are. All styling
# comes from `eos.general.figure_style`, the one home for it (section 10) —
# `paper_grid` for the geometry -- which applies the paper style itself --
# `LABELS` for the axes, `OKAB_CAT` for the model colours.
#
# The two microscopic panels are solved in the modes the DATA are quoted in,
# which is not the mode the stars are built from:
#
# * **E/A vs n_B** against the chiral-EFT band: pure NEUTRON matter, so
#   `fixed_YC_YS` at Y_C = 0, Y_S = 0 with `leptons=False`. Adding electrons
#   would put lepton pressure into a curve the band does not contain one.
# * **P vs n_B** against Danielewicz and FOPI: SYMMETRIC matter, Y_C = 0.5,
#   Y_S = 0, again `leptons=False`. Those bands are the flow constraint on
#   strongly-interacting matter alone.
#
# E/A is eps/n_B - m_N: the band is an energy per particle with the rest mass
# taken out, so the curve has to have it taken out too.

# %%
# `mode="double"` is the two-column page width: four panels, each carrying an
# overlay's worth of legend entries, do not fit the single-column preset.
# `placeholder=False` turns off the empty-panel marker paper_grid draws by
# default, which is for laying a figure out before the curves exist.
fig, axes = paper_grid("2x2", mode="double", placeholder=False, aspect=1.15)
(mr, ml), (en, pn) = axes

MICRO_NB = np.linspace(0.04, 0.8, 80)
COLD = np.array([0.0])
M_NUCLEON = 939.0                      # MeV, the rest mass E/A is measured from

# Any selected model that has no sequence yet gets one here, on the same core
# grid and the same crust as the two cells above -- did has no parameters cell
# of its own, so this is where its star gets built.
for name, (model_par, model_flags) in SELECTED.items():
    if name in STARS:
        continue
    rows = MODELS[name].rows_from_result(MODELS[name].eos_table(
        model_par, "beta_eq_neutrinoless", model_flags,
        {"nB": N_B_TOV, "T": COLD}))
    core = EOSTable_for_TOV(P=np.array([row["P"] for row in rows]),
                            epsilon=np.array([row["eps"] for row in rows]),
                            nB=np.array([row["n_B"] for row in rows]))
    sequence = compute_tov_sequence(
        core, np.geomspace(250.0, float(core.epsilon.max()), N_STARS),
        add_crust_table="BPS", n_transition=0.08,
        add_crust_mode="interpolate", delta_n=0.01, verbose=False,
        backend="fast", compute_tidal=True, compute_baryonic_mass=False)
    index, _, m_max = find_mmax_precise(sequence)
    STARS[name] = dict(core=core, sequence=sequence[:index + 1])
    print(f"  {name}: M_max = {m_max:.3f} M_sun")

for (name, (model_par, model_flags)), colour in zip(SELECTED.items(), OKAB_CAT):
    module = MODELS[name]

    # --- the two stellar panels, from the sequences computed above ----------
    if name in STARS:
        sequence = STARS[name]["sequence"]
        mr.plot(sequence[:, 3], sequence[:, 4], color=colour, label=name)
        # Lambda is the LAST column, not a fixed index: `compute_tov_sequence`
        # returns (e_c, n_c, P_c, R, M, [M_b], [k2], [Lambda]) and M_b is
        # optional, so the cells above -- which pass
        # compute_baryonic_mass=False -- get seven columns and not eight.
        ml.plot(sequence[:, 4], sequence[:, -1], color=colour, label=name)

    # --- pure neutron matter: E/A against the chiral-EFT band ---------------
    table = module.eos_table(model_par, "fixed_YC_YS", model_flags,
                             {"nB": MICRO_NB, "T": COLD,
                              "Y_C": np.array([0.0]), "Y_S": np.array([0.0])},
                             leptons=False)
    rows = module.rows_from_result(table)
    n_B = np.array([row["n_B"] for row in rows])
    en.plot(n_B, np.array([row["eps"] for row in rows]) / n_B - M_NUCLEON,
            color=colour, label=name)

    # --- symmetric matter: P against the heavy-ion flow bands --------------
    table = module.eos_table(model_par, "fixed_YC_YS", model_flags,
                             {"nB": MICRO_NB, "T": COLD,
                              "Y_C": np.array([0.5]), "Y_S": np.array([0.0])},
                             leptons=False)
    rows = module.rows_from_result(table)
    pn.plot(np.array([row["n_B"] for row in rows]),
            np.array([row["P"] for row in rows]), color=colour, label=name)

# --- the data, one call per plane ------------------------------------------
overlay(mr, "M-R")
overlay(ml, "M-Lambda")
# E-n carries both the neutron-matter and the symmetric-matter bands; this
# panel is neutron matter, so only the PNM ones belong on it.
overlay(en, "E-n", only=("chiral_eft", "chiEFT_PNM_E"))
overlay(pn, "P-n")

mr.set_xlabel("$R$ [km]")
mr.set_ylabel(r"$M$ [$M_\odot$]")
mr.set_xlim(8.0, 16.0)
mr.set_ylim(0.5, 2.7)

ml.set_xlabel(r"$M$ [$M_\odot$]")
ml.set_ylabel(r"$\Lambda$")
ml.set_xlim(0.8, 2.4)
ml.set_ylim(0.0, 1500.0)               # linear, as asked: the band lives here

en.set_xlabel(LABELS["nB"])
en.set_ylabel(r"$E/A$ [MeV]")
en.set_xlim(0.0, 0.35)
en.set_ylim(0.0, 45.0)

pn.set_xlabel(LABELS["nB"])
pn.set_ylabel(LABELS["P"])
# Density stays LINEAR here: 0.1 to 0.8 fm^-3 is less than one decade, so a
# log x buys no dynamic range and labels itself with minor ticks. Pressure
# spans four decades and does want one.
pn.set_yscale("log")
pn.set_xlim(0.05, 0.8)
# CMU Serif has no U+2212, so matplotlib's own mathtext log labels come out as
# hollow boxes below 1. `log_decades` is figure_style's protection for exactly
# that axis and is never removed (section 10).
log_decades(pn, axis="y")

for axis, lab in zip((mr, ml, en, pn), "abcd"):
    panel_label(axis, f"({lab})")
# One model legend for the whole figure -- the three curves are the same three
# everywhere, so repeating them four times spends panel area on nothing. Each
# overlay labels its own constraints in the panel they belong to.
mr.legend(loc="lower left", fontsize="x-small")
for axis in (ml, en, pn):
    axis.legend(fontsize="xx-small", loc="best")

plt.show()

# %% [markdown]
# ### Which crust join, and with what width
#
# The core stops at n_B = 0.05 fm^-3 and the crust is a separate calculation;
# `add_crust` offers three ways to put them together and `interpolate` takes a
# width. This cell runs all of them on the same core table, at T = 0 in
# beta equilibrium, and judges them on the four things that matter:
#
# * **monotonicity** — decreasing-P steps in the merged table, and the largest
#   c_s^2 = dP/deps it implies. Section 8: a table handed to a structure solver
#   has P non-decreasing and 0 <= c_s^2 <= 1.
# * **the seam itself** — the adiabatic index
#
#       Gamma = dln P / dln n_B = (n_B / P) dP/dn_B,
#
#   the log-log slope of P(n_B), evaluated over the blend window
#   n_tr +- 1.5 delta_n. It is the standard stiffness measure and it is
#   dimensionless and O(1) on both sides of the join: a BPS crust runs at
#   Gamma ~ 1.3 and these cores at ~2.5. A table can be monotone and still
#   have a KINK at the seam, and Gamma is what shows it -- G_seam = 17 means
#   the merged table locally rises as P ~ n_B^17, which is a join artefact and
#   not anything the two calculations say. dG is the largest step in Gamma
#   between neighbouring rows of the window, i.e. how abrupt that kink is.
# * **monotonicity INSIDE the window**, separately from the table as a whole.
#   `interpolate` blends P and mu_B = (P + eps)/n_B and then reconstructs
#   eps = mu_B n_B - P, so eps is not blended directly and is not guaranteed
#   monotone by construction the way P is; the window is also where PCHIP
#   extrapolates the core below its first solved point. Both are checked
#   there: dP <= 0, deps <= 0 and the minimum c_s^2 over the window alone.
# * **M_max, R_1.4, Lambda_1.4** — the same numbers, or not.
# * **time** — the join itself, and the TOV sequence run on what it produced.
#
# `attach` and `maxwell` are one-point joins with no width, so they appear once;
# `interpolate` is swept over delta_n.

# %%
# Local import: this cell is meant to be runnable on its own, without going
# back to the import cell at the top.
from eos.astro.tov import add_crust

CRUST = "BPS"
N_TRANS = 0.08                     # fm^-3, the same transition the cells above use
N_STARS_CRUST = 150                # denser than N_STARS: the joins differ by
                                   # under a per cent and the comparison must
                                   # not be reading its own interpolation error

JOINS = [("attach", {}),
         ("maxwell", {"delta_P": 0.0}),
         ("interpolate", {"delta_n": 0.002}),
         ("interpolate", {"delta_n": 0.005}),
         ("interpolate", {"delta_n": 0.01}),
         ("interpolate", {"delta_n": 0.02}),
         ("interpolate", {"delta_n": 0.04})]

CRUST_JOINS = {}

for name, (model_par, model_flags) in SELECTED.items():
    if name in STARS:
        core = STARS[name]["core"]
    else:
        rows = MODELS[name].rows_from_result(MODELS[name].eos_table(
            model_par, "beta_eq_neutrinoless", model_flags,
            {"nB": N_B_TOV, "T": np.array([0.0])}))
        core = EOSTable_for_TOV(P=np.array([row["P"] for row in rows]),
                                epsilon=np.array([row["eps"] for row in rows]),
                                nB=np.array([row["n_B"] for row in rows]))

    print(f"=== {name} ===  core {len(core.nB)} rows from "
          f"n_B = {core.nB.min():.3f} fm^-3")
    print(f"  {'join':22s} {'rows':>5s} {'dP<=0':>6s} {'cs2max':>7s} "
          f"|{'zone':>5s} {'bad':>4s} {'cs2min':>8s} {'G_seam':>7s} {'dG':>6s} "
          f"|{'M_max':>7s} {'R_1.4':>7s} {'L_1.4':>7s} {'t_join':>8s} {'t_TOV':>7s}")

    results = {}
    for mode, kwargs in JOINS:
        label = mode + (f" dn={kwargs['delta_n']}" if "delta_n" in kwargs else "")

        start = time.perf_counter()
        merged = add_crust(core, CRUST, mode=mode, n_transition=N_TRANS, **kwargs)
        time_join = time.perf_counter() - start

        # Section 8's gate, run on the table that is about to be integrated.
        dP = np.diff(merged.P)
        d_eps = np.diff(merged.epsilon)
        n_decreasing = int(np.sum(dP <= 0.0))
        rising = d_eps > 0.0
        cs2_max = float((dP[rising] / d_eps[rising]).max())

        # The same gate again, restricted to the blend window. A count over
        # the whole table can hide a handful of bad rows among three hundred
        # good ones, and the window is exactly where the merge invented rows
        # that neither calculation produced. `attach` and `maxwell` have no
        # width of their own, so they are measured over the same interval the
        # default `interpolate` uses -- otherwise the comparison is between
        # different amounts of table.
        width = kwargs.get("delta_n", 0.01)
        in_zone = (np.abs(merged.nB - N_TRANS) <= 1.5 * width)
        first, last = np.flatnonzero(in_zone)[[0, -1]]
        P_z = merged.P[first:last + 1]
        eps_z = merged.epsilon[first:last + 1]
        n_z = merged.nB[first:last + 1]
        dP_z, d_eps_z = np.diff(P_z), np.diff(eps_z)
        zone_bad = int(np.sum(dP_z <= 0.0) + np.sum(d_eps_z <= 0.0))
        cs2_zone_min = float((dP_z / d_eps_z).min())

        # The kink test, on that same window.
        gamma = np.diff(np.log(P_z)) / np.diff(np.log(n_z))
        gamma_seam = float(gamma.max())
        gamma_step = float(np.abs(np.diff(gamma)).max())

        # add_crust_table="No": the crust is already in `merged`, and adding it
        # twice is what this comparison is trying to measure.
        start = time.perf_counter()
        sequence = compute_tov_sequence(
            merged, np.geomspace(250.0, float(merged.epsilon.max()), N_STARS_CRUST),
            add_crust_table="No", verbose=False, backend="fast",
            compute_tidal=True, compute_baryonic_mass=False)
        time_tov = time.perf_counter() - start

        index, _, m_max = find_mmax_precise(sequence)
        stable = sequence[:index + 1]
        radius, mass, tidal = stable[:, 3], stable[:, 4], stable[:, -1]
        finite = np.isfinite(mass) & np.isfinite(radius) & np.isfinite(tidal)
        if finite.any() and mass[finite].max() >= 1.4:
            r_14 = float(np.interp(1.4, mass[finite], radius[finite]))
            lambda_14 = float(np.interp(1.4, mass[finite], tidal[finite]))
        else:
            r_14 = lambda_14 = np.nan

        results[label] = dict(M_max=m_max, R_14=r_14, Lambda_14=lambda_14,
                              gamma_seam=gamma_seam, gamma_step=gamma_step,
                              n_decreasing=n_decreasing, zone_bad=zone_bad,
                              cs2_zone_min=cs2_zone_min)
        # Section 8 is a gate, not a column to read later: a join that is not
        # monotone in its own blend window has no business reaching TOV.
        if zone_bad or cs2_zone_min <= 0.0:
            print(f"     ^ REJECTED: {zone_bad} non-monotone steps in the "
                  f"window, min c_s^2 = {cs2_zone_min:.4f}")
        print(f"  {label:22s} {len(merged.P):5d} {n_decreasing:6d} {cs2_max:7.3f} "
              f"|{len(P_z):5d} {zone_bad:4d} {cs2_zone_min:8.4f} {gamma_seam:7.2f} "
              f"{gamma_step:6.2f} |{m_max:7.4f} {r_14:7.3f} {lambda_14:7.1f} "
              f"{1e3 * time_join:7.2f}ms {time_tov:6.2f}s")

    CRUST_JOINS[name] = results

    # How far apart the joins are, as a fraction of the observable. M_max and
    # R_1.4 barely notice the join; Lambda_1.4 is the one that does, because
    # the tidal deformability is an integral over the whole star and the crust
    # is where the star is soft.
    for key in ("M_max", "R_14", "Lambda_14"):
        values = np.array([r[key] for r in results.values()])
        spread = float(np.ptp(values))
        print(f"     spread in {key:10s} {spread:9.4f} "
              f"({100 * spread / np.abs(values).mean():5.2f} % of mean)")

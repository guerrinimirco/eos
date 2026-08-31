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
from eos.general.figure_style import LABELS, OKAB_CAT, particle_style
from eos.general.state import EOSTable_for_TOV

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

print("nB axis      ", table.nB[:3], "...", tab)
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
rows

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
#   that order. Its Delta sector may be given either way round -- as a DEPTH,
#   `U_Delta`, which the model inverts into x_Delta_sigma, or as the ratio
#   x_Delta_sigma itself, which needs no solve. `delta_potential` reads the
#   other one back. x_Delta_omega and x_Delta_rho stay ratios either way.
# * **sfho** has no such wrapper, so the stages are written out: `invert_nmp`
#   for the nucleons, `replace` for the SU(6) factors, then
#   `from_potential_depths` for the hyperon depths, which also takes the three
#   Delta ratios. This model inverts no Delta depth, so x_Delta_sigma is
#   named directly and there is no depth form to choose.
#
# The nine SU(6)-breaking factors go in BEFORE the hyperon depths are closed,
# in both models and for the same reason: they scale the vector couplings, and
# the scalar couplings are inverted from the depths on top of them. Set them
# first and the depths stay what was asked while x_sigma re-fits; set them on
# the finished parametrization and x_sigma stays put while the depths move.
# Both are legitimate and only the first is a re-fit at fixed U_Y.
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
target

# %%
# Where both models leave their results, for the figures at the end.
STARS = {}

N_B_TOV = np.linspace(0.05, 1.6, 150)    # fm^-3, the core table. High
#   enough that both sequences turn over: a table that stops below the
#   maximum-mass central density gives a lower bound on M_max, not M_max.
N_STARS = 50                              # central densities per sequence

# --- 1. the particles ------------------------------------------------------
start = time.perf_counter()
flags = dd2.SpeciesFlags(hyperons=True, deltas=True)
species_s = time.perf_counter() - start

# --- 2. the parameters -----------------------------------------------------
# The six nuclear-matter parameters, chosen here and read by BOTH cells.
NMP = dict(n_sat=0.153,          # fm^-3
           E_sat=-16.0,          # MeV
           m_eff_ratio=0.70,     # m*/m at saturation
           K_sat=222.0,          # MeV, incompressibility
           E_sym=30.0,           # MeV
           L_sym=45.0)           # MeV, slope of the symmetry energy

# The hyperon depths at saturation in symmetric matter, MeV. Also read by both.
U_HYPERON = dict(U_Lambda=-27.0, U_Sigma=20.0, U_Xi=-15.0)

# The nine SU(6)-breaking factors, one per (vector meson, multiplet) pair, in
# both models under the same names. y = 1 is SU(6) exactly, which is DD2Y; the
# set below is the SFHoY breaking of Fortin, Oertel & Providencia, PASA 35
# (2018) e044 §2.2 -- omega and phi scaled by 1.5 for Lambda and Sigma and by
# 1.875 for Xi, rho left alone. y_phi = 0 in all three would be a set with no
# hidden-strange sector at all: the coupling IS that statement, there is no
# flag for it.
#
# ORDER MATTERS. These scale the VECTOR couplings, and the hyperon scalar
# couplings are inverted from the depths AFTER them, so the depths above stay
# what was asked and x_sigma moves instead. Setting them on the finished
# parametrization would do the opposite -- leave x_sigma alone and move U_Y --
# which is why they go in `target` and not into a `replace` afterwards. Both
# models take them the same way and for the same reason.
SU6 = dict(y_omega_Lambda=1.5, y_rho_Lambda=1.0, y_phi_Lambda=1.5,
           y_omega_Sigma=1.5,  y_rho_Sigma=1.0,  y_phi_Sigma=1.5,
           y_omega_Xi=1.875,   y_rho_Xi=1.0,     y_phi_Xi=1.875)

# The Delta scalar sector, named ONCE and in one of two ways. `U_Delta` is a
# DEPTH in SNM at saturation, which dd2 inverts into x_Delta_sigma and which
# carries the literature range [-100, -50] MeV as a guard; `x_Delta_sigma` is
# the RATIO itself, taken as it stands with no solve and no range. Whichever
# is given, `dd2_nmp.delta_potential` reads the other back. Naming both raises
# -- they are two names for one number.
DELTA = dict(U_Delta=-100.0)
# DELTA = dict(x_Delta_sigma=1.15)     # the other way round; U_Delta follows

# The Delta VECTOR ratios are free in both models either way.
DELTA_VECTOR = dict(x_Delta_omega=1.0, x_Delta_rho=1.0)

# One dict carries the lot: nuclear-matter parameters, SU(6) factors and the
# Delta sector on axes together, because that is how a sampler declares them.
# The hyperon depths stay a keyword here only because they read the same in
# both cells; they could equally ride in `target`.
target = dict(NMP, **SU6, **DELTA, **DELTA_VECTOR)

start = time.perf_counter()
par, stage, message = dd2_nmp.build_parametrization(
    target, flags,
    hyperon_potentials=U_HYPERON)
parameters_s = time.perf_counter() - start

print("=== dd2 ===")
print(f"  1. particles   {1e6 * species_s:8.1f} us   {flags}")
print(f"  2. parameters  {1e3 * parameters_s:8.1f} ms   stage={stage} {message}")

# `stage` is 'ok', 'inversion_failed' (the NMPs have no DD-RMF realisation) or
# 'sectors_failed' (they do, but the hyperon/Delta scalar inversion does not
# converge on them). Two different statements, so they are reported apart.
if stage == "ok":
    # Both directions of the Delta scalar map, whichever way round it was
    # given: the ratio the parametrization carries, and the depth it amounts
    # to. `delta_potential` is the forward map, so this line reads the same
    # under either form of `DELTA` above.
    print(f"     x_Delta_sigma {par.x_Delta_sigma:.4f}"
          f"   <->   U_Delta {dd2_nmp.delta_potential(par):9.4f} MeV")
    # And the hyperon depths read back off the SU(6)-broken couplings: the
    # check that breaking the vector sector re-fitted x_sigma rather than
    # moving the depths.
    for key, U in dd2_nmp.hyperon_potentials(par).items():
        print(f"     {key:12s} asked {U_HYPERON[key]:9.4f}   got {U:9.4f}")
    # The forward map on what the inverse map returned: did it hit the target?
    # Q_sat and K_sym are not imposed by the closure, so they are predictions.
    achieved = dd2_nmp.compute_nmp(par)
    for key, wanted in NMP.items():
        print(f"     {key:12s} asked {wanted:9.4f}   got {achieved[key]:9.4f}")
    for key in ("Q_sat", "K_sym"):
        print(f"     {key:12s} {'predicted':>9s}   {achieved[key]:9.4f}")

    # --- 3. the star -------------------------------------------------------
    start = time.perf_counter()
    table = dd2.eos_table(par, "beta_eq_neutrinoless", flags,
                          {"nB": N_B_TOV, "T": np.array([0.0])})
    rows = dd2.rows_from_result(table)
    core = EOSTable_for_TOV(P=np.array([row["P"] for row in rows]),
                            epsilon=np.array([row["eps"] for row in rows]),
                            nB=np.array([row["n_B"] for row in rows]))
    e_c = np.geomspace(250.0, 0.95 * float(core.epsilon.max()), N_STARS)
    sequence = compute_tov_sequence(core, e_c, add_crust_table="BPS",
                                    n_transition=0.08, verbose=False,
                                    backend="fast")
    index, _, m_max = find_mmax_precise(sequence)
    star_s = time.perf_counter() - start

    # Everything past the maximum-mass star is unstable and belongs on no plane.
    STARS["dd2"] = dict(core=core, sequence=sequence[:index + 1])
    print(f"  3. star        {star_s:8.2f} s    {len(rows)}/{len(N_B_TOV)} EoS "
          f"points, M_max = {m_max:.3f} M_sun at R = {sequence[index, 3]:.2f} km")
    # A maximum is only a maximum if the sequence turned over. When the heaviest
    # star is the last one computed, the table ran out before the star did and
    # the number is a LOWER BOUND on M_max, not M_max.
    turned_over = index + 1 < len(sequence)
    print("     maximum found: the sequence turned over" if turned_over else
          "     TABLE-LIMITED: the heaviest star is the last one computed, "
          "so M_max is a lower bound")

# %%
# --- 1. the particles ------------------------------------------------------
start = time.perf_counter()
flags = sfho.SpeciesFlags(hyperons=True, deltas=True)
species_s = time.perf_counter() - start

# --- 2. the parameters -----------------------------------------------------
# Two calls, because sfho has no single wrapper. `invert_nmp` returns
# `(Parameters, InversionStatus)` — non-convergence is a return value, so the
# status is tested rather than caught. The default base is nucleonic SFHo, so
# there is no strange sector yet and `hold_hyperons` does not apply.
start = time.perf_counter()
base, status = sfho_nmp.invert_nmp(**NMP)
if status.ok:
    # The SU(6) factors go on the base, BEFORE the depths are closed on it --
    # the same order dd2 uses, and here it has to be written out because sfho
    # has no wrapper with a seam in the middle. `from_potential_depths` reads
    # the nine factors off `base` and inverts x_sigma after them.
    base = replace(base, **SU6)
    # The hyperon depths and the three Delta ratios, closed on the base the
    # lines above just inverted and rescaled. sfho inverts no Delta depth, so
    # the scalar sector of the quartet is named as a ratio like the other two:
    # where `DELTA` names a depth this model cannot take it, and the two
    # quartets then differ by more than the functional does. That asymmetry is
    # the models', not the notebook's.
    par = sfho_nmp.from_potential_depths(
        U_Lambda_N=U_HYPERON["U_Lambda"], U_Sigma_N=U_HYPERON["U_Sigma"],
        U_Xi_N=U_HYPERON["U_Xi"], base=base,
        x_Delta_sigma=DELTA.get("x_Delta_sigma", 1.15), **DELTA_VECTOR)
parameters_s = time.perf_counter() - start

print("=== sfho ===")
print(f"  1. particles   {1e6 * species_s:8.1f} us   {flags}")
print(f"  2. parameters  {1e3 * parameters_s:8.1f} ms   ok={status.ok} "
      f"{status.message}")

if status.ok:
    achieved = sfho_nmp.compute_nmp(par)
    for key, wanted in NMP.items():
        print(f"     {key:12s} asked {wanted:9.4f}   got {achieved[key]:9.4f}")
    for key in ("Q_sat", "K_sym"):
        print(f"     {key:12s} {'predicted':>9s}   {achieved[key]:9.4f}")

if status.ok:
    # --- 3. the star -------------------------------------------------------
    start = time.perf_counter()
    table = sfho.eos_table(par, "beta_eq_neutrinoless", flags,
                           {"nB": N_B_TOV, "T": np.array([0.0])})
    rows = sfho.rows_from_result(table)
    core = EOSTable_for_TOV(P=np.array([row["P"] for row in rows]),
                            epsilon=np.array([row["eps"] for row in rows]),
                            nB=np.array([row["n_B"] for row in rows]))
    e_c = np.geomspace(250.0, 0.95 * float(core.epsilon.max()), N_STARS)
    sequence = compute_tov_sequence(core, e_c, add_crust_table="BPS",
                                    n_transition=0.08, verbose=False,
                                    backend="fast")
    index, _, m_max = find_mmax_precise(sequence)
    star_s = time.perf_counter() - start

    STARS["sfho"] = dict(core=core, sequence=sequence[:index + 1])
    print(f"  3. star        {star_s:8.2f} s    {len(rows)}/{len(N_B_TOV)} EoS "
          f"points, M_max = {m_max:.3f} M_sun at R = {sequence[index, 3]:.2f} km")
    # A maximum is only a maximum if the sequence turned over. When the heaviest
    # star is the last one computed, the table ran out before the star did and
    # the number is a LOWER BOUND on M_max, not M_max.
    turned_over = index + 1 < len(sequence)
    print("     maximum found: the sequence turned over" if turned_over else
          "     TABLE-LIMITED: the heaviest star is the last one computed, "
          "so M_max is a lower bound")

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

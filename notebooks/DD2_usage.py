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
# # DD2 EoS engine — usage notebook
#
# Production-style tour of the `eos.dd2` density-dependent RMF engine, laid out like
# the SFHo workflow: **one knobs cell → a job-loop that computes & writes the EoS
# tables → TOV → physics plots → a speed test**. The heavy code lives in the installed
# `eos` package; every non-trivial plotting/analysis routine lives in
# **`eos/dd2/notebook_api.py`** (imported here as `api`), so this notebook stays thin.
#
# **Paired to `DD2_usage.py` via jupytext** (`formats: ipynb,py:percent`): edit either
# file and the other updates on save. The `.py` is the review-friendly source of truth.
#
# Layout:
# - **Part I — Setup & knobs.** Imports and *every* tunable (parametrization, particles,
#   grids, table axes). Edit here, then Run-All. *Always run.*
# - **Part II — Compute & write the EoS tables** (β-eq, isentropic β-eq, fixed-Y_C,
#   trapped-ν, isentropic trapped) in one loop.
# - **Part III — TOV structure** for the selected parametrization.
# - **Part IV — Physics plots** (P–n_B, composition, sound speed, isentropic T,
#   symmetric matter vs Danielewicz+FOPI, PNM vs chiral EFT, M–R comparison, NMPs).
# - **Part V — Speed test** DD2 vs SFHo at fixed-Y_C.

# %% [markdown]
# # Part I — Setup & knobs
# ## I.1 — Imports & installs
#
# Run once from a fresh kernel. Keep the GitHub-install lines active for a clean
# environment (Colab); switch to the commented local-editable block when developing
# the `eos` package (repo edits then take effect after a kernel restart only).

# %%
import os
import sys

# ─── Prefer the local repo when developing next to it ────────────────────────
# On Colab this repo isn't present, so we fall through to the pip install below;
# locally, the parent dir (which contains ./eos) takes precedence over any stale
# pip copy, so edits to the package take effect on a kernel restart.
_repo = os.path.abspath("..")
if os.path.isdir(os.path.join(_repo, "eos")) and _repo not in sys.path:
    sys.path.insert(0, _repo)

# ─── Install the packages from GitHub (latest commit) ────────────────────────
# !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/guerrinimirco/eos.git --quiet
print("eos package loaded successfully!")
# !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/guerrinimirco/metastability-nucleation.git --quiet
print("nucleation package loaded successfully!")

# Local-dev alternative (comment the two installs above, uncomment these):
# # !{sys.executable} -m pip install -e .. --quiet
# # !{sys.executable} -m pip install -e ../../metastability-nucleation --quiet

# ─── Scientific Python ───────────────────────────────────────────────────────
import os
from dataclasses import replace

import numpy as np

# ─── DD2 engine (plotting helpers live in eos/dd2/notebook_api.py) ───────────
from eos.dd2 import Parametrization, SpeciesFlags
from eos.dd2 import notebook_api as api

# %% [markdown]
# ## I.2 — Knobs — **EDIT THIS CELL**
#
# Everything you'd normally change lives here; downstream cells only *read* these.
#
# - `PAR` — the parametrization. Default `from_dd2_defaults()` (nucleonic DD2). For
#   hyperons use `from_dd2y_defaults()`. To drive it from an explicit **NMP set**,
#   uncomment the `from_nmp` block (it inverts the NMPs to DD2 couplings; check `.ok`).
# - `FLAGS` — particle content. Hyperons/Δ need a par that carries their couplings
#   (DD2Y / a Δ-calibrated par); the plain DD2 par is nucleonic only.
# - grids — the axes of the tables and plots.
# - `TABLE_*` — fixed fractions and the output folder for Part II.

# %%
# ── parametrization ──────────────────────────────────────────────────────────
#PAR = Parametrization.from_dd2_defaults()          # ← nucleonic DD2 (default)
PAR = Parametrization.from_dd2y_defaults()        #   DD2Y (enables hyperons below)

# --- or an explicit NMP set (uncomment; inverts to DD2 nucleon couplings) -----
# NMP = dict(n_sat=0.149065, E_sat=-16.02, K_sat=242.7, Q_sat=169.0,
#            E_sym=31.67, L_sym=70.0, m_eff_ratio=0.5625)   # L_sym nudged 55→70 (stiffer sym energy)
# PAR, _status = from_nmp(NMP, return_status=True)
# assert _status.ok, f"NMP inversion did not converge: {_status.message}"

# --- custom hyperon potentials U_YN + Δ coupling ratios (uncomment) ------------
# These COMPOSE: pick a nucleon base (default DD2 above, or the NMP par), then
# attach hyperons on that base via `base=PAR`, then the Δ sector. Defaults shown
# for reference (DD2Y / universal-Δ). U_YN = hyperon single-particle potentials
# in SNM at n_sat [MeV]; x_Delta_* = g_ΔM/g_NM (σ is the uncertain one, lit. 1.0–1.25).
# U_Lambda_N, U_Sigma_N, U_Xi_N = -30.0, +30.0, -18.0        # defaults [MeV]
# x_Delta_sigma, x_Delta_omega, x_Delta_rho = 1.0, 1.0, 1.0  # defaults (universal SU(6))
# PAR = Parametrization.from_hyperon_potentials(U_Lambda_N, U_Sigma_N, U_Xi_N,
#                                               base=PAR)     # ← base=PAR keeps the NMP nucleon sector
# PAR = replace(PAR, x_Delta_sigma=x_Delta_sigma,            # attach the Δ sector directly
#               x_Delta_omega=x_Delta_omega, x_Delta_rho=x_Delta_rho)

# ── particle content ─────────────────────────────────────────────────────────
FLAGS = SpeciesFlags(hyperons=True,               # ← Λ,Σ,Ξ octet (needs DD2Y)
                     deltas=True,                 # ← Δ quartet (needs a Δ par)
                     muons=True,                   #   e always on; μ optional
                     phi_field=True)              #   hidden-strange φ (DD2Y default)

# ── axes ─────────────────────────────────────────────────────────────────────
n_sat= 0.149065
NB_GRID  = np.linspace(0.06, 10, 300)*n_sat             # β-eq density grid [fm^-3]
T_VALUES = np.concatenate([[0, 0.1], np.arange(2, 101., 2)])           # temperature axis [MeV] (bump for production)
S_VALUES = (1.0, 2.0)                              # isentropic S = s/n_B (must be reachable)
T_FIXED  = 10.0                                    # single T for T>0 diagnostics [MeV]

# ── table knobs (Part II) ────────────────────────────────────────────────────
TABLE_NB    = np.linspace(0.06, 1.2, 300)*n_sat            # density grid for the written tables
Y_C_VALUES   = (0.1, 0.3, 0.5)                     # fixed charge fractions (one fixed-Y_C file each)
YC_ELECTRONS = True                                # neutralise with electrons? (False = leptonless, 2a)
YC_MUONS     = False                               # also add muons? (only when YC_ELECTRONS; 2b)
Y_LE_VALUES  = (0.3,)                               # trapped-ν lepton fractions (one file each)
OUT_DIR     = "../output/tables_DD2/"              # written next to the SFHo convention
os.makedirs(OUT_DIR, exist_ok=True)

# Symmetric-matter plot (IV.5): Y_C=0.5, Y_S=0, no leptons, Δ active (vs FOPI/Danielewicz).
FLAGS_SNM = SpeciesFlags(hyperons=False, deltas=True, muons=False, neutrinos=False)

# ── reference data ───────────────────────────────────────────────────────────
# The raw posterior SAMPLES stay repo-relative (~160 MB, NOT shipped in the pip
# package). The M-R credible CONTOURS derived from them now ship inside the
# package, so they resolve wherever eos is installed.
from eos.general.observational_constraints import DEFAULT_CONTOUR_DIR

DATA_DIR    = "../plot/data"
SAMPLES     = DATA_DIR + "/samples"
CONTOUR_DIR = str(DEFAULT_CONTOUR_DIR)
CHIRAL_EFT  = SAMPLES + "/chiral_eft.txt"          # PNM E/N band (Y_C=0)
DANIELEWICZ = SAMPLES + "/DLL_2002_PSM.txt"        # SNM P flow constraint
FOPI_PSM    = SAMPLES + "/FOPI_2016_PSM.txt"       # SNM P flow constraint

# NS-structure flags: a hyperonic core needs a DD2Y PAR, otherwise go nucleonic.
FLAGS_TOV = FLAGS if FLAGS.hyperons else api.NUCLEONIC

print("PAR    :", PAR.__class__.__name__, "| n_sat =", round(PAR.n_sat, 4))
print("FLAGS  :", FLAGS)
print("NB_GRID:", round(NB_GRID[0], 3), "→", round(NB_GRID[-1], 3),
      "fm^-3,", len(NB_GRID), "points")

# %% [markdown]
# # Part II — Compute & write the EoS tables
#
# One loop, one row per table (the SFHo `jobs` pattern). Each job is
# `api.export_eos_table(PAR, flags, mode=…, nB=…, T|SnB=…, fixed=…, path=…)` →
# `TableSpec` + `build_table` under the hood.
#
# Modes: `beta` (charge-neutral β-eq, transparent ν); fixed-Y_C — either `YC` (leptonless,
# report §1.7 mode 2a) or `YC_e` (+ neutralising electrons, +μ iff `YC_MUONS`, mode 2b),
# selected by the `YC_ELECTRONS` / `YC_MUONS` knobs; `YL` (fixed lepton fraction, trapped ν).
# Temperature axis is `T=[…]` **or** `SnB=[…]` (entropy per baryon, adds the isentropic
# outer T-solve). The `fixed` dict is scalar, so the fixed-Y_C and trapped tables loop over
# `Y_C_VALUES` / `Y_LE_VALUES` writing one file per value.

# %%
# fixed-Y_C lepton content: electrons on/off (mode YC_e vs YC), muons via flags.
YC_MODE  = "fixed_YC_neutral" if YC_ELECTRONS else "fixed_YC"        # YC_e = neutralising e (+μ iff flags); YC = leptonless
FLAGS_YC = replace(FLAGS, muons=YC_MUONS, neutrinos=False)
FLAGS_TRAP = replace(FLAGS, neutrinos=True)        # trapped ν

# (label, mode, flags, temp-axis kwargs, fixed fractions, filename)
_lep = ("e" + ("mu" if YC_MUONS else "")) if YC_ELECTRONS else "nolep"
jobs = [
    ("betaeq",     "beta_eq_neutrinoless", FLAGS, dict(T=T_VALUES),   {}, "eos_dd2_betaeq.dat"),
    ("iso_betaeq", "beta_eq_neutrinoless", FLAGS, dict(SnB=S_VALUES), {}, "eos_dd2_betaeq_isentropic.dat"),
]
for yc in Y_C_VALUES:                              # fixed-Y_C tables, one per Y_C
    tag = f"YC{int(round(yc * 100))}_{_lep}"
    jobs.append((f"fixed_{tag}", YC_MODE, FLAGS_YC, dict(T=T_VALUES),
                 {"Y_C": yc}, f"eos_dd2_fixed{tag}.dat"))
for yl in Y_LE_VALUES:                              # trapped-ν tables, one per Y_Le
    tag = f"YL{int(round(yl * 100))}"
    jobs.append((f"trapped_{tag}", "beta_eq_neutrino_trapped", FLAGS_TRAP, dict(T=T_VALUES),
                 {"Y_Le": yl}, f"eos_dd2_trapped_{tag}.dat"))
    jobs.append((f"iso_trapped_{tag}", "beta_eq_neutrino_trapped", FLAGS_TRAP, dict(SnB=S_VALUES),
                 {"Y_Le": yl}, f"eos_dd2_trapped_{tag}_isentropic.dat"))

# skip_errors: drop points where the uniform solve doesn't converge (the low-T /
# low-density liquid-gas spinodal for constrained Y_C/Y_Le modes) instead of aborting.
results_H = {}
for label, mode, flags, temp_kw, fixed, fname in jobs:
    print(f"\n── computing DD2 table: {label}  (mode={mode}) ──")
    res, path = api.export_eos_table(PAR, flags, mode=mode, nB=TABLE_NB,
                                     fixed=fixed, path=os.path.join(OUT_DIR, fname),
                                     skip_errors=True, **temp_kw)
    results_H[label] = res
    nrows = sum(len(line) for line in res.points)
    full = len(res.nB) * len(res.temp_values)
    skipped = f" ({full - nrows} spinodal points skipped)" if nrows < full else ""
    print(f"   wrote {nrows} rows → {path}{skipped}")

# %% [markdown]
# ## II.1 — Peek at a written table
# First data line of the first fixed-Y_C table. With `YC_ELECTRONS=True` the
# neutralising leptons show up as `Y_e ≈ Y_C` (and `Y_mu` if `YC_MUONS`); with it
# off (leptonless mode `YC`) `Y_e = Y_mu = 0`.

# %%
_peek = f"eos_dd2_fixedYC{int(round(Y_C_VALUES[0] * 100))}_{_lep}.dat"
with open(os.path.join(OUT_DIR, _peek)) as f:
    head = [next(f) for _ in range(3)]
print(_peek)
print("".join(head).rstrip())

# %% [markdown]
# # Part III — TOV structure (selected parametrization)
#
# Cold β-eq core + BPS crust. Solved once and reused by the M–R / Λ–M figures.

# %%
tov = api.compute_tov(PAR, FLAGS_TOV)
print(f"M_max = {tov['M_max']:.3f} M_sun | R_1.4 = {tov['R_1p4']:.2f} km | "
      f"Lambda_1.4 = {tov['Lambda_1p4']:.0f}")

# %% [markdown]
# # Part IV — Physics plots
#
# All for the selected `PAR` / `FLAGS`. Cold / T=0 unless the plot is about temperature.

# %% [markdown]
# ## IV.1 — Pressure vs $n_B$ (β-equilibrium)

# %%
api.plot_p_vs_nb(PAR, flags=FLAGS, grid=NB_GRID);

# %% [markdown]
# ## IV.2 — Composition $Y_i$ vs $n_B$ (β-equilibrium)
# n, p, e, μ for a nucleonic par; hyperon onsets appear with a DD2Y par + `hyperons=True`.

# %%
api.plot_composition(PAR, flags=FLAGS, grid=NB_GRID);

# %% [markdown]
# ## IV.3 — Speed of sound $c_s^2$
# Frozen (fixed-composition) and equilibrium $c_s^2$, with the causal limit.

# %%
api.plot_sound_speed(PAR, flags=FLAGS);

# %% [markdown]
# ## IV.4 — Isentropic temperature
# Temperature along constant entropy-per-baryon paths S = s/n_B.

# %%
api.plot_isentropic_T(PAR, flags=FLAGS, S_values=S_VALUES);

# %% [markdown]
# ## IV.5 — Symmetric matter ($Y_C=0.5$, $Y_S=0$) vs Danielewicz + FOPI
# SNM pressure over the two heavy-ion flow constraints.

# %%
api.plot_p_vs_nb_snm(PAR, flags=FLAGS_SNM, grid=NB_GRID,
                     danielewicz=DANIELEWICZ, fopi=FOPI_PSM);

# %% [markdown]
# ## IV.6 — Pure neutron matter ($Y_C=0$, $Y_S=0$) vs chiral EFT
# PNM energy per particle E/N against the chiral-EFT band.

# %%
api.plot_pnm_chiral(PAR, chiral_path=CHIRAL_EFT);

# %% [markdown]
# ## IV.7 — Mass–radius: parametrization comparison
# The selected `PAR` against the three reference DD2 variants (default couplings):
# nucleons-only, DD2Y (hyperons), DD2YΔ (hyperons + Δ). Each solved once; $M_{max}$ in
# the legend.

# %%
curves = [
    ("selected",     PAR,                                                 FLAGS_TOV),
    ("DD2 nucleons", Parametrization.from_dd2_defaults(),                 api.NUCLEONIC),
    ("DD2Y",         Parametrization.from_dd2y_defaults(),                api.OCTET),
    ("DD2YΔ",        Parametrization.from_delta_potential(
                         base=Parametrization.from_dd2y_defaults()),
                     SpeciesFlags(hyperons=True, deltas=True, phi_field=True)),
]
api.plot_mass_radius_comparison(curves);

# %% [markdown]
# ## IV.8 — Λ–M and M–R vs observational constraints (selected par)

# %%
api.plot_lambda_mass(PAR, flags=FLAGS_TOV, tov=tov);

# %%
api.plot_mr_with_constraints(PAR, flags=FLAGS_TOV, tov=tov, contour_dir=CONTOUR_DIR);

# %% [markdown]
# ## IV.9 — Nuclear-matter parameters
# `compute_nmp(PAR)` next to the DD2 reference values.

# %%
print(api.format_nmp_comparison(PAR))

# %% [markdown]
# # Part V — Speed test: DD2 vs SFHo (fixed-$Y_C$)
#
# Matched fixed-$Y_C$ density sweep with electrons, no muons, no neutrinos. DD2 fast
# path (`sweep_octet(charge_mode='fixed', yc_leptons=True, analytic_jac=True)`, first
# call discarded for the Numba compile) vs the SFHo `fixed_yc` table generator, at T=0
# and T>0. `ratio > 1` means DD2 is faster.
#
# *Bottleneck note:* profiling this sweep found `solve_octet` was eagerly building the
# un-jitted beta-eq fallback guess on **every** point (~77% of the time) even though the
# warm start already converged. Making that fallback lazy dropped DD2 T=0 from ~0.9 to
# ~0.15 ms/pt (~3× faster than SFHo; ~parity at T>0 where the JEL integrals aren't jitted).

# %%
for T in (0.0, T_FIXED):
    b = api.benchmark_dd2_vs_sfho(PAR, T=T, Y_C=Y_C_VALUES[len(Y_C_VALUES) // 2])
    print(f"T = {T:>4} MeV | Y_C = {b['Y_C']} | {b['n_points']} pts | "
          f"DD2 {b['dd2_ms_per_pt']:.3f} ms/pt | "
          f"SFHo {b['sfho_ms_per_pt']:.3f} ms/pt | "
          f"ratio SFHo/DD2 = {b['ratio']:.2f}")

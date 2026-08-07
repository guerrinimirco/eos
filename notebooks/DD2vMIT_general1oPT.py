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
# # DD2 + vMIT — general framework for the first-order deconfinement transition
#
# A hadronic EOS (**DD2**, a density-dependent relativistic mean field) and a quark EOS
# (**vMIT**, a vector-enhanced bag model) joined across a first-order deconfinement
# phase transition, with the character of that transition controlled by a single
# continuous parameter **η**.
#
# **What η means.** The hadronic (*H*) and quark (*Q*) phases coexist in the mixed
# phase with a *quasi-permeable* interface for charged leptons: in the vicinity of the
# interface the two phases exchange charged leptons, which then help neutralize
# electric charge across both phases, while regions farther away remain effectively
# locally charge neutral. Charge neutrality is therefore realized *partly locally* and
# *partly globally*, and η ∈ [0,1] measures the fraction of charged leptons that
# enforce local charge neutrality (**LCN**) separately in each phase; the remaining
# fraction 1−η participates only in global charge neutrality (**GCN**).
#
# | η | construction | behaviour through the mixed phase |
# |---|---|---|
# | 0 | **Gibbs (GC)** | baryon number *and* electric charge are conserved globally; only the volume average is neutral and the pressure varies continuously across the mixed phase |
# | 1 | **Maxwell (MC)** | baryon number is the only globally conserved charge and neutrality is imposed locally in each phase; the pressure remains constant and the window collapses to a density jump |
# | in between | — | finite-size effects are *not* included microscopically; their net impact is emulated by tuning the balance between local and global neutrality through η |
#
# An equivalent picture is that of pure-phase "lumps" forming in the coexistence
# region. If their characteristic size is much larger than the Debye screening length,
# charged leptons are effectively tied to individual lumps and charge neutrality is
# local — roughly, the σ → ∞ limit; if it is much smaller, they experience an averaged
# background and neutrality becomes purely global (σ → 0). Intermediate sizes
# correspond to 0 < η < 1.
#
# **How a density is classified.** The quark volume fraction χ ≡ V_Q/(V_H + V_Q) comes
# out of the solve and is *not* clamped: **χ ≤ 0 → pure hadronic, χ ≥ 1 → pure quark,
# 0 < χ < 1 → mixed phase.** The table builder locates the two χ crossings first and
# then solves the expensive mixed system only between them.
#
# **The two sound speeds.** A first-order transition has two, and the gap between
# them is the clearest single picture of what η does:
#
# | | what is free | behaviour in the window |
# |---|---|---|
# | **c_eq² = dP/dε** | χ readjusts | dips at η=0, **collapses to 0** at η=1 |
# | **c_ad² (frozen)** | χ held fixed | stays finite at every η |
#
# c_eq is what enters the TOV equations; c_ad is what a fast disturbance sees.
# Section III.4 plots both.
#
# ---
# **Layout**
# - **Part I** — imports, the parameter-space funnel, then every knob in one cell.
# - **Part II** — pure phases, the mixed tables, TOV, and a parameter scan.
# - **Part III** — plots, all defined here in the notebook.
#
# **Start here if you are choosing parameters:** run Part I only. Section I.2 is a
# staged funnel over (nuclear-matter parameters × hyperon and Δ potentials × B^1/4,
# a, m_s × η) that reports which combinations land in a target M_max and R(1.4) band
# and, for each, whether the centre of the star is hadronic, mixed or pure quark. It
# ends by printing the recipe for the best one, ready to paste into I.3 before Part II
# is run at full resolution. Section II.4 then maps the (B^1/4, a) plane of that one
# choice as a picture.

# %% [markdown]
# # Part I — setup

# %% [markdown]
# ## I.1 Imports
#
# Imports the repository clone when the notebook sits inside one — the root is found by
# walking up for `pyproject.toml` — and falls back to installing the package from GitHub
# only when it does not, so a stale copy in `site-packages` can never shadow the working
# tree. The line printed at the end says which copy was actually loaded.
#

# %%
import sys
import time
import subprocess
from pathlib import Path

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "pyproject.toml").is_file() and (p / "eos").is_dir()), None)
if ROOT is not None:
    sys.path.insert(0, str(ROOT))
    for _m in [m for m in sys.modules if m == "eos" or m.startswith("eos.")]:
        del sys.modules[_m]

try:
    import eos.mixed
except ImportError:      
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps",
                           "--quiet",
                           "git+https://github.com/guerrinimirco/eos.git"])
    import eos.mixed

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from eos.dd2 import Parametrization, SpeciesFlags, hadronic_row, compute_nmp
from eos.dd2.solver import sweep_beta_eq_octet
from eos.vmit.parameters import get_vmit_default, get_vmit_custom
from eos.vmit.eos import solve_vmit_beta_eq
from eos.tov.solver import find_mmax_precise
from eos.tov.rotating import (kepler_sequence, rotating_grid,
                              GRID_COLUMNS, KEPLER_COLUMNS)
from eos.tov.rns_backend import have_rns
from eos.mixed import (
    beta_eq_neutrinoless,
    MixedTableSpec, build_mixed_table, build_mixed_eos_table,
    mass_radius_mixed, save_table, load_table, export_csv,
    locate_window, sweep_mixed,
    sound_speed_eq, frozen_along,
    scan_parameters, scan_hadronic, grid_samples, DEFAULT_U_DELTA,
)

from eos.general.figure_style import (
    set_global_style, setup_scientific_figure, apply_style, add_panel_labels,
    save_figure, particle_style, LABELS, STANDARD_COLORS,
)
from eos.general.observational_constraints import add_observational_constraints

set_global_style()

OUT = Path("eos_tables_DD2vMIT")
OUT.mkdir(exist_ok=True)
print("eos imported from:", Path(eos.__file__).parent)


# %% [markdown]
# ## I.2 Parameter-space funnel — which parameters give the star you asked for?
#
# One cell, four questions, asked in the order that fails cheapest. Each stage runs
# only on what survived the one before, so the expensive mixed-phase solves are never
# spent on parameters that were already ruled out.
#
# | stage | question | cost per sample |
# |---|---|---|
# | **1** | which nuclear-matter parameters have a DD-RMF realisation at all? | milliseconds |
# | **2** | which of those, crossed with the hyperon and Δ potentials, give a hadronic branch near the target band? | ~1 s |
# | **3** | which of those, crossed with the quark parameters, give M_max and R(1.4) in the target band — at how many η, and **where does the centre of the star actually sit**? | ~2–3 s |
#
# **The answer stage 3 gives is a phase, not a yes/no.** A transition existing in the
# equation of state and a transition being *realised inside a star* are different
# statements: the window can open at a density no stable star reaches. Every
# combination is therefore labelled by where the central baryon density of the
# maximum-mass star falls:
#
# | `core_phase` | meaning | is it a hybrid star? |
# |---|---|---|
# | `none` | no window on this grid | no — pure hadronic, judged on the stage-2 hadronic M_max and R(1.4) |
# | `H` | a window exists but n_Bc < n_onset | no — the transition is above every stable star |
# | `mix` | n_onset ≤ n_Bc < n_offset | yes — a mixed-phase core |
# | `Q` | n_Bc ≥ n_offset | yes — a genuine pure-quark core |
#
# **Which parameters can actually be varied.** `L_sym` is free — the isovector
# inversion is near-analytic and converges across 30–100 MeV. **`K_sat` is not**: the
# isoscalar system is closed by a cross-constraint that ties `K_sat` to `Q_sat`, so
# moving `K_sat` alone leaves a residual that grows away from the DD2 value (~3e-2 at
# 240 MeV against a 2e-2 gate, 0.2 by 220 MeV). Stage 1 reports it as
# `inversion_failed`, which is a statement about the DD2 functional form, not a solver
# defect. The `K_sat` axis is kept below precisely so you can see that.
#
# The hadronic *sector* knobs are all free and matter far more than the NMP do, because
# they set how soft the hadronic branch becomes: `x_wD` alone moves the viable fraction
# by a factor of ~4, while `U_Lambda` and `U_Xi` barely move it once the transition
# sits below the hyperon threshold. `from_delta_potential` restricts `U_Delta` to the
# literature range [-100, -50] MeV.
#
# **The stage-2 cut is deliberately soft.** A hadronic branch under 2 M_sun is not
# disqualifying — quark matter with a > 0 stiffens the star and routinely lifts it back
# above two solar masses — so stage 2 reports every sample and drops only those under a
# loose floor. `sweep_truncated` is kept for the same reason: the hadronic branch only
# has to reach the transition *onset*, not the top of the grid, and a Δ model driving
# the effective mass to zero at high density is expected physics.
#
# **A third stage-3 outcome, and with hyperons and Δ on it is a common one:
# `eos_unphysical`.** The χ=0 crossing the window locator finds can be spurious at a
# low bag constant — the mixed branch it locks onto sits at a *lower* pressure than the
# hadronic branch at the same density, so it is not the favoured state, and the
# stitched table steps *down* at the onset. Integrating that returns confident nonsense
# (maximum masses in the hundreds of solar masses), so P monotonicity and 0 ≤ c_s² ≤ 1
# are checked **before** the TOV run. Those rejections are the scan working.
#
# The funnel runs on the coarse `FUNNEL_GRID`; it is reconnaissance. Re-run Part II on
# the winning recipe printed at the end into I.3, then run Part II at full `NB`.
#
# The stages are separate cells on purpose: tightening `M_MAX_RANGE` or `R14_RANGE` and
# re-running the last two cells is instantaneous, where re-running the whole funnel
# is minutes.

# %%
from collections import Counter, defaultdict
from joblib import Parallel, delayed

from eos.mixed import scan_point, build_parametrization, NMP_KEYS

#: The keys `build_parametrization` understands. A finished scan row carries
#: results and string columns alongside them, and only these may be fed back in.
PAR_KEYS = NMP_KEYS + ("U_Lambda", "U_Sigma", "U_Xi", "U_Delta", "x_wD", "x_rD")

# ---- what counts as an acceptable star ------------------------------------
# The band the FINAL star must land in. Stage 3 tests these.
M_MAX_RANGE = (2.0, 2.6)          # M_sun
R14_RANGE = (11.5, 13.5)          # km, radius at M = 1.4 M_sun

# The stage-2 pre-filter on the PURE HADRONIC branch. Deliberately looser than
# the target band: quark matter changes both numbers, so a hard cut here throws
# away hybrid stars that would have qualified. A sample whose hadronic TOV did
# not integrate at all is kept — nan is "unknown", not "failed".
HAD_M_MIN = 1.6                   # M_sun, soft floor (NOT M_MAX_RANGE[0])
HAD_R14_RANGE = (10.0, 15.0)      # km, soft band

# ---- axes -----------------------------------------------------------------
# Independent of the PAR / FLAGS / VMIT chosen in I.3: this section builds its
# own parametrizations from nuclear-matter parameters, with hyperons and Delta
# isobars switched on regardless of what FLAGS says.
FUNNEL_FLAGS = SpeciesFlags(hyperons=True, deltas=True, muons=True,
                            phi_field=True, photons=True)
NUCLEON_FLAGS = SpeciesFlags(hyperons=False, deltas=False, muons=True,
                             phi_field=False, photons=True)

# ---- stage-1 axes: K_sat and Q_sat on a plain rectangular grid ------------
# Both are swept independently and widely, which is worth a word because it
# contradicts what a single-seeded inversion appears to say.
#
# The isoscalar sector has six free couplings — Gamma_sigma, b_sigma, c_sigma
# and their omega counterparts, since d_i and a_i are fixed by f_i''(0) = 0 and
# f_i(1) = 1 — matched to six conditions: P(n_sat) = 0, E_sat, m*/m, K_sat,
# Q_sat and the closing cross-constraint f_sigma''(1) = f_omega''(1). A square
# system, so the freedom is genuinely limited. But the set that a solve seeded
# from the published DD2 couplings can REACH is much smaller than the set that
# exists, and it traces a narrow band through DD2's own (K_sat, Q_sat) that is
# easy to mistake for a physical constraint curve. On a 187-cell grid over
# K_sat = 200-300 and Q_sat = 0-400 MeV, the single seed inverts 7 cells;
# 32 restarts invert 68 and 64 restarts invert 115, still without saturating.
#
# `invert_nmp` now restarts on a miss by default, so stage 1 reports the second
# number rather than the first — but a miss costs n_restarts solves, which is
# why stage 1 is no longer free and why the grid below is a few dozen cells
# rather than a few hundred.
SCAN_K_SAT = np.arange(210.0, 291.0, 20.0)        # MeV
SCAN_Q_SAT = np.arange(250.0, 351.0, 50.0)         # MeV, independent of K_sat
SCAN_L_SYM = [30.0, 50.0, 70.0, 90.0]             # free over 30-100

# How many stage-1 survivors reach the expensive stages. The grid above is wide
# so the feasible region is visible; carrying every cell of it into stage 3
# would multiply that stage's cost by the number of survivors. They are thinned
# evenly, so what goes forward still spans the region rather than clustering at
# one corner of it.
MAX_NMP = 10

# ---- stage-2 axes: spend the budget where the sensitivity is ---------------
# It is very uneven. x_wD alone moves the viable fraction by a factor of ~4 and
# U_Delta sets how soft the branch becomes, while U_Lambda and U_Xi barely move
# anything once the transition sits below the hyperon threshold — so those two
# are held at one value each and the freed factor of four goes to the axes that
# matter. U_Sigma is the least constrained hyperon potential (repulsive, but by
# how much is genuinely open), so it gets an axis rather than a default.
SCAN_U_LAMBDA = [-30.0,-25]           # MeV, hyperon potentials in SNM at n_sat
SCAN_U_SIGMA = [0.0, 30.0]
SCAN_U_XI = [-22.0,-10]
SCAN_U_DELTA = [-100.0, -50.0]   # from_delta_potential limits
SCAN_X_WD = [0.9, 1.2]  # Delta vector ratio x_omegaDelta
SCAN_X_RD = [1.0]                 # x_rhoDelta; the isovector ratio moves least

# a and B^1/4 both raise the onset density, so the viable combinations sit on
# an anticorrelated ridge: a = 0.20 wants B^1/4 ~ 160-180, a = 0.10
# wants ~190-200. A rectangular grid over the pair therefore spends most of its
# cells off that ridge, which is worth knowing when reading the reject count and
# is why both axes are swept wide rather than finely. Outside 0.05 <= a <= 0.20
# nothing survives: a = 0 leaves the quark phase too soft to reach 2 M_sun, and
# by a = 0.25 the ordered window is gone for every B^1/4 from 120 to 200 MeV.
SCAN2_B4 = [160.0, 170.0, 180.0]   # MeV
SCAN2_A = [0., 0.2, 0.4,  0.6,  0.8]               # fm^2
SCAN2_MS = [50, 150.0]                # MeV; the EoS is only weakly sensitive to it

FUNNEL_ETAS = [0.0, 0.3, 0.6, 1.0]
#: One colour per eta for the funnel's own panels. Part III builds an equivalent
#: map for ETA_LIST, but Part I runs before it and must not depend on it.
FUNNEL_ETA_COLOR = {e: c for e, c in zip(
    sorted(FUNNEL_ETAS),
    plt.cm.viridis(np.linspace(0.0, 0.88, len(FUNNEL_ETAS))))}
FUNNEL_GRID = np.linspace(0.05, 1.6, 80)   # a probe grid, coarser than NB
FUNNEL_JOBS = -1                  # 1 to keep it serial and debuggable
FUNNEL_TOP = 20                   # combinations printed in the final table

# Measured wall cost per sample on ten cores, used only so a stage can say what
# it is about to spend before it spends it. Re-measure if the machine changes.
SEC_PER_HADRONIC = 0.46
SEC_PER_HYBRID = 0.43


def _in(x, band):
    """Is x inside [lo, hi]? nan is outside — an unknown is not a pass."""
    return bool(np.isfinite(x) and band[0] <= x <= band[1])


def _budget(n, sec_each):
    """A sample count as a projected wall time, so a stage can be cut before
    it runs rather than after."""
    t = n * sec_each
    return f"{n} samples, ~{t:.0f} s" if t < 120 else f"{n} samples, ~{t/60:.0f} min"


def _hadronic_mr(par, flags):
    """(R, M) of the cold beta-equilibrium hadronic branch, stable part only.

    The scan records M_max and R(1.4) but not the sequence they came from, so
    the curve is rebuilt here. Same core table and the same Numba backend the
    scan used, so what is drawn is the sequence the reported numbers describe
    and not a second, subtly different one. Everything past M_max is cut: those
    models are unstable to radial oscillations and are not stars.
    """
    import os
    from eos.dd2.verify.tov import build_core_table, N_TRANSITION
    from eos.tov.solver import (compute_tov_sequence, find_mmax_precise,
                                generate_ec_logspace, CRUST_PATHS)
    try:
        core = build_core_table(par, flags)
        if core.P.size < 10:
            return np.array([]), np.array([])
        crust = "BPS" if os.path.isfile(CRUST_PATHS.get("BPS", "")) else "No"
        res = compute_tov_sequence(
            core, generate_ec_logspace(150.0, 3000.0, 120),
            add_crust_table=crust, add_crust_mode="attach",
            n_transition=(N_TRANSITION if crust != "No" else None),
            compute_baryonic_mass=False, compute_tidal=False, verbose=False,
            backend="fast", tov_parallel=False)
        i, _, _ = find_mmax_precise(res)
        return res[:i + 1, 3], res[:i + 1, 4]
    except Exception:
        return np.array([]), np.array([])


def _mr_axes(ax, title):
    """One mass-radius panel: the two targets, the observational posteriors and
    the axis furniture, so every stage of the funnel is drawn the same way and
    the narrowing is legible from one panel to the next.

    The two targets are deliberately NOT drawn as one rectangle. They constrain
    different points of the same curve — M_MAX_RANGE the peak, R14_RANGE where
    the curve crosses M = 1.4 — and a box would suggest a curve has to pass
    through it, which is not the condition being tested.
    """
    add_observational_constraints(ax)
    ax.axhspan(*M_MAX_RANGE, color="0.5", alpha=0.12, zorder=1)
    ax.plot(R14_RANGE, [1.4, 1.4], color="k", lw=3.0, solid_capstyle="butt",
            zorder=6, label=r"target $R_{1.4}$")
    ax.axhline(M_MAX_RANGE[0], color="k", ls="--", lw=1.0, zorder=5,
               label=r"target $M_{\max}$")
    ax.axhline(M_MAX_RANGE[1], color="k", ls="--", lw=1.0, zorder=5)
    ax.set_xlim(9.0, 16.0)
    ax.set_ylim(0.0, 2.8)
    ax.set_xlabel(r"$R$ [km]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_title(title, fontsize=10)
    return ax


# %% [markdown]
# ### Stage 1 — which nuclear-matter parameters are representable?

# %%
BASE_NMP = compute_nmp(Parametrization.from_dd2_defaults())
_held = {k: v for k, v in BASE_NMP.items()
         if k not in ("K_sat", "Q_sat", "L_sym")}

nmp_axes = [dict(_held, K_sat=float(_k), Q_sat=float(_q), L_sym=float(_L))
            for _k in SCAN_K_SAT for _q in SCAN_Q_SAT for _L in SCAN_L_SYM]

t0 = time.time()
stage1_rows, nmp_ok = [], []
print(f"STAGE 1 — {len(SCAN_K_SAT)} K_sat x {len(SCAN_Q_SAT)} Q_sat "
      f"x {len(SCAN_L_SYM)} L_sym = {len(nmp_axes)} nucleon NMP combinations "
      f"(a miss costs N_RESTARTS solves, so this is no longer instant)",
      flush=True)
for _s in nmp_axes:
    # Nucleon sector only: the hyperon and Delta inversions ride on top in
    # stage 2, and asking for them here would confuse "the NMPs have no DD-RMF
    # realisation" with "the sector potentials do not invert on them".
    _par, _stage, _msg = build_parametrization(_s, NUCLEON_FLAGS)
    stage1_rows.append(dict(_s, inversion_ok=float(_stage == "ok"),
                            status=_stage, message=_msg[:200]))
    if _stage == "ok":
        nmp_ok.append(_s)

# The feasible region as a picture, counting how many of the L_sym invert in
# each (K_sat, Q_sat) cell. Read it as a map of what the SOLVER reached, not of
# what exists: raise eos.dd2.nmp_inverter.N_RESTARTS and cells keep filling in.
_n_pass = Counter((r["K_sat"], r["Q_sat"])
                  for r in stage1_rows if r["inversion_ok"])
print(f"\n  (K_sat, Q_sat) cells that invert — out of {len(SCAN_L_SYM)} L_sym")
print("  K_sat |" + "".join(f"{_q:5.0f}" for _q in SCAN_Q_SAT) + "   <- Q_sat")
for _k in SCAN_K_SAT:
    print(f"  {_k:5.0f} |" + "".join(f"{_n_pass[(_k, _q)]:5d}"
                                     for _q in SCAN_Q_SAT))

print(f"\n-> {len(nmp_ok)}/{len(nmp_axes)} NMP combinations invert "
      f"({time.time()-t0:.1f} s)")
if not nmp_ok:
    raise RuntimeError("no NMP sample inverts — raise N_RESTARTS in "
                       "eos.dd2.nmp_inverter before concluding anything about "
                       "the physics, then widen SCAN_Q_SAT")

# Thin evenly so the survivors still span the grid instead of clustering at
# one end, and so stage 3 does not inherit the whole stage-1 map.
if len(nmp_ok) > MAX_NMP:
    _step = len(nmp_ok) / MAX_NMP
    nmp_ok = [nmp_ok[int(_i * _step)] for _i in range(MAX_NMP)]
    print(f"   thinned to {len(nmp_ok)} (MAX_NMP) for the expensive stages: "
          + ", ".join(f"K={s['K_sat']:.0f}/Q={s['Q_sat']:.0f}/L={s['L_sym']:.0f}"
                      for s in nmp_ok))

# %% [markdown]
# ### Stage 2 — crossed with the hyperon and Δ potentials, is the hadronic branch usable?

# %%
sector_axes = grid_samples(U_Lambda=SCAN_U_LAMBDA, U_Sigma=SCAN_U_SIGMA,
                           U_Xi=SCAN_U_XI, U_Delta=SCAN_U_DELTA,
                           x_wD=SCAN_X_WD, x_rD=SCAN_X_RD)
had_samples = [dict(_s, **_sec) for _s in nmp_ok for _sec in sector_axes]

print(f"STAGE 2 — {len(nmp_ok)} NMP x {len(sector_axes)} sector combinations "
      f"= {_budget(len(had_samples), SEC_PER_HADRONIC)}, hyperons+deltas "
      f"(one beta-eq sweep + one TOV each)", flush=True)
t0 = time.time()
had_rows = scan_hadronic(had_samples, FUNNEL_FLAGS, FUNNEL_GRID, tov=True,
                         n_jobs=FUNNEL_JOBS)

print(f"\n  K_sat  L_sym    U_L  U_Sig   U_Xi    U_D  x_wD | inv sec swp  "
      f"n_max  M_max_had  R_1.4_had | status")
for _r in had_rows:
    print(f"  {_r['K_sat']:6.1f} {_r['L_sym']:5.1f} {_r['U_Lambda']:6.1f} "
          f"{_r['U_Sigma']:6.1f} {_r['U_Xi']:6.1f} {_r['U_Delta']:6.1f} "
          f"{_r['x_wD']:5.2f} |  "
          f"{_r['inversion_ok']:.0f}   {_r['sectors_ok']:.0f}   "
          f"{_r['sweep_ok']:.0f}  {_r['n_sweep_max']:5.2f}  "
          f"{_r['M_max_had']:9.3f}  {_r['R_1p4_had']:9.2f} | {_r['status']}")

# The soft cut. Everything whose sector couplings exist and whose hadronic
# branch is not obviously hopeless goes through — see the markdown above for
# why this is not the target band.
had_ok = [(s, r) for s, r in zip(had_samples, had_rows)
          if r["sectors_ok"] == 1.0
          and (not np.isfinite(r["M_max_had"]) or r["M_max_had"] >= HAD_M_MIN)
          and (not np.isfinite(r["R_1p4_had"])
               or _in(r["R_1p4_had"], HAD_R14_RANGE))]

print(f"\n-> {len(had_ok)}/{len(had_rows)} pass the soft hadronic cut "
      f"(M_max_had >= {HAD_M_MIN} M_sun, R_1.4 in {HAD_R14_RANGE} km) "
      f"({time.time()-t0:.1f} s)")
print(f"   of which {sum(1 for _, r in had_ok if _in(r['M_max_had'], M_MAX_RANGE) and _in(r['R_1p4_had'], R14_RANGE))}"
      f" already sit in the TARGET band as pure hadronic stars")
print(f"   status breakdown: {dict(Counter(r['status'] for r in had_rows))}")
if not had_ok:
    raise RuntimeError("nothing survived stage 2; lower HAD_M_MIN or widen "
                       "HAD_R14_RANGE")

# %% [markdown]
# #### The hadronic family, as mass-radius curves
#
# Every sample that survived the soft cut, coloured by `L_sym`. That is the axis worth
# colouring by: over 30–90 MeV `L_sym` moves R(1.4) by **1.21 km** while moving M_max by
# 0.028 M_sun, so it is very nearly a pure radius knob. Nothing else comes close —
# n_sat over 0.145–0.155 fm⁻³ gives 0.40 km, K_sat over 220–270 MeV gives 0.20 km, and
# E_sym over 29–34 MeV gives 0.14 km. Read the spread horizontally as `L_sym` and
# vertically as almost everything else.
#
# The curves are rebuilt here rather than carried out of the scan, which costs one more
# TOV pass over the survivors.

# %%
t0 = time.time()
print(f"drawing {len(had_ok)} hadronic sequences "
      f"(~{len(had_ok)*SEC_PER_HADRONIC:.0f} s)", flush=True)
_pars = Parallel(n_jobs=FUNNEL_JOBS)(
    delayed(build_parametrization)(s, FUNNEL_FLAGS) for s, _ in had_ok)
_curves = Parallel(n_jobs=FUNNEL_JOBS)(
    delayed(_hadronic_mr)(p, FUNNEL_FLAGS) for p, st, _ in _pars if st == "ok")
_Ls = [s["L_sym"] for (s, _), (p, st, _) in zip(had_ok, _pars) if st == "ok"]

fig, ax = plt.subplots(figsize=(5.2, 4.4))
_mr_axes(ax, f"Stage 2 — {len(_curves)} hadronic branches, coloured by "
             r"$L_{\rm sym}$")
# A one-value L_sym axis would give a degenerate colour scale, so widen it.
_lo, _hi = min(SCAN_L_SYM), max(SCAN_L_SYM)
_norm = plt.Normalize(_lo, _hi if _hi > _lo else _lo + 1.0)
for (R, M), _L in zip(_curves, _Ls):
    if R.size:
        ax.plot(R, M, lw=1.0, alpha=0.75, zorder=4,
                color=plt.cm.viridis(_norm(_L)))
fig.colorbar(plt.cm.ScalarMappable(norm=_norm, cmap="viridis"), ax=ax,
             label=r"$L_{\rm sym}$ [MeV]")
ax.legend(loc="lower left", fontsize=7)
fig.tight_layout()
plt.show()
print(f"({time.time()-t0:.1f} s)")

# %% [markdown]
# ### Stage 3 — add quark matter: how many η work, and where does the centre sit?

# %%
good_nmp = [s for s, _ in had_ok]
had_row_of = {i: r for i, (_, r) in enumerate(had_ok)}
vmit_samples = grid_samples(B4=SCAN2_B4, a=SCAN2_A, m_s=SCAN2_MS)
ETA_GATE = min(FUNNEL_ETAS)
ETA_REST = [e for e in sorted(FUNNEL_ETAS) if e != ETA_GATE]

# ponytail: the eta = 0 (Gibbs) window is the widest one, so a combination with
# no window there has none at any higher eta and is not worth the other solves.
# The gate costs one eta and saves len(ETA_REST) on every reject. If a run ever
# needs to prove that, set ETA_GATE = 0.0 and ETA_REST = FUNNEL_ETAS to force
# every eta on every combination.
print(f"STAGE 3 — {len(good_nmp)} hadronic x {len(vmit_samples)} vMIT, gated "
      f"at eta={ETA_GATE}: {_budget(len(good_nmp)*len(vmit_samples), SEC_PER_HYBRID)}"
      f"\n   then eta in {ETA_REST} on whatever the gate passes — about half "
      f"of them historically, so budget roughly "
      f"{_budget(round(0.5*len(good_nmp)*len(vmit_samples)*len(ETA_REST)), SEC_PER_HYBRID)}"
      f" more", flush=True)
t0 = time.time()
gate_rows = scan_parameters(good_nmp, vmit_samples, FUNNEL_FLAGS, FUNNEL_GRID,
                            eta=ETA_GATE, T=0.0, tov=True, n_jobs=FUNNEL_JOBS)

pairs = [(i, s, v) for i, s in enumerate(good_nmp) for v in vmit_samples]
for _i, (_p, _r) in enumerate(zip(pairs, gate_rows)):
    _r["combo"] = float(_i)
    _r["nmp_index"] = float(_p[0])
survivors = [(i, p, r) for i, (p, r) in enumerate(zip(pairs, gate_rows))
             if r["window_exists"] == 1.0]

print(f"  gate: {len(survivors)}/{len(gate_rows)} have a window at "
      f"eta={ETA_GATE} ({time.time()-t0:.1f} s)")

t0 = time.time()
jobs = [(i, p, e) for i, p, _ in survivors for e in ETA_REST]
print(f"  {len(jobs)} remaining (combination, eta) solves")
rest_rows = list(Parallel(n_jobs=FUNNEL_JOBS, verbose=5)(
    delayed(scan_point)(p[1], p[2], FUNNEL_FLAGS, FUNNEL_GRID, eta=e, T=0.0,
                        tov=True, tov_parallel=False)
    for _, p, e in jobs))
for (_i, _p, _e), _r in zip(jobs, rest_rows):
    _r["combo"] = float(_i)
    _r["nmp_index"] = float(_p[0])

hyb_rows = gate_rows + rest_rows
by_combo = defaultdict(list)
for _r in hyb_rows:
    by_combo[int(_r["combo"])].append(_r)
for _rows in by_combo.values():
    _rows.sort(key=lambda r: r["eta"])
print(f"  {time.time()-t0:.1f} s for the remaining eta")

# %% [markdown]
# ### The answer

# %%
# One summary per (NMP, sector, vMIT) combination. A combination with no window
# is still a star — the pure hadronic one — so it is judged on the stage-2
# hadronic numbers rather than being dropped.
summary = []
for _i, _rows in by_combo.items():
    _first = _rows[0]
    _had = had_row_of[int(_first["nmp_index"])]
    if _first["window_exists"] != 1.0:
        summary.append(dict(
            combo=_i, rows=_rows, n_eta=0, n_ok=0,
            phases={"none": len(_rows)},
            M_best=_had["M_max_had"], R_best=_had["R_1p4_had"],
            hadronic_in_band=(_in(_had["M_max_had"], M_MAX_RANGE)
                              and _in(_had["R_1p4_had"], R14_RANGE))))
        continue
    _ok = [r for r in _rows if _in(r["M_max"], M_MAX_RANGE)
           and _in(r["R_1p4"], R14_RANGE)]
    _M = [r["M_max"] for r in _rows if np.isfinite(r["M_max"])]
    summary.append(dict(
        combo=_i, rows=_rows, n_eta=len(_rows), n_ok=len(_ok),
        phases=dict(Counter(r["core_phase"] or "failed" for r in _rows)),
        M_best=(max(_M) if _M else np.nan),
        R_best=next((r["R_1p4"] for r in _ok), np.nan),
        hadronic_in_band=(_in(_had["M_max_had"], M_MAX_RANGE)
                          and _in(_had["R_1p4_had"], R14_RANGE))))

_all_eta = [s for s in summary if s["n_eta"] and s["n_ok"] == s["n_eta"]]
_some_eta = [s for s in summary if 0 < s["n_ok"] < s["n_eta"]]
_no_trans = [s for s in summary if s["n_eta"] == 0]
_phase_tally = Counter()
for s in summary:
    _phase_tally.update(s["phases"])

print(f"{len(summary)} combinations reached stage 3\n")
print(f"  {len(_all_eta):4d} in the target band at EVERY eta  "
      f"(M_max in {M_MAX_RANGE} M_sun, R_1.4 in {R14_RANGE} km)")
print(f"  {len(_some_eta):4d} in the target band at SOME eta")
print(f"  {len(_no_trans):4d} have NO transition at eta={ETA_GATE} — pure "
      f"hadronic stars, of which "
      f"{sum(1 for s in _no_trans if s['hadronic_in_band'])} are in band")
print(f"\n  where the centre of the M_max star sits, over all (combination, eta):")
for _p, _n in sorted(_phase_tally.items(), key=lambda kv: -kv[1]):
    _what = {"none": "no window — pure hadronic star",
             "H": "window above every stable star — no quarks in the star",
             "mix": "MIXED-PHASE core",
             "Q": "PURE QUARK core",
             "failed": "no maximum mass (eos_unphysical or TOV failure)"}
    print(f"    {_p:7s} {_n:5d}   {_what.get(_p, '')}")
print(f"\n  status breakdown: {dict(Counter(r['status'] for r in hyb_rows))}")

# The table. Sorted by how many eta land in the band, then by maximum mass.
# The columns are the axes swept by default; U_Lambda, U_Xi and x_rD are held
# at one value each and are in the saved tables if they are ever re-opened.
print(f"\n  K_sat  Q_sat  L_sym  U_Sig    U_D  x_wD |    B4    a   m_s |  eta |"
      f" onset  offset |  M_max  R_1.4    n_Bc  core | status")
for _s in sorted(summary, key=lambda s: (-s["n_ok"], -(s["M_best"] if
                 np.isfinite(s["M_best"]) else -1)))[:FUNNEL_TOP]:
    _r0 = _s["rows"][0]
    _head = (f"  {_r0['K_sat']:6.1f} {_r0['Q_sat']:6.1f} {_r0['L_sym']:6.1f} "
             f"{_r0['U_Sigma']:6.1f} {_r0['U_Delta']:6.1f} {_r0['x_wD']:5.2f} | "
             f"{_r0['B4']:5.1f} {_r0['a']:.2f} {_r0['m_s']:5.1f} |")
    print(f"{_head} {_s['n_ok']}/{max(_s['n_eta'], 1)} eta in band")
    if _s["n_eta"] == 0:
        _had = had_row_of[int(_r0["nmp_index"])]
        print(f"{' ' * len(_head)}  --- | no transition   | "
              f"{_had['M_max_had']:6.3f} {_had['R_1p4_had']:6.2f}     "
              f"---  none | {_r0['status']}")
        continue
    for _r in _s["rows"]:
        _flag = "*" if (_in(_r["M_max"], M_MAX_RANGE)
                        and _in(_r["R_1p4"], R14_RANGE)) else " "
        print(f"{' ' * len(_head)} {_r['eta']:4.2f} | {_r['n_onset']:5.3f} "
              f"{_r['n_offset']:6.3f} | {_r['M_max']:6.3f} {_r['R_1p4']:6.2f} "
              f"{_r['n_c_max']:7.3f}  {(_r['core_phase'] or '?'):4s}{_flag}| "
              f"{_r['status']}")

print("\n  Note: an onset below ~2 n_sat is formally allowed by these checks but")
print("  physically doubtful — uniform matter is not the ground state there.")

save_table(stage1_rows, OUT / "funnel_stage1_nmp.h5")
save_table(had_rows, OUT / "funnel_stage2_hadronic.h5",
           meta=dict(flags=FUNNEL_FLAGS))
save_table(hyb_rows, OUT / "funnel_stage3_hybrid.h5",
           meta=dict(flags=FUNNEL_FLAGS))
export_csv(had_rows, OUT / "funnel_stage2_hadronic.csv")
export_csv(hyb_rows, OUT / "funnel_stage3_hybrid.csv")

# The recipe for the best combination, ready to paste into I.3 and re-run at
# the full NB resolution — the funnel deliberately runs on a coarse grid.
_best = max(summary, key=lambda s: (s["n_ok"], s["M_best"]
                                    if np.isfinite(s["M_best"]) else -1))
if _best["n_ok"]:
    _b = _best["rows"][0]
    print(f"\n{'=' * 78}\nBest combination — paste into I.3 and re-run Part II "
          f"on the full NB grid:\n")
    print(f"NMP = dict(n_sat={_b['n_sat']:.6f}, E_sat={_b['E_sat']:.2f}, "
          f"m_eff_ratio={_b['m_eff_ratio']:.4f},\n"
          f"           K_sat={_b['K_sat']:.1f}, Q_sat={_b['Q_sat']:.2f}, "
          f"E_sym={_b['E_sym']:.2f}, L_sym={_b['L_sym']:.2f})")
    print(f"PAR = Parametrization.from_nmp(NMP)")
    print(f"PAR = Parametrization.from_hyperon_potentials("
          f"U_Lambda={_b['U_Lambda']:.1f}, U_Sigma={_b['U_Sigma']:.1f}, "
          f"U_Xi={_b['U_Xi']:.1f}, base=PAR)")
    print(f"PAR = Parametrization.from_delta_potential("
          f"U_Delta={_b['U_Delta']:.1f}, x_wD={_b['x_wD']:.2f}, "
          f"x_rD={_b['x_rD']:.2f}, base=PAR)")
    print(f"VMIT = get_vmit_custom(B4={_b['B4']:.1f}, a={_b['a']:.2f}, "
          f"m_s={_b['m_s']:.1f})")
else:
    print(f"\nNothing landed in the target band. a and B^1/4 both raise the "
          f"onset, so move\nthem in OPPOSITE directions: lower SCAN2_B4 if a "
          f"is at 0.20, raise it towards\n200 if a is at 0.10. Going above "
          f"a = 0.20 does not help; the ordered window\nis gone there for any "
          f"B^1/4.")


# %% [markdown]
# #### The finalists, as mass-radius curves
#
# The same picture as the stage-2 panel, one η per colour, for the best `FUNNEL_TOP`
# combinations. Compare it against stage 2: the hadronic family was a broad fan set by
# `L_sym`, and adding the transition pulls the top of each curve — a first-order
# transition softens the equation of state above the onset, so M_max falls and the peak
# moves left, while R(1.4) is untouched whenever the onset sits above the central
# density of a 1.4 M_sun star.
#
# A curve whose peak lands inside the dashed band **and** whose M = 1.4 crossing lands
# on the thick bar is a combination that passes at that η. Solid means the centre of
# the maximum-mass star is mixed or quark; dashed means the window exists but no stable
# star reaches it.

# %%
_show = [s for s in sorted(summary, key=lambda s: (-s["n_ok"], -(s["M_best"]
         if np.isfinite(s["M_best"]) else -1))) if s["n_eta"]][:FUNNEL_TOP]
_jobs = [(s["combo"], r) for s in _show for r in s["rows"]
         if np.isfinite(r["M_max"])]
print(f"drawing {len(_jobs)} hybrid sequences "
      f"(~{len(_jobs)*SEC_PER_HYBRID:.0f} s)", flush=True)
t0 = time.time()


def _hybrid_mr(row):
    """(R, M) of one stitched hybrid sequence, stable branch only.

    Rebuilt from the row's own parameters, so the curve belongs to the numbers
    printed beside it. `mass_radius_mixed` returns the raw TOV array, which the
    scan row does not carry.
    """
    try:
        par, st, _ = build_parametrization(
            {k: row[k] for k in PAR_KEYS if k in row}, FUNNEL_FLAGS)
        if st != "ok":
            return np.array([]), np.array([])
        vm = get_vmit_custom(B4=row["B4"], a=row["a"], m_s=row["m_s"])
        res = mass_radius_mixed(par, FUNNEL_FLAGS, FUNNEL_GRID, row["eta"],
                                beta_eq_neutrinoless(), vmit_params=vm, T=0.0,
                                n_ec=120, compute_tidal=False,
                                tov_parallel=False)["results"]
        i, _, _ = find_mmax_precise(res)
        return res[:i + 1, 3], res[:i + 1, 4]
    except Exception:
        return np.array([]), np.array([])


_hyb_curves = Parallel(n_jobs=FUNNEL_JOBS)(
    delayed(_hybrid_mr)(r) for _, r in _jobs)

fig, ax = plt.subplots(figsize=(5.2, 4.4))
_mr_axes(ax, f"Stage 3 — {len(_show)} best combinations, one colour per "
             r"$\eta$")
_seen = set()
for ((_c, _r), (R, M)) in zip(_jobs, _hyb_curves):
    if not R.size:
        continue
    _lab = None if _r["eta"] in _seen else rf"$\eta$ = {_r['eta']:.2f}"
    _seen.add(_r["eta"])
    ax.plot(R, M, lw=1.2, alpha=0.85, zorder=4,
            color=FUNNEL_ETA_COLOR.get(_r["eta"], "0.3"), label=_lab,
            ls=("-" if _r["core_phase"] in ("mix", "Q") else "--"))
ax.legend(loc="lower left", fontsize=7)
fig.tight_layout()
plt.show()
print(f"({time.time()-t0:.1f} s)")


# %% [markdown]
# ## I.3 Parameters and grids
#
# Everything tunable lives in this cell. Edit, then run the rest.

# %%
# ---- hadronic parametrization -------------------------------------------
#   # 1. nucleons: invert the nuclear-matter parameters at saturation
#   NMP = dict(n_sat=0.149065, E_sat=-16.02, m_eff_ratio=0.5625,
#              K_sat=242.7, Q_sat=169.15, E_sym=31.67, L_sym=55.03)
#   PAR = Parametrization.from_nmp(NMP)          # add return_status=True to
#                                                # inspect the inversion residuals
#
#   # 2. hyperons: SU(6) vectors, scalars inverted from the potentials in SNM.
#   #    Re-solves SNM on `base`, so it adapts to the NMP nucleon sector.
#   PAR = Parametrization.from_hyperon_potentials(
#       U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0, base=PAR)
#
#   # 3. Delta: no SU(6) rule exists, so set x_iD = Gamma_iD/Gamma_iN by hand.
#   PAR = replace(PAR, x_Delta_sigma=1.15, x_Delta_omega=1.0, x_Delta_rho=1.0)

NMP = dict(n_sat=0.149077, E_sat=-16.02, m_eff_ratio=0.5625,
           K_sat=290.0, Q_sat=300.00, E_sym=31.67, L_sym=50.00)
PAR = Parametrization.from_nmp(NMP)
PAR = Parametrization.from_hyperon_potentials(U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-10.0, base=PAR)
PAR = Parametrization.from_delta_potential(U_Delta=-50.0, x_wD=1.20, x_rD=1.00, base=PAR)
VMIT = get_vmit_custom(B4=170.0, a=0.20, m_s=150.0)


# ---- which degrees of freedom exist --------------------------------------
# Every species is an explicit flag; nothing is switched on implicitly, and a
# flag that is not wired raises rather than being quietly ignored.
FLAGS = SpeciesFlags(
    hyperons=True,      # Lambda, Sigma, Xi        (needs hyperon couplings)
    deltas=True,        # Delta quartet
    muons=True,         # electrons are always present; muons optional
    phi_field=True,     # hidden-strange vector, required with hyperons
    photons=True,       # matters only at T > 0
    include_pseudoscalars=True,     # thermal pi, K, eta, eta'
    include_thermal_vectors=True,   # thermal rho, omega, K*, phi
)




# ---- equilibrium mode -----------------------------------------------------
# 'beta_eq_neutrinoless'      independent variables (nB, T)
# 'beta_eq_neutrino_trapped'  independent variables (nB, Y_L, T)
# 'fixed_YC'                  independent variables (nB, Y_C, T)
# 'fixed_YC_YS'               independent variables (nB, Y_C, Y_S, T)

MODE = "beta_eq_neutrinoless"
FIXED = {}                      # scalar values for fractions not swept as axes
Y_C_LIST = [0.05, 0.1, 0.3, 0.5]     # the Y_C axis, for the fixed-Y_C modes

# ---- grids ----------------------------------------------------------------
N_SAT = PAR.n_sat                                     # fm^-3
NB = np.linspace(0.1 * N_SAT, 12.0 * N_SAT, 300)      # baryon density [fm^-3]
T_LIST = np.concatenate([[0, 0.1], np.arange(2.5, 101., 2.5)])  # MeV
ETA_LIST = [0.0, 0.1, 0.3, 0.6, 1.0]

# ---- what to run ----------------------------------------------------------
TOV_ETAS = ETA_LIST     # TOV is beta-equilibrium, T = 0 only



# ---- parameter scan (II.4) ------------------------------------------------
# The map of where a hybrid star exists at all. Kept coarse by default; each
# (B4, a) pair costs roughly a second with TOV on.
SCAN_B4 = [150.0, 160.0, 170.0, 180.0, 190.0]     # MeV
SCAN_A = [0.05, 0.10, 0.15, 0.20]                 # fm^2 (>= 0.25 has no window)
SCAN_GRID = np.linspace(0.05, 1.6, 120)           # coarser than NB: a probe
SCAN_TOV = True                                   # also record M_max, R_1.4

# ---- rotation (III.7) -----------------------------------------------------
# Uniformly rotating models, computed by RNS through eos.tov.rotating. One
# axis-ratio scan per central density answers every J and every frequency at
# once, so the cost is ROT_N_SCAN solver runs per (eta, n_B_c) and is set by
# ROT_N_SCAN x len(ROT_NB_C), not by how many isolines you ask for.
ROT_NB_C = np.linspace(0.4, 1.6, 13)      # central baryon density [fm^-3]
ROT_J = [0.5, 1.0, 1.5, 2.0]              # angular momentum, cJ/(G M_sun^2)
ROT_FREQ = [300.0, 600.0, 900.0, 1200.0]  # spin frequency [Hz]
ROT_ETA_SHOW = 0.0                        # eta for the two isoline panels
ROT_N_SCAN = 14                           # axis ratios per central density

# The hyperon couplings live in DD2Y and nowhere else — DD2 has no entry for
# Lambda — so `hyperons=True` on `from_dd2_defaults()` fails deep in the
# coupling lookup with a bare KeyError. Catch the mismatch here instead.
if FLAGS.hyperons and not PAR.hyperon_coupling_map:
    raise ValueError("FLAGS.hyperons=True needs a parametrization that carries "
                     "hyperon couplings — from_dd2y_defaults() or "
                     "from_hyperon_potentials(); the DD2 defaults carry none")

print(f"n_sat = {N_SAT:.6f} fm^-3")
print(f"n_B  : {NB[0]:.4f} .. {NB[-1]:.4f} fm^-3  ({len(NB)} points, "
      f"{NB[0]/N_SAT:.1f}-{NB[-1]/N_SAT:.1f} n_sat)")
print(f"T    : {T_LIST}")
print(f"eta  : {ETA_LIST}")
print(f"mode : {MODE};  species: "
      f"{'hyperons ' if FLAGS.hyperons else ''}{'deltas ' if FLAGS.deltas else ''}"
      f"{'muons ' if FLAGS.muons else ''}nucleons+electrons")
print(f"vMIT : B^1/4={VMIT.B4} MeV, a={VMIT.a} fm^2, m_s={VMIT.m_s} MeV")
print(f"\nNote: below ~0.5 n_sat and at low T, uniform matter sits inside the")
print(f"liquid-gas spinodal and has no stable solution; those points are skipped.")


# %% [markdown]
#
# %% [markdown]
# # Part II — tables
#
# The build proceeds in the order the physics suggests:
#
# 1. **the two pure phases** on the whole density grid — cheap, and they are both the
#    seeds for the mixed solve and the wings of the final table;
# 2. **the transition boundaries**, found by reading χ and bracketing its 0 and 1
#    crossings;
# 3. **the mixed phase**, solved only between those boundaries, warm-started from one
#    density to the next.

# %% [markdown]
# ## II.1 Pure phases
#
# Useful on their own, and a quick check that both engines cover the grid before the
# expensive part starts.

# %%
def _columns(rows):
    """A list of flat dicts to a {name: array} table.

    String label columns (`phase`) are dropped — everything below slices and
    plots numerically, so carrying them would only force a dtype check at
    every use.
    """
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]
            if not isinstance(rows[0][k], str)}


def _quark_row(q):
    """One pure-quark point, keyed like `hadronic_row` / `composition_row`, so
    the pure wings and the mixed window concatenate in Part III without
    renaming anything. chi = 1: no hadronic matter left."""
    return dict(n_B=q.n_B, T=q.T, chi=1.0, P=q.P_total, eps=q.e_total,
                s=q.s_total, S_per_B=(q.s_total / q.n_B if q.n_B else 0.0),
                mu_B=q.mu_B, Y_C=q.Y_C, Y_S=q.Y_S,
                Y_u=q.Y_u, Y_d=q.Y_d, Y_s=q.Y_s, Y_e=q.Y_e)


t0 = time.time()
pure_hadronic = {}
for T in T_LIST:
    pts = sweep_beta_eq_octet(PAR, NB, FLAGS, T=T, stop_at_boundary=True)
    if pts:
        # `hadronic_row` keys a pure-hadronic point exactly the way
        # `composition_row` keys a mixed one. It also sums Y_C and Y_S over
        # every active baryon rather than reading them off the proton, which
        # matters the moment hyperons are switched on.
        pure_hadronic[T] = _columns([hadronic_row(p, FLAGS) for p in pts])
    print(f"  hadronic T={T:5.1f} MeV : {len(pts):3d}/{len(NB)} points", flush=True)
print(f"pure hadronic: {time.time()-t0:.1f} s\n")

t0 = time.time()
pure_quark = {}
for T in T_LIST:
    rows = []
    for n in NB:
        try:
            rows.append(_quark_row(solve_vmit_beta_eq(float(n), T, params=VMIT)))
        except Exception:
            continue
    if rows:
        pure_quark[T] = _columns(rows)
    print(f"  quark    T={T:5.1f} MeV : {len(rows):3d}/{len(NB)} points", flush=True)
print(f"pure quark: {time.time()-t0:.1f} s")


# %% [markdown]
# ## II.2 Mixed tables
#
# One table per η (η changes the size of the unknown vector, so it is looped outside
# rather than being an axis). Each line prints its transition window, how many points
# it converged, and the cost per point.
#
# `MODE` selects which charges are globally conserved, and with them the independent
# variables of the table:
#
# | `MODE` | globally conserved | independent variables | where it applies |
# |---|---|---|---|
# | `beta_eq_neutrinoless` | B | (n_B, T, η) | cold compact stars |
# | `beta_eq_neutrino_trapped` | B, L | (n_B, Y_L, T, η) | early neutrino-trapped stages of protoneutron stars |
# | `fixed_YC` | B, C | (n_B, Y_C, T, η) | EOS tables for CCSN and BNSM simulations, where β reactions are not always equilibrated |
# | `fixed_YC_YS` | B, C, S | (n_B, Y_C, Y_S, T, η) | timescales short compared with the weak strangeness-changing one — heavy-ion collisions, or the first critical droplet in nucleation |
#
# In the first three modes weak equilibrium with respect to non-leptonic
# strangeness-changing reactions is established *separately* in each phase, μ_S^H =
# μ_S^Q = 0; in the last, global conservation of strangeness ties the phases together
# through a common and generally nonvanishing μ_S^H = μ_S^Q.

# %%
def progress(info):
    """Called once per axis combination by build_mixed_table."""
    w = info["window"]
    span = (f"[{w.n_onset:.4f}, {w.n_offset:.4f}] fm^-3"
            if (w is not None and w.exists) else "no transition")
    per = 1e3 * info["seconds"] / max(info["n_points"], 1)
    extra = "".join(f" {k}={v:.3f}" for k, v in info["fractions"].items())
    print(f"    {info['temp_key']}={info['temp']:5.1f}{extra} | window {span:32s}"
          f" | {info['n_points']:3d} pts | {info['seconds']:6.1f} s"
          f" | {per:6.1f} ms/pt", flush=True)


tables, windows_by_eta = {}, {}
grand_total = time.time()
for eta in sorted(ETA_LIST):          # ascending, so each eta seeds the next
    print(f"\neta = {eta}")
    t0 = time.time()
    axes = {"nB": NB, "T": T_LIST}
    if MODE in ("fixed_YC", "fixed_YC_YS"):
        axes["Y_C"] = Y_C_LIST
    spec = MixedTableSpec(PAR, FLAGS, MODE, axes=axes, eta=eta,
                          vmit_params=VMIT, fixed=FIXED, leptons=True)
    rows, windows = build_mixed_table(spec, progress=progress)
    tables[eta], windows_by_eta[eta] = rows, windows

    meta = dict(mode=MODE, eta=eta, parametrization=PAR, flags=FLAGS, vmit=VMIT,
                nB_grid=NB, T_grid=T_LIST)
    path = OUT / f"mixed_{MODE}_eta{eta:.2f}.h5"
    save_table(rows, path, meta=meta, windows=windows)
    export_csv(rows, path.with_suffix(".csv"), meta=meta)
    print(f"  -> {len(rows)} rows in {time.time()-t0:.1f} s, saved {path.name}")

print(f"\nTOTAL: {time.time()-grand_total:.1f} s for "
      f"{sum(len(r) for r in tables.values())} mixed points")

# %% [markdown]
# ## II.3 TOV
#
# The stitched core EOS — pure hadronic below the onset, mixed through the window,
# pure quark above the offset — integrated to a mass-radius sequence.
#
# **This is always cold, neutrinoless β-equilibrated matter, whatever `MODE` is set
# to.** A neutron-star core is cold and neutrino-transparent; the fixed-charge-fraction
# and neutrino-trapped modes describe the conditions of CCSN and BNSM matter, not of a
# cold star, so running TOV on them would answer a different question than the one the
# tables above answer. The mass-radius curve below therefore does *not* change when you
# change `MODE`.
#
# A Maxwell (η=1) table carries a constant-pressure plateau, and `eos.tov` detects it
# and applies the tidal correction across the density discontinuity by itself.
# The integration uses the Numba backend by default (`backend="fast"`), which agrees
# with the scipy reference to ~1e-4 M_sun on M_max; pass `backend="scipy"` to check.

# %%
tov = {}
for eta in TOV_ETAS:
    t0 = time.time()
    core = build_mixed_eos_table(PAR, FLAGS, NB, eta, beta_eq_neutrinoless(),
                                 vmit_params=VMIT, T=0.0)
    res = mass_radius_mixed(PAR, FLAGS, NB, eta, beta_eq_neutrinoless(),
                            vmit_params=VMIT, T=0.0, table=core, n_ec=120)
    # Keep the stitched table: III.7 rotates *this* equation of state, and
    # rebuilding it there would risk rotating something subtly different.
    tov[eta] = dict(res, core=core)
    trans = (f"onset {core.n_onset:.3f} offset {core.n_offset:.3f} fm^-3"
             if core.has_transition else "no transition")
    print(f"eta={eta}: {trans} | M_max={res['M_max']:.3f} Msun "
          f"R(M_max)={res['R_Mmax']:.2f} km R(1.4)={res['R_1p4']:.2f} km "
          f"| {time.time()-t0:.1f} s", flush=True)

# %% [markdown]
# ## II.4 Parameter scan — where does a hybrid star exist?
#
# The pre-flight answers the question for *one* combination; this maps a whole plane
# of them. For each (B^1/4, a) the scan runs two checks in the order that fails
# cheapest, and records the outcome rather than raising — the failures are the
# boundary being mapped:
#
# 1. **is the hadronic model representable?** The nuclear-matter parameters are
#    inverted back to DD2 couplings, and not every combination has a solution;
# 2. **is there a window?** χ must cross both 0 and 1 on the grid.
#
# With `SCAN_TOV` on it also records M_max, R(1.4) and max c_s², so the map answers
# the question that actually matters when choosing parameters: not just "is there a
# transition" but "is the resulting star heavy enough to be allowed".
#
# The nuclear-matter parameters are held at the current `PAR`'s own values here, so
# the plane is purely the quark sector — this is the (B^1/4, a) picture of the one
# parametrization chosen in I.3, not a search. The search over the hadronic axes is
# I.2's job.

# %%
# The NMPs of the parametrization chosen in I.3, so this plane is purely the
# quark sector. (I.2 scans the NMPs and the sector potentials themselves.)
NMP_OF_PAR = compute_nmp(PAR)


def scan_progress(r):
    win = (f"[{r['n_onset']:.3f}, {r['n_offset']:.3f}]"
           if r["window_exists"] else "—")
    extra = (f" M_max={r['M_max']:5.3f} R_1.4={r['R_1p4']:5.2f}"
             if "M_max" in r and np.isfinite(r["M_max"]) else "")
    print(f"  B4={r['B4']:5.1f} a={r['a']:.2f} | {r['status']:16s} "
          f"{win:20s}{extra} | {r['seconds']:5.2f} s", flush=True)


t0 = time.time()
scan_rows = scan_parameters(
    [NMP_OF_PAR], grid_samples(B4=SCAN_B4, a=SCAN_A), FLAGS, SCAN_GRID,
    eta=0.0, T=0.0, tov=SCAN_TOV, n_jobs=1, progress=scan_progress)

_n_ok = sum(r["status"] == "ok" for r in scan_rows)
print(f"\n{_n_ok}/{len(scan_rows)} combinations give a complete transition "
      f"({time.time()-t0:.1f} s)")
if SCAN_TOV:
    _heavy = [r for r in scan_rows
              if np.isfinite(r.get("M_max", np.nan)) and r["M_max"] >= 2.0]
    print(f"{len(_heavy)}/{len(scan_rows)} also reach M_max >= 2.0 Msun")

save_table(scan_rows, OUT / "scan_B4_a.h5", meta=dict(flags=FLAGS, eta=0.0))
export_csv(scan_rows, OUT / "scan_B4_a.csv")

# %% [markdown]
# # Part III — plots
#
# All plotting is defined here. The mixed tables are re-read from disk, so Part II.2 need
# not be repeated; the pure wings come from Part II.1, which is the cheap cell.
#
# Everything below plots the **complete EOS** — pure hadronic, mixed phase, pure quark —
# not the mixed phase alone. `full_eos` does the joining.
#

# %%
loaded = {}
for eta in ETA_LIST:
    path = OUT / f"mixed_{MODE}_eta{eta:.2f}.h5"
    if path.is_file():
        cols, meta, wins = load_table(path)
        loaded[eta] = cols
        if not cols:
            print(f"  eta={eta}: table is empty — no mixed point converged, so"
                  f" the equation of state below is the pure hadronic branch")
print("loaded eta values:", sorted(loaded))

ETA_COLORS = plt.cm.viridis(np.linspace(0.0, 0.88, len(ETA_LIST)))
COLOR_OF = {eta: ETA_COLORS[i] for i, eta in enumerate(ETA_LIST)}


def at_temperature(cols, T, tol=1e-6):
    """Rows of one loaded table at a single temperature, sorted by density.

    An eta whose window held no converged point saves an empty table; that is a
    legitimate outcome, not an error, so it comes back as no rows rather than
    raising.
    """
    if "T" not in cols:
        return {}
    m = np.abs(cols["T"] - T) < tol
    order = np.argsort(cols["n_B"][m])
    return {k: v[m][order] for k, v in cols.items()
            if v.ndim == 1 and v.dtype.kind == "f"}


def full_eos(eta, T):
    """The complete equation of state at one (eta, T), as {name: array}.

    Three segments joined on density: pure hadronic below the onset, the
    eta-mixed phase through the window, pure quark above the offset. The
    boundaries are read off the mixed table itself — it was built only between
    the two chi crossings — so the segments meet by construction and nothing is
    interpolated to close a gap. With no transition the mixed table is empty
    and the result is pure hadronic throughout, which is the correct answer.

    A column absent from a segment (a hadron fraction in the quark wing, say)
    is filled with nan there, not zero, so a logarithmic plot ends the curve
    instead of sending it to the floor.
    """
    mix = at_temperature(loaded[eta], T) if eta in loaded else {}
    n_lo, n_hi = ((mix["n_B"].min(), mix["n_B"].max())
                  if mix and mix["n_B"].size else (np.inf, np.inf))

    segs = []
    had, qk = pure_hadronic.get(T), pure_quark.get(T)
    if had is not None:
        segs.append({k: v[had["n_B"] < n_lo] for k, v in had.items()})
    if mix and mix["n_B"].size:
        segs.append(mix)
    if qk is not None:
        segs.append({k: v[qk["n_B"] > n_hi] for k, v in qk.items()})
    segs = [s for s in segs if s["n_B"].size]
    if not segs:
        return {}

    out = {}
    for k in set().union(*(s.keys() for s in segs)):
        out[k] = np.concatenate(
            [s[k] if k in s else np.full(s["n_B"].size, np.nan) for s in segs])
    order = np.argsort(out["n_B"])
    return {k: v[order] for k, v in out.items()}



# %% [markdown]
# ## III.1 Pressure, entropy per baryon, and quark fraction vs density
#
# One panel each, at a chosen temperature, with one colour per η — the whole EOS,
# hadronic wing through mixed phase through quark wing. The χ panel says which segment
# you are looking at: flat at 0 is hadronic, rising is mixed, flat at 1 is quark.
# The two pure branches are drawn dotted in grey underneath, continued past the
# transition where they are metastable, so the gain from the transition is visible.
#

# %%
def plot_eos_panels(T, show_pure=True):
    """P, S/n_B and chi against density at one temperature, one colour per eta.

    The pure branches are drawn dotted underneath and continued past the
    transition, where they are metastable, so the gain from the transition is
    visible rather than implied.
    """
    fig, axes = setup_scientific_figure(nrows=1, ncols=3, figsize=(15, 4.6))
    for eta in sorted(loaded):
        d = full_eos(eta, T)
        if not d:
            continue
        c = COLOR_OF[eta]
        for ax, key in zip(axes, ("P", "S_per_B", "chi")):
            ax.plot(d["n_B"], d[key], "-", color=c, label=rf"$\eta={eta}$")

    if show_pure:
        for pure, lab in ((pure_hadronic.get(T), "pure hadronic"),
                          (pure_quark.get(T), "pure quark")):
            if pure is None or not pure["n_B"].size:
                continue
            axes[0].plot(pure["n_B"], pure["P"], ":", lw=1.4,
                         color=STANDARD_COLORS["Gray"], label=lab)
            axes[1].plot(pure["n_B"], pure["S_per_B"], ":", lw=1.4,
                         color=STANDARD_COLORS["Gray"])

    axes[0].set_ylabel(LABELS["P"])
    axes[1].set_ylabel(LABELS["S"] + r"  [$k_B$ / baryon]")
    axes[2].set_ylabel(r"quark volume fraction  $\chi$")
    axes[2].set_ylim(-0.02, 1.02)
    for y in (0.0, 1.0):
        axes[2].axhline(y, color=STANDARD_COLORS["Gray"], lw=0.8, ls="--")
    for ax in axes:
        ax.set_xlabel(LABELS["nB"])
        apply_style(ax)
    add_panel_labels(axes)
    fig.suptitle(rf"DD2+vMIT, {MODE}, $T = {T:g}$ MeV", y=1.02)
    fig.tight_layout()
    plt.show()
    return fig


plot_eos_panels(0.0)


# %% [markdown]
# At T = 0 the entropy panel is identically zero. The same three panels at a finite
# temperature from `T_LIST`:

# %%
plot_eos_panels(20.0, show_pure=False)


# %% [markdown]
# ## III.2 Phase boundaries in the (n_B, T) plane
#
# For each η, the density at which the quark phase appears (χ = 0, onset) and the one
# at which the hadronic phase disappears (χ = 1, offset), as functions of temperature.
# The shaded band between them is the mixed phase. A wide band is Gibbs-like, the two
# phases sharing charge freely under GCN; the band narrows as η rises and the charged
# leptons are progressively tied to their own phase, until at η = 1 it becomes the
# density jump of the Maxwell construction.

# %%
fig, ax = setup_scientific_figure()
for eta in ETA_LIST:
    wins = windows_by_eta.get(eta, {})
    Ts = sorted(k[0] for k in wins)
    onset = np.array([wins[(T,)].n_onset for T in Ts], dtype=float)
    offset = np.array([wins[(T,)].n_offset for T in Ts], dtype=float)
    good = np.isfinite(onset) & np.isfinite(offset)
    if not good.any():
        print(f"eta={eta}: no transition at any temperature")
        continue
    Ts = np.array(Ts, dtype=float)
    c = COLOR_OF[eta]
    ax.plot(onset[good], Ts[good], "-o", ms=4, color=c, label=rf"$\eta={eta}$")
    ax.plot(offset[good], Ts[good], "--s", ms=4, color=c)
    ax.fill_betweenx(Ts[good], onset[good], offset[good], color=c, alpha=0.13)

ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(LABELS["T"])
ax.set_title("Mixed-phase boundaries\n"
             "solid: onset ($\\chi=0$)   dashed: offset ($\\chi=1$)")
apply_style(ax)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## III.3 Mass-radius and tidal deformability
#
# Cold, neutrinoless β-equilibrated stars built on the stitched core EOS, drawn over
# the NICER and HESS credible regions and the PSR J0952-0607 mass band.
#
# The overlays come from `eos.general.observational_constraints`, which reads contours
# precomputed offline and shipped inside the package — nothing is refitted here. They
# are drawn at low `zorder` so the model curves stay on top. Set `INLINE_LABELS=True`
# to write each source name next to its blob instead of filling the legend with them.

# %%
INLINE_LABELS = True

fig, axes = setup_scientific_figure(nrows=1, ncols=2, figsize=(12.5, 5))

# Constraints first, so the model curves are drawn over them.
add_observational_constraints(axes[0], inline_labels=INLINE_LABELS)

for eta, res in tov.items():
    r = res["results"]
    idx, _, M_max = find_mmax_precise(r)
    M, R, Lam = r[:idx + 1, 4], r[:idx + 1, 3], r[:idx + 1, 6]
    c = COLOR_OF.get(eta, STANDARD_COLORS["Gray"])
    axes[0].plot(R, M, "-", color=c, zorder=4,
                 label=rf"$\eta={eta}$  ($M_{{\max}}={M_max:.2f}\,M_\odot$)")
    axes[0].plot(res["R_Mmax"], M_max, "o", ms=6, color=c, zorder=5)
    axes[1].semilogy(M, Lam, "-", color=c, label=rf"$\eta={eta}$")

axes[0].set_xlabel(r"$R$  [km]")
axes[0].set_ylabel(r"$M$  [$M_\odot$]")
axes[0].set_title("Mass-radius")
# Pin the frame: the constraint blobs would otherwise stretch the axes out to
# wherever the widest posterior reaches.
axes[0].set_xlim(9.0, 16.0)
axes[0].set_ylim(0.5, 2.8)
axes[1].set_xlabel(r"$M$  [$M_\odot$]")
axes[1].set_ylabel(r"$\Lambda$")
axes[1].set_title("Tidal deformability")
for ax in axes:
    apply_style(ax)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## III.4 The two sound speeds
#
# The clearest single picture of what η does.
#
# **c_eq² = dP/dε** is taken along the equilibrium sequence, where the quark volume
# fraction χ is free to readjust. A compression is then answered by converting hadrons
# into quarks instead of by raising the pressure, so c_eq **dips** through a Gibbs
# window and **collapses to zero** through a Maxwell one, where the pressure remains
# constant by construction. This is the sound speed that enters the TOV equations.
#
# **c_ad² (frozen)** holds χ fixed — the mixture is compressed faster than one phase
# can convert into the other — so the pressure has to rise and c_ad does *not*
# collapse. Freezing χ is the part that matters; freezing only the charge fractions
# would let the solve slide back onto the plateau.
#
# The grey band is the causal bound c² ≤ 1. Both curves respect it; the interesting
# feature is the gap between them, which widens with η.
#
# `sound_speed_frozen` re-solves each phase twice per point, so this cell costs a few
# seconds per η — it is the only expensive plot in Part III.

# %%
CS_ETAS = [e for e in (0.0, 0.3, 1.0) if e in ETA_LIST]

fig, axes = setup_scientific_figure(nrows=1, ncols=2, figsize=(13, 5),
                                    sharey=True)
for eta in CS_ETAS:
    c = COLOR_OF.get(eta, STANDARD_COLORS["Gray"])
    w = locate_window(PAR, FLAGS, NB, eta, beta_eq_neutrinoless(),
                      vmit_params=VMIT, T=0.0)
    if not w.exists:
        print(f"eta={eta}: no window, skipped")
        continue
    # Equilibrium: read straight off the stitched table, which already spans the
    # hadronic wing, the window and the quark wing.
    core = build_mixed_eos_table(PAR, FLAGS, NB, eta, beta_eq_neutrinoless(),
                                 vmit_params=VMIT, T=0.0, window=w)
    axes[0].plot(core.n_B, sound_speed_eq(core.P, core.eps), "-", color=c,
                 label=rf"$\eta={eta}$")

    # Frozen: only defined where there are two phases, so it is drawn across the
    # window and its two endpoints.
    inside = NB[(NB >= w.n_onset) & (NB <= w.n_offset)]
    rs = sweep_mixed(PAR, FLAGS, inside, eta, beta_eq_neutrinoless(),
                     vmit_params=VMIT, T=0.0)
    n_mix = np.array([r.n_B for r in rs])
    axes[1].plot(n_mix, frozen_along(PAR, FLAGS, rs, vmit_params=VMIT),
                 "-", color=c, label=rf"$\eta={eta}$")
    axes[1].plot(n_mix, sound_speed_eq(np.array([r.P for r in rs]),
                                       np.array([r.eps for r in rs])),
                 "--", lw=1.4, color=c, alpha=0.7)
    for ax in axes:
        ax.axvspan(w.n_onset, w.n_offset, color=c, alpha=0.07, zorder=0)

for ax, title in zip(axes, ("equilibrium  $c_{\\rm eq}^2$  (full EoS)",
                            "frozen  $c_{\\rm ad}^2$  (solid) vs "
                            "$c_{\\rm eq}^2$ (dashed), window only")):
    ax.axhline(1.0, color=STANDARD_COLORS["Gray"], lw=0.9, ls="-.")
    ax.axhline(0.0, color=STANDARD_COLORS["Gray"], lw=0.8, ls="--")
    ax.set_xlabel(LABELS["nB"])
    ax.set_title(title)
    apply_style(ax)
axes[0].set_ylabel(r"$c_s^2$  [$c^2$]")
axes[0].set_ylim(-0.05, 1.05)
add_panel_labels(axes)
fig.suptitle(rf"Sound speeds — DD2+vMIT, $T = 0$ MeV", y=1.01)
fig.tight_layout()
plt.show()


# %% [markdown]
# ## III.5 Parameter map — where the hybrid star lives
#
# The II.4 scan as a picture. Colour is the onset density where a complete transition
# exists; hatched cells have none. With `SCAN_TOV` on, the contour marks
# M_max = 2.0 M_sun, so the region that is both hybrid *and* heavy enough to be
# allowed is the coloured area on the heavy side of that line.

# %%
_B4 = np.array(sorted({r["B4"] for r in scan_rows}))
_A = np.array(sorted({r["a"] for r in scan_rows}))
_shape = (len(_A), len(_B4))
_onset = np.full(_shape, np.nan)
_mmax = np.full(_shape, np.nan)
for r in scan_rows:
    i, j = np.searchsorted(_A, r["a"]), np.searchsorted(_B4, r["B4"])
    if r["window_exists"]:
        _onset[i, j] = r["n_onset"]
    _mmax[i, j] = r.get("M_max", np.nan)

fig, ax = setup_scientific_figure(figsize=(8, 5.5))
# Sequential field -> viridis, per the palette convention in figure_style.
_im = ax.pcolormesh(_B4, _A, _onset, shading="nearest", cmap="viridis")
fig.colorbar(_im, ax=ax, label=r"onset density  $n_{\rm onset}$  [fm$^{-3}$]")
# Mark the combinations with no transition, which nan leaves blank.
for i, a in enumerate(_A):
    for j, b in enumerate(_B4):
        if not np.isfinite(_onset[i, j]):
            ax.plot(b, a, "x", color=STANDARD_COLORS["Gray"], ms=9, mew=1.8)
if SCAN_TOV and np.isfinite(_mmax).any():
    _red = STANDARD_COLORS["Red"]
    ax.contour(_B4, _A, _mmax, levels=[2.0], colors=[_red], linewidths=2.2)
    ax.plot([], [], "-", color=_red, lw=2.2, label=r"$M_{\max}=2\,M_\odot$")
ax.plot(VMIT.B4, VMIT.a, "*", color="white", ms=18, mec="k", mew=1.2,
        label="current VMIT")
ax.set_xlabel(r"$B^{1/4}$  [MeV]")
ax.set_ylabel(r"vector coupling  $a$  [fm$^2$]")
ax.set_title("Where a complete transition exists  (x = none)")
# A heatmap wants no gridlines over it; the legend still comes from apply_style.
apply_style(ax, grid=False, minor_grid=False)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
plt.show()


# %% [markdown]
# ## III.6 Composition through the transition
#
# The particle fractions of the mixed phase, obtained from the corresponding densities:
# Y_h = (1−χ) n_h^H / n_B for the hadrons and Y_q = χ n_q^Q / n_B for the quarks, so
# that the curves sum consistently across the mixed phase. Hadrons solid, quarks
# dashed. Shown across the whole density range: the hadrons start at their pure-phase
# values, hand over through the window, and are gone above the offset, where the quarks
# reach theirs. The shading marks the mixed phase.
#

# %%
ETA_COMP = 0.0
T_COMP = 20.0
Y_FLOOR = 1e-3

d = full_eos(ETA_COMP, T_COMP)
species = sorted(k[2:] for k in d
                 if k.startswith("Y_") and k not in ("Y_C", "Y_S"))

fig, ax = setup_scientific_figure(figsize=(8, 5.5))
for sp in species:
    y = d[f"Y_{sp}"]
    if not np.isfinite(y).any() or np.nanmax(y) < Y_FLOOR:
        continue
    # Colour by particle and linestyle by multiplet, from the shared table:
    # nucleons solid, hyperons dashed, Deltas dash-dot, quarks densely dashed,
    # leptons dotted — so the figure survives being printed in black and white.
    colour, style = particle_style(sp)
    ax.plot(d["n_B"], y, ls=style, color=colour, label=sp)

mixed = (d["chi"] > 0.0) & (d["chi"] < 1.0)
if mixed.any():
    ax.axvspan(d["n_B"][mixed].min(), d["n_B"][mixed].max(),
               color=STANDARD_COLORS["Gray"], alpha=0.15, zorder=0,
               label="mixed")
ax.set_yscale("log")
ax.set_ylim(Y_FLOOR, 2.0)
ax.set_xlabel(LABELS["nB"])
ax.set_ylabel(LABELS["Y_i"] + r"  (volume-weighted)")
ax.set_title(rf"Composition — $\eta={ETA_COMP}$, $T={T_COMP:g}$ MeV")
# Minor gridlines are noise under a log axis with this many curves.
apply_style(ax, minor_grid=False)
ax.legend(ncol=3, fontsize=8, frameon=False)
fig.tight_layout()
plt.show()


# %% [markdown]
# ## III.7 Rotation — constant-J and constant-frequency sequences
#
# Uniformly rotating, axisymmetric models of the *same* stitched EOSs Part II.3
# integrated, computed with the Komatsu–Eriguchi–Hachisu self-consistent
# field method as implemented in RNS (Stergioulas & Friedman 1995, ApJ 444, 306),
# driven through `eos.tov.rotating`.
#
# **Read this before trusting a number.** `rotating_grid` never asks the solver for a
# physical target directly. It scans the axis ratio r_p/r_e — where every model
# converges — and inverts the resulting curve in Python, because M, M_0, Ω and J are
# all monotone in r_p/r_e at fixed central density. Two consequences:
#
# * one scan answers every J *and* every frequency, so adding isolines is free and
#   the cost is set by `ROT_N_SCAN` × `len(ROT_NB_C)` alone;
# * the values on the isolines are **interpolated between converged models**, not
#   themselves converged models. Call `rotating_model` for a point that has to be
#   exact.
#
# A target beyond the Keplerian limit of its central density comes back as NaN rather
# than an error, which is why the high-frequency isolines simply stop at low n_B,c —
# those stars cannot spin that fast without shedding mass at the equator.
#
# The shaded band marks central densities where the star has a **mixed core**: n_B,c
# between the onset and the offset. Note that the shading is in *central* density, so
# a star to the right of the band has a pure quark core, not no quark core.

# %%
rot_kepler, rot_iso = {}, {}

if not have_rns():
    print("No `rns` executable found — III.7 skipped.\n"
          "eos.tov.rns_backend.find_rns_binary() looks on PATH and at the usual\n"
          "build locations; point it at your build or pass rns_path= to run it.")
else:
    for eta in TOV_ETAS:
        t0 = time.time()
        core = tov[eta]["core"]
        eos_rot = core.to_tov()
        # The solver is parametrised by central ENERGY density; the figures want
        # central BARYON density, so map across on the table that produced both.
        e_c = np.interp(ROT_NB_C, core.n_B, core.eps)
        rot_kepler[eta] = kepler_sequence(eos_rot, e_c, parallel=True)

        extra = ""
        if eta == ROT_ETA_SHOW:
            # Only the displayed eta needs the isolines; every eta needs its
            # Keplerian limit for the ratio panel below.
            rot_iso["J"] = rotating_grid(eos_rot, e_c, J_grid=ROT_J,
                                         n_scan=ROT_N_SCAN, parallel=True)
            rot_iso["freq"] = rotating_grid(eos_rot, e_c, freq_grid=ROT_FREQ,
                                            n_scan=ROT_N_SCAN, parallel=True)
            extra = (f"  (+{len(ROT_J)} J and {len(ROT_FREQ)} frequency "
                     f"sequences)")

        M_kep = np.nanmax(rot_kepler[eta][:, KEPLER_COLUMNS.index("M")])
        f_kep = np.nanmax(rot_kepler[eta][:, KEPLER_COLUMNS.index("freq")])
        print(f"eta={eta}: M_max^Kepler={M_kep:.3f} Msun  "
              f"M_max^TOV={tov[eta]['M_max']:.3f} Msun  "
              f"ratio={M_kep / tov[eta]['M_max']:.3f}  "
              f"f_K(max)={f_kep:.0f} Hz | {time.time() - t0:.1f} s{extra}",
              flush=True)

# %%
if rot_iso:
    ICOL = {c: k for k, c in enumerate(GRID_COLUMNS)}
    KCOL = {c: k for k, c in enumerate(KEPLER_COLUMNS)}
    core = tov[ROT_ETA_SHOW]["core"]
    static = tov[ROT_ETA_SHOW]["results"]      # columns e_c, n_c, P_c, R, M, ...
    kep = rot_kepler[ROT_ETA_SHOW]

    fig, axes = setup_scientific_figure(nrows=1, ncols=2, figsize=(12.8, 5.2),
                                        sharey=True)
    panels = (("J", ROT_J, lambda v: rf"$cJ/GM_\odot^2 = {v:g}$",
               "constant angular momentum"),
              ("freq", ROT_FREQ, lambda v: rf"$f = {v:g}$ Hz",
               "constant frequency"))

    for ax, (key, targets, label_of, title) in zip(axes, panels):
        # rotating_grid stacks one block of targets per central density.
        grid = rot_iso[key].reshape(len(ROT_NB_C), len(targets), -1)

        if core.has_transition:
            ax.axvspan(core.n_onset, core.n_offset,
                       color=STANDARD_COLORS["Gray"], alpha=0.15, zorder=0,
                       label="mixed core")
        # The static and Keplerian curves bound the whole rotating family.
        ax.plot(static[:, 1], static[:, 4], color="black", lw=1.8, zorder=5,
                label=r"static ($J=0$)")
        ax.plot(ROT_NB_C, kep[:, KCOL["M"]], color=STANDARD_COLORS["Gray"],
                lw=1.8, ls="--", zorder=5, label="Kepler limit")

        # J and f are continuous, so a sequential map rather than categorical.
        shades = plt.cm.viridis(np.linspace(0.12, 0.86, len(targets)))
        for k, value in enumerate(targets):
            M = grid[:, k, ICOL["M"]]
            good = np.isfinite(M)
            if good.any():
                ax.plot(ROT_NB_C[good], M[good], "-o", ms=3.5, lw=1.5,
                        color=shades[k], zorder=4, label=label_of(value))

        ax.set_xlabel(LABELS["nB"] + r"  (central)")
        ax.set_title(title)
        apply_style(ax)
        ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    axes[0].set_ylabel(r"$M\ [M_\odot]$")
    add_panel_labels(axes)
    fig.suptitle(rf"Rotating hybrid stars, $\eta = {ROT_ETA_SHOW:g}$", y=1.01)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ### M_max at the Kepler limit, against the static value
#
# The left panel is the quantity of interest: how much extra mass uniform rotation
# supports. For nucleonic and hyperonic EOSs this ratio is famously insensitive to
# the microphysics — Breu & Rezzolla (2016, MNRAS 459, 646) find
# 1.203 ± 0.022 across a wide set of tables, and that band is drawn for reference.
#
# **The ratio comes out flat in η, and that is not the plot failing to do anything.**
# η is a phase-construction parameter, not a stiffness parameter: it reshapes the
# EOS *inside* the mixed phase and nowhere else. The maximum-mass
# configuration sits at a central density above the offset, where every η is pure
# quark matter and the tables coincide exactly — so M_max is set by a branch η never
# touches. The η=0 and η=1 tables here differ by up to 30 MeV/fm³ in pressure and
# still give Keplerian maxima within 0.005 M_sun of each other, while the *shape* of
# the M(n_B,c) curve at intermediate densities does differ between them.
#
# `M_max^Kepler` is a maximum over the `ROT_NB_C` grid, so it is resolved only as well
# as that grid is — the curve is flat near its peak, which helps, but a number that
# has to be exact wants a finer grid or a `rotating_model` call.

# %%
if rot_kepler:
    etas = sorted(rot_kepler)
    M_kep = np.array([np.nanmax(rot_kepler[e][:, KEPLER_COLUMNS.index("M")])
                      for e in etas])
    M_tov = np.array([tov[e]["M_max"] for e in etas])

    fig, axes = setup_scientific_figure(nrows=1, ncols=2, figsize=(12.0, 4.8))

    axes[0].axhspan(1.203 - 0.022, 1.203 + 0.022, color=STANDARD_COLORS["Gray"],
                    alpha=0.25, zorder=0, label="Breu & Rezzolla (2016)")
    axes[0].axhline(1.203, color=STANDARD_COLORS["Gray"], ls="--", lw=1.3,
                    zorder=1)
    for k, eta in enumerate(etas):
        axes[0].plot(eta, M_kep[k] / M_tov[k], "o", ms=8, zorder=4,
                     color=COLOR_OF.get(eta, STANDARD_COLORS["Gray"]))
    axes[0].plot(etas, M_kep / M_tov, "-", lw=1.4, zorder=3, color="black",
                 label="this model")
    axes[0].set_ylabel(r"$M_{\max}^{\rm Kepler} / M_{\max}^{\rm TOV}$")
    # Pinned wider than the reference band: the model values span ~0.001, so on
    # autoscale the band would fill the frame and stop reading as a band.
    axes[0].set_ylim(1.15, 1.26)
    axes[0].legend(fontsize=9, frameon=False, loc="lower right")

    axes[1].plot(etas, M_kep, "-o", ms=6, color=STANDARD_COLORS["Blue"],
                 label=r"$M_{\max}^{\rm Kepler}$")
    axes[1].plot(etas, M_tov, "-s", ms=6, color=STANDARD_COLORS["Orange"],
                 label=r"$M_{\max}^{\rm TOV}$")
    axes[1].axhline(2.0, color=STANDARD_COLORS["Gray"], ls=":", lw=1.3,
                    label=r"$2\,M_\odot$")
    axes[1].set_ylabel(r"$M_{\max}\ [M_\odot]$")
    axes[1].legend(fontsize=9, frameon=False)

    for ax in axes:
        ax.set_xlabel(r"$\eta$")
        apply_style(ax)
    add_panel_labels(axes)
    fig.tight_layout()
    plt.show()

    print(f"{'eta':>5} {'M_TOV':>8} {'M_Kepler':>9} {'ratio':>7}")
    for k, eta in enumerate(etas):
        print(f"{eta:5.2f} {M_tov[k]:8.3f} {M_kep[k]:9.3f} "
              f"{M_kep[k] / M_tov[k]:7.3f}")

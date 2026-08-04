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
# # DD2 + vMIT — general first-order phase transition
#
# A hadronic equation of state (**DD2**, a density-dependent relativistic mean field)
# and a quark equation of state (**vMIT**, a vector-enhanced bag model) joined across a
# first-order phase transition, with the character of that transition controlled by a
# single continuous parameter **η**.
#
# **What η means.** In a mixed phase the two phases must be electrically neutral
# *somewhere*, and the question is where. η is the fraction of neutrality imposed
# **locally**, inside each phase, rather than **globally**, on the volume average:
#
# | η | construction | behaviour through the mixed window |
# |---|---|---|
# | 0 | **Gibbs** | only the average is neutral; the phases exchange charge freely and the pressure rises continuously |
# | 1 | **Maxwell** | each phase is separately neutral; no charge is exchanged, and the window collapses to a constant-pressure plateau with a density jump |
# | in between | — | stands in for the finite surface tension and Coulomb cost of the mixed-phase structures |
#
# **How a density is classified.** The quark volume fraction χ comes out of the solve
# and is *not* clamped: **χ ≤ 0 → pure hadronic, χ ≥ 1 → pure quark, 0 < χ < 1 → mixed.**
# The table builder locates the two χ crossings first and then solves the expensive
# mixed system only between them.
#
# ---
# **Layout**
# - **Part I** — imports, and every knob in one cell.
# - **Part II** — pure phases, then the mixed tables, then TOV.
# - **Part III** — plots, all defined here in the notebook.

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

# Import the repository clone in preference to anything pip has left behind in
# site-packages. The notebook runs from `notebooks/`, so the repository root is
# not on sys.path by default and an older installed copy of `eos` wins the
# import — one that is missing whatever subpackages have been added since it
# was installed, which is how `eos.mixed` goes absent.
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "pyproject.toml").is_file() and (p / "eos").is_dir()), None)
if ROOT is not None:
    sys.path.insert(0, str(ROOT))
    # Drop anything already imported from the wrong copy, so re-running this
    # cell fixes the kernel instead of needing a restart.
    for _m in [m for m in sys.modules if m == "eos" or m.startswith("eos.")]:
        del sys.modules[_m]

try:
    import eos.mixed
except ImportError:      # not inside a clone (Colab, say) — fetch the package
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps",
                           "--quiet",
                           "git+https://github.com/guerrinimirco/eos.git"])
    import eos.mixed

import numpy as np
import matplotlib.pyplot as plt

from eos.dd2 import Parametrization, SpeciesFlags, hadronic_row
from eos.dd2.solver import sweep_beta_eq_octet
from eos.vmit.parameters import get_vmit_default, get_vmit_custom
from eos.vmit.eos import solve_vmit_beta_eq
from eos.mixed import (
    beta_eq_neutrinoless, beta_eq_neutrino_trapped, fixed_YC, fixed_YC_YS,
    MixedTableSpec, build_mixed_table, build_mixed_eos_table,
    mass_radius_mixed, save_table, load_table, export_csv,
    locate_window, sweep_mixed, make_charge_spec, composition_row,
    seed_across_eta,
)

OUT = Path("eos_tables_DD2vMIT")
OUT.mkdir(exist_ok=True)
print("eos imported from:", Path(eos.__file__).parent)


# %% [markdown]
# ## I.2 Knobs
#
# Everything tunable lives in this cell. Edit, then run the rest.

# %%
# ---- hadronic parametrization -------------------------------------------
# from_dd2_defaults()   nucleonic DD2
# from_dd2y_defaults()  DD2Y, with the hyperon couplings (required for hyperons)
# from_nmp(NMP)         pin the nuclear-matter parameters yourself
PAR = Parametrization.from_dd2y_defaults()

# ---- which degrees of freedom exist --------------------------------------
# Every species is an explicit flag; nothing is switched on implicitly, and a
# flag that is not wired raises rather than being quietly ignored.
FLAGS = SpeciesFlags(
    hyperons=False,      # Lambda, Sigma, Xi          (needs from_dd2y_defaults)
    deltas=False,        # Delta quartet
    muons=False,         # electrons are always present; muons optional
    phi_field=True,      # hidden-strange vector, required with hyperons
    photons=True,        # matters only at T > 0
)

# ---- quark parametrization ------------------------------------------------
VMIT = get_vmit_default()
# The hyperon couplings live in DD2Y and nowhere else — DD2 has no entry for
# Lambda — so `hyperons=True` on `from_dd2_defaults()` fails deep in the
# coupling lookup with a bare KeyError. Catch the mismatch here instead.
if FLAGS.hyperons and not PAR.hyperon_coupling_map:
    raise ValueError("FLAGS.hyperons=True needs "
                     "Parametrization.from_dd2y_defaults(); the DD2 defaults "
                     "carry no hyperon couplings")
          # B^1/4 = 180 MeV, a = 0.2 fm^2, m_s = 150 MeV

# ---- STRANGE / RESONANT HADRONIC MATTER -----------------------------------
# To switch hyperons and Delta isobars on, replace the three settings above
# with:
#
#     PAR   = Parametrization.from_dd2y_defaults()
#     FLAGS = SpeciesFlags(hyperons=True, deltas=True, muons=False,
#                          phi_field=True, photons=True)
#     VMIT  = get_vmit_custom(B4=150.0)
#
# and expect to have to tune B4. Hyperons and Deltas both soften the hadronic
# equation of state, which moves the transition and can remove it entirely:
#
#   B4 = 180  the hadronic phase never becomes unfavourable — no transition;
#   B4 = 160  chi rises to about 0.3 and falls back to zero without ever
#             reaching 1, so the transition starts but never completes. The
#             window locator reports no window, which is the correct answer;
#   B4 = 150  a complete transition at eta <= 0.3 (2.3-3.9 n_sat at eta = 0),
#             but at higher eta the window moves below saturation density,
#             where uniform matter is not the ground state anyway.
#
# In other words, whether a hybrid star exists at all is a physics question
# about (B4, a, the hyperon couplings), not something the solver can decide.
# The nucleonic defaults above complete at every eta and are a good place to
# start before opening the strange sector.

# ---- equilibrium mode -----------------------------------------------------
# 'beta_eq_neutrinoless'      independent variables (nB, T)
# 'beta_eq_neutrino_trapped'  independent variables (nB, Y_L, T)
# 'fixed_YC'                  independent variables (nB, Y_C, T)
# 'fixed_YC_YS'               independent variables (nB, Y_C, Y_S, T)
#
# Y_C is the NON-leptonic charge fraction (hadrons and quarks only); Y_S counts
# +1 per s-quark. The fixed-fraction modes sweep their fraction as an extra
# table axis, taken from the list below.
MODE = "beta_eq_neutrinoless"
FIXED = {}                      # scalar values for fractions not swept as axes
Y_C_LIST = [0.05, 0.1, 0.3]     # the Y_C axis, for the fixed-Y_C modes

# ---- grids ----------------------------------------------------------------
N_SAT = PAR.n_sat                                     # fm^-3
NB = np.linspace(0.1 * N_SAT, 12.0 * N_SAT, 300)      # baryon density [fm^-3]
T_LIST = [0.0, 0.1, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
ETA_LIST = [0.0, 0.1, 0.3, 0.6, 1.0]

# ---- what to run ----------------------------------------------------------
TOV_ETAS = [0.0, 0.1, 0.3, 0.6, 1.0]      # TOV is beta-equilibrium, T = 0 only

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

    The string label columns (`phase`) are dropped: everything below plots or
    slices numerically, and carrying them would only force a dtype check at
    every use.
    """
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]
            if not isinstance(rows[0][k], str)}


def _quark_row(q):
    """One pure-quark point, keyed like `hadronic_row` / `composition_row`.
    chi = 1: no hadronic matter left."""
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
        # `composition_row` keys a mixed one, so the wings and the window
        # concatenate in Part III without renaming anything. It also sums Y_C
        # and Y_S over every active baryon, which matters the moment hyperons
        # are switched on.
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
# The stitched core equation of state — pure hadronic below the onset, mixed through
# the window, pure quark above the offset — integrated to a mass-radius sequence.
# Beta equilibrium at T = 0, which is the cold neutron-star condition.
#
# A Maxwell (η=1) table carries a constant-pressure plateau, and `eos.tov` detects it
# and applies the tidal correction across the density discontinuity by itself.

# %%
tov = {}
for eta in TOV_ETAS:
    t0 = time.time()
    core = build_mixed_eos_table(PAR, FLAGS, NB, eta, beta_eq_neutrinoless(),
                                 vmit_params=VMIT, T=0.0)
    res = mass_radius_mixed(PAR, FLAGS, NB, eta, beta_eq_neutrinoless(),
                            vmit_params=VMIT, T=0.0, table=core, n_ec=120)
    tov[eta] = res
    trans = (f"onset {core.n_onset:.3f} offset {core.n_offset:.3f} fm^-3"
             if core.has_transition else "no transition")
    print(f"eta={eta}: {trans} | M_max={res['M_max']:.3f} Msun "
          f"R(M_max)={res['R_Mmax']:.2f} km R(1.4)={res['R_1p4']:.2f} km "
          f"| {time.time()-t0:.1f} s", flush=True)

# %% [markdown]
# # Part III — plots
#
# All plotting is defined here. The mixed tables are re-read from disk, so Part II.2 need
# not be repeated; the pure wings come from Part II.1, which is the cheap cell.
#
# Everything below plots the **complete equation of state** — pure hadronic, mixed, pure
# quark — not the mixed window alone. `full_eos` does the joining.
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
# One panel each, at a chosen temperature, with one colour per η — the whole equation of
# state, hadronic wing through mixed window through quark wing. The χ panel says which
# segment you are looking at: flat at 0 is hadronic, rising is mixed, flat at 1 is quark.
# The two pure branches are drawn dotted in grey underneath, continued past the
# transition where they are metastable, so the gain from the transition is visible.
#

# %%
T_SHOW = 0.0

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for eta in sorted(loaded):
    d = full_eos(eta, T_SHOW)
    if not d:
        continue
    c = COLOR_OF[eta]
    axes[0].plot(d["n_B"], d["P"], "-", lw=2, color=c, label=rf"$\eta={eta}$")
    axes[1].plot(d["n_B"], d["S_per_B"], "-", lw=2, color=c, label=rf"$\eta={eta}$")
    axes[2].plot(d["n_B"], d["chi"], "-", lw=2, color=c, label=rf"$\eta={eta}$")

for pure, lab in ((pure_hadronic.get(T_SHOW), "pure hadronic"),
                  (pure_quark.get(T_SHOW), "pure quark")):
    if pure is None or not pure["n_B"].size:
        continue
    axes[0].plot(pure["n_B"], pure["P"], ":", lw=1.4, color="0.45", label=lab)
    axes[1].plot(pure["n_B"], pure["S_per_B"], ":", lw=1.4, color="0.45")

axes[0].set_ylabel(r"$P$  [MeV fm$^{-3}$]")
axes[1].set_ylabel(r"$S = s/n_B$  [$k_B$ / baryon]")
axes[2].set_ylabel(r"quark volume fraction  $\chi$")
axes[2].set_ylim(-0.02, 1.02)
axes[2].axhline(0.0, color="0.7", lw=0.8)
axes[2].axhline(1.0, color="0.7", lw=0.8)
for ax in axes:
    ax.set_xlabel(r"$n_B$  [fm$^{-3}$]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle(rf"DD2+vMIT, {MODE}, $T = {T_SHOW:g}$ MeV", y=1.02)
fig.tight_layout()
plt.show()


# %% [markdown]
# Reading the middle panel: at T = 0 the entropy is identically zero, so run the cell
# again with `T_SHOW` set to one of the finite temperatures in `T_LIST` to see it.

# %%
T_SHOW_HOT = 20.0

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for eta in sorted(loaded):
    d = full_eos(eta, T_SHOW_HOT)
    if not d:
        continue
    c = COLOR_OF[eta]
    axes[0].plot(d["n_B"], d["P"], "-", lw=2, color=c, label=rf"$\eta={eta}$")
    axes[1].plot(d["n_B"], d["S_per_B"], "-", lw=2, color=c, label=rf"$\eta={eta}$")
    axes[2].plot(d["n_B"], d["chi"], "-", lw=2, color=c, label=rf"$\eta={eta}$")
axes[0].set_ylabel(r"$P$  [MeV fm$^{-3}$]")
axes[1].set_ylabel(r"$S = s/n_B$  [$k_B$ / baryon]")
axes[2].set_ylabel(r"quark volume fraction  $\chi$")
axes[2].set_ylim(-0.02, 1.02)
for ax in axes:
    ax.set_xlabel(r"$n_B$  [fm$^{-3}$]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle(rf"DD2+vMIT, {MODE}, $T = {T_SHOW_HOT:g}$ MeV", y=1.02)
fig.tight_layout()
plt.show()


# %% [markdown]
# ## III.2 Phase boundaries in the (n_B, T) plane
#
# For each η, the density at which the quark phase appears (χ = 0, onset) and the one
# at which the hadronic phase disappears (χ = 1, offset), as functions of temperature.
# The shaded band between them is the mixed phase. A wide band is Gibbs-like; the band
# narrows towards η = 1, where it becomes the Maxwell density jump.

# %%
fig, ax = plt.subplots(figsize=(7.5, 5.5))
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
    ax.plot(onset[good], Ts[good], "-o", ms=4, lw=2, color=c,
            label=rf"$\eta={eta}$")
    ax.plot(offset[good], Ts[good], "--s", ms=4, lw=2, color=c)
    ax.fill_betweenx(Ts[good], onset[good], offset[good], color=c, alpha=0.13)

ax.set_xlabel(r"$n_B$  [fm$^{-3}$]")
ax.set_ylabel(r"$T$  [MeV]")
ax.set_title("Mixed-phase boundaries\n"
             "solid: onset ($\\chi=0$)   dashed: offset ($\\chi=1$)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## III.3 Mass-radius and tidal deformability
#
# Cold, beta-equilibrium stars built on the stitched core equation of state.

# %%
from eos.tov.solver import find_mmax_precise

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for eta, res in tov.items():
    r = res["results"]
    idx, _, M_max = find_mmax_precise(r)
    M, R, Lam = r[:idx + 1, 4], r[:idx + 1, 3], r[:idx + 1, 6]
    c = COLOR_OF.get(eta, "k")
    axes[0].plot(R, M, "-", lw=2, color=c,
                 label=rf"$\eta={eta}$  ($M_{{\max}}={M_max:.2f}\,M_\odot$)")
    axes[0].plot(res["R_Mmax"], M_max, "o", ms=6, color=c)
    axes[1].semilogy(M, Lam, "-", lw=2, color=c, label=rf"$\eta={eta}$")

axes[0].set_xlabel(r"$R$  [km]"); axes[0].set_ylabel(r"$M$  [$M_\odot$]")
axes[0].set_title("Mass-radius")
axes[1].set_xlabel(r"$M$  [$M_\odot$]"); axes[1].set_ylabel(r"$\Lambda$")
axes[1].set_title("Tidal deformability")
for ax in axes:
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## III.4 Composition through the transition
#
# Volume-weighted fractions Y_i = w n_i / n_B, with w = 1−χ for hadrons and w = χ for
# quarks, so the curves sum consistently across the mixed phase. Hadrons solid, quarks
# dashed. Shown across the whole density range: hadrons start at their pure values, hand
# over through the window, and are gone above the offset, where the quarks reach their
# pure ones. The shading marks the mixed window.
#

# %%
ETA_COMP = 0.0
T_COMP = 0.0
Y_FLOOR = 1e-3

d = full_eos(ETA_COMP, T_COMP)
quarks = {"u", "d", "s"}
species = sorted(k[2:] for k in d
                 if k.startswith("Y_") and k not in ("Y_C", "Y_S"))

fig, ax = plt.subplots(figsize=(8, 5.5))
for sp in species:
    y = d[f"Y_{sp}"]
    if not np.isfinite(y).any() or np.nanmax(y) < Y_FLOOR:
        continue
    ax.plot(d["n_B"], y, "--" if sp in quarks else "-", lw=1.8, label=sp)

mixed = (d["chi"] > 0.0) & (d["chi"] < 1.0)
if mixed.any():
    ax.axvspan(d["n_B"][mixed].min(), d["n_B"][mixed].max(),
               color="0.85", zorder=0, label="mixed")
ax.set_yscale("log")
ax.set_ylim(Y_FLOOR, 2.0)
ax.set_xlabel(r"$n_B$  [fm$^{-3}$]")
ax.set_ylabel(r"$Y_i = n_i / n_B$  (volume-weighted)")
ax.set_title(rf"Composition — $\eta={ETA_COMP}$, $T={T_COMP:g}$ MeV")
ax.grid(alpha=0.3, which="both")
ax.legend(ncol=3, fontsize=8)
fig.tight_layout()
plt.show()


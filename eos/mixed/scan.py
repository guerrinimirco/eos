"""
mixed/scan.py
=============
Where in parameter space does a DD2 + vMIT hybrid equation of state exist?

*Public API* (re-exported from `eos.mixed`): `scan_parameters`, `scan_point`.

A hybrid star needs two things that are not guaranteed for an arbitrary set of
parameters, and this module answers both, in the order that fails cheapest:

1. **Is the hadronic model representable at all?** The nuclear-matter
   parameters {n_sat, E_sat, m*/m, K_sat, Q_sat, E_sym, L_sym} are inverted to
   DD2 couplings by `eos.dd2.invert_nmp`. Not every combination has a solution
   — Q_sat in particular can be inconsistent with the cross-constraint that
   closes the isoscalar system — and the inversion reports that rather than
   returning a silently wrong parametrization.

2. **Does a mixed phase exist?** The quark volume fraction chi comes out of the
   mixed solve and is not clamped, so the question is whether it crosses both 0
   and 1 on the density grid. `locate_window` finds those two crossings; when
   it reports no window the honest answer is that these parameters have no
   complete transition, not that the solver failed.

Whether a hybrid star exists is a physics question about (the NMPs, B^1/4, the
vector coupling a, m_s), not something a solver can decide. Both hyperons and
Delta isobars soften the hadronic branch, which moves the transition and can
remove it entirely, so a scan over the hadronic sector alone will find regions
with no window and that is the correct result.

Which nuclear-matter parameters can actually be varied
------------------------------------------------------
Read this before designing a grid, because the answer is not "all seven":

* **L_sym is free.** The isovector inversion is near-analytic and converges
  across at least 30-100 MeV with residuals ~1e-9. Sweep it freely.
* **K_sat is effectively locked to its DD2 value (~243 MeV) at fixed Q_sat.**
  The isoscalar system is closed by a cross-constraint that ties K_sat and
  Q_sat together, so moving K_sat alone leaves a residual that grows smoothly
  away from the DD2 point — about 3e-2 at K_sat = 240 against a 2e-2 gate, and
  0.2 by K_sat = 220. `invert_nmp` reports this rather than returning a
  silently wrong parametrization, and a scan over K_sat at fixed Q_sat will
  therefore come back almost entirely 'inversion_failed'. That is a true
  statement about the DD2 functional form, not a solver defect.
* **(K_sat, Q_sat) jointly** has a narrow feasible ridge — (230, 100) inverts,
  as does (243, 169) — but it is sparse, so a rectangular grid over the pair
  mostly misses it. Finding that ridge properly wants a root solve along it,
  not a grid.

The practical consequence: a two-dimensional map is best drawn over L_sym and
a quark parameter. Include K_sat if you want to *see* the constraint — the
scan reports it honestly — but do not expect it to move.

The hadronic *sector* knobs, by contrast, are all free: the hyperon potentials
U_Lambda / U_Sigma / U_Xi and the Delta potential U_Delta with its vector
ratios x_wD / x_rD (`SECTOR_KEYS`) each invert independently on top of the
nucleon base, and they move the transition far more than the NMPs do because
they set how soft the hadronic branch gets. Put them in the sample dicts
alongside the NMPs and `grid_samples` will cross them like any other axis.
`from_delta_potential` restricts U_Delta to the literature range [-100, -50]
MeV and reports anything outside as 'sectors_failed'.

Failures are recorded as rows, never raised: a scan that aborts on the first
pathological sample is useless for mapping a boundary, and the pathological
samples are exactly the boundary one is trying to map.

Units are fm-based on every boundary, as everywhere else in `eos`: densities
fm^-3, B^1/4 and m_s in MeV, the vector coupling a in fm^2.
"""
import time

import numpy as np

from eos.dd2.parametrization import Parametrization
from eos.dd2.solver import sweep_beta_eq_octet
from eos.vmit.parameters import get_vmit_custom
from eos.mixed.equilibrium.charges import beta_eq_neutrinoless
from eos.mixed.solvers.sweep import locate_window

#: The nuclear-matter parameters `eos.dd2.invert_nmp` consumes, in the order a
#: report reads best. Every key is required — the inversion has no defaults.
NMP_KEYS = ("n_sat", "E_sat", "m_eff_ratio", "K_sat", "Q_sat",
            "E_sym", "L_sym")

#: The vMIT parameters worth varying. The quark masses m_u and m_d are held at
#: their defaults: at these densities the equation of state is insensitive to
#: them, and varying them buys nothing but scan dimensions.
VMIT_KEYS = ("B4", "a", "m_s")

#: Hyperon single-particle potentials in symmetric matter at saturation [MeV].
#: These are the DD2Y values, and they are what `from_hyperon_potentials`
#: inverts the scalar couplings from.
DEFAULT_HYPERON_POTENTIALS = {"U_Lambda": -30.0, "U_Sigma": 30.0,
                              "U_Xi": -18.0}

#: Delta-isobar potential [MeV]. `from_delta_potential` rejects anything
#: outside the literature range [-100, -50].
DEFAULT_U_DELTA = -50.0

#: Hadronic-sector coupling knobs that may be carried *inside* an NMP sample
#: dict, alongside the `NMP_KEYS`, so that `grid_samples` can put them on a scan
#: axis like any other parameter. `x_wD`/`x_rD` are the Delta vector coupling
#: ratios x_omegaDelta / x_rhoDelta; x_sigmaDelta is not free, being fixed by
#: inverting U_Delta.
SECTOR_KEYS = ("U_Lambda", "U_Sigma", "U_Xi", "U_Delta", "x_wD", "x_rD")


def _split_sample(sample, hyperon_potentials=None, U_Delta=DEFAULT_U_DELTA):
    """Separate an NMP sample into (nmp, sector kwargs).

    A sample dict may carry any of the `SECTOR_KEYS` next to the `NMP_KEYS`;
    those override the corresponding keyword defaults. This is what lets one
    `grid_samples(...)` call put L_sym, U_Xi and U_Delta on axes together —
    they are all "hadronic parameters" to the caller even though the inversion
    treats them in separate stages.
    """
    nmp = {k: v for k, v in sample.items() if k not in SECTOR_KEYS}
    pots = dict(DEFAULT_HYPERON_POTENTIALS)
    pots.update(hyperon_potentials or {})
    pots.update({k: float(sample[k]) for k in ("U_Lambda", "U_Sigma", "U_Xi")
                 if k in sample})
    sector = {"hyperon_potentials": pots,
              "U_Delta": float(sample.get("U_Delta", U_Delta)),
              "x_wD": float(sample.get("x_wD", 1.0)),
              "x_rD": float(sample.get("x_rD", 1.0))}
    return nmp, sector


def _sector_columns(sector, flags):
    """The sector knobs as flat row columns, nan where the flag is off so a
    column never claims a value the model did not use."""
    had = sector["hyperon_potentials"]
    cols = {k: (had[k] if flags.hyperons else np.nan)
            for k in ("U_Lambda", "U_Sigma", "U_Xi")}
    cols.update({k: (sector[k] if flags.deltas else np.nan)
                 for k in ("U_Delta", "x_wD", "x_rD")})
    return cols


def build_parametrization(nmp, flags, hyperon_potentials=None,
                          U_Delta=DEFAULT_U_DELTA, x_wD=1.0, x_rD=1.0):
    """Nuclear-matter parameters to a `Parametrization` with the strange and
    resonant sectors attached, as `flags` requires.

    `from_nmp` inverts the NUCLEON sector only — it carries no hyperon
    couplings, so `SpeciesFlags(hyperons=True)` on its output would fail deep in
    a coupling lookup. The hyperon and Delta sectors are attached on top here,
    each by inverting its single-particle potential in symmetric matter at
    saturation *on the inverted base*, so they adapt to that base's nucleon
    couplings rather than assuming DD2's.

    `nmp` may also carry any of the `SECTOR_KEYS` (U_Lambda, U_Sigma, U_Xi,
    U_Delta, x_wD, x_rD); those take precedence over the keyword arguments, so
    a single `grid_samples` call can put nuclear-matter parameters and sector
    potentials on axes together.

    Returns `(par, stage, message)`. `stage` is 'ok', 'inversion_failed' when
    the NMPs have no DD-RMF realisation at all, or 'sectors_failed' when they
    do but the hyperon/Delta scalar inversion does not converge on them — the
    second can happen even when the first succeeded, which is why they are
    reported separately. `par` is None unless `stage` is 'ok'.
    """
    nmp, sector = _split_sample(dict(nmp), hyperon_potentials, U_Delta)
    par, status = Parametrization.from_nmp(nmp, return_status=True)
    if not status.ok:
        return None, "inversion_failed", status.message
    try:
        if flags.hyperons:
            par = Parametrization.from_hyperon_potentials(
                base=par, **sector["hyperon_potentials"])
        if flags.deltas:
            par = Parametrization.from_delta_potential(
                U_Delta=sector["U_Delta"], x_wD=sector["x_wD"],
                x_rD=sector["x_rD"], base=par)
    except Exception as exc:
        return None, "sectors_failed", f"{type(exc).__name__}: {exc}"
    return par, "ok", ""


def _hadronic_tov(par, flags, n_ec=120, tov_parallel=True):
    """(M_max, R_Mmax, R_1p4) of the PURE hadronic branch, no quarks.

    Reuses `eos.dd2.verify.tov.build_core_table` for the cold beta-equilibrium
    core and runs the Numba TOV backend over it. nan on failure — a branch that
    will not integrate is a finding about the parameters.
    """
    from eos.dd2.verify.tov import build_core_table, N_TRANSITION
    from eos.tov.solver import (
        compute_tov_sequence, find_mmax_precise, generate_ec_logspace,
        CRUST_PATHS,
    )
    import os

    nan3 = (np.nan, np.nan, np.nan)
    try:
        core = build_core_table(par, flags)
        if core.P.size < 10:
            return nan3
        crust = ("BPS" if os.path.isfile(CRUST_PATHS.get("BPS", "")) else "No")
        res = compute_tov_sequence(
            core, generate_ec_logspace(150.0, 3000.0, n_ec),
            add_crust_table=crust, add_crust_mode="attach",
            n_transition=(N_TRANSITION if crust != "No" else None),
            compute_baryonic_mass=False, compute_tidal=False, verbose=False,
            backend="fast", tov_parallel=tov_parallel)
        idx, _, M_max = find_mmax_precise(res)
        M, R = res[:idx + 1, 4], res[:idx + 1, 3]
        R_14 = float(np.interp(1.4, M, R)) if M[-1] >= 1.4 > M[0] else np.nan
        return float(M_max), float(res[idx, 3]), R_14
    except Exception:
        return nan3


def scan_hadronic_point(nmp, flags, n_B_grid, hyperon_potentials=None,
                        U_Delta=DEFAULT_U_DELTA, tov=True, tov_parallel=True,
                        M_max_target=2.0, n_sweep_min=1.0):
    """Stage one: does this NMP give a working DD2 with the requested species?

    Four checks, in the order that fails cheapest, each recorded separately so
    a failure says *which* stage broke rather than only that something did:

    1. `inversion_ok`  — the NMPs have a DD-RMF realisation at all. This is the
       check that fails most often, and not for numerical reasons: K_sat and
       Q_sat are tied together by the cross-constraint that closes the
       isoscalar system, so K_sat cannot be moved on its own. See the note in
       the module docstring;
    2. `sectors_ok`    — the hyperon and Delta couplings invert on that base;
    3. `sweep_ok`      — the cold beta-equilibrium sweep reaches at least
       `n_sweep_min` [fm^-3]. It is NOT required to cover the whole grid: a
       Delta model drives the effective mass to zero (scalar collapse) at high
       density, so a truncated sweep is expected physics rather than a failure.
       What matters is whether it got far enough to build a star, and the
       density it actually reached is recorded as `n_sweep_max`;
    4. `M_max_had`     — the hadronic branch alone supports `M_max_target`.
       Hyperons and Deltas both soften it considerably, so failing this is
       common and does NOT rule the sample out as a hybrid star: quark matter
       can stiffen the star back above two solar masses.

    `status` is 'ok' only when every requested check passes, otherwise the name
    of the first that failed. Nothing raises.
    """
    missing = [k for k in NMP_KEYS if k not in nmp]
    if missing:
        raise ValueError(f"nmp is missing {missing}; needs all of {NMP_KEYS}")

    grid = np.asarray(n_B_grid, float)
    _, sector = _split_sample(nmp, hyperon_potentials, U_Delta)
    row = {k: float(nmp[k]) for k in NMP_KEYS}
    row.update(_sector_columns(sector, flags))
    row.update(inversion_ok=0.0, sectors_ok=0.0, sweep_ok=0.0,
               n_sweep_max=np.nan, M_max_had=np.nan, R_Mmax_had=np.nan,
               R_1p4_had=np.nan, seconds=0.0, status="", message="")

    t0 = time.time()
    par, stage, message = build_parametrization(
        nmp, flags, hyperon_potentials=hyperon_potentials, U_Delta=U_Delta)
    row["message"] = message[:200]
    if stage != "ok":
        row["inversion_ok"] = float(stage != "inversion_failed")
        row["status"] = stage
        return _timed(row, t0)
    row.update(inversion_ok=1.0, sectors_ok=1.0)

    try:
        pts = sweep_beta_eq_octet(par, grid, flags, T=0.0,
                                  include_photons=False,
                                  stop_at_boundary=True)
    except Exception as exc:
        row["status"] = "sweep_failed"
        row["message"] = f"{type(exc).__name__}: {exc}"[:200]
        return _timed(row, t0)

    # A truncated sweep is the scalar-collapse boundary, not an error; what
    # matters is whether it got high enough in density to build a star, so the
    # density it stopped at is the number recorded and tested.
    row["n_sweep_max"] = float(pts[-1].n_B) if pts else np.nan
    row["sweep_ok"] = float(bool(pts) and pts[-1].n_B >= n_sweep_min)
    if not row["sweep_ok"]:
        row["status"] = "sweep_truncated"

    if tov:
        M_max, R_Mmax, R_14 = _hadronic_tov(par, flags,
                                            tov_parallel=tov_parallel)
        row.update(M_max_had=M_max, R_Mmax_had=R_Mmax, R_1p4_had=R_14)
        if row["status"] == "" and not (M_max >= M_max_target):
            row["status"] = ("hadronic_M_max_low" if np.isfinite(M_max)
                             else "tov_failed")
    if row["status"] == "":
        row["status"] = "ok"
    return _timed(row, t0)


def scan_hadronic(nmp_samples, flags, n_B_grid, hyperon_potentials=None,
                  U_Delta=DEFAULT_U_DELTA, tov=True, n_jobs=1, progress=None,
                  M_max_target=2.0, n_sweep_min=1.0):
    """`scan_hadronic_point` over a list of NMP dicts. Returns long-format rows.

    This is the hadronic half of choosing parameters: it says which nuclear
    matter parameters give a DD2 that works at all with the species requested,
    before any quark parameter is chosen. Feed the rows whose `status` is 'ok'
    into `scan_parameters` for the hybrid half.
    """
    def one(nmp):
        return scan_hadronic_point(nmp, flags, n_B_grid,
                                   hyperon_potentials=hyperon_potentials,
                                   U_Delta=U_Delta, tov=tov,
                                   tov_parallel=(n_jobs == 1),
                                   M_max_target=M_max_target,
                                   n_sweep_min=n_sweep_min)

    if n_jobs == 1:
        rows = []
        for nmp in nmp_samples:
            row = one(nmp)
            rows.append(row)
            if progress is not None:
                progress(row)
        return rows

    from joblib import Parallel, delayed
    rows = list(Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in nmp_samples))
    if progress is not None:
        for row in rows:
            progress(row)
    return rows


def scan_point(nmp, vmit, flags, n_B_grid, eta=0.0, T=0.0, spec=None,
               analytic_jac=False, tov=False, tov_parallel=True,
               hyperon_potentials=None, U_Delta=DEFAULT_U_DELTA):
    """One (nmp, vmit) sample: invert, then look for a transition window.

    nmp   : dict with the `NMP_KEYS`, optionally carrying any of the
            `SECTOR_KEYS` as well, which is how hyperon and Delta potentials go
            on a scan axis
    vmit  : dict with any of the `VMIT_KEYS`; the rest take vMIT defaults
    flags : `SpeciesFlags`. Hyperons and Deltas are supported: their sectors
            are attached on top of the NMP inversion by `build_parametrization`,
            since `from_nmp` inverts the nucleon sector only
    hyperon_potentials, U_Delta : the potentials those sectors are inverted
            from when the sample does not name them; default to the DD2Y values
            and -50 MeV
    spec  : `ChargeSpec`; defaults to neutrino-transparent beta equilibrium,
            the cold-star condition

    tov   : also build the core equation of state and run the mass-radius
            sequence, adding M_max, R_Mmax, R_1p4, cs2_max, the central baryon
            density n_c_max and the `core_phase` it falls in to the row. Only
            attempted when a window exists. Costs ~1-2 s per sample on top of
            the ~0.5 s the window search takes, which is affordable only
            because the TOV integration defaults to the Numba backend.
    tov_parallel : passed through to the TOV sequence. Set False when the scan
            itself is running under `n_jobs > 1`, so the process pool and the
            integrator's own threads do not oversubscribe the cores —
            `scan_parameters` does this for you.

    Returns one flat dict row. `status` is 'ok' when every requested check
    passes, 'inversion_failed' when the NMPs are not representable, 'no_window'
    when they are but chi never completes its crossing, 'tov_failed' when the
    star sequence did not integrate, and 'error: ...' when the solve raised —
    which is itself a finding, so it is recorded, not raised.
    """
    missing = [k for k in NMP_KEYS if k not in nmp]
    if missing:
        raise ValueError(f"nmp is missing {missing}; needs all of {NMP_KEYS}")

    _, sector = _split_sample(nmp, hyperon_potentials, U_Delta)
    row = {k: float(nmp[k]) for k in NMP_KEYS}
    row.update(_sector_columns(sector, flags))
    vmit_full = {k: float(vmit[k]) for k in VMIT_KEYS if k in vmit}
    params = get_vmit_custom(**vmit_full)
    row.update(B4=params.B4, a=params.a, m_s=params.m_s, eta=float(eta),
               T=float(T))
    row.update(inversion_ok=0.0, window_exists=0.0,
               n_onset=np.nan, n_offset=np.nan, seconds=0.0, status="")
    if tov:
        row.update(M_max=np.nan, R_Mmax=np.nan, R_1p4=np.nan, cs2_max=np.nan,
                   n_c_max=np.nan, core_phase="", eos_ok=0.0, eos_reason="")

    t0 = time.time()
    try:
        # One cascade for both stages: NMP -> nucleon couplings, then the
        # hyperon and Delta sectors on top if the flags ask for them.
        par, stage, message = build_parametrization(
            nmp, flags, hyperon_potentials=hyperon_potentials,
            U_Delta=U_Delta)
        row["inversion_ok"] = float(stage != "inversion_failed")
        if stage != "ok":
            row["status"] = stage
            return _timed(row, t0)

        grid = np.asarray(n_B_grid, float)
        cs = spec if spec is not None else beta_eq_neutrinoless()
        window = locate_window(par, flags, grid, float(eta), cs,
                               vmit_params=params, T=float(T),
                               analytic_jac=analytic_jac)
        row["window_exists"] = float(bool(window.exists))
        row["n_onset"] = float(window.n_onset)
        row["n_offset"] = float(window.n_offset)
        row["status"] = "ok" if window.exists else "no_window"

        if tov and window.exists:
            row.update(_tov_columns(par, flags, grid, eta, cs, params, T,
                                    window, tov_parallel))
            if not row["eos_ok"]:
                # The stitch is not an equation of state; say so rather than
                # reporting the maximum mass it would have integrated to.
                row["status"] = "eos_unphysical"
            elif not np.isfinite(row["M_max"]):
                row["status"] = "tov_failed"
    except Exception as exc:                    # a finding, not a crash
        row["status"] = f"error: {type(exc).__name__}: {exc}"[:200]
    return _timed(row, t0)


def eos_is_physical(table, tol=1e-9):
    """Is this stitched core equation of state fit to integrate?

    Two conditions, and a table failing either is not an equation of state at
    all, so feeding it to TOV would produce a number with no meaning rather
    than an error:

    * **P non-decreasing in n_B.** A Maxwell plateau is flat, so equality is
      allowed, but P must never fall.
    * **0 <= c_s^2 <= 1.** A descending P gives a negative c_s^2 and a
      superluminal one is unphysical from the other side.

    This matters in practice: the chi = 0 crossing that `locate_window` finds
    can be spurious when hyperons and Deltas are active at a low bag constant.
    The mixed branch it locks onto there sits at a LOWER pressure than the
    hadronic branch at the same density — so it is not the favoured state — and
    the stitch steps down by tens of MeV/fm^3 at the onset. The resulting table
    integrates to nonsense (maximum masses in the hundreds of solar masses),
    which is why the scan checks before integrating rather than after.

    Returns (ok, reason).
    """
    from eos.mixed.coefficients import sound_speed_eq

    if table.P.size < 10:
        return False, "too_few_points"
    if np.any(np.diff(table.P) < -tol):
        worst = float(np.min(np.diff(table.P)))
        return False, f"P_not_monotone(min_dP={worst:.2e})"
    cs2 = sound_speed_eq(table.P, table.eps)
    if np.any(cs2 < -1e-6) or np.any(cs2 > 1.0 + 1e-6):
        return False, (f"cs2_out_of_range([{np.nanmin(cs2):.3f},"
                       f"{np.nanmax(cs2):.3f}])")
    return True, ""


def core_phase(n_c, table):
    """Which phase sits at the centre of the maximum-mass star.

    A transition existing in the equation of state and a transition being
    *realised inside a star* are different statements: the window can open at a
    density the heaviest stable star never reaches, in which case the model
    predicts a hybrid branch that nothing in nature occupies. The central
    baryon density decides it:

        'none'  no window at all — the star is pure hadronic
        'H'     n_c < n_onset — the window exists but no star reaches it
        'mix'   n_onset <= n_c < n_offset — a mixed-phase core
        'Q'     n_c >= n_offset — a genuine pure-quark core

    '' when the central density is unknown (the sequence did not integrate).
    """
    if not np.isfinite(n_c):
        return ""
    if not table.has_transition:
        return "none"
    if n_c < table.n_onset:
        return "H"
    return "mix" if n_c < table.n_offset else "Q"


def _tov_columns(par, flags, grid, eta, spec, params, T, window, tov_parallel):
    """M_max, R(M_max), R(1.4), max c_s^2 and the core phase for one sample.

    The window is reused rather than re-located — `scan_point` has already paid
    for it. The stitched equation of state is checked for monotonicity and
    causality BEFORE being integrated; a table that fails gives nan columns and
    an 'eos_unphysical' reason, because integrating it would return a confident
    wrong number instead of an error. A sequence that then fails to integrate
    gives nan columns too, never an exception.

    `n_c_max` is the central *baryon* density at M_max, read off the same table
    the sequence was integrated from, since `mass_radius_mixed` reports the
    central energy density and the transition boundaries are in n_B.
    """
    from eos.mixed.tables.core_eos import build_mixed_eos_table, mass_radius_mixed
    from eos.mixed.coefficients import sound_speed_eq

    table = build_mixed_eos_table(par, flags, grid, float(eta), spec,
                                  vmit_params=params, T=float(T),
                                  window=window)
    cs2 = sound_speed_eq(table.P, table.eps)
    out = {"cs2_max": float(np.nanmax(cs2)) if cs2.size else np.nan,
           "eos_ok": 0.0, "eos_reason": "",
           "M_max": np.nan, "R_Mmax": np.nan, "R_1p4": np.nan,
           "n_c_max": np.nan, "core_phase": ""}

    ok, reason = eos_is_physical(table)
    out["eos_ok"] = float(ok)
    out["eos_reason"] = reason
    if not ok:
        return out

    try:
        res = mass_radius_mixed(par, flags, grid, float(eta), spec,
                                vmit_params=params, T=float(T), table=table,
                                n_ec=120, tov_parallel=tov_parallel)
    except Exception:
        return out
    n_c = float(np.interp(res["e_c_max"], table.eps, table.n_B))
    out.update(M_max=float(res["M_max"]), R_Mmax=float(res["R_Mmax"]),
               R_1p4=float(res["R_1p4"]), n_c_max=n_c,
               core_phase=core_phase(n_c, table))
    return out


def _timed(row, t0):
    row["seconds"] = time.time() - t0
    return row


def scan_parameters(nmp_samples, vmit_samples, flags, n_B_grid, eta=0.0,
                    T=0.0, spec=None, n_jobs=1, progress=None, tov=False,
                    hyperon_potentials=None, U_Delta=DEFAULT_U_DELTA):
    """Scan the product of `nmp_samples` and `vmit_samples`.

    Returns long-format rows — one per (nmp, vmit) pair — ready for
    `eos.general.table_io.save_table` / `export_csv`.

    tov=True adds M_max, R(M_max), R(1.4) and max c_s^2 per sample, so the scan
    answers not just "is there a transition" but "is the resulting star
    astrophysically viable" — which is what choosing parameters actually needs.

    n_jobs > 1 spreads the samples over processes with joblib, which is where
    the parallelism belongs: the samples are independent and each one is a long
    serial solve, so there is nothing to gain from threading inside a solve.
    The TOV integrator's own `prange` is switched off in that case so the two
    layers of parallelism do not oversubscribe the cores. Pass n_jobs=1 (the
    default) to keep it serial and debuggable.

    `progress` is an optional callable invoked with each finished row, so a
    notebook can print without this module importing anything to print with.
    """
    pairs = [(nmp, vmit) for nmp in nmp_samples for vmit in vmit_samples]

    def one(pair):
        return scan_point(pair[0], pair[1], flags, n_B_grid, eta=eta, T=T,
                          spec=spec, tov=tov, tov_parallel=(n_jobs == 1),
                          hyperon_potentials=hyperon_potentials,
                          U_Delta=U_Delta)

    if n_jobs == 1:
        rows = []
        for pair in pairs:
            row = one(pair)
            rows.append(row)
            if progress is not None:
                progress(row)
        return rows

    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=n_jobs)(delayed(one)(p) for p in pairs)
    if progress is not None:
        for row in rows:
            progress(row)
    return list(rows)


def grid_samples(**axes):
    """Cartesian product of named axes as a list of dicts.

    Convenience for building `nmp_samples` / `vmit_samples`:

        grid_samples(B4=[150, 165, 180], a=[0.0, 0.2])

    gives six dicts. A scalar axis is treated as a one-element one, so the
    parameters being held fixed read the same way as the ones being varied.
    """
    from itertools import product
    keys = list(axes)
    grids = [np.atleast_1d(axes[k]) for k in keys]
    return [dict(zip(keys, (float(v) for v in combo)))
            for combo in product(*grids)]


if __name__ == "__main__":
    # Smallest runnable check: a representable sample finds a window, and an
    # unrepresentable one is reported rather than raising.
    from eos.dd2 import SpeciesFlags, compute_nmp, Parametrization as P

    flags = SpeciesFlags(hyperons=False, phi_field=False, muons=False)
    grid = np.linspace(0.05, 1.6, 120)
    good = compute_nmp(P.from_dd2_defaults())
    bad = dict(good, K_sat=1.0e4)               # not representable in DD-RMF

    rows = scan_parameters([good, bad], grid_samples(B4=180.0), flags, grid,
                           tov=True)
    assert len(rows) == 2, rows
    assert rows[0]["status"] == "ok" and rows[0]["window_exists"] == 1.0, rows[0]
    assert rows[1]["status"] != "ok", rows[1]
    assert all(isinstance(r["seconds"], float) for r in rows)
    assert np.isfinite(rows[0]["M_max"]) and rows[0]["M_max"] > 1.5, rows[0]
    assert np.isnan(rows[1]["M_max"]), rows[1]
    # The core phase must be consistent with where the central density landed.
    assert rows[0]["core_phase"] in ("H", "mix", "Q"), rows[0]
    assert np.isfinite(rows[0]["n_c_max"]), rows[0]
    assert (rows[0]["n_c_max"] >= rows[0]["n_onset"]) == (
        rows[0]["core_phase"] != "H"), rows[0]
    print(f"DD2 defaults : window [{rows[0]['n_onset']:.4f}, "
          f"{rows[0]['n_offset']:.4f}] fm^-3, M_max={rows[0]['M_max']:.3f} "
          f"Msun, R(1.4)={rows[0]['R_1p4']:.2f} km, "
          f"max c_s^2={rows[0]['cs2_max']:.3f}, "
          f"n_c={rows[0]['n_c_max']:.3f} fm^-3 ({rows[0]['core_phase']})")
    print(f"K_sat=1e4    : {rows[1]['status']}")
    print("scan self-check OK")

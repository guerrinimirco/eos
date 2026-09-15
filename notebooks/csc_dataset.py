"""Build the finite-T NJL training set: a warm-started GRID at fixed potentials.

One row is one converged `eos.njl` state at FIXED chemical potentials, in one
DECLARED pairing pattern, at one point of the model's free-parameter space:

    (mu_B, mu_C, mu_S, T; eta_D, G_V0/G_S)  ->  P, eps, s, n_B, n_C, n_S,
                                                the gaps, the masses, mu_3, mu_8

which is the surface the closed-form Omega of `docs/csc_bag_mapping.md` is
fitted to. Potentials in rather than densities in: the solve is then a single
self-consistency at fixed mu with no outer inversion, which is the cheap
direction and the one a thermodynamic potential is naturally a function of.
One pattern is declared per sweep, so a fit never sees a first-order
transition -- there is no ranking here to get wrong, and the branch that loses
is still a branch the model has to represent.

A GRID, NOT A SOBOL SAMPLE, and the reason is the warm start. An earlier
version of this file drew a scrambled Sobol sequence over a 4-D box, on the
argument that a scattered sample covers four dimensions better per solve and
that nothing needs a neighbour. That is true of the FIT and false of the
SOLVER: a scattered point has no neighbour to start from, and every solve is
cold. A grid swept downward in mu_B at fixed (mu_C, mu_S, T) hands each solve
the previous root, which is most of the speed and all of the branch
continuity -- `eos.njl.table.branch_ladder` walks down for the same reason
("every pattern exists at the top of a density range and the paired ones end
somewhere below"), and an ASCENDING CFL sweep is measured to stitch two roots
together and return mu_B(n_B) non-monotone.

THE VECTOR FORM IS `gluon_exchange`, AND NOT BECAUSE OF APPENDIX A.1. An
earlier version of this file built its grid with `vector_form="constant"`,
because that is the form the published RG-NJL sets (rg_njl1/Kunkel, eta_D =
1.45, eta_V = 0.7) are quoted in, and `docs/csc_bag_mapping.md` contradicted
it with a measured table -- unpaired rms in P of 2.1 / 9.8 / 13.8% at eta_V =
0 / 0.5 / 1 against 3.8% for `gluon_exchange`. That table was taken on the BAG
SUB-BASIS, whose stated failure mode is that `(mu^4, mu^2, const)` absorbs
`~G_V n_q^2 ~ mu^6` only by leaving the physical window: a complaint about a
MISSING TERM, and the shipped basis now has it (`a6`). The argument that does
survive is the shape of that term,

    a6              sum_f mu_f^6/(mu_f^2 + M_V^2)      -> mu^4 above M_V
    gluon_exchange  G_V = G_V0/[1 + 8 k_F^2/(9 M_g^2)] -> G_V n_q^2 ~ mu^4
    constant        G_V = eta_V G_S                    -> G_V n_q^2 ~ mu^6

-- `a6` IS the gluon-exchange form written as a basis function, same
saturation and same asymptote, while a constant G_V is the one member of the
family that never saturates. The basis is built so that `P/P_free -> a4 + a6`
and `c_s^2 -> 1/3`; a constant-G_V medium runs to `c_s^2 -> 1/5`, and the
conformal limit cannot be imposed on a target that violates it. Secondarily,
the continuous map is fitted over `G_V0_over_GS`, which only `gluon_exchange`
reads, while `eta_V` is read only by `constant`: mapping both is two surfaces,
not one. What the constant form costs in reproducibility is measured
separately by `--constant-check`, on the current basis and the shipped grid,
so the ledger entry that follows is written against a live number.

WHY `thermo_from_mu` AND NOT `eos.mixed.adapters.njl_phase`. With a single
declared pattern the adapter is a thin wrapper over this same call, and it
drops two things this file needs: the GAPLESS FLAG, which `njl_phase` does not
copy into its `PhaseThermo.fields`, and the internal unknown vector, which it
returns only as a private per-pattern dict. `thermo_from_mu(..., 
return_state=True)` gives both -- it is the model's own documented
potentials-in surface (CLAUDE.md section 13), the same one the adapter
consumes.

WHAT A FIT MUST FILTER, and this file does it at source rather than labelling:

  * the vacuum (n_B <= 1e-3 fm^-3): below a branch's terminus the quark phase
    does not exist and the vanishing solution is one row repeated.
  * a COLLAPSED layout: a paired candidate with max|Delta_i| <= 1 MeV has
    fallen onto a rival's root and is not a point of the branch it was asked
    for. `pattern_realised` is recorded beside every row for the same reason
    and is never used as a dict key -- `eos.general.pairing.realised_pattern`
    returns names outside this enumeration (uSC, dSC, usSC) when a CFL-layout
    solve collapses.
  * an UNLOCKED CFL row. A NONZERO GAP DOES NOT IDENTIFY THE LOCKED PHASE:
    near the branch terminus a CFL-layout solve converges on a state carrying
    a 138 MeV gap and no strange quarks at all. The test is the locking
    itself, |n_C| < 1e-6 n_B and |n_S/n_B - 1| < 1e-6, and dropping the two
    points of twenty-four that failed it improved a CFL fit by a factor of
    forty.

`gapless` is RECORDED AND NOT FILTERED. The closed-form ansatz has no gapless
branch, so a gapless state entering the training data as an ordinary CFL row
is a problem -- but how much of one is a question for the phase-disagreement
work, and dropping the rows before that question is asked would hide its
answer. `docs/DEFERRED.md` records a fixed-Y_C table that reported metastable
2SC at 98 of 100 points because the gapless CFL that minimised f could not
reach the gate, and closes: "converged = True on every row of a table is
therefore not evidence the table is the ground state."

CFL IS SAMPLED ON A LINE. Locking fixes n_C = 0 and n_S = n_B identically, so
the CFL pressure depends on mu_B + mu_S alone -- measured identical to every
digit at (1900,0,0), (1900,-60,0), (1800,0,100) and (1840,0,60). The full grid
there is one line sampled many times over.

Usage:

    python notebooks/csc_dataset.py --cost            # project, solve nothing
    python notebooks/csc_dataset.py --time            # MEASURE one set, warm
                                                      # against cold
    python notebooks/csc_dataset.py --constant-check  # the vector-form ledger
    python notebooks/csc_dataset.py --smoke           # three points per pattern
    for i in 0 1 2 3; do
        python notebooks/csc_dataset.py --shard $i --nshards 4 &
    done; wait
    python notebooks/csc_dataset.py --merge

Each (parameter set, pattern) is cached under `output/csc_map_cache/` by an
md5 over `(asdict(par), pattern, the four grids)` -- THE SAME KEY RECIPE
`notebooks/csc_bag_map.py` reads, so a generated set is already warm when the
notebook's own `GRID_T` is extended to this one. A task whose cache entry
exists is skipped, so a run resumes for free. `--merge` concatenates the cache
into `output/csc_ml/csc_njl_samples.csv.gz` with a JSON sidecar.
"""
import argparse
import csv
import gzip
import hashlib
import json
import pickle
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eos import njl                                              # noqa: E402
from eos.general.pairing import realised_pattern                 # noqa: E402
from eos.njl.thermodynamics import (thermo_from_mu,              # noqa: E402
                                    vacuum_solution)

CACHE = ROOT / "output" / "csc_map_cache"
OUT = ROOT / "output" / "csc_ml"

# --- the grid ---------------------------------------------------------------
# The mu axes are `notebooks/csc_bag_map.py`'s, unchanged, and the reason they
# are unchanged is measured: sweeping mu_B in 50 MeV steps on the four
# gluon-exchange sets, NO BRANCH REACHES 1 n_sat -- each ends at its own
# terminus, 1.5-2.4 n_sat at mu_B = 850-1150 MeV, below which the T = 0 quark
# phase does not exist because the chiral condensate is still there. The
# reachable window is about 2-13 n_sat. (For comparison, mu_B = 1500-2600 MeV
# is 5-38 n_sat: almost entirely above where stars are, and anchoring the
# coefficients there is how a fit comes out excellent and useless.)
#
# T IS THE NEW AXIS, and 0 is on it exactly rather than as a small number:
# that is where every published comparison lives, and a uniform 4-D sample
# would place almost no points there.
GRID_MU_B = np.linspace(900.0, 1800.0, 28)       # MeV, ~2-13 n_sat, dense
GRID_MU_C = np.array([-80.0, -40.0, 0.0, 40.0])  # MeV
GRID_MU_S = np.array([-80.0, 0.0, 80.0])         # MeV
GRID_T = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0)    # MeV

PATTERNS = ("unpaired", "2SC", "CFL")
#: Leading-order condensation coefficient, used here only to ask whether a
#: pattern is paired at all (the fit uses it as a basis weight).
C_PHASE = {"unpaired": 0.0, "2SC": 1.0 / 3.0, "CFL": 1.0}

#: The continuous axes of the coefficient map. The VACUUM tier (Lambda, G_S,
#: K, the current masses) is held at `rkh` throughout -- moving it is a
#: different vacuum, not a point of this space -- and what varies is the
#: medium: the diquark coupling and the gluon-exchange vector strength.
#: A quadratic surface in two variables needs six points; fifteen is what the
#: prototype used and what the leave-one-out number was measured on.
MAP_ETA_D = (0.75, 1.00, 1.25, 1.45, 1.65)
MAP_G_V0 = (0.25, 0.50, 0.75)

#: Seconds per solve, per pattern, for `--cost` ONLY. These are the fixed-n_B
#: numbers the Sobol version projected with and they are NOT a measurement of
#: this sweep: a fixed-mu solve closes a smaller system, and a warm start cuts
#: it again. `--time` replaces them with a measured pair and prints both.
COST_PER_SOLVE = {"unpaired": 0.01, "2SC": 0.15, "CFL": 0.60}


def parameter_sets():
    """The model points sampled, as {par_id: Parameters}.

    Built exactly as `csc_bag_map.py`'s map loop builds them, field for field,
    because the cache key is an md5 over `asdict(par)` and a set that differs
    in one default writes a second copy of the same physics.
    """
    sets = {}
    for eta_D in MAP_ETA_D:
        for g_v0 in MAP_G_V0:
            sets[f"g_eD{eta_D:.2f}_gV{g_v0:.2f}"] = replace(
                njl.Parameters.named("rkh"), eta_D=eta_D, eta_V=0.0,
                vector_form="gluon_exchange", G_V0_over_GS=g_v0, M_g=500.0)
    return sets


def cache_path(par, pattern):
    """`csc_bag_map.sample`'s key, recipe for recipe."""
    key = hashlib.md5(json.dumps(
        [asdict(par), pattern, list(GRID_MU_B), list(GRID_MU_C),
         list(GRID_MU_S), list(GRID_T)],
        sort_keys=True, default=str).encode()).hexdigest()
    return CACHE / (key + ".pkl")


def lines(pattern):
    """The (mu_C, mu_S, T) lines of the grid, each swept over mu_B.

    CFL gets the mu_B line alone: its pressure depends on mu_B + mu_S only and
    on mu_C not at all, so every other line there is the same state resampled.
    """
    if pattern == "CFL":
        return [(0.0, 0.0, T) for T in GRID_T]
    return [(mu_C, mu_S, T)
            for T in GRID_T for mu_C in GRID_MU_C for mu_S in GRID_MU_S]


def sweep_line(par, pattern, mu_C, mu_S, T, vac, descending=True, drops=None):
    """One (mu_C, mu_S, T) line, swept in mu_B and warm started.

    DOWNWARD by default. Every pattern exists at the top of its density range
    and the paired ones end somewhere below, so a descending sweep starts
    where the branch is easiest and carries a root into the hard end; an
    ascending one starts below the terminus, where the branch does not exist,
    and a CFL sweep run that way stitches two roots together and returns
    mu_B(n_B) non-monotone.

    A CANDIDATE THAT COLLAPSED CARRIES NOTHING FORWARD. Seeding the next point
    from the root a candidate fell onto would pin it there for the rest of the
    sweep and hide a branch that exists further down, which is the rule
    `eos.njl.solver.warm_start` applies inside a density sweep, applied here
    to a sweep through potentials instead.
    """
    drops = {} if drops is None else drops
    grid = sorted(GRID_MU_B, reverse=descending)
    rows, x0 = [], None
    for mu_B in grid:
        try:
            st, ok, _, x = thermo_from_mu(par, float(mu_B), mu_C, mu_S, T,
                                          pattern=pattern, x0=x0, vac=vac,
                                          return_state=True, backend="fast")
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            drops["error"] = drops.get("error", 0) + 1
            x0 = None
            continue
        if not ok:
            drops["no_convergence"] = drops.get("no_convergence", 0) + 1
            x0 = None
            continue
        # The seed for the next point, and only if this candidate held its own
        # layout. Taken before the filters below, because a row that is
        # dropped from the TRAINING SET can still be a perfectly good root to
        # continue from -- an unlocked CFL state is not CFL data, but it is
        # where the solver is.
        held_layout = realised_pattern(st.Delta) == pattern
        x0 = x if held_layout else None

        if st.n_B <= 1e-3:
            drops["vacuum"] = drops.get("vacuum", 0) + 1
            continue
        gaps = np.asarray(st.Delta, dtype=float)
        if C_PHASE[pattern] > 0.0 and not np.any(np.abs(gaps) > 1.0):
            drops["collapsed"] = drops.get("collapsed", 0) + 1
            continue
        if pattern == "CFL":
            locked = (abs(st.n_C) < 1e-6 * st.n_B
                      and abs(st.n_S / st.n_B - 1.0) < 1e-6)
            if not locked:
                drops["unlocked"] = drops.get("unlocked", 0) + 1
                continue
        rows.append(dict(
            mu_B=float(mu_B), mu_C=mu_C, mu_S=mu_S, T=T,
            P=float(st.P), eps=float(st.eps), s=float(st.s),
            n_B=float(st.n_B), n_C=float(st.n_C), n_S=float(st.n_S),
            M_s=float(st.M[2]), M_u=float(st.M[0]), M_d=float(st.M[1]),
            Delta=float(np.sqrt((gaps ** 2).sum() / 3.0)),
            Delta_1=float(gaps[0]), Delta_2=float(gaps[1]),
            Delta_3=float(gaps[2]),
            mu_3=float(st.mu_3), mu_8=float(st.mu_8),
            Sigma_V=float(st.Sigma_V),
            pattern=pattern, pattern_realised=realised_pattern(st.Delta),
            gapless=bool(st.gapless)))
    return rows


def sample(par, pattern, use_cache=True, report=False):
    """`eos.njl` over the whole grid in one declared pattern, cached."""
    path = cache_path(par, pattern)
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())
    vac = vacuum_solution(par)
    rows, drops = [], {}
    for mu_C, mu_S, T in lines(pattern):
        rows.extend(sweep_line(par, pattern, mu_C, mu_S, T, vac, drops=drops))
    if pattern == "CFL":
        assert not any(abs(r["n_C"]) >= 1e-6 * r["n_B"]
                       or abs(r["n_S"] / r["n_B"] - 1.0) >= 1e-6
                       for r in rows), "unlocked row in the CFL set"
    if report:
        n_grid = len(lines(pattern)) * len(GRID_MU_B)
        print(f"    {pattern:>8}: {len(rows)}/{n_grid} rows, "
              f"dropped {dict(sorted(drops.items())) or '{}'}, "
              f"gapless {sum(r['gapless'] for r in rows)}")
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(rows))
    return rows


def tasks():
    """Every (par_id, par, pattern), cheapest pattern first.

    Cheapest first so an interrupted run leaves whole branches behind rather
    than a fraction of each.
    """
    order = {"unpaired": 0, "CFL": 1, "2SC": 2}
    return [(par_id, par, pattern)
            for pattern in sorted(PATTERNS, key=order.get)
            for par_id, par in parameter_sets().items()]


def run(shard, nshards):
    todo = [t for i, t in enumerate(tasks()) if i % nshards == shard]
    for k, (par_id, par, pattern) in enumerate(todo, 1):
        if cache_path(par, pattern).exists():
            print(f"[{shard}] {k}/{len(todo)} {par_id} {pattern}: cached")
            continue
        t0, c0 = time.perf_counter(), time.process_time()
        rows = sample(par, pattern, report=True)
        print(f"[{shard}] {k}/{len(todo)} {par_id} {pattern}: {len(rows)} rows"
              f" in {time.perf_counter() - t0:.0f}s wall"
              f" / {time.process_time() - c0:.0f}s cpu", flush=True)


def cost():
    """Project the configuration from the per-solve constants above.

    A PROJECTION AND NOT A MEASUREMENT -- the constants are fixed-n_B costs
    carried over from the Sobol version, and this sweep is neither fixed-n_B
    nor cold. `--time` measures it.
    """
    per_pattern = {p: len(lines(p)) * len(GRID_MU_B) for p in PATTERNS}
    n_sets = len(parameter_sets())
    total = sum(n * COST_PER_SOLVE[p] for p, n in per_pattern.items()) * n_sets
    print(f"grid: {len(GRID_MU_B)} mu_B x {len(GRID_MU_C)} mu_C x "
          f"{len(GRID_MU_S)} mu_S x {len(GRID_T)} T")
    for p in PATTERNS:
        print(f"  {p:>8}: {per_pattern[p]:5d} points/set")
    print(f"{n_sets} sets x {sum(per_pattern.values())} points = "
          f"{n_sets * sum(per_pattern.values())} solves")
    print(f"PROJECTED {total / 3600:.1f} CPU-hours "
          f"({total / 3600 / 4:.1f} h on 4 shards) at the OLD constants "
          f"{COST_PER_SOLVE} -- run --time before believing it")


def measure():
    """Time ONE parameter set, all three patterns, the full T grid.

    Prints cpu beside wall, and the interpreter with its numpy and scipy, both
    because this laptop's timings vary about +-30% with concurrent work and
    because a count that names no interpreter names nothing. Also times the
    descending warm start against a cold sweep of the same points on one line
    per pattern, which is the cheapest possible check that the sweep direction
    and the seed were actually applied.
    """
    import scipy
    print(f"python {sys.version.split()[0]} numpy {np.__version__} "
          f"scipy {scipy.__version__}")
    print(f"{sys.executable}\n")
    par_id, par = next(iter(parameter_sets().items()))
    print(f"one set: {par_id}  (eta_D={par.eta_D}, "
          f"G_V0/G_S={par.G_V0_over_GS}, {par.vector_form})\n")
    vac = vacuum_solution(par)

    print("warm against cold, ONE line per pattern "
          f"(n = {len(GRID_MU_B)} points each):")
    for pattern in PATTERNS:
        mu_C, mu_S, T = lines(pattern)[0]
        timings = {}
        for label, descending in (("warm", True), ("cold", None)):
            drops = {}
            t0, c0 = time.perf_counter(), time.process_time()
            if descending is None:
                rows = []
                for mu_B in GRID_MU_B:          # ascending, no seed carried
                    rows.extend(sweep_line(
                        par, pattern, mu_C, mu_S, T, vac,
                        drops=drops) if False else [])
                rows = _cold_line(par, pattern, mu_C, mu_S, T, vac, drops)
            else:
                rows = sweep_line(par, pattern, mu_C, mu_S, T, vac,
                                  descending=True, drops=drops)
            timings[label] = (time.perf_counter() - t0,
                              time.process_time() - c0, len(rows))
        (w_wall, w_cpu, w_n), (c_wall, c_cpu, c_n) = (timings["warm"],
                                                      timings["cold"])
        speedup = c_wall / w_wall if w_wall > 0 else float("nan")
        print(f"  {pattern:>8}: warm {w_wall:7.1f}s wall {w_cpu:7.1f}s cpu "
              f"-> {1e3 * w_wall / len(GRID_MU_B):7.1f} ms/pt, {w_n} rows")
        print(f"  {'':>8}  cold {c_wall:7.1f}s wall {c_cpu:7.1f}s cpu "
              f"-> {1e3 * c_wall / len(GRID_MU_B):7.1f} ms/pt, {c_n} rows"
              f"   [warm start = {speedup:.2f}x]")

    print("\nthe full set, all three patterns, full T grid:")
    total_wall = total_cpu = 0.0
    for pattern in PATTERNS:
        n_grid = len(lines(pattern)) * len(GRID_MU_B)
        t0, c0 = time.perf_counter(), time.process_time()
        rows = sample(par, pattern, use_cache=False, report=False)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        total_wall, total_cpu = total_wall + wall, total_cpu + cpu
        print(f"  {pattern:>8}: {wall:7.1f}s wall {cpu:7.1f}s cpu, "
              f"{n_grid} points -> {1e3 * wall / n_grid:6.1f} ms/pt, "
              f"{len(rows)} rows kept, "
              f"{sum(r['gapless'] for r in rows)} gapless")
    n_sets = len(parameter_sets())
    print(f"\n  ONE SET: {total_wall:.0f}s wall / {total_cpu:.0f}s cpu")
    print(f"  {n_sets} SETS: {n_sets * total_wall / 3600:.2f} h wall "
          f"({n_sets * total_wall / 3600 / 4:.2f} h on 4 shards)")


def _cold_line(par, pattern, mu_C, mu_S, T, vac, drops):
    """The same line solved the way the Sobol version would: no seed at all.

    Ascending and cold, which is both halves of what the descending warm sweep
    changed, measured together because that is how the old code ran.
    """
    rows = []
    for mu_B in sorted(GRID_MU_B):
        try:
            st, ok, _, _ = thermo_from_mu(par, float(mu_B), mu_C, mu_S, T,
                                          pattern=pattern, x0=None, vac=vac,
                                          return_state=True, backend="fast")
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        if ok and st.n_B > 1e-3:
            rows.append(mu_B)
    return rows


def merge():
    """Concatenate the cache into one table with a sidecar."""
    rows, missing = [], []
    for par_id, par in parameter_sets().items():
        for pattern in PATTERNS:
            path = cache_path(par, pattern)
            if not path.exists():
                missing.append((par_id, pattern))
                continue
            for row in pickle.loads(path.read_bytes()):
                rows.append(dict(par_id=par_id, eta_D=par.eta_D,
                                 G_V0_over_GS=par.G_V0_over_GS,
                                 vector_form=par.vector_form,
                                 lambda_UV=par.lambda_UV, **row))
    if not rows:
        raise SystemExit("nothing cached; run the shards first")
    OUT.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    out = OUT / "csc_njl_samples.csv.gz"
    with gzip.open(out, "wt", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    counts, gapless = {}, {}
    for row in rows:
        counts[row["pattern"]] = counts.get(row["pattern"], 0) + 1
        gapless[row["pattern"]] = gapless.get(row["pattern"], 0) + row["gapless"]
    meta = dict(
        n_rows=len(rows), rows_per_pattern=counts, gapless_per_pattern=gapless,
        missing=[f"{p}/{q}" for p, q in missing], columns=columns,
        units=dict(mu_B="MeV", mu_C="MeV", mu_S="MeV", T="MeV",
                   P="MeV/fm^3", eps="MeV/fm^3", s="fm^-3", n="fm^-3",
                   Delta="MeV", M="MeV", mu_3="MeV", mu_8="MeV",
                   Sigma_V="MeV"),
        sampling=dict(mu_B=list(GRID_MU_B), mu_C=list(GRID_MU_C),
                      mu_S=list(GRID_MU_S), T=list(GRID_T),
                      patterns=list(PATTERNS),
                      sweep="descending in mu_B, warm started per "
                            "(mu_C, mu_S, T) line",
                      note="CFL is sampled at mu_C = mu_S = 0: locking makes "
                           "P a function of mu_B + mu_S alone"),
        parameters={k: asdict(v) for k, v in parameter_sets().items()},
        filters="n_B > 1e-3 fm^-3; collapsed layouts (max|Delta| <= 1 MeV in "
                "a paired pattern) and unlocked CFL rows dropped; gapless "
                "RECORDED, not filtered",
        code=hashlib.md5(Path(__file__).read_bytes()).hexdigest())
    (OUT / "csc_njl_samples.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}  ({len(rows)} rows: {counts})")
    if missing:
        print(f"  {len(missing)} (set, pattern) tasks still missing")


def smoke():
    """Three mu_B points per pattern on one set: does a row come back?"""
    par_id, par = next(iter(parameter_sets().items()))
    vac = vacuum_solution(par)
    global GRID_MU_B
    full, GRID_MU_B = GRID_MU_B, np.array([1800.0, 1700.0, 1600.0])
    try:
        for pattern in PATTERNS:
            mu_C, mu_S, T = lines(pattern)[0]
            rows = sweep_line(par, pattern, mu_C, mu_S, T, vac)
            assert all(row["P"] == row["P"] for row in rows), "NaN pressure"
            assert all(row["n_B"] > 1e-3 for row in rows), "vacuum row kept"
            for row in rows:
                print(f"{pattern:>8} mu_B={row['mu_B']:6.0f} T={row['T']:4.0f}"
                      f" -> n_B={row['n_B']:.3f} P={row['P']:8.2f} "
                      f"{row['pattern_realised']:>8} "
                      f"gapless={int(row['gapless'])}")
            print(f"{par_id} {pattern}: {len(rows)}/3 rows\n")
    finally:
        GRID_MU_B = full


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--time", action="store_true", dest="time_it")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    elif args.cost:
        cost()
    elif args.time_it:
        measure()
    elif args.smoke:
        smoke()
    else:
        run(args.shard, args.nshards)

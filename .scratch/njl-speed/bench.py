#     sed -n '/BENCH_BLOCK_BEGIN/,$p' notebooks/quark_timing.py \
#         | NJL_BENCH=1 python3 -
#
# from the repository root, or, in a notebook, by setting BENCH_RUN = True in
# this cell. The marker runs to END OF FILE rather than to a closing one, so
# this block stays LAST in this file: a closing marker cannot be written in a
# comment that also shows the command, because the command's own text matches
# it first.
#
# WHAT IS FIXED, AND WHY IT IS NOT COSMETIC
#
#   The parameter set is `rg_njl1` -- the shipped RG-consistent set,
#   lambda_UV = 10, eta_D = 1.45, eta_V = 0.7, vector_form "constant" -- taken
#   unmodified, so nothing here is a held variant somebody has to reproduce.
#
#   THE SPAN. BENCH_NB runs 0.5 -> 1.55 fm^-3 and stays ABOVE chiral
#   restoration: every point converges, so ms/point measures solver speed and
#   not the cost of failures. BENCH_NB_CHIRAL runs 0.30 -> 1.55 and CROSSES
#   the restoration, which is the span a hybrid star actually needs and a
#   materially different problem. Both are legitimate; the pinned number is
#   the first, and the second is measured alongside so the cost of crossing is
#   on the record rather than discovered later.
#
#   THE GRID DENSITY. 200 points, because the sweep is warm-started: a coarser
#   grid takes larger steps, needs more Newton iterations per step, and
#   reports a DIFFERENT ms/point for the same solver. A per-point number is
#   only comparable at a fixed grid, so the grid is part of the benchmark.
#
#   THE PATTERN LIST, three ways. `None` is njl's own enumeration
#   (DEFAULT_PATTERNS, four candidates); "three" drops "free"; and each single
#   pattern is timed alone, because "solve fewer patterns" is one of the three
#   levers this effort tests and it cannot be costed without the parts.
#
#   THE STACK. python.org 3.14, not anaconda 3.9. CLAUDE.md section 12 names
#   3.14 as the stack the baselines are frozen against, and the two disagree:
#   scipy 1.13's root(..., method="hybr") reports success while returning its
#   seed unchanged on one of eos's closures. A solver-speed measurement taken
#   on the stack whose solver silently no-ops measures nothing. The
#   interpreter and its numpy and scipy are printed with every result, so a
#   number that names no stack cannot be quoted by accident.
#
#   n = 3, MEDIAN, AND CPU TIME BESIDE WALL TIME. Wall-clock on this laptop
#   moves +-30% with concurrent work, so one run is an anecdote; worse, a
#   machine that idles or throttles mid-run corrupts a wall-clock median
#   silently, and "nothing else was running" is not something a reader of the
#   number can check. Measured here on a first attempt: 63 minutes of elapsed
#   time against 9 minutes of CPU. So every result carries `time.process_time`
#   beside `time.perf_counter`, and the two agreeing IS the quiet-machine
#   check. They should track on this single-threaded path; where they do not,
#   the run is contaminated and the number is not quotable.
#
#   A measurement is never wrapped in `timeout` -- that binary is x86_64 here
#   and drags the interpreter under Rosetta, where numpy fails with an error
#   naming a cause it does not have.
#
# THE MIXED HALF is not a second harness either: same parameter set, same
# flags, same pattern restriction, with a real hadronic partner. That partner
# is DID, the pairing this effort's own scale figures were taken on, and DID
# is nucleonic here so the hadronic branch does not end in scalar collapse
# before the quark pressure catches it.
#
# WHAT IS MEASURED IS ONE `eos.mixed.eos_point`, NOT A TABLE, and the reason
# is measured rather than assumed: `eos.mixed.eos_table` with an ENUMERATING
# njl adapter spends its time in the window locator, and on this pairing that
# locator did not return in 24 minutes on a four-point grid. An unbounded
# quantity cannot be a baseline. A single point at a density inside the window
# IS bounded, and it is also what the 60 s target is really made of:
# window_only=True means a 200-point hybrid grid solves the mixed system only
# at the densities between the boundaries, so
#
#     table cost  ~  window location  +  (points inside) x (cost per point)
#
# and this block measures the third factor and counts the second. The first is
# ticket 07's problem, and its size is the finding above.
#
# THE WINDOW IS SCANNED WITH THE SAME ENUMERATION THAT IS TIMED. Holding one
# pattern for the scan is cheaper and was tried; it is also wrong, and visibly
# so -- it put the coexistence window somewhere the enumeration calls pure
# quark, so the timed point was a chi = 1 solve wearing a mixed-phase label. A
# held pattern is a different physical statement and can seat a metastable
# branch, and this is where that bites. The engine reports which side of the
# transition a density is on by returning chi outside [0, 1] on a CONVERGED
# point, so the scan needs no construction of its own: it reads chi, and the
# timed point reports its own chi so a reader can see it was inside.
#
# --- BENCH_BLOCK_BEGIN -----------------------------------------------------
# ===========================================================================
import os
import sys
import time
from dataclasses import replace

import numpy as np

from eos import njl
from eos.did.parameters import Parameters as DIDParameters
from eos.did.species import SpeciesFlags as DIDFlags
from eos.mixed.adapters import did_phase, njl_phase
from eos.mixed.api import eos_point as mixed_eos_point
from eos.mixed.species import SpeciesFlags as MixedFlags

BENCH_SET = "rg_njl1"
BENCH_MODE = "beta_eq_neutrinoless"
BENCH_T = 0.0                                    # MeV
BENCH_SPECIES = njl.SpeciesFlags(csc=True)       # muons off, in both halves
BENCH_BACKEND = "fast"
BENCH_NB = np.linspace(0.5, 1.55, 200)           # fm^-3, above restoration
BENCH_NB_CHIRAL = np.linspace(0.30, 1.55, 200)   # fm^-3, crosses it
BENCH_REPEATS = 3
#: name -> the `patterns` argument. None is njl's own four-candidate default.
BENCH_PATTERN_SETS = {
    "default (4)": None,
    "three": ("unpaired", "2SC", "CFL"),
    "unpaired": ("unpaired",),
    "2SC": ("2SC",),
    "CFL": ("CFL",),
    "free": ("free",),
}
#: the restriction the mixed half and the chiral-span run both use.
BENCH_PATTERNS_MAIN = ("unpaired", "2SC", "CFL")

BENCH_MIXED_ETA = 0.0                            # Gibbs
BENCH_MIXED_SCAN = np.linspace(0.60, 1.55, 20)   # where to look for chi in (0,1)
BENCH_MIXED_NB_TARGET = np.linspace(0.08, 1.60, 200)   # what 60 s must cover

BENCH_RUN = bool(os.environ.get("NJL_BENCH"))
# ---------------------------------------------------------------------------


def bench_stack():
    """The line every number in this effort has to carry with it."""
    import scipy
    return (f"python {sys.version.split()[0]} ({sys.executable}), "
            f"numpy {np.__version__}, scipy {scipy.__version__}")


def timed(call, repeats):
    """`call` run `repeats` times: (median wall, median cpu, wall runs, last).

    CPU time is not decoration. A wall-clock median taken while the machine
    idled or throttled is wrong and looks fine; the two clocks agreeing is
    what says it did not happen.
    """
    wall = []
    cpu = []
    out = None
    for _ in range(repeats):
        w0, c0 = time.perf_counter(), time.process_time()
        out = call()
        wall.append(time.perf_counter() - w0)
        cpu.append(time.process_time() - c0)
    return float(np.median(wall)), float(np.median(cpu)), wall, out


def bench_table(patterns, n_B, repeats=BENCH_REPEATS):
    """`njl.eos_table` over the pinned configuration.

    Returns (median wall, median cpu, per-run wall, n_rows). A warm-up call on
    two densities is DISCARDED first: the fast backend jits on first entry and
    that compile is not part of a per-point cost.
    """
    par = njl.Parameters.named(BENCH_SET)
    axes = {"nB": n_B, "T": np.array([BENCH_T])}

    njl.eos_table(par, BENCH_MODE, BENCH_SPECIES,
                  {"nB": n_B[:2], "T": np.array([BENCH_T])},
                  patterns=patterns, backend=BENCH_BACKEND)

    def once():
        return njl.eos_table(par, BENCH_MODE, BENCH_SPECIES, axes,
                             patterns=patterns, backend=BENCH_BACKEND)

    wall, cpu, runs, result = timed(once, repeats)
    return wall, cpu, runs, len(njl.table.rows_from_result(result))


def bench_phases(patterns):
    """The DID + NJL pairing, njl restricted to `patterns`."""
    return (did_phase(DIDParameters.default(), DIDFlags()),
            njl_phase(njl.Parameters.named(BENCH_SET), BENCH_SPECIES,
                      patterns=patterns, backend=BENCH_BACKEND))


def mixed_chi(patterns, n_B):
    """chi at one density, or None if the point did not converge.

    A converged point whose chi is outside [0, 1] is not a failure: it is the
    engine saying this density sits on one side of the transition.
    """
    result = mixed_eos_point(bench_phases(patterns), BENCH_MODE, MixedFlags(),
                             n_B=n_B, T=BENCH_T, eta=BENCH_MIXED_ETA)
    return result.point.chi if result.ok else None


def bench_mixed_window(patterns=BENCH_PATTERNS_MAIN,
                       densities=BENCH_MIXED_SCAN):
    """The densities on `densities` where chi lies in (0, 1).

    Returns (inside, chis) -- the coexistence densities and the whole scan,
    since a scan that found nothing has to be readable too."""
    chis = {}
    for n_B in densities:
        chis[float(n_B)] = mixed_chi(patterns, float(n_B))
    inside = [n_B for n_B, chi in chis.items()
              if chi is not None and 0.0 < chi < 1.0]
    return inside, chis


def bench_mixed_point(patterns, n_B, repeats=BENCH_REPEATS):
    """Median wall time of ONE converged mixed point at the enumeration."""
    phases = bench_phases(patterns)

    def once():
        return mixed_eos_point(phases, BENCH_MODE, MixedFlags(), n_B=n_B,
                               T=BENCH_T, eta=BENCH_MIXED_ETA)

    wall, cpu, runs, result = timed(once, repeats)
    chi = result.point.chi if result.ok else None
    return wall, cpu, runs, result.ok, chi


if BENCH_RUN:
    print(f"=== njl-speed benchmark ===\n{bench_stack()}")
    print(f"set {BENCH_SET}, {BENCH_MODE}, T = {BENCH_T} MeV, "
          f"csc={BENCH_SPECIES.csc}, backend {BENCH_BACKEND!r}, "
          f"median of {BENCH_REPEATS}\n")

    print(f"--- table, n_B = {BENCH_NB[0]:.2f} -> {BENCH_NB[-1]:.2f} fm^-3, "
          f"{len(BENCH_NB)} points (above chiral restoration) ---")
    for name, patterns in BENCH_PATTERN_SETS.items():
        wall, cpu, runs, n_rows = bench_table(patterns, BENCH_NB)
        spread = f"{min(runs):.1f}-{max(runs):.1f}"
        print(f"  {name:12s} {wall * 1e3 / len(BENCH_NB):9.1f} ms/pt wall   "
              f"{cpu * 1e3 / len(BENCH_NB):9.1f} cpu   "
              f"{wall:8.1f} s median (runs {spread})   {n_rows} rows",
              flush=True)

    print(f"\n--- table, n_B = {BENCH_NB_CHIRAL[0]:.2f} -> "
          f"{BENCH_NB_CHIRAL[-1]:.2f} fm^-3, {len(BENCH_NB_CHIRAL)} points "
          f"(crosses chiral restoration) ---")
    wall, cpu, runs, n_rows = bench_table(BENCH_PATTERNS_MAIN, BENCH_NB_CHIRAL)
    print(f"  {'three':12s} {wall * 1e3 / len(BENCH_NB_CHIRAL):9.1f} ms/pt wall"
          f"   {cpu * 1e3 / len(BENCH_NB_CHIRAL):9.1f} cpu   "
          f"{wall:8.1f} s median   {n_rows} rows", flush=True)

    print(f"\n--- mixed, did + njl, eta = {BENCH_MIXED_ETA:g}, T = "
          f"{BENCH_T} MeV ---", flush=True)
    w0, c0 = time.perf_counter(), time.process_time()
    inside, chis = bench_mixed_window()
    print(f"  window scan   {time.perf_counter() - w0:8.1f} s wall, "
          f"{time.process_time() - c0:8.1f} cpu, "
          f"{len(BENCH_MIXED_SCAN)} densities, {BENCH_PATTERNS_MAIN}")
    solved = [n for n, chi in chis.items() if chi is not None]
    print(f"  converged at  {len(solved)} of {len(chis)} densities; "
          f"chi in (0, 1) at {len(inside)}")
    if not inside:
        print("  NO coexistence density found -- no mixed point to time. "
              "The scan, n_B: chi ->")
        for n_B, chi in chis.items():
            print(f"    {n_B:6.3f}  {'-' if chi is None else f'{chi:+.4f}'}")
    else:
        n_B_mixed = inside[len(inside) // 2]        # mid-window
        lo, hi = min(inside), max(inside)
        target_inside = int(np.sum((BENCH_MIXED_NB_TARGET >= lo)
                                   & (BENCH_MIXED_NB_TARGET <= hi)))
        print(f"  coexistence   {lo:.3f} -> {hi:.3f} fm^-3 on this scan; a "
              f"{len(BENCH_MIXED_NB_TARGET)}-point grid puts {target_inside} "
              f"points inside it")
        print(f"  timing one point at n_B = {n_B_mixed:.3f} fm^-3", flush=True)
        for name in ("three", "CFL", "2SC"):
            patterns = (BENCH_PATTERNS_MAIN if name == "three" else (name,))
            wall, cpu, runs, ok, chi = bench_mixed_point(patterns, n_B_mixed)
            spread = f"{min(runs):.1f}-{max(runs):.1f}"
            # chi is REPORTED, not assumed: the scan that chose this density
            # held one pattern, and the enumeration may put the transition
            # somewhere else, which would make this a pure-phase solve.
            print(f"    {name:10s} {wall:8.1f} s/point wall, {cpu:8.1f} cpu "
                  f"(runs {spread})  converged={ok}  chi="
                  f"{'-' if chi is None else f'{chi:+.4f}'}", flush=True)

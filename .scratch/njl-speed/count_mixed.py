"""Ticket 04: count the NJL solves one hybrid row costs.

Instrumentation only -- nothing in `eos/` is edited. Every counter is a
monkey-patch applied from here, so the tree stays landable.

WHAT IS COUNTED, at three depths:

  mixed residual      `eos.mixed.solver.residual`, one per scipy call; split
                      into step evaluations and finite-difference Jacobian
                      evaluations by reading the x-trace of each `root` call
                      (MINPACK's fdjac1 perturbs one coordinate at a time from
                      a common base, so a group of n such calls is a Jacobian).
  Phase.thermo        one per phase per residual, plus the extra pair
                      `result_from_root` takes; this is the ticket's "NJL
                      solve per hybrid row".
  thermo_from_mu      the NJL internal solve at fixed potentials -- exactly
                      len(patterns) per quark `thermo` call.

WHERE, from the call stack: refine_window, locate.bisect, locate.walk,
locate.scan, solve_fixed_chi, wing, or the bare sweep (a table row). The
sweep's own retry bisection is read off the innermost `step` frame's `depth`.

Configuration is ticket 02's pinned benchmark: rg_njl1, csc=True, backend
'fast', beta_eq_neutrinoless, T = 0, eta = 0, DID + NJL, three patterns.
"""
import dataclasses
import os
import sys
import time
from collections import defaultdict

import numpy as np

import eos.mixed.boundaries as B
import eos.mixed.solver as S
import eos.njl.thermodynamics as NT
from eos import njl
from eos.did.parameters import Parameters as DIDParameters
from eos.did.species import SpeciesFlags as DIDFlags
from eos.mixed.adapters import did_phase, njl_phase
from eos.mixed.species import SpeciesFlags as MixedFlags

BENCH_SET = "rg_njl1"
BENCH_MODE = "beta_eq_neutrinoless"
BENCH_T = 0.0
BENCH_SPECIES = njl.SpeciesFlags(csc=True)
BENCH_BACKEND = "fast"
BENCH_PATTERNS = ("unpaired", "2SC", "CFL")
BENCH_ETA = 0.0
#: the grid a 60 s hybrid table has to cover (ticket 02's BENCH_MIXED_NB_TARGET)
TARGET_GRID = np.linspace(0.08, 1.60, 200)
#: where ticket 02's 20-density scan put the window
WINDOW = (0.800, 1.200)


# ---------------------------------------------------------------------------
# the counters
# ---------------------------------------------------------------------------

#: site -> {counter name: value}
TALLY = defaultdict(lambda: defaultdict(float))
#: per-`root`-call x traces, tagged with the site they were taken at
ROOT_CALLS = []
_stack = []                      # open root calls

#: innermost stack frame name -> the algorithmic site it belongs to
SITES = {
    "refine": "refine_window",
    "bisect": "locate.bisect",
    "walk_to_crossing": "locate.walk",
    "scan": "locate.scan",
    "solve_fixed_chi": "solve_fixed_chi",
    "wing_sweep": "wing",
    "_wing_point": "wing",
}


def site_and_depth():
    """(algorithmic site, sweep retry depth) of the current call."""
    site, depth = "sweep", 0
    f = sys._getframe(1)
    while f is not None:
        name = f.f_code.co_name
        if site == "sweep" and name in SITES:
            site = SITES[name]
        if name == "step" and "depth" in f.f_locals:
            depth = max(depth, int(f.f_locals["depth"]))
        f = f.f_back
    return site, depth


def bump(site, key, value=1.0):
    TALLY[site][key] += value


def wrap_root(real):
    def counted(fun, x0, *a, **kw):
        site, depth = site_and_depth()
        rec = {"site": site, "depth": depth, "x": [], "n": len(x0)}
        ROOT_CALLS.append(rec)
        _stack.append(rec)
        try:
            sol = real(fun, x0, *a, **kw)
            rec["nfev"] = int(getattr(sol, "nfev", 0))
            rec["njev"] = int(getattr(sol, "njev", 0))
            return sol
        finally:
            _stack.pop()
    return counted


def wrap_residual(real):
    def counted(x, ctx):
        site, depth = site_and_depth()
        bump(site, "residual")
        bump(site, f"residual.depth{depth}")
        if _stack:
            _stack[-1]["x"].append(np.array(x, dtype=float))
        return real(x, ctx)
    return counted


def wrap_thermo(phase, tag):
    real = phase.thermo

    def counted(*a, **kw):
        site, depth = site_and_depth()
        t0 = time.perf_counter()
        try:
            return real(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            bump(site, f"thermo.{tag}")
            bump(site, f"thermo.{tag}.s", dt)
    return dataclasses.replace(phase, thermo=counted)


def wrap_seed(phase):
    real = phase.seed
    if real is None:
        return phase

    def counted(*a, **kw):
        site, _ = site_and_depth()
        t0 = time.perf_counter()
        try:
            return real(*a, **kw)
        finally:
            bump(site, "seed")
            bump(site, "seed.s", time.perf_counter() - t0)
    return dataclasses.replace(phase, seed=counted)


def install():
    """Patch every name the mixed engine reaches these through."""
    for mod in (S, B):
        mod.root = wrap_root(mod.root)
        mod.residual = wrap_residual(mod.residual)

    real_tfm = NT.thermo_from_mu

    def counted_tfm(par, mu_B, *a, **kw):
        site, _ = site_and_depth()
        t0 = time.perf_counter()
        try:
            return real_tfm(par, mu_B, *a, **kw)
        finally:
            bump(site, "thermo_from_mu")
            bump(site, "thermo_from_mu.s", time.perf_counter() - t0)

    NT.thermo_from_mu = counted_tfm

    # The wings and the adapter's cold start go through njl's OWN solver, not
    # through `thermo_from_mu`, so they need their own counter or a wing row
    # reads as free.
    import eos.njl.solver as NS
    real_sp = NS.solve_pattern

    def counted_sp(*a, **kw):
        site, _ = site_and_depth()
        t0 = time.perf_counter()
        try:
            return real_sp(*a, **kw)
        finally:
            bump(site, "solve_pattern")
            bump(site, "solve_pattern.s", time.perf_counter() - t0)

    NS.solve_pattern = counted_sp
    # `njl_phase` closes over `thermo_from_mu` at factory time, so the factory
    # must be called AFTER this patch lands. `make_phases` below does.

    real_guess = S.default_guess

    def counted_guess(ctx):
        site, _ = site_and_depth()
        bump(site, "cold_guess")
        return real_guess(ctx)

    S.default_guess = counted_guess
    B.default_guess = counted_guess


def reset():
    TALLY.clear()
    ROOT_CALLS.clear()


# ---------------------------------------------------------------------------
# the pairing
# ---------------------------------------------------------------------------

def warmup():
    """Compile the jitted kernel before anything is timed.

    The fast backend jits on first entry; ticket 02's medians exclude that
    compile and so must these numbers, or the first run of a process reports
    the compiler's time as the solver's.
    """
    _, p_Q = make_phases()
    p_Q.thermo(1600.0, -120.0, 0.0, BENCH_T)
    reset()


def make_phases():
    p_H = did_phase(DIDParameters.default(), DIDFlags())
    p_Q = njl_phase(njl.Parameters.named(BENCH_SET), BENCH_SPECIES,
                    patterns=BENCH_PATTERNS, backend=BENCH_BACKEND)
    return (wrap_seed(wrap_thermo(p_H, "H")),
            wrap_seed(wrap_thermo(p_Q, "Q")))


# ---------------------------------------------------------------------------
# reading the x-traces
# ---------------------------------------------------------------------------

def classify(rec):
    """(step evaluations, Jacobian evaluations) in one `root` call.

    MINPACK's fdjac1 perturbs ONE coordinate at a time from a common base, so
    a residual whose x differs from the running base in exactly one coordinate
    is a Jacobian column; anything else is a trial step and becomes the new
    base.
    """
    xs = rec["x"]
    if not xs:
        return 0, 0
    base, steps, jac = xs[0], 1, 0
    for x in xs[1:]:
        if int(np.count_nonzero(x != base)) == 1:
            jac += 1
        else:
            steps += 1
            base = x
    return steps, jac


def report(title, extra=()):
    print(f"\n=== {title} ===", flush=True)
    for line in extra:
        print(f"  {line}")
    sites = sorted(TALLY)
    print(f"  {'site':16s} {'resid':>6s} {'thH':>6s} {'thQ':>6s} "
          f"{'tfm':>7s} {'seed':>5s} {'cold':>5s} {'thQ s':>8s} {'tfm s':>8s} {'sPat':>5s}")
    for site in sites:
        t = TALLY[site]
        print(f"  {site:16s} {t['residual']:6.0f} {t['thermo.H']:6.0f} "
              f"{t['thermo.Q']:6.0f} {t['thermo_from_mu']:7.0f} "
              f"{t['seed']:5.0f} {t['cold_guess']:5.0f} "
              f"{t['thermo.Q.s']:8.1f} {t['thermo_from_mu.s']:8.1f} "
              f"{t['solve_pattern']:5.0f}")
    tot = defaultdict(float)
    for t in TALLY.values():
        for k, v in t.items():
            tot[k] += v
    print(f"  {'TOTAL':16s} {tot['residual']:6.0f} {tot['thermo.H']:6.0f} "
          f"{tot['thermo.Q']:6.0f} {tot['thermo_from_mu']:7.0f} "
          f"{tot['seed']:5.0f} {tot['cold_guess']:5.0f} "
          f"{tot['thermo.Q.s']:8.1f} {tot['thermo_from_mu.s']:8.1f} "
          f"{tot['solve_pattern']:5.0f}")

    by_depth = {k: v for k, v in tot.items() if k.startswith("residual.depth")}
    if by_depth:
        print("  sweep retry depth:",
              ", ".join(f"{k.split('depth')[1]}={v:.0f}"
                        for k, v in sorted(by_depth.items())))

    nfev = sum(r.get("nfev", 0) for r in ROOT_CALLS)
    print(f"  scipy nfev {nfev}")
    print(f"  {len(ROOT_CALLS)} root calls   "
          f"{'site':16s} {'n':>2s} {'steps':>6s} {'jac':>5s}")
    agg = defaultdict(lambda: [0, 0, 0])
    for rec in ROOT_CALLS:
        st, jc = classify(rec)
        a = agg[(rec["site"], rec["n"])]
        a[0] += 1
        a[1] += st
        a[2] += jc
    for (site, n), (calls, st, jc) in sorted(agg.items()):
        print(f"      {site:16s} n={n} calls={calls:4d} steps={st:6d} "
              f"jac={jc:6d}  ({jc / max(st + jc, 1):.0%} of residuals)")
    # Per-call, because an average over `root` calls hides the one row that
    # cost four times its neighbours -- which is the ticket's question.
    print("  per root call (site:residuals/jac-cols, depth>0 marked *):")
    line = []
    for rec in ROOT_CALLS:
        st, jc = classify(rec)
        mark = "*" if rec["depth"] else ""
        line.append(f"{rec['site']}{mark}:{len(rec['x'])}/{jc}")
    for i in range(0, len(line), 6):
        print("     ", "  ".join(line[i:i + 6]))


def stack_line():
    import scipy
    return (f"python {sys.version.split()[0]} ({sys.executable}), "
            f"numpy {np.__version__}, scipy {scipy.__version__}")


# ---------------------------------------------------------------------------
# the runs
# ---------------------------------------------------------------------------

def run_cold_point():
    """One mixed row from a cold start, deep inside the window."""
    from eos.mixed.charges import beta_eq_neutrinoless
    phases = make_phases()
    reset()
    t0, c0 = time.perf_counter(), time.process_time()
    r = S.solve(phases, 1.000, BENCH_ETA, beta_eq_neutrinoless(), T=BENCH_T)
    report("A. one COLD mixed row, n_B = 1.000 fm^-3",
           [f"wall {time.perf_counter() - t0:.1f} s   "
            f"cpu {time.process_time() - c0:.1f} s   chi = {r.chi:+.4f}"])
    return r


def run_cold_repeat(repeats=3):
    """The cold row three times, to compare against ticket 02's 29.6 s."""
    from eos.mixed.charges import beta_eq_neutrinoless
    walls, cpus = [], []
    for _ in range(repeats):
        phases = make_phases()
        reset()
        t0, c0 = time.perf_counter(), time.process_time()
        r = S.solve(phases, 1.000, BENCH_ETA, beta_eq_neutrinoless(), T=BENCH_T)
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
    report(f"A3. the COLD row x{repeats}, n_B = 1.000 fm^-3 (last run's counts)",
           [f"wall median {np.median(walls):.1f} s  runs "
            f"{' '.join(f'{w:.1f}' for w in walls)}",
            f"cpu  median {np.median(cpus):.1f} s  runs "
            f"{' '.join(f'{c:.1f}' for c in cpus)}",
            f"chi = {r.chi:+.4f}"])


def run_warm_sweep(n_steps=6):
    """Consecutive rows on the target grid, warm-started -- a table row."""
    from eos.mixed.charges import beta_eq_neutrinoless
    phases = make_phases()
    cs = beta_eq_neutrinoless()
    inside = TARGET_GRID[(TARGET_GRID >= 1.000)][:n_steps + 1]
    reset()
    t0, c0 = time.perf_counter(), time.process_time()
    out = S.sweep(phases, inside, BENCH_ETA, cs, T=BENCH_T)
    report(f"B. {len(inside)} WARM rows, n_B = {inside[0]:.3f} -> "
           f"{inside[-1]:.3f} fm^-3 (target-grid spacing "
           f"{inside[1] - inside[0]:.4f})",
           [f"wall {time.perf_counter() - t0:.1f} s   "
            f"cpu {time.process_time() - c0:.1f} s   "
            f"{len(out)} of {len(inside)} solved   "
            f"chi {out[0].chi:+.4f} -> {out[-1].chi:+.4f}"])
    return out


def run_onset(n_steps=6):
    """Rows just above the onset -- is the cost flat across the window?

    Deep-window rows (run B) sit at chi ~ 0.98. The onset is where chi leaves
    0, and a few hard rows there against a flat profile call for completely
    different fixes, which is what the ticket asks.
    """
    from eos.mixed.charges import beta_eq_neutrinoless
    start = float(os.environ.get("NB_START", WINDOW[0]))
    phases = make_phases()
    grid = TARGET_GRID[TARGET_GRID >= start][:n_steps + 1]
    reset()
    t0, c0 = time.perf_counter(), time.process_time()
    out = S.sweep(phases, grid, BENCH_ETA, beta_eq_neutrinoless(), T=BENCH_T)
    chis = " ".join(f"{r.chi:+.3f}" for r in out)
    report(f"E. {len(grid)} rows at the ONSET, n_B = {grid[0]:.3f} -> "
           f"{grid[-1]:.3f} fm^-3",
           [f"wall {time.perf_counter() - t0:.1f} s   "
            f"cpu {time.process_time() - c0:.1f} s   "
            f"{len(out)} of {len(grid)} solved",
            f"chi: {chis}"])
    return out


def run_locator(n_probe=12, refine="exact", hint=(0.60, 1.55)):
    """`locate_window` on the target grid -- the table's first act.

    HINTED, at ticket 02's own scan range. A hint also switches off the probe
    refinement passes (`max_refine` is ignored with one), so this is the
    CHEAPEST form the locator takes -- the second and later temperatures of a
    table, which `_locate_chained` hands the previous line's window. The
    unhinted first line is strictly more expensive, and ticket 02 already
    measured that it does not return in 24 minutes on a four-point grid.
    """
    from eos.mixed.charges import beta_eq_neutrinoless
    phases = make_phases()
    reset()
    t0, c0 = time.perf_counter(), time.process_time()
    w = B.locate_window(phases, TARGET_GRID, BENCH_ETA,
                        beta_eq_neutrinoless(), T=BENCH_T,
                        n_probe=n_probe, refine=refine, hint=hint)
    inside = TARGET_GRID[(TARGET_GRID >= w.n_onset) & (TARGET_GRID <= w.n_offset)]
    print(f"  rows inside the window on the 200-point target grid: "
          f"{len(inside)}")
    report(f"C. locate_window, n_probe={n_probe}, refine={refine!r}, "
           f"hint={hint}",
           [f"wall {time.perf_counter() - t0:.1f} s   "
            f"cpu {time.process_time() - c0:.1f} s",
            f"window {w.n_onset:.4f} -> {w.n_offset:.4f}  "
            f"exists={w.exists}  {len(w.probes)} probes"])
    return w


def run_wing(n_points=6):
    """Pure-quark rows above the window -- the hybrid table's upper wing."""
    from eos.mixed.charges import beta_eq_neutrinoless
    _, p_Q = make_phases()
    grid = TARGET_GRID[TARGET_GRID > WINDOW[1]][:n_points]
    reset()
    t0, c0 = time.perf_counter(), time.process_time()
    out = p_Q.wing_sweep(beta_eq_neutrinoless(), grid, BENCH_T)
    report(f"D. quark wing, {len(grid)} rows, n_B = {grid[0]:.3f} -> "
           f"{grid[-1]:.3f} fm^-3",
           [f"wall {time.perf_counter() - t0:.1f} s   "
            f"cpu {time.process_time() - c0:.1f} s   "
            f"{len(out)} rows back"])
    return out


RUNS = {"cold": run_cold_point, "cold3": run_cold_repeat,
        "onset": run_onset, "warm": run_warm_sweep,
        "locator": run_locator, "wing": run_wing}

if __name__ == "__main__":
    install()
    print(f"ticket 04: NJL solves per hybrid row\n{stack_line()}")
    print(f"{BENCH_SET}, {BENCH_MODE}, T = {BENCH_T}, eta = {BENCH_ETA}, "
          f"patterns {BENCH_PATTERNS}, backend {BENCH_BACKEND!r}", flush=True)
    warmup()
    for name in (sys.argv[1:] or ["cold", "warm", "wing"]):
        RUNS[name]()

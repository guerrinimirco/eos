"""Ticket 11 (and 17): every mixed solve the locator makes, in order.

Layered on `t17_count_lm.py` (itself on ticket 04's `count_mixed.py`), so one
run carries ticket 04's per-site counts, ticket 17's lm census and, here, a
trace of each `eos.mixed.solver.solve` call: which site made it, at what
`sweep` retry depth, at which density, whether it converged (and to what chi),
and what it cost in mixed residuals, NJL internal solves, NJL internal
residual evaluations and cpu. Nothing in `eos/` is edited.

    python3 t11_trace_locator.py hinted | unhinted | bisect-unhinted | onset | warm
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import count_mixed as CM                                        # noqa: E402
import t17_count_lm as L                                        # noqa: E402
import eos.mixed.boundaries as B                                # noqa: E402
import eos.mixed.solver as S                                    # noqa: E402

TRACE = []


def _tot(key):
    return sum(t[key] for t in CM.TALLY.values())


def _evals():
    return sum(sum(r["evals"].values()) for r in L.CALLS)


def _lm_evals():
    return sum(r["evals"].get("lm", 0) for r in L.CALLS)


def install():
    L.install()
    real_solve = S.solve

    def traced(phases, n_B, *a, **kw):
        site, depth = CM.site_and_depth()
        before = (_tot("residual"), len(L.CALLS), _evals(), _lm_evals(),
                  time.process_time())
        rec = {"site": site, "depth": depth, "n_B": float(n_B), "ok": False,
               "chi": float("nan")}
        try:
            r = real_solve(phases, n_B, *a, **kw)
            rec.update(ok=True, chi=float(r.chi))
            return r
        except RuntimeError:
            rec.update(ok=False, chi=float("nan"))
            raise
        finally:
            after = (_tot("residual"), len(L.CALLS), _evals(), _lm_evals(),
                     time.process_time())
            rec.update(resid=int(after[0] - before[0]), tfm=after[1] - before[1],
                       evals=after[2] - before[2], lm=after[3] - before[3],
                       cpu=after[4] - before[4])
            TRACE.append(rec)
            _stream(rec)

    S.solve = traced          # `sweep` reaches `solve` through this global

    real_fc = B.solve_fixed_chi

    def traced_fc(phases, chi, *a, **kw):
        before = (_tot("residual"), len(L.CALLS), _evals(), _lm_evals(),
                  time.process_time())
        rec = {"site": "solve_fixed_chi", "depth": 0,
               "n_B": float(kw.get("n_B0") or float("nan")), "chi_imposed": chi,
               "ok": False, "chi": float("nan")}
        try:
            r = real_fc(phases, chi, *a, **kw)
            rec.update(ok=True, chi=float(r.chi), n_found=float(r.n_B))
            return r
        except (RuntimeError, ValueError):
            rec.update(ok=False, chi=float("nan"))
            raise
        finally:
            after = (_tot("residual"), len(L.CALLS), _evals(), _lm_evals(),
                     time.process_time())
            rec.update(resid=int(after[0] - before[0]), tfm=after[1] - before[1],
                       evals=after[2] - before[2], lm=after[3] - before[3],
                       cpu=after[4] - before[4])
            TRACE.append(rec)
            _stream(rec)

    B.solve_fixed_chi = traced_fc

    variants = set(os.environ.get("T11_VARIANT", "").split(","))
    if "nolm" in variants:
        # Ticket 17's candidate, applied from outside: the adapter's internal
        # NJL solve declines Levenberg-Marquardt. `thermo_from_mu` reaches
        # `solve_system` through its module global, so wrapping that global
        # and keying on the caller touches nothing else.
        import eos.njl.thermodynamics as NT
        real_ss = NT.solve_system

        def solve_system(*a, **kw):
            if sys._getframe(1).f_code.co_name == "thermo_from_mu":
                kw["methods"] = ("hybr",)
            return real_ss(*a, **kw)
        NT.solve_system = solve_system

    if "walkdown0" in variants:
        # The candidate, applied from outside: a DOWNWARD walk step gets no
        # retry ladder. Going down, a failed step IS the boundary (the walk's
        # own docstring), and the walk returns the same midpoint whatever the
        # ladder did, so the ladder can only change the answer by rescuing a
        # step -- which the traces say it never does.
        real_sweep = B.sweep

        def sweep(phases, grid, *a, **kw):
            if (sys._getframe(1).f_code.co_name == "walk_to_crossing"
                    and float(grid[-1]) < float(grid[0])):
                kw["max_bisect"] = 0
            return real_sweep(phases, grid, *a, **kw)
        B.sweep = sweep


def _line(i, r):
    extra = (f"  -> n {r['n_found']:.6f}" if "n_found" in r else "")
    return (f"  {i:3d} {r['site']:16s} {r['depth']:1d} {r['n_B']:9.6f} "
            f"{'Y' if r['ok'] else '-':>3s} {r['chi']:+8.4f} {r['resid']:5d} "
            f"{r['tfm']:5d} {r['evals']:6d} {r['lm']:5d} {r['cpu']:7.1f}{extra}")


def _stream(rec):
    """One line per solve as it lands, so a killed run still leaves its trace."""
    print("  solve" + _line(len(TRACE) - 1, rec), flush=True)


def report(title):
    print(f"\n##### ticket 11 trace: {title} -- {len(TRACE)} mixed solves")
    print(f"  {'#':>3s} {'site':16s} {'d':>1s} {'n_B':>9s} {'ok':>3s} "
          f"{'chi':>8s} {'resid':>5s} {'tfm':>5s} {'evals':>6s} {'lm':>5s} "
          f"{'cpu s':>7s}")
    for i, r in enumerate(TRACE):
        print(_line(i, r))
    agg = {}
    for r in TRACE:
        k = (r["site"], r["depth"] > 0, r["ok"])
        a = agg.setdefault(k, [0, 0, 0, 0, 0, 0.0])
        a[0] += 1
        a[1] += r["resid"]
        a[2] += r["tfm"]
        a[3] += r["evals"]
        a[4] += r["lm"]
        a[5] += r["cpu"]
    print(f"  {'site':16s} {'retry':>5s} {'ok':>3s} {'solves':>6s} "
          f"{'resid':>6s} {'tfm':>6s} {'evals':>7s} {'lm':>6s} {'cpu s':>8s}")
    for (site, retry, ok), a in sorted(agg.items()):
        print(f"  {site:16s} {'>=1' if retry else '0':>5s} "
              f"{'Y' if ok else '-':>3s} {a[0]:6d} {a[1]:6d} {a[2]:6d} "
              f"{a[3]:7d} {a[4]:6d} {a[5]:8.1f}")
    tot = [sum(a[i] for a in agg.values()) for i in range(6)]
    print(f"  {'TOTAL':16s} {'':>5s} {'':>3s} {tot[0]:6d} {tot[1]:6d} "
          f"{tot[2]:6d} {tot[3]:7d} {tot[4]:6d} {tot[5]:8.1f}")


def run_locator(hint, refine="exact"):
    from eos.mixed.charges import beta_eq_neutrinoless
    phases = L.make_phases()
    CM.reset()
    t0, c0 = time.perf_counter(), time.process_time()
    w = B.locate_window(phases, CM.TARGET_GRID, CM.BENCH_ETA,
                        beta_eq_neutrinoless(), T=CM.BENCH_T, refine=refine,
                        hint=hint)
    print(f"\nlocate_window hint={hint} refine={refine!r}: "
          f"window {w.n_onset!r} -> {w.n_offset!r} exists={w.exists} "
          f"{len(w.probes)} probes   wall {time.perf_counter() - t0:.1f} s  "
          f"cpu {time.process_time() - c0:.1f} s", flush=True)
    return w


RUNS = {
    "hinted": lambda: run_locator((0.60, 1.55)),
    "unhinted": lambda: run_locator(None),
    "bisect-unhinted": lambda: run_locator(None, refine="bisect"),
    "onset": CM.run_onset,          # NB_START=0.855 gives ticket 04's 0.859
    "warm": CM.run_warm_sweep,
}

if __name__ == "__main__":
    install()
    CM.make_phases = L.make_phases
    print(f"ticket 11/17 trace, variant "
          f"{os.environ.get('T11_VARIANT', 'HEAD')}\n{CM.stack_line()}",
          flush=True)
    CM.warmup()
    for name in sys.argv[1:]:
        TRACE.clear()
        L.CALLS.clear()
        L._call_id[0] = 0
        RUNS[name]()
        CM.report(name)
        L.report(name)
        report(name)

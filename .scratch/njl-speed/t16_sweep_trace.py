"""Every candidate of the reference-backend sweep, 0.5 -> 0.664 fm^-3."""
import sys, numpy as np
from eos import njl
import eos.njl.solver as ns
par = njl.Parameters.named("rg_njl1"); flags = njl.SpeciesFlags(csc=True)
log = []
real_sp = ns.solve_pattern
def spy(*a, **kw):
    p = real_sp(*a, **kw)
    log.append((a[5], "none" if kw.get("x0") is None else ("CROSS" if kw.get("cross_seeded", "?") is True else ("own" if kw.get("cross_seeded", "?") is False else "?")),
                p.converged, p.pattern_realised, float(p.f)))
    return p
ns.solve_pattern = spy
x0 = None
for n_B in np.linspace(0.5, 1.55, 200)[:33]:
    log.clear()
    w = ns.solve(par, "beta_eq_neutrinoless", float(n_B), 0.0, flags, x0=x0, backend="reference")
    x0 = ns.warm_start(w)
    cands = "  ".join(f"{pat}[{src},{'ok' if c else 'NO'},{r},{f:.4f}]" for pat, src, c, r, f in log)
    print(f"{sys.argv[1]} {n_B:.4f} win={w.pattern_realised:3s} seeds={sorted(x0)}  {cands}", flush=True)

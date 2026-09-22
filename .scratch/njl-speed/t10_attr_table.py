"""Ticket 03's attribution, per arm, from a t10_tables.py `attr` run.

    python3 t10_attr_table.py ATTR.json
"""
import json
import sys

d = json.load(open(sys.argv[1]))
A = d["attribution"]
arms = list(A)
names = ["pair_pass", "pair_pass[vac]", "hess_pass", "hess_pass[vac]",
         "gapless_momenta", "unpaired_ref", "unpaired_ref[vac]",
         "unpaired_ref_hess", "unpaired_ref_hess[vac]", "panel_nodes",
         "panel_nodes[vac]", "crossing_terms", "modes_thermo", "modes_jacobian"]
VAC = [n for n in names if n.endswith("[vac]")]
print(f"{d['pattern']} {d['backend']!r}  {d['stack']}")
print(f"{'% of table':24s}" + "".join(f"{a:>9s}" for a in arms))
for n in names:
    print(f"{n:24s}" + "".join(f"{A[a]['self'].get(n, 0) / A[a]['total']:9.1%}"
                               for a in arms))
for g in ("jitted", "numpy", "python"):
    print(f"{g:24s}" + "".join(f"{A[a][g] / A[a]['total']:9.1%}" for a in arms))
print(f"{'vacuum half, all of it':24s}" + "".join(
    f"{sum(A[a]['self'].get(n, 0) for n in VAC) / A[a]['total']:9.1%}"
    for a in arms))
res = {a: d["results"][a][0] for a in arms}
print(f"{'cpu ms/pt, table':24s}" + "".join(f"{res[a]['cpu'] * 5:9.1f}" for a in arms))
print(f"{'cpu ms/pt, converged':24s}" + "".join(
    f"{res[a]['conv_cpu'] * 1e3 / res[a]['n_conv']:9.1f}" for a in arms))
print(f"{'converged / solve_at':24s}" + "".join(
    f"{res[a]['n_conv']:5d}/{res[a]['n_solve_at']:<3d}" for a in arms))
for label, key in (("hot pass calls", "pair_pass"), ("vac pass calls", "pair_pass[vac]"),
                   ("hot hessian calls", "hess_pass"), ("vac hessian calls", "hess_pass[vac]"),
                   ("residual evals", "modes_thermo"), ("jacobians", "modes_jacobian")):
    print(f"{label:24s}" + "".join(f"{A[a]['calls'].get(key, 0):9d}" for a in arms))
print(f"{'vac block hit rate':24s}" + "".join(
    f"{A[a]['vac_block_cache']['hits'] / max(1, A[a]['vac_block_cache']['hits'] + A[a]['vac_block_cache']['misses']):9.1%}"
    for a in arms))

"""PROTOTYPE (wayfinder ticket 13) -- throwaway. The map's correctness gate,
plus the two things ticket 13 names on top of it.

Judged on `pattern_realised`, not on `pattern`: ticket 05's census showed the
LAYOUT column is not stable even in the baseline, so the name that says what
the matter IS is the one to gate on.

    python3 .scratch/njl-speed/proto13_gate.py VD [baseline]
"""
import json
import sys

ONSET = 0.6583          # fm^-3, the 2SC -> CFL switch the map pins
CFL_ROWS = 170


def load(name):
    return json.load(open(f".scratch/njl-speed/proto13_{name}.json"))


def onset_of(rows):
    """First density whose realised state is CFL, with 2SC below it."""
    for r in sorted(rows, key=lambda r: r["n_B"]):
        if r["realised"] == "CFL":
            return r["n_B"]
    return None


def main():
    name = sys.argv[1]
    base_name = sys.argv[2] if len(sys.argv) > 2 else "V0"
    base, new = load(base_name), load(name)
    b = {round(r["n_B"], 6): r for r in base["winners"]}
    n = {round(r["n_B"], 6): r for r in new["winners"]}

    print(f"{name} vs {base_name}: {len(b)} vs {len(n)} rows")
    print(f"  wall {base['wall_s']:.1f}s -> {new['wall_s']:.1f}s   "
          f"cpu {base['cpu_s']:.1f}s -> {new['cpu_s']:.1f}s   "
          f"speedup {base['wall_s'] / new['wall_s']:.2f}x wall, "
          f"{base['cpu_s'] / new['cpu_s']:.2f}x cpu")

    for label, miss in (("MISSING", sorted(set(b) - set(n))),
                        ("EXTRA", sorted(set(n) - set(b)))):
        if miss:
            print(f"  {label} {len(miss)} densities: {miss[:8]}")

    bad, worst = [], (0.0, None)
    for k in sorted(set(b) & set(n)):
        if b[k]["realised"] != n[k]["realised"]:
            bad.append((k, b[k]["realised"], n[k]["realised"]))
        rel = abs(n[k]["P"] - b[k]["P"]) / max(abs(b[k]["P"]), 1e-30)
        if rel > worst[0]:
            worst = (rel, k)
    print(f"  realised-state mismatches: {len(bad)}")
    for k, x, y in bad[:12]:
        print(f"     n_B={k:.4f}  {base_name} {x} -> {name} {y}")
    ok_P = worst[0] <= 1e-8
    print(f"  worst |dP|/P = {worst[0]:.3e} at n_B={worst[1]}  "
          f"({'PASS' if ok_P else 'FAIL'} at 1e-8)")

    # ticket 13's two extra conditions
    on_b, on_n = onset_of(base["winners"]), onset_of(new["winners"])
    cfl_b = sum(1 for r in base["winners"] if r["realised"] == "CFL")
    cfl_n = sum(1 for r in new["winners"] if r["realised"] == "CFL")
    print(f"  2SC->CFL onset: {base_name} {on_b}  {name} {on_n}  "
          f"(map pins {ONSET}) "
          f"{'PASS' if on_n is not None and abs(on_n - ONSET) < 5e-4 else 'FAIL'}")
    print(f"  CFL rows: {base_name} {cfl_b}  {name} {cfl_n}  "
          f"(map pins {CFL_ROWS}) {'PASS' if cfl_n == CFL_ROWS else 'FAIL'}")

    nc_b = [k for k in b if not b[k].get("converged", True)]
    nc_n = [k for k in n if not n[k].get("converged", True)]
    print(f"  non-converged ROWS: {base_name} {len(nc_b)}  {name} {len(nc_n)}")
    if nc_n:
        print(f"     {sorted(nc_n)[:12]}")
    lay = sum(1 for k in set(b) & set(n) if b[k]["pattern"] != n[k]["pattern"])
    print(f"  layout-column (`pattern`) differences: {lay}  [not gated]")


if __name__ == "__main__":
    main()

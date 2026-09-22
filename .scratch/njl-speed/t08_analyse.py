"""Ticket 08: the table and the gate, from one or more t08.py drive records.

    python3 t08_analyse.py OUT_mismatches.json t08_rep0.json [t08_reps12.json ...]

Timing: per arm and case, the median cpu over every repeat in the records
given; the ratio is base/landed of those medians, with the per-repeat ratios
beside it (each repeat interleaves the arms, so a per-repeat ratio shares one
load window). Accuracy: the FIRST repeat's rows, landed against base, on
pattern_realised and |dP|/P <= 1e-8; every other repeat is checked
bit-identical to the first within its arm. Every row that fails the gate, or
that one arm solved and the other dropped, is written to OUT_mismatches.json
for t08_adjudicate.py (ticket 08: none).
"""
import json
import sys

import numpy as np

GATE = 1e-8


def load(paths):
    runs, cases, fps = [], None, []
    for k, path in enumerate(paths):
        rec = json.load(open(path))
        cases = rec["cases"]
        fps.append((path, rec.get("fingerprint_before"),
                    rec.get("fingerprint_after")))
        for r in rec["runs"]:
            r["rep"] = f"{k}.{r['rep']}"
            runs.append(r)
    return runs, cases, fps


def main(out, paths):
    runs, cases, fps = load(paths)
    for path, before, after in fps:
        print(f"{path}: fingerprint {'UNCHANGED' if before == after else 'CHANGED'}"
              f" {after}")
    mismatches = []
    print("\n| case | base cpu | landed cpu | ratio | rows solved (base/landed) "
          "| realised mismatches | worst |dP|/P | load |")
    print("|---|---|---|---|---|---|---|---|")
    for i, label in enumerate(cases):
        by = {"base": [r for r in runs if r["case"] == i and r["arm"] == "base"],
              "landed": [r for r in runs if r["case"] == i and r["arm"] == "landed"]}
        cpu = {a: float(np.median([r["cpu"] for r in by[a]])) for a in by}
        per = [b["cpu"] / l["cpu"] for b, l in zip(by["base"], by["landed"])]
        loads = [x for a in by for r in by[a] for x in r["load"]]
        for a in by:                            # determinism within an arm
            first = by[a][0]["rows"]
            for r in by[a][1:]:
                if r["rows"] != first:
                    print(f"  NOTE {label} {a}: repeat {r['rep']} rows differ "
                          f"from the first repeat")
        errs = {a: by[a][0]["error"] for a in by if by[a][0]["error"]}
        rb = {round(x["n_B"], 9): x for x in by["base"][0]["rows"]}
        rl = {round(x["n_B"], 9): x for x in by["landed"][0]["rows"]}
        shared = sorted(set(rb) & set(rl))
        bad_state, worst = 0, 0.0
        for n in shared:
            dP = abs(rl[n]["P"] - rb[n]["P"]) / max(abs(rb[n]["P"]), 1e-30)
            worst = max(worst, dP)
            state = rb[n]["pattern_realised"] != rl[n]["pattern_realised"]
            bad_state += state
            if state or dP > GATE:
                mismatches.append({"case": i, "n_B": n, "why": "state" if state
                                   else "dP", "dP": dP, "base": rb[n],
                                   "landed": rl[n]})
        for n in sorted(set(rb) ^ set(rl)):
            mismatches.append({"case": i, "n_B": n, "why": "dropped",
                               "base": rb.get(n), "landed": rl.get(n)})
        print(f"| {label} | {cpu['base']:.1f} s | {cpu['landed']:.1f} s | "
              f"**{cpu['base'] / cpu['landed']:.2f}x** "
              f"({', '.join(f'{x:.2f}' for x in per)}) | {len(rb)}/{len(rl)} | "
              f"{bad_state} | {worst:.1e} | {min(loads):.0f}-{max(loads):.0f} |"
              + (f" ERR {errs}" if errs else ""))
    allrun = [r for r in runs]
    print(f"\nwall/cpu over all runs: {min(r['cpu'] / r['wall'] for r in allrun):.2f}"
          f" .. {max(r['cpu'] / r['wall'] for r in allrun):.2f}; loadavg "
          f"{min(min(r['load']) for r in allrun):.1f} .. "
          f"{max(max(r['load']) for r in allrun):.1f}; repeats per arm/case "
          f"{len(allrun) // (2 * len(cases))}")
    print(f"\n{len(mismatches)} rows to adjudicate")
    for m in mismatches:
        b, l = m["base"], m["landed"]
        print(f"  case {m['case'] + 1} n_B {m['n_B']:.4f} {m['why']:7s} base "
              f"{b and b['pattern_realised']} landed {l and l['pattern_realised']}"
              + (f" dP/P {m['dP']:.1e}" if "dP" in m else ""))
    json.dump(mismatches, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

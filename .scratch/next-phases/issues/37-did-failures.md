# Three of the four suite failures are did — one cause or three?

Type: task
Status: resolved
Parent: ../map.md

## Question

The clean baseline ([ticket 01](01-pytest-baseline.md)) leaves four failures on
`main`, and three of them are `did`:

    FAILED test/baseline/test_baseline.py::test_baseline[did]
    FAILED test/did/test_did_tov.py::test_mass_radius_matches_the_paper[DID-flags0]
    FAILED test/did/test_did_tov.py::test_mass_radius_matches_the_paper[DIDY-flags1]

**The baseline half is already characterised.** A standalone re-run gives exactly
12 mismatches, every one a hyperon effective potential in `fixed_YC_YS` at
**Y_S = 0**:

    ycys.YS0.n0.32.mu_eff_i.Sigma+   rel=3.280e-07  abs=2.321e-05
    ycys.YS0.n0.32.mu_eff_i.Lambda   rel=1.663e-07  abs=2.321e-05
    ycys.YS0.n0.32.mu_eff_i.Xi-      rel=1.609e-07  abs=4.642e-05
    ...  (Sigma0, Sigma-, Xi0 at n0.32; the same six at n0.64, rel ~1e-10)

`eps`, `P`, `mu_B` and every density are untouched — only the potentials move.
That is precisely the cross-cutting entry already in `docs/DEFERRED.md`:
*"A potential is only pinned as tightly as its conjugate density responds."* At
Y_S = 0 no strange species is thermally populated, so the strangeness constraint
`n_S = n_B Y_S` holds for a whole range of `mu_S`, the Jacobian is singular in
that direction, and the solver reports whichever `mu_S` its path reached.

**But that ledger entry names `sfho` and `vmit`, not `did`.** So either `did`
belongs in it — a one-line ledger fix plus a decision about whether a baseline
can pin an underdetermined potential at all — or `did`'s case is something else
wearing the same signature.

**The M–R half is uncharacterised.** Both parameter sets fail
`test_mass_radius_matches_the_paper`. Whether that shares a cause with the
potentials (it should not — `eps` and `P` are unaffected, and those are what TOV
integrates) or is an independent disagreement with the paper is the open
question. Note the two known `did` paper traps: the Eq. (6) typo and
`tau_3 = 2 I_3`.

Resolve by deciding, for each of the three: fix the code, re-pin the baseline in
its own commit quoting the before/after delta (never regenerate everything), or
record in `docs/DEFERRED.md` with reasoning. §12 makes the frozen baseline ground
truth, so "the baseline is stale" needs an argument, not an assumption.

Related: [ticket 31](31-did-documents.md) rewrites `did`'s documents and will
need whatever this settles about the model's M–R predictions.

## Finding 1 — the baseline mismatch IS the ledger's underdetermined-mu_S case

Proven, not inferred. Two independent lines of evidence.

**The strange sector is empty at Y_S = 0.** Re-solving the four `ycys` cases:

| Y_S | n_B | mu_S [MeV] | n_S | sum n_strange |
|---|---|---|---|---|
| 0.0  | 0.32 | **−515.32** | 1.6e-16 | 3.4e-11 |
| 0.0  | 0.64 | **−532.46** | −5.8e-24 | 2.9e-10 |
| 0.05 | 0.32 | +70.55 | 1.60e-02 | 1.52e-02 |
| 0.05 | 0.64 | +19.00 | 3.20e-02 | 3.13e-02 |

At Y_S = 0 no strange species is thermally populated — densities are round-off,
1e-10 and below — so `n_S = n_B Y_S` is satisfied for a whole range of `mu_S`,
the Jacobian is singular in that direction, and the solver lands wherever its
path took it. At Y_S = 0.05 the same solver pins `mu_S` cleanly.

**Every mismatch is exactly S_i × one shared shift.** All 12 mismatched keys are
`ycys.YS0.*.mu_eff_i.<hyperon>`; `delta / |S_i|` is constant to six digits within
each density:

| species | S | delta at n=0.32 | delta/|S| | delta at n=0.64 | delta/|S| |
|---|---|---|---|---|---|
| Lambda, Sigma+, Sigma0, Sigma− | 1 | 2.320888e-05 | 2.320888e-05 | 4.11584e-08 | 4.11584e-08 |
| Xi−, Xi0 | 2 | 4.641776e-05 | 2.320888e-05 | 8.23169e-08 | 4.11584e-08 |

Which is `mu_eff_i = B_i mu_B + C_i mu_C + S_i mu_S + ...` read off directly: one
`Delta mu_S` of 2.320888e-05 MeV propagating into each hyperon in proportion to
its strangeness. **No nucleon is affected** (S = 0), and no density, `eps`, `P`
or `mu_B` moved — consistent with `mu_S` being a free direction that changes
nothing physical.

**Conclusion: `did` belongs in the `docs/DEFERRED.md` cross-cutting entry that
currently names only `sfho` and `vmit`.** Not a regression, not model drift, and
not something to fix in `did`'s solver. Two things follow, and both need a
ruling:

1. **The ledger is incomplete** — a one-line fix, adding `did` (and noting the
   entry applies to any model exposing `fixed_YC_YS` with hyperons, which is what
   its own "Models:" line already half-says).
2. **The baseline should not pin an underdetermined potential.** `mu_eff_i` for a
   hyperon at Y_S = 0 is not a reproducible quantity — its value is the path's,
   not the physics'. Pinning it at rtol = 1e-10 guarantees a red suite whenever
   the solver path shifts, which is exactly what happened. The map already
   carries this as an open design question; this ticket supplies the measurement
   it was waiting for.

Note the shift is ~500× larger at n_B = 0.32 than at 0.64, so the looseness is
not uniform — worth stating in whatever ledger entry results.

## Finding 2 — the two M–R failures are a missing data file, not physics

Both fail on **`R_1.4` only**; `M_max` passes for both parameter sets:

    DID:  R_1.4 = 11.35 vs published 11.99
    DIDY: R_1.4 = 11.35 vs published 11.99

Identical to 2 dp across both, which is right — hyperons do not populate at
1.4 M_sun. But 0.64 km short against a 0.25 km tolerance, where the test's own
docstring says the crust difference should move `R_1.4` by "around a tenth of a
km". That explanation never covered it.

**Cause: the BPS crust table is absent, and the test helper silently drops it.**
`test/did/did_tov_sequence.py` (and `test/dd2/dd2_tov_sequence.py` identically)
open with

    if crust == "BPS" and not have_crust("BPS"):
        crust = "No"

`have_crust("BPS")` returns `False` in this checkout: the search path is only
`<repo>/data/crust`, which does not exist, and `EOS_CRUST_DIR` is unset. The file
itself is on disk at `/Users/mircoguerrini/Desktop/Research/Crust/BPST0.dat`,
outside the repo. So both tests build a bare-core star — missing the outer
kilometre — and then assert against published values that include a crust.

**Proven by setting the variable.** With
`EOS_CRUST_DIR=/Users/mircoguerrini/Desktop/Research/Crust`:

    pytest test/did/test_did_tov.py test/dd2/test_dd2_m4_tov.py -q
    6 passed in 111.09s

So `did`'s M–R is **not** a defect, `did`'s published numbers reproduce, and this
finding also resolves [ticket 38](38-dd2-tov-radius.md) — dd2's 12.33 km against
13.2 was the same missing crust, which is why both models came out short in the
same direction.

`eos/astro/tov/crust.py` itself is exemplary and needs no change: `crust_path`
raises `MissingCrustData` naming the file, every directory tried and the
override variable — exactly the §10 standard. **The defect is the silent
downgrade in the two test helpers**, which converts a missing-data condition into
a physics failure 0.64–0.87 km deep. Graduated to
[ticket 39](39-crust-silent-fallback.md).

## Ruling

**Make `mu_S` determined at Y_S = 0** rather than excluding it from the baseline
or re-pinning `did`. Graduated to [ticket 40](40-determine-mu-s.md), which carries
the scope: `fixed_YC_YS` is exposed by 11 models, and `docs/DEFERRED.md`'s entry
already generalises to "any model exposing `fixed_YC_YS` or a charge-neutral mode
at Y_C = 0", so this is repo-wide.

Both of this ticket's own questions are answered and it closes:

- the baseline mismatch is the underdetermined-`mu_S` case, proven by
  `Delta mu_eff_i = S_i x Delta mu_S` holding to six digits with no nucleon and
  no physical quantity affected (Finding 1);
- the two M–R failures are the missing BPS crust table, proven by the tests
  passing with `EOS_CRUST_DIR` set (Finding 2), which also resolved
  [ticket 38](38-dd2-tov-radius.md) and raised
  [ticket 39](39-crust-silent-fallback.md).

`did` itself has no defect. Adding `did` to the DEFERRED entry is superseded by
ticket 40, which removes the underlying looseness instead of documenting it.

Status: resolved.

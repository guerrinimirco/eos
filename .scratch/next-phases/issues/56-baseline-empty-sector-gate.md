# The baseline's empty-sector gate is absolute where the physics is relative

Type: task
Status: open
Parent: ../map.md

## Question

[Ticket 40](40-determine-mu-s.md) taught the baseline generator to drop a
potential its residual never pinned: where the strange sector is empty, `mu_S`
and everything carrying it through `S_i` come out of the stored record. The
mechanism is right. **The gate that triggers it is not** — it tests an absolute
density against `1e-12`, and the sector's emptiness is a statement about
`n_S / n_B`, not about `n_S`.

`test/baseline/generate_baseline.py:181`:

    if n_S is not None and abs(n_S) < 1e-12:

**Measured, one affected row in the whole suite** (scan of every `.npz` for a
stored `mu_S` beside a near-zero `n_S`):

    test/baseline/sfho.npz   ycys.n0.16.matter
        n_B  = 0.16000000280432228
        n_S  = 2.492122e-09        ->  Y_S = 1.558e-08
        mu_S = 8.449600

`n_S` is **nine orders of magnitude above the gate**, so the exclusion never
fires, and `mu_S = 8.4496` is frozen at rtol = 1e-10 although nothing determined
it. Ticket 40's own paragraph describes this exact number; it simply lands at a
density where its gate does not recognise the case.

The fingerprint is ticket 40's mechanism, not round-off — one undetermined
`Delta mu_S` propagating linearly through `S_i`, with Xi at exactly twice
Lambda, across thirteen quantities (`mu_S`, `mu_i.{Lambda,Sigma+,Sigma0,Sigma-,
Xi0,Xi-}` and the six `mu_eff_i` twins, all stored in this record).

**The blast radius is exactly one row, and the second gate already does its
half of the job.** Verified across all three models that store a `mu_S`:

| model | stored `mu_S` rows | affected |
|---|---|---|
| `sfho` | — | **1** (`ycys.n0.16`, `mu_S = 8.4496` free) |
| `did` | 55 | **0** — 20 rows DO have a tiny `n_S`, but every one has `mu_S = 0` exactly |
| `dd2` | 12 | **0** — every `mu_S` is `0` exactly, and every `n_S` is large (6.2e-06 to 0.42) |

`did`'s twenty rows are the case the generator's comment at `:183-189` protects:
in beta equilibrium strangeness is not conserved, so `mu_S = 0` is **imposed**
rather than solved and the hyperon potentials are perfectly determined. The
second gate (`:190`, `abs(mu_S) > 1e-12`) declines to drop them, correctly.
**That structure survives the fix** — a relative first gate would fire on did's
rows too, and the second gate would still keep them, because their `mu_S` is
zero. So the fix cannot cost the "234 good numbers" the comment defends.

This also corrects a guess made while the finding was being passed around:
**`dd2` does not carry the same thing and could not**, its `mu_S` being imposed
in every stored record. Only `sfho` was ever exposed.

## What to decide

The one-line version is "test whether the sector is empty relative to `n_B`
rather than against an absolute `1e-12`", but the threshold is a real choice and
should be argued rather than picked:

- `Y_S < 1e-12`? Then sfho's row (1.6e-08) is still kept. Too tight to fix it.
- `Y_S < 1e-6`? Fires on sfho and on all twenty did rows; did's survive the
  second gate. This is the smallest threshold that closes the measured case with
  room, and it has a physical reading: below one strange baryon per million.
- Or drop the density test entirely and gate on the **residual's sensitivity** —
  the honest criterion, since what makes `mu_S` free is a singular Jacobian
  column, not a small number. More faithful, and more work.

`mu_e` at `Y_C = 0` is the same class (`:198`, the same absolute `1e-12` against
`n_e`) and should be ruled the same way in the same pass. It was not measured
here; measure it before changing it.

## Constraints

- **`test/` is gitignored** (`.gitignore:75`, per CLAUDE.md §11), so this fix
  lands **outside version control**, as tickets 39 and 40's did. Anyone
  reconstructing `test/` reintroduces the gap. That is the standing question the
  map records under "Several real fixes now live outside version control", and
  this ticket is a third instance of it, not a new problem.
- Regenerating `sfho.npz` is what applies the fix. §12 makes the baselines
  golden, so the regeneration must **drop keys and change nothing else**: every
  surviving key bit-identical at rtol = 1e-10. Diff the key sets before and
  after and report the count dropped — expected: the 13 listed above, from one
  record.
- **Do not loosen a tolerance** (§12). The fix removes an unpinnable number from
  the record; it does not widen the gate that pins the rest.
- Coordinate the suite run — `test/dd2/test_dd2_speed.py` goes flaky under CPU
  contention, so do not run a full suite concurrently with another session.

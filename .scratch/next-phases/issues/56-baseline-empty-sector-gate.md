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

## Measured: the lepton side and the strange side do NOT have the same shape

Both gates were then measured across all nine baselines — every `n_e`/`mu_e`
and `n_S`/`mu_S` pair, expressed as a fraction of that record's own `n_B`.

**The lepton side has a clean gap, so a relative gate works.** Sorting all 579
lepton rows by `Y_e = n_e/n_B`:

    empty cluster      0  ...  6.929e-11      (the largest "empty" row)
                       ---- gap: x81,220 ----
    populated cluster  5.627e-06  ...         (the smallest real electron gas)

Nothing lives in between. **The absolute gate at `n_e < 1e-12` sits inside the
empty cluster rather than inside the gap**, which is why it is flaky: six rows
straddle it, and three sit ABOVE it by less than a factor of 70 —
`vmit ycys.n0.8` (n_e = 1.168e-12, **1.17x the gate**),
`vmit yc.lep.YC0.n0.8` (3.012e-12) and `alphabag yc.lep.YC0.n0.45` (3.118e-11).
Those three keep `mu_e` today; a recomputation that moves `n_e` by a factor of
two drops it, and the baseline then fails on **"quantity no longer produced"**,
a key-set mismatch rather than a value change. That is vmit's current failure,
and its "2 quantities" is exactly the two vmit rows above. A gate placed in the
gap — `Y_e < 1e-8`, the log-midpoint — has four orders of margin on each side
and no row within reach of it.

**The strange side has no such gap, and this is the finding that matters.**
sfho's free row does not sit at the edge of a cluster; it is *bracketed* by
did's rows on both sides:

    Y_S = 4.538e-09   did   yc.nolep.YC0.5.n0.08     mu_S = 0  (imposed)
    Y_S = 5.544e-09   did   yc.lep.YC0.5.n0.16       mu_S = 0  (imposed)
    Y_S = 1.558e-08   sfho  ycys.n0.16               mu_S = 8.4496   <-- FREE
    Y_S = 2.222e-08   did   yc.lep.YC0.3.n0.08       mu_S = 0  (imposed)
    Y_S = 4.597e-08   did   beta.Y.T10.n0.04         mu_S = 0  (imposed)

**No threshold in `Y_S` separates the one free row from the imposed ones** —
they interleave. So the magnitude proxy that works for leptons cannot work for
strangeness, and two of the three thresholds this ticket originally proposed are
dead on arrival for the sector that motivated it.

What rescues it is that **the first gate does not need to discriminate.** The
second gate (`:190`, `abs(mu_S) > 1e-12`) already separates free from imposed
*exactly* — an imposed `mu_S` is identically `0.0`, never nearly zero — and it
is doing that job correctly today. The first gate's only job is to be permissive
enough to admit every empty-sector row for the second gate to judge. Read that
way, `Y_S < 1e-6` is fine: it fires on sfho and on most of did's rows, and every
one of did's survives.

## The root cause is upstream of both gates

sfho's three `fixed_YC_YS` rows all target `Y_S = 0`. Compare how tightly each
closed the strangeness row:

    n_B = 0.160    n_S = 2.4921e-09    <-- seven orders looser than its siblings
    n_B = 0.320    n_S = 3.9248e-16
    n_B = 0.640    n_S = 3.0107e-16

Same model, same mode, same target. **The n = 0.16 solve closed the strangeness
row seven orders less tightly than the other two**, which is why it alone landed
above a gate that caught its siblings correctly. The gate's units are wrong, but
what put this row on the wrong side of it is solver conditioning — the map's
open "Scaling the strangeness residual" question, which measured that a 1e-10
residual gate admits 0.079 MeV of `mu_S`.

So there are three fixes at three depths, and they are not alternatives:

1. **Tighten the solve** so n = 0.16 closes like n = 0.32 and n = 0.64. Then the
   existing gate catches it, nothing else changes, and the row stops being
   special. This is the root cause and the only one that fixes the physics
   rather than the bookkeeping.
2. **Put the gates in the gap**: `Y_e < 1e-8` on the lepton side, which removes
   the flakiness band entirely and is what vmit's failure actually needs;
   `Y_S < 1e-6` on the strange side, permissive by design, with the second gate
   doing the discrimination.
3. **Gate on the residual's sensitivity** rather than on any density — the
   honest criterion, since what makes a potential free is a singular Jacobian
   column and a small density is only a symptom. Correct, and the most work.

(2) is what makes the suite deterministic and is cheap. (1) is what makes the
question go away. They are worth doing in that order, and (1) may belong with
the strangeness-residual scaling rather than here.

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

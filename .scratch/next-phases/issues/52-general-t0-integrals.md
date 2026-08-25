# general/ owes a public T = 0 entry point, and one loop is unbounded

Type: task
Status: resolved
Assignee: mirco (session)
Blocked by: 11
Parent: ../map.md

## Question

Two rows in `eos/general/`, grouped because both sit under every model's numbers
and therefore share one golden-reference gate.

**1. dd2 re-derives the T = 0 Fermi gas (finding 24).**
`eos/dd2/thermodynamics.py:66-94` defines `number_density_t0`,
`scalar_density_t0`, `eps_kin_t0` and `P_kin_t0` — the exact T = 0 ideal Fermi
gas, duplicating `eos/general/fermi_integrals.py:220 _compute_exact_T0`. §7 is
absolute: "All Fermi and Bose integrals, **at T = 0 and finite T**, come from
`eos/general/`. No model implements its own."

[Ticket 11](11-conformance-triage.md) ruled the fix lands in `general/`, not in
`dd2`, and the reason matters: **`general/fermi_integrals.py` exports no public
T = 0 entry point.** Everything at `:220` is private; the public names
(`:426 solve_fermi_jel`, `:453 solve_fermi_gl`, `:546 Fermi_Numerical`,
`:724 kinetic_thermo`) are finite-T. dd2 did not ignore the rule, it found no
door. So: promote the T = 0 closed forms to a public name, then dd2 imports them
and its four functions go.

The formulas are already algebraically identical and the audit found no numeric
discrepancy, so this must be a **no-op on every number**. §12 checks:
the **DD2 golden SNM point at n_B = 0.16 fm^-3**, the **DD2 published NMP/TOV
values**, and `test/baseline/dd2` at rtol = 1e-10. Movement beyond round-off
means the promotion is wrong and gets reverted, not accommodated.

While promoting, decide whether the new public name serves only dd2's four call
shapes or the general T = 0 case — §7 says the integral implementations may be
improved and alternatives added alongside, "each validated against JEL", and
**JEL is never removed**.

**2. The one unbounded loop (finding 20).**

    eos/general/fermi_integrals.py:519   while n_hi < n_target:   (mu_hi *= 1.5)
    eos/general/fermi_integrals.py:524   while n_lo > n_target:

No iteration counter, no cap. §6: "every solver has a bounded iteration count."
Geometric growth makes a hang practically unreachable, which is why this is a
ledger line as well as a fix — but every other loop in the repository is already
bounded (`grep -rn "while True" eos/` returns nothing; `general/tabulate.py:98`
caps at `T_cap=400.0`; `general/thermodynamics_leptons.py:343-345` raises past
`mu_max`; `general/solve.py:72-83` allows three attempts "because a parameter
scan must always get an answer back"). Bound it in the same style, and make
exhaustion a **returned status**, not a raise (§6).

Report added failures against `output/_audit/pytest_before_with_crust.txt`.

## Answer

**Half shipped, half measured and reverted, and the measurement is the finding.**
`ffae9db`, two files, `eos/general/fermi_integrals.py` and one docstring
reference in `eos/sfho/backends/jacobian.py`. Gate on **anaconda 3.9.7 / numpy
1.26.4 / scipy 1.13.1** (the stack `test/baseline/*.npz` was frozen on — ticket
57's canonical 3.14 fails 12 baseline tests for BLAS reasons, so a like-for-like
before/after there would have been read through a known-red control), in an
**isolated copy beside a control copy of the same tree**: the live checkout had a
concurrent session editing `dd2/api.py` between the before- and after-runs, which
turned a `test_photons_flag` failure that is not this ticket's into one that
looked like it. **A/B in the pair: 137 passed in both.**

### 1. The T = 0 door exists — `solve_fermi_t0`

`_compute_exact_T0` is promoted, not wrapped: same jitted body, same
`(n, P, e, s, ns)` in the same fm units as `solve_fermi_jel` and
`solve_fermi_gl`, so it reads as the third solver of the family and JEL is
untouched (§7). The docstring writes out all four closed forms and `s = 0`. Its
three existing callers — JEL's `T < 1e-4` branch, `solve_fermi_gl`'s, and sfho's
Jacobian comment — are a rename apart, so **no number in the repository moves**:
`test/general` plus the dd2, zl, vmit and enjl baselines at `rtol = 1e-10` are
identical in both copies.

**Scope, the question the ticket asked to decide: the general T = 0 case, not
dd2's four call shapes.** dd2's shapes are a strict subset of one call
(`n`, `P`, `eps`, `n_s` at one potential), a mask over the general signature
would have been a second door, and the existing private function already WAS the
general case — promoting it costs nothing and inventing anything else would have
been a fourth implementation of the same four formulas.

### 2. The loop is bounded, and exhaustion returns NaN

`invert_fermi_density`'s two bracket loops carry a step counter capped at
`_MAX_BRACKET_STEPS = 200`, and falling out of either without a bracket
**returns NaN** — a status the caller's residual carries out to the public
boundary, not a raise (§6). Verified against the HEAD control on every case that
reaches the loops: `n = 0` at T = 10 (5.21249710061511e-14 both), `n = 1e300`
(NaN both), `n = inf` (NaN both), `n = nan` (NaN both), and the ordinary
`n = 0.16` at T = 0 and T = 10, bit-identical. New test at
`test/general/test_fermi_integrals.py`, 4 passed on both interpreters.

**Two corrections to what the ledger says about this loop.** `docs/DEFERRED.md`
row 4 records that "a hang needs a non-finite `n_target`" — measured, **not even
that hangs**: `n_hi` overflows to `inf` and the comparison goes False, so the
worst case in double precision is a spin of order 1700 iterations, never an
infinite loop. And the UPPER bracket cannot be exhausted by a finite target at
all: it opens at twice the T = 0 estimate, which overshoots by a factor of eight
in n. The lower bracket is the one a target can outrun (`n = 0` at finite T
shrinks `mu_lo` for hundreds of steps), so that is the half the test exercises.

### 3. dd2 keeps its four functions, and that is now [ticket 67](67-dd2-t0-adoption.md)

The adoption was implemented and run, and **it is not a no-op on every number**:
of 4692 stored dd2 quantities 3434 are bit-identical and every EoS quantity moves
at most 5.9e-15 relative, but `nmp.K_sat` moves 1.9e-8, `nmp.K_sym` 2.2e-8 and
`nmp.Q_sat` 3.6e-4 (0.061 MeV) — the finite-difference NMP map amplifying
last-bit noise by 1e6 to 1e8, one factor of ~1/h per derivative order. Two added
failures, both there: `test_baseline[dd2]` on those three entries, and
`test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion`, whose assertion
is a solver verdict on a knife-edge target. The golden SNM(0.16) point (1.40e-05)
and CompOSE HS(DD2) (2.83e-05) do not move at all.

This ticket's rule was written for exactly this and it is honoured: **reverted,
not accommodated** — no tolerance touched, no `.npz` regenerated (ticket 62's
work, not this one's). What the measurement changes is the SHAPE of the remaining
half: no reformulation makes it bit-exact, so it is not a task any more but a
ruling — adopt and re-freeze three FD curvatures, or defer finding 24 with the
reason recorded. [Ticket 67](67-dd2-t0-adoption.md) carries it, blocked by
[62](62-regenerate-baselines-py314.md), with the diff and every number above.

**Added failures against `output/_audit/pytest_before_with_crust.txt`: none.**
The one failure seen in the live checkout,
`test/dd2/test_photons_flag.py::test_table_honours_the_flag` (`TypeError: leptons
is a flag, not a condition`), reproduces in the HEAD control copy and belongs to
the concurrent [ticket 54](54-signature-corrections.md) session — it is that
ticket's `leptons` fix landing in `dd2/api.py` ahead of the test that passes
`leptons` inside `fixed=`. Not fixed here, and not this ticket's to fix.

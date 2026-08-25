# Four missing verify/ invariants, and who owes the delivery gate

Type: task
Status: open
Blocked by: 11
Parent: ../map.md

## Question

[Ticket 11](11-conformance-triage.md) ruled these closed now rather than in
Phase 5: they are pure additions, no number moves, and two of them are real holes
in the models most likely to hide a defect.

1. **dd2's verify suite checks neither the free energy nor the rearrangement
   placement — and dd2 carries Σ^R** (finding 25).
   `eos/dd2/verify/run_full_check.py:134-156` registers golden points, thermo
   identities, responses, coeff analytic~FD, backend parity and CompOSE. Euler
   appears only inside `_check_identities` (`:70-78`, via `p.euler_residual()`).
   There is no `f = eps - T s` check and **no rearrangement check**, though
   `dd2/thermodynamics.py:492` returns `Sigma_R` and `dd2/solver.py:174,398,666`
   consume it. §8 makes "Σ^R enters mu and P, never eps" an invariant, and it is
   exactly the invariant that catches a wrong density-dependent RMF.
   `eos/ccdm/verify/run_full_check.py:263-297` is the model implementation — it
   asserts both the identity and that the term is non-trivial. Copy it.

2. **`eos/mixed`'s verify suite has the same gap** (finding 26).
   `mixed/verify/run_full_check.py` has Euler/HVH (`:106`), causality and
   monotonicity (`:187-194`) and backend parity (`:173`), no free-energy identity.
   It couples DD2, which carries Σ^R. Downstream of item 1.

3. **`eos/ccdm`'s verify suite has no causality or monotonicity check at all**
   (finding 27). `grep -n "cs2\|sound\|monoton"
   eos/ccdm/verify/run_full_check.py` returns nothing. Every other model checks
   `0 <= cs^2 <= 1` somewhere. CCDM is a colour-superconducting model where a
   wrong gap contribution shows up in the sound speed first — and it sits next to
   the best rearrangement check in the repository.

4. **`eos/njl`'s causality check is disabled by default** (finding 28).
   `njl/verify/run_full_check.py:545 check_sound_speed` exists but `:602 if
   include_sound` gates it on the `--sound` CLI flag, so `run_all()` with no
   arguments runs no causality check. An invariant that does not run by default
   is not one. If cost is the reason, turn it on anyway and mark it `slow` using
   `pyproject.toml`'s existing marker convention, and say so in the ledger.

5. **dd2 and did adopt the delivery gate** (finding 29, the (a) half).
   §8 scopes P-monotonicity to tables "DELIVERED to a structure solver", and it
   runs in only 3 of 12 suites (`sfho:234`, `enjl:889`, `mixed:191`). Eight
   models never build a table, so their absence is correct by the letter — but
   `eos/dd2/table.py:316 build_core_table` and `eos/did/table.py:203` do build
   one and do not check it. `eos/enjl/verify/run_full_check.py:866
   check_delivered_table` is the model implementation.
   The (b) half — §8 naming the delivery gate as belonging to whoever builds a
   table — is [ticket 22](22-phase5-claudemd.md)'s, and this ticket assumes that
   ruling rather than waiting on it.

Resolved when all five run in their suites by default and pass, and the
added-failure count against `output/_audit/pytest_before_with_crust.txt` is
reported. **No golden reference is at risk** — nothing here changes a computed
quantity. If one of the new checks *fails* on shipped physics, that is a finding:
report it, do not weaken the check and do not loosen a tolerance (§12).

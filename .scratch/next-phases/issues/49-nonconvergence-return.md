# Non-convergence escapes as an exception at seven public boundaries

Type: task
Status: open
Blocked by: 11
Parent: ../map.md

## Question

CLAUDE.md §6 calls this non-negotiable: "**NON-CONVERGENCE IS A RETURN VALUE** at
every public boundary, not an exception and never a hang." Seven live sites break
it, all inside the three §5 entry points.

**The `SnB=` path (finding 18).** `eos/general/tabulate.py:102` raises when an
entropy target is unreachable. `zl/api.py:113`, `vmit/api.py:113`,
`alphabag/api.py:119`, `mixed/api.py:160` and `dd2/api.py:97` all call
`temperature_at_entropy` **inside** their `try`; `eos/njl/api.py:109` and
`eos/ccdm/api.py:122` call it bare. So an unreachable isentrope escapes
`eos_point` as a `RuntimeError`. Two-line fix each.

**`eos_response` (finding 19).**

    eos/zl/api.py:199          raise RuntimeError("eos_response could not solve its stencil point ...")
    eos/vmit/api.py:199        (same)
    eos/alphabag/api.py:207    (same)
    eos/mixed/api.py:345       central point
    eos/mixed/api.py:374       stencil points
    eos/abpr/api.py:234        raise RuntimeError(f"eos_response could not invert n_B = {n_B}")

`abpr` is the starkest: `mu_from_nB` returns `(mu, converged)` and the status is
converted straight into an exception. `sfho`, `dd2`, `did`, `njl` and `ccdm`
already return dicts, so the compliant shape exists in-repo to copy.

**The return shape, ruled by [ticket 11](11-conformance-triage.md):** the SAME
dict shape the converged path returns, with `converged=False` and **NaN in every
quantity** — not a minimal `{converged: False, reason: ...}`. A caller writing
`result["cs2_adiabatic"]` into an array column must not need a second code path
for the failure case, and NaN propagates to a plot honestly. Carry a `reason`
string alongside; do not drop the message the exception used to carry.

Resolved when all seven return rather than raise, a check exists that a
deliberately unconvergeable point comes back with `converged=False`, and the
added-failure count against `output/_audit/pytest_before_with_crust.txt` is
reported. **No converged number may move** — this touches failure paths only.

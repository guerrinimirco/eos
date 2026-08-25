# dd2 raises a bare KeyError when hyperons are asked of a nucleonic set

Type: task
Status: resolved
Assignee: session e5e1c4c9 (also the other of 60/61)
Blocked by: 44
Parent: ../map.md

## Question

Found by [ticket 20](20-phase5-api-readme.md) while executing its README blocks:
the first draft of the quick-start example was

```python
par   = eos.dd2.Parameters.default()          # the nucleonic DD2 set
flags = eos.dd2.SpeciesFlags(hyperons=True)
eos.dd2.eos_point(par, "beta_eq_neutrinoless", flags, n_B=0.32, T=10.0)
```

and it dies at `eos/dd2/thermodynamics.py:289`, inside `build_baryon_specs`,
with

```
KeyError: 'Lambda'
```

— `Parameters.default()` carries `hyperon_couplings=()`, so the lookup
`hyp[b.name]` misses on the first hyperon in the loop.

The refusal is correct physics: DD2 and DD2Y are different published
parameterisations, not one set read through two flag settings, and asking for
the octet from the nucleonic couplings has no answer. What is wrong is HOW it
refuses. CLAUDE.md §4: "Setting a flag a model does not implement RAISES; a
`NotImplementedError` is never turned into a silent no-op" — and §6 makes the
public boundary the place where a malformed call is named. A bare `KeyError`
from three layers down names neither the flag nor the parameter set, and the
caller cannot tell it from an internal bug.

Two things to settle while fixing:

- **Where the check goes.** `SpeciesFlags.__post_init__` cannot see `par`, so
  this is not the shape the other nine models use — it is a
  (parameters, flags) compatibility check. `eos_point`/`eos_table` see both;
  so does `build_baryon_specs`, which is where the information already is.
- **Whether the same hole exists for `deltas`.** `x_Delta_sigma` and friends
  default to 1.0 rather than being absent, so the Delta path probably fails
  differently or not at all — measure rather than assume.

The message should name the flag, the parameter set and the set that does
carry the couplings, i.e. `Parameters.named("DD2Y")`.

Changes no converged number: every path that reaches a converged point today
carries the couplings. Gate on `test/dd2/` plus the `dd2` baseline.


## Resolution

**Fixed, in one place, and both "things to settle" are settled by measurement.**

### Where the check goes: `build_baryon_specs`, and it is the only funnel

The ticket offered three candidate homes and named `build_baryon_specs` as
where the information already is. That is also the only place it *needs* to be:
`build_baryon_specs` has exactly **one** caller in the whole repository,
`build_matter_ctx` at `eos/dd2/thermodynamics.py:339`, and every mode, every
table sweep and every `eos/mixed` hadronic adapter reaches the hyperon
couplings through it. Grepped, not assumed. So the guard is a single membership
test at the lookup that used to raise, and there is no sibling caller left
holding the old behaviour:

```python
else:
    if b.name not in hyp:
        raise NotImplementedError(...)
    mass, xs, xw, xr, xphi = hyp[b.name]
```

It reads `not in hyp` rather than `not par.hyperon_couplings`, so a **partial**
coupling map is caught the same way an empty one is, and the message names the
baryon that is actually missing.

### The exception type is NOT ValueError, and that is the whole fix

This is the part that would have silently defeated a copy of `sfho`'s guard.
`eos/dd2/api.py:106-108` reads

```python
except NotImplementedError:
    raise                       # an unwired request must never be a status
except (RuntimeError, ValueError) as err:
    return PointResult(False, str(err))
```

so a `ValueError` raised anywhere under `eos_point` comes back as a
**non-converged PointResult**, not as an error the caller sees. Measured: with
the guard first written as `ValueError`, `eos_point(Parameters.default(),
"beta_eq_neutrinoless", SpeciesFlags(hyperons=True), ...)` returned
`ok=False` instead of raising — a malformed call reported as a failed solve,
which is worse than the bare `KeyError` the ticket opened on, because a sampler
would score it and move on. `NotImplementedError` is what §4 names and what
`api.py`'s own comment reserves for exactly this. The choice is recorded as a
comment at the raise so nobody "harmonises" it back to `ValueError`.

Note the asymmetry with `sfho`, which raises `ValueError` from
`species.check_couplings` and gets away with it only because `sfho/solver.py`
calls it OUTSIDE `api.py`'s try. Same defect one refactor away; recorded for the
Stage 7 report, not changed here.

### `deltas` has no such hole — measured, not assumed

`x_Delta_sigma`, `x_Delta_omega` and `x_Delta_rho` all default to **1.0** on
`Parameters`, which is the usual "the Delta couples like a nucleon" choice
rather than a stand-in for absent data, so there is nothing to look up and
nothing to miss. `eos_point(Parameters.default(), "beta_eq_neutrinoless",
SpeciesFlags(deltas=True), n_B=0.32, T=10.0)` **converges**. No guard added:
adding one would refuse a configuration the model can genuinely compute.

### The message

```
SpeciesFlags(hyperons=True) with a parameter set that carries no couplings for
Lambda (par.hyperon_couplings is empty or partial). DD2 and DD2Y are different
published parameterisations, not one set read through two flag settings, so the
octet has no answer from the nucleonic couplings. Use Parameters.named('DD2Y').
```

It names the flag, the missing species, the parameter set's own field, the
physics reason, and the set that does carry the couplings — the four things the
ticket asked for.

### Numbers

Unchanged, as the ticket predicted: the guard fires only where `hyp[b.name]`
raised `KeyError` before, so no path that reached a converged point takes it.
Confirmed by `Parameters.named("DD2Y") + hyperons=True` still converging, by
`eos/dd2/verify/run_full_check.py` **PASS** (golden SNM point 1.40e-05, HVH
1.50e-15, backend parity 8.88e-16, CompOSE HS(DD2) 2.83e-05) and by
`eos/mixed/verify/run_full_check.py` **PASS** (TOV M_max = 2.340, R = 12.64 km,
unchanged).

### The gate

**Interpreter: python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0**
([ticket 57](57-canonical-stack.md)'s canonical stack).

`test/dd2`: **3 failed, 203 passed, 206 collected** —
`test_inversion_without_Q_sat_predicts_it`,
`test_inversion_with_Q_sat_still_imposes_it` and
`test_restarts_recover_a_seed_limited_inversion`, the three
[ticket 47](47-dd2-nmp-inversion.md) NMP-inversion failures already in
`output/_audit/pytest_after_ticket48_py314.txt`; `diff` of the sorted node-id
lists EMPTY. **0 added, 0 cleared.**

`test/baseline`: **6 failed, 10 passed**, the six node ids identical to the
before-image, `diff` EMPTY — so the `dd2` baseline row is unmoved, which is
this ticket's stated gate. Recorded as
`output/_audit/pytest_after_ticket61_baseline_py314.txt`.

`eos/dd2/verify/run_full_check.py` **PASS**, every check at its previous
tolerance.

Shares its gate with [ticket 61](61-dd2-species-flags.md), which was worked in
the same session and the same files; the full evidence, including the matched
HEAD-control pair and why an isolated copy was needed, is written up there.

# dd2 raises a bare KeyError when hyperons are asked of a nucleonic set

Type: task
Status: open
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

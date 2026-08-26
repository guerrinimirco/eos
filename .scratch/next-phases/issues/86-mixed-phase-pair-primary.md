# Make the Phase pair the parameter argument in `eos/mixed`

Type: task
Status: open
Blocked by: -   (29 and 84 both resolved)
Parent: ../map.md

## Question

Execution of [ticket 84](84-vmit-params-in-the-plumbing.md)'s ruling. Read it
first: the design is settled and the user's instruction is that **`dd2` and
`vmit` must not be preferred or special with respect to the other models**.

**[Ticket 29](29-mixed-species-flags.md) is DONE** (`8bb546c`) and was the
first half: `eos/mixed/species.py` exists, `api.py` no longer imports
`eos.dd2.species`, the `species` argument is no longer a DD2 type, and the
`muons=None` kwarg is gone from all four signatures — the chain that carried
it now threads the flag object as `species=`.

Two things 29 left for this ticket to finish. The front door's DEFAULT flags
moved out of `api.py` into **`adapters.default_flags()`** — delete it here
along with the rest of the front door. And `_engine_fractions`' sibling check
in `hybrid_table`, the trapped-mode `species.neutrinos` guard, is now narrowed
to `phases is None`, because for a `Phase` pair the adapter's own
`_dd2_wing_kwargs` already raises; when the front door goes, that guard goes
with it.

`eos/mixed/__init__.py:51` was deliberately NOT touched by 29 — the re-export
is this ticket's.

### The inversion

`phases=(Phase, Phase)` becomes **the** parameter argument on all four public
entry points — `eos_point`, `eos_table`, `hybrid_table`, `eos_response`.
`(par, flags, vmit_params)` **retires entirely**, not as a compatibility
overload: a callable that wants DD2+vMIT writes
`phases=default_pair(par, flags, vmit_params)`, and `default_pair` stays in
`adapters.py`. The convenience survives; the privileged position does not.

**The gate is a grep.** `vmit_params` must return **zero** hits in `eos/`. It is
**264** today: `solver.py` 16, `boundaries.py` 13, `table.py` 11,
`responses.py` 9, `hybrid.py` 8, `api.py` 4, `scan.py` 3 (scan goes to
[ticket 87](87-remove-mixed-scan.md)), plus the rest.

### The five module-level model imports that must go

`adapters.py` importing both models is CORRECT — that is what an adapter layer
is. These are not adapters:

    api.py:49        from eos.dd2.species import SpeciesFlags      -> ticket 29
    charges.py:56    from eos.dd2.species import hadronic_qn, hadronic_charges
    responses.py:68  from eos.dd2.solver import warm_start
    scan.py:73,76,77 dd2.nmp, dd2.solver.sweep, vmit.parameters    -> ticket 87
    __init__.py:51   from eos.dd2 import Parameters, SpeciesFlags

**`hadronic_qn` / `hadronic_charges` move to `general/basis.py`.** They read only
`general/particles` and a flags object with `hyperons`/`deltas` — nothing
DD2-specific — so this is §7's single-home rule, NOT a §2 duplication:
`eos/dd2/species.py:23` imports the shared `Particle` objects and re-declares
nothing. There is no duplicate to delete, only a file to move. Check whether
`general/basis.charges_from_densities` then does the same job; if it does, the
merged one wins.

**`responses.py`'s DD2 warm start needs no new contract.** The `Phase` surface
ALREADY provides it: `thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None,
return_state=False)` returns `(block, state)` where "`state` is an opaque
internal vector", and the class docstring says outright that "the engine never
sees a model type". `responses.py:68` is a shortcut around a surface that has
been there all along. Note that `:208` also reaches into `flags.phi_field and
flags.hyperons` — DD2-specific FIELDS, not just a DD2 function.

### Gate

- `grep -rn vmit_params eos/` returns nothing.
- No module outside `eos/mixed/adapters.py` imports `eos.dd2` or `eos.vmit`.
- `test/baseline/` for `mixed` unmoved at rtol = 1e-10 — this changes how the
  engine is CALLED, not what it computes. If a number moves, STOP AND REPORT.
- `eos/mixed/verify/run_full_check.py` all `[ok ]`.
- A pairing that is neither DD2 nor vMIT — `sfho_phase` with `njl_phase` — runs
  end to end through the new primary signature. That is the whole point, so it
  is part of the gate rather than a nice-to-have.

CLAUDE.md §5's front-door clause is retired by
[ticket 85](85-claudemd-sentences-owed.md); do not edit CLAUDE.md here.

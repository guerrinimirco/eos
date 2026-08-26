# Make the Phase pair the parameter argument in `eos/mixed`

Type: task
Status: open
Blocked by: 29, 84
Parent: ../map.md

## Question

Execution of [ticket 84](84-vmit-params-in-the-plumbing.md)'s ruling. Read it
first: the design is settled and the user's instruction is that **`dd2` and
`vmit` must not be preferred or special with respect to the other models**.

**Blocked by [ticket 29](29-mixed-species-flags.md)**, which is the first half:
`mixed` borrows DD2's flag class (`api.py:49 from eos.dd2.species import
SpeciesFlags`), and 29's own `species.py` is what lets the `species` argument
stop being a DD2 type. 29 also removes the `muons=None` kwarg sitting beside
`vmit_params` in the same four signatures.

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

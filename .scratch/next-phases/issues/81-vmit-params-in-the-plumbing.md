# `dd2` and `vmit` are structurally privileged inside `eos/mixed`

Type: grilling
Status: in progress
Assignee: session 9a857509
Blocked by: -
Parent: ../map.md

## Question

Raised by the user, twice and sharpening each time: *"In hybrid_table an input
is vmit_params? why? It should work for all the models"* — then, decisively,
*"dd2 and vmit should not be 'preferred' or 'special' with respect to others
models in mixed."*

**The first framing was too kind and this ticket now carries the second.** The
privileging is not a convenience keyword. It is structural, it sits in five
modules that are not the adapter layer, and it inverts CLAUDE.md's own stated
priority.

### Measured: module-level imports of `dd2` / `vmit` inside `eos/mixed`

`adapters.py` importing both models is **correct** — that is what an adapter
layer is for, and §5 names the shipped adapters explicitly. Every other site is
the finding:

    api.py:49        from eos.dd2.species import SpeciesFlags
    charges.py:56    from eos.dd2.species import hadronic_qn, hadronic_charges
    responses.py:68  from eos.dd2.solver import warm_start
    scan.py:73       from eos.dd2.nmp import from_nmp, from_hyperon_potentials,
                                             from_delta_potential
    scan.py:76       from eos.dd2.solver import sweep
    scan.py:77       from eos.vmit.parameters import Parameters
    __init__.py:51   from eos.dd2 import Parameters, SpeciesFlags

So the engine's **public API** types its species argument as DD2's class; its
**charge bookkeeping** uses DD2's quantum-number tables; its **response
functions** warm-start through DD2's solver; its **scan** builds parameters
through DD2's NMP inversion and vMIT's dataclass; and its **package `__init__`**
re-exports DD2's two classes as the engine's own.

The public docstrings say it outright: `par : dd2 Parameters`,
`species : dd2 SpeciesFlags` (`api.py:99-140`).

### The inversion

CLAUDE.md §5 states the priority in one direction:

> for the composite engine the **Phase pair IS the parameter argument** (which
> is how §6's "parameters are arguments" reads there), with the plain
> `(par, flags, vmit_params)` signatures **remaining** the DD2+vMIT front door.

The code has it the other way round. `par` is **positional #1** and is a DD2
object; `species` is **positional #3** and is a DD2 object; `vmit_params` is a
named slot in ten-plus functions (264 sites: `solver.py` 16, `boundaries.py` 13,
`table.py` 11, `responses.py` 9, `hybrid.py` 8, `scan.py` 3). `phases=` — the
thing CLAUDE.md calls THE parameter argument — is a **keyword escape hatch with
a `None` default**.

A front door that occupies the first and third positional arguments and types
them to one model is not a front door. It is the building, with a side entrance
for everyone else.

### It is also a §2 breach

§2: the conserved-charge basis maps and the species quantum numbers live in
`general/`, and "**No model carries its own copy of these algebraic maps.**"
`charges.py:56` has the composite engine importing a MODEL's copy of
`hadronic_qn` / `hadronic_charges` — a layer below where §2 puts them, borrowed
sideways. Whatever this ticket rules, that import goes to `general/`.

### Same disease as ticket 29

[Ticket 29](29-mixed-species-flags.md) is "eos/mixed has no `species.py`". It
does not need one **because it borrowed DD2's** (`api.py:49`). The two tickets
are one problem seen from two sides, and 29's ruling — that `mixed` gains its
own flag set and delegates the per-phase sectors — is the first half of this
one's answer.

### The decision

Not whether `phases=` should exist; it does, nine adapters ship, and it works.
The question is **which route is the primary one**, and how much of the engine
is allowed to know the words "dd2" and "vmit":

1. **Invert to match §5.** The Phase pair becomes the parameter argument, in the
   signature, positionally. `(par, flags, vmit_params)` is resolved into a pair
   at each public entry point and appears nowhere below it. `mixed` imports no
   model outside `adapters.py`; `charges.py` takes its quantum numbers from
   `general/` (§2); `responses.py` gets a warm start through the adapter surface
   rather than DD2's solver; `scan.py`'s DD2+vMIT specialisation either moves
   out of the engine or is renamed for what it is. Largest change, and the only
   one after which §5's "couples phases only through this surface" describes the
   code.
2. **Keep the DD2+vMIT signature primary, delete the pretence.** Amend CLAUDE.md
   §5 via [ticket 22](22-phase5-claudemd.md) to say the engine IS a DD2+vMIT
   engine with a generic escape hatch, and stop claiming otherwise. Cheapest,
   and honest — but it contradicts the shipped adapters for `sfho`, `did`, `zl`,
   `alphabag`, `njl`, `ccdm` and `enjl`, which exist precisely because the
   engine is meant to be general.
3. **Split the difference**: fix the three that are outright breaches
   (`charges.py`'s §2 import, `api.py`'s borrowed `SpeciesFlags`,
   `responses.py`'s DD2 warm start), leave `scan.py` as a declared DD2+vMIT
   study, and record the positional asymmetry in `docs/DEFERRED.md`.

**Also in these signatures:** `muons=None` sits beside `vmit_params` in all four
public entry points — [ticket 29](29-mixed-species-flags.md)'s ruling removes it
by making `muons` a flag. Rule 29 first or alongside; not after.

Unrecorded before now: `docs/DEFERRED.md` has zero hits for `vmit_params`, and
the tickets that mention it only quote signatures in passing.

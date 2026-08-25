# Rename did's and dd2's phase-adapter surface to thermo_from_mu

Type: task
Status: open
Blocked by: 36
Parent: ../map.md

## Question

[Ticket 36](36-quark-engine-documents.md) settled the name ticket
[10](10-rename-approvals.md) deferred to `mixed.tex`: the §5 phase-adapter
surface — `(baryon potential, mu_C, mu_S, T) -> PhaseThermo`, solving the
phase's own internal self-consistency — is **`thermo_from_mu`** in every model,
and a lower evaluation layer that additionally takes the solved mean fields is
**`thermo_from_fields`**.

`sfho` carries that ruling under [ticket 45](45-rename-sfho.md). `did` is the
third model with the split, and [ticket 42](42-rename-internal.md) — which
covered `eos/mixed` and `eos/did` — closed before the name was settled, so it
has no ticket.

**dd2's half is also still open, and belongs here.** Ticket 44 carried the same
"Added by ticket 36" instruction and its list of 19 renames does not include
this one: `eos/dd2/thermodynamics.py:571` is still `thermo_at_potentials`, with
no `thermo_from_mu` in the package at all. Found while working ticket 45.
dd2 is the easy case the ticket-44 text already described — one name, no lower
layer to re-spell — but its call sites reach further than did's:

    eos/dd2/thermodynamics.py:17, 571      the def and its module docstring
    eos/mixed/adapters.py:52, 243, 280     a BARE module-level import, not an
                                           alias like the sfho and did ones
    eos/sfho/thermodynamics.py:566         a cross-reference in a docstring
    eos/enjl/thermodynamics.py:741         the same
    test/dd2/test_thermodynamics.py        4 sites
    docs/REFACTOR_PLAN.md:288              names it in prose
    docs/DEFERRED.md:769                   names it in prose

The bare import at `adapters.py:52` is the one to watch: once dd2's surface is
`thermo_from_mu`, that file will hold a module-level `thermo_from_mu` beside two
function-local aliased imports of the same name from sfho and did — shape 3 of
the three-shape collision check, which is exactly what broke
`test/mixed/test_hybrid_modes.py` under ticket 44. Alias it (`_dd2_at_mu`)
like its five neighbours.

`did/thermodynamics.py:542` is `thermo_at_potentials`, the surface;
`did/thermodynamics.py:358` is `thermo_from_mu(par, flags, fields,
mu_tilde_B, mu_C, mu_S, T, matter=None)`, the layer beneath it. **Rename the
lower one to `thermo_from_fields` first**, or the second rename lands on an
occupied name — the pattern that cost ticket 42 twelve silently-red tests
(`mixed/api.py`'s local `solve`) and ticket 43 five collisions
(`vmit/table.py`'s `warm_start`, plus four local `default_guess` bindings the
AST check cannot see).

Call sites: `eos/mixed/adapters.py:797, 813` aliases it as `_did_at_mu`, and
`did`'s own `solver.py` / `verify/`. Run the AST collision check tickets 43-45
carry before moving anything, and `test/baseline/` must not move at
rtol = 1e-10 — a rename that changes a number is not a rename.

# Rename did's phase-adapter surface to thermo_from_mu

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

`dd2` and `sfho` carry that ruling under tickets
[44](44-rename-dd2.md) and [45](45-rename-sfho.md). `did` is the third model
with the split, and [ticket 42](42-rename-internal.md) — which covered
`eos/mixed` and `eos/did` — closed before the name was settled, so it has no
ticket.

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

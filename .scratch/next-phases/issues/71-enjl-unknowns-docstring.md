# `eos.enjl.api` documents a name the module does not export

Type: task
Status: open
Blocked by: none
Parent: ../map.md

## Question

`eos/enjl/api.py:106` tells the caller that `x0` is "a starting guess in the
order of `eos.enjl.solver.UNKNOWNS`". **There is no `UNKNOWNS`.**
`eos/enjl/solver.py:133` defines `BASE_UNKNOWNS`, the ten of the base vector,
and `:137 unknown_slots(spec)` returns those ten plus one potential per held
charge — so the true ordering is mode-dependent and the docstring names neither
function.

This is the documented contract for a public argument: a caller who follows it
gets an `ImportError`, and a caller who guesses gets an ordering that is right
for `beta_eq_neutrinoless` and wrong for a mode with an extra held charge.

Found while writing [ticket 19](19-enjl-stepwise.md)'s step-1 cell, which
prints the ten unknowns by name rather than importing anything. It is not that
ticket's to fix (notebook-only scope) and
[ticket 54](54-signature-corrections.md), where a signature correction would
have belonged, is resolved.

Done when the docstring names `unknown_slots` (and `BASE_UNKNOWNS` as the base
of it), says the ordering depends on the mode, and no other public docstring in
`eos/enjl` points at a name the module does not carry — the same grep is worth
running over the other models while it is open.

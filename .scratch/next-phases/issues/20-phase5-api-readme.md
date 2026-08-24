# Phase 5 items 1, 2 and 4 — top-level imports and a runnable README

Type: task
Status: open
Blocked by: 10, 11
Parent: ../map.md

## Question

`docs/REFACTOR_PROMPTS.md` Phase 5, items 1 and 2, with item 4 applied to both:

1. `eos/eos/__init__.py` so the common tasks are one import deep, and the modes
   and species flags are importable from the top level.
2. Rewrite `eos/README.md`: what the library computes, install, and four runnable
   examples — (a) one point from one model in `beta_eq_neutrinoless`, (b) a
   `fixed_YC` table swept over `n_B` and `T`, (c) that table fed to
   `eos/astro/tov/` for an M–R curve and maximum mass, (d) the same model with
   hyperons enabled, showing the species flags. Then a fifth: an M–R figure with
   the observational constraints overlaid, in the house style, in under ten
   lines. **Copy-paste runnable, no pseudocode.**

**Item 4 is not optional**: execute every code block in the README and paste the
real output. Do not write an example you have not run.

Blocked on the rename approvals (ticket 10) and the conformance triage (ticket
11) because both can move the public surface these examples import.

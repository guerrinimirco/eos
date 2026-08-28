# SU(6) is an assumption: give the vector hyperon couplings a rescaling factor

Type: task
Status: open
Blocked by: -
Parent: ../map.md

## Question

Requested by the user (2026-08-28), and it is the concrete form of the
hyperon-sector question [ticket 103](103-nmp-closures-four-models.md) asks in
general.

Today the vector hyperon couplings are SU(6) and nothing else:
`dd2/couplings.py::SU6_HYPERON` and `sfho/parameters.py::SU6_RATIOS` hardcode
`x_omega = 2/3, 2/3, 1/3` for Lambda, Sigma, Xi, `x_phi = -sqrt2/3, -sqrt2/3,
-2 sqrt2/3` and `x_rho = 0, 2, 1`. Only the SCALAR couplings are inverted,
from the potential depths. So the numbers a hyperon inference would most want
to vary are the ones with no knob.

SU(6) spin-flavour symmetry is a quark-model assumption, not a measurement,
and breaking it is a standard move — Fortin, Oertel & Providencia and the
SFHoY family scale the SU(6) vector couplings by factors the user gives as
1.5, 1.5 and 1.875, against all-1 for the SU(6) variant.

## Downstream ping from [ticket 102](102-retire-phi-field-flag.md) (2026-08-28)

**102 put a first knob on the `x_phi` column and this ticket should subsume
it, not sit beside it.** Retiring `phi_field` left the phi sector controlled by
its coupling, and dd2's named route is now
`nmp.from_hyperon_potentials(..., x_phi=None)`: `None` keeps the SU(6) column,
a float replaces it in EVERY row. That is a single scalar, deliberately the
smallest thing that expresses "no phi sector" (`x_phi = 0.0`) — it is not
per-multiplet and it cannot express the 1.5 / 1.5 / 1.875 pattern this ticket
is about.

`R_phi_Y` per multiplet, defaulting to 1.0, is the general form of the same
knob: `R_phi_Y = 0` for every multiplet IS `x_phi = 0.0`. So this ticket should
**replace** the scalar argument rather than add a second way to reach the same
column, and it inherits one constraint from 102's gate — CLAUDE.md §4 now says
the phi sector is controlled by its coupling, and
`test/test_imports.py::test_phi_sector_is_off_exactly_when_its_coupling_is_zero`
asserts `from_hyperon_potentials(x_phi=0.0)` builds a phi-free set through
`eos_point`. Whatever replaces the argument must keep that check meaningful,
which means keeping a reachable way to say "no phi".

`sfho`'s side has no such argument: 102 was pure deletion there, and
`SFHo_2fam` (g_phi = 0 for every hyperon) against `SFHo_2fam_phi` remains the
only route, hardcoded per named set rather than parameterised — which is
exactly the gap this ticket closes.

## Work

1. Rescaling factors as PARAMETERS on the parameter object, per hyperon
   multiplet and per vector meson: `R_omega_Y`, `R_phi_Y`, `R_rho_Y`, each
   defaulting to **1.0 = SU(6)**. The rho factor is 1 in both published sets
   the user names but must be free for the same reason the others are.
2. The published breaking set as a NAMED alternative
   (`Parameters.named(...)`), not as a new default — §6's "published sets are
   named defaults, never hardcoded values".
3. **Pin down which factor attaches to which vertex before writing anything.**
   The user gives 1.5, 1.5, 1.875 without naming the mesons, and SU(6)'s
   omega ratios are 2/3, 2/3, 1/3 — so 1.5 x 2/3 = 1 for Lambda and Sigma
   while 1.875 x 1/3 = 0.625 for Xi, which is a coherent reading but IS a
   reading. Take it from Fortin et al.'s own table, not from arithmetic that
   happens to come out round.
4. Both models, one vocabulary: `dd2` carries the ratios in
   `hyperon_couplings` rows and `sfho` in `couplings_map`, so the factor must
   enter each model's own constructor
   (`from_hyperon_potentials`, `from_potential_depths`) rather than being
   applied by the caller afterwards.
5. **The scalar inversion must re-run after the rescaling, not before.** The
   depths fix `g_sigma_Y` through `U_Y = -g_sigma_Y sigma + g_omega_Y omega`,
   so changing `g_omega_Y` changes the scalar coupling that reproduces the
   same depth. Whether the depths or the scalar ratios are what is held is
   the caller's physics and must be an argument, not a default — this is the
   same trap `docs/DEFERRED.md` records for sfho's `invert_nmp`.

## Gate

`R = 1` everywhere reproduces every current number bit-for-bit — that is the
whole regression argument, and `test/baseline/{dd2,sfho}.npz` must not move.
The named breaking set reproduces its published U_Y depths through
`compute_hyperon_potentials`. One test that a rescaling actually moves
`g_omega_Y` and that the scalar coupling follows it at fixed depth.

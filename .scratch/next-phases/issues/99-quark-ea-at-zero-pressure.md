# E/A at P = 0, T = 0 for two- and three-flavour quark matter

Type: grilling
Status: open
Blocked by: - (98 ruled 2026-08-27; both arms unblocked)
Parent: ../map.md

## Question

Requested by the user (2026-08-27). The quark models must report

    E/A (P = 0, T = 0) = eps / n_B  at the self-bound surface   [MeV]

for BOTH flavour contents: three-flavour matter (u, d, s, beta-equilibrated,
neutral) and two-flavour matter (u, d only — Y_S = 0, neutral,
mu_u + mu_e = mu_d).

**Why both, and why this pair of numbers.** It is the Bodmer–Witten window,
and it is a two-sided gate on a quark parameter set: three-flavour E/A below
the 930 MeV of iron says strange quark matter is absolutely stable, and
two-flavour E/A ABOVE 930 MeV says ordinary nuclei are not already decaying
into it. A set that fails either is excluded, so an inference run over quark
parameters wants both numbers per sample. `eos/abpr/parameters.py:92` and
`eos/abpr/verify/run_full_check.py:190` already state the three-flavour half
in exactly these terms for the shipped ABPR set (E/A = 831.58 MeV); the
two-flavour half has no implementation anywhere.

**There is a consumer waiting.** [Ticket 98](98-fixed-ys-undeclared-mode.md)
records that BayEoS holds `two_flavour_stable: skipped` rather than passing
its gate silently, with its `OPEN_QUESTIONS` pointed at that ticket. Whoever
rules this owes the same ping 98 does.

### The physics is one line at P = 0, and that line is also a free check

At T = 0 the Euler relation (§8) is eps + P = sum_i mu_i n_i, so at the P = 0
surface eps/n_B is the Gibbs energy per baryon exactly. In §2's basis, with
beta equilibrium (mu_C + mu_e = 0) and total neutrality (n_C = n_e), the
lepton terms cancel against the charge term and

    E/A = mu_B + Y_S mu_S

so the two-flavour number at Y_S = 0 is **mu_B at the P = 0 root**, and the
whole computation is a one-dimensional root find plus one read. Computing
eps/n_B directly and comparing against mu_B + Y_S mu_S is then free, and it
is the identity `abpr`'s `check_surface` already asserts.

**Two mu_B conventions meet at that identity and a shared helper must not
read the wrong one.** `eos/abpr/verify/run_full_check.py:185` asserts
E/A == mu_B, while `eos/alphabag/solver.py:767` records that in CFL "the
energy per baryon at P = 0 is mu_B + mu_S, not mu_B". Both are right in their
own model: ABPR is a single-mu model (`thermodynamics.py:7`, mu_B = 3 mu, so
mu_S vanishes identically), and alphaBag's locked phase pairs equal densities
at unequal masses, so mu_S is nonzero there. Reconcile the two before any
shared code reads either.

### Three routes to two-flavour matter, and the map has already paid for
### evidence on the first

The three-flavour arm needs nothing new — it is `beta_eq_neutrinoless` at the
P = 0 root. The fork is entirely about how two-flavour matter is REACHED.

1. **A `fixed_YS` mode** (Y_S = 0, charge-equilibrated, neutral). This is
   [ticket 98](98-fixed-ys-undeclared-mode.md)'s subject and that ticket
   records BayEoS asking for exactly this, on exactly these models. Cost this
   ticket adds to 98's ledger: holding Y_S = 0 where no populated species
   carries S is precisely what
   [ticket 75](75-undetermined-potential-check.md)'s screen was built to fire
   on — mu_S becomes a null Jacobian column — and
   [ticket 72](72-enjl-branch-selection.md) measured what that costs when it
   went unnoticed: three decades of residual, and round-off choosing the
   chiral branch. The route is not wrong, but it walks into a hazard this map
   has already diagnosed twice, and the locator would be reading mu_S out of a
   solve that does not determine it.

2. **A species flag** that switches the s quark off, so the flavour leaves the
   species list and mu_S leaves the unknown vector. This is §4's shape ("if a
   sector is off, its flag is False"), it has no null column to condition
   around, and it needs no §3 amendment at all — two-flavour matter becomes
   `beta_eq_neutrinoless` with one sector off, which is what it physically is.
   Cost: **no quark model has such a flag today** — checked, `vmit`,
   `alphabag`, `njl`, `ccdm` and `abpr` all carry
   photons/gluons/muons/hyperons/deltas/thermal_mesons/thermal_neutrinos and
   nothing for strangeness — and §4's vocabulary has no name for it
   (`hyperons` is the strange BARYON sector; the quark sector's name would be
   new, and §4 names are mandatory once coined).

3. **A standalone routine** that computes only the two numbers, the user's
   second option. Smallest diff by far. The question it must answer is
   whether it can reach two-flavour matter without (1) or (2) — a private
   two-flavour residual inside the routine is a fourth place the equilibrium
   conditions get written, which §5's thermodynamics/solver boundary and §13's
   one-name-per-job rule both push back on.

### Where it lives, if it is more than a script

§7's single-home rule bites: `abpr` computes this already, and a second model
writing its own P = 0 search makes two. But ABPR's is closed-form
(`mu_from_P`), so what generalises is **the locator over a callable**, not
ABPR's code — a P = 0 bracket taking `P(n_B)` can sit in `general/` without
importing a model, which keeps §1's layering intact. Each quark model then
owes one §13 name for the entry point, the same name in every model.

### Which models

`vmit`, `alphabag`, `njl`, `ccdm`, `enjl`'s quark branch, and `abpr` (which
has the three-flavour half). **`cfl` has no two-flavour arm by construction**
— §3 says the locking fixes Y_S = +1 identically — so for `abpr`, whose only
mode is `cfl`, the two-flavour number does not exist rather than being
unimplemented, and the gate must say so instead of reporting a NaN.

## Gate

- Both numbers reported for every quark model's default parameter set at
  T = 0, with the flavour content each was computed at named beside it.
- `E/A = mu_B + Y_S mu_S` holds at the located root to ~1e-12 relative, per
  model — the free cross-check above, and the one that catches a locator that
  found a root of something other than P.
- **`abpr`'s 831.58 MeV reproduces through whatever new path is built.** It is
  a §12 golden reference pinned in a `verify/` suite; new code that needs a
  different number is wrong until proven otherwise.
- Two-flavour E/A > 930 MeV and three-flavour E/A < 930 MeV are REPORTED per
  set, not asserted — whether a set sits in the Bodmer–Witten window is a
  property of the set, not an invariant of the code, and a `verify/` entry
  that asserts it would fail on a legitimately-excluded parameter point.
- No number moves anywhere else. A moved baseline means the route reached
  further than it was meant to.
- The ruling is pinged downstream to BayEoS, with ticket 98's.

## Fog this opens

Whether E/A(P = 0) is a `verify/` invariant, a reported quantity on the public
API, or both — §5 answers this shape of question for the response functions
and does not answer it here. It should be decided WITH the route rather than
after it, because route 3 answers it by construction and routes 1 and 2 do
not.


## Unblocked by ticket 98 (2026-08-27)

[Ticket 98](98-fixed-ys-undeclared-mode.md) ruled, and it rules **route 2**
above: `fixed_YS` is not a §3 mode and is demoted to an internal `ModeSpec`
label. Two-flavour quark matter is reached as **`beta_eq_neutrinoless` with the
strange sector's flag False**.

Route 1 is closed, and closed on this ticket's own evidence: holding Y_S = 0
where no populated species carries S is [ticket 75](75-undetermined-potential-check.md)'s
null column, [ticket 72](72-enjl-branch-selection.md) priced it, and §4 forbids
switching a sector off through a fraction that happens to vanish. Route 3 (a
standalone routine) is not ruled out as the *shape* of the deliverable — the
question this ticket still owns — but it no longer needs a private two-flavour
residual, because route 2 gives it one through the existing mode.

### Inherited, not open

98 ruled the flag's **category**, which is binding here:

- **two legal values in the unpaired and 2SC modes**, defaulting False (§4);
- **RAISES under CFL pairing** — `alphabag`, `njl`, `ccdm` each carry both
  regimes, so each gets both behaviours;
- **`abpr` refuses it outright**, as it does `gluons`: `cfl` is its only mode,
  so this ticket's "`cfl` has no two-flavour arm by construction" is expressed
  by the flag refusing rather than by a NaN, which is what the Gate asked for.

The flag is in §4's `phi_field`/`gluons`/`csc` class — physics only quark
models have — **not** a seventh mandatory name, so its footprint is five
models, not ten. `test_every_species_flag_defaults_off_or_raises` checks the
category the day the flag lands; nothing new is owed there.

### Still this ticket's to decide

- **The flag's NAME.** §13: the same name in every model, mandatory once
  coined. §4's `hyperons` is the strange BARYON sector and is taken.
- **Where the P = 0 locator lives** and its §13 entry-point name (the ticket's
  own "locator over a callable" in `general/`).
- **The two `mu_B` conventions** at `E/A = mu_B + Y_S mu_S`, unchanged by 98.
- The Fog below, unchanged: `verify/` invariant, public API, or both.

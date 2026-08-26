# A composition contract in general/, so astro/gmode stops importing dd2.solver

Type: task
Status: resolved
Blocked by: 11
Parent: ../map.md

## Question

`eos/astro/gmode` imports model internals, the last live §1 breach now that
`dd2/notebook_api.py` is gone:

    eos/astro/gmode/rates.py:85                   from eos.dd2.solver import solve_composition
    eos/astro/gmode/sound_speeds.py:94            from eos.mixed.responses import sound_speed_eq, sound_speed_frozen
    eos/astro/gmode/sound_speeds.py:149           from eos.dd2.solver import solve_composition   (function-local)
    eos/astro/gmode/verify/run_full_check.py:39-41  eos.dd2.Parametrization, eos.dd2.responses, eos.dd2.solver

§1: "`astro/` … consumes tables and arrays produced by models and engines; **it
never imports model internals**." `solve_composition` is a solver internal, and
`rates.py:85` is a **top-level** import, so `import eos.astro.gmode` pulls DD2 in.

[Ticket 11](11-conformance-triage.md) ruled out the cheap answer — amending §1 to
name gmode as a second exception beside `mixed/` — on the grounds that the astro
half of §1 was tightened *because* this ambiguity existed, and that a carve-out
would make gmode DD2-only by specification when the physics need is general. The
ledger entry recording the gap belongs to [ticket 55](55-deferred-ledger.md); the
design is this ticket's.

**The question: what is the contract?** A composition g-mode needs
d(composition)/dn_B along the equilibrium sequence, which no `EOSTable_for_TOV`
carries — that is why the import exists. `EOSTable_for_TOV` is the shape to copy:
it lives in `general/`, the layer both `astro/` and the models may import, so a
model's side of the contract is *producing* one and astro's side is *consuming*
it. The open questions:

- Does the contract carry the composition **derivative**, or the composition on a
  fine enough grid that gmode differentiates it itself? The second is a smaller
  interface and moves the numerical choice to the consumer.
- Does it also cover the two sound speeds gmode currently takes from
  `eos.mixed.responses`, or is `eos_response` (§5, already returning
  `cs2_equilibrium` and `cs2_frozen`) the surface for those?
- Which models must produce it before gmode can drop the DD2 import? Only `dd2`
  and `mixed` are consumed today.
- `gmode/verify/run_full_check.py` reaches into dd2 too. §1's `verify/` carve-out
  ([ticket 22](22-phase5-claudemd.md), finding 3) is written for the
  model-to-model half of the rule; whether it extends to an astro suite reaching
  down into a model is a separate call.

**Rides along, same files (finding 17a').** `eos/astro/gmode/rates.py:90-97`
declares `G2_FERMI = 1.1e-22`, `G_A = 1.26`, `F_PI_NN = 1.0`, `M_PI = 139.57039`
as module constants with no override path (§6), and `M_PI` additionally
duplicates a mass §7 puts in `eos/general/particles.py`. Fix both while the file
is open: the mass comes from `general/particles.py`, the weak couplings become
arguments.

Resolved when the contract is designed and ruled — not necessarily built. If
building it is a session's work in its own right, that becomes its own ticket.

## Note from ticket 69 (2026-08-26)

[Ticket 69](69-cs2-eq-naming.md) renamed `cs2_eq` -> `cs2_isothermal` across
the ten models and deliberately stopped at this surface. The reason is that
`mixed` and `gmode` are not two spellings but ONE: `eos/astro/gmode/sound_speeds.py`
imports `sound_speed_eq` and `sound_speed_frozen` from `eos/mixed/responses.py`,
so `mixed.eos_response`'s `cs2_eq`/`cs2_frozen` keys and gmode's `cs2_eq`/`cs2_ad`
arguments and `Background` fields share one vocabulary. Renaming half of it is
worse than renaming none.

So this ticket now also settles the third spelling: whether the g-mode surface
is `cs2_equilibrium`/`cs2_frozen`, and whether `mixed.eos_response`'s keys move
with it. §5's rule is that the key names the THERMAL axis and the `frozen=`
argument names the composition axis — but gmode's two speeds are not one call
with a `frozen=`; they are two arrays a caller supplies, so the rule does not
transfer unexamined. That is the decision.

An AST check run for ticket 69 found `cs2_eq` bound as an identifier at 35
sites under `eos/astro/gmode/` plus `test/gmode/`, all parameters, assignments
or dataclass fields. None is a dict key, so a rename here is a real refactor
rather than a string sweep.

## Ruling

Agreed with the user, and grounded in **Zhao & Lattimer, arXiv:2204.03037**
Eq. (1): `nu_g^2 = g^2 (1/c_e^2 - 1/c_s^2) e^(nu-lambda)`. **The g-mode is the
difference between the equilibrium and frozen sound speeds** — given one alone,
`nu_g` is identically zero. That is exactly why `gmode` reaches into
`dd2.solver`.

**The contract.** Model-general, like TOV: it lives in `general/` beside
`EOSTable_for_TOV` — the layer both `astro/` and the models may import — every
model PRODUCES one, `gmode` CONSUMES it, and no model internal is imported.

**Payload: the two sound speeds along the sequence, not a composition
derivative.** Zhao Eq. (1) needs `c_e` and `c_s` per point and nothing else, and
`eos_response` already returns both per model. This is a SMALLER interface than
the ticket imagined, and it answers the ticket's second sub-question: the sound
speeds ARE the contract, not a separate surface.

**T = 0 only, as a first approach** (user's ruling; finite T when it is useful).
This is clean rather than a compromise: Zhao's operative clause is "without
varying chemical composition", NOT the zero temperature. At T = 0 with varying
composition `c_e != c_s` and the g-mode is nonzero — the dd2-with-hyperons case.
**T = 0 collapses only the THERMAL axis**, leaving the composition axis intact:
exactly two numbers per point and no thermal-axis naming problem at all.

**The blocker is not what the ticket thought.** Measured: `frozen='composition'`
is implemented in **`dd2` alone**. Six models expose only `equilibrium`;
`njl`, `ccdm` and `enjl` expose no freezes at all. So nine models cannot compute
the second sound speed under any conditioning, and `C_P` was never the
constraint.

**Therefore the contract ENDS THE SECTION 1 BREACH but does not make `gmode`
general** — and that is still worth doing. It converts "gmode is DD2-only by
accident, hidden inside `from eos.dd2.solver import solve_composition`" into
"gmode is DD2-only until nine models implement one freeze": visible, per-model,
and ticketable. A model that cannot fill the contract raises saying so, which is
§3's own answer to a partly-filled surface.

Execution is [ticket 77](77-gmode-contract-build.md). The nine-model freeze gap
is [ticket 78](78-composition-freeze-nine-models.md).

**Rides along, same files** (finding 17a'): `eos/astro/gmode/rates.py:90-97`
declares `G2_FERMI`, `G_A`, `F_PI_NN`, `M_PI` as module constants with no
override path (§6), and `M_PI` duplicates a mass §7 puts in
`general/particles.py`. The mass comes from there; the weak couplings become
arguments.

Status: resolved.

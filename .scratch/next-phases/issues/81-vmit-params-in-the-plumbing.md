# `vmit_params` is threaded through the engine's internals, not just its front door

Type: grilling
Status: open
Blocked by: -
Parent: ../map.md

## Question

Raised by the user: *"In hybrid_table an input is vmit_params? why? It should
work for all the models."*

**It does work for all the models.** All four public entry points
(`eos_point`, `eos_table`, `hybrid_table`, `eos_response`) take
`phases=(Phase, Phase)` with `par=None` (`api.py:127`), and nine adapters ship:
`dd2_phase`, `sfho_phase`, `did_phase`, `zl_phase`, `vmit_phase`,
`alphabag_phase`, `njl_phase`, `ccdm_phase`, `enjl_branch_pair`.

And the front-door slot is a **documented carve-out**, not an oversight.
CLAUDE.md:277: the Phase pair is the parameter argument, *"with the plain
`(par, flags, vmit_params)` signatures remaining the DD2+vMIT front door."*
`adapters.py:685 default_pair` implements exactly that and calls itself "the
engine's historical default".

**The defect is that the slot did not stay at the front door.** Measured:
`vmit_params` appears at **264 sites in `eos/`**, and the distribution is the
finding:

    solver.py 16   boundaries.py 13   table.py 11   responses.py 9
    hybrid.py  8   scan.py        3   api.py     4

So `solve_fixed_chi`, `refine_window`, `locate_windows`,
`sound_speed_frozen`, `sound_speed_frozen_quark`, `_sweep_at_entropy` and
`solve_at_entropy` each carry a named slot for **one specific quark model's
parameter object**.

§5 says `eos/mixed` "couples phases **only through this surface**" — the phase
adapter. A `vmit_params=` argument threaded through the engine's internals is a
second, model-specific coupling surface running beside the contract. The
carve-out CLAUDE.md granted was for the *front door*; nothing granted it to the
plumbing.

Two consequences, neither hypothetical:

- A new quark pairing inherits ten internal functions whose signatures name a
  model it does not use, and must pass `vmit_params=None` through all of them.
- `vmit_params=None` is ambiguous by construction: in `default_pair` it means
  "build vMIT with its defaults", but on a call that passed `phases=` it must
  mean "no vMIT anywhere". The same value carries two meanings depending on an
  argument three frames up.

**The decision this ticket owes.** Not whether to delete the front door —
CLAUDE.md grants it and `notebooks/hybrid_eos` shows both forms deliberately
(ticket 58). The question is whether the INTERNALS keep the slot:

1. **Push the pairing to the boundary.** Every public entry point resolves
   `(par, flags, vmit_params)` into a `Phase` pair on entry, and every internal
   function below takes only the pair. `vmit_params` then appears in exactly
   four signatures instead of ten-plus, and §5's "only through this surface"
   becomes true of the code and not only of the documentation.
2. **Rename it `quark_params` and keep it threaded.** Cheaper, honest about the
   shape, but leaves a second coupling surface that the contract says does not
   exist — and does not fix the two-meanings-of-`None` problem.
3. **Record it in `docs/DEFERRED.md`** with the measurement, as the cost of the
   historical front door.

Related, same signatures: **`muons=None` sits beside `vmit_params` in all four
public entry points**, which is the phase-common-sector problem
[ticket 29](29-mixed-species-flags.md) rules on — once `mixed` has a
`species.py`, `muons=` is a flag and not a kwarg, and that removes one of the
two special cases from these signatures. Whatever this ticket decides should be
consistent with 29's ruling.

Unrecorded before now: `docs/DEFERRED.md` has zero hits for `vmit_params`, and
the four tickets that mention it only quote signatures in passing.

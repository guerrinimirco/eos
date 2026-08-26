# Nine models cannot compute a frozen-composition response

Type: task
Status: open
Blocked by: 53
Parent: ../map.md

## Question

Measured while ruling [ticket 53](53-gmode-contract.md), and not previously
recorded anywhere:

    dd2                        RESPONSE_FREEZES = ['composition', 'equilibrium']
    sfho did zl vmit alphabag abpr              = ['equilibrium']
    njl ccdm enjl                               = []   (no freezes at all)

§5 makes the conditioning of a second derivative explicit and names
`equilibrium`, `fast` and `slow` as presets over a SET of held quantities. Nine
models implement one preset; three implement none.

Two consequences, both live:

- **The g-mode contract cannot be filled by anyone but `dd2`.** Zhao & Lattimer
  Eq. (1) needs the equilibrium AND frozen sound speeds; with one alone the
  g-mode frequency is identically zero. This is why `gmode` is DD2-only, and it
  is a physics gap, not a layering one.
- **`docs/DEFERRED.md` records the per-model freeze menus** (`:915`, `:954`,
  `:1041`, `:1078`) but nowhere states that the frozen-composition speed —
  the one asteroseismology needs — exists in exactly one model.

Not one ticket's work: each model's freeze is its own physics (an RMF holds
`Y_i` differently from a bag model holding flavour fractions, and `mixed` must
also hold `chi`). This ticket **decides the order and the shape**, then spawns
per-model work; it does not implement nine freezes itself.

Start from `dd2/responses.py:35-44`, the one working implementation: `T` held on
both stencil points, `n_B` perturbed at fixed `Y_p`.

Done when the gap is recorded in `docs/DEFERRED.md` with the measurement above,
and the per-model order is decided with `mixed` (which owes `chi` as well as
`Y_i`) placed explicitly.


## Note from ticket 77 (2026-08-26)

Building the contract ([ticket 77](77-gmode-contract-build.md)) turned up a
**tenth** gap this ticket's measurement does not show, and it changes the
starting point above.

`dd2.eos_response(frozen='composition')` returns `sound_speed_adiabatic`, which
is **LEPTONLESS**. That is the right probe of the nucleonic sector and is what
its docstring promises, but it is the wrong half of a g-mode: the equilibrium
speed follows the beta-equilibrium sequence WITH the neutralising leptons, so
differencing the two compares two different fluids. The lepton contribution to
`c_s` is a few per cent — comparable to the whole `c_s^2 - c_e^2` signal — so
the mismatch is a leading-order error in `N^2`, not a refinement. Measured on
DD2 it flips the sign of the buoyancy over part of the density range, i.e. a
spurious convective instability; `gmode/verify`'s "convective stability" check
exists to catch exactly that.

So **`dd2` is not one working implementation to copy; it is one working
implementation of the leptonless half.** What the contract needs is §5's THIRD
conditioning axis, the one CLAUDE.md already names — *whether leptons
re-neutralize against the held charge* — wired onto
`eos_response(frozen='composition', leptons=...)`. Both variants are wanted:
leptonless for the nucleonic probe, with-leptons for stellar matter.

That axis is missing in all ten models, not nine, and deciding its spelling
belongs here rather than in each per-model ticket, since a per-model answer
would drift immediately.

The working with-leptons implementation exists and is tested: `dd2_frozen_cs2`
in `eos/astro/gmode/verify/run_full_check.py`, reproducing the shipped DD2
g-mode numbers bit-identically. It belongs in `eos/dd2/` and sits there only
because `eos/dd2/` was under concurrent edit; moving it is the first per-model
step this ticket orders. `test/gmode/test_sound_speeds.py` already pins
`leptons=False` against dd2's own function to machine precision, so the move
has its check waiting.

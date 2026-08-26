# Nine models cannot compute a frozen-composition response

Type: task
Status: resolved
Assignee: session 78-freezes
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


## Ruling

Recorded in `docs/DEFERRED.md`, new section **"The composition freeze: what each
model owes, and the order it lands in"**, placed after the `astro/gmode` section
whose deferral it completes. Docs only; no `eos/` code touched.

**The measurement in this ticket was wrong and is corrected.** Taken live rather
than restated:

    dd2                          ('equilibrium', 'composition')
    mixed                        ('equilibrium', 'chi')
    sfho did zl vmit alphabag
      abpr njl ccdm              ('equilibrium',)
    enjl                         ()

`njl` and `ccdm` do NOT have empty menus — they have carried `equilibrium` since
the commits that introduced them (`75b617e`, `971f6ad`), so `enjl` alone has no
response surface. And `mixed` was missing from the count altogether: it is the
eleventh unit and carries a third spelling, `chi`. The stale sentence in the
`astro/gmode` section is corrected in the same diff rather than left to
contradict the new one.

**Why it is nine separate jobs, which is the substance of the order.** A
composition freeze needs the model at PRESCRIBED SPECIES DENSITIES with no
equilibrium condition — §13's `thermo_from_n`, the direction that inverts the
Fermi integrals. It cannot be reached by re-tuning `(mu_B, mu_C, mu_S)`: three
conserved potentials cannot hold eight species fractions. That block exists in
`zl`, `vmit` and `enjl`; `dd2` has the nucleonic special case
(`solve_composition`, a single `brentq` on the sigma gap over n and p — so
**dd2 cannot freeze with hyperons or deltas on either**); the other seven have
nothing.

**The order decided**, driven by the g-mode consumer first and by
already-existing blocks second:

    1. dd2        the with-leptons producer home from gmode/verify + the third
                  axis; its test already waits
    2. zl, vmit   `thermo_from_n` exists; wiring, and they prove the axis across
                  a hadronic AND a quark model before anything expensive
    3. abpr       a RULING, not work: CFL locks n_u = n_d = n_s, so no fraction
                  is free, the frozen speed IS the equilibrium one, and an ABPR
                  g-mode is identically zero for a physical reason
    4. sfho, did  must WRITE the block. sfho's four field equations are
                  nonlinear (c3 omega^4, the omega-rho mixing) so prescribed
                  densities give a coupled 4-D root find, not dd2's scalar gap.
                  did carries TWO rearrangement self-energies (couplings depend
                  on n_B and on beta); beta is constant along a frozen sequence,
                  so that channel drops out by construction and DID's frozen
                  speed is structurally not the same object — a choice to make
                  deliberately, not inherit
    5. alphabag   flavours decouple, so three 1-D inversions — but of the
                  alpha_s-corrected density, so it owes its own inverter
    6. njl, ccdm  most expensive: gap equations (plus ccdm's dielectric) re-solved
                  at prescribed flavour densities with cutoff-regularised
                  integrals, AND pairing means the composition is not free (CFL
                  degenerate as abpr, 2SC ties two of three). Both SELECT their
                  pattern by comparing pressures, so the freeze must first
                  declare whether the pattern is held with the composition or
                  re-selected per stencil point — the same question open for
                  enjl's branch pair
    7. enjl       `equilibrium` first (it has none); composition is then cheap,
                  its `thermo_from_n` being the most complete in the repository
    8. mixed      **last by construction, not by priority.** Its `chi` freeze
                  already holds each phase's Y_C and Y_S. §5's `fast` additionally
                  holds every Y_i, and the engine has no species of its own —
                  they live in the phases, behind the phase-adapter contract. So
                  mixed can hold {Y_i} | {chi} only once BOTH phases can hold
                  their own

**Also ruled here, because ticket 77 handed the spelling over and it collides.**
§5's third axis (whether leptons re-neutralize) cannot be spelled `leptons=`:
that keyword is already taken on `eos_response` and already means two different
things — the §3 MODE flag in `sfho zl did vmit alphabag njl ccdm` (routed into
`mode_spec`), and the RESPONSE axis in `mixed` (routed into
`sound_speed_frozen`). One name, two jobs, shipped in opposite senses.

**`leptons=` keeps its §3 mode meaning on `eos_response`; the response axis
takes `reneutralize=` (bool, default True).** Reasons: `eos_point` and
`eos_table` take the same argument with the mode meaning and a uniform API
cannot have siblings read a keyword differently; [ticket 70](70-leptons-on-a-beta-mode.md)
just landed that rule in nine models; and the two axes are genuinely orthogonal
— `leptons=` says whether the STATE has leptons, `reneutralize=` whether the
PERTURBATION re-neutralises — so a caller can want both at once. `mixed` is the
one surface that renames, against nine models left alone. Consequence worth
watching: since `leptons=False` on a beta mode now RAISES, the leptonless probe
MUST be reached through `reneutralize=False`, never by turning the mode's
leptons off.

**Not done, deliberately.** No freeze implemented, and no per-model tickets
created — the nine steps are **out of scope of this map** (see the map's Out of
scope): nothing in the Acceptance criteria block measures a response freeze, and
`docs/DEFERRED.md` is exactly the repository's tracked ledger for a per-model gap
carried past the refactor. The order now lives there, where the work will be
picked up from.

**Gate.** Docs only — `docs/DEFERRED.md` and this ticket. No `eos/` file touched,
so no test can move; the three test files that mention `DEFERRED.md` do so in
prose and none parses it. The live session's concurrent edit to `DEFERRED.md`
(the `build_mixed_eos_table` -> `build_hybrid_table` rename, now at `:2106`) was
checked before and after and is intact.

**Found and not fixed** (Stage 7 report, not this diff): the map's `## Out of
scope` section has a run of Decisions-so-far entries misfiled beneath its two
real bullets — appended to end-of-file rather than to the Decisions section.

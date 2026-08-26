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

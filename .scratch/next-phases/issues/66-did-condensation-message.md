# `did` reports meson condensation as a residual below its own gate

Type: task
Status: open
Blocked by: —
Parent: ../map.md

## Question

Surfaced by [ticket 12](12-hadronic-skeleton.md), which was told to report gaps
rather than fix them.

`eos/did/api.py:99` formats a non-convergence message as

    f"residual {point.error:.3e} above the gate at n_B={n_B:g} fm^-3"

for every point whose `converged` flag is False. But `eos/did/solver.py:574`
clears that flag for a **second, unrelated reason**:

    if point.converged and point.condensation >= 1.0:
        point = replace(point, converged=False)

So with `thermal_mesons=True` at `n_B >= 0.4`, every temperature tried, `did`
returns

    ok=False   "residual 5.684e-15 above the gate at n_B=0.4 fm^-3"

with `condensation = 1.16` and `RESIDUAL_TOL = 1e-10`. The residual named is
five orders of magnitude BELOW the gate it is said to exceed, so a reader who
checks the number against the gate — the one thing the message invites — is
told something false. The point is correctly rejected; only the reason is wrong.

`sfho` and `dd2` do this right and are the model to copy:

    the thermal meson gas Bose-condenses at n_B=0.4, T=10.0
    (max |mu*|/m = 1.322); a condensate is not implemented, so this
    state is outside the model

§3's rule for a refusal is that it says *which*, and §6 makes non-convergence a
return value the caller can act on — a caller cannot act on a reason that is not
the reason. Give the condensation rejection its own message, naming
`condensation` and the density and temperature, and leave the residual message
to the residual.

Check whether `n_B <= 0.2`, which converges at every T tried, is the whole of
the domain or whether the threshold moves with the parametrisation; and check
whether any other model clears `converged` for a second reason while formatting
only the first.

Done when a `did` meson-condensation rejection says so in its own words, the
residual message is reached only by an actual residual miss, and the notebook's
species-flag cell prints the new message.

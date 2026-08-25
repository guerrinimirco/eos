# `did` reports meson condensation as a residual below its own gate

Type: task
Status: resolved
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

## Resolution

**The defect was in three places, not one, and `sfho` — named in the ticket as
the model to copy — was one of them.** `sfho` formats the condensation
rejection correctly in `eos_point` and NOT in `responses._solve`
(`eos/sfho/responses.py:41`), which raises the same "did not converge (residual
...)" sentence `did` did. `did` had it in both `api.py:99` and
`responses.py:36`. So fixing `did` alone would have left a sibling broken in
exactly the way the ticket predicted.

**The root-cause sweep, and its answer.** Two models in the repository clear
`converged` for a second, unrelated reason: `sfho/solver.py:617` and
`did/solver.py:574`, both the condensation gate. Nobody else does — `njl` and
`ccdm` return the best iterate of a failed candidate enumeration, which is
still the residual reason; `zl`, `vmit`, `alphabag`, `abpr`, `enjl` and `mixed`
set `converged` from the residual and nothing else. `dd2` and `mixed` refuse a
condensed gas too but by RAISING with their own message
(`dd2/solver.py:793`, `mixed/solver.py:676`), so they never had the defect and
are not touched.

**The message now lives once**, in `eos/general/thermal_mesons.py` beside the
quantity it names — the single home of the thermal meson gas machinery
(CLAUDE.md section 7) — as `condensation_message(condensation, n_B, T)`. The
wording is `dd2`'s, the one the ticket quotes as the model. Four call sites
consume it: `did/api.py`, `did/responses.py`, `sfho/api.py`,
`sfho/responses.py`.

`sfho/api.py`'s own message changed with it: it said "Bose-condenses **here**"
and named neither the density nor the temperature, which is the half of the
ticket's requirement copying `sfho` verbatim would have failed.

    did, n_B=0.4, T=10, thermal_mesons=True:
    ok=False  the thermal meson gas Bose-condenses at n_B=0.4 fm^-3, T=10 MeV
              (max |mu*|/m = 1.161); a condensate is not implemented, so this
              state is outside the model

and `eos_response` at the same state now reports "the response stencil needs a
converged neighbour and the thermal meson gas Bose-condenses at n_B=0.4004
fm^-3 ...". The residual message is reachable only by an actual residual miss.

**The notebook needed no edit.** `notebooks/hadronic_eos.py:182` prints
`result.message` verbatim, so its species-flag cell prints the new sentence
already — which is why nothing under `notebooks/` is in this commit.

### The domain question: `n_B <= 0.2` is not the domain, and the threshold moves

The ticket's `n_B <= 0.2` was the grid the notebook happened to try. The onset
was bisected to 1e-9 in `beta_eq_neutrinoless` with `thermal_mesons=True`:

    T [MeV]      nucleons   +hyperons   g_rho_N_N x 1.10
      5          0.30179    0.30179     0.29331
     10          0.30167    0.30169     0.29362
     20          0.30225    0.30502     0.29547
     30          0.31111    0.32978     0.30581
     50          0.36615    0.51396     0.36337

So it is a curve near 0.302 fm^-3, shallow up to T ~ 20 MeV and rising past it,
**and it moves with the parametrisation** — a 10% change in one isovector
coupling shifts it by 0.008 fm^-3, and turning the hyperons on moves it by 0.15
at T = 50. It is not a constant a caller can hardcode, which is the reason the
message names the density and the ratio rather than a fixed domain: an
inference run varying couplings crosses it at a different density every draw.
`did` ships one published parameter set (`DID` and `DIDY` are the same numbers),
so the coupling row is a perturbation of that set, not a second published fit.

### Gate

**0 added failures.** python.org **3.14.2** (`python3`), measured as an
**isolated PAIR** built with `git archive HEAD` (`508bab9`) plus one snapshot of
the gitignored `test/`: a control carrying HEAD only and a copy carrying HEAD
plus exactly this ticket's five files. The live tree is shared with three
notebook sessions, so a run in it has no before-image.

    control (HEAD)          6 failed, 277 passed   2:18
    mine    (HEAD + t66)    6 failed, 277 passed   2:51

Collected **283** over `test/did test/sfho test/baseline test/general`. The
failure sets are identical node id by node id: the six are
`test/baseline/test_baseline.py::test_baseline[{ccdm,dd2,enjl,njl,tov,zlvmit}]`,
the same six the map records at HEAD. `test/baseline/` for `did`, `sfho` and
every other model that passes at HEAD is **unmoved** — the diff is message text
and one new pure function, so no converged number is reachable from it.

`did`'s `verify/` suite reports **13/13 `[ok ]`** and `sfho`'s **8/8 `[ok ]`**,
no failures in either. `eos/general/` has no `verify/` suite to run — that is
[ticket 64](64-general-verify-suite-missing.md), still open.

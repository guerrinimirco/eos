# Is the acceleration mode-agnostic, or has it quietly become beta-eq-only?

Type: task
Status: open
Blocked by: 05, 06
Parent: ../map.md

## Question

The map may not close without answering this, and the reason is a specific
failure this repository has already had.

`build_fast_table` reached 18 ms/pt and looked entirely ordinary at
`fixed_YC` — and was **3-5% out in P**, because its Hermite spline needs
`dP/dmu_B = n_B` along the sweep, an identity only beta equilibrium delivers.
Measured `|dP/dmu_B / n_B - 1|`: beta_eq_neutrinoless 4.0e-4, `fixed_YC` 3.0e-2
leptonless and 4.2e-2 **with** leptons, beta_eq_neutrino_trapped 6.9e-2,
`fixed_YC_YS` 6.0e-2. Hence `FAST_MODES` and a raise.

Nothing in this map's route (pre-screen, compiled Newton loop) has any obvious
reason to be mode-dependent — a compiled loop does not care what the charge rows
say. **But "no obvious reason" is not a measurement**, and finding out late is
what the last effort did.

### What to measure

Re-run tickets 05 and 06's speedups, on the same grid, at:

- `fixed_YC` with `leptons=True` — the mode the previous fast path could not do,
  and one of the two the user named
- `fixed_YC` with `leptons=False` — what `eos/mixed` needs per pure phase
- `beta_eq_neutrinoless` at **T > 0** — pick a temperature where the thermal
  occupations genuinely matter, not 1 MeV
- `beta_eq_neutrino_trapped` if it costs little extra

For each: ms/pt, and the same-pattern / P-to-1e-8 gate against the unaccelerated
path in that same mode.

### What a failure here looks like, and what it means

- **Speedup holds, accuracy holds** — the route is mode-agnostic, the map can
  close, and the fog entry on finite-T and `fixed_YC` production paths stays
  a scheduling matter rather than a physics one.
- **Speedup shrinks in a mode** — say the pre-screen's abort threshold is tuned
  to beta-eq residual scales. Fixable, and this ticket names the fix.
- **Accuracy fails in a mode** — the serious case. It would mean something in
  the compiled loop depends on the charge rows, and ticket 09's verdict has to
  say so plainly rather than shipping a second `FAST_MODES`.

## Gate

- The table above, filled in, on the pinned benchmark stack.
- An explicit verdict sentence: mode-agnostic, or not, and if not, which mode
  and why.
- If a mode fails, it is recorded in `docs/DEFERRED.md` per CLAUDE.md section 3
  rather than silently left out.

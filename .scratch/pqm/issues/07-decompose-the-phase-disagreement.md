# What is the 22-42% phase disagreement actually made of?

Type: prototype
Status: open
Blocked by: 06
Parent: ../map.md

## Question

The model's pressure is close to target. Its **branch ordering is not**:

| mode | phase agreement with `eos.njl` |
|---|---|
| beta equilibrium | **58-78%** -- the weakest |
| `fixed_YC` | 65-95% |
| `Y_C` = 0.5, `Y_S` = 0 | 88-100% |

and the map makes >= 90% everywhere a **blocking gate**, because a model that
inverts the ordering is useless however small its residual
(`docs/csc_bag_mapping.md:464-467`).

**Do not try to fix it before knowing what it is.** There are three candidate
causes and they have three different -- in one case opposite -- responses:

### Candidate 1: gapless states the ansatz cannot represent

`eos.njl` carries gapless branches; `pqm` does not and structurally cannot. The
evidence that this matters here is specific rather than theoretical:

- `eos/njl/solver.py:474-476`: *"a gapless state is physical, but comparing
  candidates by Omega across one is not, so it is reported rather than silently
  ranked."*
- `docs/DEFERRED.md` records that giving the quadrature breakpoints at the
  quasiparticle zero-crossings (`eos.general.pairing.gapless_momenta`) is *"what
  made gapless CFL converge at T = 0 -- before it, a fixed-Y_C = 0.1 table
  reported a metastable 2SC at 98 of 100 points because the gapless CFL that
  minimised f could not reach the gate."*
- Gapless phases are exactly what occupies the Fermi-surface mismatch window,
  and beta equilibrium at these densities sits in that window -- which would
  explain why beta-eq is the WORST mode, not the best.

`th.fields` carries the flag and ticket 04 records it per row. **If NJL's
winner is gapless at a point, the disagreement there is not a fit error and no
amount of refitting will close it.** The response is a stated domain and a
`docs/DEFERRED.md` entry, not more terms.

### Candidate 2: genuine near-degeneracy

The memory of the earlier pass names this directly: *"the pressures of the
candidates differ by less than the fit error in some modes."* Where two
branches' pressures cross at a shallow angle, a 0.5% error in P moves the
crossing a long way in `n_B` while getting both pressures right. That is not a
ranking failure in any useful sense -- the physics itself is undetermined at
that resolution -- and the response is to REPORT the margin, not to chase it.

Measurable directly: at every disagreeing point, compute `|P_winner -
P_runner_up| / max|P|` in BOTH models and compare it to the local fit error.

### Candidate 3: ordinary fit error

The winner is wrong and the margin is wide. This is the only one of the three
that more terms, a better basis or a denser grid can fix, and it is the only
one ticket 08 should try to fix.

### What to build

A per-point decomposition over every benchmark slice and every NJL parameter
set. For each point where the model's min-Omega winner differs from
`eos.njl`'s `pattern_realised`:

- is NJL's winner flagged gapless?
- what is the margin `|P_1 - P_2|/max|P|` in NJL, and in the model?
- is the margin smaller than the local `|dP|/max|P|`?
- which pair of patterns is confused, and in which direction?

Then the three-way split, per mode: **gapless / degenerate / fit error**, as
percentages that add to the disagreement.

**Two traps in the comparison itself.** Read `pattern_realised`, never
`pattern` -- the solved-layout column is not stable, and `realised_pattern` can
return names outside the enumeration (uSC, dSC, usSC) when a CFL-layout solve
collapses, so never index a dict on it. And the comparison must be made at
matched conditions: NJL's `eos_table` ranks by `f = eps - T s` at fixed `n_B`
while a fixed-mu comparison ranks by `P`, and those disagree inside a
first-order window BY CONSTRUCTION (`eos/njl/solver.py:36-53`). Compare
like with like or the ticket will manufacture disagreement that is not there --
which is itself a fourth candidate worth ruling out first.

## Gate

- The three-way split -- gapless / degenerate / fit error -- as percentages
  per mode and per parameter set, adding to the measured disagreement.
- **A fourth number: how much of the apparent disagreement was the `f`-versus-`P`
  comparison artefact**, ruled out before the other three are believed.
- The confusion matrix: which pattern pairs are swapped, in which direction.
- **A verdict on whether >= 90% is reachable at all**, and if it is not, which
  slices it is not reachable on and why. That verdict sets ticket 08's scope,
  and if it says "not reachable because gapless", the map's blocking gate is
  renegotiated with the user rather than quietly missed.
- Whatever proves structural recorded for `docs/DEFERRED.md`.

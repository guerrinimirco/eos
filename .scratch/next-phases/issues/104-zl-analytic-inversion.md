# `zl.invert_nmp`: the closed form exists and was verified

Type: task
Status: open
Blocked by: -
Parent: ../map.md

## Question

`zl/nmp.py` says the inverse map is "deliberately absent" and `invert_nmp`
raises. **The closed form exists**, was supplied by the user from an earlier
derivation, and was re-derived and checked against this repository's own
forward map this session.

With kinetic parts subtracted (`Kb = K_sat - K_sat,K` etc.),
`X = Eb n0 + P_K,0` and `D = (9 Eb + Kb) n0^2 + 9 P_K,0 n0`:

    a0     = [Kb Eb n0^2 - 9 P_K,0 (Eb n0 + P_K,0)] / D
    b0     = 9 X^2 / D
    gamma  = -Kb n0 / (9 X)
    a1     = [3 g1 Sb - Lb + 3 a0 (g1 - 1) + 3 b0 (g1 - gamma)] / [3 (g1 - 1)]
    b1     = [3 Sb - Lb + 3 b0 (1 - gamma)] / [3 (1 - g1)]

with `gamma1 = g1` a FREE input (it tunes the high-density isovector
behaviour and no saturation-density observable constrains it).

Verified, target {n0 = 0.16, E_sat = -16, K_sat = 250, E_sym = 31.6,
L_sym = 43, gamma1 = 2.45}:

    recovered  a0=-96.6555  b0=58.8619  gamma=1.39854  a1=-25.1985  b1=7.1850
    published  a0=-96.64    b0=58.85    gamma=1.40     a1=-25.19    b1=7.18

and round-tripped through `zl.nmp.compute_nmp`: n_sat to 1.2e-14, E_sat to
5.6e-13, E_sym to 5.4e-07, K_sat and L_sym to 9.5e-03 and 6.4e-03 — the last
two being the forward map's own stencil, not the algebra.

**Why this matters beyond one model.** It makes `zl` the only model in the
repository whose inversion has no seed, no basin, no restart count and no
lottery — see [ticket 105](105-dd2-isoscalar-conditioning.md) and
[ticket 93](93-invert-nmp-basin-lottery.md) for what the alternative costs.
For an inference run that is a different class of object.

## Work

1. `zl/nmp.py::invert_nmp` and `from_nmp`, closed form, replacing the raise.
   `gamma1` an explicit argument, no default that hides the choice.
2. Kinetic pieces from the model's own code with the interaction switched off
   (`replace(par, a0=0, b0=0, a1=0, b1=0)`), not a second transcription of the
   Fermi integrals (§7).
3. **State the two conventions in the docstring, both of which silently give
   wrong answers if mixed:**
   - the rest-mass bookkeeping — `Eb = E_sat - E_sat,K` holds only when both
     are binding energies or both include the mass; the `+ m_H` in the
     original derivation belongs to the first reading;
   - which `E_sym`. This repository's `symmetry_energy` is the quadratic
     coefficient; Constantinou et al. use `E(n0, 0.5) - E(n0, 0)`. That is
     exactly why the shipped set carries `a1 = -26.06, b1 = 7.34` where the
     same NMPs through this inversion give `-25.19, 7.18`. A round-trip test
     that does not fix the convention will fail for the wrong reason.
4. **Two ZL-specific facts worth stating in `zl.md`/`.tex`**, both verified:
   - `Q_sat,V = 3 (gamma - 2) K_sat,V` exactly (checked to 6 digits,
     -1.804378 both ways). So Q_sat is NOT free in ZL: once
     {n_sat, E_sat, K_sat} fix gamma and b0, Q_sat is determined. A prior over
     (K_sat, Q_sat) in ZL lives on a curve, not in a plane.
   - the isovector sector has THREE knobs (a1, b1, gamma1), so ZL is the one
     model here that can impose K_sym, through
     `K_sym,V = 9[gamma1(gamma1-1) b1 - gamma(gamma-1) b0]`. Offer it as an
     option; leave gamma1 free by default.

## Gate

Round trip on the published set to the forward map's own stencil; the XOA
numbers above reproduced; `test/baseline/zl.npz` unmoved (this adds a
constructor, it changes no evaluation); one test asserting the Q_sat
rigidity, since it is the identity most likely to be broken by a later edit.

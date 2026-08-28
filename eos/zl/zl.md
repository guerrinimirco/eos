# ZL — the Zhao-Lattimer nucleonic density functional

`zl.tex` is the same description typeset, with the bibliography compiled
against `../../docs/eos.bib`. Both files carry the same physics: this one is not
a summary of that one, and neither defers to the other for an equation.

**Model.** Protons and neutrons as free Fermi gases at their vacuum mass, plus
a density-dependent interaction energy density. Zhao & Lattimer, PRD 102,
023021 (2020); the parameter set is the one used by Constantinou et al., PRD
104, 123032 (2021), PRD 107, 074013 (2023) and arXiv:2506.20418 (2025):

    V(n_p, n_n) = 4 n_p n_n [a0/n0 + b0/n0 u^(gamma-1)]
                + (n_n - n_p)^2 [a1/n0 + b1/n0 u^(gamma1-1)],   u = n_B/n0

There is no scalar field, so no gap equation and no effective mass: the whole
self-consistency of the model is between the densities and the interaction
potentials they generate,

    mu_Hv_i = dV/dn_i     mu_eff_i = mu_i - mu_Hv_i     n_i = n_i(mu_eff_i, T, m_i)

Differentiating `V` at the other density fixed, with `du/dn_i = 1/n0`:

    mu_Hv_p = 4 n_n [a0/n0 + (b0/n0) u^(gamma-1)]
            - 2 (n_n - n_p) [a1/n0 + (b1/n0) u^(gamma1-1)]
            + 4 b0 n_p n_n (gamma-1) u^(gamma-2) / n0^2
            + b1 (n_n - n_p)^2 (gamma1-1) u^(gamma1-2) / n0^2

    mu_Hv_n = 4 n_p [a0/n0 + (b0/n0) u^(gamma-1)]
            + 2 (n_n - n_p) [a1/n0 + (b1/n0) u^(gamma1-1)]
            + 4 b0 n_p n_n (gamma-1) u^(gamma-2) / n0^2
            + b1 (n_n - n_p)^2 (gamma1-1) u^(gamma1-2) / n0^2

The two differ only in their first two terms; the last two, which come from
differentiating the powers of `u = n_B/n0`, are common to both, because `u`
depends on the densities only through their sum.

Per baryon, with delta = (n_n-n_p)/n_B,

    V/n_B = (1-delta^2) [a0 u + b0 u^gamma] + delta^2 [a1 u + b1 u^gamma1]

so BOTH brackets enter the symmetry energy: the a0/b0 term is a proton-neutron
cross interaction, not an isoscalar one, and its potential part is
`(a1-a0) u + b1 u^gamma1 - b0 u^gamma`. Reading `a1+b1` as the potential
symmetry energy is the standard way to misread the functional, and gives the
wrong sign.

**Parameters.** Eight numbers, all fields of `Parameters` and arguments
everywhere — nothing in the package reads a module-level coupling.

| symbol | code | value | meaning |
|---|---|---|---|
| `n0` | `n0` | 0.16 fm^-3 | reference density of the functional |
| `a0` | `a0` | -96.64 MeV | cross term, linear in `u` |
| `b0` | `b0` | +58.85 MeV | cross term, proportional to `u^gamma` |
| `gamma` | `gamma` | 1.40 | exponent of the cross term |
| `a1` | `a1` | -26.06 MeV | isovector term, linear in `u` |
| `b1` | `b1` | +7.34 MeV | isovector term, proportional to `u^gamma1` |
| `gamma1` | `gamma1` | 2.45 | exponent of the isovector term |
| `m_p`, `m_n` | `m_p`, `m_n` | 939.5 MeV | nucleon masses |

`Parameters.default()` returns exactly this set — the one used in Constantinou
et al. (2021, 2023, 2025). Both nucleons carry the SAME mass: the model has no
isospin splitting of the kinetic term, and the asymmetry enters only through
`V`. And `n0` is a **parameter of the functional**, not the saturation density
the functional predicts; the two differ by 0.3 % (below).

**Three routes to a parameter set.** CLAUDE.md section 6 makes model
parameters arguments, so all three have to exist -- and one of the three is
refused here rather than written. *By name:* `Parameters.default()` is the
published set of Constantinou et al., and `Parameters.named('ZL_Constantinou')`
takes it by name. ZL ships exactly one set, so the map has a single entry; it
exists so that a caller sweeping parameter sets need not know which models
happen to have more than one. *A new set:* every field carries a default, so
`Parameters(a0=..., gamma=...)` names only what changes, and
`dataclasses.replace` modifies one already in hand. *From nuclear-matter
parameters:* `nmp.from_nmp` and `nmp.invert_nmp`, in **closed form** — the
interaction enters the nuclear-matter parameters linearly in `a0, b0, a1, b1`
once the two exponents are known, and the exponents come out of ratios of the
isoscalar data, so there is no seed, no basin and no restart count. Six
couplings against the five NMPs of the standard list still leaves one free
choice, and the caller names it: either `gamma1`, or a sixth datum `K_sym`.
`nmp.compute_nmp` is the forward direction.

**Single-nucleon thermodynamics.** Each species is a free Fermi gas of mass
`m_i` and degeneracy `g = 2` (spin), evaluated at its effective potential
`mu_eff_i`, antiparticles included. With `f(x) = 1/(1 + exp(x/T))` and
`E_k = sqrt(k^2 + m_i^2)`:

    n_i   = g/(2 pi^2 (hc)^3) int_0^inf dk k^2   [f(E_k - mu_eff_i) - f(E_k + mu_eff_i)]
    P_i   = g/(6 pi^2 (hc)^3) int_0^inf dk k^4/E_k [f(E_k - mu_eff_i) + f(E_k + mu_eff_i)]
    eps_i = g/(2 pi^2 (hc)^3) int_0^inf dk k^2 E_k [f(E_k - mu_eff_i) + f(E_k + mu_eff_i)]

The entropy density is NOT integrated separately. It comes from the Euler
relation of the free gas,

    s_i = (eps_i + P_i - mu_eff_i n_i) / T

which is exact for an ideal gas at the EFFECTIVE potential, and is the reason
the interaction never enters `s`.

At T = 0 the antiparticle terms vanish, the occupations become step functions at
`k_F_i = sqrt(mu_eff_i^2 - m_i^2)` (with `n_i = 0` when `mu_eff_i <= m_i`), and
these close in elementary functions, with `E_F_i = sqrt(k_F_i^2 + m_i^2) = mu_eff_i`:

    n_i   = g k_F_i^3 / (6 pi^2 (hc)^3)
    eps_i = g/(16 pi^2 (hc)^3) [ k_F_i (2 k_F_i^2 + m_i^2) E_F_i
                                 - m_i^4 ln((k_F_i + E_F_i)/m_i) ]
    P_i   = g/(48 pi^2 (hc)^3) [ k_F_i (2 k_F_i^2 - 3 m_i^2) E_F_i
                                 + 3 m_i^4 ln((k_F_i + E_F_i)/m_i) ]
    s_i   = 0

These integrals are not implemented in this subpackage: they come from
`eos.general.fermi_integrals`, which evaluates them through the
Johns-Ellis-Lattimer analytic approximation — uniformly valid from the
degenerate to the non-degenerate limit and exact at T = 0. The same module
supplies the inverse `mu_eff_i(n_i, T, m_i)` used by the density-first entry
point. They are written out here anyway, because a paper-style description is
self-contained.

**There is no scalar density `n_s`.** Every model with an effective mass returns
one through `n_s = (eps - 3P)/m*`; ZL has no scalar field and no `m*`, so the
quantity is not defined and is not returned. That is an absence of the physics,
not an omission.

**The totals.** Summed over the active species, plus leptons and photons where
a mode includes them:

    eps = sum_i eps_i + V              P = sum_i P_i + P_int
    s   = sum_i s_i                    (V carries no T)

and the Euler relation `eps + P = T s + sum_i mu_i n_i` holds identically at the
physical potentials, which is what `verify/` checks.

**The interaction** adds

    eps_int = V                                       (the functional itself)
    P_int   = sum_i n_i mu_Hv_i - V
            = 4 n_p n_n [a0/n0 + gamma b0/n0 u^(gamma-1)]
            + (n_n-n_p)^2 [a1/n0 + gamma1 b1/n0 u^(gamma1-1)]
    s_int   = 0                                       (V carries no T)

The power-law pieces of `P_int` pick up the factors `gamma`, `gamma1` and the
linear pieces do not; that identity is the numerical content of "V is a
functional of the densities and nothing else", and `verify/` checks it. There
is no rearrangement term to place separately: `mu_Hv_i` is the full derivative,
so the Euler relation `eps + P = T s + sum_i mu_i n_i` holds identically at the
physical potentials.

**Charges.** `n_B = n_p + n_n`, `n_C = n_p`, `n_S = 0`; `mu_B = mu_n`,
`mu_C = mu_p - mu_n`, and `mu_S` is reported as zero by convention — no
equation of any mode responds to it.

## Equilibrium modes and their residuals

Every mode imposes the two self-consistency equations
`n_i(mu_eff_i, T, m_i) = n_i`. The unknown vector always carries the physical
potentials, and carries `(n_p, n_n)` as well wherever the composition is not
fixed in advance. Keeping the densities as unknowns rather than substituting
the self-consistency into `mu_Hv_i(n_p, n_n)` is a conditioning choice: it
keeps the residual polynomial in the interaction potentials instead of nesting
Fermi integrals inside them. It is not a statement about the state, which is
`(mu_p, mu_n, T)` with `mu_Hv_i` playing the part a mean field plays elsewhere.

**`beta_eq_neutrinoless`**, conditions `(n_B, T)` —
`x = [mu_p, mu_n, mu_e, n_p, n_n]`, five rows in the order assembled:

    r1 = n_p(mu_eff_p, T, m_p) - n_p
    r2 = n_n(mu_eff_n, T, m_n) - n_n
    r3 = n_p + n_n - n_B
    r4 = mu_n - mu_p - mu_e          (= -mu_C - mu_e)
    r5 = n_C - n_e(mu_e, T)          (= n_p - n_e)

    scales  (n_B, n_B, n_B, mu_B, n_B)

`r4` is beta equilibrium with free-streaming neutrinos (`mu_nue = 0`), `r5`
total electric neutrality.

**`beta_eq_neutrino_trapped`**, conditions `(n_B, Y_Le, T)` —
`x = [mu_p, mu_n, mu_e, mu_nue, n_p, n_n]`, six rows: `r1, r2, r3, r5`
unchanged, with

    r4 = mu_n - mu_p - mu_e + mu_nue      (= -mu_C - mu_e + mu_nue)
    r6 = (n_e + n_nue)/n_B - Y_Le

    scales  (n_B, n_B, n_B, mu_B, n_B, 1)

— `r6` is already dimensionless. The muon family is not tracked, so `Y_Lmu`
raises.

**`fixed_YC`**, conditions `(n_B, Y_C, T)`. Here the composition is known
before the solve — `n_p = Y_C n_B`, `n_n = (1 - Y_C) n_B` — so both densities
leave the unknown vector and rows `r3` and `r4` are not needed:

    leptons=True    x = [mu_p, mu_n, mu_e]   rows r1, r2, r5'
    leptons=False   x = [mu_p, mu_n]         rows r1, r2

    r5' = n_e(mu_e, T) - n_C

    scales  (n_B, ...) — every row of this mode balances a density

**The neutrality row's SIGN differs between modes in the code**: it is
`n_C - n_e` in the two beta-equilibrium modes and `n_e - n_C` in `fixed_YC`.
The root is of course unchanged; the residual is not, so each mode's row is
given here as the code assembles it rather than picking one spelling and
implying the other matches.

**`fixed_YC_YS`** raises `NotImplementedError`. The mode is not unimplemented
but meaningless here: `n_S = 0` for any state of the model, so the only `Y_S`
it could satisfy is zero and any other request has no solution. Silently
ignoring `Y_S` would return symmetric nuclear matter under a name that promised
a strangeness condition.

`leptons=True/False` applies to `fixed_YC`: without leptons the result is
charged nucleonic matter, which is what a mixed phase needs per pure phase
before global neutrality is imposed. Photons are a separate flag. Entropy per
baryon may replace `T`, through an outer 1-D solve.

**Nuclear-matter parameters.** `nmp.compute_nmp(par)` is the forward map, at
T = 0. Saturation is the `P = 0` root of symmetric matter and is NOT `n0`, the
functional's reference density. Every value is a prediction — ZL imposes no
saturation condition:

    n_sat = 0.15951 fm^-3   E_sat = -15.99648   K_sat = 250.174
    E_sym = 30.84803        L_sym =  41.27034
    Q_sat = -352.9          K_sym =  -88.5 +/- 0.2      (all MeV)

The first five are the published set of Constantinou et al. and are pinned in
`verify/run_full_check.py` at the published precision. `Q_sat` and `K_sym` are
reported but not pinned: they are not in the published set, and a check cannot
assert a number nobody published.

`K_sym` is quoted to one decimal on purpose. It is a SECOND density derivative
of a quantity that is itself a second derivative in `beta`, and the estimate
drifts with the step — -88.47, -88.42, -88.46, -88.60 for `h/n_sat` = 0.05,
0.02, 0.01, 0.005 — as round-off starts to dominate before truncation has
finished falling. So it is determined to roughly +/- 0.2, and any four-digit
value for it, in this document or elsewhere, claims a precision the finite
difference does not have. `L_sym`, one derivative lower, converges cleanly
(41.307, 41.270, 41.265, 41.264 over the same steps).

`S(n)` is the curvature at `beta = 0`, `S = (1/2) d^2(E/A)/d beta^2`, which is
the definition the published numbers use. Note that `eos.did.nmp` estimates the
same coefficient with a full step to pure neutron matter and a Richardson
correction; that route gives 30.776 and 41.124 here, carrying `beta^4`
contamination the published values do not include. The difference is real, not
numerical: DID needs the full step because its `E/A` difference at small
asymmetry sits in noise, and ZL's does not.

**The inverse map, in closed form.** ZL has six couplings — `a0, b0, gamma,
a1, b1, gamma1` — against the five NMPs of the standard list, so imposing that
list leaves one free choice. `invert_nmp` makes the caller name it and then
solves by algebra, not by a root find. Writing `Eb = E_sat - E_sat,K` and so on
for each interaction remainder (the total minus its free-Fermi-gas part at
`n0`, both rest-mass-subtracted), and setting the functional's `n0` equal to
the requested `n_sat` so that saturation is imposed at `u = 1`:

    X      = Eb n0 + P_K,0
    D      = (9 Eb + Kb) n0^2 + 9 P_K,0 n0
    gamma  = -Kb n0 / (9 X)
    b0     = 9 X^2 / D
    a0     = [Kb Eb n0^2 - 9 P_K,0 (Eb n0 + P_K,0)] / D
    b1     = [3 Sb - Lb + 3 b0 (1 - gamma)] / [3 (1 - gamma1)]
    a1     = [3 gamma1 Sb - Lb + 3 a0 (gamma1 - 1)
              + 3 b0 (gamma1 - gamma)] / [3 (gamma1 - 1)]

which is Eqs. (nmp-pot) read backwards: `a0 + b0 = Eb`, `n0 (a0 + gamma b0) =
-P_K,0` (the total pressure vanishes), `9 gamma (gamma - 1) b0 = Kb`, and the
two `delta^2` conditions. For `{n_sat = 0.16, E_sat = -16, K_sat = 250,
E_sym = 31.6, L_sym = 43}` with `gamma1 = 2.45` it returns `a0 = -96.6555,
b0 = 58.8619, gamma = 1.39854, a1 = -25.1985, b1 = 7.1850`, and the round trip
through `compute_nmp` returns `n_sat` to 1e-14, `E_sat` to 6e-13 and `E_sym` to
5e-7, with `K_sat` and `L_sym` to 1e-2 — the forward map's own stencil, not the
algebra.

**Two conventions decide whether that round trip means anything.** First, the
rest mass: every remainder is a difference between a total and its kinetic
part, which is the interaction piece only when both sides are binding energies
(they are here — `energy_per_baryon` subtracts `m_p n_p + m_n n_n`), so a
derivation carrying an explicit `+ m_H` is reading the same symbol the other
way. Second, **which `E_sym`**: this repository uses the quadratic coefficient
and Constantinou et al. use the full PNM − SNM step. On the shipped set those
read `30.848 / 41.270` and `31.561 / 42.718` — one functional, two conventions.
The familiar target `{31.6, 43}` is therefore the shipped set stated in the
*second* convention, which is exactly why putting it through this inversion
gives `a1 = -25.20, b1 = 7.19` where the shipped set carries `-26.06` and
`7.34`. Neither is wrong; they answer different questions.

**`Q_sat` is not free in ZL.** The interaction carries one power-law term, so
its skewness and its incompressibility come from the same `b0 u^gamma`:

    Q_sat,pot = 27 gamma (gamma - 1) (gamma - 2) b0
              = 3 (gamma - 2) K_sat,pot        exactly.

Once `{n_sat, E_sat, K_sat}` fix `gamma` and `b0`, `Q_sat` follows — so it
cannot be the sixth datum, and a prior over `(K_sat, Q_sat)` in ZL lives on a
curve rather than in a plane. Nothing isovector reaches `gamma` or `b0`, which
is how the identity is pinned in `test/zl/test_zl_nmp.py`: moving `E_sym`,
`L_sym` and `gamma1` leaves the predicted `Q_sat` bit-identical.

**`K_sym` can be imposed, and only here.** The isovector sector has THREE knobs
— `a1, b1, gamma1` — against three isovector data, so with

    K_sym,pot = 9 [gamma1 (gamma1 - 1) b1 - gamma (gamma - 1) b0]

`gamma1` is determined rather than chosen, again in closed form: with
`(gamma1 - 1) b1 = Lb/3 - Sb - X/n0` fixed by `E_sym` and `L_sym` alone,

    gamma1 = [K_sym,pot / 9 + gamma (gamma - 1) b0] / [(gamma1 - 1) b1] .

ZL is the one model in this repository that can do this; every other hadronic
model has two isovector knobs and reports `K_sym` as a prediction. It is
offered rather than imposed — pass `K_sym` in the NMP dict *instead of*
`gamma1`, never both.

## What a solved point returns

A solve returns an `EoSPoint` carrying every quantity below; nothing is omitted
because nothing downstream consumes it.

| field | symbol | unit / meaning |
|---|---|---|
| `converged` | — | the status a caller must test first |
| `error` | — | largest scaled residual, dimensionless |
| `n_B`, `T` | n_B, T | fm^-3, MeV: the conditions |
| `Y_C`, `Y_S`, `Y_L` | Y_C, Y_S, Y_Le | the fractions; `Y_S = 0` always |
| `mu_p`, `mu_n` | mu_p, mu_n | MeV, *physical* potentials (not `mu_eff_i`) |
| `mu_e`, `mu_nu` | mu_e, mu_nue | MeV |
| `mu_B`, `mu_C`, `mu_S`, `mu_L` | mu_B, mu_C, mu_S, mu_L | MeV; `mu_S = 0` by convention |
| `n_p`, `n_n`, `n_e`, `n_nu` | n_p, n_n, n_e, n_nue | fm^-3 |
| `P_total` | P | MeV/fm^3 |
| `e_total` | eps | MeV/fm^3 |
| `s_total` | s | fm^-3 |
| `Y_p`, `Y_n`, `Y_e` | n_i/n_B | per-species fractions |

When `converged` is False every other field holds the best iterate reached,
which is not a physical state.

**`s`** is the sum of the sectors' entropy densities, each obtained through the
free-gas Euler identity at that sector's EFFECTIVE potential. The interaction
never enters `s`, because `V` depends on the densities and not on `T`. The
model therefore does not compute `s` from the total Euler relation, which is
what leaves that relation available as an independent check.

**There is no scalar density `n_s`** — see above. This is the one
returned-quantity requirement the model discharges by not having the quantity.

`eos_table` flattens each solved point into the long-format row
`eos.general.table_io` writes, keyed so a nucleonic table and a hybrid table
concatenate without renaming.

## The API surface

Three entry points, the same three every model exposes, with the parameters
always first and the mode always required:

- `eos_point(par, mode, species, n_B=, T= | SnB=, leptons=, **conditions)` —
  one state, returned inside a `PointResult` carrying `ok`, a `message` and the
  `point`. Exactly one of `T` and `SnB`.
- `eos_table(par, mode, species, axes, fixed=, leptons=, progress=, verbose=)`
  — a solved grid over `{n_B} x {T} x` the fraction axes the mode fixes, the
  density axis warm-started. `progress` is called once per completed line with
  `{mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
  elapsed_s}`, the same dictionary in every model. Deep solver code never
  prints.
- `eos_response(par, mode, species, frozen='equilibrium', n_B=, T=, leptons=,
  rel_step=, **conditions)` — second derivatives.

`eos_response` implements `frozen='equilibrium'` and nothing else. Both
quantities are central differences over a relative step `rel_step` (default
1e-3) in the variable differentiated:

    cs2_isothermal = (dP/deps)_T             always
    C_V            = (T/n_B) (ds/dT)_n_B     at T > 0 only

The adiabatic speed, larger by `C_P/C_V` at T > 0, is not computed by this
model: `C_P` is not among the returned quantities, so there is no factor to
form it with.

Every other freeze — frozen composition, frozen conserved fractions, the
leptonic re-neutralization variants — and the susceptibility matrix `chi_ab`
raise `NotImplementedError` naming the gap.

## The `verify/` suite

Ten physics invariants, each returning a structured pass/fail with the largest
error it saw:

1. **Euler relation** `eps + P = T s + sum_i mu_i n_i`, at 1e-8 relative.
2. **Free energy** `f = eps - Ts = -P + sum_i mu_i n_i`, at 1e-8.
3. **Interaction identities**: the two expressions for `P_int` agree, and
   `mu_Hv_i` is the numerical derivative of `V`.
4. **Mode closures**: each mode's own conditions at its own solution, at 1e-8.
5. **Free-gas limit**: with `a0 = b0 = a1 = b1 = 0` the solved state is two
   free Fermi gases at the physical potentials, at 1e-10.
6. **Isospin symmetry**: at `Y_C = 0.5` symmetric matter comes back with
   `mu_p = mu_n`, i.e. `mu_C = 0`, and `n_p = n_n`, at 1e-10. Both nucleons
   carry the same mass and `V` is symmetric under `n_p <-> n_n`, so this must
   hold exactly; it catches a sign slip in the isovector term, which is
   otherwise invisible in the totals.
7. **Residual gate**: every state the suite solved is inside 1e-10.
8. **Causality**: `0 <= c_s^2 <= 1` along the cold beta-equilibrium sequence.
9. **No strangeness**: `n_S = 0` and `mu_S = 0` in every returned state, and
   `fixed_YC_YS` raises.
10. **Nuclear-matter parameters**: the five published values reproduce at the
    precision they are published to.

**Numerics.** Two to six equations, Powell hybrid then Levenberg-Marquardt,
with one further cold-start attempt when a warm start was supplied and failed.
Convergence is judged on a dimensionless residual — density and charge rows
divided by `n_B`, potential equalities by `mu_B` — gated on the largest scaled
component at 1e-10. Non-convergence is a return value, not an exception.

**Not implemented** (see `docs/DEFERRED.md`): muons, hyperons, deltas, thermal
mesons, thermal neutrinos, `fixed_YC_YS`, the NMP inversion (above), and the
freezes of `eos_response` beyond `equilibrium`.

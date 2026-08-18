# ABPR — the analytic colour-flavour locked parametrization at T = 0

The full description, with equations and bibliography, is `abpr.tex`
(compiled against `../../docs/eos.bib`). This file is the plain-text summary.

**Model.** Colour-flavour locked (CFL) quark matter at zero temperature, as
the closed-form pressure of Alford, Braby, Paris and Reddy, ApJ 629, 969
(2005). The condensate locks the three flavour densities together,
`n_u = n_d = n_s`, so the composition is fixed and the phase has a single
independent potential: the common quark chemical potential `mu = mu_B/3`.

    P(mu)   = 3 a4 mu^4/(4 pi^2 (hc)^3)          free gas + leading pQCD
              - 3 ms^2 mu^2/(4 pi^2 (hc)^3)      strange mass, to O(ms^2)
              + 3 Delta^2 mu^2/(pi^2 (hc)^3)     CFL condensation energy
              - B/(hc)^3                         the bag

    n_B     = dP/dmu_B = dP/dmu / 3 = a4 mu^3/(pi^2 (hc)^3)
                                    + 2 (Delta^2 - ms^2/4) mu/(pi^2 (hc)^3)
    eps     = -P + mu_B n_B                      the Euler relation itself
    s       = 0                                  T = 0
    f       = eps - T s = eps
    c_s^2   = (2 A mu^2 + C)/(6 A mu^2 + C) -> 1/3

with `A = 3 a4/(4 pi^2 (hc)^3)` and `C = 3(Delta^2 - ms^2/4)/(pi^2 (hc)^3)`.
The bag enters `eps` and `P` with opposite signs and neither `s` nor any
`n_q`, so it cancels out of `eps + P` and the Euler relation
`eps + P = T s + sum_q mu_q n_q = mu_B n_B` carries no bag term.

**Parameters** (`Parameters`, a frozen dataclass; all four are arguments):

    ms    = 150 MeV    strange current quark mass
    Delta =  80 MeV    CFL pairing gap, constant (there is no T here)
    a4    = 0.7        pQCD factor;  alpha_s = pi/2 (1 - a4) = 0.4712
    B4    = 135 MeV    bag constant B^(1/4);  B/(hc)^3 = 43.23 MeV/fm^3

`Parameters.B` returns B in MeV^4 — the same unit, for the same attribute, as
`eos.alphabag.Parameters.B` and `eos.vmit.Parameters.B` — and the one division
by `(hc)^3` happens where the pressure is assembled. There is no published
single ABPR set: the four numbers span a range a hybrid study scans, and the
values above are `Parameters.default()`, the set the numerical baseline is
frozen at.

**Charges.** With `n_u = n_d = n_s = n_B` and `S = +1` per s quark (this
repository's convention, the opposite of the PDG's):

    n_C = (2 n_u - n_d - n_s)/3 = 0        ->  Y_C = 0   identically
    n_S = n_s = n_B                        ->  Y_S = +1  identically
    mu_C = mu_u - mu_d = 0 ,  mu_S = mu_s - mu_d = 0 ,  mu_B = 3 mu

The phase is electrically neutral by construction and carries no leptons of
any kind. `mu_S = 0` is a choice of this parametrization rather than a
property of CFL matter: locking equal densities at unequal masses needs
unequal potentials, and `eos/alphabag` solves for exactly that. Here the
difference is absorbed into the `-3 ms^2 mu^2/(4 pi^2)` term.

**Modes.** One mode, `cfl`, taking `(n_B, T = 0)` and closed by flavour
locking. The gap is a parameter, not a per-call condition (unlike
`eos/alphabag`, where it selects between two phases of one potential).

The four repository modes each raise, naming the physics:

    beta_eq_neutrinoless        locking already fixed the composition and left
                                mu_C = 0 with no electrons; the beta condition
                                has no free variable. Use eos/alphabag or
                                eos/vmit for unpaired beta-equilibrated matter.
    beta_eq_neutrino_trapped    the same, and there are no leptons for Y_Le to
                                fix.
    fixed_YC                    Y_C = 0 identically; any other value has no
                                state, and Y_C = 0 IS the cfl mode.
    fixed_YC_YS                 the same, and Y_S = +1 identically. Both are
                                outputs of the closure, not inputs to it.

`T > 0` raises (that is `eos/alphabag`'s `cfl` mode), and every species flag
raises when set: the hadronic sectors are meaningless in a deconfined phase,
there are no leptons, and the thermal sectors are identically zero at T = 0.

**Inverses.** All three are closed forms, so the model iterates nowhere.
`n_B(mu)` is a depressed cubic solved by Cardano, in the numerically stable
form `mu = u - p/(3u)`; `P(mu)` and `eps(mu)` are quadratics in `mu^2`. Over
the baseline's targets they agree with the `scipy.optimize.root` solves they
replaced to 2.5e-13 relative, inside the 1e-10 the baseline is frozen at.

**The P = 0 surface.** A self-bound phase ends at finite density with no
crust. At the shipped set the surface is `mu_0 = 277.195 MeV`,
`n_B = 0.2023 fm^-3`, `eps = 168.21 MeV/fm^3` and `E/A = mu_B = 831.58 MeV` —
below the 930 MeV of Fe-56, so the set describes absolutely stable strange
quark matter. The pairing term is what buys that: at `Delta = 0` with
everything else unchanged, `E/A = 932.94 MeV`.

**Relation to `eos/alphabag`.** This model is the T = 0 analytic limit of that
package's CFL phase, and the parameters map exactly (`alpha_s = pi/2 (1 - a4)`,
`Delta -> Delta0`, `m_u = m_d = 0`). They are not the same expression:
`eos/alphabag` carries `m_s` exactly through the Fermi integrals, this model
carries it as the `O(ms^2)` term. The whole gap between them is therefore the
`ms^4` term of the expansion,

    dP = P_abpr(mu) - P_alphabag_cfl(mu)
       ~ -ms^4/(8 pi^2 (hc)^3) [9/4 + 3 ln(2 mu/ms)]  + O(ms^6/mu^2)

Measured at the shipped set, at three equal potentials:

    mu [MeV]    350      400      500      600      700      800
    dP          -5.694   -6.038   -6.608   -7.070   -7.460   -7.796
    dP/P        -8.1e-2  -4.2e-2  -1.6e-2  -8.1e-3  -4.5e-3  -2.8e-3
    ratio to    0.9931   0.9950   0.9971   0.9981   0.9987   0.9991
    the formula

so the two agree to exactly the order at which they are supposed to differ.
`verify/run_full_check.py` asserts the last row to within 1%. Compared at
matched n_B — the way a table pairs them — `dP/P` runs from -7.9e-2 to
-2.8e-3 and `deps/eps` from 1.3e-2 to 1.1e-3 over
`n_B = 0.3 -> 3 fm^-3`.

**Layout.**

    parameters.py       Parameters, its default set, alpha_s and B
    species.py          SpeciesFlags -- every sector off, and setting one raises
    thermodynamics.py   P, n_B, eps, s, f, c_s^2 from mu
    solver.py           the inverse maps, solve_cfl, CFLPoint, the modes
    api.py              eos_point / eos_table / eos_response
    verify/             the invariants above, one entry point

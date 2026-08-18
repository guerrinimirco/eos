# ENJL — the extended Nambu-Jona-Lasinio model of dense matter at T = 0

The full description, with every equation and the bibliography, is `enjl.tex`
(compiled against `../../docs/eos.bib`). This file is the plain-text summary.

**Model.** Baryons and quarks from ONE functional, as in Xia, PRD 110, 014022
(2024) [arXiv:2405.02946]. A baryon is a three-quark cluster whose mass is
built from the same constituent masses the NJL gap equation determines, so the
chiral, quarkyonic and deconfinement transitions all come out of a single mean
field rather than out of two models joined at a boundary. Density-dependent
couplings, hence rearrangement self-energies. The NJL three-momentum cut-off
enters as a T-independent vacuum subtraction on the quark sector alone.

**Species.** p, n, Lambda (g = 2, B = 1); u, d, s (g = 6, B = 1/3); e, mu
(g = 2). Fixed by the model, not configurable. Repo conventions: S = +1 per s
quark (so Lambda and s both carry +1), and C excludes leptons.

**The self-consistency** is the three constituent masses and nothing else;
everything below them is algebraic. With `N^q_i` the valence content of baryon
i, `x = kF/M`, `y = Lambda/M`:

    n           = g kF^3/(6 pi^2)
    n^s_medium  = (g M^3/4 pi^2)  [ x(x^2+1)^1/2 - asinh x ]
    eps_medium  = (g M^4/16 pi^2) [ x(2x^2+1)(x^2+1)^1/2 - asinh x ]
    n^s         = n^s_medium - (g M^3/4 pi^2) [ y(y^2+1)^1/2 - asinh y ]
    eps         = eps_medium - (g M^4/16 pi^2)[ y(2y^2+1)(y^2+1)^1/2 - asinh y]
    s           = 0                                            T = 0

    nbar^s_q    = min( n^s_q + alpha_S sum_i N^q_i n^s_i , 0 )          Eq. (6)
    M_q         = m_q0 - 4 G_S nbar^s_q + 2 K nbar^s_q' nbar^s_q''      Eq. (5)
    M_i         = sum_q N^q_i [ m_q0 + alpha_S (M_q - m_q0) ] + B n_B^Q Eq. (4)

    J_omega     = sum_i f_i N_i n_i     N_i = 3 (baryons), 1 (quarks)
    J_rho       = sum_i f_i tau_i n_i
    g_omega w   = Gamma_omega(n_B) J_omega,  g_rho r = Gamma_rho(n_B) J_rho

    Sigma^R_b   = 1/2 Gamma_omega' J_omega^2 + 1/2 Gamma_rho' J_rho^2
                + alpha_S' sum_i [ sum_q N^q_i (M_q - m_q0) ] n^s_i    Eq. (17)
    Sigma^R_q   = 1/3 B sum_i n^s_i + 1/3 Sigma^R_b                    Eq. (18)

    mu_i        = nu_i + f_i (3 g_omega w + tau_i g_rho r) + Sigma^R_b  baryons
    mu_q        = nu_q + f_q (  g_omega w + tau_q g_rho r) + Sigma^R_q  quarks
    mu_l        = nu_l                                                  leptons

    eps         = sum_i eps_i + 2 G_S sum_q (nbar^s_q)^2
                - 4 K nbar^s_u nbar^s_d nbar^s_s
                + 1/2 Gamma_omega J_omega^2 + 1/2 Gamma_rho J_rho^2 - E0
    P           = T s + sum_i mu_i n_i - eps                            Eq. (19)
    f           = eps - T s = eps = -P + sum_i mu_i n_i

Sigma^R is in every mu_i and hence in P, and NEVER in eps. That is checked
directly, as `mu_i = d eps / d n_i`, by `verify/run_full_check.py`: it holds to
1.8e-6 MeV everywhere the functional is smooth. The one exception is the cap in
Eq. (6) — see DEFERRED.

**The cut-off enters exactly one way** and it matters: only the VACUUM
subtraction carries Lambda, never the medium integral. The quark kinetic
potential exceeds Lambda = 602.3 MeV above n_B ~ 3 fm^-3, so cutting the medium
integral would truncate the physical Fermi sea. Because the vacuum terms depend
on (M, g, Lambda) alone, they are independent of both kF and T — which is what
makes `thermodynamics.kinetic_thermo` the single seam a finite-temperature
extension touches.

**The 't Hooft term** is written over the OTHER two flavours,
`2 K nbar_q' nbar_q''`, not as the paper's `2 K nbar_u nbar_d nbar_s / nbar_q`.
Same number wherever nbar_q is nonzero, no 0/0 at chiral restoration.

**Parameters** (`Parameters`, a frozen dataclass; every number an argument):

    RKH set (Rehberg, Klevansky, Huefner, PRC 53, 410 (1996)):
      Lambda = 602.3 MeV   m_u0 = m_d0 = 5.5   m_s0 = 140.7 MeV
      G_S = 1.835/Lambda^2   K = 12.36/Lambda^5

    Table I of the paper:
      alpha_S  : aS = 0.4413715,  bS = 0.4076285,  nS = 0.16  fm^-3
      Gamma_w  : aV = 3.566049,   bV = 1.062771,   nV = 0.214 fm^-3
      Gamma_r  : aTV = 0.5014459, bTV = 0.0117601, nTV = 0.1  fm^-3

      f_Lambda = 1.0626 (fixed by U_Lambda(n_sat) = -30 MeV)
      f_q in {0.5, 0.7, 1.0}   B in {0, 1} GeV/fm^3

`Parameters.default()` is (f_q = 0.5, B = 1), the set of the paper's Figs. 4-6
and the one `test/baseline` is frozen at; `Parameters.named("fq1.0_B0")` and
its five siblings are the study's other combinations, named as the author's
own tables are.

**Gamma_rho carries a factor 9** that the printed Eq. (22) does not. It is
required by the isospin source used here and is confirmed twice: the published
symmetry energies come out (25.50 and 31.55 MeV measured, against 25.5 and
31.5) where the literal Eq. (22) gives 13.3 and 20.2; and reading g_rho rho off
the tables' isospin splitting gives exactly 9.0000x Eq. (22) at every nucleonic
density, with no fit. It is `parameters.RHO_FACTOR`. Do not "fix" it back.

**Modes.** One is closed; three raise naming the physics, and so does any
T > 0.

| mode | status |
|---|---|
| `beta_eq_neutrinoless` | closed at T = 0: `mu_i = B_i mu_b - q_i mu_e` and `sum_i q_i n_i = 0`, ten unknowns |
| `beta_eq_neutrino_trapped` | raises — no neutrinos in the species set, and Eq. (23) has mu_nu = 0 built in |
| `fixed_YC` | raises — cheap with `leptons=False`, needs an extra unknown with `leptons=True`, and half a mode is not a mode |
| `fixed_YC_YS` | raises — mu_S = 0 identically here, so Y_S is an output; imposing it means promoting mu_S to an unknown |

`thermodynamics.thermo_from_n` is NOT a fifth mode: it is handed all eight
densities and determines nothing about the composition, solving only the gap
equation. It is the "block at given densities" of the shared vocabulary, and
it is how the paper's Figs. 1-3 are evaluated.

**The beta-equilibrium residual**, ten rows in the order the code assembles
them, on the unknowns
`(M_u, M_d, M_s, mu_b, mu_e, n_B^Q, g_omega w, g_rho r, Sigma^R_b, Sigma^R_q)`:

    1-3   M_q - gap_q(nbar^s)                       scale 100 MeV
    4     sum_i B_i n_i - n_B_target                scale n_B
    5     sum_i q_i n_i                             scale n_B   (leptons in)
    6     n_B^Q - (n_u + n_d + n_s)/3               scale n_B
    7     g_omega w - Gamma_omega J_omega           scale 3 Gamma_omega n_B
    8     g_rho r   - Gamma_rho   J_rho             scale Gamma_rho n_B
    9     Sigma^R_b - Sigma^R_b[x]                  scale 3000 MeV
    10    Sigma^R_q - Sigma^R_q[x]                  scale 1000 MeV

Accepted on the largest scaled residual at `general.solve.RESIDUAL_TOL` = 1e-10;
the shipped grid reaches 8.8e-13 worst, 4.0e-15 median. Bounded least squares
rather than the shared `solve_system`: the box is what keeps the iteration out
of the unphysical regions an unbounded root find walks into at these scales.

**Tables are continuations, not phase diagrams.** Each point is warm-started
from its neighbour, so a sweep follows one branch past any first-order
transition into the metastable region beyond, and `direction="up"/"down"`
selects the branch rather than merely the loop order. Where several branches
exist the two differ, and that difference IS the branch structure. Choosing
between them is a Maxwell construction, which needs both branches at once and
is not implemented. So a raw branch may violate dP/dn_B >= 0; that is real
physics and must be resolved by a construction before the table reaches TOV.

**Measured against the paper** (fixed composition, central differences):

    n_sat  = 0.158297 fm^-3   (0.158)      E_sym(n_sat) = 31.549 MeV  (31.5)
    E/A    = -16.010 MeV      (-16.0)      L_sym        = 42.35 MeV   (42.4)
    K_sat  = 234.20 MeV       (234.5)      E_sym(0.1)   = 25.500 MeV  (25.5)
    M_N(0.16) = 519.23 MeV    (519.2)      U_Lambda(n_sat) = -30.020  (-30)
    vacuum: M_u = M_d = 367.648, M_s = 549.479 MeV  (367.6, 549.5)
            M_N = 938.89, M_Lambda = 1113.68 MeV    (938.9, 1113.7)
            E0 = -4263.8455 MeV/fm^3, identical across all six parameter sets

**Measured against `eos/general`.** The free-gas parts come from
`general.fermi_integrals`; the local closed forms they replaced agreed with
them to 1e-12 or better everywhere a species is populated enough to matter
(kF >= 50 MeV), and routing through them moved no frozen quantity of
`test/baseline` by more than 1.5e-12 relative against a 1e-10 gate. The one
loose entry is the pressure at kF ~ 1 MeV, where both forms are a cancellation
of two nearly equal terms; there the shared closed form is the less accurate
of the two, and the species carrying it has a density of 4e-9 fm^-3.

**Not implemented.** Finite temperature; the Maxwell construction and the
branch rule; `eos_response`; the three modes above. All are recorded in
`docs/DEFERRED.md` with what closing each would take.

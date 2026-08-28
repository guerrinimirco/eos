# The DD2 isoscalar closure is degenerate in the shape coefficients, and the
# coordinates are why

Type: grilling
Status: open
Blocked by: -   (93 is a sibling: this is the closure, 93 is the solver's verdict)
Parent: ../map.md

## Question

Raised by the user (2026-08-28): can the 6+2 closure be made well conditioned
rather than abandoned, and if not, which couplings should stay free?

**Measured this session, and it changes the diagnosis the map has carried.**
Isoscalar Jacobian d(NMP)/d(coupling) at the published DD2 point, central
differences, rows scaled by a physical size and columns by the coupling:

    rows  P, E_sat, m*/m, K_sat, Q_sat, cross
    cols  Gamma_sigma, b_sigma, c_sigma, Gamma_omega, b_omega, c_omega

    closure                                condition number
    6x6 all six                            5.1e2
    5x5 drop Q_sat, pin c_omega (shipped)  2.9e2
    5x5 keep Q_sat, drop cross             2.6e2
    4x4 no cross, pin b_omega and c_omega  3.5e2

**So the Jacobian is NOT badly conditioned** — 5e2 is unremarkable. The map's
"ill-conditioned" language was wrong, and the real cause is two other things.

### 1. The four shape coefficients are one direction, not four

|cos| between Jacobian COLUMNS:

                Gam_s   b_s     c_s     Gam_w   b_w     c_w
    Gamma_s     1.000   0.747   0.550   0.837   0.712   0.670
    b_sigma     0.747   1.000   0.961   0.269   0.997   0.992
    c_sigma     0.550   0.961   1.000   0.024   0.974   0.987
    Gamma_w     0.837   0.269   0.024   1.000   0.219   0.164
    b_omega     0.712   0.997   0.974   0.219   1.000   0.998
    c_omega     0.670   0.992   0.987   0.164   0.219*  1.000

`b_sigma, c_sigma, b_omega, c_omega` are mutually collinear at **|cos| >= 0.96,
and b_omega/c_omega at 0.998**. They contribute essentially ONE independent
direction to the nuclear-matter parameters. Only `Gamma_sigma` and
`Gamma_omega` are genuinely distinguishable (`Gamma_omega` against `c_sigma`
is 0.024 — orthogonal).

That is the answer to "why does it not work": four shape-sensitive conditions
are being imposed on what is numerically a one-dimensional shape freedom, and
the solve succeeds only on the small residual differences between the four
columns — exactly the differences a stencil floor swamps.

### 2. The amplification arithmetic closes

    6x6:  515 x Q_sat's 1.5e-3 stencil floor  =  0.77 RELATIVE coupling error
    5x5:  288 x the h-exact rows' ~1e-12      =  2.9e-10

**So the shipped 5x5 is not a compromise, it is the only well-posed member of
the family**, and the reason is arithmetic rather than taste. The 6x6's basin
lottery (7/187 cells at zero restarts) is this number, not a solver defect.

### The proposal, and it is analytic

**Change coordinates.** The nuclear-matter parameters do not feel `(b_i, c_i)`;
they feel the coupling function's derivatives at saturation — `f_i(1) = 1`
(fixed), `f'_i(1)` (enters P and K_sat), `f''_i(1)` (enters K_sat and Q_sat).
Solve the system in

    (Gamma_sigma, f'_sigma(1), f''_sigma(1), Gamma_omega, f'_omega(1), f''_omega(1))

and map back to `(b_i, c_i)` by a 2x2 per meson afterwards. Two consequences,
both free:

- The Jacobian is near-diagonal by construction, so the collinearity above is
  a coordinate artifact and disappears.
- **The cross-constraint becomes a coordinate identity**,
  `f''_sigma(1) = f''_omega(1)`, eliminable by SUBSTITUTION rather than
  carried as a residual row. 6 unknowns and 6 rows become 5 and 5 with no
  row deleted and no coupling pinned.

**Second, take Q_sat analytically.** Its 1.5e-3 floor is a third finite
difference of E/A. The isoscalar E/A at saturation is a closed form in the
couplings and their density derivatives, so the third derivative can be
differentiated by hand. With the floor gone, the 6x6's amplification falls
from 0.77 to nothing and imposing Q_sat becomes legitimate.

### If neither is done: which couplings stay free, and why

Ordered by the measurement, not by intuition:

- **`Gamma_sigma` and `Gamma_omega` are never pinned.** They are the only two
  independent directions; a closure without both is under-determined in fact
  whatever its row count says.
- **Pin `c_omega` first** (the shipped choice). Now justified quantitatively:
  `b_omega`/`c_omega` is the most collinear pair in the matrix at 0.998, so
  removing one of them costs the least information of any single pin.
- **For a 4+2 closure without the cross-constraint, pin the whole omega shape**
  (`b_omega` AND `c_omega`) and let the sigma shape carry K_sat. With the
  cross row gone the softest direction moves to `(b_sigma, c_sigma)`, so one
  sigma shape freedom is the most the data supports.

### And the cross-constraint can be made optional today

Checked: it appears ONLY in `nmp.py`. `Parameters.__post_init__` validates
`f_i(1) = 1` and `d_i = 1/sqrt(3 c_i)` and NOT the cross-constraint, so the
forward path — `couplings_at`, the solver, every table — never sees it. A
`Parameters` violating it is already legal and already computes. So
"couplings as input, no cross-constraint" is not a change, it is the current
behaviour; what is owed is that the option is DECLARED (a flag on
`invert_nmp`, and one sentence in `dd2.md`/`.tex` saying the constraint binds
the inverse map only).

## Gate

A ruling plus, if the reparametrisation is adopted, the measured
condition numbers in the new coordinates beside the four above. No published
number moves: this is the same closure in different coordinates, so
`compute_nmp` on the shipped set must return the same eight values.

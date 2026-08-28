# The DD2 isoscalar closure is degenerate in the shape coefficients, and the
# coordinates are why

Type: grilling
Status: resolved (2026-08-28)
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


## What [ticket 93](93-invert-nmp-basin-lottery.md) hands over (2026-08-28)

93 removed the solver half of the complaint, so what is left here is the
closure alone, and it is now measured at DD2's own point rather than inferred.

- **The 6x6 no longer stalls; it converges to a wrong answer and says ok.**
  With the stall counted as a miss the restarts fire, and at DD2's own NMPs the
  6x6 reaches max|residual| = **1.408e-02** — under `ISO_GATE` by a factor 1.4,
  so `ok=True` — while imposing **Q_sat only to 1.585 MeV** and K_sat to
  1.9e-03. It **saturates**: 64 and 128 restarts find nothing better. So the
  6x6's `ok` is a statement about a residual whose Q_sat row carries the
  stencil floor, and no verdict change can rescue it. This is the arithmetic
  of this ticket (515 x 1.5e-03) arriving as a measured Q_sat miss.
- **The 5x5 was never the noise problem.** Its own residual floor is
  2e-10 .. 1e-07 and its whole defect was the suppressed restarts. After 93 it
  converges at all seven target perturbations over eps in [0, 1e-8], at
  `coupling_shift` 3.948e-02 every time. So the "amplified noise, not a
  singular Jacobian" framing splits the two closures rather than covering both:
  the noise is the entire story in the 6x6 and no part of it in the 5x5.
- **A routing question this ticket is the natural home for.** A whole
  `compute_nmp` dict carries `Q_sat`, so `impose_Q_sat=None` sends
  `from_nmp(compute_nmp(par))` — the natural round trip — to the **6x6**, the
  worse-conditioned closure, while the shipped default is the 5x5. Two dd2
  tests were 6x6 tests that read as 5x5 ones for exactly this reason. If the
  reparametrisation lands, decide whether the presence of a key should still
  choose a closure.


## Resolution (2026-08-28)

**Ruled by the user: DD2's inverse map drops the cross-constraint entirely.**
Six isoscalar couplings free, the isovector pair fixed by E_sym and L_sym. Both
of the ticket's analytic proposals were REFUTED by measurement before the
ruling, and the ruling then made the first one moot.

### The literature check changed the question

The ticket asked for the constraint's stated motivation. It has one, and it is
not DD2's:

> Typel, *Phys. Rev. C* **71**, 064301 (2005), §IV: "In order to reduce the
> number of free parameters it is required that the functions f_sigma and
> f_omega obey the conditions f_sigma(1) = f_omega(1) = 1,
> f'_sigma(0) = f'_omega(0) = 0, and **f''_sigma(1) = f''_omega(1)**."

Parameter economy, stated. No physics gloss — so "sigma and omega run
together" was not merely unsourced, it invented a motivation for a counting
device. (`f'(0)` there is a typo for `f''(0)`; the DD table's own
d_i = 1/sqrt(3 c_i) to 7 digits.) That paper counts **eight** independent
parameters for DD. Typel et al., *Phys. Rev. C* **81**, 015803 (2010) — the
DD2 paper — states only the first two conditions and counts **ten**. The
difference of one is exactly this constraint, and the tables agree:

    DD  (Typel 2005)   f''_sigma(1) - f''_omega(1) = -6.0e-08   imposed
    DD2 (Typel 2010)   f''_sigma(1) - f''_omega(1) =  2.201e-03   not imposed

**The 2.200718e-3 was never a fit imperfection. It is the constraint's
absence.** `invert_nmp` had been closing DD2 with DD's condition.

### Proposal 1 (reparametrise) — REFUTED, then moot

The map (b_i, c_i) -> (f'_i(1), f''_i(1)) is **block-diagonal**, one 2x2 per
meson, and principal angles between the sigma-shape and omega-shape planes in
NMP space are invariant under exactly that class of map. Measured: **0.000 deg
and 12.62 deg**. The zero is forced, and the reason is a rank statement the
ticket did not have:

    row sensitivities to the four shape columns (dQ/dlog p):
      P       -1.729e+01   1.711e+01   1.482e+01  -1.508e+01
      E_sat    0.000e+00  -1.137e-08   0.000e+00   0.000e+00
      m*/m     0.000e+00  -5.551e-12   0.000e+00   0.000e+00
      K_sat   -1.669e+02  -7.400e+01   5.820e+02  -3.692e+02
      Q_sat    8.134e+03  -4.067e+03  -5.846e+03   5.084e+03

**E_sat and m*/m are shape-blind** — at fixed n_sat they need only f_i(1) = 1.
Four shape knobs answer to THREE rows; the shape block's singular values are
[77.0, 7.16, 1.94, 1.8e-10], an exact rank deficiency of one. Two 2-planes in
a 3-space always intersect. So the diagnosis is not "four columns are one
direction" but **four knobs in a three-dimensional response**, and it is
coordinate-invariant. Running the reparametrisation confirms it: cond
2623 -> **3082 (worse)**, f'_s/f''_s at 0.995 against b_s/c_s 0.962, shape
block still rank 3. With the cross row then removed by the ruling, the
proposal's one surviving payoff — eliminating that row by substitution —
had nothing left to eliminate.

### Proposal 2 — the only remedy that works, and it is deferred

    6x6 all six                          2623
    5x5 drop Q_sat, pin c_omega (was)    1473
    5x5 substituted (proposal 1)          698
    4x4 no cross, pin b_om and c_om       354
    5x5 pin c_omega, DROP CROSS, keep Q   259   <- best available

259 x 1.5e-3 = **0.39 relative coupling error**. No conditioning on offer
rescues an imposed Q_sat; proposal 1 buys 3.8x where ~1e9 is needed. Kill the
floor and even the 6x6 at 2623 is fine. **The two proposals were never a
package: 2 alone suffices, 1 cannot substitute for it in any amount.**
Carried to [ticket 111](111-dd2-analytic-nmp-derivatives.md) under the user's
Q4 ruling, "all the relations that can be written analytically should."

### The closure that ships

Q1's four bullets, answered by measurement:

- **Option C (a sixth NMP) is refused.** Z_sat is the only isoscalar datum
  left (m*/m is spent; the tower is E_sat, K_sat, Q_sat, Z_sat) and nobody
  quotes it — imposing it converts one free coupling into one free NMP, the
  same freedom behind an extra nonlinear map, which is worse for §6 inference.
  Measured: cond **550340**, and the fourth difference spans **4.8e+04 on a
  value of 4547**. Not a datum; noise with a name.
- **Option A (5 rows, one pin): pin `c_omega`** — 259, against b_omega 354,
  c_sigma 703, b_sigma 4191. Exposed, and documented as not trustworthy until
  108 lands.
- **Option B (4 rows, two pins): pin `b_sigma` and `c_omega`** — **128**, the
  best number in the exercise, and every row h-exact (K_sat spans 5.2e-04 MeV
  over h in [5e-5, 5e-4] against Q_sat's 2.48 MeV). **This is the new
  default.**
- **This refutes the ticket's own fallback ordering**, which said pin the whole
  omega shape and let the sigma shape carry K_sat. Measured, (b_om, c_om) is
  354 and (b_sig, c_sig) is 305 — both beaten by one pin per meson. What should
  be left free is the least collinear surviving pair, and c_sigma against
  b_omega at |cos| 0.974 is the least collinear pair in the matrix. (The
  ticket's cosine table is otherwise reproduced to three digits; its c_om row
  carried a transcription slip, b_om/c_om is 0.998 there too.)

### What the change bought, beyond the ruling

**The published DD2 couplings are now a root of their own inverse map.** All
four default rows vanish at the published table, so `from_nmp(compute_nmp(par))`
returns it to **1.1e-05** where the old closure reached a root 3.9% away. The
predicted Q_sat goes from **117.5 to 168.5** against a forward-map 168.9.
`test_roundtrip_recovers_couplings` — asserted, then withdrawn by ticket 93
because the premise was false — is true again, and now for a reason.

Q5 answered: **presence of a "Q_sat" key selects nothing.** `impose_Q_sat`
defaults to a plain `False`; the None-routing sent the natural round trip into
the noisier closure on the accident of a key being present.

### Reported, not fixed

`ISO_GATE = 2e-2` was set wide to clear the cross row's 2.2e-3 and Q_sat's
stencil, and this ticket retired both reasons. On a 105-cell (K_sat, m*/m)
grid with restarts on, the 101 passing cells split **95 below 1e-5 and 6 in
[1e-3, 2e-2], with nothing in between** — six certified without being roots.
Ticket 93 ruled the gate could not be tightened, and that ruling rested on the
cross row making accurate solves land at 1.9e-3, a premise this ticket
removed. Retuning moves `ok` for real targets, so it is 108's, not a
side effect here. Stalls themselves survive the change and `_stalled` stays
load-bearing: 12 of 18 misses on that grid at zero restarts are stalls.

### Gate

dd2 `run_full_check` **PASS**, golden SNM(0.16) `1.40e-05` and CompOSE
HS(DD2) `2.83e-05` both UNMOVED; `test/dd2` 211 passed. No published number
moved: `compute_nmp`, the DD2 table and `test/baseline/dd2.npz` are all
forward-path and untouched — what moved is what the INVERSE map returns, which
is the point. The ticket's own gate line ("same closure, different
coordinates") no longer describes the outcome; the closure is what changed.

Landed: `eos/dd2/nmp.py` (cross row removed from both closures, `PINNED_COEFF`
-> `PINNED_DEFAULT`/`PINNED_WITH_Q_SAT`, `_f2_at1` deleted, `impose_Q_sat`
default False), `eos/dd2/dd2.md`, `eos/dd2/dd2.tex`, `docs/eos.bib` (new
`Typel2005`), `eos/dd2/verify/run_full_check.py`, `test/dd2/test_api.py`,
`test/dd2/test_dd2_m8.py`, `docs/DEFERRED.md`, and **`CLAUDE.md` §5**, which
stated the cross-constraint as specification and gained the general sentence:
a closure condition belongs to the parametrization that imposed it, and is
checked against the paper that fitted THAT set.

Status: resolved (2026-08-28).

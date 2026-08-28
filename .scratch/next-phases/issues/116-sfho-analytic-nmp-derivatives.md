# SFHo's nuclear-matter derivatives should be analytic too, for
# reproducibility rather than for conditioning

Type: task
Status: resolved (2026-08-29)
Blocked by: -   (111 is the worked precedent)
Parent: ../map.md

## Question

Raised on [ticket 103](103-nmp-closures-four-models.md) (2026-08-29).
`sfho.nmp.compute_nmp` takes K_sat as a second finite difference of E/A, Q_sat
as a third, and K_sym as a second difference of `esym`, each of a quantity that
is itself the output of a nonlinear solve.
[Ticket 111](111-dd2-analytic-nmp-derivatives.md) did the same program for dd2
and found something that has nothing to do with conditioning:

> `test/baseline` documents a 0.351 MeV Q_sat shift between anaconda 3.9.7 /
> numpy 1.26.4 / scipy 1.13.1 and python.org 3.14.2 / numpy 2.3.5 / scipy
> 1.17.0 -- 2e-3 relative, all of it stencil roundoff. The four analytic values
> now agree across those same two stacks to 1.2e-13 .. 3.1e-16.

That argument transfers unchanged: SFHo's published Q_sat is a stencil number
and therefore an interpreter-dependent one. SFHo's own h-sweep puts the plateau
spread at 0.13 MeV on 467 (2.8e-4 relative), with the h = 5e-5 end 2.5 MeV away.

**This does NOT make Q_sat imposable in sfho, and the ticket must not claim it
will.** 103 measured why: sfho's only candidate fifth isoscalar knob is `c3`,
whose Jacobian column is 550x weaker than `g_sigma`'s, and admitting it drops
the closure's `sigma_min` from 0.051 to 0.0063. That is a statement about what
saturation constrains -- a high-density omega^4 term is not a saturation
observable -- and no derivative accuracy touches it. The default four-row
closure is already h-exact, so nothing shipped waits on this ticket either.

SFHo should be easier than dd2 was: its couplings are constants, so there is no
density-dependent coupling to differentiate and no rearrangement term. The
program is 111's -- differentiate the gap equation implicitly for the sigma
field's density derivatives, substitute into the closed-form isoscalar E/A --
with `E_sym`'s closed form (Steiner et al., Phys. Rept. 411 (2005) Eq. 20)
giving L_sym and K_sym directly.

**Forward and inverse go analytic together.** 111's constraint holds here for
the same reason: the finite-difference bias cancels on a round trip only while
both sides difference identically.

## Gate

Analytic values inside the (h, h/2) Richardson pair's own scatter; the four
values identical across two interpreters where the stencils differ; the
h-exact keys (n_sat, E_sat, m*/m, E_sym, P_sat) bit-identical; `test/baseline`
`sfho.npz` passing, with every moved published number named old -> new; sfho
`run_full_check` PASS and `test/sfho` green. A `verify/` entry mirroring dd2's
`analytic NMP derivatives`.


## Resolution (2026-08-29)

**Done. Every derivative in `sfho/nmp.py` is analytic, forward and inverse
together.** The ticket's argument held — the four values are now the same
number on two interpreters where the stencils were not — but the mechanism it
predicted was wrong, and correcting it is what moved more than Q_sat.

### The derivation

SFHo is easier than dd2 in the way the ticket said and harder in one way it
did not. Easier: the couplings are constants, so there is no rearrangement
self-energy and nothing density-dependent to differentiate. Harder: dd2 treats
symmetric matter as one g = 4 gas at an averaged Dirac mass, while sfho keeps
m_p != m_n, so at Y_C = 0.5 (where n_p = n_n exactly, hence rho = phi = 0) it
is TWO g = 2 gases at one k_F and two Dirac masses.

Writing `Phi(sigma) = m_sigma^2 sigma + g2 sigma^2 + g3 sigma^3` and
`Psi(omega) = m_omega^2 omega + c3 omega^3`, the fields at rho = 0 obey
`Phi(sigma) = g_sigma n_s(sigma, n)` and `Psi(omega) = g_omega n`. The second
carries **no scalar density at all**, so omega is an exact function of n alone
and costs two lines; only the sigma gap needs implicit differentiation.

111's shortcut transfers unchanged once the right potential is identified.
Holding Y_C = 0.5 means `deps = mu_p dn_p + mu_n dn_n = mu_bar dn` with
`mu_bar = (mu_p + mu_n)/2`, so `eps' = mu_bar` even though mu_p != mu_n, and
with `P = mu_bar n - eps`

    K_sat = 9 n mu_bar',   Q_sat = 27 n (n mu_bar'' - 3 mu_bar')   at P = 0

**A third derivative of E/A is a SECOND derivative of mu_bar**, so the gap is
differentiated twice. `L_sym`/`K_sym` come from the same `E_F*` derivatives
plus A(sigma, omega); f is separable in sigma and omega, so there is no mixed
second partial — a property of the SFHo form, and the two new partials live
on `Parameters` beside the first ones. Z_sat is not reported, for 111's reason.

`snm_derivatives(par, n_B)` is the one home; `compute_nmp`,
`_isoscalar_quantities` and the isovector residual all call it, so forward and
inverse went analytic together as the ticket required. It solves symmetric
matter once instead of seven times.

### The finding: the stencil's floor was never h

The ticket's premise was the h-plateau — 0.13 MeV of spread, with the h = 5e-5
end 2.5 MeV away. That is true and it is not the binding constraint. Two
measurements found the real one.

**T_COLD was buying JEL's approximation error.** `_snm` solved at 0.01 MeV, on
the argument that a strictly cold solve puts a threshold kink where the
differences straddle. That reason never applied here — symmetric matter is
nucleons only and a nucleon has no threshold — and with the differences gone
nothing is left of it. What it did buy: sfho evaluates the Fermi integrals in
**closed form on its T = 0 branch and through JEL above it**, so at
n_B = 0.158 the T = 0.01 MeV solve returned a sigma 1.5e-07 relative away from
the exact gap root and an eps 4.5e-08 away. Not a thermal effect — T = 1e-4
and T = 0.01 MeV give the SAME displaced answer, so it is the branch, not the
temperature. **`T_COLD = 0.0`**, and the whole NMP path is exactly T = 0.

**What remains is `hybr`'s own xtol, and no h reaches past it.** The solver
returns a state whose density is up to 5e-11 relative from the one it was
asked for; dividing by the requested density leaves a SMOOTH ~5e-08 MeV wobble
on E/A whose curvature is a FIXED offset from the analytic value — 1.9e-06
relative on K_sat and 2.8e-04 on Q_sat, the same at h = 1e-3 and at h = 5e-3,
and reproduced to four digits on both interpreters. So the stencil route could
never have been improved by choosing h better; the analytic values are not a
refinement of the plateau but a different and correct limit. (Stencilling a
hand-solved gap equation instead of the solver moves the K_sat agreement from
1.9e-06 to 1.4e-06 and the offset vanishes, which is what identifies it as the
solver's.)

### Gate, line by line

- **Analytic values inside the stencil pair's own scatter.** Held, and the
  ticket's preferred comparison — against the h-plateau MEAN, not the shipped
  h — is the one reported. Plateau h in [2e-4, 2e-3] on python.org 3.14.2:

                  plateau mean    spread     analytic     analytic - mean
      K_sat        245.221306    3.26e-03   245.219693       -1.6e-03
      Q_sat       -467.417901    9.80e-02  -467.547098       -1.3e-01
      L_sym         47.077861    3.58e-03    47.076598       -1.3e-03
      K_sym       -205.378953    2.23e-03  -205.378186       +7.7e-04

  Three of the four land INSIDE the plateau's own spread; **Q_sat lands 1.3
  spreads outside it**, and that is the whole correction — 0.129 MeV, against
  dd2's 0.073 MeV on a similar-sized number.

- **The four values identical across two interpreters where the stencils
  differ.** Held. anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1 against
  python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0: **5.9e-14 (K_sat), 2.2e-13
  (Q_sat), 7.0e-15 (L_sym), 6.8e-16 (K_sym)**, and n_sat to 1.5e-14. The old
  path differed by 9.7e-05 MeV on Q_sat at the shipped h and by 1.4e-02 MeV on
  the plateau mean. Nine orders.

- **The h-exact keys bit-identical.** **NOT HELD, and it could not be** — this
  is the one line of the gate the work refuted rather than met. The gate was
  copied from 111, whose `solve_snm_t0` was already exactly T = 0; sfho's was
  not, and `T_COLD` above is why. n_sat, E_sat, m*/m and E_sym move by 3.3e-06,
  2.6e-06, 8.7e-07 and 1.7e-06 relative. The alternative was to keep the
  analytic derivatives on a state that does not satisfy its own gap equation,
  which is incoherent (the reported K_sat would not be the curvature of the
  reported E_sat curve) and which the new verify entry fails at a ratio of ~5.
  P_sat stays ~1e-14, zero by construction as before.

- **`test/baseline/sfho.npz` passing.** Held, unmodified, on both
  interpreters. It freezes no NMP key — only `dd2.npz` carries an `nmp` block
  — so nothing in it was ever in reach of this change.

- **sfho `run_full_check` PASS and `test/sfho` green.** Held on both stacks
  (below).

- **A `verify/` entry mirroring dd2's.** Landed, below.

- **The ticket's own prohibition.** Nothing here makes Q_sat imposable in
  sfho and nothing claims it does. `docs/DEFERRED.md` now says so with 103's
  measurement (c3's Jacobian column 550x weaker than g_sigma_N's, sigma_min
  0.051 -> 0.0063) and states explicitly that derivative accuracy was never
  the obstacle.

### The moved published numbers, old -> new

On python.org 3.14.2, old = stencil at the shipped h = 1e-3 with the T = 0.01
solve; new = analytic at T = 0:

    n_sat        0.1582409773  ->   0.1582415032    (rel 3.3e-06)
    E_sat      -16.1723618256  -> -16.1724036674    (rel 2.6e-06)
    m*/m         0.7615642360  ->   0.7615635772    (rel 8.7e-07)
    E_sym       31.5456784436  ->  31.5457311218    (rel 1.7e-06)
    K_sat      245.2210817033  -> 245.2196926301    (rel 5.7e-06)
    Q_sat     -467.4237080394  -> -467.5470984057   (rel 2.6e-04)
    L_sym       47.0775696160  ->  47.0765975489    (rel 2.1e-05)
    K_sym     -205.3787719172  -> -205.3781863893   (rel 2.9e-06)

The list is in the module docstring. Nothing pins any of them: `test/sfho`
carries no golden NMP value, `verify`'s `published NMPs` check compares
E_sat/E_sym/L against Steiner et al. at 2e-2 (worst 2.01e-03, unmoved), and
`test/baseline` freezes none of them. The largest move is 2.6e-04 relative.

### The round trip got four orders sharper

Forward and inverse now differentiate identically by construction rather than
by both being the same stencil. On the (0.160, -16.00, 0.75, 240, 32, 60)
target, worst |forward - target| over the six imposed values went
**3.5e-06 -> 2.7e-11**, and the isoscalar residual **3.7e-09 -> 1.0e-12**.
`status.predictions` is bit-identical to a forward map of its own couplings on
both sides of the change. `ISO_GATE` was already 1e-6 and is untouched; it now
sits six orders above the residual instead of two and a half, and the ticket
did not ask for it to be retuned.

### The verify entry

**`analytic NMP derivatives`**, the same estimator as dd2's: each of the four
must sit within the (h, h/2) stencil pair's own scatter of their Richardson
extrapolation. Self-calibrating, no tolerance to tune.

**h = 8e-3, and that is a measured choice made for the OPPOSITE reason to
dd2's.** There the small-h end failed on roundoff, which jittered with the
interpreter. Here the two stacks agree to four digits at every h, because what
stops the stencil converging is the solver floor above, not roundoff — a fixed
offset, so the estimator is honest only once the scatter has grown past it.
Worst of the four, Q_sat every time:

    h        1.5e-3   2e-3    4e-3    6e-3    8e-3    1.2e-2
    py3.9     4.674   3.152   0.800   0.356   0.279   0.089
    py3.14    4.674   3.244   0.800   0.356   0.279   0.089

At 8e-3 both stacks pass with 3.6x of margin and the ratio is still falling,
so anywhere above ~5e-3 does the same job. What the gate has to CATCH was
measured by ablation — zeroing one term at a time in `snm_derivatives`, the
smallest signature any real dropped term leaves is 3.2e-03 relative (K_sym,
from omega''), and most are 1e-2 to 6.

### Landed

`eos/sfho/nmp.py` (the `snm_derivatives` section, `T_COLD`, `compute_nmp` and
`_isoscalar_quantities` rewired, the isovector residual's stencil gone, `h`
gone from three signatures, the moved-numbers docstring),
`eos/sfho/parameters.py` (`compute_d2A_dsigma2`, `compute_d2A_domega2`),
`eos/sfho/__init__.py` (`snm_derivatives` exported),
`eos/sfho/verify/run_full_check.py` (the new check, renumbered header),
`eos/sfho/sfho.md`, `eos/sfho/sfho.tex` (the full derivation, a new subsection
of the nuclear-matter-parameters section; compiles), and `docs/DEFERRED.md`
(the sfho NMP entry: derivatives analytic, Q_sat still not imposable and now
with the measurement saying why).

Status: resolved (2026-08-29).

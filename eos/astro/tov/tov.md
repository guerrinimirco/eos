# astro/tov — stellar structure from an equation of state

The full description, with every equation and the bibliography, is `tov.tex`
(compiled against `../../../docs/eos.bib`). This file is the plain-text summary.

`eos/astro/tov` turns a table into a star: TOV mass and radius, the l = 2 tidal
Love number and Lambda, and — through the external RNS code — uniformly
rotating models. It also assembles the table it integrates, joining a tabulated
crust to the core a model produced.

    crust.py        building ONE table from a crust and a core
    solver.py       integrating the TOV and tidal equations, and sequences
    solver_fast.py  a jitted integrator on a uniform log-P grid: four ODE
                    variables (m, P, Y, M_b) in dimensionless r, so the Love
                    number and the baryonic mass cost no extra passes. Its
                    tolerances and its M_b algorithm differ from solver.py --
                    it is a second implementation, not a compiled copy
    rotating.py     uniformly rotating sequences, Kepler limit, turning points
    rns_backend.py  writing RNS input, running it, parsing its output

The table it consumes is `eos.general.state.EOSTable_for_TOV` — three parallel
arrays (P, epsilon, n_B), fm-based. It lives in `general/` because it is the
CONTRACT between the models that produce a table and this layer that consumes
one, and CLAUDE.md section 1 lets both import `general/` while forbidding a
model to import `astro/`.

## The variables the integration actually uses

Not physical units. With r_sun = 2 G M_sun/c^2 = 2.953 km, M_sun in MeV/c^2,
and P_c the central pressure of the star being built:

    r^ = r / r_sun      m^ = m / M_sun      p^ = P / P_c        (p^ = 1 at centre)
    rho_sol = 3 M_sun / (4 pi r_sun^3)

That choice of r_sun is what makes the metric factor collapse to

    1 - 2Gm/rc^2  =  1 - m^/r^                      no constant left over

ONE EXPRESSION, TWO FACTORS — read before comparing with the source.
4 pi r^2 eps appears both as a MASS PER UNIT RADIUS (in dm/dr) and as a
DIMENSIONLESS geometric quantity (inside F and Q below), and it does NOT carry
the same factor in the two places:

    4 pi r^3 P / M_sun     = 3   (P/rho_sol)   r^^3        a mass
    dm^/dr^                = 3   (eps/rho_sol) r^^2        a mass per radius
    [4 pi r^2 eps]_geom    = 3/2 (eps/rho_sol) r^^2        dimensionless

The dimensionless one picks up a further r_sun/M_sun = 2G/c^4, which halves it.
That is why the code carries 1.5 in F and in the first term of r^2 Q, but 3.0
in dm^/dr^ and in (m + 4 pi r^3 P). Deriving one from the other and expecting
the same coefficient is the mistake to avoid.

and the TOV system becomes

    dm^/dr^ = 3 (eps/rho_sol) r^^2
    dp^/dr^ = -(1/2)(eps/P_c) m^ (1 + P/eps)(1 + 3 (P_c/rho_sol) p^ r^^3 / m^)
              / (r^ (r^ - m^))

Start at r^ = 1e-3 with the uniform-core m^, p^ = 1; stop on a terminal event
at p^ = 1e-10. DOP853, rtol 1e-10, atol 1e-12, dense output.

Baryonic mass counts PROPER volume:

    M_b = m_N INT 4 pi r^2 n_B / sqrt(1 - 2Gm/rc^2) dr

## Tidal deformability

Integrated in Y(r) = r H'(r)/H(r), which stays finite where H spans many
decades:

    r dY/dr + Y^2 + Y F(r) + r^2 Q(r) = 0,      Y(0) = 2

    F     = [1 - 4 pi r^2 (eps - P)] / (1 - 2m/r)
    r^2 Q = 4 pi r^2 [5 eps + 9P + (eps+P)/(dP/deps)] / (1 - 2m/r)
            - 6/(1 - 2m/r) - [(m + 4 pi r^3 P)/(r(1 - 2m/r))]^2

then k2 from the surface value Y(R) and compactness C = GM/Rc^2 (Hinderer 2008
Eq. 23), and Lambda = (2/3) k2 C^-5. Returns k2 = Lambda = 0 outside
0.005 < C < 0.5, where the closed form is ill-conditioned.

cs^2 = dP/deps is TABULATED ONCE on the table's own P grid before integrating
and interpolated like any other column, clipped to [1e-6, 1]. Differentiating
inside the right-hand side would make every step depend on a numerical
derivative of an interpolant. The lower clip matters on a Maxwell plateau,
where deps/dP -> infinity and cs^2 -> 0 exactly.

**The jump at a first-order interface.** Where eps is discontinuous at
continuous P — a self-bound surface, or an internal Maxwell transition — Y
jumps and integrating through is simply wrong:

    dY = -4 pi r_d^3 delta_eps / (m(r_d) + 4 pi r_d^3 P(r_d))

THE DENOMINATOR IS m + 4 pi r^3 P, NOT m. The two agree only at a surface where
P = 0, which is why the familiar -4 pi R^3 eps_s / M is right for a bare quark
star and wrong inside a hybrid star. Takatsy & Kovacs 2020 (PRD 102 028501),
correcting Postnikov 2010. Getting it wrong biases Lambda.

Crossing one: a second terminal event fires at p^ = P_t/P_c; Y gets its jump,
the interpolant is restricted to the low-density branch, cs^2 is retabulated on
it, and the integration resumes.

## The crust, and why the obvious join fails

The crust and the core are independent calculations that DISAGREE on P at the
same n_B: at n_B = 0.080 fm^-3, BPS gives P = 0.406 MeV/fm^3 where one DD2
parametrization gives 0.225. Concatenating at a chosen density therefore steps
DOWN in P at the seam — and a table whose P decreases with n_B is not an EoS:
eps(P) is double-valued, cs^2 is negative there, and the integration diverges.

Three join modes, differing precisely in how they avoid that:

    attach        keep crust points with n_B <= n_t AND P < min(P_core). The
                  second condition is the safe one: the join lands where the
                  two P(n_B) curves CROSS, not at a density fixed in advance.
    interpolate   tanh blend in n_B over n_t +- delta_n, applied to P and to
                  mu_B = (P + eps)/n_B, recovering eps = mu_B n_B - P. Blending
                  mu_B rather than eps keeps the blended region
                  thermodynamically consistent.
    maxwell       find P_t where mu_B^crust(P) = mu_B^core(P) — the real
                  Maxwell condition — and tanh-blend eps(P) across it; sharp as
                  delta_P -> 0.

Crust tables are large external data, not shipped and not in git. Resolved by
name against an explicit path, then each directory of $EOS_CRUST_DIR (a SEARCH
PATH — the BPS file and the CompOSE tree do not live together), then
<repo>/data/crust. A missing table raises naming the file, every directory
tried and the variable. `have_crust(name)` is the predicate to call before
falling back: falling back to no crust moves M_max by ~1%, so it is visible at
the call site rather than silent.

## Sequences

One integration per central energy density; columns
(eps_c, n_c, P_c, R, M, M_b, k2, Lambda).

M_max is not read off the grid: the maximum is located on the FINITE subset
(central states past the table's validity integrate to NaN and must not enter
an interpolation), a +-5-point window is fitted with a monotone cubic, and the
peak of the fit is returned. The stable branch ends at dM/deps_c = 0;
`truncate_to_stable_branch` removes what lies beyond, which are not stars.

## Rotation

Computed by RNS (Stergioulas & Friedman 1995), third-party Fortran run as a
subprocess. Binary from an explicit argument, then $RNS_BIN, then PATH, then
<repo>/external/rns.

### What RNS solves

Stated here because the columns this module writes and parses cannot be checked
without it. A uniformly rotating, axisymmetric, stationary perfect fluid in the
quasi-isotropic gauge of Komatsu, Eriguchi & Hachisu (1989), in the form Cook,
Shapiro & Teukolsky (1994) put it:

    ds^2 = -e^(gamma+rho) dt^2 + e^(2 alpha) (dr^2 + r^2 dtheta^2)
           + e^(gamma-rho) r^2 sin^2(theta) (dphi - omega dt)^2

with gamma, rho, alpha, omega functions of (r, theta). Rotation is UNIFORM:
Omega = u^phi/u^t is one number, not a profile. omega(r, theta) is the
frame-dragging rate and is a different quantity — the two are easy to confuse
in the output.

Three potentials satisfy elliptic equations, which KEH turns into integral
equations with flat-space Green's functions and iterates; alpha follows by
quadrature. That iteration is what the accuracy and relaxation numbers in the
retry ladder below control.

The matter side is one equation. For a barotrope the Euler equation has a first
integral over the whole star,

    H(r, theta) + (1/2)(gamma + rho) - (1/2) ln(1 - v^2) = const
    v = (Omega - omega) r sin(theta) e^(-rho)

with v the fluid's proper velocity as seen by a zero-angular-momentum observer,
and H the specific enthalpy

    H(P) = integral_0^P dP' / (eps(P') + P')

That integral is the ONLY place the equation of state enters, which is why the
table handed to RNS carries an enthalpy column, and why `_specific_enthalpy`
integrates it on 16003 points uniform in ln P rather than on the table's own
rows: 1/(eps+P) falls by orders of magnitude between adjacent rows of a
log-spaced table, and a 1% error in H moves the masses by ~1%.

A model is two numbers: the central energy density eps_c and the axis ratio

    r_ratio = r_p / r_e        (polar / equatorial COORDINATE radius)

with r_ratio = 1 the non-rotating star. Omega is an OUTPUT, not an input — the
solver finds the rotation rate for which the first integral holds at the
requested shape. Lowering r_ratio spins the star up to mass-shedding, where the
equatorial surface velocity equals the orbital velocity of a free particle;
that is the Kepler limit and it ends the sequence.

### Driving the external code

Four RNS properties shape the interface, each a constraint the caller cannot
see:

    200 rows max      it declares double[201] and does not check; a denser
                      table is RESAMPLED, not rejected
    surface at        7.8 g/cm^3, hardwired. A table not reaching it is
                      rejected saying to attach a crust — the check catches a
                      MISSING crust, wrong by seven decades, so it carries a
                      factor-of-ten tolerance
    char[80] paths    filled by sscanf("%s"), so the EoS file is symlinked to
                      a short fixed name in the run directory
    its own c         2.9979e10 cm/s divides the enthalpy column on read;
                      writing it with the same constant makes the round trip
                      exact rather than accurate to 3e-6

Non-convergence retries on (accuracy, relaxation) = (1e-5, 1.0), (1e-6, 0.8),
(1e-7, 0.8), (1e-4, 0.6) — tightening BEFORE loosening, because most failures
are an iteration oscillating about the solution, which a smaller step fixes and
a looser tolerance only hides.

Along a rotating sequence the stability boundary is the turning point
dM/deps_c = 0 at fixed J or fixed M_0 (Friedman, Ipser & Sorkin 1988);
`turning_point` takes the FIRST stationary point, the one bounding the stable
branch.

# astro/gmode — composition g-modes

The full description, with every equation, the observability discussion and the
bibliography, is `gmode.tex` (compiled against `../../../docs/eos.bib`). This
file is the plain-text summary.

Gravity (g) modes are restored by BUOYANCY, which exists only when a displaced
fluid element cannot chemically re-equilibrate within an oscillation period.
Their frequencies therefore measure the COMPOSITION GRADIENT of dense matter —
something mass, radius and tidal deformability are blind to.

    sound_speeds.py  the two sound speeds, and what a finite rate does to them
    rates.py         Urca equilibration rates and the susceptibility A
    background.py    the stellar background: TOV metric, g, N^2, crust
    cowling.py       the Cowling eigenvalue problem and the mode search

## The two sound speeds, and why the difference is the physics

    c_eq^2 = dP/deps                    equilibrium: composition free to track
    c_ad^2 = (dP/deps)_x                frozen: composition x held

`c_eq` is what enters TOV and what a mass-radius measurement constrains.
`c_ad` is what a displaced element actually feels. The mode exists only in
their difference:

    N^2 = g^2 (1/c_eq^2 - 1/c_ad^2) e^(nu-lambda),   g = -(1/(eps+P)) dP/dr

If the two are equal — a barotrope, a polytrope, anything with one sound speed
— then N^2 = 0 and there is no g-mode at all. That is the test the
`polytrope` fixture in the suite exists to make.

Both speeds are computed the SAME way (proper central differences on re-solved
states, neutralising leptons in both), so their difference is the composition
effect and not a mismatch of conventions.

## The eigenvalue problem

Relativistic Cowling approximation: the spacetime is held fixed and only the
fluid is perturbed. This reduces the fourth-order Thorne-Campolattaro system to
two first-order equations confined to the star. The cost is the
gravitational-wave damping time; for g-modes the frequency error is a few per
cent, because the mode is concentrated in the core and nearly divergence-free.

With e^(-i omega t), degree l, U = r^2 e^(lambda/2) xi_r and
V = delta P/(eps+P):

    dU/dr = (g/c_ad^2) U + e^(lambda/2) [ l(l+1) e^nu / omega^2 - r^2/c_ad^2 ] V
    dV/dr = e^(lambda/2 - nu) (omega^2 - N^2)/r^2 U
            + g (1/c_eq^2 - 1/c_ad^2) V

NOTE WHICH SPEED APPEARS WHERE: c_ad governs the element's response to
compression; c_eq enters only through the buoyancy terms. Swapping them is the
easy mistake and it silently changes what the mode measures.

Modes are found by scanning the surface discriminant in frequency and counting
nodes in U, so a returned mode carries its order rather than being whichever
root the solver happened to land on.

## Finite reaction rates

Freezing is not binary. With an equilibration rate gamma the response is
between the two limits, and the sound speed becomes COMPLEX:

    c_dy^2 = c_dy^2(c_eq^2, c_ad^2, gamma, omega)

For gamma << omega it tends to c_ad^2 (frozen); for gamma >> omega to c_eq^2
(equilibrated, no mode). In between Im[c_dy^2] is positive — the mode damps —
and peaks at gamma ~ omega, which is also where the bulk viscosity peaks. In
the slow regime Im[c_dy^2] -> (c_ad^2 - c_eq^2) gamma/omega, linear in gamma,
so the damping time scales as tau ~ 1/gamma. Fast equilibration destroys the
buoyancy and with it the mode: Q = Re(omega)/2 Im(omega) falls monotonically,
passes Q ~ 1 near gamma = omega, and beyond that the root ceases to exist and
the solver says so rather than inventing one.

Rates come from direct and modified Urca, combined with the susceptibility
A = d(mu_n - mu_p - mu_e)/d Y_p.

## The background needs a crust

`with_crust` attaches one before building the background, and the tests skip
without it rather than running core-only: the mode's outer turning point sits
in the crust, so a core-only background is a different problem, not an
approximation to this one. Crust resolution is `eos.astro.tov.crust`
($EOS_CRUST_DIR, see `../tov/tov.md`).

## A caveat the document spends a section on

The split between "equilibrium" and "frozen" is CONVENTION DEPENDENT: which
quantities are held fixed defines c_ad, and different choices give different
N^2 for the same matter. In a quark-hadron mixed phase this is not a detail —
whether the quark volume fraction chi is held or allowed to relax changes the
answer qualitatively. `gmode.tex` section "Convention dependence" states which
choice is made here and why; `cs2_frozen_isobaric(cs2_H, cs2_Q, chi)` is the
mixed-phase case.

## Running the tests

The suite marks its long tests, and `pyproject.toml` documents the convention:
`markers = ["slow: long-running (excluded with -m 'not slow')"]`. Measured on
this machine:

    no crust data, -m "not slow"      69 s     29 passed, 5 skipped
    crust configured, -m "not slow"   17.5 min 34 passed
    one slow test on its own          3.6 min

NOTHING HERE HANGS — it is simply expensive, and the cost is the crust-attached
background: without the crust those five tests skip and the suite is a minute.
A run that appears stuck after a dozen dots is a `slow` test doing a complex
frequency scan. Use `-m "not slow"` for a gate, and budget twenty minutes if
you want the crust-backed tests too.

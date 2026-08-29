# mixed — the hadron-quark mixed phase, Gibbs to Maxwell in one parameter

A *composite engine*, not a model: it supplies no matter of its own. It builds
the first-order deconfinement transition between a hadronic phase and a quark
phase by coupling two validated bulk equations of state — DD2 (Typel et al.,
PRC 81, 015803 (2010); Hempel and Schaffner-Bielich, NPA 837, 210 (2010)) for
hadrons and vMIT (Chodos et al., PRD 9, 3471 (1974); Gomes et al., ApJ 877, 139
(2019)) for quarks — through a single narrow interface, the phase-adapter
contract.

A continuous parameter `eta` in [0, 1] selects how much of electric-charge
neutrality is imposed phase by phase rather than on the mixture as a whole, so
that `eta = 0` is the Gibbs construction (Glendenning, PRD 46, 1274 (1992)),
`eta = 1` is Maxwell, and intermediate values interpolate between them in the
sense of Constantinou et al., PRD 107, 074013 (2023) and its finite-temperature
successor.

This file states every equation the engine solves and every quantity it
returns. `mixed.tex` is the same document typeset, compiled against
`../../docs/eos.bib`; neither defers to the other. Every equation here is
asserted by the `verify/` suite or by `test/mixed/`.

**The parameters of the phases are not here.** A composite engine's parameter
argument is the pair of phases (see the contract below), each closing over its
own model's numbers; those numbers, and the equations they pin down, are in the
sibling documents — `eos/dd2/dd2.{md,tex}`, `eos/vmit/vmit.{md,tex}`,
`eos/sfho/`, `eos/zl/`, `eos/did/`, `eos/alphabag/`, `eos/enjl/`. What is
stated here is everything the engine itself carries: the contract, the
equilibrium conditions, the assembly, the numerical constants of the solve, and
what a mixed table reports beyond a point equation of state.


## What is being solved

At a given total baryon density `n_B`, temperature `T` and construction
parameter `eta`, a hadronic phase H occupying volume fraction `1-chi` coexists
with a quark phase Q occupying `chi`. The two phases are separate thermodynamic
systems, each with its own composition and its own set of conserved-charge
chemical potentials; coexistence is the statement that certain of those
potentials are shared and that certain charges are conserved only in the volume
average. Which ones is exactly the content of the construction, and it is the
only thing that changes between `eta = 0` and `eta = 1`.

`chi` is an unknown of the solve, not an input, and it is deliberately *not*
clamped to [0, 1]. Its value classifies the density: `chi <= 0` means the point
is still pure hadronic, `chi >= 1` that it is already pure quark, and
`0 < chi < 1` that it lies inside the coexistence window. The onset and offset
densities are located by exactly that sign change, which is why the solver must
be allowed to return the analytic continuation outside the window.


## The phase-adapter contract

Everything this engine knows about any bulk model passes through one interface,
and a *pairing* is two declared phases handed to the solver: DD2+vMIT is the
shipped default, SFHo, ZL, DID and alphaBag adapters ship alongside it, and two
*branches* of one functional (ENJL) pair the same way — the engine cannot tell
a two-model pairing from a two-branch one, which is the point. An adapter is a
map

    ( the phase's conserved-charge potentials, T )  ->  PhaseThermo   (adapter)

which solves that phase's own internal self-consistency — the meson field
equations for a density-dependent relativistic mean field, the vector-field
fixed point for a bag model, the gap equation for a chirally broken one — at
the *given* potentials, and reports the resulting block. There is no `eta` in
(adapter), no mixing and no neutrality: those are conditions on the *pair* of
phases. An adapter describes one phase in isolation, and pairing a different
hadronic or quark model is writing an adapter, not editing the solver.

**The block.** `PhaseThermo` (`eos.general.state`) is matter only — no leptons
and no photons, because those are shared by the whole system and in a mixed
phase may be shared differently from the matter. A thermal meson gas, on the
other hand, *is* part of the phase and is listed in `densities` beside the
baryons, which is what makes the reported `n_C` and `n_S` the TOTAL
non-leptonic charge and strangeness the neutrality and fixed-fraction
conditions are stated in terms of. The fields are:

| field | meaning |
|---|---|
| `T` | temperature [MeV] |
| `mu_B`, `mu_C`, `mu_S` | the conserved-charge potentials, `mu_C = mu_p - mu_n` |
| `fields` | the model's own self-consistent solution, in its own names; empty for a model that has none |
| `densities` | every active species: baryons, quarks, any thermal meson gas [fm^-3] |
| `mu_i` | `mu_i = B_i mu_B + C_i mu_C + S_i mu_S`, derived, never independent |
| `mu_eff_i` | `mu_i - Sigma0_i`, the potential entering the Fermi integrals — carried because it cannot be reconstructed without the fields and the per-species couplings |
| `m_eff_i` | the Dirac effective masses, per species, for the same reason |
| `n_B`, `n_C`, `n_S` | the phase's conserved-charge densities [fm^-3] |
| `P`, `eps`, `s` | [MeV/fm^3, MeV/fm^3, fm^-3] |
| `mu_dot_n` | `sum_i mu_i n_i` over this phase — supplied by the model, not derived here, because a mean field splits the field energy between `mu` and `P` in a way only the model knows (a thermal meson gas contributes `sum_j mu*_j n_j` at its *effective* potentials) |
| `Sigma_R` | rearrangement self-energy; enters `mu` and `P`, never `eps`; zero for constant couplings |
| `condensation` | `kappa = max_j |mu*_j|/m_j` over the thermal meson gas; 0 without one |

`mu_dot_n` is not decoration: it is what the engine's 1e-8 Euler check consumes,
and `euler_residual()` on the block is
`(eps + P - T s - mu_dot_n)/eps`.

**Why `condensation` is on the contract**, and it is not cosmetic. A
Bose-Einstein condensate is implemented nowhere in this repository; the shared
Bose routine caps `mu*_j` at `m_j` instead of diverging, so a condensed phase
does not blow up — it quietly stops absorbing charge and returns an entropy
that drifts negative, while every solve around it converges beautifully. The
mixed system has no other way to see that, because it sees a phase only through
(adapter), so a point with `kappa >= 1` in either phase is REFUSED rather than
returned. A quark phase has no meson gas and reports `kappa = 0`, which is the
physical answer rather than a missing value.

Two further properties of (adapter) are load-bearing.

**One decomposition.** Both adapters report the same projection

    mu_i = B_i mu_B + C_i mu_C + S_i mu_S                     (projection)

with `C` the electric charge of strongly-interacting matter only (leptons
excluded) and `S = +1` per `s` quark, the opposite of the PDG sign. Because
both phases speak (projection), the coexistence conditions can be written in
terms of `(mu_B, mu_C, mu_S)` without either side knowing which engine produced
the other.

**Determinism.** An adapter must be a function of its arguments alone: the same
potentials must give the same block, to the last digit, however the point was
reached. The mixed residual is differentiated numerically, so an adapter that
remembered the previous trial point would make the Jacobian differentiate that
memory along with the physics. This is why warm starts are passed in as
explicit arguments and why the one thing that *is* cached — the hadronic
phase's starting field configuration — is a constant of the solve rather than a
running state.

**The baryon-potential flavour is DECLARED, not assumed.** A phase's slot in
the unknown vector carries either the kinetic potential
`mu_tilde_B = mu_B - Sigma_R` or the physical `mu_B`, and which one is a
declared property of the phase (`potential_kind`), not a constant of the
engine. DD2 declares kinetic: its rearrangement self-energy `Sigma_R` is a
function of the phase density, itself an unknown of the phase-internal solve,
so carrying `mu_B` would put that circularity inside the outer iteration —
carrying `mu_tilde_B` leaves it inside the adapter, and the smooth variation of
the effective potentials along a density sweep is what makes warm starts work.
DID declares kinetic for the same reason, having the same rearrangement term
plus a second one for the isospin dependence. SFHo (constant couplings,
`Sigma_R = 0`), vMIT, alphaBag, ZL and the ENJL branches (which carry both
rearrangement terms as unknowns with their defining rows) declare physical. The
matching row always compares the PHYSICAL potentials — read off the slot for a
physical-kind phase, restored at assembly for a kinetic one — so the two kinds
mix freely in one pair.

**The Phase declaration.** Concretely a pairing is two `Phase` records
(`eos.mixed.adapters`), each closing over its own model's parameters — for the
composite engine the pair IS the parameter argument, in the first position of
every public entry point, and DD2+vMIT is one pairing among the shipped ones
with `adapters.default_pair(par, flags, vmit_params)` as its named
constructor. Beyond
the adapter map (adapter) and the potential kind, a phase declares how it may
be seeded (a per-solve cache is FORBIDDEN for a branch-declared adapter,
because there the seed chooses the root — caching would change physics, not
speed), an optional cold start from its own equilibrium, its validity limits
(`supports_S`, `max_T`), and its optional capabilities: the pure per-mode wing
sweep the stitched hybrid table uses (`wing_sweep`), the frozen-composition
block the adiabatic sound speed uses (`frozen_thermo`), and an analytic
Jacobian block (`jacobian_block`). Capabilities are optional by design — a
phase without one makes the feature that needs it raise, naming the phase, and
the analytic Jacobian in particular is never generalized: each block
differentiates one model's own field equations, and the numeric Jacobian
remains the reference.

**The name of the surface.** In this repository the map (adapter) is
`thermo_from_mu`, in every model that ships one. Three models — dd2, sfho and
did — once spelled it `thermo_at_potentials`, and in sfho and did that name sat
above a lower evaluation layer already called `thermo_from_mu`, which takes the
solved mean fields as arguments as well as the potentials. The ruling this
document fixes is: **the contract surface is `thermo_from_mu`, and a lower
layer that also takes the fields is `thermo_from_fields`** — the name says what
the function takes, which is the distinction that matters, and it removes the
one-job-two-names split rather than freezing it. All three have since been
renamed to match, under their own package tickets.


## The eta family of constructions

Write the electron (and, when enabled, muon) gas as two populations. A *local*
population lives inside the structures and neutralises its own phase; a
*global* population is spread over the mixture and neutralises only the
average. They carry weights `eta` and `1-eta`:

    local:  mu_e^{L,H}, mu_e^{L,Q}      (weight eta)
    global: mu_e^{G}                    (weight 1-eta)          (leptonsplit)

Neutrality is then imposed twice, each where its population lives:

    local  (eta > 0):  n_C^H = n_l^H  and  n_C^Q = n_l^Q      (neutral_local)
    global (eta < 1):  (1-chi) n_C^H + chi n_C^Q = n_l^G     (neutral_global)

where `n_l` is the total negatively-charged lepton density of that domain. At
`eta = 1` only (neutral_local) survives: each phase is separately neutral, no
charge is exchanged between them, and the two phases can only coexist at one
pressure — the Maxwell plateau, with a genuine density jump. At `eta = 0` only
(neutral_global) survives: each phase may be charged, charge is exchanged
freely, and the pressure rises continuously through the window — Gibbs. In
between, both populations exist. Physically the intermediate case stands in for
the finite surface tension and Coulomb energy of the mixed-phase structures,
which suppress charge separation without forbidding it; here it is a controlled
interpolation, not a derivation from a surface tension.

Mechanical equilibrium carries the same weighting. Only the local leptons sit
inside the structures whose pressures must balance, so

    P^H + eta P_l^H = P^Q + eta P_l^Q                          (mechanical)

The global leptons, the photons and any trapped neutrinos are common to both
phases and cancel from (mechanical) identically.


## Conserved charges as a declaration

Each conserved quantity in `{B, C, S, L_e}` is treated in one of three ways:

- **GLOBAL** — the potential is shared, `mu_j^H = mu_j^Q`, and the charge is
  conserved in the volume average, `(1-chi) n_j^H + chi n_j^Q = Y_j n_B`;
- **LOCAL** — the potential is per-phase and the charge is conserved inside
  each phase separately;
- **NOT CONSERVED** — the charge is not conserved at all: its potential is
  eliminated from the unknown vector.

`B` is GLOBAL in every mode — `mu_B^H = mu_B^Q` is what makes the phases
coexist at all. The four named equilibrium modes are four assignments of the
remaining three charges:

| mode | B | C | S | L_e | independent variables |
|---|---|---|---|---|---|
| `beta_eq_neutrinoless`     | global | — | — | — | `(n_B, T)` |
| `beta_eq_neutrino_trapped` | global | — | — | global | `(n_B, Y_Le, T)` |
| `fixed_YC`                 | global | global | — | — | `(n_B, Y_C, T)` |
| `fixed_YC_YS`              | global | global | global | — | `(n_B, Y_C, Y_S, T)` |

("—" is NOT CONSERVED.) Nothing else in the engine enumerates modes. The
unknown vector, the residual list and the analytic Jacobian are all *assembled
by reading the regimes*, so the four named modes are four configurations of one
solver and an unnamed combination needs no new code.

A regime is not a third kind of declaration. A `ChargeSpec` is a `ModeSpec` —
the same object every single-phase model takes, saying which charges are held
and at what fractions — plus one `Locality` per charge, which is the only axis
a second phase adds. The regime is the two composed: a charge the mode does not
hold is NOT CONSERVED; one it holds is GLOBAL or LOCAL according to its
locality. So the four modes are declared once, in `eos/general/modes.py`, and
this engine says only *where* each conserved charge is conserved.

Three combinations are refused rather than mis-assembled, each with its own
message:

- **`S` LOCAL** — per-phase strangeness conservation is not wired; `fixed_YC_YS`
  conserves strangeness globally over H+Q.
- **`L_e` LOCAL** — not a defined mode at all: the neutrino mean free path is
  far larger than the mixed-phase structures, so neutrinos cannot be localised
  in one of them.
- **`L_e` GLOBAL together with `C` conserved** — trapped neutrinos are defined
  on top of beta-equilibrium charge, so combining a fixed `Y_C` with a fixed
  `Y_Le` is not a defined mode.

One further refusal is a declaration of a phase rather than of a mode: a
`ChargeSpec` conserving `S` globally raises if either phase declares
`supports_S = False`. A sector a phase does not implement is that adapter's
own refusal — `SpeciesFlags.sigma_star`, a hidden-strange scalar, is refused
by the DD2 adapter and by DD2's own flag object, not by the engine.


## The equilibrium system

Because the adapters have already absorbed each phase's internal
self-consistency, the unknowns here are only conserved-charge potentials,
`chi`, and the split lepton potentials — four to nine numbers, rather than the
full per-species `(mu_i, n_i)` vector a direct formulation would carry:

    x = ( mu_tilde_B^H, mu_B^Q, chi                    always
        , mu_C^H, mu_C^Q  (with leptons) or mu_C       C global
        , mu_S                                         S global
        , mu_nue                                       L_e global
        , mu_e^{L,H}, mu_e^{L,Q}                       eta > 0
        , mu_e^{G} )                                   eta < 1   (unknowns)

The residual rows are, in the order `residual()` assembles them:

    (1)  mu_B^H - mu_B^Q = 0                    baryon potentials match
    (2)  (1-chi) n_B^H + chi n_B^Q - n_B = 0    the average reproduces n_B
    (3)  P^H + eta P_l^H - P^Q - eta P_l^Q = 0  mechanical equilibrium

followed, by regime, by

    (4a) (1-chi) n_C^H + chi n_C^Q - Y_C n_B = 0            C global
    (4b) mu_C^H + eta mu_e^{L,H}
             - mu_C^Q - eta mu_e^{L,Q} = 0                  C global, leptons
    (5)  (1-chi) n_S^H + chi n_S^Q - Y_S n_B = 0            S global
    (6)  eta[(1-chi) n_e^H + chi n_e^Q]
             + (1-eta) n_e^G + n_nue - Y_Le n_B = 0         L_e global

and closed by the neutrality rows

    (7a) n_C^H - n_l^H = 0,  n_C^Q - n_l^Q = 0     leptons and eta > 0
    (7b) (1-chi) n_C^H + chi n_C^Q - n_l^G = 0     leptons and eta < 1

each present exactly when its lepton population is. Row (4b) follows (4a)
immediately — before the strangeness and lepton-number rows, not after them.

The charge-matching row (4b) is where the construction enters the charge
sector: at `eta = 0` it reduces to `mu_C^H = mu_C^Q`, the Gibbs statement that
the two phases share a charge potential, and at `eta = 1` the local electron
potentials shift it by exactly the amount that separate neutrality demands. The
global electron potential does not appear in it, because it neutralises the
average rather than either phase.

In beta equilibrium (`C` NOT CONSERVED) the charge potential is not an unknown
at all: it is eliminated by the weak condition applied with the `eta`-weighted
electron potential of that phase,

    mu_C^I = mu_nue - [ eta mu_e^{L,I} + (1-eta) mu_e^{G} ],
    I in {H, Q}                                                    (beta)

which for transparent matter (`mu_nue = 0`) is the repository's sign convention
`mu_C + mu_e = 0`, i.e. `mu_C = mu_p - mu_n`. Strangeness self-equilibrates in
that case, `mu_S = 0`.


## Thermodynamics of the mixture

The matter phases are volume-averaged; the local leptons carry weight `eta` and
are themselves volume-averaged; the global leptons carry weight `1-eta`;
photons and trapped neutrinos are uniform across the whole mixture and are
counted once:

    P   = P^H + eta P_l^H + (1-eta) P_l^G + P_nu + P_gamma        (Ptot)
    eps = (1-chi) eps^H + chi eps^Q
          + eta <eps_l> + (1-eta) eps_l^G + eps_nu + eps_gamma    (epstot)
    s   = (1-chi) s^H + chi s^Q
          + eta <s_l> + (1-eta) s_l^G + s_nu + s_gamma            (stot)

with `<X_l> = (1-chi) X_l^H + chi X_l^Q`. The pressure is read off one phase
rather than averaged because row (3) has made the two equal. That is the one
place `P` and `eps` are assembled differently, and it is why: the pressure is
uniform across the mixture by mechanical equilibrium, the energy density is
not.

Each lepton block is an ideal Fermi gas at its own potential, with
antiparticles: for a species of mass `m`, degeneracy `g = 2` (`g = 1` and
`m = 0` for neutrinos), `E = sqrt(k^2 + m^2)` and
`f_± = [1 + exp((E ∓ mu)/T)]^-1`,

    n   = g/(2 pi^2 (hc)^3) int_0^inf dk k^2      (f_+ - f_-)
    eps = g/(2 pi^2 (hc)^3) int_0^inf dk k^2 E    (f_+ + f_-)
    P   = g/(6 pi^2 (hc)^3) int_0^inf dk k^4 / E  (f_+ + f_-)
    s   = (eps + P - mu n)/T                                  (leptonkin)

evaluated through the same shared routines the phases use. Each lepton
population is evaluated at the potential its own neutrality row fixed:
`mu_e^{L,H}` and `mu_e^{L,Q}` for the local gases of (neutral_local),
`mu_e^{G}` for the global one of (neutral_global). The muon, where enabled, is
transparent: `mu_mu = mu_e - mu_nue` within each population, and `n_l` in
(neutral_local)-(neutral_global) is `n_e + n_mu`.

The photons are

    P_gamma = pi^2 T^4/(45 (hc)^3),  eps_gamma = 3 P_gamma,
    s_gamma = 4 pi^2 T^3/(45 (hc)^3),  mu_gamma = 0             (photons)

and they enter only when `SpeciesFlags.photons` is set, and only at `T > 0`.
The engine carries its own `SpeciesFlags` (`eos/mixed/species.py`) with the
six names of CLAUDE.md section 4, all defaulting to False. Every model carries
the same six names, so a caller may hand one model's flag object to both a
`Phase` and the mixture without translating it (`mixture_flags`).

The six split by where they are consumed. `hyperons`, `deltas`,
`thermal_mesons` and the `muons` of the lepton gases are sectors of the
models being coupled, and are delegated: each `Phase` carries its own model's
flags. `photons` and `thermal_neutrinos` belong to NEITHER phase — like the
eta-split leptons they are uniform across the mixture — and are consumed at
the mixture level, counted once. That is why no adapter's `thermo` — the
surface the mixture assembles from — adds a photon gas: the phases contribute
matter, the mixture contributes the radiation, and (photons) appears exactly
once in (Ptot)-(stot). A phase's `wing_sweep` takes the opposite rule and
carries the caller's own `photons`, because its rows are stitched into the
hybrid table as they stand with no mixture layer above them; the two paths
meet at n_offset, where chi = 1 and both describe the same matter, so a wing
short of the gas would put a spurious step of (photons) there.
`thermal_neutrinos` is carried and raises: the flavours a
mode does not track are not wired in the engine.

**Euler / Hugenholtz-Van Hove.** The identity

    eps + P = T s + sum_i mu_i n_i                              (euler)

must hold for the mixture as a whole, with the sum weighted exactly as
(epstot)-(stot) weight `eps` and `s`, and with `mu_gamma = 0`. Unlike the
pure-phase case this is not an algebraic identity: it holds only if every
thermal and lepton term has been weighted consistently, so it is the sharpest
single check on the assembly and it is asserted at 1e-8 relative
(`HVH_RTOL`) on every solved point.

There is no `n_s` and no `n_s = (eps - 3P)/m*` here, and no single-species
kinetic block beyond (leptonkin): the engine consumes each phase through
(adapter) and forms no scalar density of its own. The scalar densities and
effective masses live in the phases and are reported per phase in
`m_eff_i`; the identities that produce them are in the models' own documents.


## The two sound speeds

A first-order transition has two, and which is physical depends on how fast the
matter is disturbed relative to the rate at which one phase converts into the
other.

The *equilibrium* speed `cs2_eq = dP/deps` is taken along the solved sequence,
with `chi` free to readjust: a compression is answered by converting hadrons
into quarks rather than by raising the pressure. Through a Maxwell window it
therefore vanishes identically, and through a Gibbs window it dips but stays
finite. This is the speed that enters the TOV equations and whose causality
bound `0 <= c^2 <= 1` is checked before any table is integrated.

The *frozen* (adiabatic) speed `cs2_frozen` holds `chi` fixed. The mixture is
compressed faster than the phases can convert, the pressure has to rise, and it
does not collapse in the window. Freezing `chi` is the part that matters:
freezing only the charge fractions would let the solve readjust `chi` and
return to the plateau. Which *compositional* variables are held alongside it is
the further choice that separates the fast and slow propagation limits of
Constantinou et al. — every particle fraction in the first, only the lepton
fraction in the second, since there every chemical equilibrium but the beta one
is imposed before differentiating. In addition each phase is compressed by the
same factor, each keeps its own `Y_C` and `Y_S`, and the leptons are
re-neutralised against the frozen total charge. That convention is stated in
full in `eos/mixed/responses.py`, because a different one gives a different
number; with `chi = 0` and the leptons switched off it reduces exactly to the
pure-hadronic adiabatic sound speed of `eos/dd2`.

The gap between the two is what drives a composition g-mode across the
transition.


## What a mixed table reports

A composite engine returns more than a point equation of state. `eos_table`
returns `(rows, windows)`: the long-format rows — bulk thermodynamics, `chi`,
the phase label, and every conserved charge resolved by phase — and the
`Window` per (temperature, fraction) line. The windows are part of the result
rather than something the caller recovers afterwards by scanning the rows for
where `chi` crossed 0 and 1.

A `Window` carries

| field | meaning |
|---|---|
| `n_onset` | the density at which `chi` reaches 0 — the last hadronic point |
| `n_offset` | the density at which `chi` reaches 1 — the first pure quark point |
| `probes` | the solved points used to find the boundaries, kept so a caller can reuse them |
| `onset_state`, `offset_state` | the converged fixed-`chi` solves AT the boundaries when the window was refined exactly; `None` when the boundary is only a bisected estimate. They carry the full unknown vector at `chi = 0` and `chi = 1`, which is what seeds a neighbouring temperature's boundary search and the window sweep's first point |

Both densities are `nan` when there is no transition on the grid — a physics
outcome for those parameters, not a failure. `exists` is True only for a
well-ordered window (`n_offset > n_onset`), because `chi` is a solved quantity
rather than a monotone parameter and a sparse or noisy probe set can bracket
the two crossings out of order. `reason` reports *why* it is False as a
distinct label, since the four outcomes are not interchangeable and reporting
them as one would cost a scan the ability to tell physics from failure:

    ok                      exists is True
    no_transition           chi never crossed either target on this grid
    onset_unbracketed       chi = 1 was located, chi = 0 was not
    offset_unbracketed      chi = 0 was located, chi = 1 was not
    crossings_out_of_order  both located, but offset <= onset

The per-phase decomposition satisfies three cheap invariants at every solved
point,

    Y_B^H + Y_B^Q = 1,   Y_C^H + Y_C^Q = Y_C,   Y_S^H + Y_S^Q = Y_S
                                                             (partition)

on the volume-weighted convention `Y_j^I = w_I n_j^I / n_B` with `w_H = 1-chi`
and `w_Q = chi`. The partition is what the global sums cannot show: at fixed
total `Y_C` the hadronic phase can be far more positively charged than the
average while the quark phase carries the compensating negative charge, and how
far that separation goes is exactly what `eta` controls.

A solved point (`Result`) carries `converged`, `error`, `n_B`, `T`, `eta`,
`chi`, the two `PhaseThermo` blocks `th_H` and `th_Q`, the solved unknown-vector
slots as `potentials`, the matched physical `mu_B`, the totals `P`, `eps`, `s`
of (Ptot)-(stot), and an `extras` dictionary. `in_mixed_phase` is
`0 < chi < 1`, and `phase` is `'H'`, `'mix'` or `'Q'`.


### The complete hybrid equation of state

The stitched table (`build_hybrid_table`, and the mode-facing entry
`hybrid_table`) covers the whole density range with three segments cut on
`chi`,

    n_B < n_onset                : pure hadronic
    n_onset <= n_B <= n_offset   : eta-mixed
    n_B > n_offset               : pure quark

and the *whole* hybrid is at one equilibrium: the mode the charge declaration
fixes holds in the wings and in the window alike. If `Y_C` is fixed, every
segment is solved at that `Y_C`; if neutrinos are trapped at `Y_Le`, both wings
trap them too. The wings are the pure models' own per-mode solves — `eos.dd2`'s
octet solver in the declared mode below the onset, `eos.vmit`'s mode solvers
above the offset — dispatched from the same regime assignment that shapes the
mixed system, so a regime combination the window can solve gets a consistent
wing without new code. Only the neutrality locality `eta` is specific to the
mixed region: a pure phase has a single phase to neutralize, so there is no
local/global distinction left for `eta` to interpolate, and the leptonless
fixed-`Y_C` hybrid is a charged slice whose window is `eta`-independent.

Because the segments are cut on `chi` rather than matched on pressure, they
meet by construction; before the table reaches a structure solver, round-off
pressure inversions on a Maxwell plateau are clamped — a drop is round-off when
`|dP| <= 1e-12 x scale` (`_P_ROUNDOFF`) — and any larger inversion, mechanical
instability no construction has resolved, is refused.


## The API surface

Three entry points, with the signatures every model in this repository carries,
plus the two the composite engine adds:

    eos_point(phases, mode, species, n_B=, T=|SnB=, eta=0.0,
              leptons=True, x0=, analytic_jac=, check_consistency=True,
              **conditions)
    eos_table(phases, mode, species, axes=, eta=0.0, fixed=, leptons=True,
              window_only=True, analytic_jac=, refine="exact",
              progress=, verbose=)
    eos_response(phases, mode, species, frozen="equilibrium", n_B=, T=0.0,
                 eta=0.0, leptons=True, rel_dn=1e-3, **conditions)

`phases` is the pairing — two `Phase` objects — and it occupies the position
`par` occupies in a single-phase model, because for a composite engine the pair
IS the parameter argument. There is no second signature: a caller who wants the
DD2+vMIT hybrid writes `phases=adapters.default_pair(par, flags, vmit_params)`,
and `phases=(sfho_phase(...), njl_phase(...))` is written the same way.
`species` is the ENGINE's own `eos.mixed.SpeciesFlags` — the phase-common
sectors (photons) and the muons of the eta-split lepton domains; the per-phase
sectors travel inside each `Phase`, in that model's own flag object.

`n_B` is always the TOTAL baryon density of the mixture, volume-averaged over
both phases, not the density of either one. `x0` is a warm start in the slot
order of `mixed_slots(spec, eta, phases)`.

`eos_table` takes `axes = {'nB': grid, exactly one of 'T'/'SnB': grid, and
optionally any of 'Y_C'/'Y_S'/'Y_Le' to sweep that fraction}`. The density axis
is warm-started within each line. `window_only=True` solves the mixed system
only between the located boundaries, where it is the only thing that can
answer; outside, the far cheaper pure-phase solvers give the same state — set
it False to solve at every grid point, e.g. when studying `chi` outside [0, 1].
`refine` chooses the boundary resolution: `"exact"` finishes each boundary with
one fixed-`chi` solve and marches those solves along the temperature axis,
`"bisect"` stops at half a grid spacing. The `progress` callback is invoked
once per completed line with the repository's standard dictionary plus this
engine's two extras,

    {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
     elapsed_s} + {eta, window}

and `verbose=True` installs the built-in printer.

`eos_response` implements two freezes, and every result carries `chi` and
`phase` alongside them, because in a composite engine which regime the point is
in decides which numbers in the dict mean anything:

- `frozen="equilibrium"` — everything re-equilibrates. Central differences over
  a relative density step `rel_dn` give `cs2_eq = dP/deps` at fixed `T`, and
  `C_V = (T/n_B) ds/dT` at fixed `n_B`, the latter only at `T > 0`. Through a
  Maxwell window the pressure is constant along that sequence, so `cs2_eq` goes
  to zero: that is the physics, not a failure. **Both are defined only inside
  the coexistence window.** Outside it the mixed system still has a root — `chi`
  runs negative or past one — but that root is an analytic continuation, not
  the state: at `eta = 1` it sits on the pressure plateau at every density, so
  differentiating it returns zero for matter whose real sound speed is not
  small. Both quantities are therefore `nan` at a pure-phase point, and the
  equilibrium response there comes from that phase's own `eos_response`.
- `frozen="chi"` — the quark volume fraction is held, and with it each phase's
  charge and strangeness fractions; `leptons` chooses whether the mixture is
  re-neutralised under the perturbation. Returns `cs2_frozen`, defined at every
  density: outside the window `chi` is clipped and the answer is the pure
  phase's own frozen sound speed, so the curve joins continuously at the
  boundaries.

The remaining freezes of the specification — frozen per-species composition,
frozen conserved fractions — raise `NotImplementedError` naming the gap, and
are recorded in `docs/DEFERRED.md`.


## Numerics

**Scaling and the gate.** Every residual row is made dimensionless before it is
judged: density rows are divided by `n_scale = max(n_B, 0.01) fm^-3`, potential
equalities by the fixed `mu_scale = 100.0 MeV`, and the mechanical row by
`max(|P^H + eta P_l^H|, |P^Q + eta P_l^Q|, 1.0) MeV/fm^3`. A state is accepted
when the largest scaled residual is below 1e-10. The rows carry mixed units —
densities of order 1e-1 fm^-3 against potentials of order 1e3 MeV — so a gate
on the raw norm would be dominated by whichever row happens to be largest and
would accept states satisfying the others only loosely.

**Warm starts.** These dominate the cost, because the adapters are called once
per residual evaluation. Three levels are used, in order of what they buy: the
hadronic phase's starting field configuration is a constant of the solve and is
computed once; each density seeds the next along the stiff `n_B` axis, with a
missed step bisected up to `max_bisect = 6` levels; and each isotherm seeds the
next. On top of that the mixed system is solved only *inside* the located
window, and the window search on one isotherm is told where the previous
isotherm found its boundaries, since they move smoothly with `T`. A hint that
produces a boundary on the edge of the hinted range is discarded and the full
search repeated, so a window that genuinely disappears is still reported as
gone.

The cold start is physical: each phase is placed at its OWN pure equilibrium
point at the target density, which matches the `eta = 1` structure — where each
phase is separately neutral — far better than a single shared `mu_e` would, and
is a sound seed at `eta = 0` too; `chi` starts mid-window. A pair whose first
phase declares no cold start raises rather than guessing, and the caller passes
`x0`.

**Exact boundaries by the fixed-chi solve.** A phase boundary is a `chi`
crossing, and the crossing can be solved for directly: impose `chi` and move
`n_B` into the unknown vector in its place,

    x = ( mu_tilde_B^H, mu_B^Q, n_B, ... )   at given chi

with every residual row of the equilibrium system unchanged — only which symbol
is solved for moves. `chi = 0` then returns `n_onset` and `chi = 1` returns
`n_offset`, each in one solve, exactly, with no grid resolution in the answer
(a normal solve at the reported onset gives `chi` at round-off). The scan stays
as the cold-start finder — the fixed-`chi` system can walk to the wrong root
from far away — so the scan decides *which* root and the exact solve decides
*where*: the onset solve is seeded from the lowest mixed probe, the offset
solve from the converged onset state, and a solve that fails or leaves the
probes' bracket is rejected in favour of the bisected estimate. The scan itself
uses `n_probe = 12` probes per line by default, refined up to `max_refine = 2`
times (each refinement adding `3 x n_probe` probes on the bracketing
subinterval) when no hint is available, and the bisection that backs it is
bounded by `MAX_WALK = 64` steps. Along a temperature axis the scan is then
needed only twice: once two isotherms carry converged boundary states, the next
isotherm's boundaries are two warm-started fixed-`chi` solves seeded by linear
extrapolation of the entire boundary vector in `T`, with the scan as fallback
whenever that march declines. Only converged states enter the extrapolation
history, so a failed isotherm cannot poison it.

**Jacobian.** The reference path uses the solver's own numeric Jacobian and is
the correctness oracle. A hand-assembled analytic Jacobian is available as the
fast path, validated against finite differences and required to reach the same
root; where a trial point drives a phase solve out of its domain, both paths
answer with a large penalty residual so the outer iteration backs off instead
of aborting. The analytic path lives in `backends/` and is deletable: a pairing
in which either phase advertises no analytic block silently uses the numeric
Jacobian, because `analytic_jac=True` asks for the fast path where one exists,
not for an error where none does.

**Non-convergence.** Reported, never raised, at the public boundary: the engine
is used inside parameter scans that walk into regions where no hybrid equation
of state exists, and a scan must be able to score such a point and move on. A
converged point outside [0, 1] in `chi` is not a failure — it is how the engine
says which side of the transition the density is on.


## Not implemented

Recorded in `docs/DEFERRED.md`: the frozen-per-species and
frozen-conserved-fraction response freezes; `S` LOCAL; combining a fixed `Y_C`
with a fixed `Y_Le`; `SpeciesFlags.sigma_star` in the hadronic phase; a Bose
condensate in either phase (refused through `condensation`); and
`SpeciesFlags.thermal_neutrinos`, the flavours a mode does not track, which
the engine carries and refuses.


## References

- N. K. Glendenning, Phys. Rev. D 46, 1274 (1992) — the Gibbs construction for
  a two-conserved-charge transition.
- C. Constantinou, T. Zhao, S. Han and M. Prakash, Phys. Rev. D 107, 074013
  (2023), and its finite-temperature successor — the eta interpolation and the
  fast/slow propagation limits.
- S. Typel et al., Phys. Rev. C 81, 015803 (2010); M. Hempel and
  J. Schaffner-Bielich, Nucl. Phys. A 837, 210 (2010) — DD2.
- A. Chodos et al., Phys. Rev. D 9, 3471 (1974); R. O. Gomes et al.,
  Astrophys. J. 877, 139 (2019) — the bag and its vector extension, vMIT.

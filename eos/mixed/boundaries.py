"""
mixed/boundaries.py
===================
Location of the mixed-phase window: where the first-order transition sits on a
density line.

*Public API* (re-exported from `eos.mixed`): `Window`, `locate_window`.

Solving the mixed system at a density that turns out to be pure phase is
wasted work, and most of a realistic grid is pure phase. `locate_window` finds
the two boundaries first, using chi as the indicator (chi <= 0 hadronic,
chi >= 1 quark, in between mixed), so the expensive mixed solve runs only
where it is the answer.

There is a trap in that point worth stating plainly, because it produced
silently missing transitions. chi is only readable where the mixed system has a
solution, and below the onset it has none — so a density down there does not
report chi <= 0, it drops out of the sweep entirely. The probe set can
therefore begin *above* the onset with every chi > 0, whereupon no sign change
exists to bracket and a real transition is reported as absent. `locate_window`
answers that by walking to the boundary from a point known to be inside the
window (`walk_to_crossing`), where a step that will not converge is itself the
hadronic side rather than missing information.

The empirical bracketing heuristics live here rather than in the residual,
mirroring the split between physics and continuation.
"""
from dataclasses import dataclass

import numpy as np

from scipy.optimize import brentq, root

from eos.mixed.thermodynamics import charged_leptons
from eos.mixed.solver import (
    RESIDUAL_TOL, build_mixed_ctx, default_guess, mixed_slots, residual,
    result_from_root, sweep, warm_start,
)

#: Most single-tolerance steps `locate_window` will take when it has to walk to
#: boundary -- so this is a cost ceiling, not a working limit.
MAX_WALK = 64


@dataclass
class Window:
    """Where the first-order transition sits on one density line.

    n_onset  : density at which chi reaches 0 — the last hadronic point
    n_offset : density at which chi reaches 1 — the first pure quark point
    Both are nan when there is no transition on the grid (the quark phase never
    becomes favourable for these parameters, which is a physics outcome, not a
    failure). `probes` keeps the solved points used to find the boundaries so a
    caller can reuse them.

    `onset_state` / `offset_state` are the converged fixed-chi solves AT the
    boundaries when the window was refined exactly (`refine="exact"`), and
    None when the boundary is only a bisected estimate. They carry the full
    unknown vector at chi = 0 and chi = 1, which is what seeds a neighbouring
    temperature's boundary search and the window sweep's first point.
    """
    n_onset: float
    n_offset: float
    probes: list
    onset_state: object = None
    offset_state: object = None

    @property
    def exists(self):
        """True only for a well-ordered window. chi is a solved quantity, not a
        monotone parameter, so a sparse or noisy probe set can bracket the two
        crossings out of order; that is not a transition, it is a failed
        location, and callers must not treat it as one."""
        return (np.isfinite(self.n_onset) and np.isfinite(self.n_offset)
                and self.n_offset > self.n_onset)

    @property
    def reason(self):
        """Why `exists` is False, as a distinct label; 'ok' when it is True.

        The four outcomes below are not interchangeable, and reporting them as
        one label costs the caller the ability to tell physics from failure: a
        parametrization with no transition and one whose boundary was never
        bracketed both look like 'no window' in a scan table, so a scan cannot
        say how much of its reject count is real.

          no_transition          chi never crossed either target on this grid
          onset_unbracketed      chi = 1 was located, chi = 0 was not
          offset_unbracketed     chi = 0 was located, chi = 1 was not
          crossings_out_of_order both located, but offset <= onset
        """
        if self.exists:
            return "ok"
        on, off = np.isfinite(self.n_onset), np.isfinite(self.n_offset)
        if on and off:
            return "crossings_out_of_order"
        if off:
            return "onset_unbracketed"
        if on:
            return "offset_unbracketed"
        return "no_transition"

    def contains(self, n_B):
        return self.exists and self.n_onset <= n_B <= self.n_offset


def locate_window(phases, n_B_grid, eta, spec, T=0.0,
                  n_probe=12, tol=None, analytic_jac=False, x0=None,
                  hint=None, max_refine=2, refine="exact", species=None):
    """Find the mixed window on `n_B_grid` by bracketing the chi crossings.

    Probes the grid coarsely, reading chi as the regime indicator (chi <= 0
    hadronic, chi >= 1 quark), then bisects the chi = 0 and chi = 1 crossings
    to `tol` (default: half a grid spacing). In exchange for a couple of dozen
    solves, the caller can then skip the mixed system everywhere outside the
    window, which on a realistic density grid is most of it.

    `refine` chooses how far each located boundary is sharpened:
    `"exact"` (the default) finishes with one fixed-chi solve per boundary
    (`refine_window`), so the reported density is the chi crossing itself,
    with no grid resolution in the answer; `"bisect"` stops at the scan's
    resolution (half a grid spacing).

    `hint` is an optional (n_lo, n_hi) span to concentrate the probes in, for
    when a neighbouring temperature or eta has already shown roughly where the
    transition sits. Do not assume a window at one eta brackets the window at
    another: raising eta narrows the window but also moves it, sometimes to
    lower density, so a hint taken from a different eta should be generous.

    Without a hint the probe set is refined up to `max_refine` times, each time
    tripling the density of probes over the sub-range that brackets the
    crossings, which recovers narrow windows at the cost of a few more solves.
    """
    grid = np.asarray(n_B_grid, dtype=float)
    if grid.size < 2:
        return Window(np.nan, np.nan, [])
    if tol is None:
        tol = 0.5 * float(np.min(np.diff(np.sort(grid))))
    slots = mixed_slots(spec, eta, phases)

    def scan(lo, hi, count):
        """Solve at `count` densities across [lo, hi], warm-started along the way.

        Delegates to `sweep` so the probes get the same continuation a
        table sweep gets: a step that misses is bisected rather than dropped,
        and the warm start survives a point that is skipped anyway.

        That matters more here than it looks. The probe spacing is the whole
        grid divided by `n_probe`, which is a far bigger step than any table
        sweep takes, and a step that large inside the window regularly misses.
        Dropping those points loses every probe above the first failure, so the
        chi >= 1 side of the transition is never seen, the refinement below
        concludes there is no transition, and the offset falls through to the
        last probe that happened to converge -- a boundary that then jumps
        around from one temperature to the next.
        """
        return sweep(phases, np.unique(np.linspace(lo, hi, count)),
                           eta, spec, T=T, x0=x0,
                           analytic_jac=analytic_jac, species=species)

    lo, hi = (float(grid[0]), float(grid[-1])) if hint is None else (
        max(float(grid[0]), float(hint[0])), min(float(grid[-1]), float(hint[1])))
    probes = scan(lo, hi, min(n_probe, grid.size))

    # Narrow the search to the stretch that actually straddles the transition,
    # then re-probe it more finely. A window thinner than the coarse spacing is
    # invisible to the first pass but obvious to the second.
    for _ in range(max_refine if hint is None else 0):
        partial = [r for r in probes if 0.0 < r.chi < 1.0]
        below = [r.n_B for r in probes if r.chi <= 0.0]
        above = [r.n_B for r in probes if r.chi >= 1.0]
        if partial and below and above:
            break                              # both crossings already bracketed
        if not (below and above):
            break                              # no transition on this grid at all
        sub_lo, sub_hi = max(below), min(above)
        if sub_hi <= sub_lo or (sub_hi - sub_lo) <= tol:
            break
        probes += scan(sub_lo, sub_hi, 3 * min(n_probe, grid.size))

    if not probes:
        return Window(np.nan, np.nan, [])

    def bisect(target, lo, hi):
        """Density at which chi crosses `target`, bracketed by (lo, hi).

        Each midpoint is reached by continuation from the low end of the
        bracket, not by a lone solve. The starting bracket is a full coarse
        probe spacing wide -- a tenth of the grid -- and a single solve across
        a gap that size misses often, especially deep in the window.

        A midpoint that still will not converge returns nan rather than the
        current bracket midpoint. Accepting the midpoint looks like a graceful
        degradation and is not: it reports a boundary up to half a probe
        spacing wrong *as if it had converged*, and since the error depends on
        where the probes happened to land it moves unpredictably from one
        temperature to the next. nan says the crossing was bracketed but not
        located, which is what actually happened.
        """
        r_lo, r_hi = lo, hi
        while (r_hi.n_B - r_lo.n_B) > tol:
            n_mid = 0.5 * (r_lo.n_B + r_hi.n_B)
            stepped = sweep(phases, [r_lo.n_B, n_mid], eta, spec, T=T,
                                  x0=warm_start(r_lo, slots),
                                  nH0=r_lo.th_H.n_B,
                                  analytic_jac=analytic_jac, species=species)
            if not stepped or abs(stepped[-1].n_B - n_mid) > 1e-12:
                return np.nan
            r = stepped[-1]
            probes.append(r)
            if (r.chi - target) * (r_lo.chi - target) > 0.0:
                r_lo = r
            else:
                r_hi = r
        return 0.5 * (r_lo.n_B + r_hi.n_B)

    def crossing(target, above=None):
        """Bracket the first chi = `target` crossing among the probes.

        `above` restricts the search to densities beyond a boundary already
        found, so the offset is always looked for on the quark side of the
        onset. Without that the two crossings can be picked up in the wrong
        order — chi is a solved quantity, not a monotone parameter, and a
        coarse probe set can straddle both crossings inside one interval.
        """
        ordered = sorted(probes, key=lambda r: r.n_B)
        if above is not None and np.isfinite(above):
            ordered = [r for r in ordered if r.n_B >= above]
        for a, b in zip(ordered, ordered[1:]):
            if (a.chi - target) * (b.chi - target) <= 0.0:
                return bisect(target, a, b)
        return np.nan

    def walk_to_crossing(target, direction):
        """Step from the nearest mixed probe towards the chi = `target`
        crossing, one `tol` at a time, when the probe set never bracketed it.

        This covers the case the probe scan is blind to. `sweep` drops a
        density it cannot solve, and below the onset the mixed system has no
        solution at all, so the probe set that comes back can *begin above the
        onset* with every chi > 0. `crossing` then finds no sign change and a
        transition that plainly exists — its chi = 1 side is located — is
        reported as absent. Which side of the lowest surviving probe the onset
        falls on decides it, so the outcome moves with the parametrization, the
        temperature and the probe count, for no physical reason.

        Walking sees the boundary either way: chi crosses the target, or the
        solve stops converging. Going down those are the same event, because
        the formal mixed solution ceases to exist on the hadronic side — that
        is what a dropped probe down there means. Going up, a failure is only a
        failure; a missing pure-quark boundary is not evidence of one, so it
        returns nan rather than inventing an offset.

        Steps are `tol` and warm-started, which is the regime the mixed solve
        is reliable in, and the answer lands already refined to `tol`, so no
        bisection follows.
        """
        inside = [r for r in probes if r.in_mixed_phase]
        if not inside:
            return np.nan
        r = (min(inside, key=lambda p: p.n_B) if direction < 0
             else max(inside, key=lambda p: p.n_B))
        for _ in range(MAX_WALK):
            n_next = r.n_B + direction * tol
            if not (lo <= n_next <= hi):
                return np.nan                  # ran off the grid, not a crossing
            stepped = sweep(phases, [r.n_B, n_next], eta, spec, T=T,
                                  x0=warm_start(r, slots), nH0=r.th_H.n_B,
                                  analytic_jac=analytic_jac, species=species)
            if not stepped or abs(stepped[-1].n_B - n_next) > 1e-12:
                return 0.5 * (n_next + r.n_B) if direction < 0 else np.nan
            r = stepped[-1]
            probes.append(r)
            if (r.chi <= target) if direction < 0 else (r.chi >= target):
                return r.n_B - 0.5 * direction * tol
        return np.nan

    n_onset = crossing(0.0)
    if not np.isfinite(n_onset):
        n_onset = walk_to_crossing(0.0, -1)
    n_offset = crossing(1.0, above=n_onset)
    if not np.isfinite(n_offset):
        n_offset = walk_to_crossing(1.0, +1)
    # A crossing may be missing because the grid begins or ends inside the
    # window rather than because there is no transition; fall back to the
    # extent of the probes that actually came out mixed.
    #
    # Only when the probes reached that end of the range, though. If the solves
    # stopped converging partway up, the last mixed probe is where the SOLVER
    # gave up, not where the phase boundary is, and returning it turns a
    # convergence failure into a confident wrong number -- one that moves with
    # the probe spacing and so wanders from one temperature to the next. nan
    # here means `exists` is False and the caller reports no window, which is
    # the honest answer when the boundary was never bracketed.
    mixed = sorted(r.n_B for r in probes if r.in_mixed_phase)
    if mixed:
        reached = sorted(r.n_B for r in probes)
        # "The grid ended inside the window" means two things at once: the
        # probes got to that end of the range, AND that side of the transition
        # was never seen. If a chi >= 1 probe exists, the offset was bracketed
        # and merely failed to refine, and the last mixed probe is nowhere near
        # it -- so the fallback must not fire.
        if (not np.isfinite(n_onset) and reached[0] <= lo + tol
                and not any(r.chi <= 0.0 for r in probes)):
            n_onset = mixed[0]
        if (not np.isfinite(n_offset) and reached[-1] >= hi - tol
                and not any(r.chi >= 1.0 for r in probes)):
            n_offset = mixed[-1]
    window = Window(float(n_onset), float(n_offset), probes)
    if refine == "exact":
        window = refine_window(window, phases, eta, spec, T=T,
                               species=species)
    elif refine != "bisect":
        raise ValueError(f"refine must be 'bisect' or 'exact', got {refine!r}")
    return window


def solve_fixed_chi(phases, chi, eta, spec, T=0.0,
                    n_B0=None, x0=None, check_consistency=False,
                    species=None):
    """The density at which the mixed system has volume fraction `chi`:
    chi is IMPOSED and the total density n_B is solved for.

    Same residual as `solve` with one slot swapped
    (`mixed_slots(..., fixed_chi=True)`), so chi = 0 returns the onset and
    chi = 1 the offset of the coexistence window, each in one solve, exactly —
    no grid resolution enters the answer. The idea is salvaged from the
    first-generation hybrid codes (docs/SALVAGE.md section 1).

    n_B0 : required starting density — the unknown must start somewhere, and
           it also seeds the hadronic phase's internal solve.
    x0   : warm start in the slot order of
           `mixed_slots(spec, eta, phases, fixed_chi=True)`. Without one the
           physical cold start is used, which is reliable only near the
           transition: the fixed-chi system can walk to the wrong root from
           far away, which is why the scan of `locate_window` stays the
           cold-start finder and this solve is its finisher.

    Returns a `Result` whose `n_B` is the located boundary; raises
    RuntimeError when no guess meets the residual gate.
    """
    if n_B0 is None:
        raise ValueError("n_B0 is required: the density is the unknown and "
                         "its solve must start somewhere")
    ctx = build_mixed_ctx(spec, eta, None, phases, T=T,
                          n_B_guess=float(n_B0), chi=float(chi),
                          species=species)

    def guesses():
        if x0 is not None:
            yield list(x0)
        yield default_guess(ctx)

    sol = None
    for guess in guesses():
        ctx.cache.clear()          # a fresh guess invalidates the phase cache
        sol = root(residual, guess, args=(ctx,), method="hybr", tol=1e-12)
        res_max = max(abs(r) for r in residual(sol.x, ctx))
        if res_max <= RESIDUAL_TOL:
            break
    else:
        raise RuntimeError(
            f"fixed-chi solve failed at chi={chi}, T={T}, eta={eta} "
            f"(started from n_B0={n_B0}): {sol.message} "
            f"(max residual {res_max:.2e}, tol {RESIDUAL_TOL:.0e})")
    return result_from_root(sol.x, ctx, res_max,
                            check_consistency=check_consistency)


def refine_window(window, phases, eta, spec, T=0.0, species=None):
    """Sharpen a scan-located window with one exact fixed-chi solve per
    boundary.

    The division of labour: the scan decides WHICH root — its probes establish
    that a transition exists and roughly where — and the exact solve decides
    WHERE, by imposing chi = 0 (onset) and chi = 1 (offset) and solving for
    the density. Each solve is seeded from the scan's own probes: the onset
    from the lowest mixed probe, the offset from the converged onset state
    (the best predictor across the window, docs/SALVAGE.md section 2), falling
    back to the highest mixed probe.

    A solve that fails, or lands outside the bracket the probes established,
    is rejected and the bisected estimate kept — the exact solve is an
    accelerator on top of the scan, never a second opinion against it. A
    boundary the scan never located (nan) is left alone.

    Returns a new `Window` carrying `onset_state` / `offset_state` for
    the boundaries that were refined.
    """
    mixed = [p for p in window.probes if p.in_mixed_phase]
    if not mixed:
        return window
    slots = mixed_slots(spec, eta, phases, fixed_chi=True)

    def seed(point, n_scan):
        # The probe's potentials, with the scan estimate in the n_B slot.
        filled = dict(point.potentials, n_B=n_scan)
        return [filled[name] for name in slots]

    def accepted(n_exact, n_scan):
        # Within the tightest probe bracket around the scan estimate, widened
        # by its own width: the bisection narrowed that bracket to its
        # tolerance, so an exact root belonging to the same crossing cannot
        # be far outside it.
        below = [p.n_B for p in window.probes if p.n_B <= n_scan]
        above = [p.n_B for p in window.probes if p.n_B >= n_scan]
        lo = max(below) if below else n_scan
        hi = min(above) if above else n_scan
        margin = max(hi - lo, 1e-3)
        return (lo - margin) <= n_exact <= (hi + margin)

    def refine(chi_target, n_scan, x0, n_B0):
        try:
            r = solve_fixed_chi(phases, chi_target, eta, spec, T=T,
                                n_B0=n_B0, x0=x0, species=species)
        except (RuntimeError, ValueError):
            return None
        return r if accepted(r.n_B, n_scan) else None

    n_onset, onset_state = window.n_onset, None
    if np.isfinite(window.n_onset):
        probe = min(mixed, key=lambda p: p.n_B)
        # chi = 0: the hadronic phase IS the mixture, so the scan estimate
        # seeds the phase-internal solve too.
        onset_state = refine(0.0, window.n_onset,
                             seed(probe, window.n_onset), window.n_onset)
        if onset_state is not None:
            n_onset = onset_state.n_B

    n_offset, offset_state = window.n_offset, None
    if np.isfinite(window.n_offset):
        hi_probe = max(mixed, key=lambda p: p.n_B)
        base = onset_state if onset_state is not None else hi_probe
        # chi = 1: the hadronic phase has thinned out and its density no
        # longer tracks n_B, so its internal solve is seeded from the deepest
        # mixed probe's own hadronic density.
        offset_state = refine(1.0, window.n_offset,
                              seed(base, window.n_offset), hi_probe.th_H.n_B)
        if offset_state is not None:
            n_offset = offset_state.n_B

    return Window(float(n_onset), float(n_offset), window.probes,
                       onset_state, offset_state)


# ---------------------------------------------------------------------------
# the Maxwell branch crossing (eta = 1)
# ---------------------------------------------------------------------------
# At eta = 1 the two phases exchange no charge: each is separately neutral, they
# coexist at one pressure, and the density jumps. That construction needs no
# mixed system and no chi -- there is no mixture at the transition, only two
# states of the same matter at one (mu_B, P). It is two one-dimensional solves:
# neutralize each phase against its own leptons, then find the mu_B at which the
# total pressures cross.
#
# `locate_window` above is the other construction: eta < 1 puts a genuine mixed
# region on the density axis and chi is what locates its edges. Both are in this
# module because both answer "where is the transition"; they do not share code
# because they do not share a physical picture.
#
# Everything here takes a `phase` CALLABLE, (mu_B, mu_C) -> PhaseThermo, so it
# is model-agnostic: a DD2/vMIT pairing supplies two different adapters, and an
# ENJL pairing supplies `eos.mixed.adapters.enjl_phase` twice at two different
# declared branches.

#: Range of mu_C scanned for the neutralizing value, and how finely. Charge
#: neutrality puts mu_e between roughly 20 and 300 MeV over the density range
#: any of these models is used at; the scan exists rather than a bracket
#: because a branch does not necessarily EXIST across the whole span, and a
#: bracket built from its endpoints would fail on the endpoint that is missing.
#: Kept coarse deliberately: the residual is smooth in mu_C and the bisection
#: below refines it, while every extra scan point off the branch costs a full
#: failed seed ladder, which is the dominant cost of this construction.
#:
#: The upper end must stay close to zero. Nearly symmetric quark matter is
#: nearly neutral on its own, so it neutralizes at a very small mu_e: the
#: deconfined branch of the f_q = 0.7, B = 0 set does it at mu_C = -7.9 MeV.
#: A scan that stops at -20 MeV reports that branch as non-existent and the
#: deconfinement transition as absent -- which is what it did, until this
#: comment was written.
MU_C_SCAN = (-320.0, -2.0, 15)

#: A residual this close to zero [fm^-3] IS neutrality, whether or not the
#: residual changes sign around it. Fifteen orders below the densities in play,
#: so it can only fire where a phase carries no net charge to begin with; see
#: the tangential root in `neutral_phase`.
NEUTRAL_TOL = 1.0e-8


@dataclass
class MaxwellPoint:
    """One first-order transition located at eta = 1.

    P        : the coexistence pressure, matter + leptons [MeV/fm^3]
    g        : the coexistence Gibbs free energy per baryon [MeV],
               g = (eps + P) / n_B
    mu_B_lo, mu_B_hi : each branch's OWN baryon potential [MeV]
    n_B_lo   : baryon density of the low-density branch [fm^-3]
    n_B_hi   : baryon density of the high-density branch [fm^-3]
    mu_e_lo, mu_e_hi : each branch's own neutralizing electron potential
               [MeV], nan where the phases carry no leptons
    branches : the two branch labels, low then high

    `n_B_lo` and `n_B_hi` are the window edges: between them no single phase is
    stable and the delivered EoS is the constant-pressure plateau.

    **P and g are the pair that coexistence equates, in EVERY closure**, and
    they are the two fields here that carry one value rather than two. The
    baryon potential is not that pair: at fixed composition Euler gives

        g = mu_B + Y_C mu_C + Y_S mu_S

    and each phase solves its own (mu_C, mu_S) for the held fractions, so the
    two branches meet at one g with two different mu_B. What collapses the
    pair to a single number is the BETA closure specifically: there neutrality
    determines mu_C from mu_B and mu_S = 0, the Y_C mu_C term cancels against
    the lepton contribution to eps + P, and g = mu_B on both sides. So
    `locate_maxwell` returns `mu_B_lo == mu_B_hi` and `g` equal to both, while
    `locate_maxwell_composition` returns three different numbers. A single
    `mu_B` field would state the beta closure as though it were the
    construction.
    """
    P: float
    g: float
    mu_B_lo: float
    mu_B_hi: float
    n_B_lo: float
    n_B_hi: float
    mu_e_lo: float
    mu_e_hi: float
    branches: tuple

    @property
    def exists(self):
        return np.isfinite(self.g) and self.n_B_hi > self.n_B_lo



def _bisect_where_defined(f, a, fa, b, fb, xtol=1.0e-11, max_steps=60):
    """Bisect f between a and b, tolerating arguments where f does not exist.

    A branch has a domain, and its edge can fall inside a bracket that the
    coarse scan reported as a clean sign change: the two scanned points exist
    and the states between them need not. `brentq` cannot express that -- it
    propagates the RuntimeError and loses an otherwise good bracket.

    So: ordinary bisection, and where the midpoint does not exist, try a few
    offset points before giving up on the bracket. Returning None says the root
    was bracketed but not located, which is what actually happened; the caller
    moves to the next bracket rather than reporting a boundary it did not find.
    """
    for _ in range(max_steps):
        if abs(b - a) <= xtol:
            return 0.5 * (a + b)
        mid = fmid = None
        for fraction in (0.5, 0.4, 0.6, 0.3, 0.7):
            trial = a + fraction * (b - a)
            try:
                fmid, mid = f(trial), trial
            except RuntimeError:
                continue
            break
        if mid is None:
            return None
        if fa * fmid <= 0.0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid
    return 0.5 * (a + b)


def neutral_phase(phase, mu_B, T=0.0, muons=True):
    """The phase at `mu_B`, with mu_C set so it neutralizes its OWN leptons.

    This is what eta = 1 means for one phase: n_C = n_e + n_mu inside it, with
    no charge borrowed from anywhere else. Returns
    `(PhaseThermo, LeptonDomain, mu_C)`, or None where the branch does not
    exist across the scan or the residual neither vanishes nor changes sign
    on it.

    Beta equilibrium fixes mu_e = -mu_C (CLAUDE.md section 2: mu_C + mu_e = 0),
    so the single unknown is mu_C and the single condition is neutrality.

    **NEUTRALITY IS NOT ALWAYS A SIGN CHANGE.** A phase that is neutral BY
    CONSTRUCTION touches zero without crossing it, and no bisection can find a
    root it cannot bracket. Colour-flavour-locked quark matter is the case:
    the locking forces n_u = n_d = n_s, hence n_C = 0 at every mu_C, so the
    residual is -n_e(-mu_C) <= 0 throughout and reaches zero only at mu_C = 0,
    where the phase neutralizes with no electrons at all. So mu_C = 0 is
    scanned too, and a scanned point whose residual is within `NEUTRAL_TOL` is
    taken as the root -- the one closest to mu_C = 0, since a self-neutral
    phase carries no leptons. Every phase that neutralizes transversally is
    located by the sign change below exactly as before: a scanned point that
    close to zero is its root anyway, to a precision the bisection could not
    improve on.
    """
    if T != 0.0:
        raise NotImplementedError(
            f"the eta = 1 construction is written at T = 0; got T = {T} MeV")

    def residual(mu_C):
        return phase(mu_B, mu_C).n_C - charged_leptons(-mu_C, T, muons).n

    lo, hi, count = MU_C_SCAN
    scanned = []
    for mu_C in list(np.linspace(lo, hi, count)) + [0.0]:
        try:
            scanned.append((float(mu_C), residual(mu_C)))
        except RuntimeError:            # this branch does not exist here
            continue

    touching = [c for c, r in scanned if abs(r) <= NEUTRAL_TOL]
    if touching:
        mu_C = min(touching, key=abs)
        return (phase(mu_B, mu_C), charged_leptons(-mu_C, T, muons),
                float(mu_C))

    for (c0, r0), (c1, r1) in zip(scanned, scanned[1:]):
        if r0 * r1 <= 0.0:
            mu_C = _bisect_where_defined(residual, c0, r0, c1, r1)
            if mu_C is None:              # branch stops existing inside here
                continue
            return (phase(mu_B, mu_C), charged_leptons(-mu_C, T, muons),
                    float(mu_C))
    return None


def total_pressure(phase, mu_B, T=0.0, muons=True):
    """(P_matter + P_leptons, n_B, mu_C) for the neutralized phase, or None.

    The leptons belong in the balance because at eta = 1 they live inside the
    structures whose pressures must equal -- `mixed.tex`'s Eq. for mechanical
    equilibrium with eta = 1, where the local lepton pressure carries weight 1.
    Dropping them moves the located mu_B by tens of MeV.
    """
    found = neutral_phase(phase, mu_B, T=T, muons=muons)
    if found is None:
        return None
    th, leptons, mu_C = found
    return th.P + leptons.P, th.n_B, mu_C


def locate_maxwell(phase_lo, phase_hi, mu_B_grid, T=0.0, muons=True,
                   labels=("lo", "hi"), xtol=1.0e-8):
    """Where two branches coexist: the eta = 1 construction.

    `phase_lo` and `phase_hi` are callables (mu_B, mu_C) -> PhaseThermo, the
    low- and high-density branch. Scans `mu_B_grid` for a sign change in
    P_lo - P_hi with both branches neutralized, then brackets it to `xtol`.

    Returns a `MaxwellPoint`; `.exists` is False when the two branches never
    both exist on the grid, or when their pressures do not cross on it. That is
    a physics outcome -- these parameters have no first-order transition in this
    range -- and not a failure, so it is a value rather than an exception.

    The grid is scanned rather than bracketed from its ends for the same reason
    `neutral_phase` scans: neither branch need exist across the whole span, and
    above a transition the low-density branch typically stops existing
    altogether.
    """
    if T != 0.0:
        raise NotImplementedError(
            f"the eta = 1 construction is written at T = 0; got T = {T} MeV")

    def gap(mu_B):
        a = total_pressure(phase_lo, mu_B, T=T, muons=muons)
        b = total_pressure(phase_hi, mu_B, T=T, muons=muons)
        return None if (a is None or b is None) else a[0] - b[0]

    none = no_coexistence(labels)
    scanned = [(float(m), gap(m)) for m in np.asarray(mu_B_grid, dtype=float)]
    scanned = [(m, g) for m, g in scanned if g is not None]
    bracket = next((((m0, g0), (m1, g1))
                    for (m0, g0), (m1, g1) in zip(scanned, scanned[1:])
                    if g0 * g1 <= 0.0), None)
    if bracket is None:
        return none

    (m0, _), (m1, _) = bracket
    mu_B = brentq(lambda m: gap(m), m0, m1, xtol=xtol, rtol=1e-14)
    lo = total_pressure(phase_lo, mu_B, T=T, muons=muons)
    hi = total_pressure(phase_hi, mu_B, T=T, muons=muons)
    if lo is None or hi is None:
        return none
    # g = mu_B here, and not by assertion: on a neutral phase at T = 0 the
    # Euler relation reads eps + P = mu_B n_B + mu_C n_C + sum_l mu_l n_l with
    # mu_C = -mu_e and n_C = n_e + n_mu, so the charge and lepton terms cancel
    # and (eps + P)/n_B is the one potential both phases were evaluated at.
    return MaxwellPoint(P=0.5 * (lo[0] + hi[0]), g=mu_B,
                        mu_B_lo=mu_B, mu_B_hi=mu_B,
                        n_B_lo=lo[1], n_B_hi=hi[1],
                        mu_e_lo=-lo[2], mu_e_hi=-hi[2], branches=tuple(labels))


# --------------------------------------------------------------------------
# The same construction under the COMPOSITION closure
# --------------------------------------------------------------------------
#
# `neutral_phase` above closes a phase by making it neutralize its own
# leptons, which is what a beta-equilibrium branch pair needs. A mixed-phase
# construction consumes the other thing (CLAUDE.md section 3): each pure phase
# at a held, non-leptonic (Y_C, Y_S), electrically charged, with global
# neutrality imposed later. That closure needs its own phase solve and its own
# locator, because the collapse that makes `locate_maxwell` a one-variable
# bisection is a property of beta equilibrium and not of the construction.

#: Starting (mu_C, mu_S) for the composition solve, tried in order until one
#: meets `COMPOSITION_TOL`. mu_C is near zero in symmetric matter and tens of
#: MeV negative in neutron-rich matter; mu_S is positive where strangeness is
#: demanded, since every carrier in this repository has S = +1 (CLAUDE.md
#: section 2) and a positive mu_S is what populates them.
COMPOSITION_SEEDS = ((0.0, 0.0), (-60.0, 100.0), (-150.0, 0.0), (0.0, 200.0))

#: Relative tolerance the held fractions must be met to, as |n_X - Y_X n_B|
#: against n_B. The solve is accepted on this residual rather than on the root
#: finder's own success flag: where the composition is symmetric the residual
#: starts AT round-off, and `hybr` then reports "not making good progress"
#: while sitting on an exact answer. CLAUDE.md section 6 judges convergence on
#: a residual norm, which is the test that reads that situation correctly.
COMPOSITION_TOL = 1.0e-10


def composition_phase(phase, mu_B, Y_C, Y_S, T=0.0):
    """The phase at `mu_B`, with (mu_C, mu_S) solved so it carries (Y_C, Y_S).

    `phase` is a callable (mu_B, mu_C, mu_S) -> PhaseThermo -- one argument
    more than `neutral_phase` takes, because here mu_S is not pinned by an
    equilibrium relation. Returns `(PhaseThermo, mu_C, mu_S)`, or None where
    the branch does not exist at `mu_B` or no seed meets `COMPOSITION_TOL`.

    Y_C and Y_S are the NON-leptonic fractions of CLAUDE.md section 2: the
    phase carries no leptons and is electrically charged, which is what a
    mixed-phase construction consumes for each pure phase before imposing
    global neutrality.

    **At T = 0 a demanded Y_S = 0 leaves mu_S undetermined**, and it is
    dropped from the unknowns rather than carried as a null column. Every
    strangeness carrier in this repository has S = +1 and at T = 0 there are
    no antiparticles, so every term of n_S = sum_i S_i n_i is non-negative and
    n_S = 0 forces each of them to vanish term by term: the row reads 0 = 0
    for every mu_S. That is the same statement, and the same remedy, as
    `eos.enjl.solver.strangeness_row_is_empty` -- and it is a T = 0 statement
    only, since at T > 0 the Fermi tails populate the carriers and their
    antiparticles and the row acquires a gradient.
    """
    if T != 0.0:
        raise NotImplementedError(
            f"the eta = 1 construction is written at T = 0; got T = {T} MeV")

    empty_S = (Y_S == 0.0)

    def residual(x):
        mu_C = float(x[0])
        mu_S = 0.0 if empty_S else float(x[1])
        th = phase(mu_B, mu_C, mu_S)
        rows = [th.n_C - Y_C * th.n_B]
        if not empty_S:
            rows.append(th.n_S - Y_S * th.n_B)
        return rows

    for mu_C0, mu_S0 in COMPOSITION_SEEDS:
        start = [mu_C0] if empty_S else [mu_C0, mu_S0]
        try:
            sol = root(residual, start, method="hybr")
            mu_C = float(sol.x[0])
            mu_S = 0.0 if empty_S else float(sol.x[1])
            th = phase(mu_B, mu_C, mu_S)
        except RuntimeError:          # this branch does not exist here
            continue
        worst = float(np.max(np.abs(np.atleast_1d(sol.fun))))
        if worst <= COMPOSITION_TOL * th.n_B:
            return th, mu_C, mu_S
    return None


def composition_state(phase, mu_B, Y_C, Y_S, T=0.0):
    """(P, g, n_B, mu_C, mu_S) for the phase held at (Y_C, Y_S), or None.

    The composition-closure twin of `total_pressure`, and it reports g as well
    because P alone no longer closes the construction. There are no leptons in
    the balance here, which is the point of the closure: each pure phase is
    charged, and the charge is reconciled between the phases rather than
    inside each of them.

    g is the Gibbs free energy per baryon (eps + P)/n_B. At fixed composition
    the Euler relation makes that g = mu_B + Y_C mu_C + Y_S mu_S, so it is not
    mu_B unless both fraction terms vanish.
    """
    found = composition_phase(phase, mu_B, Y_C, Y_S, T=T)
    if found is None:
        return None
    th, mu_C, mu_S = found
    return th.P, (th.eps + th.P) / th.n_B, th.n_B, mu_C, mu_S


def no_coexistence(labels):
    """The MaxwellPoint that says "no transition here"; `.exists` is False."""
    nan = float("nan")
    return MaxwellPoint(P=nan, g=nan, mu_B_lo=nan, mu_B_hi=nan,
                        n_B_lo=nan, n_B_hi=nan, mu_e_lo=nan, mu_e_hi=nan,
                        branches=tuple(labels))


def locate_maxwell_composition(phase_lo, phase_hi, mu_B_grid, Y_C, Y_S,
                               T=0.0, labels=("lo", "hi"), ftol=1.0e-8):
    """Where two branches held at (Y_C, Y_S) coexist: a 2-D root find.

    `phase_lo` and `phase_hi` are callables (mu_B, mu_C, mu_S) -> PhaseThermo.
    Returns a `MaxwellPoint`, whose `.exists` is False when the two branches
    never both exist on `mu_B_grid` or their (P, g) curves do not cross on it
    -- a physics outcome for these parameters, so a value and not an
    exception (CLAUDE.md section 6).

    **Why this is not `locate_maxwell` with a different phase closure.** That
    one bisects `P_lo(mu_B) - P_hi(mu_B)` in ONE variable, and may, because
    beta equilibrium with neutrality determines mu_C from mu_B and forces
    g = mu_B: the two branches meeting at one mu_B already have equal g, and
    equal P is all that is left to impose. Here each branch solves its own
    (mu_C, mu_S) for the held fractions, g = mu_B + Y_C mu_C + Y_S mu_S
    differs from mu_B by a different amount on each side, and coexistence is
    two conditions --

        P_lo  = P_hi        mechanical equilibrium
        g_lo  = g_hi        chemical equilibrium at fixed composition

    -- in two unknowns, the branches' own mu_B. Nothing in a one-variable
    bisection can express that.

    The two-dimensional solve is SEEDED from the grid rather than started
    anywhere: g is monotone along a branch (Gibbs-Duhem at fixed composition
    and T = 0 gives n_B dg = dP, and n_B > 0), so tabulating (mu_B, g, P) on
    the grid and reading each branch's mu_B and P at a common g turns the
    seed into a one-dimensional sign change in P_lo - P_hi. That interpolated
    crossing is a grid-resolution answer -- the accuracy this module exists to
    avoid -- so it is the starting point and not the result.
    """
    if T != 0.0:
        raise NotImplementedError(
            f"the eta = 1 construction is written at T = 0; got T = {T} MeV")

    none = no_coexistence(labels)
    tables = []
    for phase in (phase_lo, phase_hi):
        rows = []
        for mu_B in np.asarray(mu_B_grid, dtype=float):
            state = composition_state(phase, float(mu_B), Y_C, Y_S, T=T)
            if state is not None:
                rows.append((float(mu_B), state[1], state[0]))   # mu_B, g, P
        if len(rows) < 2:
            return none
        table = np.array(rows)
        tables.append(table[np.argsort(table[:, 1])])            # by g

    g_lo, g_hi = tables[0][:, 1], tables[1][:, 1]
    span_lo = max(g_lo[0], g_hi[0])
    span_hi = min(g_lo[-1], g_hi[-1])
    if not span_hi > span_lo:
        return none
    probe = np.linspace(span_lo, span_hi, 8 * len(mu_B_grid))
    gap = (np.interp(probe, g_lo, tables[0][:, 2])
           - np.interp(probe, g_hi, tables[1][:, 2]))
    crossing = np.flatnonzero(gap[:-1] * gap[1:] <= 0.0)
    if crossing.size == 0:
        return none
    g_seed = float(probe[crossing[0]])
    seed = [float(np.interp(g_seed, g_lo, tables[0][:, 0])),
            float(np.interp(g_seed, g_hi, tables[1][:, 0]))]

    def conditions(x):
        a = composition_state(phase_lo, float(x[0]), Y_C, Y_S, T=T)
        b = composition_state(phase_hi, float(x[1]), Y_C, Y_S, T=T)
        if a is None or b is None:      # walked off a branch's domain
            return [np.nan, np.nan]
        return [a[0] - b[0], a[1] - b[1]]

    sol = root(conditions, seed, method="hybr")
    lo = composition_state(phase_lo, float(sol.x[0]), Y_C, Y_S, T=T)
    hi = composition_state(phase_hi, float(sol.x[1]), Y_C, Y_S, T=T)
    if lo is None or hi is None:
        return none
    if not np.all(np.abs(np.atleast_1d(sol.fun)) <= ftol):
        return none
    return MaxwellPoint(P=0.5 * (lo[0] + hi[0]), g=0.5 * (lo[1] + hi[1]),
                        mu_B_lo=float(sol.x[0]), mu_B_hi=float(sol.x[1]),
                        n_B_lo=lo[2], n_B_hi=hi[2],
                        mu_e_lo=float("nan"), mu_e_hi=float("nan"),
                        branches=tuple(labels))


def locate_windows(phases, n_B_grid, eta, spec, T=0.0,
                   max_windows=4, species=None, **kw):
    """Every coexistence window on the grid, in density order.

    A model with several branches (ENJL) can put MORE than one first-order
    transition on one density axis. This is the thin loop
    docs/enjl/PHASE_TRANSITION_DESIGN.md section 4 asks for: `locate_window`
    repeatedly, each search starting just above the previous offset, stopping
    at the first search that finds nothing or after `max_windows`. Which
    branch PAIRS to sweep is the caller's business — the engine stays a
    two-phase engine, and each call of this function is one pair.

    `**kw` passes `locate_window`'s tuning through (n_probe, tol, refine, ...).
    Returns a list of `Window`, possibly empty.
    """
    grid = np.asarray(n_B_grid, dtype=float)
    if grid.size < 2:
        return []
    step = float(np.min(np.diff(np.sort(grid))))
    windows = []
    lo = float(grid[0])
    for _ in range(max_windows):
        sub = grid[grid >= lo]
        if sub.size < 2:
            break
        w = locate_window(phases, sub, eta, spec, T=T,
                          species=species, **kw)
        if not w.exists:
            break
        windows.append(w)
        lo = w.n_offset + step
    return windows

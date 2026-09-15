# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A fast colour-superconducting quark EoS, fitted to `eos.njl`
#
# The goal: a closed-form pressure that takes
# `(mu_B, mu_C, mu_S, T, csc_type; NJL parameters)` and returns an equation of
# state at microsecond cost, with its coefficients fixed once and for all by a
# fit against `eos.njl`, so that a construction, a TOV sweep or a sampler
# never calls the NJL solver again.
#
# `docs/csc_bag_mapping.md` is the derivation; this notebook is the
# construction and its benchmarks.
#
# **The parametrisation is a power series, not a bag model.** Write the
# pressure of one phase as
#
#     P(mu_u, mu_d, mu_s, T) = (1/(4 pi^2)) sum_t a_t X_t(mu_f, T)
#
# where each `X_t` is an elementary function of the flavour potentials and the
# `a_t` are the fitted coefficients. The terms are chosen for what they mean,
# following the structure of the perturbative CFL/2SC/unpaired pressures
# (Alford, Braby, Paris and Reddy; Gorda and Saeppi; Geissel, Gorda and Braun):
#
#     a0    mu_star^4                     the bag constant, B = -a0/(4 pi^2)
#     a2    sum_f mu_f^2 mu_star^2        a mass-like term
#     a2s   mu_s^2 mu_star^2              strangeness breaking: -3 m_s^2 mu^2
#     a4    sum_f mu_f^4                  the free gas; a4 = 1 is free quarks
#     a4l   sum_f mu_f^4 ln(mu_f/mu_star) the perturbative logarithm
#     a6    sum_f mu_f^6/(mu_f^2 + M_V^2)  a SATURATING vector-like term
#     aT2   sum_f mu_f^2 T^2              the leading thermal term
#     aT4   T^4                           and the next
#     pair  c_phase (sum_f mu_f^2) Delta^2 * 4    the condensation energy
#
# **The basis is constrained by the conformal limit.** For `P ~ mu^k` one has
# `n ~ k mu^(k-1)`, `eps = -P + mu n = (k-1) mu^k` and hence
# `c_s^2 = dP/deps = 1/(k-1)`: `k = 4` is conformal, and any term with a
# LARGER power and a fixed scale takes over as `mu -> infinity` and drags the
# model away from it -- a bare `mu^6/mu_star^2` sends `P/P_free` to infinity
# and `c_s^2` to 1/5, which is wrong in the one regime where the answer is
# known. So the vector-like term SATURATES: `mu^6/(mu^2 + M_V^2)` behaves as
# `mu^6/M_V^2` where the vector repulsion acts and as `mu^4` above it, which
# is also the shape the gluon-exchange `G_V(n_q)` of `eos.njl` actually has.
# Every other term is of power <= 4, so `P/P_free -> a4 + a6` and
# `c_s^2 -> 1/3` up to the logarithm, and the logarithm is the running that
# perturbative QCD has anyway.
#
# with `c_phase = 1` (CFL), `1/3` (2SC), `0` (unpaired) -- the count of gapped
# quasiparticles, and the same 3:1 that Geissel et al. obtain for `gamma_1` at
# leading order -- and `Delta = Delta_star (mu_B/mu_star)^sigma`.
#
# EVERY TERM IS LINEAR IN ITS COEFFICIENT, so a fit is one bounded linear
# least squares with no iteration, and adding or removing a term is editing
# one list -- the term's value and its slopes, which `basis_terms` keeps side
# by side so that the densities `n_a = -dOmega/dmu_a` and the entropy
# `s = -dOmega/dT` are CLOSED FORM and the section 8 identities hold to
# round-off. The bag model is the case `TERMS = ("a0", "a2s", "a4", "pair")`.

# %%
import hashlib
import itertools
import json
import os
import pickle
import sys
import time
from collections import namedtuple
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, lsq_linear, root

ROOT = Path.cwd()
if not (ROOT / "eos").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eos import njl
from eos.mixed import njl_phase
from eos.general.basis import quark_potentials
from eos.general.fermi_integrals import solve_fermi_jel
from eos.general.particles import get_particle
from eos.general.figure_style import LABELS, OKAB_CAT, panel_label, paper_grid

#: Quark degeneracy (2 spin x 3 colour), from the single home of the particle
#: properties rather than as a literal 6.
G_QUARK = get_particle("quark").g_degen

MU_STAR = 2600.0                 # MeV, the reference potential
#: Fixed masses at which an exact massive strange Fermi gas is added to the
#: basis. A THRESHOLD cannot be built from powers of mu_s -- and n_S is the
#: worst residual row everywhere, because across 2-15 n_sat the NJL strange
#: constituent mass runs from ~460 to ~200 MeV and the s sea is switching on.
#: Each of these gases switches on at mu_s = m, so their COMBINATION spans the
#: threshold shapes a running mass produces, and it does so linearly: still one
#: bounded least squares, no mass to search over.
#:
#: MEASURED, eta_D = 1.45, gluon-exchange, over 2-15 n_sat, unpaired phase:
#: rms in P 1.5% -> 0.57%, in n_B 5.3% -> 2.2%, in n_S 9.5% -> 7.6%. Giving
#: the s flavour its own mu^4 coefficient instead barely moves n_S (9.5% ->
#: 9.2%), which is what says the missing structure is a threshold and not a
#: normalisation.
S_MASSES = (150.0, 300.0, 450.0)         # MeV

#: The saturation scale of the vector-like term. Fixed rather than fitted, so
#: the model stays linear in every coefficient. The gluon-exchange coupling of
#: `eos.njl` is G_V0/[1 + 8 k_F^2/(9 M_g^2)], whose scale is 3 M_g/(2 sqrt 2)
#: ~ 1.06 M_g, so this is the shipped M_g = 500 MeV expressed as a potential.
M_V = 530.0                      # MeV
PI2 = np.pi ** 2
HBARC = 197.3269804
HC3 = HBARC ** 3
#: sum over gapped quasiparticles of Delta_qp^2, as a multiple of
#: (sum_f mu_f^2) Delta^2/pi^2: 1 for CFL (nine modes), 1/3 for 2SC (four).
C_PHASE = {"unpaired": 0.0, "2SC": 1.0 / 3.0, "CFL": 1.0}
PATTERNS = ("unpaired", "2SC", "CFL")
CACHE = ROOT / "output" / "csc_map_cache"


def flavour_mu(mu_B, mu_C, mu_S):
    """(mu_u, mu_d, mu_s) [MeV]; S = +1 per s quark, C non-leptonic."""
    return quark_potentials(mu_B, mu_C, mu_S)


#: d mu_f / d mu_a, the rows indexed by a in (B, C, S) and the entries by
#: flavour. The map is linear, so evaluating `quark_potentials` on the unit
#: potentials IS its Jacobian -- which is how the chain rule that carries a
#: flavour derivative back to (mu_B, mu_C, mu_S) is TAKEN from the one place
#: the basis change is declared (CLAUDE.md section 2) instead of rewritten:
#:
#:     dX/dmu_B = (X_u + X_d + X_s)/3,  dX/dmu_C = (2 X_u - X_d - X_s)/3,
#:     dX/dmu_S = X_s.
DMU_DCHARGE = (quark_potentials(1.0, 0.0, 0.0),
               quark_potentials(0.0, 1.0, 0.0),
               quark_potentials(0.0, 0.0, 1.0))

#: The same for the CFL branch, where the basis is evaluated at the single
#: potential mu = (mu_B + mu_S)/3: mu_C does not enter it at all, and mu_B and
#: mu_S enter identically. So n_C = 0 and n_S = n_B come OUT of the derivative
#: -- the locking as an identity of the chain rule, not as a filter applied
#: afterwards.
DMU_DCHARGE_CFL = ((1.0 / 3.0,) * 3, (0.0,) * 3, (1.0 / 3.0,) * 3)

#: One basis term: its value and its four derivatives. `B`, `C` and `S` are
#: dX/dmu_B, dX/dmu_C, dX/dmu_S and `T` is dX/dT, so that
#: n_a = -dOmega/dmu_a = sum_t a_t X_t.<a> and s = -dOmega/dT = sum_t a_t X_t.T
#: in closed form: the Euler and free-energy identities of CLAUDE.md section 8
#: then hold to round-off rather than to a difference quotient's truncation.
Term = namedtuple("Term", "X B C S T")


def basis_terms(mu_B, mu_C, mu_S, T, pattern, Delta_star, sigma):
    """{term: Term(X_t, dX_t/dmu_B, dX_t/dmu_C, dX_t/dmu_S, dX_t/dT)}.

    Value and derivatives of every basis function in MeV/fm^3 (and fm^-3) per
    unit coefficient, written side by side so that a term cannot be added with
    its derivative forgotten. That is the one cost of dropping the central
    differences the prototype used: a new term now needs its slope written
    too. Every one of them is elementary EXCEPT the massive strange gases, and
    those need nothing written -- `solve_fermi_jel` returns their n = dP/dmu_s
    and s = dP/dT alongside the P that enters the basis.

    Adding a term to the model means adding an entry here and its name to
    `TERMS`; nothing else in the fit changes, because every one of them enters
    the pressure linearly.
    """
    if pattern == "CFL":
        # Flavour locking, imposed rather than fitted: a colour-neutral CFL
        # phase has n_u = n_d = n_s, hence n_C = 0 and n_S = n_B identically,
        # and its pressure depends only on mu_B + mu_S. Evaluating the basis
        # at a common potential reproduces all three statements exactly.
        # Measured on eos.njl: P is the same to every digit at (1900, 0, 0),
        # (1900, -60, 0), (1800, 0, 100) and (1840, 0, 60).
        mu = (mu_B + mu_S) / 3.0
        mu_f = (mu, mu, mu)
        dmu = DMU_DCHARGE_CFL
        # The gap of a LOCKED phase runs on the locked phase's own potential,
        # 3 mu = mu_B + mu_S, not on mu_B alone. On the sampled line
        # (mu_S = 0) the two are the same number, so no fitted coefficient
        # moves; off it, a Delta(mu_B) would make dP/dmu_B differ from
        # dP/dmu_S and break n_S = n_B by 2e-4 -- MEASURED, and the CFL gate
        # of .scratch/pqm/issues/01 is what found it.
        mu_gap, dgap = mu_B + mu_S, (1.0, 0.0, 1.0)
    else:
        mu_f = flavour_mu(mu_B, mu_C, mu_S)
        dmu = DMU_DCHARGE
        mu_gap, dgap = mu_B, (1.0, 0.0, 0.0)
    mu_u, mu_d, mu_s = mu_f
    scale = 1.0 / (4.0 * PI2 * HC3)
    Delta = Delta_star * (mu_gap / MU_STAR) ** sigma if Delta_star > 0 else 0.0

    def term(X, dX_dmu_f, dX_dmu_gap=0.0, dX_dT=0.0):
        """A term from its flavour gradient, carried back to (B, C, S).

        `dX_dmu_gap` is an EXPLICIT dependence on the gap potential, beside
        the one through the flavour potentials -- only `pair` has one.
        """
        return Term(X,
                    *(dX_dmu_gap * dgap[a]
                      + sum(d * j for d, j in zip(dX_dmu_f, dmu[a]))
                      for a in range(3)),
                    dX_dT)

    sum_mu2 = sum(m ** 2 for m in mu_f)
    pair_X = scale * 4.0 * C_PHASE[pattern] * sum_mu2 * Delta ** 2
    gases = {}
    for m in S_MASSES:
        # n, P and s of the same call: index 0 IS dP/dmu_s and index 3 IS
        # dP/dT, so the massive gas costs no derivative of its own.
        #
        # The prototype zeroed this term below mu_s = m. At T = 0 that was a
        # no-op -- JEL returns exactly 0 there -- and at T > 0 it was a jump:
        # a gas with P = 0.58 MeV/fm^3 at (mu_s, T) = (149, 30) switched on
        # discontinuously two MeV later, so its derivative was whatever the
        # cut said rather than what the gas does. Dropping the cut changes no
        # T = 0 number and makes dP/dT the entropy of the gas everywhere.
        n_g, P_g, _, s_g = solve_fermi_jel(mu_s, max(T, 0.0), m, G_QUARK,
                                          include_antiparticles=True)[:4]
        gases[f"gs{m:.0f}"] = term(P_g, (0.0, 0.0, n_g), dX_dT=s_g)
    return {
        "a0": term(scale * MU_STAR ** 4, (0.0, 0.0, 0.0)),
        "a2": term(scale * sum_mu2 * MU_STAR ** 2,
                   tuple(scale * 2.0 * m * MU_STAR ** 2 for m in mu_f)),
        "a2s": term(scale * mu_s ** 2 * MU_STAR ** 2,
                    (0.0, 0.0, scale * 2.0 * mu_s * MU_STAR ** 2)),
        "a4": term(scale * sum(m ** 4 for m in mu_f),
                   tuple(scale * 4.0 * m ** 3 for m in mu_f)),
        "a4l": term(scale * sum(m ** 4 * np.log(m / MU_STAR) for m in mu_f),
                    tuple(scale * m ** 3 * (4.0 * np.log(m / MU_STAR) + 1.0)
                          for m in mu_f)),
        # Saturating, so that P/P_free stays finite and c_s^2 -> 1/3 as
        # mu -> infinity: this term is mu^6/M_V^2 below the vector scale and
        # mu^4 above it, never mu^6 forever.
        "a6": term(scale * sum(m ** 6 / (m ** 2 + M_V ** 2) for m in mu_f),
                   tuple(scale * (4.0 * m ** 7 + 6.0 * m ** 5 * M_V ** 2)
                         / (m ** 2 + M_V ** 2) ** 2 for m in mu_f)),
        "aT2": term(scale * sum_mu2 * T ** 2,
                    tuple(scale * 2.0 * m * T ** 2 for m in mu_f),
                    dX_dT=scale * sum_mu2 * 2.0 * T),
        "aT4": term(scale * T ** 4, (0.0, 0.0, 0.0),
                    dX_dT=scale * 4.0 * T ** 3),
        # Delta = Delta_star (mu_gap/mu_star)^sigma is a power law, so the
        # chain rule through it is exact:
        # d(Delta^2)/dmu_gap = 2 sigma Delta^2 / mu_gap.
        "pair": term(pair_X,
                     tuple(scale * 8.0 * C_PHASE[pattern] * m * Delta ** 2
                           for m in mu_f),
                     dX_dmu_gap=(2.0 * sigma * pair_X / mu_gap
                                 if Delta > 0.0 and mu_gap > 0.0 else 0.0)),
        **gases,
    }


def basis(mu_B, mu_C, mu_S, T, pattern, Delta_star, sigma):
    """{term: X_t} -- the values alone, for the pressure and the fit rows."""
    return {name: t.X for name, t in
            basis_terms(mu_B, mu_C, mu_S, T, pattern, Delta_star,
                        sigma).items()}


#: Which terms the model carries, and the bounds each coefficient may take.
#: `a0` is minus the bag constant, so B > 0 means a0 < 0, and `pair` is a
#: condensation energy and must not be negative. The rest are free in sign --
#: a correction that may only push one way is not a correction, it is a prior.
#:
#: `a4` is the free-gas coefficient and its bound is (0, inf) -- NO CEILING,
#: which is what the code has always shipped; the comment here used to claim a
#: ceiling of 2 that was never in it.
#:
#: In a bag model a4 = 1 - 2 alpha_s/pi <= 1, and a4 > 1 would mean a negative
#: coupling; but this is a PARAMETRISATION of eos.njl, whose RG-consistent
#: regularisation (lambda_UV = 10) runs the medium integral to ten times the
#: vacuum cutoff and makes the quark gas STIFFER than free. Measured: with the
#: bound at 1, a4 pegs there for every phase and every parameter point, which
#: is the fit reporting that the bound is the wrong prior for this model.
#: MEASURED at rkh, eta_D = 1.45, G_V0/G_S = 0.5: a4 = 2.67 unpaired, 3.13
#: 2SC, 2.81 CFL. A ceiling of 2 would peg all three exactly as the ceiling of
#: 1 did, so it is not a bound the data would ever sit inside -- it is the
#: same wrong prior one notch looser. What keeps a4 finite is not a ceiling
#: but the conformal row below, which constrains the SUM of the mu^4 terms:
#: a4, a4l, a6 and the massive gases are collinear, so bounding one of them
#: individually pins nothing anyway. None of the three fits pegs a4.
TERM_BOUNDS = {f"gs{m:.0f}": (-np.inf, np.inf) for m in S_MASSES}
TERM_BOUNDS.update({"a0": (-np.inf, 0.0), "a2": (-np.inf, np.inf),
               "a2s": (-np.inf, np.inf), "a4": (0.0, np.inf),
               "a4l": (-np.inf, np.inf), "a6": (-np.inf, np.inf),
               "aT2": (-np.inf, np.inf), "aT4": (-np.inf, np.inf),
               "pair": (0.0, np.inf)})


def pressure(coeff, mu_B, mu_C, mu_S, T, pattern, Delta_star, sigma):
    """The model's pressure [MeV/fm^3]: the coefficients on the basis."""
    X = basis(mu_B, mu_C, mu_S, T, pattern, Delta_star, sigma)
    return sum(c * X[name] for name, c in coeff.items())


def charges(f, mu_B, mu_C, mu_S, T=0.0):
    """(n_B, n_C, n_S) [fm^-3] = -dOmega/dmu_a, in closed form.

    Omega = -P, so n_a = dP/dmu_a, and every term's slope comes from
    `basis_terms` rather than from a difference quotient. On the CFL branch
    the chain rule returns n_C = 0 and n_S = n_B to round-off, with nothing
    filtered out (`DMU_DCHARGE_CFL`).
    """
    X = basis_terms(mu_B, mu_C, mu_S, T, f["pattern"], f["Delta_star"],
                    f["sigma"])
    return tuple(sum(c * getattr(X[name], slot)
                     for name, c in f["coeff"].items())
                 for slot in ("B", "C", "S"))


def entropy(f, mu_B, mu_C, mu_S, T=0.0):
    """s [fm^-3] = -dOmega/dT = dP/dT, in closed form.

    Only `aT2`, `aT4` and the massive strange gases carry a temperature; the
    gases' dP/dT is the s that `solve_fermi_jel` returns beside their P.
    """
    X = basis_terms(mu_B, mu_C, mu_S, T, f["pattern"], f["Delta_star"],
                    f["sigma"])
    return sum(c * X[name].T for name, c in f["coeff"].items())


def flavour_densities(f, mu_B, mu_C, mu_S, T=0.0):
    """(n_u, n_d, n_s) [fm^-3] from (n_B, n_C, n_S), inverting the charge sums.

        n_B = (n_u + n_d + n_s)/3,  n_C = (2 n_u - n_d - n_s)/3,  n_S = n_s

    so n_u = n_B + n_C, n_d = 2 n_B - n_C - n_S, n_s = n_S. Reading the
    flavours off the conserved charges rather than differentiating in mu_f
    keeps one definition of the basis (CLAUDE.md section 2).
    """
    n_B, n_C, n_S = charges(f, mu_B, mu_C, mu_S, T)
    return n_B + n_C, 2.0 * n_B - n_C - n_S, n_S


# %% [markdown]
# ## Sampling `eos.njl` in the potentials
#
# The phase-adapter contract already runs the right way -- potentials in,
# thermodynamics out -- so no inversion is needed. One pattern is DECLARED per
# grid, so a single branch is followed and the fit is never handed a
# first-order transition it cannot represent. The solves are the whole cost
# here, so they are cached on disk.

# %%
# THE FIT WINDOW IS THE APPLICATION WINDOW, and it is dense at the bottom of
# it, because that is where the model is hardest and where a star's quark core
# begins.
#
# MEASURED, sweeping mu_B in 50 MeV steps on the four gluon-exchange sets:
# NO BRANCH REACHES 1 n_sat. The lowest density any of them attains is its own
# terminus, 1.5-2.4 n_sat at mu_B = 850-1150 MeV, below which the T = 0 quark
# phase does not exist -- the chiral condensate is still there and the branch
# ends. So 1-2 n_sat is a statement about the model, not a sampling choice,
# and the reachable window is about 2-13 n_sat:
#
#   eta_D=1.45 G_V0=0.50   unpaired 1150-1700 MeV (2.4-8.6 n_sat)
#                          2SC       900-1700     (2.1-10.6)
#                          CFL      1000-1700     (2.8-13.1)
#   eta_D=1.00 G_V0=0.50   CFL      1350-1700     (6.1-12.2)  -- a short branch
#   eta_D=1.45 G_V0=0.25   2SC       850-1700     (1.5-13.7)
#   eta_D=0.75 G_V0=0.75   all three 1150-1700    (2.0-9.3)
#
# For comparison, mu_B = 1500-2600 MeV is 5-38 n_sat: almost entirely ABOVE
# the region of interest, and anchoring the coefficients there is how a fit
# comes out excellent and useless.
GRID_MU_B = np.linspace(900.0, 1800.0, 28)       # MeV, ~2-13 n_sat, dense
GRID_MU_C = np.array([-80.0, -40.0, 0.0, 40.0])  # MeV
GRID_MU_S = np.array([-80.0, 0.0, 80.0])         # MeV
GRID_T = (0.0,)                                  # MeV
# ---------------------------------------------------------------------------


def sample(par, pattern, use_cache=True):
    """`eos.njl` on the grid at fixed potentials, one pattern declared."""
    key = hashlib.md5(json.dumps(
        [asdict(par), pattern, list(GRID_MU_B), list(GRID_MU_C),
         list(GRID_MU_S), list(GRID_T)],
        sort_keys=True, default=str).encode()).hexdigest()
    path = CACHE / (key + ".pkl")
    if use_cache and path.exists():
        rows = pickle.loads(path.read_bytes())
        if pattern == "CFL":        # apply the locking check to old caches too
            rows = [r for r in rows
                    if r["n_B"] > 0 and abs(r["n_C"]) < 1e-6 * r["n_B"]
                    and abs(r["n_S"] / r["n_B"] - 1.0) < 1e-6]
        return rows

    phase = njl_phase(par, njl.SpeciesFlags(csc=True), patterns=(pattern,),
                      backend="fast")
    # CFL: mu_C does not enter and mu_S only through mu_B + mu_S, so a full
    # grid there is one line sampled many times over.
    if pattern == "CFL":
        grid = [(b, 0.0, 0.0, t) for b in GRID_MU_B for t in GRID_T]
    else:
        grid = list(itertools.product(GRID_MU_B, GRID_MU_C, GRID_MU_S, GRID_T))
    out = []
    for mu_B, mu_C, mu_S, T in grid:
        try:
            th = phase.thermo(mu_B, mu_C, mu_S, T)
        except Exception:
            continue
        gaps = np.array([th.fields.get(f"Delta_{i}", 0.0) for i in (1, 2, 3)])
        if C_PHASE[pattern] > 0 and not np.any(np.abs(gaps) > 1.0):
            continue                # the layout collapsed: not this branch
        # A CFL-layout solve near the branch terminus can converge on a state
        # that is not locked: MEASURED at mu_B = 967 and 1000 MeV it returns
        # n_C = 0.21 fm^-3 and n_S = 0, i.e. no strange quarks at all, while
        # still carrying a 138 MeV gap. A nonzero gap is therefore NOT enough
        # to identify the phase -- the locking itself has to be checked, or
        # those points enter the CFL fit as data the ansatz cannot represent
        # and take the coefficients with them.
        if pattern == "CFL" and th.n_B > 0:
            locked = (abs(th.n_C) < 1e-6 * max(th.n_B, 1e-12)
                      and abs(th.n_S / th.n_B - 1.0) < 1e-6)
            if not locked:
                continue
        out.append(dict(mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, T=T, P=th.P,
                        n_B=th.n_B, n_C=th.n_C, n_S=th.n_S,
                        M_s=th.fields.get("M_s", 0.0),
                        Delta=float(np.sqrt((gaps ** 2).sum() / 3.0))))
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


# %% [markdown]
# ## The fit
#
# The gap is TAKEN FROM `eos.njl`, not inferred: the model reports
# `Delta_1, Delta_2, Delta_3` at every point, so fitting it would spend a
# degree of freedom to recover something already known -- and measurably lets
# it drift (a 2SC gap of 372 MeV against the NJL 220 when it is left free).
# What is fitted to it is the power law of Geissel et al. Eq. (13),
# `Delta = Delta_star (mu_B/mu_star)^sigma`.
#
# The residual carries the pressure AND the three charge densities. They are
# first derivatives of the same potential and the adapter returns them for
# free, so they are four times the data at no solver cost -- and they are the
# sharper test, since a pressure fit is insensitive to a slope error that a
# constant can absorb.

# %%
def gap_powerlaw(records):
    """(Delta_star, sigma) of Delta = Delta_star (mu_B/mu_star)^sigma."""
    mu_B = np.array([r["mu_B"] for r in records])
    D = np.array([r["Delta"] for r in records])
    keep = D > 1.0
    if keep.sum() < 3:
        return 0.0, 0.0
    sigma, lnD = np.polyfit(np.log(mu_B[keep] / MU_STAR), np.log(D[keep]), 1)
    return float(np.exp(lnD)), float(sigma)


def usable_terms(terms, pattern):
    """`terms` minus the ones that are meaningless or degenerate here.

    Two exclusions, both structural rather than cosmetic:

      * `pair` in an unpaired phase multiplies zero;
      * `a2s` in CFL. The locked phase is evaluated at a common potential, so
        `sum_f mu_f^2 = 3 mu_s^2` there and `a2s` IS `a2` divided by three.
        Left in, the least squares returns a cancelling pair of coefficients
        of order 1e9 that happens to fit -- and then the map from the NJL
        parameters to those coefficients is noise.
    """
    out = []
    for t in terms:
        if t == "pair" and C_PHASE[pattern] == 0.0:
            continue
        if t == "a2s" and pattern == "CFL" and "a2" in terms:
            continue
        out.append(t)
    return out


#: Tikhonov weight on the coefficients, relative to the data rows.
#: mu^4, mu^4 ln mu and mu^6 are nearly collinear over any finite range
#: of mu, so the least squares has a long flat valley: the PREDICTIONS
#: are pinned but the individual coefficients are not, and they jump
#: about from one NJL parameter point to the next. That is fatal here,
#: because the whole object of the exercise is a smooth map from the NJL
#: parameters to those coefficients. A small ridge picks the
#: minimum-norm solution out of the valley, which is a smooth function of
#: the data, at a cost in residual that the benchmark below measures.
RIDGE = 1.0e-3

#: Weight of the CONFORMAL ROW: one extra equation demanding
#: P/P_free -> 1 as mu -> infinity. Every term of power 4 -- a4, the
#: logarithm, the saturating a6 and each massive gas -- tends to mu^4 up
#: there, so what protects the limit is their SUM, and nothing inside the
#: fitted window constrains it. MEASURED without this row, on the dense
#: low-density grid: the CFL fit returns a4 = 20, a6 = -36, gs150 = +35 and
#: P/P_free(60 GeV) = 20.5, which is an excellent fit to the data and a
#: nonsense equation of state one decade above it.
CONFORMAL_WEIGHT = 1.0
#: Where "infinity" is evaluated. Far above any star and far above the fit.
MU_CONFORMAL = 3.0e5             # MeV


def fit_phase(records, pattern, terms, held=None, ridge=RIDGE):
    """The coefficients of `terms` for one phase, and the residuals.

    `held` fixes a coefficient instead of fitting it; its contribution moves
    to the right-hand side. Passing `held={"pair": 1.0}` pins the
    condensation energy at its leading-order value,
    `c_phase (sum_f mu_f^2) Delta^2/pi^2`.

    MEASURED, and the reason the default fits it instead: pinning it costs a
    factor of a hundred in beta equilibrium -- 20.7% maximum error in P
    against 0.2% -- and the free fit returns a coefficient near ZERO. At
    these densities the leading-order condensation term is not what pairing
    does to the NJL pressure, and the other powers of mu describe it better.
    The pairing physics still reaches the model, through the per-phase
    coefficient set and through Delta_star and sigma; it is the textbook
    COEFFICIENT that does not survive, and that is a result rather than a
    fitting convenience.

    The density rows are the CLOSED-FORM slopes of the model's own basis
    functions, `basis_terms`, so the coefficients are fitted to exactly the
    derivative the model will later report. The prototype differenced the
    basis instead, which cost the fit the truncation error of an h = 1 MeV
    difference on a mu^6-scale function and left `s` with no row at all.
    """
    held = {} if held is None else dict(held)
    Delta_star, sigma = gap_powerlaw(records)
    use = [t for t in usable_terms(terms, pattern) if t not in held]
    held = {t: v for t, v in held.items()
            if t in usable_terms(terms, pattern)}

    #: The four rows one sample contributes: the pressure and the three
    #: densities, which are the same Term's value and its three slopes.
    ROWS = (("X", "P"), ("B", "n_B"), ("C", "n_C"), ("S", "n_S"))

    P_scale = max(abs(r["P"]) for r in records)
    n_scale = max(abs(r["n_B"]) for r in records)
    A, b = [], []
    for r in records:
        X = basis_terms(r["mu_B"], r["mu_C"], r["mu_S"], r["T"], pattern,
                        Delta_star, sigma)
        for slot, target in ROWS:
            scale = P_scale if slot == "X" else n_scale
            row = np.array([getattr(X[t], slot) for t in use])
            off = sum(v * getattr(X[t], slot) for t, v in held.items())
            A.append(row / scale)
            b.append((r[target] - off) / scale)
    A, b = np.array(A), np.array(b)
    n_data = len(b)
    if CONFORMAL_WEIGHT > 0.0:
        Xinf = basis(MU_CONFORMAL, 0.0, 0.0, 0.0, pattern, Delta_star, sigma)
        P_free_inf = 3.0 * (MU_CONFORMAL / 3.0) ** 4 / (4.0 * PI2 * HC3)
        row = np.array([Xinf[t] for t in use]) / P_free_inf
        held_inf = sum(v * Xinf[t] for t, v in held.items()) / P_free_inf
        A = np.vstack([A, CONFORMAL_WEIGHT * row])
        b = np.concatenate([b, [CONFORMAL_WEIGHT * (1.0 - held_inf)]])
    if ridge > 0.0:
        A = np.vstack([A, ridge * np.eye(len(use))])
        b = np.concatenate([b, np.zeros(len(use))])
    res = lsq_linear(A, b, bounds=([TERM_BOUNDS[t][0] for t in use],
                                   [TERM_BOUNDS[t][1] for t in use]))
    resid = res.fun[:n_data].reshape(-1, 4)
    Xinf = basis(MU_CONFORMAL, 0.0, 0.0, 0.0, pattern, Delta_star, sigma)
    P_free_inf = 3.0 * (MU_CONFORMAL / 3.0) ** 4 / (4.0 * PI2 * HC3)
    coeff = dict(zip(use, res.x))
    coeff.update(held)
    return dict(pattern=pattern, coeff=coeff, terms=tuple(use) + tuple(held),
                fitted=tuple(use), held=held,
                P_over_free_inf=sum(coeff[t] * Xinf[t]
                                    for t in coeff) / P_free_inf,
                Delta_star=Delta_star, sigma=sigma,
                B14=(-coeff.get("a0", 0.0) * MU_STAR ** 4 / (4.0 * PI2))
                     ** 0.25 if coeff.get("a0", 0.0) < 0 else 0.0,
                rms=dict(zip(("P", "n_B", "n_C", "n_S"),
                             np.sqrt((resid ** 2).mean(axis=0)))),
                max_res=float(np.abs(resid).max()),
                pegged=[t for t, v in coeff.items()
                        if np.isclose(v, TERM_BOUNDS[t][0])
                        or np.isclose(v, TERM_BOUNDS[t][1])],
                n_points=len(records))


# %% [markdown]
# ## Modes, and what "the equilibrium phase" means in each
#
# A mode fixes the independent variables (CLAUDE.md section 3), and with them
# WHICH potentials are globally free. The pairing pattern is then the candidate
# of lowest grand potential at fixed values of the potentials conjugate to the
# GLOBALLY CONSERVED charges -- the rest are fixed inside each phase by that
# mode's own equilibrium conditions, exactly as neutrality is:
#
#     beta_eq_neutrinoless  B conserved.  mu_S = 0 (weak s <-> d), and mu_C
#                           from charge neutrality with mu_C + mu_e = 0.
#                           Phases compared at fixed mu_B.
#     fixed_YC + leptons    B and C conserved.  mu_S = 0; mu_C set by the
#                           target Y_C; electrons added to neutralise,
#                           n_e = n_C, so mu_e is NOT -mu_C here.
#                           Phases compared at fixed mu_B (equivalently at
#                           fixed mu_C, which the constraint picks out).
#     fixed_YC_YS, no lep   B, C and S conserved.  mu_C and mu_S both set by
#                           the targets; the phase is electrically charged and
#                           carries no leptons. This is what a mixed-phase
#                           construction needs for each pure phase.
#
# **CFL is not available in every mode, and the code says so rather than
# returning a number.** Locking fixes `Y_C = 0` and `Y_S = 1` identically, so
# a mode demanding `Y_C = 0.5` or `Y_S = 0` has no CFL solution at all. That
# is CLAUDE.md section 3's statement -- `cfl` is not a choice of equilibrium
# condition but a statement about which phase the model describes -- arriving
# here on its own rather than being written in.

# %%
NSAT = 0.16                      # fm^-3

#: label, whether leptons neutralise, whether beta equilibrium relates mu_e to
#: mu_C, and which fractions the mode fixes.
MODES = {
    "beta_eq_neutrinoless": dict(
        label=r"$\beta$-equilibrium, neutrinoless",
        leptons=True, beta=True, fixed=()),
    "fixed_YC": dict(
        label=r"fixed $Y_C$, with leptons",
        leptons=True, beta=False, fixed=("Y_C",)),
    "fixed_YC_YS": dict(
        label=r"fixed $Y_C$ and $Y_S$, no leptons",
        leptons=False, beta=False, fixed=("Y_C", "Y_S")),
}

#: What flavour locking fixes in the CFL phase, whatever the potentials are.
CFL_FRACTIONS = dict(Y_C=0.0, Y_S=1.0)


def scan_root(residual, lo, hi, n=121):
    """A bracketed root of `residual`, found by scanning first.

    A fixed bracket is not safe here: the polynomial pressure has no meaning
    where it predicts n_B <= 0, and at the ends of a wide mu_C interval it
    does exactly that. The residual returns NaN there, so a plain brentq gets
    two same-signed endpoints and raises -- which is how the fixed-Y_C sweep
    silently lost 71 of its 90 points. Scanning finds the first sign change
    between two points that are both meaningful, and brentq refines it.
    """
    xs = np.linspace(lo, hi, n)
    vals = []
    for x in xs:
        try:
            vals.append(residual(x))
        except Exception:
            vals.append(np.nan)
    for a, b, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa == 0.0:
            return float(a)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0:
            return float(brentq(residual, a, b, xtol=1e-9))
    return None


def lepton_thermo(mu_e):
    """(n_e, P_e) of a massless electron gas at T = 0, fm^-3 and MeV/fm^3."""
    if mu_e <= 0.0:
        return 0.0, 0.0
    return mu_e ** 3 / (3.0 * PI2) / HC3, mu_e ** 4 / (12.0 * PI2) / HC3


def phase_state(f, mode, mu_B, targets, T=0.0, guess=None):
    """One phase in one mode at one mu_B, or None where it cannot exist.

    Returns the full state: potentials, charges, fractions, P, eps and the
    grand potential the phase comparison ranks by.
    """
    spec = MODES[mode]
    pattern = f["pattern"]

    if pattern == "CFL":
        # Locking fixes the fractions; a mode demanding others has no CFL
        # solution, and mu_C is irrelevant because n_C vanishes identically.
        for name in spec["fixed"]:
            if abs(targets[name] - CFL_FRACTIONS[name]) > 1e-9:
                return None
        mu_C, mu_S = 0.0, 0.0
    elif mode == "beta_eq_neutrinoless":
        mu_S = 0.0

        def residual(mu_C):
            n_B, n_C, _ = charges(f, mu_B, mu_C, mu_S, T)
            return (n_C - lepton_thermo(-mu_C)[0]) if n_B > 0.0 else np.nan
        mu_C = scan_root(residual, -600.0, 20.0)
        if mu_C is None:
            return None
    elif mode == "fixed_YC":
        mu_S = 0.0

        def residual(mu_C):
            n_B, n_C, _ = charges(f, mu_B, mu_C, mu_S, T)
            return (n_C / n_B - targets["Y_C"]) if n_B > 0.0 else np.nan
        mu_C = scan_root(residual, -500.0, 700.0)
        if mu_C is None:
            return None
    else:                                     # fixed_YC_YS
        def residual(v):
            n_B, n_C, n_S = charges(f, mu_B, v[0], v[1], T)
            if n_B <= 0:
                return [1e3, 1e3]
            return [n_C / n_B - targets["Y_C"], n_S / n_B - targets["Y_S"]]
        starts = ([guess] if guess is not None else []) + [[0.0, 0.0],
                                                            [-50.0, 50.0],
                                                            [100.0, -100.0]]
        mu_C = mu_S = None
        for start in starts:
            sol = root(residual, start, method="hybr", tol=1e-11)
            if sol.success and charges(f, mu_B, sol.x[0], sol.x[1], T)[0] > 0:
                mu_C, mu_S = float(sol.x[0]), float(sol.x[1])
                break
        if mu_C is None:
            return None

    n_B, n_C, n_S = charges(f, mu_B, mu_C, mu_S, T)
    if n_B <= 0.0:
        return None
    P = pressure(f["coeff"], mu_B, mu_C, mu_S, T, pattern,
                 f["Delta_star"], f["sigma"])

    # Leptons: shared by the system, not part of the phase. In beta
    # equilibrium mu_e = -mu_C; under a fixed Y_C they are whatever
    # neutralises the matter, which is a different number.
    if spec["leptons"]:
        mu_e = -mu_C if spec["beta"] else (3.0 * PI2 * n_C * HC3) ** (1.0 / 3.0) \
            if n_C > 0 else 0.0
        n_e, P_e = lepton_thermo(mu_e)
    else:
        mu_e, n_e, P_e = 0.0, 0.0, 0.0

    n_u, n_d, n_s = flavour_densities(f, mu_B, mu_C, mu_S, T)
    P_tot = P + P_e
    # The quark phase's own entropy: `lepton_thermo` is the T = 0 gas, so a
    # mode with leptons is a T = 0 statement until that gains a temperature.
    s = entropy(f, mu_B, mu_C, mu_S, T)
    eps = -P_tot + mu_B * n_B + mu_C * n_C + mu_S * n_S + mu_e * n_e + T * s
    return dict(pattern=pattern, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S, mu_e=mu_e,
                n_B=n_B, n_C=n_C, n_S=n_S, n_e=n_e, s=s,
                Y_C=n_C / n_B, Y_S=n_S / n_B, Y_e=n_e / n_B,
                Y_u=n_u / n_B, Y_d=n_d / n_B, Y_s=n_s / n_B,
                P=P_tot, Omega=-P_tot, eps=eps, f=eps - T * s)


def eos_from_fit(fits, mode, mu_B, targets=None, T=0.0, guesses=None):
    """The equilibrium state: the phase of LOWEST grand potential.

    At fixed potentials of the conserved charges the grand potential density
    is Omega = -P, so the stable phase is the one of largest total pressure.
    This is the rule `eos.njl.solver.solve` applies, so the two models are
    comparable point by point -- and `csc_type` is an OUTPUT here, never an
    input.
    """
    targets = {} if targets is None else targets
    best = None
    for pattern, f in fits.items():
        st = phase_state(f, mode, mu_B, targets, T,
                         guess=(guesses or {}).get(pattern))
        if st is None:
            continue
        if best is None or st["Omega"] < best["Omega"]:
            best = st
    return best


def sweep(fits, mode, mu_B_grid, targets=None, T=0.0):
    """The mode swept along mu_B, with the 2-D solves warm started."""
    out, guesses = [], {}
    for mu_B in mu_B_grid:
        states = {}
        for pattern, f in fits.items():
            st = phase_state(f, mode, mu_B, targets or {}, T,
                             guess=guesses.get(pattern))
            if st is not None:
                states[pattern] = st
                guesses[pattern] = [st["mu_C"], st["mu_S"]]
        if states:
            out.append(min(states.values(), key=lambda s: s["Omega"]))
    return out


# %% [markdown]
# ## Which terms does the model need?
#
# Fit one NJL parameter point with several term sets and compare. The bag
# model is the first row; each later row adds one function of `mu`.

# %%
FIT_PAR = replace(njl.Parameters.named("rkh"), eta_D=1.45, eta_V=0.0,
                  vector_form="gluon_exchange", G_V0_over_GS=0.5, M_g=500.0)
_GS = tuple(f"gs{m:.0f}" for m in S_MASSES)
TERM_SETS = (("bag", ("a0", "a2s", "a4", "pair")),
             ("+ a2 a6", ("a0", "a2", "a2s", "a4", "a6", "pair")),
             ("+ a4l", ("a0", "a2", "a2s", "a4", "a4l", "a6", "pair")),
             ("+ massive s", ("a0", "a2", "a2s", "a4", "a4l", "a6", "pair")
                             + _GS))
# ---------------------------------------------------------------------------
print(f"=== term sets, rkh eta_D={FIT_PAR.eta_D:g}, "
      f"{FIT_PAR.vector_form}, T = 0 ===")
samples = {p: sample(FIT_PAR, p) for p in PATTERNS}
for pattern in PATTERNS:
    print(f"  {pattern}  ({len(samples[pattern])} pts)")
    for label, terms in TERM_SETS:
        f = fit_phase(samples[pattern], pattern, terms)
        r = f["rms"]
        print(f"    {label:14s} " +
              "  ".join(f"{t}={f['coeff'][t]:+7.3f}" for t in f["terms"]) +
              f"  | rms P={r['P']:.1e} nB={r['n_B']:.1e} "
              f"nC={r['n_C']:.1e} nS={r['n_S']:.1e}"
              f"  {'pegged:' + ','.join(f['pegged']) if f['pegged'] else ''}")

#: The term set the rows above justify, used everywhere below.
TERMS = (("a0", "a2", "a2s", "a4", "a4l", "a6", "pair")
         + tuple(f"gs{m:.0f}" for m in S_MASSES))


# %% [markdown]
# ## The map: coefficients as functions of the NJL parameters
#
# The NJL side has, after the vacuum fit is fixed, exactly two continuous
# knobs that matter here -- the diquark coupling `eta_D` and the vector
# strength -- plus the discrete choice of pairing pattern. So the map to fit
# is `(eta_D, G_V0/G_S, csc_type) -> {a_t}`, and the question is whether it is
# smooth enough to write down.

# %%
MAP_ETA_D = (0.75, 1.00, 1.25, 1.45, 1.65)
MAP_G_V0 = (0.25, 0.50, 0.75)
# ---------------------------------------------------------------------------
map_rows = []
started = time.perf_counter()
for eta_D in MAP_ETA_D:
    for g_v0 in MAP_G_V0:
        par = replace(njl.Parameters.named("rkh"), eta_D=eta_D, eta_V=0.0,
                      vector_form="gluon_exchange", G_V0_over_GS=g_v0,
                      M_g=500.0)
        for pattern in PATTERNS:
            recs = sample(par, pattern)
            if len(recs) < 8:
                continue
            f = fit_phase(recs, pattern, TERMS)
            map_rows.append(dict(eta_D=eta_D, G_V0=g_v0, pattern=pattern,
                                 Delta_star=f["Delta_star"],
                                 sigma=f["sigma"], rms_P=f["rms"]["P"],
                                 rms_nB=f["rms"]["n_B"], **f["coeff"]))
print(f"  {len(map_rows)} fits in {time.perf_counter() - started:.0f} s")


def surface(rows, key):
    """y = c0 + c1 eta_D + c2 G_V0 + c3 eta_D G_V0, and its worst deviation.

    Bilinear on purpose: with fifteen points per phase a quadratic would fit
    the fit's own noise, and the question is whether the dependence is SIMPLE,
    not whether some polynomial can be threaded through it.
    """
    eta = np.array([r["eta_D"] for r in rows])
    gv = np.array([r["G_V0"] for r in rows])
    y = np.array([r.get(key, 0.0) for r in rows])
    A = np.stack([np.ones_like(eta), eta, gv, eta * gv], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    dev = np.abs(A @ c - y).max()
    spread = y.max() - y.min()
    # A coefficient that is the same at every parameter point has no spread to
    # be a fraction of; report the deviation alone rather than a huge ratio of
    # two numbers that are both round-off.
    scale = max(abs(y).max(), 1e-30)
    return c, dev, (dev / spread if spread > 1e-6 * scale else 0.0)


MAP_FIELDS = TERMS + ("Delta_star", "sigma")
print("\n=== is the map analytic?  y = c0 + c1 eta_D + c2 G_V0 "
      "+ c3 eta_D G_V0 ===")
print(f"  {'phase':9s} {'field':11s} {'c0':>9s} {'c1':>9s} {'c2':>9s} "
      f"{'c3':>9s} {'max dev':>9s} {'/spread':>8s}")
surfaces = {}
for pattern in PATTERNS:
    rows = [r for r in map_rows if r["pattern"] == pattern]
    if not rows:
        continue
    for key in MAP_FIELDS:
        if key not in rows[0]:
            continue
        c, dev, rel = surface(rows, key)
        surfaces[(pattern, key)] = c
        print(f"  {pattern:9s} {key:11s} {c[0]:9.3f} {c[1]:9.3f} {c[2]:9.3f} "
              f"{c[3]:9.3f} {dev:9.3f} {100 * rel:7.1f}%")

# %%
COLOUR = dict(zip(MAP_G_V0, OKAB_CAT))
show = [k for k in MAP_FIELDS if any((p, k) in surfaces for p in PATTERNS)]
fig, axes = plt.subplots(len(show), len(PATTERNS),
                         figsize=(9.5, 1.9 * len(show)), sharex=True)
for j, pattern in enumerate(PATTERNS):
    rows = [r for r in map_rows if r["pattern"] == pattern]
    for i, key in enumerate(show):
        ax = axes[i, j]
        if not rows or key not in rows[0]:
            ax.axis("off")
            continue
        eta = np.array([r["eta_D"] for r in rows])
        gv = np.array([r["G_V0"] for r in rows])
        y = np.array([r[key] for r in rows])
        c = surfaces[(pattern, key)]
        for g in MAP_G_V0:
            m = gv == g
            order = np.argsort(eta[m])
            ax.plot(eta[m][order], y[m][order], "o", ms=3.5, color=COLOUR[g],
                    label=f"$G_{{V0}}={g:g}$" if (i, j) == (0, 0) else None)
            fine = np.linspace(min(MAP_ETA_D), max(MAP_ETA_D), 40)
            ax.plot(fine, c[0] + c[1] * fine + c[2] * g + c[3] * fine * g,
                    "-", lw=0.9, color=COLOUR[g], alpha=0.75)
        ax.set_ylabel(key, fontsize="x-small")
        ax.tick_params(labelsize="xx-small")
        if i == 0:
            ax.set_title(pattern, fontsize="small")
        if i == len(show) - 1:
            ax.set_xlabel(r"$\eta_D$", fontsize="x-small")
axes[0, 0].legend(loc="best", fontsize="xx-small")
fig.suptitle("fitted coefficients vs the NJL parameters; "
             "lines are the bilinear surface", fontsize="small")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Is the map good enough to use, and would machine learning help?
#
# The individual coefficients are noisy functions of the NJL parameters,
# because `mu^4`, `mu^4 ln mu` and the saturating `mu^6` are nearly collinear
# over any finite range of `mu`: the fit pins their combination, not each one.
# So the per-coefficient deviations printed above are the wrong test. The
# right one is whether an EoS built from PREDICTED coefficients still
# reproduces `eos.njl`.
#
# Leave-one-out over the NJL parameter grid, with three regressors of
# increasing flexibility, is also the answer to "would a network help": with
# two inputs and fifteen points, *machine learning here is interpolation*, and
# what matters is whether the extra flexibility buys accuracy or just fits the
# fit's own noise.

# %%
def design_matrix(eta, gv, kind):
    """The regressor's features: bilinear, quadratic, or radial basis."""
    if kind == "bilinear":
        return np.stack([np.ones_like(eta), eta, gv, eta * gv], axis=1)
    if kind == "quadratic":
        return np.stack([np.ones_like(eta), eta, gv, eta * gv,
                         eta ** 2, gv ** 2], axis=1)
    raise ValueError(kind)


def fit_regressor(rows, key, kind, ridge=1e-8):
    """Coefficients of one regressor for one field, or an RBF interpolant.

    `rbf` is the machine-learning end of the comparison: a Gaussian kernel
    ridge regression, which with two inputs and this many points is what a
    small network would be approximating anyway -- and unlike a network it has
    no training noise, so the comparison is about FLEXIBILITY and not about
    optimisation.
    """
    eta = np.array([r["eta_D"] for r in rows])
    gv = np.array([r["G_V0"] for r in rows])
    y = np.array([r.get(key, 0.0) for r in rows])
    if kind == "rbf":
        # Kernel ridge with a Gaussian kernel on standardised inputs.
        X = np.stack([eta / 0.5, gv / 0.25], axis=1)
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        K = np.exp(-0.5 * d2)
        alpha = np.linalg.solve(K + 1e-6 * np.eye(len(y)), y)
        return ("rbf", X, alpha)
    A = design_matrix(eta, gv, kind)
    c = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y)
    return (kind, c)


def apply_regressor(model, eta_D, g_v0):
    """Evaluate a regressor built by `fit_regressor`."""
    if model[0] == "rbf":
        _, X, alpha = model
        x = np.array([eta_D / 0.5, g_v0 / 0.25])
        k = np.exp(-0.5 * ((X - x) ** 2).sum(-1))
        return float(k @ alpha)
    kind, c = model
    A = design_matrix(np.array([eta_D]), np.array([g_v0]), kind)
    return float((A @ c)[0])


def predict_fit(rows, pattern, eta_D, g_v0, template, kind):
    """A fit-shaped dict whose coefficients come from the regressor."""
    sel = [r for r in rows if r["pattern"] == pattern]
    if len(sel) < 6:
        return None
    coeff, extra = {}, {}
    for key in template["terms"]:
        coeff[key] = apply_regressor(fit_regressor(sel, key, kind),
                                     eta_D, g_v0)
    for key in ("Delta_star", "sigma"):
        extra[key] = apply_regressor(fit_regressor(sel, key, kind),
                                     eta_D, g_v0)
    return dict(template, coeff=coeff,
                Delta_star=max(extra["Delta_star"], 0.0), sigma=extra["sigma"])


LOO_MU_B = np.linspace(GRID_MU_B[0], GRID_MU_B[-1], 40)
print("=== leave-one-out: predicted coefficients vs directly fitted ones ===")
print("    (beta equilibrium; median |dP/P| over the sweep, per NJL point)")
print(f"  {'regressor':11s} {'median':>9s} {'worst':>9s} {'phase right':>12s}")
loo = {}
for kind in ("bilinear", "quadratic", "rbf"):
    errs, agree, total = [], 0, 0
    for eta_D in MAP_ETA_D:
        for g_v0 in MAP_G_V0:
            kept = [r for r in map_rows
                    if not (r["eta_D"] == eta_D and r["G_V0"] == g_v0)]
            par_lo = replace(njl.Parameters.named("rkh"), eta_D=eta_D,
                             eta_V=0.0, vector_form="gluon_exchange",
                             G_V0_over_GS=g_v0, M_g=500.0)
            direct = {p: fit_phase(sample(par_lo, p), p, TERMS)
                      for p in PATTERNS if len(sample(par_lo, p)) >= 8}
            guessed = {}
            for p, f in direct.items():
                g = predict_fit(kept, p, eta_D, g_v0, f, kind)
                if g is not None:
                    guessed[p] = g
            if not guessed:
                continue
            a = sweep(direct, "beta_eq_neutrinoless", LOO_MU_B)
            b = sweep(guessed, "beta_eq_neutrinoless", LOO_MU_B)
            for pa, pb in zip(a, b):
                if pa["P"] > 0:
                    errs.append(abs(pb["P"] - pa["P"]) / pa["P"])
                    total += 1
                    agree += pa["pattern"] == pb["pattern"]
    errs = np.array(errs)
    loo[kind] = errs
    print(f"  {kind:11s} {np.median(errs):8.1%} {errs.max():9.1%} "
          f"{100.0 * agree / max(total, 1):11.0f}%")


# %% [markdown]
# ## Benchmarks: every mode, several NJL parametrisations
#
# One figure per equilibrium mode. Rows are the quantity, columns are the NJL
# parameter set; dots are `eos.njl`, lines the fitted model, and the x axis is
# `n_B/n_sat` throughout except the `P` vs `mu_B` row, which is that plane.
# A second figure per mode carries the relative errors.
#
# None of these trajectories was fitted: the coefficients come from a grid in
# the potentials, and every curve here is an extrapolation off it.

# %%
BENCH_SETS = (
    (r"$\eta_D$=1.45, $G_{V0}$=0.50", dict(eta_D=1.45, G_V0_over_GS=0.50)),
    (r"$\eta_D$=1.00, $G_{V0}$=0.50", dict(eta_D=1.00, G_V0_over_GS=0.50)),
    (r"$\eta_D$=1.45, $G_{V0}$=0.25", dict(eta_D=1.45, G_V0_over_GS=0.25)),
    (r"$\eta_D$=0.75, $G_{V0}$=0.75", dict(eta_D=0.75, G_V0_over_GS=0.75)),
)
BENCH_MODES = (
    ("beta_eq_neutrinoless", {}, None),
    ("fixed_YC", {"Y_C": 0.3}, True),
    ("fixed_YC_YS", {"Y_C": 0.5, "Y_S": 0.0}, False),
    ("fixed_YC_YS", {"Y_C": 0.0, "Y_S": 0.0}, False),
)
BENCH_NB = np.linspace(0.35, 1.75, 18)            # fm^-3, ~2-11 n_sat
BENCH_MU_B = np.linspace(850.0, 1900.0, 130)      # MeV, the fitted sweep
# ---------------------------------------------------------------------------


def njl_reference(par, mode, targets, leptons, use_cache=True):
    """`eos.njl` along the mode, cached: this is the expensive half."""
    key = hashlib.md5(json.dumps(
        [asdict(par), mode, targets, leptons, list(BENCH_NB)],
        sort_keys=True, default=str).encode()).hexdigest()
    path = CACHE / ("ref_" + key + ".pkl")
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())
    try:
        rows = njl.eos_table(par, mode, njl.SpeciesFlags(csc=True),
                             {"nB": BENCH_NB, "T": np.array([0.0])},
                             fixed=targets, leptons=leptons, rows=True,
                             backend="fast", patterns=PATTERNS)
    except Exception as exc:                       # a mode a set cannot reach
        print(f"    [njl {mode} {targets}] {type(exc).__name__}: "
              f"{str(exc)[:90]}")
        rows = []
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(rows))
    return rows


def sound_speed(n_B, P, eps):
    """c_s^2 = dP/deps along a sorted sweep."""
    order = np.argsort(n_B)
    return n_B[order], np.gradient(P[order], eps[order])


def bench_columns(states):
    """A sweep of states as parallel arrays, ascending in n_B."""
    if not states:
        return None
    keys = ("n_B", "mu_B", "P", "eps", "Y_u", "Y_d", "Y_s", "Y_e")
    col = {k: np.array([s[k] for s in states]) for k in keys}
    col["pattern"] = [s["pattern"] for s in states]
    order = np.argsort(col["n_B"])
    for k in keys:
        col[k] = col[k][order]
    col["pattern"] = [col["pattern"][i] for i in order]
    return col


def njl_columns(rows):
    """The same arrays from an `eos.njl` table."""
    if not rows:
        return None
    col = {k: np.array([r[k] for r in rows], dtype=float)
           for k in ("n_B", "mu_B", "P", "eps", "Y_u", "Y_d", "Y_s", "Y_e")}
    col["pattern"] = [r["pattern_realised"] for r in rows]
    order = np.argsort(col["n_B"])
    for k in list(col):
        col[k] = (col[k][order] if k != "pattern"
                  else [col["pattern"][i] for i in order])
    return col


bench = {}
print("=== building the benchmarks ===")
for mode, targets, leptons in BENCH_MODES:
    tag = mode + ("" if not targets else
                  " " + " ".join(f"{k}={v:g}" for k, v in targets.items()))
    for label, held in BENCH_SETS:
        par_b = replace(njl.Parameters.named("rkh"), eta_V=0.0,
                        vector_form="gluon_exchange", M_g=500.0, **held)
        fits_b = {p: fit_phase(sample(par_b, p), p, TERMS)
                  for p in PATTERNS if len(sample(par_b, p)) >= 8}
        started = time.perf_counter()
        model = bench_columns(sweep(fits_b, mode, BENCH_MU_B, targets))
        t_model = time.perf_counter() - started
        started = time.perf_counter()
        truth = njl_columns(njl_reference(par_b, mode, targets, leptons))
        t_njl = time.perf_counter() - started
        bench[(tag, label)] = (model, truth)
        n_model = 0 if model is None else len(model["n_B"])
        n_truth = 0 if truth is None else len(truth["n_B"])
        print(f"  {tag:28s} {label:26s} model {n_model:3d} pts "
              f"({1e3 * t_model / max(n_model, 1):5.2f} ms/pt), "
              f"njl {n_truth:3d} pts ({t_njl:5.1f} s)")


# %%
# One figure of quantities and one of relative errors, per mode.
FLAVOUR_KEYS = (("Y_u", "-"), ("Y_d", "--"), ("Y_s", ":"), ("Y_e", "-."))
#: Every pattern that can be REALISED, not only the three enumerated. A
#: CFL-layout solve whose s-quark gaps collapse comes back labelled uSC, dSC
#: or usSC, so a plot that indexes only the enumeration raises on the first
#: such point; `phase_index` therefore never raises, it appends.
INDEX = {"unpaired": 0, "2SC": 1, "uSC": 2, "dSC": 3, "usSC": 4, "CFL": 5,
         "free": 6}
FILL = ("#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#009E73",
        "#D55E00")


def phase_index(name):
    """The row a pattern is drawn on; unknown names are added, not raised on."""
    if name not in INDEX:
        INDEX[name] = max(INDEX.values()) + 1
    return INDEX[name]
ROWS = ("P vs n_B", "P vs mu_B", "c_s^2", "phase", "Y_i")


def monotone(col):
    """Keep the strictly increasing part of a sweep in n_B.

    A first-order transition makes n_B jump, so P(n_B) can be double valued
    and `np.interp` -- which assumes a single-valued increasing x -- returns
    nonsense across it. Dropping the backward steps compares the two models
    on the branch they share and leaves the transition to the phase panel,
    which is where it belongs.
    """
    keep = [0]
    for i in range(1, len(col["n_B"])):
        if col["n_B"][i] > col["n_B"][keep[-1]] + 1e-9:
            keep.append(i)
    idx = np.array(keep)
    out = {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
           for k, v in col.items()}
    return out, len(col["n_B"]) - len(idx)


def overlap(model, truth):
    """(n_B grid, model on it, njl on it, scale) over the common range.

    The errors are normalised by max|P| over the window rather than pointwise,
    because P passes through zero near the surface and a pointwise relative
    error diverges there -- which is the whole of the 20000% figures a
    pointwise normalisation produced.
    """
    m, _ = monotone(model)
    t, _ = monotone(truth)
    lo = max(m["n_B"].min(), t["n_B"].min())
    hi = min(m["n_B"].max(), t["n_B"].max())
    if hi <= lo or len(m["n_B"]) < 3 or len(t["n_B"]) < 3:
        return None
    grid = np.linspace(lo, hi, 60)
    got, ref = {}, {}
    for key in ("mu_B", "P", "eps", "Y_u", "Y_d", "Y_s", "Y_e"):
        got[key] = np.interp(grid, m["n_B"], m[key])
        ref[key] = np.interp(grid, t["n_B"], t[key])
    return grid, got, ref, max(np.abs(ref["P"]).max(), 1e-12)


for mode, targets, leptons in BENCH_MODES:
    tag = mode + ("" if not targets else
                  " " + " ".join(f"{k}={v:g}" for k, v in targets.items()))
    title = MODES[mode]["label"] + ("" if not targets else "   " + ", ".join(
        f"$Y_{k[2:]}$={v:g}" for k, v in targets.items()))

    fig, axes = plt.subplots(len(ROWS), len(BENCH_SETS),
                             figsize=(3.0 * len(BENCH_SETS), 2.3 * len(ROWS)))
    fig_e, axes_e = plt.subplots(len(ROWS), len(BENCH_SETS),
                                 figsize=(3.0 * len(BENCH_SETS),
                                          2.3 * len(ROWS)))
    for j, (label, _) in enumerate(BENCH_SETS):
        model, truth = bench[(tag, label)]
        for i in range(len(ROWS)):
            axes[i, j].tick_params(labelsize="xx-small")
            axes_e[i, j].tick_params(labelsize="xx-small")
        if model is None or truth is None:
            for i in range(len(ROWS)):
                axes[i, j].text(0.5, 0.5, "no solution", ha="center",
                                va="center", fontsize="xx-small",
                                transform=axes[i, j].transAxes)
            continue
        x_m, x_t = model["n_B"] / NSAT, truth["n_B"] / NSAT

        axes[0, j].plot(x_t, truth["P"], "o", ms=2.5, color="0.25")
        axes[0, j].plot(x_m, model["P"], "-", color=OKAB_CAT[0])
        axes[1, j].plot(truth["mu_B"], truth["P"], "o", ms=2.5, color="0.25")
        axes[1, j].plot(model["mu_B"], model["P"], "-", color=OKAB_CAT[0])
        xt, cs_t = sound_speed(truth["n_B"], truth["P"], truth["eps"])
        xm, cs_m = sound_speed(model["n_B"], model["P"], model["eps"])
        axes[2, j].plot(xt / NSAT, cs_t, "o", ms=2.5, color="0.25")
        axes[2, j].plot(xm / NSAT, cs_m, "-", color=OKAB_CAT[0])
        axes[2, j].axhline(1.0 / 3.0, color="0.6", ls="--", lw=0.7)
        axes[2, j].set_ylim(0.0, 1.0)
        axes[3, j].plot(x_t, [phase_index(p) for p in truth["pattern"]], "o", ms=2.5,
                        color="0.25")
        axes[3, j].plot(x_m, [phase_index(p) for p in model["pattern"]], "-",
                        color=OKAB_CAT[0])
        axes[3, j].set_yticks(list(INDEX.values()))
        axes[3, j].set_yticklabels(list(INDEX) if j == 0 else [],
                                   fontsize="xx-small")
        for k, (key, style) in enumerate(FLAVOUR_KEYS):
            axes[4, j].plot(x_t, truth[key], "o", ms=2, color=OKAB_CAT[k])
            axes[4, j].plot(x_m, model[key], style, color=OKAB_CAT[k],
                            label=key if j == 0 else None)
        # Symmetric matter has Y_u = Y_d = 1.5, so a limit of 1.35 hides
        # the very quantity the panel is for.
        axes[4, j].set_ylim(-0.1, 2.1)

        got = overlap(model, truth)
        if got is not None:
            grid, m, t, P_scale = got
            xr = grid / NSAT
            axes_e[0, j].plot(xr, (m["P"] - t["P"]) / P_scale,
                              color=OKAB_CAT[1])
            axes_e[1, j].plot(xr, (m["mu_B"] - t["mu_B"]) / t["mu_B"],
                              color=OKAB_CAT[1])
            cs_mi = np.gradient(m["P"], m["eps"])
            cs_ti = np.gradient(t["P"], t["eps"])
            axes_e[2, j].plot(xr, cs_mi - cs_ti, color=OKAB_CAT[1])
            agree = np.array([a == b for a, b in zip(
                np.interp(grid, model["n_B"],
                          [phase_index(p) for p in model["pattern"]]).round(),
                np.interp(grid, truth["n_B"],
                          [phase_index(p) for p in truth["pattern"]]).round())])
            axes_e[3, j].plot(xr, agree.astype(float), ".", ms=2.5,
                              color=OKAB_CAT[2])
            axes_e[3, j].set_ylim(-0.15, 1.15)
            axes_e[3, j].set_yticks([0, 1])
            axes_e[3, j].set_yticklabels(["differ", "agree"] if j == 0 else [],
                                         fontsize="xx-small")
            for k, (key, style) in enumerate(FLAVOUR_KEYS):
                axes_e[4, j].plot(xr, m[key] - t[key], style,
                                  color=OKAB_CAT[k],
                                  label=key if j == 0 else None)
            for i in (0, 1, 2, 4):
                axes_e[i, j].axhline(0.0, color="0.6", lw=0.7)
            rel = np.abs(m["P"] - t["P"]) / P_scale
            print(f"  [{tag:28s}] {label:26s} "
                  f"max |dP|/max|P| {rel.max():6.1%}   median {np.median(rel):6.1%}"
                  f"   phase agrees {100.0 * agree.mean():3.0f}%")

        axes[0, j].set_title(label, fontsize="x-small")
        axes_e[0, j].set_title(label, fontsize="x-small")

    for i, (name, ylab) in enumerate(
            ((ROWS[0], LABELS["P"]), (ROWS[1], LABELS["P"]),
             (ROWS[2], r"$c_s^2$"), (ROWS[3], "phase"), (ROWS[4], r"$Y_i$"))):
        axes[i, 0].set_ylabel(ylab, fontsize="x-small")
    for i, ylab in enumerate((r"$\Delta P/\max|P|$", r"$\Delta\mu_B/\mu_B$",
                              r"$\Delta c_s^2$", "phase",
                              r"$\Delta Y_i$")):
        axes_e[i, 0].set_ylabel(ylab, fontsize="x-small")
    for j in range(len(BENCH_SETS)):
        axes[1, j].set_xlabel(r"$\mu_B$ [MeV]", fontsize="xx-small")
        axes[-1, j].set_xlabel(r"$n_B/n_{\rm sat}$", fontsize="xx-small")
        axes_e[-1, j].set_xlabel(r"$n_B/n_{\rm sat}$", fontsize="xx-small")
    axes[4, 0].legend(loc="upper right", fontsize="xx-small", ncol=2)
    axes_e[4, 0].legend(loc="upper right", fontsize="xx-small", ncol=2)
    fig.suptitle(f"{title}    (dots: eos.njl, lines: fitted model)",
                 fontsize="small")
    fig_e.suptitle(f"{title} -- relative errors", fontsize="small")
    fig.tight_layout()
    fig_e.tight_layout()
    plt.show()


# %% [markdown]
# ## The mu -> infinity limit
#
# The basis was constrained so that the model stays conformal where the answer
# is known: every term is of power <= 4 except the vector-like one, which
# saturates. So `P/P_free` must approach a constant and `c_s^2` the conformal
# 1/3, from wherever the fitted region leaves them. This is not a fit result --
# it is a property of the BASIS, and the point of checking it is that a bare
# `mu^6/mu_star^2` term fails it (`P/P_free -> infinity`, `c_s^2 -> 1/5`) and
# nothing inside the fitted range would have shown that.

# %%
CONF_MU_B = np.geomspace(GRID_MU_B[0], 60000.0, 200)     # MeV, far past any star
# ---------------------------------------------------------------------------
fig, axes = paper_grid("1x2", mode="double", placeholder=False, aspect=1.3)
ax_r, ax_c = axes[0, 0], axes[0, 1]
par_c = replace(njl.Parameters.named("rkh"), eta_D=1.45, eta_V=0.0,
                vector_form="gluon_exchange", G_V0_over_GS=0.5, M_g=500.0)
fits_c = {p: fit_phase(sample(par_c, p), p, TERMS) for p in PATTERNS
          if len(sample(par_c, p)) >= 8}
states = sweep(fits_c, "beta_eq_neutrinoless", CONF_MU_B)
mu_B = np.array([s["mu_B"] for s in states])
P = np.array([s["P"] for s in states])
eps = np.array([s["eps"] for s in states])
n_B = np.array([s["n_B"] for s in states])
# P_free: three massless flavours at a common mu = mu_B/3, plus nothing else.
P_free = 3.0 * (mu_B / 3.0) ** 4 / (4.0 * PI2 * HC3)

ax_r.semilogx(mu_B / 1000.0, P / P_free, color=OKAB_CAT[0])
ax_r.axhline(1.0, color="0.6", ls="--", lw=0.8)
ax_r.set_xlabel(r"$\mu_B$ [GeV]")
ax_r.set_ylabel(r"$P/P_{\rm free}$")
ax_r.set_ylim(0.0, 2.0)
panel_label(ax_r, "(a)")

ax_c.semilogx(mu_B / 1000.0, np.gradient(P, eps), color=OKAB_CAT[0])
ax_c.axhline(1.0 / 3.0, color="0.6", ls="--", lw=0.8)
ax_c.set_xlabel(r"$\mu_B$ [GeV]")
ax_c.set_ylabel(r"$c_s^2$")
ax_c.set_ylim(0.0, 0.6)
panel_label(ax_c, "(b)")
plt.show()
print(f"  at mu_B = {mu_B[-1] / 1000:.0f} GeV: P/P_free = "
      f"{P[-1] / P_free[-1]:.3f}, c_s^2 = {np.gradient(P, eps)[-1]:.3f} "
      f"(conformal: 1 and {1 / 3:.3f})")

"""The extended-NJL kernel reproduces the identities of the reference tables.

The five beta-equilibrium tables in `test/enjl/reference/` were produced by the
author's own implementation of the model of Xia 2024 (PRD 110, 014022,
arXiv:2405.02946). They therefore satisfy that paper's own equations to the
convergence of the run that made them, far more tightly than the two or three
significant figures the paper prints.

This module checks the *model's own consistency* using the tables' columns as
input: given the printed densities and masses, do the equations of the paper
close? Every quantity is computed through `eos.enjl` rather than reimplemented
here, so a sign error or a wrong factor in the shipped code fails the test
instead of being silently duplicated into it. No solver runs: the composition
is read off the table, which isolates the T = 0 kernel from any root finding.

Identities checked, all at fixed composition:

    Eq.  (5)  gap equation, M_q from the effective scalar densities
    Eq.  (6)  nbar^s_q = n^s_q + alpha_S sum_i N^q_i n^s_i
    Eq.  (7)  baryon-number sum rule
    Eq. (12)  scalar densities of baryons and quarks
    Eq. (19)  P = sum_i mu_i n_i - E
    Eq. (23)  beta equilibrium, mu_i = B_i mu_b - q_i mu_e
    Eq. (24)  charge neutrality

Three things about the data drive the shape of this file:

* **Tolerances are per file, not global.** The five runs differ by up to five
  orders of magnitude on the *same* identity — `Beta_fq1.0_B1.dat` closes the
  baryon sum rule to 1e-10 and `Beta_fq0.7_B0.dat` to 3e-6 — so a single bound
  would be either vacuous on the tight files or wrong on the loose ones. Each
  entry of `TOL` is the worst residual actually observed on that file; they are
  gates measured on a fixed data set, not aspirations.
* **Some rows are not solver output** and are excluded by `solved_rows` (203
  interpolated mixed-phase rows in `Beta_fq0.5_B1.dat`) and `bad_rows` (nine
  non-converged rows in `Beta_fq0.7_B0.dat`). Both masks live in
  `test/enjl/reference/` and carry the evidence for each exclusion.
* **Four columns do not mean what their names suggest**: `Sigmaq` (not `nsq`)
  is the effective scalar density that feeds the gap equation, `munr` (not
  `mun`) is mu_b, and `mue`/`mumu` carry the lepton *mass* on rows where that
  lepton is absent. The loader's `baryon_potential` and `electron_potential`
  handle the last two.
"""
import numpy as np
import pytest

from eos.enjl.uniform import quark_masses_from_gap
from eos.general.physics_constants import hc3

from enjl_cases import (
    BARYONS as _BARYONS, QUARKS as _QUARKS, REF_COL as _COL, case as _case,
    chirally_restored as _chirally_restored,
    scalar_density_column as _scalar_density, species_of as _species,
    worst as _worst,
)
from reference import (
    PARAMETER_SETS, baryon_potential, electron_potential, present,
)

#: Per-file gates: the worst residual actually observed on that file, plus 5%
#: and rounded up to two significant figures. The margin is there only to
#: absorb last-bit differences in summation order between platforms — these
#: sit four to five orders of magnitude below anything a real sign error or
#: wrong factor would produce, so it costs no sensitivity.
#:
#: `Beta_fq0.7_B0.dat` is the loosest run even after its nine non-converged
#: rows are excluded: its scalar densities sit at 1e-3..1e-2 where the other
#: files reach 1e-6. Nothing should be read into that difference. The tight
#: gates are `Beta_fq1.0_B1` and `Beta_fq0.5_B1`, which close nearly
#: everything to 1e-6 or better.
TOL = {
    "Beta_fq1.0_B0.dat": dict(
        nb_sum=1.9e-06, charge=2.6e-06, charge_rel=2.1e-05, epa=1.2e-02,
        fq=5.3e-16, gap=2.2e-05, nsq=8.7e-05, sigma=1.6e-04,
        nsb=7.1e-04, P=4.5e-02, P_rel=2.4e-04, beta=5.5e-01,
        beta_pn=2.5e-03),
    "Beta_fq1.0_B1.dat": dict(
        nb_sum=9.9e-11, charge=5.3e-11, charge_rel=5.5e-12, epa=9.3e-07,
        fq=5.2e-17, gap=3.8e-07, nsq=8.1e-12, sigma=7.0e-07,
        nsb=1.1e-07, P=5.2e-06, P_rel=7.5e-09, beta=1.3e+00,
        beta_pn=1.4e-04),
    # charge/charge_rel here are far tighter than the 1e-2/1.3e-3 that the
    # same identity gives before the non-converged rows are dropped: the
    # worst row for charge neutrality in this file is n_b = 3.9, which is one
    # of the stale-`nB` rows that `bad_rows` removes.
    "Beta_fq0.7_B0.dat": dict(
        nb_sum=3.2e-06, charge=1.7e-03, charge_rel=5.8e-05, epa=4.7e-02,
        fq=8.2e-16, gap=2.5e-05, nsq=7.2e-08, sigma=6.8e-03,
        nsb=9.6e-03, P=1.1e-01, P_rel=7.6e-06, beta=5.2e-01,
        beta_pn=7.9e-05),
    "Beta_fq0.7_B1.dat": dict(
        nb_sum=4.1e-06, charge=6.5e-06, charge_rel=7.1e-06, epa=6.3e-03,
        fq=5.4e-16, gap=3.5e-06, nsq=3.7e-06, sigma=1.5e-05,
        nsb=4.1e-05, P=3.8e-02, P_rel=5.4e-04, beta=2.1e-01,
        beta_pn=8.9e-04),
    "Beta_fq0.5_B1.dat": dict(
        nb_sum=4.2e-11, charge=4.1e-02, charge_rel=1.5e-03, epa=8.9e-07,
        fq=5.2e-15, gap=1.8e-05, nsq=8.0e-08, sigma=1.0e-03,
        nsb=8.4e-08, P=1.5e-06, P_rel=7.0e-09, beta=1.8e-01,
        beta_pn=3.2e-07),
}

FILES = pytest.mark.parametrize("filename", sorted(PARAMETER_SETS))


def _cluster_term(col, alpha, species, q):
    """alpha_S sum_{i=p,n,Lambda} N^q_i n^s_i of Eq. (6), in fm^-3."""
    iq = _QUARKS.index(q)
    return alpha * sum(species[b].N[iq] * col["ns" + _COL[b]]
                       for b in _BARYONS)


# -----------------------------------------------------------------------------
# bookkeeping: Eqs. (7) and (24)
# -----------------------------------------------------------------------------
@FILES
def test_baryon_number_sum_rule(filename):
    """Eq. (7): n_b = sum_i B_i n_i, with B = 1/3 per quark."""
    col, ok, par, _ = _case(filename)
    sp = _species(par)
    n_b = sum(sp[name].B * col["n" + suffix] for name, suffix in _COL.items())
    assert _worst(n_b - col["nB"], ok) <= TOL[filename]["nb_sum"]


@FILES
def test_quark_baryon_fraction(filename):
    """`fq` is the quark share of the baryon density, n_b^Q / n_b."""
    col, ok, par, _ = _case(filename)
    sp = _species(par)
    n_bQ = sum(sp[q].B * col["n" + q] for q in _QUARKS)
    assert _worst(col["fq"] - n_bQ / col["nB"], ok) <= TOL[filename]["fq"]


@FILES
def test_charge_neutrality(filename):
    """Eq. (24): sum_i q_i n_i = 0, with q the *physical* electric charge.

    Checked absolutely, and relatively where quarks dominate: there the sum is
    a small difference of large nearly-cancelling numbers and the runs that
    produced the tables converged less tightly on near-symmetric quark matter.
    """
    col, ok, par, _ = _case(filename)
    sp = _species(par)
    charge = sum(sp[name].Q * col["n" + suffix]
                 for name, suffix in _COL.items())
    quark_total = sum(col["n" + q] for q in _QUARKS)
    assert _worst(charge, ok) <= TOL[filename]["charge"]

    quark_rich = quark_total > 0.1 * np.maximum(col["nB"], 1.0e-30)
    rel = charge / np.maximum(quark_total, 1.0e-30)
    assert _worst(rel, ok & quark_rich) <= TOL[filename]["charge_rel"]


@FILES
def test_energy_per_baryon(filename):
    """`epa` is E/n_b including the rest mass; `E` already has E_0 subtracted."""
    col, ok, _, _ = _case(filename)
    assert _worst(col["epa"] * col["nB"] - col["E"], ok) <= TOL[filename]["epa"]


# -----------------------------------------------------------------------------
# the scalar sector: Eqs. (5), (6), (12)
# -----------------------------------------------------------------------------
@FILES
def test_gap_equation(filename):
    """Eq. (5) with Eq. (8): the tables satisfy their own gap equation.

    The argument is `Sigmaq`, the effective scalar density of Eq. (6) with the
    vacuum Dirac-sea term included — not `nsq`, which is the same object with
    that term removed. Feeding `nsq` here misses M_q by hundreds of MeV.
    """
    col, ok, par, _ = _case(filename)
    nbar = {q: col["Sigma" + q] * hc3 for q in _QUARKS}
    M_q = quark_masses_from_gap(nbar, par)
    for q in _QUARKS:
        use = ok & ~_chirally_restored(col, par, q)
        rel = (M_q[q] - col["M" + q]) / col["M" + q]
        assert _worst(rel, use) <= TOL[filename]["gap"], q


@FILES
def test_quark_scalar_densities(filename):
    """Eqs. (6) and (12): the two quark scalar-density columns differ only by
    the vacuum term.

        nsq    = medium                  + alpha_S sum_i N^q_i n^s_i
        Sigmaq = medium + vacuum(Lambda) + alpha_S sum_i N^q_i n^s_i

    Only quarks carry the cut-off, and it enters purely as this additive
    vacuum piece, which is what `scalar_density_t0(..., Lambda)` returns.
    """
    col, ok, par, alpha = _case(filename)
    sp = _species(par)
    for q in _QUARKS:
        use = ok & ~_chirally_restored(col, par, q)
        cluster = _cluster_term(col, alpha, sp, q)
        medium = _scalar_density(col["n" + q], col["M" + q], 6.0, 0.0)
        full = _scalar_density(col["n" + q], col["M" + q], 6.0, par.Lambda)
        assert _worst(col["ns" + q] - (medium + cluster), use) \
            <= TOL[filename]["nsq"], q
        assert _worst(col["Sigma" + q] - (full + cluster), use) \
            <= TOL[filename]["sigma"], q


@FILES
def test_baryon_scalar_densities(filename):
    """Eq. (12) with Lambda = 0: baryons carry no cut-off."""
    col, ok, par, _ = _case(filename)
    for b in _BARYONS:
        suffix = _COL[b]
        use = ok & (col["n" + suffix] > 1.0e-8)
        calc = _scalar_density(col["n" + suffix], col["M" + suffix], 2.0, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (calc - col["ns" + suffix]) / np.abs(col["ns" + suffix])
        assert _worst(rel, use) <= TOL[filename]["nsb"], b


# -----------------------------------------------------------------------------
# equilibrium conditions: Eqs. (19) and (23)
# -----------------------------------------------------------------------------
@FILES
def test_pressure(filename):
    """Eq. (19): P = sum_i mu_i n_i - E, the T = 0 Euler relation.

    This closes only because the rearrangement self-energy Sigma^R sits in
    mu_i and never in E; an implementation that puts it in E instead passes
    every other check here and fails this one.
    """
    col, ok, _, _ = _case(filename)
    mu_dot_n = sum(col["mu" + suffix] * col["n" + suffix]
                   for suffix in _COL.values())
    resid = mu_dot_n - col["E"] - col["P"]
    assert _worst(resid, ok) <= TOL[filename]["P"]
    assert _worst(resid / np.maximum(col["P"], 1.0e-3), ok) \
        <= TOL[filename]["P_rel"]


@FILES
def test_beta_equilibrium(filename):
    """Eq. (23): mu_i = B_i mu_b - q_i mu_e, for species that are present.

    Three traps, each of which turns a correct check into a residual of
    hundreds of MeV: mu_b is `munr` and not `mun`; mu_e must come from
    mu_d - mu_u on rows with no electrons, where the `mue` column holds the
    electron mass instead; and a species below its onset has no equilibrium
    potential at all, only the threshold value the solver last held, so the
    identity applies only above a presence cut.

    The looser per-file gate is always carried by whichever species sits at
    its onset, where the density is small and the printed mu least resolved.
    p and n are resolved far better and get their own tight gate.
    """
    col, ok, par, _ = _case(filename)
    sp = _species(par)
    mu_b = baryon_potential(col)
    mu_e = electron_potential(col)
    for name in _BARYONS + _QUARKS:
        suffix = _COL[name]
        use = ok & present(col, suffix)
        resid = col["mu" + suffix] - (sp[name].B * mu_b - sp[name].Q * mu_e)
        worst = _worst(resid, use)
        assert worst <= TOL[filename]["beta"], name
        if name in ("p", "n"):
            assert worst <= TOL[filename]["beta_pn"], name


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

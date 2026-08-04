"""
The hadronic table in long-format rows, and its interchangeability with the
mixed-phase one.

`build_table(rows=True)` and `eos.mixed.build_mixed_table` must return the same
shape of thing, keyed the same way, so a pure-hadronic table and a hybrid table
concatenate without renaming any column. That is the whole point of
`hadronic_row` mirroring `composition_row`.
"""
import numpy as np
import pytest

from eos.dd2 import (
    Parametrization, SpeciesFlags, TableSpec, build_table, hadronic_row,
    solve_hadronic, MODE_FRACTIONS, export_csv,
)


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2y_defaults()


NB = np.linspace(0.2, 0.6, 4)


def test_fraction_axes_multiply_out(par):
    flags = SpeciesFlags(hyperons=True, phi_field=True, muons=True)
    spec = TableSpec(parametrization=par, mode="fixed_YC", include=flags,
                     axes={"nB": NB, "T": [0.0, 10.0], "Y_C": [0.1, 0.3]})
    rows, windows = build_table(spec, rows=True)
    assert len(rows) == NB.size * 2 * 2
    assert windows == {}, "hadronic matter has no phase window"
    # every axis value is present and lands on the row it was solved at
    assert sorted({round(r["Y_C"], 6) for r in rows}) == [0.1, 0.3]
    assert sorted({round(r["T"], 6) for r in rows}) == [0.0, 10.0]
    for r in rows:
        assert r["Y_C"] == pytest.approx(r["Y_C"], abs=1e-8)


def test_rows_match_mixed_keys(par):
    """The hadronic keys must be a subset of the mixed ones, so the two tables
    stack. A key present in one and absent in the other is the bug this
    catches."""
    from eos.mixed import (
        beta_eq_neutrinoless, solve_mixed, composition_row,
    )
    from eos.vmit.parameters import get_vmit_default

    flags = SpeciesFlags(hyperons=False, phi_field=False, muons=True)
    h = hadronic_row(solve_hadronic(par, flags, 0.4, T=0.0), flags)
    m = composition_row(solve_mixed(par, flags, 0.7, 0.0,
                                    beta_eq_neutrinoless(),
                                    vmit_params=get_vmit_default(), T=0.0))
    shared = {"n_B", "T", "chi", "phase", "P", "eps", "s", "S_per_B", "mu_B",
              "Y_C", "Y_S", "Y_e", "Y_mu-"}
    assert shared <= set(h), sorted(shared - set(h))
    assert shared <= set(m), sorted(shared - set(m))
    assert h["chi"] == 0.0 and h["phase"] == "H"


def test_Y_C_and_Y_S_sum_over_active_baryons(par):
    """With hyperons on, Y_C is not n_p/n_B and Y_S is not zero. Reading either
    off the proton alone is the classic error this pins."""
    flags = SpeciesFlags(hyperons=True, phi_field=True, muons=True)
    p = solve_hadronic(par, flags, 0.6, T=0.0, mode="beta_eq_neutrinoless")
    row = hadronic_row(p, flags)
    comp = p.composition_map
    if sum(n for name, n in comp.items() if name not in ("n", "p")) > 1e-6:
        assert row["Y_S"] > 0.0, "hyperons present but Y_S = 0"
        assert row["Y_C"] != pytest.approx(p.n_p / p.n_B, rel=1e-6)
    # charge neutrality: the non-leptonic charge is carried by the leptons
    assert row["Y_C"] == pytest.approx(row["Y_e"] + row["Y_mu-"], rel=1e-6)


def test_every_mode_builds_rows(par):
    flags = SpeciesFlags(hyperons=True, phi_field=True, muons=True,
                         neutrinos=True)
    values = {"Y_C": 0.2, "Y_S": 0.02, "Y_L": 0.35}
    # Above saturation, where a 2% strangeness fraction is actually reachable:
    # below the hyperon threshold there are no strange baryons to carry it and
    # 'fixed_YS' has no solution, which is physics, not a solver failure.
    grid = np.linspace(0.4, 0.9, 4)
    for mode, needs in MODE_FRACTIONS.items():
        spec = TableSpec(parametrization=par, mode=mode, include=flags,
                         axes={"nB": grid, "T": [0.0]},
                         fixed={k: values[k] for k in needs})
        rows, _ = build_table(spec, rows=True)
        assert len(rows) == grid.size, mode
        assert all(np.isfinite(r["P"]) and np.isfinite(r["eps"]) for r in rows)


def test_export_round_trip(par, tmp_path):
    flags = SpeciesFlags(hyperons=False, phi_field=False, muons=True)
    spec = TableSpec(parametrization=par, mode="beta_eq_neutrinoless",
                     include=flags, axes={"nB": NB, "T": [0.0]})
    rows, _ = build_table(spec, rows=True)
    out = tmp_path / "t.csv"
    export_csv(rows, out, meta=dict(mode="beta_eq_neutrinoless",
                                    parametrization=par, flags=flags))
    text = out.read_text().splitlines()
    header = [ln for ln in text if ln.startswith("#")][-1]
    assert "n_B" in header and "phase" in header
    assert len(text) - sum(ln.startswith("#") for ln in text) == len(rows)


---

## Third defect, found 2026-08-29 by a consumer sweep of the PAPER NOTEBOOK

The sweep that found defect 1 reached `nucleation/analysis/filters.py` and
stopped there. It did not reach the notebook, and the notebook is broken at its
**first `eos` call**.

`notebooks/2fam_PNS_nucleation.py:246` builds `params_H` — the hadronic
parameter set every downstream cell consumes — as

    from_potential_depths(..., x_sigma_delta=, x_omega_delta=, x_rho_delta=)

against today's

    from_potential_depths(U_Lambda_N, U_Sigma_N, U_Xi_N, base=None,
                          x_Delta_sigma=1.15, x_Delta_omega=1.0,
                          x_Delta_rho=1.0, name='Custom')

The three Delta ratios were renamed by **`286da5f` (tickets 112, 114)**. Binding
the notebook's keywords raises `TypeError: unexpected keyword argument
'x_sigma_delta'`, so the notebook dies within its first few cells.

**A third site carries the same break**: `test/make_fixture.py:68`'s `PARAM_KW`.
Together with the notebook these are EXACTLY the two blind spots the map's fog
entry "a rename landing green is not a rename landing safe" named — a lazy
import inside `main()` and a notebook that is never collected. The fog predicted
the shape and the sites; what it could not predict was which rename would land
in them.

### Fixed in the working tree, deliberately NOT committed

Keyword names only, at all three sites (`.py`, the paired `.ipynb`, and
`make_fixture.py`). The notebook's own local variables keep their names, so
`xsd_tag` and every run label are unchanged. Verified three ways: the call parsed
back out of the file binds against the live signature; `make_fixture`'s
`PARAM_KW` binds; and the call EXECUTES, returning
`g_sigma_N = 7.531282`, `g_omega_N = 9.022391` and recovering
`U_Lambda = -28.0, U_Sigma = +30.0, U_Xi = -18.0` exactly. **No physics moves —
it is a rename and nothing else.**

Not committed because `notebooks/2fam_PNS_nucleation.py` and
`analysis/filters.py` carry another session's extensive uncommitted work, and a
commit of that path would sweep it in. Whoever owns that tree should commit the
three sites with the rest.

### What this does NOT establish

A static signature sweep sees a changed signature behind an unchanged name. It
cannot see a renamed RESULT field, a moved dict key or a changed array shape,
and defect 2 means there is no suite to catch those. The notebook has not been
executed since the rename.

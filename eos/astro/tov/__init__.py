"""Stellar structure: the TOV sequence, the crust, and uniform rotation.

The input is one `EOSTable_for_TOV` (`eos.general.state`) — P, epsilon and
n_B in the repository's fm-based units, ordered by increasing density. That
dataclass is the whole contract between the models and this layer: a model
builds one, `astro/` integrates it, and neither imports the other
(CLAUDE.md section 1).

    from eos.astro.tov import compute_tov_sequence, find_mmax_precise

    seq = compute_tov_sequence(core, e_c_vec, add_crust_table="BPS",
                               n_transition=0.08)
    i, e_c, M_max = find_mmax_precise(seq)

`compute_tov_sequence` returns one row per central energy density; column 3
is R [km] and column 4 is M [M_sun], and `find_mmax_precise` gives the index
of the maximum-mass star, so `seq[:i + 1]` is the stable branch.

The crust tables ship in `data/`, so `add_crust_table="BPS"` works from a
fresh clone; `$EOS_CRUST_DIR` overrides the search path. A missing table
raises `MissingCrustData` rather than silently producing a star most of a
kilometre too small.

The rotating (RNS) backend is `eos.astro.tov.rotating`, imported directly:
it shells out to a compiled solver, which is a heavier dependency than a
`from ... import` line should carry.
"""
from eos.astro.tov.solver import (
    TOVResult, solve_tov_single, compute_tov_sequence,
    generate_ec_logspace, find_mmax_precise, truncate_to_stable_branch,
)
from eos.astro.tov.crust import (
    MissingCrustData, add_crust, load_crust_table, have_crust, crust_path,
)

__all__ = [
    "TOVResult", "solve_tov_single", "compute_tov_sequence",
    "generate_ec_logspace", "find_mmax_precise", "truncate_to_stable_branch",
    "MissingCrustData", "add_crust", "load_crust_table", "have_crust",
    "crust_path",
]

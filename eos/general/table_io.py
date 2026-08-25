"""
general/table_io.py
===================
Reading and writing generated EoS tables.

*Public API* (re-exported from both `eos.dd2` and `eos.mixed`): `save_table`,
`load_table`, `export_csv`.

Engine-neutral: nothing here knows about hadrons, quarks or mixed phases. A
table is a list of dicts — the long format that `eos.dd2.build_table(rows=True)`
and `eos.mixed.build_table` both return — plus metadata describing how it
was made. HDF5 is the working format: one dataset per column, metadata in file
attributes. That keeps large multi-axis grids compact and lets a Bayesian
pipeline memory-map a single column without reading the rest.

`export_csv` writes the same rows as plain text with a commented header, for
sharing with tools that do not read HDF5.

The metadata is deliberately complete enough to reproduce the table: the mode,
eta, every species flag, every parametrization field, the vMIT parameters, and
the phase boundaries found on each line. A table that cannot say how it was
generated is not much use months later.
"""
from dataclasses import asdict, is_dataclass

import numpy as np


def _flatten_meta(meta):
    """Metadata dict -> flat {str: scalar-or-array} that HDF5 attrs accept.

    Dataclasses (Parameters, SpeciesFlags, Parameters) are expanded field
    by field under a prefix; anything else is stringified as a last resort so a
    table is never rejected for carrying an unexpected note.
    """
    flat = {}
    for key, value in meta.items():
        if is_dataclass(value) and not isinstance(value, type):
            for k, v in asdict(value).items():
                if isinstance(v, (int, float, bool, str)):
                    flat[f"{key}.{k}"] = v
                elif isinstance(v, (list, tuple)) and not v:
                    continue                       # empty tuple: nothing to say
                else:
                    flat[f"{key}.{k}"] = str(v)
        elif isinstance(value, (int, float, bool, str)):
            flat[key] = value
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            flat[key] = arr if arr.dtype.kind in "fiub" else str(value)
        elif value is None:
            continue
        else:
            flat[key] = str(value)
    return flat


def _columns(rows):
    """Long-format rows -> {name: array}, union of all keys, nan where absent.

    Different modes and different eta values populate different columns (a
    trapped-neutrino run has mu_nue, a Maxwell run has no mu_eG), so the union is
    taken rather than assuming every row has the same keys.
    """
    names = []
    for row in rows:
        for k in row:
            if k not in names:
                names.append(k)
    cols = {}
    for name in names:
        values = [row.get(name, np.nan) for row in rows]
        if any(isinstance(v, str) for v in values):
            cols[name] = np.array([("" if v is np.nan else str(v))
                                   for v in values], dtype=object)
        else:
            cols[name] = np.asarray(values, dtype=float)
    return cols


def save_table(rows, path, meta=None, windows=None):
    """Write long-format `rows` to an HDF5 file at `path`.

    meta    : dict of anything describing the run — pass the Parameters,
              SpeciesFlags, Parameters, mode name and eta.
    windows : the {axis key: Window} dict `build_table` returns; the
              onset/offset densities are stored as a table of their own so the
              phase boundaries survive without re-solving.
    """
    import h5py

    cols = _columns(rows)
    with h5py.File(path, "w") as f:
        grp = f.create_group("table")
        for name, arr in cols.items():
            if arr.dtype == object:
                grp.create_dataset(name, data=np.array(arr, dtype="S32"))
            else:
                grp.create_dataset(name, data=arr, compression="gzip")
        grp.attrs["n_rows"] = len(rows)
        for key, value in _flatten_meta(meta or {}).items():
            f.attrs[key] = value
        if windows:
            keys = sorted(windows)
            wg = f.create_group("windows")
            wg.create_dataset("axis_key", data=np.array(keys, dtype=float))
            wg.create_dataset(
                "n_onset", data=np.array([windows[k].n_onset if windows[k]
                                          else np.nan for k in keys]))
            wg.create_dataset(
                "n_offset", data=np.array([windows[k].n_offset if windows[k]
                                           else np.nan for k in keys]))
    return path


def load_table(path):
    """Read a table written by `save_table`.

    Returns `(columns, meta, windows)` where `columns` is {name: array},
    `meta` is the flat metadata dict, and `windows` is a
    {axis key tuple: (n_onset, n_offset)} dict (empty if none were stored).
    Wrap `columns` in `pandas.DataFrame` if you want one.
    """
    import h5py

    with h5py.File(path, "r") as f:
        grp = f["table"]
        columns = {}
        for name in grp:
            arr = grp[name][()]
            if arr.dtype.kind == "S":
                arr = np.array([b.decode() for b in arr])
            columns[name] = arr
        meta = {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in f.attrs.items()}
        windows = {}
        if "windows" in f:
            wg = f["windows"]
            for key, on, off in zip(wg["axis_key"][()], wg["n_onset"][()],
                                    wg["n_offset"][()]):
                windows[tuple(np.atleast_1d(key))] = (float(on), float(off))
    return columns, meta, windows


def export_csv(rows, path, meta=None):
    """Write long-format `rows` as plain text with a commented header.

    Column order follows first appearance; missing values are written as nan.
    """
    cols = _columns(rows)
    names = list(cols)
    with open(path, "w") as f:
        for key, value in sorted(_flatten_meta(meta or {}).items()):
            f.write(f"# {key} = {value}\n")
        f.write("# " + "  ".join(names) + "\n")
        for i in range(len(rows)):
            out = []
            for name in names:
                v = cols[name][i]
                out.append(v if isinstance(v, str) else f"{float(v):.8e}")
            f.write("  ".join(out) + "\n")
    return path


# ----------------------------------------------------------- automatic names

_FLAG_TOKENS = {"hyperons": "hyp", "deltas": "del", "muons": "mu",
                "thermal_mesons": "mes", "thermal_neutrinos": "nu",
                "photons": "ph"}


def standard_name(model, mode, conditions, axes, species, leptons=True,
                  eta=None, ext="h5"):
    """The automatic file name for a generated table.

    Every choice that changes a number is in the name, so two tables in one
    folder cannot collide silently and a name alone says how it was made:

        dd2_fixed_YC_YC0.100_T0.0-30.0x4_nB0.1-1.2x64_mu+ph.h5
        vmit_beta_eq_neutrinoless_T0.0x1_nB0.1-1.5x32_ph_nolep.h5
        dd2vmit_fixed_YC_YC0.100_eta0.30_T0.0-30.0x4_nB0.1-1.2x64_mu+ph.h5

    Order: model, mode, the mode's fractions, eta if a composite engine, the
    thermal axis, the density axis, the sectors that are ON, and `nolep` only
    when leptons are off (their presence is the common case). `bare` is the
    literal token when no sector is on, rather than an empty gap that would
    make the name ambiguous.

    The complete metadata still goes into the file through `save_table(meta=)`;
    the name is for the human reading the folder months later.
    """
    parts = [model, mode]
    for key in ("Y_C", "Y_S", "Y_Le", "Y_Lmu"):
        if conditions.get(key) is not None:
            parts.append(f"{key.replace('_', '')}{conditions[key]:.3f}")
    if eta is not None:
        parts.append(f"eta{eta:.2f}")
    for key in ("T", "SnB"):
        if key in axes:
            parts.append(f"{key}{_span(axes[key])}")
    parts.append(f"nB{_span(axes['nB'])}")

    on = [token for flag, token in _FLAG_TOKENS.items() if species.get(flag)]
    parts.append("+".join(on) if on else "bare")
    if not leptons:
        parts.append("nolep")
    return "_".join(parts) + "." + ext


def _span(grid):
    """A grid as `lo-hix n` — or just `lo x1` when it is a single value."""
    grid = np.atleast_1d(np.asarray(grid, dtype=float))
    if grid.size == 1:
        return f"{grid[0]:.1f}x1"
    return f"{grid[0]:.1f}-{grid[-1]:.1f}x{grid.size}"


def table_path(model, name, root="output/tables"):
    """`output/tables/<model>/<name>` — CLAUDE.md section 11's per-model folder.

    The folder is created on demand, so a notebook that saves a table does not
    need a setup step before it.
    """
    import os

    folder = os.path.join(root, model)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)

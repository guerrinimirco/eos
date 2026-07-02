#!/usr/bin/env python
"""Download the LVK GW190425 posterior release and extract (m1, m2, Λ1, Λ2).

Pulls the official GWTC-2.1 posterior HDF5 from Zenodo (record 6513631) if it is
not already on disk, reads the preferred waveform run, and writes a plain
4-column table (masses in M_sun, dimensionless tidal deformabilities) with the
same column orientation as gw170817.dat — samples as rows, one parameter per
column.

    python fetch_gw190425.py

Robust to: missing h5py, network timeouts, and either HDF5 layout (the modern
pesummary compound array `run/posterior_samples`, or the older
`run/posterior/samples` + `run/posterior/parameter_names`).
"""
from pathlib import Path
import sys
import urllib.request
import urllib.error

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "data" / "samples"

# NOTE: the filename in the original spec ("GW190425_nocosmo.h5") 404s on Zenodo.
# The real GWTC-2.1 release file for GW190425 is named as below (record 6513631).
H5_NAME = "IGWN-GWTC2p1-v2-GW190425_081805_PEDataRelease_mixed_nocosmo.h5"
URL = f"https://zenodo.org/records/6513631/files/{H5_NAME}?download=1"

LOCAL_H5 = SAMPLES / "GW190425_nocosmo.h5"          # keep the short local name
OUT_TABLE = SAMPLES / "gw190425_extracted_table.txt"
PREFERRED_RUN = "C01:IMRPhenomPv2_NRTidal"
PARAMS = ("mass_1", "mass_2", "lambda_1", "lambda_2")
TIMEOUT = 120                                        # s, per network read


def download_if_missing(url=URL, dest=LOCAL_H5, timeout=TIMEOUT):
    """Fetch the HDF5 to `dest` unless already present.  Downloads to a .part
    file first so an interrupted transfer never leaves a truncated .h5 behind."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already present: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}\n        → {dest}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(part, "wb") as fh:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while chunk := r.read(1 << 20):          # 1 MiB chunks
                fh.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:6.1f} / {total/1e6:.1f} MB", end="", flush=True)
        print()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        part.unlink(missing_ok=True)
        raise SystemExit(f"ERROR: download failed ({e}). Check network / URL.") from e
    part.replace(dest)
    return dest


def _resolve_run_key(f, preferred=PREFERRED_RUN):
    """Preferred run if present, else any NRTidal (BNS) run with posterior data,
    else the first group that actually carries samples."""
    keys = list(f.keys())
    if preferred in keys and _has_posterior(f[preferred]):
        return preferred
    for k in keys:
        if "NRTidal" in k and _has_posterior(f[k]):
            return k
    for k in keys:
        if _has_posterior(f[k]):
            return k
    raise KeyError(f"no posterior group among {keys}")


def _has_posterior(grp):
    try:
        return "posterior_samples" in grp or (
            "posterior" in grp and "samples" in grp["posterior"])
    except (TypeError, AttributeError):
        return False                                 # not a group (e.g. a dataset)


def read_params(grp, params=PARAMS):
    """Return {param: 1D array} handling both HDF5 layouts."""
    import numpy as np
    if "posterior_samples" in grp:                   # pesummary compound array
        ds = grp["posterior_samples"]
        names = ds.dtype.names
        missing = [p for p in params if p not in names]
        if missing:
            raise KeyError(f"missing {missing}; available: {names}")
        return {p: np.asarray(ds[p], dtype=float) for p in params}
    # older layout: 2D samples + a parallel list of column names
    samples = np.asarray(grp["posterior"]["samples"], dtype=float)
    raw = grp["posterior"]["parameter_names"][:]
    names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in raw]
    idx = {n: i for i, n in enumerate(names)}
    missing = [p for p in params if p not in idx]
    if missing:
        raise KeyError(f"missing {missing}; available: {names}")
    return {p: samples[:, idx[p]] for p in params}


def main():
    try:
        import numpy as np
        import h5py
    except ImportError as e:
        raise SystemExit(f"ERROR: missing dependency ({e}). pip install h5py numpy") from e

    path = download_if_missing()
    with h5py.File(path, "r") as f:
        run = _resolve_run_key(f)
        if run != PREFERRED_RUN:
            print(f"preferred run '{PREFERRED_RUN}' not found — using '{run}'")
        cols = read_params(f[run])

    table = np.column_stack([cols[p] for p in PARAMS])   # same orientation as gw170817.dat
    np.savetxt(OUT_TABLE, table, fmt="%.6e", delimiter="\t",
               header="mass_1_msun\tmass_2_msun\tlambda_1\tlambda_2")
    print(f"wrote {OUT_TABLE}  ({table.shape[0]} samples × {table.shape[1]} cols) from run '{run}'")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Extract (m1, m2, Λ1, Λ2) for GW170817 from the official GWTC-1 PE release.

Mirrors fetch_gw190425.py so the two BNS events sit on equal footing: same
low-spin prior, source-frame masses, same output layout.

Source: LVK GWTC-1 parameter-estimation data release, LIGO DCC P1800370
(https://dcc.ligo.org/LIGO-P1800370/public), file `GW170817_GWTC-1.hdf5`
(Abbott et al. 2019, "GWTC-1", arXiv:1811.12907 / PRX 9, 031040).

The `IMRPhenomPv2NRT_lowSpin_posterior` dataset carries **detector-frame**
masses only (`m1_detector_frame_Msun`, ...), so we convert to source frame with
the known host (NGC 4993) redshift z≈0.0099 — a <1% correction here, applied for
consistency with the GW190425 source-frame table.

    python fetch_gw170817.py
"""
from pathlib import Path
import urllib.request, urllib.error

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "data" / "samples"
H5_URL = "https://dcc.ligo.org/public/0157/P1800370/005/GW170817_GWTC-1.hdf5"
LOCAL_H5 = SAMPLES / "GW170817_GWTC-1.hdf5"
OUT_TABLE = SAMPLES / "gw170817_extracted_table.txt"
RUN = "IMRPhenomPv2NRT_lowSpin_posterior"
Z_NGC4993 = 0.0099                                  # host redshift (Hjorth+ 2017)


def download_if_missing(url=H5_URL, dest=LOCAL_H5, timeout=120):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already present: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}\n        → {dest}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(part, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        part.unlink(missing_ok=True)
        raise SystemExit(f"ERROR: download failed ({e}).") from e
    part.replace(dest)
    return dest


def main():
    try:
        import numpy as np, h5py
    except ImportError as e:
        raise SystemExit(f"ERROR: missing dependency ({e}). pip install h5py numpy") from e

    with h5py.File(download_if_missing(), "r") as f:
        ds = f[RUN]
        m1 = np.asarray(ds["m1_detector_frame_Msun"], float) / (1 + Z_NGC4993)
        m2 = np.asarray(ds["m2_detector_frame_Msun"], float) / (1 + Z_NGC4993)
        l1 = np.asarray(ds["lambda1"], float)
        l2 = np.asarray(ds["lambda2"], float)

    table = np.column_stack([m1, m2, l1, l2])       # same layout as gw190425 table
    np.savetxt(OUT_TABLE, table, fmt="%.6e", delimiter="\t",
               header=f"m1_source_msun\tm2_source_msun\tlambda_1\tlambda_2  (run {RUN}, z={Z_NGC4993})")
    print(f"wrote {OUT_TABLE}  ({table.shape[0]} samples) from '{RUN}'")


if __name__ == "__main__":
    main()

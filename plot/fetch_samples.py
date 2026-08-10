"""Download the raw posterior samples the contour build needs.

These files are large (~160 MB total) and are NOT tracked in git. They are
inputs to `compute_contours.py`, which turns them into the small 68 % / 95 %
contour CSVs shipped inside the package at `eos/general/data/contours/`.
Nothing at runtime reads the raw samples: the library, the notebooks and the
figures all read the derived contours. You only need these to REBUILD a
contour, or to build a new one.

    python plot/fetch_samples.py            # fetch whatever is missing
    python plot/fetch_samples.py --check    # report only, download nothing
    python plot/fetch_samples.py J0030.txt  # one file

Every entry carries the sha256 of the copy the shipped contours were built
from. A download whose checksum differs is written to `<name>.mismatch` and
reported rather than installed, because a different posterior release is a
different physics input: silently swapping one in would change published
contours with no visible cause. Provenance for each file — paper, data
release, column layout, and which run was chosen and why — is in
`data/samples/SOURCES.md`.
"""
from __future__ import annotations

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "data" / "samples"
TIMEOUT = 180  # s per network read


class Sample:
    """One raw sample file: where it comes from and how to recognise it."""

    def __init__(self, name, size, sha256, record, key, header="", note="",
                 member=None):
        self.name = name          # filename under data/samples/
        self.size = size          # bytes, of the copy contours were built from
        self.sha256 = sha256
        self.record = record      # Zenodo record id, or a script name
        self.key = key            # file within that record
        self.header = header      # the `# col1 col2` line this repo prepends
        self.note = note
        self.member = member      # path inside the archive, if key is a tarball

    @property
    def url(self):
        return f"https://zenodo.org/records/{self.record}/files/{self.key}?download=1"


# Only the files too large to track. Everything else in data/samples/ —
# the extracted GW tables, the heavy-ion and chiral-EFT bands, SOURCES.md —
# is small and stays in git.
SAMPLES_INDEX = [
    Sample("J0030.txt", 20000319,
           "5bb01bfafa106a2249e04e3f424fa2318438b29cfe348546705bc4a2fc88e1f0",
           "3473466", "J0030_2spot_RM.txt",
           header="# R_km  M_sun",
           note="NICER PSR J0030+0451. UNRESOLVED: SOURCES.md describes the "
                "three-oval (ST+PST) run, but this copy's size matches the "
                "2-spot RM release and its posterior gives R = 13.27 km "
                "against the published 13.02 km. The Zenodo key below is "
                "therefore a best guess; confirm it before rebuilding this "
                "contour."),
    Sample("J0740.txt", 63857893,
           "fae7dea9b2272331c0307b698a1c07b9483a685ccf03c8ebc3c596417eed7f65",
           "4670689", "NICER+XMM-relative_J0740_RM.txt",
           header="# R_km  M_sun  weight",
           note="NICER+XMM PSR J0740+6620. Weighted samples: the third "
                "column must be used."),
    Sample("HESS.txt", 24998444,
           "c9c111f35da1d64d6fe34d622c91022d5384988e5a985e41e59b6ec3f48e30f1",
           "8232233", "xray_only_carbatm.txt",
           header="# R_km  M_sun",
           note="HESS J1731-347, X-ray-only carbon-atmosphere fit. Do not "
                "substitute full_priors_carbatm_corr.txt."),
    Sample("J0614.dat", 10047398,
           "93412719fcb978b96e0a4a1cc5edd09dfc503c14ea090704ae92b0b0e88d151c",
           "17380576", "Headline_Contours_and_Samples.tar.gz",
           header="# M_sun  R_km   [note: mass first, then radius]",
           member=None,
           note="NICER PSR J0614-3329, headline M-R samples from inside the "
                "archive. Columns are mass first, radius second."),
]

# The two gravitational-wave files already have dedicated fetchers, because
# they need a run selected out of a multi-run HDF5 rather than a plain copy.
DELEGATED = {
    "GW170817_GWTC-1.hdf5": "plot/fetch_gw170817.py",
    "GW190425_nocosmo.h5": "plot/fetch_gw190425.py",
}


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, timeout=TIMEOUT):
    """Download to a .part file, then move into place."""
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with open(part, "wb") as fh:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if total:
                    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
        print()
    part.replace(dest)
    return dest


def extract_member(archive, member, dest):
    """Pull one file out of a tarball, choosing it by size when unnamed."""
    with tarfile.open(archive) as tf:
        entries = [m for m in tf.getmembers() if m.isfile()]
        if member:
            chosen = next((m for m in entries if m.name.endswith(member)), None)
        else:
            chosen = max(entries, key=lambda m: m.size)
        if chosen is None:
            raise SystemExit(
                f"{archive.name}: no member matching {member!r}; "
                f"contains {[m.name for m in entries][:10]}")
        with tf.extractfile(chosen) as src, open(dest, "wb") as out:
            out.write(src.read())
    return dest


def fetch(sample, check_only=False):
    dest = SAMPLES / sample.name
    if dest.exists():
        actual = dest.stat().st_size
        tag = "ok" if actual == sample.size else f"SIZE {actual} != {sample.size}"
        print(f"  present  {sample.name:24s} {actual / 1e6:7.1f} MB  {tag}")
        return True
    if check_only:
        print(f"  MISSING  {sample.name:24s} "
              f"expect {sample.size / 1e6:.1f} MB  (zenodo {sample.record})")
        if sample.note:
            print(f"           {sample.note}")
        return False

    print(f"  fetching {sample.name} from zenodo {sample.record}")
    tmp = SAMPLES / (sample.key.split("/")[-1])
    download(sample.url, tmp)
    if sample.key.endswith((".tar.gz", ".tgz")):
        extract_member(tmp, sample.member, dest)
        tmp.unlink()
    elif tmp != dest:
        tmp.replace(dest)

    # This repo prepends a uniform `# col1 col2 ...` token header to every
    # sample file so numpy.loadtxt callers can see the column layout; the
    # upstream releases have no header. Re-apply it, or the checksum below
    # fails on a file whose data is byte-identical.
    if sample.header:
        body = dest.read_bytes()
        if not body.startswith(b"#"):
            dest.write_bytes(sample.header.encode() + b"\n" + body)

    actual_size = dest.stat().st_size
    actual_sha = sha256_of(dest)
    if actual_sha != sample.sha256:
        bad = dest.with_suffix(dest.suffix + ".mismatch")
        dest.replace(bad)
        print(f"  MISMATCH {sample.name}\n"
              f"           got      {actual_size} bytes, sha256 {actual_sha}\n"
              f"           expected {sample.size} bytes, sha256 {sample.sha256}\n"
              f"           Kept as {bad.name} rather than installing it: a "
              f"different posterior release is a different physics input and "
              f"would change the contours with no visible cause. "
              f"See data/samples/SOURCES.md.")
        if sample.note:
            print(f"           {sample.note}")
        return False
    print(f"  ok       {sample.name}  {actual_size / 1e6:.1f} MB  sha256 verified")
    return True


def main(argv):
    check_only = "--check" in argv
    wanted = [a for a in argv if not a.startswith("--")]
    index = [s for s in SAMPLES_INDEX if not wanted or s.name in wanted]

    print(f"raw posterior samples in {SAMPLES}")
    missing = [s for s in index if not fetch(s, check_only=check_only)]

    for name, script in DELEGATED.items():
        if not wanted or name in wanted:
            if (SAMPLES / name).exists():
                print(f"  present  {name:24s} "
                      f"{(SAMPLES / name).stat().st_size / 1e6:7.1f} MB  ok")
            else:
                print(f"  MISSING  {name:24s} run `python {script}`")
                missing.append(name)

    if missing:
        print(f"\n{len(missing)} file(s) missing. They are only needed to "
              f"rebuild contours with plot/compute_contours.py; the shipped "
              f"contours in eos/general/data/contours/ work without them.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

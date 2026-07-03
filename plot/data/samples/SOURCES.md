# Observational & nuclear input samples — provenance & column audit

Every raw file in this directory, its authoritative source (paper + data
release), the run/release we deliberately picked and **why**, and its verified
column layout. Processed 68 % / 95 % contours are built from these by
`../../compute_contours.py` (and `../../fetch_gw190425.py` for GW190425).

**Uniform format:** every file's first line is a `# col1 col2 …` token header
naming columns and units (space/tab-separated data below; `#` lines are skipped
by `numpy.loadtxt`). Column identities were re-verified from value ranges on
2026-07-03 — see the "verified" note per entry.

---

## Mass–Radius posteriors (NICER / X-ray)

### `J0030.txt` — PSR J0030+0451 (NICER)
- **Paper:** Miller, Lamb, Dittmann et al. **2019**, *ApJL* **887**, L24 — [arXiv:1912.05705](https://arxiv.org/abs/1912.05705)
- **Data:** Illinois–Maryland MCMC release, Zenodo **10.5281/zenodo.3473466** — <https://zenodo.org/record/3473466>
- **Columns:** `R_km  M_sun`. Verified: R∈[9.3,19.7], M∈[1.0,2.37]; three-oval model, M≈1.44, R≈13.0 km.

### `J0740.txt` — PSR J0740+6620 (NICER + XMM-Newton)
- **Paper:** Miller et al. **2021**, *ApJL* **918**, L28 — [arXiv:2105.06979](https://arxiv.org/abs/2105.06979)
- **Data:** Illinois–Maryland posterior release, Zenodo **10.5281/zenodo.4670689** — <https://zenodo.org/record/4670689>
- **Columns:** `R_km  M_sun  weight`. Verified: R∈[9.2,26.6], M∈[1.59,2.4], w∈[1,312]; M≈2.08, R≈13.7 km. **Weighted** samples — the `weight` column must be used (it is, via the MR_SOURCES spec).

### `HESS.txt` — HESS J1731-347 central compact object
- **Paper:** Doroshenko, Suleimanov, Pühlhofer, Santangelo **2022**, *Nature Astronomy* **6**, 1444 — [arXiv:2211.07485](https://arxiv.org/abs/2211.07485)
- **Data:** Zenodo **10.5281/zenodo.8232233**. This file = `xray_only_carbatm.txt` (X-ray-only, carbon-atmosphere fit) → R=10.4⁺⁰·⁸⁶₋₀.₇₈, M=0.77⁺⁰·²⁰₋₀.₁₇, the paper's headline posterior (tail to ~8 km).
- **Columns:** `R_km  M_sun`. Verified: R∈[8.2,14.5], M∈[0.32,1.38]; lightest known compact object.
- **Why this file:** the X-ray-only fit is the model-independent headline result. Do **not** use `HESS_full_priors_carbatm_corr.txt` (kept here, unused): adding χEFT/NICER/GW EoS priors pulls R up ~0.8 km and halves the width — that was a previous *wrong* HESS.txt.

### `J0614.dat` — PSR J0614-3329 (NICER)
- **Paper:** Mauviard, Watts et al. **2025**, "A NICER view of the 1.4 M⊙ edge-on pulsar PSR J0614-3329", *ApJ* — [arXiv:2506.14883](https://arxiv.org/abs/2506.14883), [doi:10.3847/1538-4357/ae145d](https://iopscience.iop.org/article/10.3847/1538-4357/ae145d)
- **Data (used):** Zenodo **10.5281/zenodo.17380576** — <https://zenodo.org/records/17380576> ("Data and Reproduction package…"; headline M–R samples are inside `Headline_Contours_and_Samples.tar.gz`). Supersedes the initial release 10.5281/zenodo.15603406.
- **Columns:** `M_sun  R_km` — **mass first, radius second** (opposite order to the NICER files above). Verified: M median 1.445 (68 % 1.38–1.51), R median 10.29 (68 % 9.43–11.30), full R tail [6.9,14.8].
- **Which run (resolved):** an **exact match** to the paper's *headline NS* result (R=10.29⁺¹·⁰¹₋₀.₈₆, M=1.44⁺⁰·⁰⁶₋₀.₀₇), unimodal. It is **not** a strange-quark-star run — the small-R values are just the lower tail. (A separate SQS interpretation exists in [arXiv:2508.02652](https://arxiv.org/abs/2508.02652); that is *not* this file.)

---

## Gravitational-wave posteriors (LVK)

### `GW170817_GWTC-1.hdf5` + `gw170817_extracted_table.txt` — GW170817 BNS (USED)
- **Paper:** Abbott et al. **2019**, "GWTC-1", *PRX* **9**, 031040 — [arXiv:1811.12907](https://arxiv.org/abs/1811.12907).
- **Data:** LVK GWTC-1 PE release, LIGO DCC **P1800370** — <https://dcc.ligo.org/LIGO-P1800370/public> (file `GW170817_GWTC-1.hdf5`, [doi:10.7935/82H3-HH23](https://doi.org/10.7935/82H3-HH23)). Downloaded + extracted by `../../fetch_gw170817.py`.
- **Which run + why:** dataset `IMRPhenomPv2NRT_lowSpin_posterior` — **low-spin** prior, matching GW190425 for a like-for-like comparison. (`highSpin` also present.)
- **Extracted table columns:** `m1_source_msun  m2_source_msun  lambda_1  lambda_2` (tab-sep). Verified: Mc_src=1.186 (exactly the published source-frame chirp mass), m1=1.465, m2=1.268.
- **Frame:** GWTC-1 gives detector-frame masses only, converted to **source-frame** with host NGC 4993 redshift z=0.0099 (<1 % correction), applied for symmetry with GW190425. Λ is frame-independent.

### `gw170817.dat` — GW170817 (older PRX-properties release, kept for reference)
- **Paper:** Abbott et al. **2019**, "Properties of the binary neutron star merger GW170817", *PRX* **9**, 011001 — [arXiv:1805.11579](https://arxiv.org/abs/1805.11579); LVK DCC **P1800061** — <https://dcc.ligo.org/LIGO-P1800061/public>.
- **Columns (10, header-less originally):** `c0  dL_Mpc  m1_sun  m2_sun  Λ1  Λ2  a1  a2  c8  c9` (cols 2–5 = m1,m2,Λ1,Λ2; low-spin, detector-frame). **No longer used by the pipeline** — superseded by the GWTC-1 extract above; retained as a cross-check.

### `GW170817_MR.txt` — GW170817 with tidal + radii (parametrized EoS)
- **Paper:** Abbott et al. **2018**, "GW170817: Measurements of neutron star radii and equation of state", *PRL* **121**, 161101 — [arXiv:1805.11581](https://arxiv.org/abs/1805.11581)
- **Data:** LVK release, LIGO DCC **P1800115** — <https://dcc.ligo.org/LIGO-P1800115/public>
- **Columns:** `M1_sun  M2_sun  Λ1  Λ2  R1_km  R2_km`. Verified: M1∈[1.36,1.77]>M2∈[1.06,1.36], Λ1∈[0.1,1.3e3], R∈[5.3,15] km. Small-R tail is the parametrized-EoS (Mmax≥1.97 M⊙) prior. Two components share one EoS → same R(M) band downstream.

### `GW190425_nocosmo.h5` + `gw190425_extracted_table.txt` — GW190425 BNS
- **Paper:** Abbott et al. **2020**, "GW190425: Observation of a Compact Binary Coalescence with Total Mass ~3.4 M⊙", *ApJL* **892**, L3 — [arXiv:2001.01761](https://arxiv.org/abs/2001.01761)
- **Data:** GWTC-2.1 posterior release, Zenodo record **6513631** — <https://zenodo.org/records/6513631> (file `IGWN-GWTC2p1-v2-GW190425_081805_PEDataRelease_mixed_nocosmo.h5`). Downloaded by `../../fetch_gw190425.py`.
- **Extracted table columns:** `m1_source_msun  m2_source_msun  lambda_1  lambda_2` (tab-separated).
- **Which run (resolved) + why:** the h5 contains **two** runs, `C01:IMRPhenomPv2_NRTidal:LowSpin` and `:HighSpin`. We use **LowSpin** — same prior as GW170817 for a like-for-like comparison; the earlier default silently grabbed HighSpin (broad, extreme mass ratios), now fixed. Verified LowSpin source-frame: m1=1.75⁺⁰·¹⁷₋₀.₀₉, m2=1.56⁺⁰·⁰⁸₋₀.₁₃, Mc=1.436⁺⁰·⁰²₋₀.₀₂ ≈ literature 1.44±0.02.
- **Frame:** **source-frame** masses (`mass_1_source`), redshift-corrected (z≈0.03 → ~3 % below detector-frame). Λ is dimensionless (frame-independent). Λ̃ formula matches the LVK-native `lambda_tilde` to 1e-12.

---

## Heavy-ion & chiral-EFT bands (density vs energy/pressure)

3-column filled bands. Also distributed via the **nucleardatapy** toolkit
(<https://jeromemargueron.github.io/nucleardatapy/>) as HIC inferences
`2002-DLL`, `2016-FOPI`, `2016-ASYEOS`.

### `DLL_2002_PSM.txt` — flow constraint, pressure of symmetric matter
- Danielewicz, Lacey, Lynch **2002**, *Science* **298**, 1592 — [arXiv:nucl-th/0208016](https://arxiv.org/abs/nucl-th/0208016).
- **Columns:** `rho_fm3  P_low_MeVfm3  P_up_MeVfm3`. Verified: ρ∈[0.32,0.74], P∈[7,207].

### `FOPI_2016_eSM.txt` / `FOPI_2016_PSM.txt` — FOPI flow, symmetric matter (E/A and P)
- Le Fèvre, Leifels, Reisdorf et al. **2016**, *Nucl. Phys. A* **945**, 112 (IQMD analysis of FOPI flow).
- **Columns:** `eSM` = `rho_fm3  EoverA_low_MeV  EoverA_up_MeV` (E/A reaches −16 MeV at n₀, i.e. symmetric-matter binding); `PSM` = `rho_fm3  P_low_MeVfm3  P_up_MeVfm3`.

### `ASYEOS_2016_Esym.txt` — symmetry energy at supra-saturation density
- Russotto et al. (ASY-EOS) **2016**, *Phys. Rev. C* **94**, 034608 — [arXiv:1608.04332](https://arxiv.org/abs/1608.04332).
- **Columns:** `rho_fm3  Esym_low_MeV  Esym_up_MeV`. Verified: ρ∈[0,0.32], Esym∈[0,60].

### `chiral_eft.txt` — chiral-EFT pure-neutron-matter energy band
- Hebeler, Lattimer, Pethick, Schwenk **2013**, *ApJ* **773**, 11 — [arXiv:1303.4662](https://arxiv.org/abs/1303.4662) (χEFT PNM E/A band; also compiled in Huth et al. 2021).
- **Provenance:** privately supplied file (no direct data-release DOI). **Cross-checked** against the database version below and confirmed to be this 2013 calculation.
- **Columns:** `rho_fm3  EoverA_low_MeV  EoverA_up_MeV`. Verified: ρ∈[0.05,0.16], E∈[6.8,18] MeV.
- **Database version (for citation / swap-in):** the **nucleardatapy** toolkit exposes this as `nda.setupMicro(model='2013-MBPT-NM')` (`nm_e2a_int_low/up`) — matches this file closely (upper edge near-identical; lower edge slightly wider). Newer χEFT alternatives in the same toolkit: `2016-MBPT-AM` (Drischler 2016), `2020-MBPT-AM` (Drischler GP-B 2020), `2024-MBPT-AM-*`, plus QMC/AFDMC series. Toolkit paper: [EPJA (2025)](https://link.springer.com/article/10.1140/epja/s10050-025-01760-w) / [arXiv:2506.20434](https://arxiv.org/abs/2506.20434). `../../compute_contours.py::fetch_nucleardatapy_bands()` already pulls a χEFT band from this API.

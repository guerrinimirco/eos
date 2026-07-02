# Observational & nuclear input samples — provenance

References and download locations for every raw file in this directory.
Processed 68% / 95% contours are built from these by `../../compute_contours.py`
(and `../../fetch_gw190425.py` for GW190425).

`⚠ confirm` marks a file whose *exact* data release I could not verify from the
file alone — the scientific reference is right, but double-check you grabbed the
model/run you intend.

---

## Mass–Radius posteriors (NICER / X-ray)

### `J0030.txt` — PSR J0030+0451 (NICER)
- Miller, Lamb, Dittmann et al. **2019**, *ApJL* **887**, L24 — arXiv:1912.05705
- Illinois–Maryland MCMC samples, Zenodo DOI **10.5281/zenodo.3473466**
  <https://zenodo.org/record/3473466>
- Columns: `R_km  M_sun` (three-oval model; M ≈ 1.44, R ≈ 13.0 km)

### `J0740.txt` — PSR J0740+6620 (NICER + XMM-Newton)
- Miller et al. **2021**, *ApJL* **918**, L28 — arXiv:2105.06979
- Illinois–Maryland posterior release, Zenodo <https://zenodo.org/record/4670689>
- Columns: `R_km  M_sun  weight` (±10% calibration; M ≈ 2.08, R ≈ 13.7 km)

### `HESS.txt` — HESS J1731-347 central compact object
- Doroshenko, Suleimanov, Pühlhofer, Santangelo **2022**, *Nature Astronomy* **6**, 1444
  — arXiv:2211.07485
- Lightest known compact object (M ≈ 0.77, R ≈ 10.4 km)
- Columns: `R_km  M_sun`

### `J0614.dat` — PSR J0614-3329 (NICER)  ⚠ confirm release
- Mauviard, Watts et al. **2025**, "A NICER view of the 1.4 M⊙ edge-on pulsar
  PSR J0614-3329", *ApJ* — arXiv:2506.14883 (R ≈ 10.3, M ≈ 1.44).
- Strange-quark-star interpretation: arXiv:2508.02652.
- Header-less; columns: `M_sun  R_km`. The small radii in this file (~7–9 km)
  suggest a specific compact/quark-star model run — verify which posterior it is.

---

## Gravitational-wave posteriors (LVK)

### `gw170817.dat` — GW170817 BNS  ⚠ confirm release
- Abbott et al. **2019**, "Properties of the binary neutron star merger GW170817",
  *PRX* **9**, 011001 — arXiv:1805.11579; LVK data release (GWOSC / DCC P1800061).
- Header-less, 10 columns; the ones used here (verified by value ranges):
  `col2 = m1, col3 = m2, col4 = Λ1, col5 = Λ2` (M_sun / dimensionless).

### `GW170817_MR.txt` — GW170817 with tidal + radii  ⚠ confirm release
- Likely from Abbott et al. **2018**, "GW170817: Measurements of neutron star radii
  and equation of state", *PRL* **121**, 161101 — arXiv:1805.11581.
- Header-less, 6 columns: `M1  M2  Λ1  Λ2  R1  R2` (M_sun / dimensionless / km).

### `GW190425_nocosmo.h5` — GW190425 BNS (downloaded by `fetch_gw190425.py`)
- Abbott et al. **2020**, "GW190425: Observation of a Compact Binary Coalescence
  with Total Mass ~3.4 M⊙", *ApJL* **892**, L3 — arXiv:2001.01761.
- GWTC-2.1 posterior data release, Zenodo record **6513631**
  <https://zenodo.org/records/6513631>
  (file `IGWN-GWTC2p1-v2-GW190425_081805_PEDataRelease_mixed_nocosmo.h5`).
- Extracted to `gw190425_extracted_table.txt`: `mass_1  mass_2  lambda_1  lambda_2`
  (run `C01:IMRPhenomPv2_NRTidal:HighSpin`; a `:LowSpin` run is also available).

---

## Heavy-ion & chiral-EFT bands (density vs energy/pressure)

All are 3-column `rho_fm3  lower  upper` filled bands.

### `DLL_2002_PSM.txt` — flow constraint, pressure of symmetric matter
- Danielewicz, Lacey, Lynch **2002**, *Science* **298**, 1592.

### `FOPI_2016_eSM.txt` / `FOPI_2016_PSM.txt` — FOPI flow, symmetric matter (E/A and P)
- Le Fèvre, Leifels, Reisdorf et al. **2016**, *Nucl. Phys. A* **945**, 112 (IQMD
  analysis of FOPI flow data).

### `ASYEOS_2016_Esym.txt` — symmetry energy at supra-saturation density
- Russotto et al. (ASY-EOS collaboration) **2016**, *Phys. Rev. C* **94**, 034608
  — arXiv:1608.04332.

> The three HIC bands above are also distributed via the **nucleardatapy** toolkit
> as HIC inferences `2002-DLL`, `2016-FOPI`, `2016-ASYEOS`
> (<https://jeromemargueron.github.io/nucleardatapy/>).

### `chiral_eft.txt` — chiral-EFT pure-neutron-matter energy band  ⚠ confirm reference
- χEFT PNM E/A band (`rho, E_low, E_up`). Typical sources: Hebeler et al. 2013 /
  Drischler et al. 2016 / Huth et al. 2021 — confirm which calculation this is.

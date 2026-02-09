# Nuclear Equation of State (EOS) Library

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)


A Python library for computing finite-temperature Equations of State (EOS) for dense nuclear and quark matter. 
Designed for astrophysical applications (including neutron star structure, core-collapse supernovae and compact binary mergers) and heavy-ion collisions.

This is a recent (2026) Python rewrite of some of the code I wrote in Mathematica and Python during my PhD (2022-2026). It is mainly for personal use, but if you need help to use it, please contact me!

**Author:** Mirco Guerrini (University of Ferrara)

**Contact:** mirco.guerrini@unife.it

**Publications:** [INSPIRE-HEP Profile](https://inspirehep.net/authors/2775420)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models Overview](#models-overview)
- [Package Structure](#package-structure)
- [Usage Examples](#usage-examples)
  - [SFHo Hadronic EOS](#1-sfho-hadronic-eos)
  - [Zhao-Lattimer Nucleonic EOS](#2-zhao-lattimer-nucleonic-eos)
  - [AlphaBag Quark EOS](#3-alphabag-quark-eos)
  - [vMIT Quark EOS](#4-vmit-quark-eos)
  - [ZL+vMIT Hybrid EOS](#5-zlvmit-hybrid-eos)
  - [ABPR CFL Quark EOS](#6-abpr-cfl-quark-eos)
  - [TOV Solver](#7-tov-neutron-star-structure)
- [Equilibrium Modes](#equilibrium-modes)
- [Output Format](#output-format)
- [Physical Conventions](#physical-conventions)
- [References](#references)
- [Future Development](#future-development)
- [Collaborators](#collaborators)

---

## Features

- **Multiple EOS Models**: Hadronic (SFHo, ZL), quark (AlphaBag, vMIT, ABPR), and hybrid constructions
- **Finite Temperature**: Full thermal treatment with Fermi-Dirac/Bose-Einstein integrals
- **Multiple Equilibrium Modes**: Beta equilibrium, fixed composition, trapped neutrinos
- **Phase Transitions**: Gibbs, Maxwell, and intermediate constructions
- **Neutron Star Applications**: Built-in TOV solver with crust matching and baryonic mass


---

## Installation

!pip install git+https://github.com/guerrinimirco/eos.git --quiet


### Dependencies

- Python >= 3.9
- NumPy
- SciPy
- Matplotlib
- Numba

All dependencies are automatically installed via pip.

---

## Quick Start

```python
import numpy as np
from eos.sfho.compute_tables import TableSettings, compute_table

# Generate a simple beta-equilibrium EOS table
settings = TableSettings(
    parametrization='sfho',
    particle_content='nucleons',
    equilibrium='beta_eq',
    n_B_values=np.linspace(0.1, 8, 100) * 0.16,  # fm^-3
    T_values=[0.1, 10.0, 30.0],                   # MeV
    save_to_file=True
)

results = compute_table(settings)
```

---

## Models Overview

| Model | Type | Description | Particles |
|-------|------|-------------|-----------|
| **SFHo** | Hadronic | Relativistic Mean Field with σ-ω-ρ-φ mesons | N, Y, Δ |
| **ZL** | Hadronic | Zhao-Lattimer nucleonic model | n, p |
| **AlphaBag** | Quark | MIT bag model with α_s QCD corrections | u, d, s |
| **vMIT** | Quark | Vector-MIT bag model with repulsive interactions | u, d, s |
| **ABPR** | Quark | CFL quark matter at T=0 (analytical) | u, d, s |
| **ZLvMIT** | Hybrid | ZL hadronic + vMIT quark mixed phase | Mixed |
| **SFHo+AlphaBag** | Hybrid | SFHo hadronic + AlphaBag quark mixed phase | Mixed |

---

## Package Structure

```
eos/
├── pyproject.toml          # Package configuration
├── setup.py                # Installation script
├── README.md               # This file
│
├── eos/                    # Main package
│   ├── __init__.py         # Package initialization, REPO_ROOT path
│   │
│   ├── general/            # Shared infrastructure
│   │   ├── physics_constants.py    # Physical constants (PDG values)
│   │   ├── particles.py            # Particle definitions
│   │   ├── fermi_integrals.py      # Fermi-Dirac integrals
│   │   ├── bose_integrals.py       # Bose-Einstein integrals
│   │   ├── thermodynamics_leptons.py  # e, μ, ν thermodynamics
│   │   └── plotting_info.py        # Plotting utilities
│   │
│   ├── sfho/               # SFHo RMF hadronic model
│   │   ├── parameters.py           # Model parametrizations
│   │   ├── thermodynamics_hadrons.py  # Baryon thermodynamics
│   │   ├── eos.py                  # EOS solvers
│   │   ├── compute_tables.py       # Table generation script
│   │   ├── nuclear_saturation_properties.py
│   │   └── compare_with_compose.py # Validation tools
│   │
│   ├── zl/                 # Zhao-Lattimer nucleonic model
│   │   ├── parameters.py           # Model parameters
│   │   ├── thermodynamics_nucleons.py  # Nucleon thermodynamics
│   │   ├── eos.py                  # EOS solvers
│   │   └── compute_tables.py       # Table generation script
│   │
│   ├── alphabag/           # AlphaBag quark model
│   │   ├── parameters.py           # Bag constant, α_s, masses
│   │   ├── thermodynamics_quarks.py   # Quark thermodynamics
│   │   ├── eos.py                  # EOS solvers (unpaired, CFL)
│   │   └── compute_tables.py       # Table generation script
│   │
│   ├── vmit/               # Vector-MIT bag model
│   │   ├── parameters.py           # Bag constant, vector coupling
│   │   ├── thermodynamics_quarks.py   # Quark thermodynamics
│   │   ├── eos.py                  # EOS solvers
│   │   └── compute_tables.py       # Table generation script
│   │
│   ├── abpr/               # ABPR analytical CFL model (T=0)
│   │   └── eos.py                  # Analytical P(μ), ε(μ), n_B(μ)
│   │
│   ├── zlvmit/             # ZL+vMIT hybrid model
│   │   ├── mixed_phase_eos.py      # Phase transition solvers
│   │   ├── hybrid_table_generator.py  # Configuration & runner
│   │   ├── trapped_solvers.py      # Trapped neutrino mode
│   │   ├── isentropic.py           # Isentropic trajectories
│   │   ├── table_reader.py         # Table I/O utilities
│   │   └── plot_results.py         # Visualization tools
│   │
│   ├── sfhoalphabag/       # SFHo+AlphaBag hybrid model
│   │   ├── mixed_phase_eos.py      # Phase transition solvers
│   │   └── hybrid_table_generator.py  # Configuration & runner
│   │
│   └── tov/                # Neutron star structure
│       └── solver.py               # TOV equation solver
│
├── notebooks/              # Jupyter notebooks
│   └── ZLvMIT_hybrid.ipynb # Interactive hybrid EOS exploration
│
└── output/                 # Generated EOS tables
```

---

## Usage Examples

### 1. SFHo Hadronic EOS

The SFHo model is a Relativistic Mean Field (RMF) model with σ-ω-ρ-φ meson exchange.

```python
from eos.sfho.compute_tables import TableSettings, compute_table
import numpy as np

settings = TableSettings(
    # Model selection
    parametrization='sfhoy',           # 'sfho', 'sfhoy', 'sfhoy_star', '2fam_phi', '2fam'
    particle_content='nucleons_hyperons',  # 'nucleons', 'nucleons_hyperons', 'nucleons_hyperons_deltas'

    # Equilibrium mode
    equilibrium='beta_eq',             # 'beta_eq', 'fixed_yc', 'fixed_yc_ys', 'trapped_neutrinos',
                                       # 'isentropic_beta_eq', 'isentropic_trapped'

    # Density and temperature grid
    n_B_values=np.linspace(0.1, 10, 300) * 0.1583,  # fm^-3
    T_values=[0.1, 10.0, 30.0, 50.0],               # MeV
    S_values=[1.0, 2.0],                            # For isentropic modes (entropy per baryon)

    # For fixed_yc or trapped modes:
    Y_C_values=np.arange(0.0, 0.55, 0.05),  # Charge fraction
    Y_L_values=np.arange(0.1, 0.45, 0.05),  # Lepton fraction

    # Physics options
    include_photons=True,
    include_electrons=True,
    include_thermal_neutrinos=False,
    include_pseudoscalar_mesons=False,

    # Output
    save_to_file=True,
    output_filename="sfhoy_beta_eq.dat"
)

results = compute_table(settings)
```

**Available Parametrizations:**

| Name | Description |
|------|-------------|
| `sfho` | Standard SFHo (nucleons only) |
| `sfhoy` | SFHo with hyperon couplings (Fortin et al.) |
| `sfhoy_star` | Alternative hyperon couplings |
| `2fam_phi` | Two-families model with φ meson |
| `2fam` | Two-families model |

**Custom Parametrization:**

```python
from eos.sfho.parameters import create_custom_parametrization

my_params = create_custom_parametrization(
    U_Lambda_N=-28.0, U_Sigma_N=+30.0, U_Xi_N=-18.0,
    name="MyCustom"
)
settings = TableSettings(
    custom_params=my_params,
    particle_content='nucleons_hyperons'
)
```

### 2. Zhao-Lattimer Nucleonic EOS

A phenomenological nucleonic model based on Zhao & Lattimer (2020).

```python
from eos.zl.compute_tables import ZLTableSettings, compute_zl_table
import numpy as np

settings = ZLTableSettings(
    equilibrium='beta_eq',             # 'beta_eq', 'fixed_yc', 'trapped_neutrinos'
    n_B_values=np.linspace(0.1, 12, 300) * 0.16,
    T_values=[0.1, 10.0, 50.0, 100.0],

    # For fixed_yc mode
    Y_C_values=[0.1, 0.2, 0.3, 0.4, 0.5],

    include_photons=True,
    include_leptons=True,  # Include electrons for fixed_yc mode
    save_to_file=True
)

results = compute_zl_table(settings)
```

### 3. AlphaBag Quark EOS

MIT bag model with perturbative QCD corrections. Supports unpaired and CFL phases.

```python
from eos.alphabag.compute_tables import AlphaBagTableSettings, compute_alphabag_table
import numpy as np

# Unpaired quark matter
settings = AlphaBagTableSettings(
    phase='unpaired',                  # 'unpaired' or 'cfl'
    equilibrium='beta_eq',             # 'beta_eq', 'fixed_yc', 'fixed_yc_ys'

    # Model parameters
    alpha=0.3,                         # QCD coupling α_s
    B4=165.0,                          # Bag constant B^(1/4) in MeV
    m_s=150.0,                         # Strange quark mass in MeV

    n_B_values=np.linspace(0.1, 12, 300) * 0.16,
    T_values=[0.1, 10.0, 30.0, 50.0],

    Y_C_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # For fixed_yc
    Y_S_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # For fixed_yc_ys

    include_photons=True,
    include_gluons=True,
    include_thermal_neutrinos=True,
    save_to_file=True
)

results = compute_alphabag_table(settings)
```

**CFL Phase:**

```python
cfl_settings = AlphaBagTableSettings(
    phase='cfl',
    Delta0_values=[80.0, 100.0, 150.0],  # Pairing gap in MeV
    B4=165.0,
    m_s=100.0,
    n_B_values=np.linspace(0.1, 12, 300) * 0.16,
    T_values=[0.1, 10.0, 50.0],
    save_to_file=True
)

cfl_results = compute_alphabag_table(cfl_settings)
```

### 4. vMIT Quark EOS

Vector-MIT bag model with repulsive vector interactions.

```python
from eos.vmit.compute_tables import vMITTableSettings, compute_vmit_table
import numpy as np

settings = vMITTableSettings(
    equilibrium='beta_eq',
    B4=165.0,                          # Bag constant B^(1/4) in MeV
    a=0.2,                             # Vector coupling in fm²
    m_s=150.0,                         # Strange quark mass in MeV

    n_B_values=np.linspace(0.1, 12, 300) * 0.16,
    T_values=[0.1, 10.0, 30.0],
    save_to_file=True
)

results = compute_vmit_table(settings)
```

### 5. ZL+vMIT Hybrid EOS

Hadron-quark phase transition between fully local and fully global electric charge neutrality (Gibbs ↔ Maxwell).

see notebooks/ZLvMIT_hybrid.ipynb for details and implementation
```

### 6. ABPR CFL Quark EOS

Analytical CFL quark matter EOS at T=0 (Alford-Braby-Paris-Reddy).

```python
from eos.abpr.eos import (
    ABPRParams,
    pressure_abpr,
    baryon_density_abpr,
    energy_density_abpr,
    mu_from_nB_abpr,
    generate_abpr_tables
)
import numpy as np

# Single point calculation
params = ABPRParams(
    ms=150.0,      # Strange quark mass (MeV)
    Delta=100.0,   # Pairing gap (MeV)
    a4=0.7,        # QCD factor (dimensionless), α = π/2 × (1 - a4)
    B4=145.0       # Bag constant B^(1/4) (MeV)
)

mu = 400.0  # Quark chemical potential (MeV)
P = pressure_abpr(mu, params)           # MeV/fm³
n_B = baryon_density_abpr(mu, params)   # fm⁻³
epsilon = energy_density_abpr(mu, params)  # MeV/fm³

# Inverse: find μ from n_B
mu_solved, converged = mu_from_nB_abpr(n_B=0.5, params=params)

# Generate tables for parameter scan
results = generate_abpr_tables(
    input_type='nB',                        # 'mu', 'nB', 'P', 'epsilon'
    input_values=np.linspace(0.16, 1.6, 300),
    a4_list=[0.6, 0.7, 0.8, 0.9, 1.0],
    B4_list=[135, 165],
    Delta_list=[80, 100, 150, 200],
    ms_list=[150],
    output_dir="./output",
    single_table=False  # Separate file for each parameter set
)
```

### 7. TOV (Neutron Star Structure)

Solve Tolman-Oppenheimer-Volkoff equations for neutron star mass-radius relations.

```python
from eos.tov.solver import (
    compute_tov_sequence,
    EOSTable_for_TOV,
    add_crust,
    generate_ec_logspace
)
import numpy as np

# Generate M-R curve from EOS file
e_c_values = generate_ec_logspace(100, 2000, 50)  # MeV/fm³

results = compute_tov_sequence(
    eos_file="my_eos.dat",
    e_c_vec=e_c_values,

    # Crust options
    add_crust_table='BPS',             # 'No', 'BPS', 'compose_sfho', 'personalized'
    add_crust_mode='interpolate',       # 'attach', 'interpolate', 'maxwell'
    n_transition=0.08,                  # Transition density (fm⁻³)
    delta_n=0.01,                       # Interpolation width

    # For 'personalized' crust:
    # custom_crust_path="/path/to/crust.dat",

    # Computed quantities
    compute_baryonic_mass=True,
    compute_tidal=False,                 # not yet implemented

    output_file="tov_results.dat",
    eos_columns=(0, 1, 2),              # (P, epsilon, nB) column indices
    skip_header=0,                      # Header lines to skip
    verbose=True
)

# Results array columns: e_c, n_c, P_c, R, M, [M_b], [k2], [Lambda]
```

**Crust Matching Modes:**

| Mode | Description |
|------|-------------|
| `attach` | Simple attachment at transition density |
| `interpolate` | Smooth tanh interpolation of P and μ_B |
| `maxwell` | Maxwell construction matching μ_B(P) |

---

## Equilibrium Modes

| Mode | Description | Constraints | Use Case |
|------|-------------|-------------|----------|
| `beta_eq` | Beta equilibrium | μ_n = μ_p + μ_e, charge neutrality | Cold neutron stars |
| `fixed_yc` | Fixed charge fraction | Y_C = n_C/n_B fixed | astrophysical simulations |
| `fixed_yc_ys` | Fixed charge & strangeness | Y_C and Y_S fixed | Heavy-ion collisions |
| `trapped_neutrinos` | Trapped neutrinos | Y_L = (n_e + n_ν)/n_B fixed | Proto-neutron stars |
| `isentropic_beta_eq` | Constant entropy | s/n_B = const, beta equilibrium | Adiabatic evolution |
| `isentropic_trapped` | Constant entropy + trapped | s/n_B = const, Y_L fixed | Proto-neutron star |

---

## Output Format

Tables are saved as whitespace-separated `.dat` files:

```
# SFHo EOS Table: sfho, nucleons
# Equilibrium: beta_eq
# Components: photons, electrons
#          n_B             T         sigma         omega  ...
  1.583000e-02  1.000000e-01  3.924156e+01  2.312847e+01  ...
```

**Standard Columns:**

| Column | Description | Units |
|--------|-------------|-------|
| `n_B` | Baryon number density | fm⁻³ |
| `T` | Temperature | MeV |
| `sigma`, `omega`, `rho`, `phi` | Meson mean fields | MeV |
| `mu_B`, `mu_C`, `mu_S` | Baryon, charge, strangeness chemical potentials | MeV |
| `mu_e`, `mu_nu` | Electron, neutrino chemical potentials | MeV |
| `P_total` | Total pressure | MeV/fm³ |
| `e_total` | Total energy density | MeV/fm³ |
| `s_total` | Total entropy density | fm⁻³ |
| `f_total` | Free energy density (ε - Ts) | MeV/fm³ |
| `Y_C`, `Y_S`, `Y_L` | Charge, strangeness, lepton fractions | dimensionless |
| `converged` | Solver convergence flag | 0 or 1 |

---

## Physical Conventions

### Units
- **Energy/Mass:** MeV
- **Length:** fm
- **Density:** fm⁻³
- **Pressure/Energy density:** MeV/fm³
- **Entropy density:** fm⁻³

### Sign Conventions
- **Strangeness (Y_S):** Number of strange quarks per baryon. S = +1 for hyperons.
  *(Note: some literature uses S = −1)*
- **Charge fraction (Y_C):** Electric charge per baryon of hadrons and quarks (no leptons), Y_C = n_C/n_B

### Key Constants (PDG values)

| Constant | Value | Description |
|----------|-------|-------------|
| ℏc | 197.327 MeV·fm | Reduced Planck constant × c |
| m_n | 939.565 MeV | Neutron mass |
| m_p | 938.272 MeV | Proton mass |
| m_e | 0.511 MeV | Electron mass |
| α | 1/137.036 | Fine structure constant |

---

## References

### Models

**SFHo:**
- Steiner, Hempel & Fischer, ApJ 774, 17 (2013)
- Fortin et al. Astronomical Society of Australia (2018)

**Zhao-Lattimer:**
- Zhao & Lattimer, PRD 102, 023021 (2020)

**ABPR:**
- Alford, Braby, Paris & Reddy, ApJ 629, 969 (2005)

### Phase Transitions
- Constantinou, Guerrini, Zhao, Han, Prakash Phys.Rev.D 112 (2025) 9, 094014
- Constantinou et al. Phys.Rev.D 107 (2023) 7, 074013

### A summary of the models and all the references can be found in:
- M. Guerrini, PhD Thesis, University of Ferrara (2026) - Chapter 2

---

## Future Development

- DD2+vMIT hybrid model with hyperons
- Improved crust models and matching procedures
- Tidal deformability (k₂ and Λ) in TOV solver
- Response functions (heat capacities, susceptibilities, speed of sound)
- Quark matter nucleation rates
- Hydrodynamic hadron-quark conversion flames

---

## Main Collaborators

- **A. Drago** (University of Ferrara) - PhD supervisor
- **G. Pagliara** (University of Ferrara) - PhD supervisor
- **C. Constantinou** (ECT* Trento)
- **A. Lavagno** (Politecnico di Torino)
- **T. Zhao** (N3AS, Berkeley)
- **S. Han** (Tsung-Dao Lee Institute, Shanghai)


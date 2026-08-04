"""
enjl/species.py
====================
Species bookkeeping for the extended NJL model of Xia 2024 (PRD 110, 014022,
arXiv:2405.02946).

Degrees of freedom: baryons p, n, Lambda; quarks u, d, s; leptons e, mu.
Each species carries the conserved charges (B, Q) and strangeness S (S = +1
per s-quark, repo convention), the isospin eigenvalue tau_3 used by the rho
coupling, the valence-quark composition N_i^q of Eq. (4), the degeneracy and
the coupling rescaling f_i (paper Sec. II.B).

Charge neutrality uses the physical electric charge Q; the conserved-charge
decomposition mu_i = B_i mu_b - q_i mu_e (paper Eq. (23)) is implemented in
eos.enjl.eos_beta.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
    """One fermion species of the ENJL model."""
    name: str                # "p", "n", "Lambda", "u", "d", "s", "e", "mu"
    g: float                 # degeneracy (spin x color): 2 baryons/leptons, 6 quarks
    B: float                 # baryon number
    Q: float                 # electric charge [e]
    S: float                 # strangeness (S = +1 per s-quark, repo convention)
    tau3: float              # isospin eigenvalue for the rho coupling
    f: float                 # vector-coupling rescaling f_i (f_p=f_n=1)
    N: tuple                 # valence quarks (N^u, N^d, N^s), for baryons only
    mass: float = 0.0        # bare mass [MeV] (m_q0 for quarks; unused for baryons)
    is_baryon: bool = False
    is_quark: bool = False
    is_lepton: bool = False


def make_species(f_Lambda, f_q):
    """Build the species table for given f_Lambda and f_q = f_u = f_d = f_s."""
    s = []
    # Baryons: valence content N = (N^u, N^d, N^s), g = 2, B = 1, tau3 in
    # the RMF rho convention (p:+1, n:-1, Lambda:0).
    s.append(Species(name="p", g=2.0, B=1.0, Q=1.0, S=0.0, tau3=1.0, f=1.0,
                     N=(2, 1, 0), is_baryon=True))
    s.append(Species(name="n", g=2.0, B=1.0, Q=0.0, S=0.0, tau3=-1.0, f=1.0,
                     N=(1, 2, 0), is_baryon=True))
    s.append(Species(name="Lambda", g=2.0, B=1.0, Q=0.0, S=1.0, tau3=0.0,
                     f=f_Lambda, N=(1, 1, 1), is_baryon=True))
    # Quarks: g = 6, B = 1/3, Q = +2/3 (u), -1/3 (d, s).
    s.append(Species(name="u", g=6.0, B=1.0 / 3.0, Q=2.0 / 3.0, S=0.0,
                     tau3=1.0, f=f_q, N=(1, 0, 0), is_quark=True))
    s.append(Species(name="d", g=6.0, B=1.0 / 3.0, Q=-1.0 / 3.0, S=0.0,
                     tau3=-1.0, f=f_q, N=(0, 1, 0), is_quark=True))
    s.append(Species(name="s", g=6.0, B=1.0 / 3.0, Q=-1.0 / 3.0, S=1.0,
                     tau3=0.0, f=f_q, N=(0, 0, 1), is_quark=True))
    # Leptons: g = 2, B = 0, charge -1, no vector coupling (f = 0).
    s.append(Species(name="e", g=2.0, B=0.0, Q=-1.0, S=0.0, tau3=0.0, f=0.0,
                     N=(0, 0, 0), is_lepton=True))
    s.append(Species(name="mu", g=2.0, B=0.0, Q=-1.0, S=0.0, tau3=0.0, f=0.0,
                     N=(0, 0, 0), is_lepton=True))
    return s


BARYONS = ("p", "n", "Lambda")
QUARKS = ("u", "d", "s")
LEPTONS = ("e", "mu")


def baryon_count(name, q):
    """N_i^q: valence-quark content of baryon `name` for flavor `q`.

    Used in Eq. (4) for M_i and Eq. (6) for the effective scalar density.
    """
    idx = {"u": 0, "d": 1, "s": 2}[q]
    table = {"p": (2, 1, 0), "n": (1, 2, 0), "Lambda": (1, 1, 1)}
    return table[name][idx]

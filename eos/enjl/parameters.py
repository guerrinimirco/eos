"""
enjl/parameters.py
====================
Parameter container for the extended NJL (ENJL) model of Xia 2024 (PRD 110,
014022, arXiv:2405.02946), including the density-dependent couplings of
Table I and the Pauli-blocking parameter B of Eq. (4).

All masses/couplings in MeV units; densities natural (MeV^3) at the internal
boundary. The meson masses only appear in combination with the couplings
(g^2/m^2), exactly as in the paper; they are stored here for the Paper-2 TFA
engine, which needs the bare m_sigma, m_omega, m_rho.
"""
from dataclasses import dataclass
import math

from eos.general.physics_constants import hc3

#: Isospin-source normalization of the rho channel: the effective coupling is
#: 9x the literal reading of Eq. (22), used with J_rho = sum_i f_i tau_i n_i
#: (no factor 3) and tau_p = +1, tau_n = -1. Two independent confirmations:
#:
#:   * it reproduces the paper's symmetry energies S(0.1) = 25.5 MeV and
#:     S(0.158) = 31.5 MeV; the literal Eq. (22) gives 13.3 and 20.2.
#:   * on the nucleonic rows of the author's own beta-equilibrium tables in
#:     test/enjl/reference/, the isospin splitting of the printed potentials
#:     gives g_rho rho = [(E_F^n - E_F^p) - (mu_n - mu_p)] / 2, and dividing
#:     that by J_rho = n_p - n_n yields exactly 9.0000x Eq. (22) at every one
#:     of those densities — with no fit and no symmetry-energy argument.
#:
#: The omega channel needs no such factor, because J_omega already carries
#: N_i = 3 for the baryons; Gamma_w keeps the paper Eq. (21) form.
RHO_FACTOR: float = 9.0


@dataclass(frozen=True)
class ENJLParams:
    # --- RKH NJL parameter set (paper Sec. II.B) ---
    Lambda: float = 602.3          # 3-momentum cut-off [MeV]
    m_u0: float = 5.5              # current u-quark mass [MeV]
    m_d0: float = 5.5              # current d-quark mass [MeV]
    m_s0: float = 140.7            # current s-quark mass [MeV]
    # G_S = g_sigma^2 / (4 m_sigma^2) = 1.835 / Lambda^2
    GS: float = 1.835 / 602.3 ** 2       # [MeV^-2]
    K: float = 12.36 / 602.3 ** 5        # 't Hooft determinant coupling [MeV^-5]

    # --- density-dependent structural function alpha_S(n_b), Eq. (20)/Table I ---
    aS: float = 0.4413715
    bS: float = 0.4076285
    nS: float = 0.16              # fm^-3 (reference density for alpha_S)

    # --- density-dependent vector couplings, Eqs. (21)-(22)/Table I ---
    aV: float = 3.566049
    bV: float = 1.062771
    nV: float = 0.214             # fm^-3
    aTV: float = 0.5014459
    bTV: float = 0.0117601
    nTV: float = 0.1              # fm^-3

    # --- quark / hyperon coupling rescaling (f_p = f_n = 1) ---
    f_Lambda: float = 1.0626      # fixed by U_Lambda(n0) = -30 MeV
    f_q: float = 0.5              # f_u = f_d = f_s = f_q in {0.5, 0.7, 1.0}

    # --- Pauli blocking strength, Eq. (4): M_i += B * n_b^Q ---
    # Paper: "B = 1 GeV/fm^3". In natural units (densities MeV^3) the
    # coefficient B_nat = 1000 * hc^-3 MeV^-2 gives the mass shift in MeV.
    B_GeV_fm3: float = 1.0

    # --- bare meson masses [MeV], Table I of Xia, Maruyama, Yasutake &
    #     Tatsumi, arXiv:2409.12489. They enter the uniform engine only
    #     through g^2/m^2 (i.e. G_S and Gamma_w/r) and so are inert here;
    #     they matter once the gradient terms of that paper are switched on,
    #     which fix the interaction ranges.
    #
    #     m_omega is 1e5 MeV and that is not a typo for 105: the paper adopts
    #     "a rather large omega meson mass" to suppress density fluctuations
    #     in its Thomas-Fermi approximation, and notes it could be reduced
    #     given a better treatment. It is not a physical omega mass and must
    #     not be used as one.
    m_sigma: float = 630.0
    m_omega: float = 1.0e5
    m_rho: float = 769.0

    # --- lepton masses [MeV] ---
    m_e: float = 0.511
    m_mu: float = 105.66

    # ------------------------------------------------------------ derived
    @property
    def Gamma_s(self):
        """g_sigma^2 / m_sigma^2 = 4 G_S [MeV^-2] (constant scalar coupling)."""
        return 4.0 * self.GS

    @property
    def B_nat(self):
        """Pauli-blocking coefficient in natural units [MeV^-2]."""
        return self.B_GeV_fm3 * 1000.0 / hc3

    # ------------------------------------------------ density dependence
    def alpha_S(self, n_b):
        """Structural function alpha_S(n_b), Eq. (20); n_b [MeV^3], dimless."""
        x = n_b / (self.nS * hc3)
        return self.aS * math.exp(-x) + self.bS

    def d_alpha_S(self, n_b):
        """d alpha_S / d n_b [MeV^-3] with n_b in MeV^3."""
        x = n_b / (self.nS * hc3)
        return -self.aS * math.exp(-x) / (self.nS * hc3)

    def Gamma_w(self, n_b):
        """g_omega^2/m_omega^2 = 4 G_S [aV e^{-n_b/nV} + bV] [MeV^-2]."""
        x = n_b / (self.nV * hc3)
        return 4.0 * self.GS * (self.aV * math.exp(-x) + self.bV)

    def d_Gamma_w(self, n_b):
        """d Gamma_w / d n_b [MeV^-5] with n_b in MeV^3."""
        x = n_b / (self.nV * hc3)
        return -4.0 * self.GS * self.aV * math.exp(-x) / (self.nV * hc3)

    def Gamma_r(self, n_b):
        """g_rho^2/m_rho^2 = 9 * 4 G_S [aTV e^{-n_b/nTV} + bTV] [MeV^-2].

        The factor 9 is the isospin-source normalization (see RHO_FACTOR);
        the paper Eq. (22) prints the coupling without it.
        """
        x = n_b / (self.nTV * hc3)
        return RHO_FACTOR * 4.0 * self.GS * (self.aTV * math.exp(-x) + self.bTV)

    def d_Gamma_r(self, n_b):
        """d Gamma_r / d n_b [MeV^-5] with n_b in MeV^3."""
        x = n_b / (self.nTV * hc3)
        return -RHO_FACTOR * 4.0 * self.GS * self.aTV * math.exp(-x) / (self.nTV * hc3)


#: Default parameter set (paper Table I + RKH).
def get_enjl_default() -> ENJLParams:
    return ENJLParams()

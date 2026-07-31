"""
verify_eta_equilibrium.py
=========================
Symbolic verification of the eta-mixed-phase equilibrium conditions
(thesis Ch. 3 / Appendix A), generalized beyond what the thesis derives:
  * nonzero HADRONIC strangeness (hyperons active),
  * modes A (beta-eq), C (fixed Y_C), D (fixed Y_C + Y_S),
  * both readings of "fixed Y_S": globally conserved vs. per-phase (local).

Results (see docs/phase2/STEP0_AUDIT.md):
  - chi-stationarity reduces to P_H = P_Q in every mode, including with
    n_S^H != 0. The eta-dependence cancels out of mechanical equilibrium.
  - For LOCALLY conserved Y_S, the n_B^H variation gives the Appendix-A (A.93)
    combined form  mu_B^H + Ybar_S mu_S^H = mu_B^Q + Ybar_S mu_S^Q,
    NOT a separate mu_S^H = mu_S^Q.

Run: python verify_eta_equilibrium.py
"""
import sympy as sp

chi, eta = sp.symbols('chi eta', positive=True)
nB, nC, nS = sp.symbols('n_B n_C n_S')
nBH, nCH, nSH, neGH = sp.symbols('nBH nCH nSH neGH')
nSQ_ind = sp.Symbol('nSQ_ind')
muBH, muCH, muSH, mueLH, mueGH = sp.symbols('muBH muCH muSH mueLH mueGH')
muBQ, muCQ, muSQ, mueLQ, mueGQ = sp.symbols('muBQ muCQ muSQ mueLQ mueGQ')
PH, PQ = sp.symbols('P_H P_Q')
YSbar = sp.Symbol('YSbar')


def chi_stationarity(mode):
    """Reduce d f_M / d chi = 0 under the chemical conditions of `mode`."""
    nBQ = (nB - (1 - chi) * nBH) / chi                    # Eq. 3.16
    neLH = nCH                                            # Eq. 3.20
    nCQ = ((nC - (1 - chi) * nCH) / chi if mode in ('C', 'D')
           else sp.Symbol('nCQ_ind'))                     # Eq. 3.17
    neLQ = nCQ                                            # Eq. 3.19
    neGQ = ((1 - chi) * (nCH - neGH) + chi * nCQ) / chi    # Eq. 3.10
    nSQ = (nS - (1 - chi) * nSH) / chi if mode == 'D' else nSQ_ind

    # Euler relation per phase, Eq. 3.45, with the eta weights of Eqs. 3.4-3.5
    fH = (-PH + muBH * nBH + muCH * nCH + muSH * nSH
          + eta * mueLH * neLH + (1 - eta) * mueGH * neGH)
    fQ = (-PQ + muBQ * nBQ + muCQ * nCQ + muSQ * nSQ
          + eta * mueLQ * neLQ + (1 - eta) * mueGQ * neGQ)
    dfQ = (muBQ * sp.diff(nBQ, chi) + muCQ * sp.diff(nCQ, chi)
           + muSQ * sp.diff(nSQ, chi) + eta * mueLQ * sp.diff(neLQ, chi)
           + (1 - eta) * mueGQ * sp.diff(neGQ, chi))
    stat = -fH + fQ + chi * dfQ                           # Eq. 3.38

    subs = {muBQ: muBH, mueGQ: mueGH}                     # Eqs. 3.24, 3.33
    if mode == 'A':                                       # Eqs. 3.56-3.57
        subs[muCH] = -eta * mueLH - (1 - eta) * mueGH
        subs[muCQ] = -eta * mueLQ - (1 - eta) * mueGH
        subs[muSH] = 0
        subs[muSQ] = 0
    elif mode == 'C':                                     # Eq. 3.27
        subs[muCQ] = muCH + eta * mueLH - eta * mueLQ
        subs[muSH] = 0
        subs[muSQ] = 0
    elif mode == 'D':                                     # Eq. 3.27 + global S
        subs[muCQ] = muCH + eta * mueLH - eta * mueLQ
        subs[muSQ] = muSH
    return sp.simplify(sp.expand(stat.subs(subs)))


def local_strangeness():
    """Y_S fixed PER PHASE: n_S^I = Ybar_S n_B^I. Returns (n_B^H variation,
    chi-stationarity under the resulting (A.93) condition)."""
    nBQ = (nB - (1 - chi) * nBH) / chi
    nSH_, nSQ_ = YSbar * nBH, YSbar * nBQ
    neLH = nCH
    nCQ = (nC - (1 - chi) * nCH) / chi
    neLQ = nCQ
    neGQ = ((1 - chi) * (nCH - neGH) + chi * nCQ) / chi

    fH = (-PH + muBH * nBH + muCH * nCH + muSH * nSH_
          + eta * mueLH * neLH + (1 - eta) * mueGH * neGH)
    fQ = (-PQ + muBQ * nBQ + muCQ * nCQ + muSQ * nSQ_
          + eta * mueLQ * neLQ + (1 - eta) * mueGQ * neGQ)
    fM = (1 - chi) * fH + chi * fQ

    dnBH = sp.factor(sp.simplify(sp.expand(sp.diff(fM, nBH))))

    dfQ = (muBQ * sp.diff(nBQ, chi) + muCQ * sp.diff(nCQ, chi)
           + muSQ * sp.diff(nSQ_, chi) + eta * mueLQ * sp.diff(neLQ, chi)
           + (1 - eta) * mueGQ * sp.diff(neGQ, chi))
    stat = -fH + fQ + chi * dfQ
    subs = {muBQ: muBH + YSbar * muSH - YSbar * muSQ,      # (A.93)
            mueGQ: mueGH,
            muCQ: muCH + eta * mueLH - eta * mueLQ}
    return dnBH, sp.simplify(sp.expand(stat.subs(subs)))


if __name__ == '__main__':
    for m in ('A', 'C', 'D'):
        r = chi_stationarity(m)
        assert sp.simplify(r - (PH - PQ)) == 0, (m, r)
        print(f'mode {m}: d f_M/d chi = 0  ->  0 = {r}')

    dnBH, stat = local_strangeness()
    print(f'local-S: d f_M/d nBH = 0  ->  0 = {dnBH}')
    assert sp.simplify(stat - (PH - PQ)) == 0, stat
    print(f'local-S: d f_M/d chi = 0  ->  0 = {stat}')
    print('all checks passed')

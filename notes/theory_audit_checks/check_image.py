"""Check 4: crawfish pair weights are the R=0 lattice-image element, so they depend on atom wrapping.

1D dimerized chain (SSH model), lattice constant 1, one s orbital on A and on B.
Short bond (hopping t_s=-1.0), long bond (hopping t_l=-0.3). Half filling (lowest band full).
Bloch sums follow JDFTx's convention phi_k = sum_R e^{ik.R} phi(r - tau - R) (Vnl_calc uses
cis(-2 pi pos.(k+G))), so H_AB(k) = sum_R e^{2 pi i k R} t_AB(R).

Cell choice 1: A at 0.0, B at 0.2 -> same-cell (R=0) partner of A is the SHORT bond.
Cell choice 2: A at 0.0, B at -0.8 (same crystal, B wrapped one cell left) -> R=0 partner is the LONG bond.
"""
import numpy as np
from crawfish.core.operations.matrix import _get_gen_tj_coef_uu

nk = 60
ks = np.arange(nk) / nk
wk = np.full(nk, 2.0 / nk, dtype=np.float32)  # spin-unpolarized JDFTx convention: weights sum to 2
t_s, t_l = -1.0, -0.3


def chain_proj(tau_B):
    # t_AB(R): B image at tau_B + R. Short bond where |tau_B + R| = 0.2, long where it's 0.8.
    proj = np.zeros((nk, 2, 2), dtype=np.complex64)
    e = np.zeros((nk, 2))
    for i, k in enumerate(ks):
        hab = 0j
        for R in range(-2, 3):
            d = abs(tau_B + R)
            if np.isclose(d, 0.2):
                hab += t_s * np.exp(2j * np.pi * k * R)
            elif np.isclose(d, 0.8):
                hab += t_l * np.exp(2j * np.pi * k * R)
        H = np.array([[0, hab], [np.conj(hab), 0]])
        ev, C = np.linalg.eigh(H)
        e[i] = ev
        proj[i] = C.T  # proj[t, j, u] = C_uj  (orthonormal orbitals: <phi_u|psi_j> = C_uj)
    return proj, e


occ = np.zeros((nk, 2), dtype=np.float32)
occ[:, 0] = 1.0  # lower band full
ones = np.ones((2, 2), dtype=np.float32)
for tau_B in (0.2, -0.8):
    proj, e = chain_proj(tau_B)
    bo = np.sum(_get_gen_tj_coef_uu(proj, wk, ones, [0], [1]) * occ)  # crawfish pCOMO weights, integrated
    # Proper R-resolved bond order: P_AB(R) = sum_k w f Re[conj(C_A) C_B e^{2 pi i k R}]
    resolved = {}
    for R in (-1, 0, 1):
        ph = np.exp(2j * np.pi * ks * R)[:, None]
        resolved[R] = float(np.sum(wk[:, None] * occ * (np.conj(proj[:, :, 0]) * proj[:, :, 1] * ph).real))
    print(f"tau_B={tau_B:+.1f}: crawfish IpCOMO(A,B) = {bo:+.4f} | R-resolved bond orders "
          + ", ".join(f"R={R:+d} (d={abs(tau_B+R):.1f}): {v:+.4f}" for R, v in resolved.items()))

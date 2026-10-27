"""Check 1: odd-l projections in bandProjections carry an extra i^l phase.

N2 along x. Orbital order per atom (JDFTx real Ylm): s, py, pz, px.
At Gamma the KS states are real up to a per-band global phase, so for real
orbitals T*_s T_px must be real. If JDFTx's p orbitals are i * (real p), the
product is purely imaginary instead.
"""
import sys
import numpy as np
from crawfish.io.data_parsing import get_proj_tju_from_file, parse_kptsfile

calc = sys.argv[1]
proj = get_proj_tju_from_file(f"{calc}/bandProjections")  # raw, no crawfish processing
wk, ks, nstates = parse_kptsfile(f"{calc}/kPts")
ks = np.array(ks)
e = np.fromfile(f"{calc}/eigenvals").reshape(proj.shape[:2])
f = np.fromfile(f"{calc}/fillings").reshape(proj.shape[:2])
print("fillings max:", f.max(), " sum_k wk:", sum(wk), " N_e = sum wk*f:", float(np.sum(np.array(wk)[:, None] * f)))

s_A, py_A, pz_A, px_A = 0, 1, 2, 3
s_B, py_B, pz_B, px_B = 4, 5, 6, 7
gam = [t for t in range(nstates) if np.allclose(ks[t], 0)]
print("Gamma states:", gam)
t = gam[0]
print(f"{'band':>4} {'e(Ha)':>8} {'f':>5} | {'Re s_A*s_B':>10} {'Im s_A*s_B':>10} | {'Re s_A*px_B':>11} {'Im s_A*px_B':>11} | {'Re pxA*pxB':>10} {'Im pxA*pxB':>10}")
for j in range(proj.shape[1]):
    T = proj[t, j]
    ss = np.conj(T[s_A]) * T[s_B]
    sp = np.conj(T[s_A]) * T[px_B]
    pp = np.conj(T[px_A]) * T[px_B]
    print(f"{j:4d} {e[t,j]:8.4f} {f[t,j]:5.2f} | {ss.real:10.5f} {ss.imag:10.5f} | {sp.real:11.5f} {sp.imag:11.5f} | {pp.real:10.5f} {pp.imag:10.5f}")

# Across all states/bands: fraction of |T*_s T_p| that sits in the imaginary part
num_re = num_im = 0.0
for (a, b) in [(s_A, px_B), (s_B, px_A), (s_A, px_A)]:
    prod = np.conj(proj[:, :, a]) * proj[:, :, b]
    num_re += np.sum(np.abs(prod.real))
    num_im += np.sum(np.abs(prod.imag))
print(f"\nall k: sum|Re(T*_s T_px)| = {num_re:.4f}, sum|Im(T*_s T_px)| = {num_im:.4f}")

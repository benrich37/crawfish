"""Check 2: crawfish ICOHP vs. a reference projected-COHP on the same N2 data.

Reference (LOBSTER-style projected COHP, in crawfish array layout P[t,j,u] = <phi_u|O|psi_j>):
  1. phase fix:   P[..., u] *= i**l_u              (JDFTx orbitals are i^l * real orbital)
  2. Lowdin within the band space (crawfish's los_projs_for_orbs), so C[t,j,u] = P'[t,j,u]
  3. H_t[u,v] = sum_j P'[t,j,u] e[t,j] conj(P'[t,j,v])      (= <chi_u|H|chi_v>)
  4. ICOHP_uv = sum_tj w_t f_tj Re[conj(C[t,j,u]) H_t[u,v] C[t,j,v]]
Sum rule: sum_{u,v} ICOHP_uv = sum_tj w f (Pi e Pi)_jj  ~ band energy when spilling is small.
"""
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from crawfish.core.elecdata import ElecData
from crawfish.core.operations.matrix import los_projs_for_orbs
from crawfish.funcs.pcohp import get_ipcohp, get_ipcomo

calc = sys.argv[1]
edata = ElecData.from_calc_dir(calc)
print("crawfish settings: los_orbs", edata.los_orbs, "p_uu_consistent", edata.p_uu_consistent,
      "norm_eigs_for_h", edata.norm_eigs_for_h, "trim", edata.trim_excess_bands)

names = ["s", "py", "pz", "px"]
lvals = [0, 1, 1, 1]
pairs = [("s", "s"), ("s", "px"), ("px", "px"), ("py", "py"), ("pz", "pz")]


def crawfish_val(func, o1, o2):
    es, cs = func(edata, idcs1=0, idcs2=1, orbs1=o1, orbs2=o2, use_cache=False)
    return float(np.asarray(cs).ravel()[-1])


# --- reference ---
from crawfish.io.data_parsing import get_proj_tju_from_file
raw = get_proj_tju_from_file(f"{calc}/bandProjections").astype(np.complex128)
e = edata.e_tj.astype(np.float64)
f = np.fromfile(f"{calc}/fillings").reshape(e.shape)
w = edata.wk_t.astype(np.float64)
phase = np.array([1j ** l for l in lvals * 2])
P = los_projs_for_orbs(raw * phase[None, None, :])
H = np.einsum("tju,tj,tjv->tuv", P, e, P.conj())
icohp = np.einsum("t,tj,tju,tuv,tjv->uv", w, f, P.conj(), H, P).real
band_energy = np.sum(w[:, None] * f * e)
print(f"\nreference sum rule: sum_uv ICOHP = {icohp.sum():.5f} Ha, band energy sum w f e = {band_energy:.5f} Ha")

idx = {n: i for i, n in enumerate(names)}
print(f"\n{'pair':>8} | {'crawfish IpCOHP':>15} {'crawfish IpCOMO':>15} | {'reference ICOHP (Ha)':>20}")
for o1, o2 in pairs:
    cf = crawfish_val(get_ipcohp, o1, o2)
    cm = crawfish_val(get_ipcomo, o1, o2)
    u, v = idx[o1], 4 + idx[o2]
    ref = icohp[u, v]  # same one-direction convention as crawfish (orbs1 on atom 0, orbs2 on atom 1)
    print(f"{o1+'-'+o2:>8} | {cf:15.5f} {cm:15.5f} | {ref:20.5f}")
inter = icohp[:4, 4:].sum()
print(f"\nreference total inter-atomic ICOHP (A->B, one direction): {inter:.4f} Ha  (negative = bonding)")

# --- basis / energy-zero dependence ---
# Non-orthogonal band-projected basis: S = T T^dag, C = S^-1 T, H = T e T^dag (no Lowdin step).
Pn = raw * phase[None, None, :]
s_conj = np.einsum("tju,tjv->tuv", Pn.conj(), Pn)  # crawfish's s_uu convention (= conj(S))
C = np.einsum("tjw,twu->tju", Pn, np.linalg.inv(s_conj))
S = np.einsum("tju,tjv->tuv", Pn, Pn.conj())
mu = float(edata.mu)
for label, shift in (("e", 0.0), ("e - mu", mu)):
    Hn = np.einsum("tju,tj,tjv->tuv", Pn, e - shift, Pn.conj())
    ic = np.einsum("t,tj,tju,tuv,tjv->uv", w, f, C.conj(), Hn, C).real
    print(f"non-orthogonal basis, H from {label:>6}: s-s {ic[0,4]:+.5f}  s-px {ic[0,7]:+.5f}  px-px {ic[3,7]:+.5f}"
          f"  py-py {ic[1,5]:+.5f}  A-B total {ic[:4,4:].sum():+.4f}")
ioop = np.einsum("t,tj,tju,tuv,tjv->uv", w, f, C.conj(), S, C).real
print(f"non-orthogonal basis ICOOP:            s-s {ioop[0,4]:+.5f}  s-px {ioop[0,7]:+.5f}  px-px {ioop[3,7]:+.5f}"
      f"  py-py {ioop[1,5]:+.5f}  A-B total {ioop[:4,4:].sum():+.4f}")
S_los = np.einsum("tju,tjv->tuv", P, P.conj())
print(f"after Lowdin, max |S_uv - delta_uv| = {np.abs(S_los - np.eye(8)).max():.2e}  -> COOP vanishes off-diagonal")

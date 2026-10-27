"""Check 3: normalization of Gaussian vs LTI spectra, occupation helpers, spilling diagnostics."""
import sys
import warnings
import numpy as np
from scipy.integrate import trapezoid

warnings.filterwarnings("ignore")
from crawfish.core.elecdata import (
    ElecData, calculate_filling_nobroad, calculate_filling_gauss,
)
from crawfish.funcs.dos import get_dos
from crawfish.io.data_parsing import get_proj_tju_from_file

calc = sys.argv[1]
edata = ElecData.from_calc_dir(calc)
nk = np.prod(edata.kfolding)
print("kfolding", edata.kfolding, "nspin", edata.nspin, "sum wk", edata.wk_t.sum(), "lti_allowed", edata.lti_allowed)

# --- DOS normalization: integral over all energies should be sum_t w_t * nbands = 2 * nbands
erange = np.arange(edata.e_tj.min() - 0.05, edata.e_tj.max() + 0.05, 0.0005, dtype=np.float32)
expected = float(edata.wk_t.sum() * edata.nbands)
for lti in (False, True):
    er, dos = get_dos(edata, erange=erange, lti=lti)
    print(f"lti={lti}: integral of DOS = {trapezoid(np.asarray(dos).ravel(), er):.5f}   (expected {expected:.1f})")
print(f"Gaussian amplitude factor sqrt(pi*sig) for sig=1e-5: {np.sqrt(np.pi*1e-5):.5f};  1/nk = {1/nk:.5f}")

# --- occupation helpers
eig = np.array([-0.5, -0.1, 0.1, 0.5])
print("\ncalculate_filling_nobroad(mu=0.0) for eig", eig, "->", calculate_filling_nobroad(0.0, eig, 0.0),
      " (expected [1,1,0,0])")
try:
    calculate_filling_gauss(0.01, eig, 0.0)
    print("calculate_filling_gauss ok")
except TypeError as err:
    print("calculate_filling_gauss on an array raises TypeError:", err)

# --- spilling diagnostics (with ortho-orbital projections, |T|^2 sums are meaningful)
raw = get_proj_tju_from_file(f"{calc}/bandProjections").astype(np.complex128)
f = np.fromfile(f"{calc}/fillings").reshape(edata.e_tj.shape)
w = edata.wk_t
band_cov = np.sum(np.abs(raw) ** 2, axis=2)  # (t, j): how much of band j lies in the orbital span
occ_spill = 1 - np.sum(w[:, None] * f * band_cov) / np.sum(w[:, None] * f)
orb_cov = np.einsum("tju,t->u", np.abs(raw) ** 2, w) / w.sum()  # (u): how much of orbital u lies in the band span
print(f"\ncharge spilling (occupied, Sanchez-Portal definition): {occ_spill:.4%}")
print("orbital completeness <phi_u|P_bands|phi_u> averaged over k:", np.round(orb_cov, 4))

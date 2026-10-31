"""Shared helpers for the benchmarking notebooks in this directory.

The notebooks show how crawfish's results converge with the JDFTx settings a user chooses (number of bands,
pseudopotential, projection type) and compare crawfish with orbital bases it does not implement yet: frozen-window
projected Wannier functions, their projectability-disentangled variant, JDFTx's maximally localized Wannier
functions, LOBSTER, and two band-independent bases built from exact atomic-orbital matrices. The derivations and
the history of these results are in `notes/wannier_and_adaptive_basis.md` (sections 4 and 10).

Conventions follow the README (section "Provided analysis techniques"):
- T_{mu,j}(t) = <phi_mu|O|psi_j(t)> are JDFTx's projections with the i^l phase applied (real orbitals), stored as
  `proj_tju[t, j, mu]`;
- a basis is a set of orthonormal functions w_mu; `coef_tju[t, j, mu]` = C_{mu,j}(t) = <w_mu|O|psi_j(t)>;
- its Hamiltonian is H_{mu,nu}(t) = <w_mu|H|w_nu>, by default the band sum sum_j C_{mu,j} eps_j C*_{nu,j};
- the integrated pCOHP of orbitals (mu on atom 1, nu on the image R of atom 2) is
  sum_{t,j} w_t f_j(t) Re[C*_{mu,j} H_{mu,nu}(R) e^{2 pi i k_t.R} C_{nu,j}], with
  H(R) = sum_t w_t e^{-2 pi i k_t.R} H(t) / sum_t w_t (crawfish's image convention, README).
Energies are in Hartree, k-point weights sum to 2 (no-spin), occupations per state are in [0, 1].
Everything here is float64/complex128 and independent of crawfish's code paths; `crawfish_ipcohp` runs crawfish
itself on band-truncated copies of the same files. The spectra are pCOHP(E) = sum_{t,j} (the same weights, without
f) delta(E - eps_j), Gaussian-broadened (`bond_pcohp`, `crawfish_pcohp`, `lobster_bond_pcohp`).
"""

from __future__ import annotations

import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*Numba.*")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"

GRAPHENE_BONDS = ((0, 0, 0), (-1, 0, 0), (0, 1, 0))  # image R of atom 2 for the three equivalent C1-C2 bonds


@dataclass(frozen=True)
class System:
    """One JDFTx calculation in `data/<name>/` (see that folder's `*.in` files)."""

    name: str
    molecule: str
    psp: str
    bonds: tuple  # lattice image R of atom 2 for each equivalent bond with atom 1 in the home cell
    nbands_fixture: int  # elec-n-bands of the matching crawfish test fixture
    nbands_series: tuple  # band counts emulated by truncation (the largest is the calculation's elec-n-bands)
    e_frozen: float  # Ha: upper edge of the frozen (inner) window, occupied states only (N2) or mu + 0.05 Ha
    sigma: float  # Ha: Fermi smearing of the calculation
    kfolding: tuple
    wannier_runs: tuple = ()  # JDFTx `wannier` outputs present in the folder: (projection-only, minimized)


SYSTEMS = {
    s.name: s
    for s in (
        System("N2_gbrv", "N2", "GBRV ultrasoft, 20 Ha", ((0, 0, 0),), 15, (8, 10, 12, 15, 20, 30, 40, 60, 80),
               -0.09, 0.001, (1, 1, 1), ("wproj_occ", "wmlwf_pin")),
        System("N2_sg15", "N2", "SG15 norm-conserving, 30 Ha", ((0, 0, 0),), 15, (8, 10, 12, 15, 20, 30, 40),
               -0.09, 0.001, (1, 1, 1)),
        System("graphene_gbrv", "graphene", "GBRV ultrasoft, 20 Ha", GRAPHENE_BONDS, 20,
               (8, 10, 12, 16, 20, 30, 40, 48), 0.026, 0.005, (12, 12, 1), ("wproj_mu05", "wmlwf_pin")),
        System("graphene_sg15", "graphene", "SG15 norm-conserving, 30 Ha", GRAPHENE_BONDS, 20,
               (8, 10, 12, 16, 20, 30, 40, 48), 0.026, 0.005, (12, 12, 1)),
    )
}

# projection files: name -> (file prefix, JDFTx `band-projection-params`)
PROJECTIONS = {
    "ortho": ("aomo", "yes no (orthonormalized orbitals; crawfish's default input)"),
    "nonortho": ("aom", "no no + full overlap operator (patched JDFTx `fullO`)"),
    "nonortho_jdftx": ("nonortho", "no no (JDFTx as released: single-centre O*phi for ultrasoft)"),
}
ATOM1 = range(0, 4)  # s, py, pz, px of atom 1 (JDFTx's real-harmonic order)
ATOM2 = range(4, 8)
M_LABELS = {0: ["s"], 1: ["py", "pz", "px"], 2: ["dxy", "dyz", "dz2", "dxz", "dx2-y2"]}


# ----------------------------------------------------------------------------------------- reading JDFTx output

def read_bandprojections(path: Path) -> tuple[np.ndarray, list[tuple]]:
    """Projections as dumped (no phase applied) and the orbital list [(species, atom, l, label)]."""
    lines = Path(path).read_text().splitlines()
    m = re.match(r"(\d+) states, (\d+) bands, (\d+) (?:ortho-)?orbital-projections, (\d+) species", lines[0])
    nstates, nbands, nproj, nspecies = (int(m[i]) for i in range(1, 5))
    orbs, iatom = [], 0
    for line in lines[2:2 + nspecies]:
        tok = line.split()
        sym, nat, lmax = tok[0], int(tok[1]), int(tok[3])
        nshells = [int(x) for x in tok[4:5 + lmax]]
        for _ in range(nat):
            for l in range(lmax + 1):
                for n in range(nshells[l]):
                    orbs += [(sym, iatom, l, (f"{n}" if nshells[l] > 1 else "") + lab) for lab in M_LABELS[l]]
            iatom += 1
    proj_tju = np.zeros((nstates, nbands, nproj), complex)
    i = 2 + nspecies
    for t in range(nstates):
        block = np.array(" ".join(lines[i + 1:i + 1 + nbands]).split(), float).reshape(nbands, nproj, 2)
        proj_tju[t] = block[..., 0] + 1j * block[..., 1]
        i += 1 + nbands
    return proj_tju, orbs


@dataclass
class Calc:
    system: System
    proj_tju: np.ndarray  # T_{mu,j}(t) = <phi_mu|O|psi_j(t)>, real orbitals
    e_tj: np.ndarray
    occ_tj: np.ndarray
    wk_t: np.ndarray
    kpts_t: np.ndarray  # (t, 3) in reciprocal-lattice coordinates
    orbs: list
    mu: float

    @property
    def nbands(self) -> int:
        return self.e_tj.shape[1]

    @property
    def projectability_tj(self) -> np.ndarray:
        """sum_mu |T_{mu,j}|^2 (meaningful for orthonormalized-orbital projections)."""
        return (np.abs(self.proj_tju) ** 2).sum(-1)


def load(system: System | str, proj: str = "ortho") -> Calc:
    system = SYSTEMS[system] if isinstance(system, str) else system
    d = DATA_DIR / system.name
    raw_tju, orbs = read_bandprojections(d / f"{PROJECTIONS[proj][0]}.bandProjections")
    nstates, nbands, _ = raw_tju.shape
    lphase_u = np.array([1j ** o[2] for o in orbs])  # README "Orbital phase convention"
    kpts, wk = [], []
    for line in (d / "totalE.kPts").read_text().splitlines():
        m = re.search(r"\[\s*(\S+)\s+(\S+)\s+(\S+)\s*\]\s+(\S+)", line)
        if m:
            kpts.append([float(m[i]) for i in (1, 2, 3)])
            wk.append(float(m[4]))
    mu = float(re.findall(r"mu:\s+(\S+)", (d / "totalE.out").read_text())[-1])
    return Calc(system=system, proj_tju=raw_tju * lphase_u, orbs=orbs, mu=mu,
                e_tj=np.fromfile(d / "totalE.eigenvals").reshape(nstates, nbands),
                occ_tj=np.fromfile(d / "totalE.fillings").reshape(nstates, nbands) / 2.0,  # no-spin file is [0, 2]
                wk_t=np.array(wk), kpts_t=np.array(kpts))


def load_ao_matrices(calc: Calc) -> tuple[np.ndarray, np.ndarray]:
    """Exact S_{mu,nu}(t) = <phi_mu|O|phi_nu> and H_{mu,nu}(t) = <phi_mu|H|phi_nu> (patched JDFTx `aoMatrices`),
    converted to the real-orbital convention of `proj_tju` (phi_real = (-i)^l phi_JDFTx)."""
    d = DATA_DIR / calc.system.name
    n, nt = len(calc.orbs), len(calc.wk_t)
    dphase_u = np.array([(-1j) ** o[2] for o in calc.orbs])
    out = []
    for name in ("aoOverlap", "aoHamiltonian"):
        m_tuu = np.fromfile(d / f"aom.{name}", np.complex128).reshape(nt, n, n).transpose(0, 2, 1)  # column-major
        out.append(dphase_u.conj()[None, :, None] * m_tuu * dphase_u[None, None, :])
    return out[0], out[1]


# ----------------------------------------------------------------------------------------- bases

@dataclass
class Basis:
    """Orthonormal functions w_mu described by their coefficients on a set of rows (bands, or bands plus extra
    directions) at each state: coef_tju[t, row, mu] = <w_mu|O|row>, with energies/occupations per row."""

    coef_tju: np.ndarray
    e_tj: np.ndarray
    occ_tj: np.ndarray
    h_tuu: np.ndarray | None = None  # <w_mu|H|w_nu>; None -> band sum sum_j C eps C*

    @property
    def hamiltonian_tuu(self) -> np.ndarray:
        if self.h_tuu is not None:
            return self.h_tuu
        return np.einsum("tju,tj,tjv->tuv", self.coef_tju, self.e_tj, self.coef_tju.conj())


def msqrt(x: np.ndarray, p: float) -> np.ndarray:
    e, v = np.linalg.eigh(x)
    return (v * e ** p) @ v.conj().T


def _invsqrt_cut(g: np.ndarray, rtol: float) -> np.ndarray:
    """G^(-1/2) on the eigenvalues above rtol * max (crawfish's Löwdin cutoff, README / audit F10)."""
    e, v = np.linalg.eigh(g)
    keep = e > rtol * e.max()
    return (v * np.where(keep, 1 / np.sqrt(np.where(keep, e, 1.0)), 0.0)) @ v.conj().T


def lowdin(calc: Calc, nbands: int | None = None, rtol: float | None = 1e-6) -> Basis:
    """crawfish's default (`los_orbs=True`): the band-projected orbitals chi_mu = sum_{j<=nb} psi_j <psi_j|O|phi_mu>,
    Löwdin-orthonormalized: C = T (T^dag T)^(-1/2) over the lowest nb bands. rtol=None gives the pure polar factor
    (no cutoff), which is what LOBSTER does with nb = number of orbitals."""
    nb = nbands or calc.nbands
    t_tju = calc.proj_tju[:, :nb, :]
    if rtol is None:
        w, _, vh = np.linalg.svd(t_tju, full_matrices=False)
        c_tju = w @ vh
    else:
        c_tju = np.array([t @ _invsqrt_cut(t.conj().T @ t, rtol) for t in t_tju])
    return Basis(c_tju, calc.e_tj[:, :nb], calc.occ_tj[:, :nb])


def crawfish_basis(calc: Calc, nbands: int | None = None) -> Basis:
    """`lowdin` exactly as crawfish evaluates it by default: eigenvalues measured from mu (`norm_eigs_for_h=True`).
    That shift only changes diagonal H elements while sum_j C C^dag = 1, i.e. unless the Löwdin cutoff dropped an
    orbital direction (`n_dropped_states`); then it also changes the pair values."""
    b = lowdin(calc, nbands)
    return Basis(b.coef_tju, b.e_tj - calc.mu, b.occ_tj)


def n_dropped_states(calc: Calc, nbands: int, rtol: float = 1e-6) -> int:
    """Number of states at which crawfish's Löwdin cutoff drops an orbital direction (eigenvalue of T^dag T below
    rtol of the largest), i.e. where the lowest nb bands do not span the orbitals."""
    e_tu = np.linalg.eigvalsh(np.einsum("tju,tjv->tuv", calc.proj_tju[:, :nbands].conj(), calc.proj_tju[:, :nbands]))
    return int((e_tu.min(1) <= rtol * e_tu.max(1)).sum())


def frozen_window_pwf(calc: Calc, e_frozen: float, nbands: int | None = None, p_min: float | None = None,
                      p_max: float | None = None, projectability_tj: np.ndarray | None = None) -> Basis:
    """Frozen-window projected Wannier functions in the space of the lowest nb bands (JDFTx `wannier` with
    `nIterations 0`, i.e. WannierMinimizer::initRotations). The trial orbitals g_mu are the orbitals `calc` was
    projected on (Löwdin AOs for "ortho", bare phi for "nonortho"). Bands with eps <= e_frozen are kept exactly;
    the other n_W - n_F directions are the combinations of the remaining bands that best overlap the trial
    directions the frozen bands do not cover; the trial orbitals are then Löwdin-orthonormalized in that subspace.
    p_min / p_max: projectability thresholds (JDFTx `projectionThresholds`, PDWF): bands with p <= p_min are left
    out, bands with p > p_max are frozen too."""
    nb = nbands or calc.nbands
    nt, _, nw = calc.proj_tju.shape
    c_all = []
    for t in range(nt):
        a = calc.proj_tju[t, :nb].conj()  # A_{j,mu} = <psi_j|O|g_mu>
        frozen = calc.e_tj[t, :nb] <= e_frozen
        outer = np.ones(nb, bool)
        if projectability_tj is not None:
            if p_max is not None:
                frozen |= projectability_tj[t, :nb] > p_max
            if p_min is not None:
                outer = projectability_tj[t, :nb] > p_min
        fixed, partial = np.where(frozen)[0], np.where(outer & ~frozen)[0]
        nfix = len(fixed)
        nfree = nw - nfix
        if nfix > nw or len(partial) < nfree:
            raise ValueError(f"state {t}: {nfix} frozen and {len(partial)} other bands for {nw} functions")
        u1 = np.zeros((nb, nw), complex)  # orthonormal basis of the subspace, as band combinations
        u1[fixed, np.arange(nfix)] = 1.0
        if nfree:
            vfree = np.linalg.svd(a[fixed], full_matrices=True)[2][nfix:].conj().T if nfix else np.eye(nw)
            ub, _, vbh = np.linalg.svd(a[partial] @ vfree, full_matrices=False)
            u1[partial, nfix:] = ub @ vbh
        wg = u1.conj().T @ a  # trial orbitals in the subspace
        lam = np.linalg.eigvalsh(wg.conj().T @ wg)
        if lam.min() <= 1e-12 * lam.max():
            raise ValueError(f"state {t}: the trial orbitals do not cover the subspace (too little orbital "
                             f"character in the non-frozen bands)")
        u = u1 @ (wg @ msqrt(wg.conj().T @ wg, -0.5))
        c_all.append(u.conj())
    return Basis(np.array(c_all), calc.e_tj[:, :nb], calc.occ_tj[:, :nb])


def exact_lowdin(calc_ortho: Calc, s_tuu: np.ndarray, h_tuu: np.ndarray) -> Basis:
    """nb -> infinity limit of crawfish's basis: the exact Löwdin orbitals phi~ = phi S^(-1/2). Their coefficients
    <phi~_mu|O|psi_j> are JDFTx's ortho projections (exact band by band) and <phi~|H|phi~> = S^-1/2 H S^-1/2."""
    sm = np.array([msqrt(s, -0.5) for s in s_tuu])
    return Basis(calc_ortho.proj_tju, calc_ortho.e_tj, calc_ortho.occ_tj, h_tuu=sm @ h_tuu @ sm)


def complete_frozen_window(calc_nonortho: Calc, s_tuu: np.ndarray, h_tuu: np.ndarray, e_frozen: float,
                           trial: str = "lowdin", occ_tol: float = 1e-3) -> Basis:
    """Frozen-window projected Wannier functions in the complete limit, from the exact AO matrices (no empty bands
    needed). Subspace = span(psi_F) + span(f_alpha), f = g V with V the trial combinations having zero overlap
    with every frozen band (exact, from the full-O non-orthogonal projections). H is block diagonal:
    <psi_i|H|psi_j> = eps_j delta_ij, <psi_j|H|f> = eps_j <psi_j|O|f> = 0, <f|H|f> = V^dag <g|H|g> V.
    Rows of the returned basis are the frozen bands followed by the f directions (occupation 0)."""
    nt, nb, nw = calc_nonortho.proj_tju.shape
    c_all, h_all, occ_all, e_all = [], [], [], []
    for t in range(nt):
        m = msqrt(s_tuu[t], -0.5) if trial == "lowdin" else np.eye(nw)
        sg, hg = m.conj().T @ s_tuu[t] @ m, m.conj().T @ h_tuu[t] @ m
        ag = calc_nonortho.proj_tju[t].conj() @ m  # <psi_j|O|g_mu>
        f = np.where(calc_nonortho.e_tj[t] <= e_frozen)[0]
        if calc_nonortho.occ_tj[t, len(f):].max(initial=0) > occ_tol:
            raise ValueError(f"state {t}: occupied band above the frozen window")
        nf = len(f)
        vfree = np.linalg.svd(ag[f], full_matrices=True)[2][nf:].conj().T
        ffun = vfree @ msqrt(vfree.conj().T @ sg @ vfree, -0.5)  # orthonormal f
        b = np.vstack([ag[f], ffun.conj().T @ sg])  # trial orbitals in the (psi_F, f) basis
        usub = b @ msqrt(b.conj().T @ b, -0.5)
        hsub = np.zeros((nw, nw), complex)
        hsub[:nf, :nf] = np.diag(calc_nonortho.e_tj[t, f])
        hsub[nf:, nf:] = ffun.conj().T @ hg @ ffun
        c_all.append(usub.conj())
        h_all.append(usub.conj().T @ hsub @ usub)
        occ_all.append(np.r_[calc_nonortho.occ_tj[t, f], np.zeros(nw - nf)])
        e_all.append(np.r_[calc_nonortho.e_tj[t, f], np.full(nw - nf, np.nan)])
    return Basis(np.array(c_all), np.array(e_all), np.array(occ_all), h_tuu=np.array(h_all))


def fermi(e, calc: Calc):
    """Fermi occupations with the calculation's mu and smearing."""
    return 0.5 * (1 - np.tanh((e - calc.mu) / (2 * calc.system.sigma)))


def jdftx_wannier(calc: Calc, run: str) -> Basis:
    """A basis from JDFTx `wannier` output (`<run>.mlwfH`, cell map and weights): H(k) is Wannier-interpolated on
    the DFT k-mesh; its eigenvectors play the role of bands, occupied with the calculation's Fermi smearing."""
    d, kfold = DATA_DIR / calc.system.name, np.array(calc.system.kfolding)
    cell_map = np.loadtxt(d / f"{run}.mlwfCellMap", ndmin=2)[:, 0:3].astype(int)
    weights = np.fromfile(d / f"{run}.mlwfCellWeights")
    nw = int(round(np.sqrt(weights.size / len(cell_map))))
    weights = weights.reshape(len(cell_map), nw, nw).swapaxes(1, 2)
    h_red = np.fromfile(d / f"{run}.mlwfH").reshape(np.prod(kfold), nw, nw).swapaxes(1, 2)
    i_red = np.mod(cell_map, kfold) @ np.array([kfold[1] * kfold[2], kfold[2], 1])
    h_k = np.tensordot(np.exp(2j * np.pi * calc.kpts_t @ cell_map.T), weights * h_red[i_red], axes=1)
    e_tj, v = np.linalg.eigh(h_k)
    return Basis(v.transpose(0, 2, 1), e_tj, fermi(e_tj, calc), h_tuu=h_k)


def ritz_states(basis: Basis, calc: Calc) -> Basis:
    """The same functions w_mu, with the rows replaced by the eigenstates of <w|H|w> within their span (energies
    are Ritz values; Fermi occupations). Needed for spectra of `complete_frozen_window`, whose f rows are not
    eigenstates; its frozen rows come back unchanged because H is block diagonal there."""
    e_tj, v = np.linalg.eigh(basis.hamiltonian_tuu)
    return Basis(v.transpose(0, 2, 1), e_tj, fermi(e_tj, calc), h_tuu=basis.hamiltonian_tuu)


# ----------------------------------------------------------------------------------------- analysis

def pair_icohp(basis: Basis, calc: Calc, orbs_a=ATOM1, orbs_b=ATOM2, image=(0, 0, 0), per_pair: bool = False):
    """Integrated pCOHP (Ha, negative = bonding) between orbitals orbs_a of atom 1 and orbs_b of atom 2's image."""
    ph_t = np.exp(2j * np.pi * calc.kpts_t @ np.asarray(image, float))
    h_r = np.einsum("t,tuv->uv", calc.wk_t * ph_t.conj(), basis.hamiltonian_tuu) / calc.wk_t.sum()
    f_tj = basis.occ_tj * calc.wk_t[:, None]
    c = basis.coef_tju
    out = {(a, b): float((f_tj * np.real(c[:, :, a].conj() * h_r[a, b] * ph_t[:, None] * c[:, :, b])).sum())
           for a in orbs_a for b in orbs_b}
    return out if per_pair else sum(out.values())


def bond_icohp(basis: Basis, calc: Calc) -> float:
    """Mean over the system's equivalent bonds."""
    return float(np.mean([pair_icohp(basis, calc, image=r) for r in calc.system.bonds]))


def charge_spilling(basis: Basis, calc: Calc) -> float:
    """Fraction of the occupied density outside the basis: 1 - sum w f sum_mu |C|^2 / sum w f."""
    f_tj = basis.occ_tj * calc.wk_t[:, None]
    return 1.0 - float((f_tj * (np.abs(basis.coef_tju) ** 2).sum(-1)).sum() / f_tj.sum())


def sum_rule_miss(basis: Basis, calc: Calc) -> float:
    """sum over all orbital pairs and images of ICOHP (= sum w f <psi|P_W H P_W|psi>) minus the band energy."""
    c = basis.coef_tju
    tot_tj = np.einsum("tju,tuv,tjv->tj", c.conj(), basis.hamiltonian_tuu, c).real
    return float((basis.occ_tj * calc.wk_t[:, None] * tot_tj).sum()
                 - (calc.occ_tj * calc.wk_t[:, None] * calc.e_tj).sum())


def n2_channels(basis: Basis, calc: Calc) -> dict[str, float]:
    """N2 (bond along x) ICOHP split into sigma (s-s, s-px + px-s, px-px) and pi (py-py + pz-pz)."""
    p = pair_icohp(basis, calc, per_pair=True)
    return {"σ s–s": p[(0, 4)], "σ s–px": p[(0, 7)] + p[(3, 4)], "σ px–px": p[(3, 7)], "π": p[(1, 5)] + p[(2, 6)]}


def bond_weights_tj(basis: Basis, calc: Calc, orbs_a=ATOM1, orbs_b=ATOM2) -> np.ndarray:
    """pCOHP weight of each row, w_t Re[sum_{mu,nu} C*_{mu,j} H_{mu,nu}(R) e^{2 pi i k_t.R} C_{nu,j}] (Ha, no
    occupation) for orbitals orbs_a of atom 1 and orbs_b of atom 2, averaged over the system's equivalent bonds.
    sum f * weights = `bond_icohp`."""
    orbs_a, orbs_b = list(orbs_a), list(orbs_b)
    out = 0.0
    for image in calc.system.bonds:
        ph_t = np.exp(2j * np.pi * calc.kpts_t @ np.asarray(image, float))
        h_r = np.einsum("t,tuv->uv", calc.wk_t * ph_t.conj(), basis.hamiltonian_tuu) / calc.wk_t.sum()
        c_a, c_b = basis.coef_tju[:, :, orbs_a], basis.coef_tju[:, :, orbs_b]
        g_tj = np.einsum("tju,uv,tjv->tj", c_a.conj(), h_r[np.ix_(orbs_a, orbs_b)], c_b) * ph_t[:, None]
        out = out + calc.wk_t[:, None] * g_tj.real
    return out / len(calc.system.bonds)


def gauss_spectrum(e_tj: np.ndarray, w_tj: np.ndarray, erange: np.ndarray, std: float) -> np.ndarray:
    """sum_{t,j} w_tj g(E - e_tj), with g a unit-area Gaussian of standard deviation std (crawfish's kernel with
    sig = 2 std^2). Rows with a NaN energy are skipped."""
    ok = np.isfinite(e_tj)
    x = (erange[:, None] - e_tj[ok][None, :]) / std
    return np.exp(-0.5 * x ** 2) @ w_tj[ok] / (std * np.sqrt(2 * np.pi))


def bond_pcohp(basis: Basis, calc: Calc, erange: np.ndarray, std: float) -> np.ndarray:
    """Bond pCOHP(E) on erange (absolute Ha) for every row of the basis (occupied or not); negative = bonding."""
    return gauss_spectrum(basis.e_tj, bond_weights_tj(basis, calc), erange, std)


# ----------------------------------------------------------------------------------------- crawfish itself

def write_truncated_view(system: System, nbands: int, proj: str, out_dir: Path) -> Path:
    """A calc dir crawfish can read (files named by suffix) holding the lowest nb bands of the calculation:
    the same states a run with `elec-n-bands nb` computes."""
    src = DATA_DIR / system.name
    lines = (src / f"{PROJECTIONS[proj][0]}.bandProjections").read_text().splitlines()
    m = re.match(r"(\d+) states, (\d+) bands, (.*)", lines[0])
    nstates, nb_all = int(m[1]), int(m[2])
    nspecies = int(lines[0].split(",")[-1].split()[0])
    out = [f"{nstates} states, {nbands} bands, {m[3]}"] + lines[1:2 + nspecies]
    i = 2 + nspecies
    for _ in range(nstates):
        out += lines[i:i + 1 + nbands]
        i += 1 + nb_all
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bandProjections").write_text("\n".join(out) + "\n")
    for name in ("eigenvals", "fillings"):
        np.fromfile(src / f"totalE.{name}").reshape(nstates, nb_all)[:, :nbands].tofile(out_dir / name)
    for name in ("kPts", "out"):
        (out_dir / name).write_bytes((src / f"totalE.{name}").read_bytes())
    return out_dir


def crawfish_ipcohp(system: System | str, nbands: int, proj: str = "ortho", norm_eigs_for_h: bool | None = None):
    """crawfish's bond IpCOHP (Ha, mean over the equivalent bonds) for the lowest nb bands, with crawfish's default
    settings unless norm_eigs_for_h is given, and the messages of the warnings it raised (e.g. the too-few-bands
    warning of README "JDFTx settings")."""
    from crawfish.core.elecdata import ElecData
    from crawfish.funcs.pcohp import get_ipcohp

    system = SYSTEMS[system] if isinstance(system, str) else system
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edata = ElecData.from_calc_dir(write_truncated_view(system, nbands, proj, Path(tmp)))
        if norm_eigs_for_h is not None:
            edata.norm_eigs_for_h = norm_eigs_for_h
        vals = [float(np.asarray(get_ipcohp(edata, idcs1=[0], idcs2=[1], image=list(r))[1]).ravel()[-1])
                for r in system.bonds]
    msgs = sorted({str(w.message).split(".")[0] for w in caught if "Numba" not in str(w.message)})
    return float(np.mean(vals)), msgs


def crawfish_pcohp(system: System | str, nbands: int, erange: np.ndarray, std: float, proj: str = "ortho"):
    """crawfish's bond pCOHP spectrum (`get_pcohp`, default settings, Gaussian of standard deviation std in Ha) on
    erange (absolute Ha) for the lowest nb bands, averaged over the equivalent bonds."""
    from crawfish.core.elecdata import ElecData
    from crawfish.funcs.pcohp import get_pcohp

    system = SYSTEMS[system] if isinstance(system, str) else system
    erange = np.asarray(erange, np.float32)
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edata = ElecData.from_calc_dir(write_truncated_view(system, nbands, proj, Path(tmp)))
        specs = [np.asarray(get_pcohp(edata, idcs1=[0], idcs2=[1], image=list(r), erange=erange,
                                      sig=np.float32(2 * std ** 2), save_spectrum=False)[1], float).ravel()
                 for r in system.bonds]
    return np.mean(specs, axis=0)


# ----------------------------------------------------------------------------------------- LOBSTER output

LOBSTER_ORB = {"2s": 0, "2p_y": 1, "2p_z": 2, "2p_x": 3}
HARTREE_EV = 27.211386245988


def read_lobster_icohplist(path: Path) -> dict:
    """{(R, a, b): ICOHP in Ha}; a, b = orbital index on atom 1 / atom 2 (None for the atom-pair total).
    Pairs listed as atom 2 -> atom 1 are flipped (image negated) so atom 1 is always first."""
    out = {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) != 8 or not f[0].isdigit():
            continue
        (am, om), (an, on) = [(s.split("_", 1) + [None])[:2] for s in f[1:3]]
        r = tuple(int(x) for x in f[4:7])
        if am[-1] == "2" and an[-1] == "1":
            om, on, r = on, om, tuple(-x for x in r)
        out[(r, LOBSTER_ORB.get(om), LOBSTER_ORB.get(on))] = float(f[7]) / HARTREE_EV
    return out


def lobster_bond_icohp(system: System | str, basis_set: str = "custom") -> float:
    system = SYSTEMS[system] if isinstance(system, str) else system
    icohp = read_lobster_icohplist(DATA_DIR / system.name / f"lobster_{basis_set}" / "ICOHPLIST.lobster")
    return float(np.mean([icohp[(r, None, None)] for r in system.bonds]))


def lobster_bond_pcohp(system: System | str, erange: np.ndarray, std: float, basis_set: str = "custom"):
    """LOBSTER's bond pCOHP (COHPCAR.lobster: eV from its Fermi level, Gaussian `gaussianSmearingWidth`), as the
    mean over the bonds, moved to absolute Ha (LOBSTER's Fermi level is the JDFTx mu it was given) and broadened to
    a total standard deviation std (Ha) by a further Gaussian, so it can be compared with `bond_pcohp`."""
    system = SYSTEMS[system] if isinstance(system, str) else system
    d = DATA_DIR / system.name / f"lobster_{basis_set}"
    lines = (d / "COHPCAR.lobster").read_text().splitlines()
    nlabels, _, _, _, _, e_fermi = (float(x) for x in lines[1].split())
    labels = lines[2:2 + int(nlabels)]
    data = np.loadtxt(lines[2 + int(nlabels):])
    bonds = [i for i, lab in enumerate(labels) if lab.startswith("No.") and "[" not in lab]
    pcohp = data[:, [1 + 2 * i for i in bonds]].mean(1)  # per eV over eV, the same number per Ha over Ha
    std0 = float(re.search(r"gaussianSmearingWidth\s+(\S+)", (d / "lobsterin").read_text())[1]) / HARTREE_EV
    e_ha = (data[:, 0] + e_fermi) / HARTREE_EV
    de = e_ha[1] - e_ha[0]
    return gauss_spectrum(e_ha[None, :], pcohp[None, :] * de, np.asarray(erange), np.sqrt(std ** 2 - std0 ** 2))


def lobster_spilling(system: System | str, basis_set: str = "custom") -> tuple[float, float]:
    """LOBSTER's (abs. total, abs. charge) spilling in percent."""
    system = SYSTEMS[system] if isinstance(system, str) else system
    txt = (DATA_DIR / system.name / f"lobster_{basis_set}" / "lobsterout").read_text()
    return tuple(float(re.search(rf"abs\. {k}\s+spilling:\s+(\S+)%", txt)[1]) for k in ("total", "charge"))


# ----------------------------------------------------------------------------------------- plotting

# Reference categorical palette (light mode, fixed slot order); one entity keeps its color in every figure.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {
    "crawfish": "#2a78d6",
    "exact-Löwdin": "#eb6834",
    "complete frozen-window": "#1baf7a",
    "LOBSTER": "#eda100",
    "frozen-window PWF": "#e87ba4",
    "PDWF": "#008300",
    "JDFTx wannier": "#6250d6",
}


def _contrast(a, b) -> float:
    """WCAG contrast ratio of two colors."""
    from matplotlib.colors import to_rgb

    def lum(c):
        c = np.array(to_rgb(c))
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) @ [0.2126, 0.7152, 0.0722]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def ordinal_ramp(key: str, n: int) -> list:
    """n shades of one entity's color (`COLORS[key]`), light to dark, for an ordered quantity such as the band
    count. The lightest is tinted toward white only as far as it keeps 2:1 contrast with the surface."""
    from matplotlib.colors import to_rgb

    base = np.array(to_rgb(COLORS[key]))
    tint = 0.0
    while tint < 0.6 and _contrast(base + (tint + 0.02) * (1 - base), SURFACE) >= 2.0:
        tint += 0.02
    light, dark = base + tint * (1 - base), base * 0.45
    return [tuple(light + (dark - light) * x) for x in np.linspace(0, 1, n)]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)

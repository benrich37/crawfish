"""Module for common matrix operations.

Module for common matrix operations.
"""

from __future__ import annotations
import numpy as np
from numba import jit


def get_p_uvjsabc(proj_sabcju: np.ndarray, orbs_u: list[int] | None, orbs_v: list[int] | None) -> np.ndarray:
    r"""Return the projection matrix P_{uv}^{j,s,a,b,c} = <\phi_{s,a,b,c}^j | \hat{P}_{uv} | \phi_{s,a,b,c}^j>.

    Return the projection matrix P_{uv}^{j,s,a,b,c} = <\phi_{s,a,b,c}^j | \hat{P}_{uv} | \phi_{s,a,b,c}^j>.
    Evaluated at P_uv as T_u^* T_v, where T is the bandprojections vector.
    Parameters
    ----------
    proj_sabcju : np.ndarray
        The projection matrix S_{a,b,c,j,u} = <\phi_{s,a,b,c}^j | \phi_{s,a,b,c}^u>
    orbs_u : list[int] | None
        The list of orbitals to project from
    orbs_v : list[int] | None
        The list of orbitals to project to
    """
    shape = np.shape(proj_sabcju)
    nspin = shape[0]
    nka = shape[1]
    nkb = shape[2]
    nkc = shape[3]
    nbands = shape[4]
    nproj = shape[5]
    if orbs_u is None:
        orbs_u = list(range(nproj))
    if orbs_v is None:
        orbs_v = list(range(nproj))
    p_uvjsabc = np.zeros([nproj, nproj, nbands, nspin, nka, nkb, nkc], dtype=np.float32)
    orbs_u = np.asarray(orbs_u)
    orbs_v = np.asarray(orbs_v)
    p_uvjsabc = _get_p_uvjsabc_jit(proj_sabcju, p_uvjsabc, nproj, nbands, nka, nkb, nkc, nspin, orbs_u, orbs_v)
    return np.real(p_uvjsabc)


@jit(nopython=True)
def _get_p_uvjsabc_jit(proj_sabcju, p_uvjsabc, nproj, nbands, nka, nkb, nkc, nspin, orbs_u, orbs_v):
    for u in orbs_u:
        for v in orbs_v:
            for j in range(nbands):
                for a in range(nka):
                    for b in range(nkb):
                        for c in range(nkc):
                            for s in range(nspin):
                                t1 = proj_sabcju[s, a, b, c, j, u]
                                t2 = proj_sabcju[s, a, b, c, j, v]
                                p_uvjsabc[u, v, j, s, a, b, c] += np.real(np.conj(t1) * t2)
    return p_uvjsabc

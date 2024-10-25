<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/d339ce1f-b041-433c-a7c3-19204bac4061">
    <img alt="Logo" src="https://github.com/user-attachments/assets/d339ce1f-b041-433c-a7c3-19204bac4061"
height="200">
  </picture>
</h1>

Crawfish is a python library for pcohp analysis on JDFTx calculations. 

## About `crawfish`

Crawfish (originally called ultraSoftCrawfish) is a python library intended primarily for performing bonding analysis on the output of JDFTx calculations. Its reason for existing (as alluded to in the original name) is that the state-of-the-art COHP analysis software (LOBSTER) only supports calculations with PAW pseudopotentials. While the researchers of LOBSTER have shown unavoidable pitfalls when attempting cohp analysis on non-PAW calculations, cohp analysis on calculations of other pseudopotential-type calculations is far from meaningless and still provides tremendous insight. Thus the goal of crawfish is to allow access to cohp analysis for DFT users who do not use PAW pseudopotentials. While this library is intended for JDFTx, it can theoretically be used on the output of any DFT package provided the user is savvy enough to mimic JDFTx output or feed the ElecData class the required arrays (and will likely soon provide better support for arbitrary projection/eigenvalue assignments).

### ** Notation - format **
All arrays are named in the general format "<name>_<indices>", where "<name>" provides insight to the meaning of the array, and "<indices>" tells the user the array's dimensionality, and the significance of each dimension. ie for `h_uu`, "h" would signify the system hamiltonian, and "uu" would signify the array is 2-dimensional, where both dimensions correspond to atomic orbitals (meaning of each index name given below). Parts of the indices are also occasionally separated by an underscore for clarity, but are meaningless (ie `s_tj_uu` would be assumed equivalent to `s_tjuu`)

### ** Notation - index definitions **
Spin and k-points are collapsed to a single index `t`. When un-collapsed, spin is given the index `s` and steps along the first, second, and third reciprocal lattice vector are given the indices `a`, `b`, and `c`. Bands are indexed always by `j`. Orbitals are indexed by either `u` ( $\mu$ ) or `v` ( $\nu$ ) (the latter only when distinction of a second orbital index is required)

### ** Notation - array definitions **
1. `proj` is used to signify the projection vector, typically in shape `tju`. In braket notation, proj_tju[t,j,u] = $\bra{\phi_\mu}\psi_j(t)\rangle$.
2. `s` is used to signify orbital overlaps, thus will either have shape `uu` ($\bra{\phi_\mu} \phi_\nu\rangle$) or `tj_uu` ($\bra{\phi_\mu}\psi_j(t)\rangle\langle\psi_j(t)\ket{\phi_\nu}$)

### Provided analysis techniques

1. **DOS/pDOS** 

## Why and when should I use `crawfish`?

1. **Non-PAW JDFTx calculations** The intended audience for `crawfish` is anyone curious about the bondinging within a non-PAW pseudopotential calculation performed using JDFTx. While LOBSTER is not explicitly supported by JDFTx, the output of any unsupported calculation with PAW pseudopotentials can be converted by the user to mimic the output of a calculation which is supported by LOBSTER, circumventing the need of explicit support. If this is not the case, `crawfish` is here for you.
2. **General non-PAW calculations** 

## How to use `crawfish`



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
Spin and k-points are collapsed to a single index `t`, called a "state" (and `nstates` gives the total number of states for a calculation) . When un-collapsed, spin is given the index `s` (`nspin`) and steps along the first, second, and third reciprocal lattice vector are given the indices `a`, `b`, and `c` (`nka`, `nkb`, `nkc` = `kfolding`). Bands are indexed always by `j` (`nbands`). Orbitals are indexed by either `u` (`nproj`) ( $\mu$ ) or `v` ( $\nu$ ) (the latter only when distinction of a second orbital index is required)

### ** Notation - array definitions **
1. `proj` is used to signify the projection vector, typically in shape `tju`. In braket notation, proj_tju[t,j,u] = $\bra{\phi_\mu}\psi_j(t)\rangle$.
2. `e` ($\epsilon$) is used to signify the Kohn-Sham eigenvalues of the DFT calculation, and has either the shape `tju` or `sabcju`.
3. `wk` is used to signify the weights of each k-point, and has only the shape `t`
4. `occ` ($f$) is used to signify the occupation at each state (k-point + spin) and band, and thus has either shape `tj` or `sabcj`
5. `s` is used to signify orbital overlaps, thus will either have shape `uu` ($\bra{\phi_\mu} \phi_\nu\rangle$) or `tj_uu` ($\bra{\phi_\mu}\psi_j(t)\rangle\langle\psi_j(t)\ket{\phi_\nu}$)
6. `p` is used to signify orbital-overlap populations,thus will either have shape `uu` ($\bra{\phi_\mu} \hat{\rho} \ket{\phi_\nu}$) or `tj_uu` ($f_j(t)\bra{\phi_\mu}\psi_j(t)\rangle\rho_j(t)\langle\psi_j(t)\ket{\phi_\nu}$).

### Projection modifications
1. `trim_excess_bands` is a bool class variable of `ElecData` in which only `nproj` bands are included in analysis. By trimming excess bands, the projection vector `proj_tju` becomes square at each state. This has been primarily useful so far as means of allowing the projections at each state to be normalized for each band and each orbital. Theoretically, this also allows for projections to undergo a band-lowdin-orthogonalization, but the usefulness has not been investigated. Theoretically this also allows for using the dual space of the projections (allowing for less ad-hoc approaches to charge conservation), but this has yet to be implemented.
2. `los_orbs` is a bool class variable of `ElecData` in which orbitals are made orthogonal to one another via the Lowdin-Orthogonalization technique. This may seem counterintuitive in a framework centered around how orbitals interact with each other, but remember that this orthogonality ($\langle\phi_\mu|\phi_\nu\rangle=\delta_{\mu,\nu}$) does not eliminate overlap between orbitals at individual bands and states ($\langle\phi_\mu|\psi_j(t)\rangle\langle\psi_j(t)|\phi_\nu\rangle)$), only over the sum of all bands and states ($\sum_{j,t}wk(t)\langle\phi_\mu|\psi_j(t)\rangle\langle\psi_j(t)|\phi_\nu\rangle)=\delta_{\mu,\nu}$). This is an incredibly useful technique when trying to reformulate our calculation in a LCAO picture, as it ensures that for all bonding interactions (band/states `j`,`t` where $c_{\mu,j}(t)^*c_{\nu,j}(t)>0), there are enough antibonding interactions (band/states `j`,`t` where $c_{\mu,j}(t)^*c_{\nu,j}(t)<0) such that the sum over all bands and states for any orbital pair $\mu,\nu$ sums to $\delta_{\mu,\nu}$.

### Matrix modifications
1. `p_uu_consistent` is a bool class variable of `ElecData` ensuring charge conservation when building the orbital-overlap population matrix. When `True`, it will temporarily re-scale `proj` such that summing over `u` and `v` for `p_tj_uu[t,j,u,v]` equals `occ_tj[t,j]` ($\sum_{\mu,\nu}P_{u,v}(t,j)=f_j(t)$). 
2. `s_tj_uu_real` is a bool class variable of `ElecData` ensuring that orbital overlap is a real value. Since planewaves have a complex component, orbital/band projections `proj_tju` ($\bra{\phi_\mu |\psi_j(t)\rangle$) are typical complex.
3. `s_tj_uu_pos` is a bool class variable of `ElecData` ensuring that orbital overlap is a positive value. This is done by subtracting out the smallest value from the entire tensor, and then rescaling the entire tensor such the sum over all indices `tjuv` matches the original sum.

### Provided analysis techniques

1. **DOS/pDOS** 

## Why and when should I use `crawfish`?

1. **Non-PAW JDFTx calculations** The intended audience for `crawfish` is anyone curious about the bondinging within a non-PAW pseudopotential calculation performed using JDFTx. While LOBSTER is not explicitly supported by JDFTx, the output of any unsupported calculation with PAW pseudopotentials can be converted by the user to mimic the output of a calculation which is supported by LOBSTER, circumventing the need of explicit support. If this is not the case, `crawfish` is here for you.
2. **General non-PAW calculations** 

## How to use `crawfish`



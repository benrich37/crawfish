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

### Notation - format
All arrays are named in the general format "<name>_<indices>", where "<name>" provides insight to the meaning of the array, and "<indices>" tells the user the array's dimensionality, and the significance of each dimension. ie for `h_uu`, "h" would signify the system hamiltonian, and "uu" would signify the array is 2-dimensional, where both dimensions correspond to atomic orbitals (meaning of each index name given below). Parts of the indices are also occasionally separated by an underscore for clarity, but are meaningless (ie `s_tj_uu` would be assumed equivalent to `s_tjuu`)

### Notation - index definitions
Spin and k-points are collapsed to a single index `t`, called a "state" (and `nstates` gives the total number of states for a calculation) . When un-collapsed, spin is given the index `s` (`nspin`) and steps along the first, second, and third reciprocal lattice vector are given the indices `a`, `b`, and `c` (`nka`, `nkb`, `nkc` = `kfolding`). Bands are indexed always by `j` (`nbands`). Orbitals are indexed by either `u` (`nproj`) ( $\mu$ ) or `v` ( $\nu$ ) (the latter only when distinction of a second orbital index is required)

### Notation - array definitions
1. `proj` is used to signify the projection vector, typically in shape `tju`. In braket notation, proj_tju[t,j,u] = $\bra{\phi_\mu}\psi_j(t)\rangle$.
2. `e` ($\epsilon$) is used to signify the Kohn-Sham eigenvalues of the DFT calculation, and has either the shape `tju` or `sabcju`.
3. `wk` is used to signify the weights of each k-point, and has only the shape `t`
4. `occ` ($f$) is used to signify the occupation at each state (k-point + spin) and band, and thus has either shape `tj` or `sabcj`
5. `s` is used to signify orbital overlaps, thus will either have shape `uu` ($\bra{\phi_\mu} \phi_\nu\rangle$) or `tj_uu` ($\bra{\phi_\mu}\psi_j(t)\rangle\langle\psi_j(t)\ket{\phi_\nu}$)
6. `p` is used to signify orbital-overlap populations,thus will either have shape `uu` ($\bra{\phi_\mu} \hat{\rho} \ket{\phi_\nu}$) or `tj_uu` ($f_j(t)\bra{\phi_\mu}\psi_j(t)\rangle\rho_j(t)\langle\psi_j(t)\ket{\phi_\nu}$).

### Units
Unless otherwise indicated, all energies are in **Hartrees** and are not normalized to the Fermi level!!

### Projection modifications
1. `trim_excess_bands` is a bool class variable of `ElecData` in which only `nproj` bands are included in analysis. By trimming excess bands, the projection vector `proj_tju` becomes square at each state. This has been primarily useful so far as means of allowing the projections at each state to be normalized for each band and each orbital. Theoretically, this also allows for projections to undergo a band-lowdin-orthogonalization, but the usefulness has not been investigated. Theoretically this also allows for using the dual space of the projections (allowing for less ad-hoc approaches to charge conservation), but this has yet to be implemented.
2. `los_orbs` is a bool class variable of `ElecData` in which orbitals are made orthogonal to one another via the Lowdin-Orthogonalization technique. This may seem counterintuitive in a framework centered around how orbitals interact with each other, but remember that this orthogonality ($\langle\phi_\mu|\phi_\nu\rangle=\delta_{\mu,\nu}$) does not eliminate overlap between orbitals at individual bands and states ($\langle\phi_\mu|\psi_j(t)\rangle\langle\psi_j(t)|\phi_\nu\rangle)$), only over the sum of all bands and states ($\sum_{j,t}wk(t)\langle\phi_\mu|\psi_j(t)\rangle\langle\psi_j(t)|\phi_\nu\rangle)=\delta_{\mu,\nu}$). This is an incredibly useful technique when trying to reformulate our calculation in a LCAO picture, as it ensures that for all bonding interactions (bands `j` at state `t` where $c_{\mu,j}(t)^* c_{\nu,j}(t)>0$), there are enough antibonding interactions ($c_{\mu,j}(t)^* c_{\nu,j}(t)<0) such that the sum over all bands at that state for any orbital pair $\mu,\nu$ sums to $\delta_{\mu,\nu}$. The Lowdin-Orthogonalization technique is the obvious choice for this orthogonalization, as it is a simple to employ (takes 5 lines of vectorized numpy processes here) and minimizes the deviation of each projection from the true value (JDFTx will orthogonalize the orbitals if given the argument `band-projection-params yes no` prior to evaluating and dumping the band projections. However, due to the incompleteness of the space spanned by the bands at each state, this orthogonality will be lost when evaluating total overlap with the dumped projections. The same can also be realized for the bands due to the incompleteness of the space spanned by the orbitals). 

### Matrix modifications
1. `p_uu_consistent` is a bool class variable of `ElecData` ensuring charge conservation when building the orbital-overlap population matrix. When `True`, it will temporarily re-scale `proj` such that summing over `u` and `v` for `p_tj_uu[t,j,u,v]` equals `occ_tj[t,j]` ($\sum_{\mu,\nu}P_{u,v}(t,j)=f_j(t)$). 
2. `s_tj_uu_real` is a bool class variable of `ElecData` ensuring that orbital overlap is a real value. Since planewaves have a complex component, orbital/band projections `proj_tju` ($\bra{\phi_\mu}\psi_j(t)\rangle$) are typical complex.
3. `s_tj_uu_pos` is a bool class variable of `ElecData` ensuring that orbital overlap is a positive value. This is done by subtracting out the smallest value from the entire tensor, and then rescaling the entire tensor such the sum over all indices `tjuv` matches the original sum.

### Provided analysis techniques

For the following equations, projections (`proj_tju[t,j,u]`) are short-handed as $T_{\mu,j}(t)=\bra{\phi_\mu}\psi_j(t)\rangle$. Eigenvalues (`e_tj[t,j]`) are notated as $\epsilon_j(t)$, and $\delta(x)$ signifies the delta function ($\delta(x)=\infty$ for $x=0$,$\delta(x)=0$ for $x\neq0$). The following equations are evaluated on an even-space energy array `erange`. By default, gaussian smearing is employed, by which $delta(E-x)=e^{-\frac{(E-x)^2}{\sigma}}$, where $\sigma$ is an optional user parameter `sig`. If linear tetrahedron integration (`lti`) is requested, $\sum_{j,t}f(j,t)w_t\delta(E-\epsilon_j(t))$ is replaced by $\sum_{s,j}f(j,s)\int d\vec{k}\delta(E-\epsilon_j(\vec{k})$, where $\int d\vec{k}\delta(E-\epsilon_j(\vec{k})$ signifies linear tetrahedral integration performed by the `libtetrabz` package, and $s$ signifies spin.

1. **pDOS**
Projected density-of-states (pDOS) is primarily included in this package for sanity checks, and is evaluated as

$$
pDOS(E,\mu)=\sum_{j,t}|T_{\mu,j}|^2w_t\delta(E-\epsilon_j(t))
$$

2. **pCOHP**

$$
pCOHP(E,\mu,\nu)=H_{\mu,\nu}\sum_{j,t}Re\left[T_{\mu,j}^*T_{\nu,j}\right]w_t\delta(E-\epsilon_j(t))
$$

where

$$
H_{\mu,\nu} = \sum_{j,t} P_{j,t,\mu,\nu} \epsilon_{j,t}
$$

and

$$
P_{j,t\mu,\nu} = Re\left[T_{\mu,j}^*(t)T_{\nu,j}(t)\right] f_j(t) w_t
$$

Similar techniques (pCOOP and COBI) are available but not reccomended as they are currently benchmarking very poorly in this implementation.

## Why and when should I use `crawfish`?

1. **Non-PAW JDFTx calculations** The intended audience for `crawfish` is anyone curious about the bondinging within a non-PAW pseudopotential calculation performed using JDFTx. While LOBSTER is not explicitly supported by JDFTx, the output of any unsupported calculation with PAW pseudopotentials can be converted by the user to mimic the output of a calculation which is supported by LOBSTER, circumventing the need of explicit support. If this is not the case, `crawfish` is here for you.
2. **General non-PAW calculations** The techniques used by `crawfish` are made available to other DFT calculators, so long as the user is able to acquire the required data to construct an `ElecData` object. The instructions for how to do so are available in the "Creating your own `ElecData`" section of this readme. This process requires providing `crawfish` with the Kohn-Sham eigenvalues, and the projections of each Kohn-Sham wavefunction onto each orbital (as well as some other information that is typically much easier to obtain). If you are interested in doing so, please reach out to me (beri9208@colorado.edu) to help you with any obstacles that might require fixing some less-tested parts of the code. 

## How to use `crawfish`

1. **Create an `ElecData` object** `ElecData` is the class used to house all electronic data and derived tensors for a given calculation. Provided the JDFTx calculation has been run with the required settings (`band-projection-params yes no`, `dump End BandProjections` and `dump End BandEigs`), this can be done in one line as `edata = ElecData.from_calc_dir(calc_dir)`, where `ElecData` has been imported from `crawfish.core.elecdata`, and `calc_dir` is either `str` or `Path` giving the full path to your directory containing the calculation output data.
2. **Change desired settings** If there are any parameters you wish to change that effect the computed tensors required for pCOHP analysis (ie `edata.los_orbs`), you can change these values in the typical fashion (`edata.los_orbs = False`) triggering a re-evaluation of the affected tensors with this change in mind. If you have multiple settings you want to change, you can avoid repeated re-evaluations by changing the setting's private value (`edata._los_orbs = False`) and either remembering to change the final setting through the public value or by running `edata.alloc_elec_data()`.
3. **Import function(s) for desired analysis** All spectrum-generating functions for a given analysis technique "<mode>" can be imported from `crawfish.funcs.<mode>`, which will contain a dos-like spectrum-generating function `get_<mode>` and a spectrum integrating function `get_i<mode>`.
4. **Generate spectra and plot** Dos-like generating functions and spectrum integrating function will both return a length-2 tuple `erange, spectrum`, corresponding to the spectrum values and their corresponding energy-axis values. If you are unfamiliar with plotting in python, this can be easily done with `matplotlib` (the installation of this library ensures your python environment has this package) by importing `import matplotlib.pyplot as plt` and running `plt.plot(erange, spectrum)`. The arguments for these functions can all be checked in the docustrings for the function definitions.

## Creating your own `ElecData`

### Required
1. Collect (or artificially construct) the following objects
- A pymatgen `Structure` of your system. It is critical that the species in this structure are ordered by their atomic number.
- The band eigenvalues in shape `tj` as a numpy array.
- The k-point folding and the corresponding k-points. (if you are unsure but know the total number of k-points, the kfolding can be set arbitrarily and the k-points can be left as None)
- The k-point weights (if you are unsure but know there was no symmetry reduction (ie every k-point on your MK grid was evaluated) then all your k-point weights are equal to nspin/nkpts. If there was symmetry reduction of your k-mesh but know how many k-points were reduced into each of the output k-pts, multiply the weights by the number of k-points each output k-point "represents")
- The number of projections (orbitals) gathered for each atom type and their corresponding quantum numbers (exact principal quantum number n matters less, as long as you known the ordering of them for multiple shells of a given angular momentum)
- The projection coefficients for each band+state on each orbital in shape `tju`. It is critical that you have the actual projections and not their absolute values (the latter typically dumped as it is all that is needed for pDOS analysis) as taking the absolute value removes all information about the phase of the orbital at that band+state (bonding vs antibonding interaction is determined solely by the matching of phases between two orbitals). The ordering of the projections must match the ordering of the atoms as given in the Structure (ie for an all-electron calculation of an Li2 structure, projections 0-3 should correspond to Li #1's 1s, Li #1's 2s, Li #2's 1s, and Li #2's 2s)
2. Initialize an empty `ElecData` object with the class method `edata = ElecData.as_empty()` to circumvent initialization procedures for JDFTx calculations.
3. Set `user_proj_tju` (ie `edata.user_proj_tju = np.random([10,5,4])`). This will not be touched, and all projection manipulations will be performed on a copy of this array. This will automatically define `nstates`, `nbands`, and `nproj`
4. Set the `atom_orb_labels_dict` property for the class as a dictionary mapping each element in your calculation to a string representing of all the quantum numbers for the projections gathered for that element (ie for a calculation of C2H4O, set `edata.atom_orb_labels_dict = {"H": ["s"], "C": ["s", "px", "py", "pz], "O": ["s", "px", "py", "pz"]}`. If you do not wish to perform orbital resolution on your analysis, it does not matter what you put here, as long as all the elements of each list are unique and the list length matches the number of projections for that element. The ordering of the projections must match the ordering as they are listed in your `user_proj_tju`. If you have multiple projections for a given angular momentum, include the principal quantum number as well (ie `["1s", "2s"]` - they do not need to be the true principal quantum number.
5. Set `kfolding` (ie `edata.kfolding = [3,3,3]`). If you are unsure but know `nspin`, set as `[int(edata.nstates/nspin), 1, 1]`
6. Set `wk_t` (ie `edata.wk_t = np.ones(edata.nstates)*edata.nspin/np.prod(edata.kfolding)`). If you are unsure, just make sure they sum to `nspin`.
8. Set the fermi level as `edata.mu` (if you are going to set `occ_tj` explicitly, this step becomes optional but is still useful for plotting)
### Optional
9. If you have the state/band occupation, set it as `edata.occ_tj`. Otherwise, it will be calculated for your using `edata.broadening` and `edata.broadening_type`.

## Installation

### pip

Any github-url pip installation method should work, but below are the steps I have tested and know should work.

1. Clone this repo somewhere
```sh
git clone https://github.com/benrich37/crawfish.git
```
2. Activate the python environment you wish to use when performing pCOHP analysis **NOTE: At the moment, the JDFTx IO module that part of this library depends on only exists on an independent fork of pymatgen. At the time of writing this (10/24/24) this fork is fully up-to-date, but later on this installation may roll back your pymatgen to an older version.** If you are worried about dependency conflicts, I would reccomend creating a conda virtual environment with python version 3.12 (latest as of writing this)
3. Navigate to ~/crawfish/ where you cloned this repo (not ~/crawfish/src/crawfish/) and install via pip
```sh
cd ./crawfish
```
```sh
pip install .
```



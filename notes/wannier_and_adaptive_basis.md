# Beyond fixed pseudo-atomic projectors: Wannier functions and other adaptive bases for crawfish

Started 2026-10-28. Research notes only: **no crawfish code was changed** (another session is fixing bugs on
`theory_check`). Living document; add dated entries rather than rewriting results.

- Prototype scripts, JDFTx inputs and raw outputs: `~/Desktop/jdftx_hold/wannier_cohp/` (README there).
- Baseline numbers come from crawfish at `8943d71` (a `git archive` snapshot, so the other session's uncommitted
  edits were not used). The prototype reproduces crawfish's default IpCOHP exactly (section 4.2).
- Related: `notes/theory_audit.md` (F7, F10, F11 and U1 are the closest findings).

---------------------------------------------------------------------------------------------------------------

## 0. Summary and recommendation

**What crawfish does now is already a Wannier construction.** With `los_orbs=True` crawfish builds
U = A(A†A)^(-1/2) from the band projections A_ju = ⟨ψ_j|O|φ_u⟩ over *all computed bands*. These are projected
Wannier functions (Ku et al. 2002) with the outer window set to "every band in the file", no frozen (inner)
window, and no disentanglement or localization. The WOHP of Kundu et al. (2021) and pengWann (2025) uses the
same algebra as crawfish's Löwdin-basis pCOHP: Re[c*_a H_ab(R) e^{ik·R} c_b] weighted by w_k f δ(E−ε).
So moving to a Wannier basis changes how U is built, not the spectrum pipeline.

**Measured (N2 and graphene, GBRV USPP and SG15 NC; section 4):**

| property | crawfish Löwdin (today) | frozen-window PWF (JDFTx `wannier`, 0 iterations, or the same thing computed from crawfish's own data) | MLWF (JDFTx `wannier`, minimized) |
|---|---|---|---|
| occupied-state spilling | 1.0% (both systems, all bands) | **0** (exact, by construction) | 0 |
| band-energy sum rule Σ ICOHP − Σwfε | +0.11 Ha (2–4% of the band energy), grows with bands | **0** to 1e-14 | 0 |
| bond ICOHP vs. band count | not converged (N2 −0.859 → −0.892, 8 → 80 bands; graphene −0.337 → −0.529, 8 → 48) | not converged in the box continuum (N2 −0.871 → −0.910, 12 → 80); refuses to run when bands are too few (graphene < 30) instead of giving a wrong number | not tested vs. bands |
| with projectability thresholds (PDWF) | n/a | graphene **exactly converged** at 30 bands (p_out = 0.05), but the value depends on p_out by ~3% | n/a |
| atom/orbital identity | atomic (character 0.88–1.00) | atomic (0.86–1.00) | **lost for N2**: the σ manifold localizes into lone pairs + an on-atom bond function, and the N–N ICOHP drops from −0.92 to −0.36 Ha. Fine for graphene. |
| equivalent bonds equal | yes | yes | spread 0.003 Ha unpinned (graphene) |

**Verdict.**
1. **Yes, a JDFTx `wannier` run can give better-founded COHP than the current workflow,** when it is run as
   *projection-only with a frozen inner window* (`addAtomicOrbitals yes`, `innerWindow` covering the occupied
   states, `wannier-minimize nIterations 0`, optionally `projectionThresholds`). That gives zero spilling, an exact
   sum rule, atom-centred orbitals with the same labels as `bandProjections`, and an explicit error instead of
   silent garbage when the antibonding directions are missing. Bond ICOHPs move by 1–4% relative to crawfish's
   current basis (graphene −0.529 → −0.535; N2 −0.892 → −0.927).
2. **The same basis can be built inside crawfish from data it already reads** (projections + eigenvalues), with
   no new JDFTx run. For norm-conserving pseudopotentials it reproduces JDFTx's `wannier` Hamiltonian to 2e-6 Ha
   (section 4.2). This is the cheapest high-value step: one new basis option next to `los_projs_for_orbs`
   (section 7, phase A).
3. **Do not use maximally-localized (spread-minimized) Wannier functions for orbital-resolved COHP by default.**
   Spread minimization rotates bonding pairs into single bond orbitals when the antibonding partner is missing
   or delocalized (N2's σ*), and that moves bonding into "on-site" terms. Pinning the centres
   (`pinAtomicOrbitals yes`) does not prevent it.
4. **No basis removes the residual band-count dependence of the *bond* ICOHP.** Section 5.1 explains why. The
   virtual (antibonding) part of the basis cancels in the total but moves weight between on-site and
   inter-site pairs, so whatever fixes the antibonding closure (band count, window, threshold, localization)
   shifts bond ICOHP by a few %. LOBSTER has the same dependence through NBANDS. Projectability thresholds make
   the choice explicit and reproducible. The only fully band-independent route needs the exact atomic-orbital
   matrices S(k) = ⟨φ|O|φ⟩ and H(k) = ⟨φ|H|φ⟩ (section 6.2). S(k) can already be recovered for norm-conserving
   runs (section 4.8); H(k) needs a small upstream JDFTx dump.
5. **Overcomplete molecular-style (Gaussian) basis from `.wfns`:** feasible for norm-conserving pseudopotentials
   (JDFTx has `dump End Gvectors`), but it is more work and less useful for COHP than the frozen-window route.
   Large non-orthogonal bases make pair populations basis-set dependent (the Mulliken problem), and the
   basis must be fitted to *pseudo*-atoms, not taken from all-electron libraries. Its real value is as a
   reference space for building environment-adapted *minimal* orbitals (NAO, IAO or QUAMBO) when the
   pseudopotential lacks channels (no d or polarization PAOs). Section 6.3.

**Side finding (relevant to the theory audit, F7 and U1):** for ultrasoft pseudopotentials, JDFTx's
*non-orthogonal* projections (`band-projection-params no no`) use a single-centre `OpsiRadial`, while the ortho
projections and the `wannier` code apply the full O operator, including neighbouring atoms' augmentation.
The two dumps of the same state are therefore not related by S^(-1/2) (projection residual 3e-3 in N2, 4e-2 in
graphene; exact, 1e-7, for norm-conserving). crawfish's USPP pCOOP path uses these non-orthogonal
projections. Section 4.8.

---------------------------------------------------------------------------------------------------------------

## 1. The problem, stated in crawfish's terms

Two separate deficiencies of the fixed pseudo-atomic orbital (PAO) basis from the pseudopotential file:

1. **Accuracy / spilling.** The PAOs are free-atom functions. The occupied Kohn–Sham states are not exactly in
   their span. The *exact* occupied-state charge spilling of the fixed PAOs (from the ortho projections, which
   are exact per band and independent of the band count) is **2.07% for N2 and 1.08% for graphene** (GBRV USPP,
   this work). That is small for these covalent p-block systems. It should be larger for cations, transition
   metals, semicore and metals (untested; see section 8).
2. **Completeness / conditioning.** crawfish never has the PAOs themselves, only their projections onto the
   computed bands. The band-projected orbitals χ_u = Σ_j ψ_j A_ju become φ_u only as the band count goes to
   infinity. That is the slow band convergence of F11 (graphene IpCOHP −36% at 8 bands, −0.9% at 30). Even with
   80 bands N2's s and bond-axis p are only 82% and 73% represented (σ* is spread over the box continuum,
   every state there has projectability < 0.03).

crawfish's Löwdin step makes the band-projected orbitals orthonormal *within the band space*. This is the
Marzari et al. RMP "projection + Löwdin" initial guess, A(A†A)^(-1/2) (already noted in theory_audit.md
references), with the outer window equal to all computed bands and no frozen window. Consequences:

- occupied states leak out of the basis. crawfish's own spilling (1 − Σ_occ |U_ju|²) is 1.0% at 80 and 48
  bands, *increasing* with band count (N2: 0.06% at 8 bands, 1.0% at 80), because the basis drifts towards the
  exact PAOs, which spill 2%;
- the band-energy sum rule misses by +0.106 Ha (N2) and +0.109 Ha (graphene): 1.9% and 3.8% of the band energy;
- nothing tells crawfish which empty bands are "antibonding partners" and which are vacuum or box states. They
  all enter in proportion to their projection.

---------------------------------------------------------------------------------------------------------------

## 2. Literature survey

Verification status: **[read]** = text fetched and read this session (via a fetch agent); **[search]** = seen
in search-result abstracts or metadata only; **[memory]** = from memory, not re-checked: verify before quoting.

### 2.1 Wannier-based COHP (the direct analogue)

- **WOOP/WOHP: Kundu, Bhattacharjee, Lee, Jain**, "Population analysis with Wannier orbitals", J. Chem. Phys.
  154, 104111 (2021), doi:10.1063/5.0032605; arXiv:2009.01130 **[read arXiv]**.
  - Equations (arXiv v1): WOHP_{μR,νR'}(E) = −H_{μR,νR'} Σ_{j,k} C*_{μR,j}(k) C_{νR',j}(k) δ(ε_j(k) − E), with
    C from the inverse of the Wannier rotation U^k. WOOP is the same with S, which is δ for orthonormal WFs. So
    off-diagonal WOOP is zero (same as crawfish F7) and diagonal WOOP is a pDOS.
  - Spilling is "strictly zero for insulators", and small for metals when more bands enter disentanglement.
  - QE + Wannier90 (with an undescribed modified version for WOOP/WOHP); systems: diamond, GaAs, hcp Ti, (5,5) CNT,
    LiNi0.5Mn1.5O4. No quantitative LOBSTER comparison. Uses local axes for CNT and LNMO because global-axis
    orbitals gave "qualitatively incorrect" or unequal populations.
  - Whether they ran spread minimization or projection-only is not stated.
  - Sign convention: positive WOHP = bonding (opposite to crawfish's IpCOHP sign).
- **pengWann: P. J. Taylor, B. J. Morgan**, "pengWann: Descriptors of chemical bonding from Wannier functions",
  JOSS 10(108), 7890 (2025), doi:10.21105/joss.07890 **[read docs and paper.md]**.
  - Reads Wannier90 `.eig`, `_u.mat`, `_hr.dat`, `.wout`, `_centres.xyz`. Computes WOHP, WOBI, pDOS, populations
    and charges; k- and orbital-resolved; vectorized (Rust extension).
  - C^α_{nk} = exp[ik·R] (U^k_{nα})*; WOHP^R_{αβ}(E) = −H^R_{αβ} Σ_{nk} Re[(C^α_{nk})* C^β_{nk}] δ(E − ε_nk);
    WOBI uses the Wannier density matrix P^R.
  - Assigns each Wannier centre to the nearest atom; atom pairs selected by distance cutoffs; requires
    `translate_home_cell = false` in Wannier90 for correct R.
  - Footnote in the paper: LOBSTER spilling 9.8e-3 (diamond) and 9.5e-3 (rutile) vs. < 1e-13 with pengWann "for
    the same set of Bloch states". No COHP-number comparison with LOBSTER.
  - Explicit caveat: Wannier functions are "strongly non-unique"; MLWFs "tend to be atom-centred".
  - `_u_dis.mat` (disentanglement) is not mentioned in the docs I read; whether it handles entangled bands
    correctly was not checked.
- **WOBSTER** (C. Xiao, github.com/Chengcheng-Xiao/WOBSTER): WOHP and pDOS from Wannier90, loop-based. Cited by
  pengWann **[read, via pengWann]**; not inspected.
- Earlier: Bhattacharjee & Waghmare, PCCP 12, 1564 (2010), Wannier orbital-based population analysis **[read, via
  Kundu]**.

### 2.2 Choosing the Wannier functions (what matters for COHP)

- **Disentanglement**: Souza, Marzari, Vanderbilt, PRB 65, 035109 (2001) **[read, via pengWann]**: inner
  (frozen) window kept exactly, outer window supplies the remaining directions chosen by minimizing Ω_I.
- **MLWF**: Marzari & Vanderbilt, PRB 56, 12847 (1997); review: Marzari et al., RMP 84, 1419 (2012).
- **PDWF: Qiao, Pizzi, Marzari**, "Projectability disentanglement for accurate and automated electronic-structure
  Hamiltonians", npj Comput. Mater. 9, 208 (2023), doi:10.1038/s41524-023-01146-w **[read arXiv v2 + npj
  part]**. The method closest to what crawfish needs:
  - projectors = the PAOs from the pseudopotential, Bloch-summed and **Löwdin-orthonormalized**; projectability
    p_mk = Σ_n |⟨g_n|ψ_mk⟩|² (≤ 1);
  - p < p_min → discarded; p ≥ p_max → frozen; in between → disentangled. Defaults p_max = 0.95, p_min = 0.01
    (graphene used 0.85 and 0.02). They also recommend freezing everything below E_F (metals) or the CBM
    (insulators) **plus ~2 eV**;
  - then standard disentanglement and **unconstrained** maximal localization: "there is no constraint on the WF
    centers";
  - atom-centring: 82% of WFs atom-centred (d_NN ≤ 0.1 Å) after disentanglement, before localization, but only
    30% of final MLWFs. Graphene ends up as p_z, p_x, s±p_y hybrids;
  - success: η2 ≤ 20 meV band interpolation for 96.5% of 200 materials (SCDM 91.5%) and 97.8% of 21,737;
  - failure modes: p_max/p_min not universal (η non-monotonic); Si CBM p ≈ 0.83 with s+p only (0.99 with d PAOs
    added); PAOs with the wrong node count project onto semicore;
  - they note "the WFs lack the anti-bonding part of the bonding/anti-bonding closure" (citing Thygesen et al.
    2005 [memory: PRL 94, 026405, partly occupied Wannier functions]). This is the same issue as section 5.1;
  - extension to magnetism and SOC: Jiang, Qiao, Paulish, Zhao, Marzari, Pizzi, npj Comput. Mater. 11, 353 (2025)
    **[search]**.
- **SCDM**: Damle, Lin, Ying, JCTC 11, 1463 (2015); J. Comp. Phys. 334, 1 (2017) **[read, via pengWann]**.
  Automatic, but WFs are not atom-centred (14% atom-centred before localization per Qiao 2023), so it is a poor
  fit for orbital-resolved COHP.
- **Constrained centres (SLWF)**: Wang, Lazar, Park, Millis, Marianetti, PRB 90, 165125 (2014) **[memory]**;
  Wannier90 `slwf_*`. Qiao 2023 mentions them as out of scope, "at the cost of potentially having worse
  interpolators".
- "Manifold-remixed Wannier functions" (separating bonding and antibonding manifolds): not found in Qiao 2023;
  **unverified**, don't cite yet.

### 2.3 Environment-adapted *minimal* atomic bases from plane waves

These are the non-Wannier ways to get "AOs that exactly span the occupied space", the other half of the user's
request.

- **QUAMBO**: Lu, Wang, Yao, Ho et al., PRB 70, 041101 (2004) **[search; author list not verified]**: a minimal
  set of quasi-atomic orbitals that span exactly the occupied subspace of a converged large-basis calculation,
  used for bonding analysis. Follow-up on tight-binding parameters: PRB 72, 205123 (2005) **[search]**.
- **QO**: Qian, Li, Qi, Wang, Chan, Yao, Ho, Yip, "Quasiatomic orbitals for ab initio tight-binding analysis",
  PRB 78, 245112 (2008) **[search; volume/page from memory]**. Non-orthogonal pseudo-angular-momentum orbitals
  from PW-DFT that reproduce all electronic structure below an energy threshold, with Mulliken charge and bond
  order "which satisfy sum rules". Construction: occupied (or below-threshold) states fully, plus the unoccupied
  combinations that maximize similarity to the free-atom orbitals. **This is the same criterion as JDFTx's
  projection-only disentanglement and the frozen-window PWF here** (the best-projecting partial-band
  combinations). QO = the non-orthogonal version, FW-PWF = its Löwdin orthonormalization.
- **IAO/IBO**: Knizia, JCTC 9, 4834 (2013) **[memory]**: intrinsic atomic orbitals, built from the occupied
  space and a free-atom minimal basis, exactly span the occupied space with no virtual-band input. Periodic, PW:
  Wöckinger, Rumpf, Schäfer, JCTC 21, 10515 (2025) **[search]** (IBOs in VASP plus a standalone package).
  The IAO construction needs the *exact* free-atom AO overlaps (and, for COHP, ⟨φ|H|φ⟩): section 6.2.
- **PAOFLOW**: Agapito, Curtarolo, Buongiorno Nardelli, PRB 88, 165127 (2013) and later **[memory]**:
  projections onto the pseudopotential's PAOs with projectability filtering and a shifted TB Hamiltonian. Same
  ingredients as crawfish; it filters bands rather than adapting the basis.

### 2.4 Projection onto an extended / overcomplete localized basis

- **LOBSTER** (Maintz, Deringer, Tchougréeff, Dronskowski, JCC 34, 2557 (2013); JCC 37, 1030 (2016); Nelson et
  al., JCC 41, 1931 (2020), doi:10.1002/jcc.26353) **[memory; cohp.de failed TLS and the publisher returned 403
  this session]**. PAW reconstruction, then projection onto Slater-type local basis sets (pbeVaspFit2015, Koga,
  Bunge), with charge spilling reported. A QE-users mailing-list post (Feb 2026) says LOBSTER 5.1.1 supports only
  PAW potentials **[search]**, which agrees with the README's premise; not confirmed from LOBSTER's own docs.
- **Periodic NBO / projection code**: Dunnington & Schmidt, JCTC 8, 1902 (2012) **[search]**. Projects plane-wave
  bands (VASP PAW, `LNBO`) onto a user-specified Gaussian (Gaussian94-format) basis, reports spillage, and feeds
  periodic NBO analysis. Code and guide: github.com/yohsama/periodic-NBO (mirror) **[search]**. Same projection
  for SSAdNDP: Galeev, Dunnington, Schmidt, Boldyrev, PCCP 15, 5022 (2013) **[search; page from memory]**, which
  adds overlap corrections for the PAW augmenters.
- **Spillage-optimized basis sets**: Sánchez-Portal, Artacho, Soler, Solid State Commun. 95, 685 (1995) and
  J. Phys.: Condens. Matter 8, 3859 (1996) **[search]**; Chen, Guo, He, J. Phys.: Condens. Matter 22, 445501 (2010)
  (ABACUS numerical atomic orbitals fitted to PW dimers by minimizing spillage) **[search]**. These give
  pseudopotential-consistent extended bases (multiple-ζ and polarization) that would be the right basis family for
  section 6.3.

---------------------------------------------------------------------------------------------------------------

## 3. What JDFTx's `wannier` executable can do (verified in source, `../jdftx/jdftx/wannier/`, at `a37d055f`)

Executable: `~/Desktop/jdftx_hold/build_master/wannier` (built, works: 0.7–5 s for the runs below).
Input = the total-energy input plus `wannier`, `wannier-center*`, `wannier-initial-state`,
`wannier-dump-name`, `wannier-minimize`. The total-energy run must `dump End State BandEigs`.

Relevant `wannier` keys (`wannier/commands.cpp`):

- `addAtomicOrbitals yes`: one trial orbital per PAO of every atom, so no `wannier-center` lines are needed.
  `Wannier::setup` orders them species → atom → l → n → m (→ spin), **the same order as `bandProjections`**, so
  crawfish's orbital labels carry over. `ignoreSemiCore yes` (default) drops inner shells of each l and
  excludes that many lowest bands. `ignorePtillGroup` drops p shells of s/d-block metals.
- `pinAtomicOrbitals yes`: adds the "pinned" centre constraint (a penalty towards the atom position).
- `innerWindow eMin eMax`, `outerWindow eMin eMax` (absolute Hartree, not relative to μ). Without
  `outerWindow`, a fixed band range `bStart .. bStart+nCenters` is used.
- `projectionThresholds pOuter pInner`: band projectability onto the *orthonormalized* trial orbitals
  (`CdagG * orthoMatrix(trial ^ O(trial))`). In `initRotations`: withinOuter = (energy in outer window) **and**
  p > pOuter; withinInner = (energy in inner window) **or** p > pInner. So the PDWF scheme is built in.
- Trial orbitals can also be `Gaussian x0 x1 x2 [sigma] [orbDesc]` (any l, m) or `Numerical` (read from a file in
  the `mlwfC` format). This is the way to add polarization or diffuse functions in the Wannier route (section 6.3).
- `wannier-minimize nIterations 0` → projection-only (no Ω_I or spread minimization).
- `frozenCenters`, `loadRotations`, `saveWfns`, `saveWfnsRealSpace`, `saveMomenta`, `saveR`, `spinMode`, etc. are not
  needed here.

Algorithm with `nIterations 0` (`WannierMinimizer_init.cpp::initRotations`), per k:
1. `CdagG = ⟨ψ|O|trial⟩`, with trial = `lPhase` × PAO, `lPhase = i^l`. Since JDFTx's PAOs are i^l × real
   orbitals (theory_audit F1), the trial orbitals are **(−1)^l × the real orbitals**: p-type Wannier functions
   carry the opposite sign to crawfish's real orbitals. COHP is invariant to this; pCOMO is not.
2. Bands are sorted into full (inner), partial (outer only) and excluded.
3. The trial directions not covered by the full bands (null space of `CdagG[full]`) are projected onto the partial
   bands, and the Löwdin-orthonormalized result fills the remaining nW − nFull columns of U1 (the "best
   projecting" combinations: QO/QUAMBO criterion).
4. U2 = Löwdin of `U1† CdagG` (projection of the trial orbitals into the chosen subspace). U = U1·U2.

Outputs (`WannierMinimizer_save.cpp`; filenames `<dump-name with $VAR>`):

- `mlwfH`: H(R) = Σ_k w_k e^{−2πik·R} U†εU for the kfold-product "unique cells" (real part only for collinear
  spin; "Relative discarded imaginary part" is printed, 1e-5 to 1e-7 here). Expand with `mlwfCellMap` (Wigner–Seitz
  cells, text) and `mlwfCellWeights` (binary doubles) as in the tutorial (`doc/tutorials/wannier/
  SeparatedBands.dox`): `Hwannier = W * Hreduced[mod(cellMap,kfold)·kStride]`, H(k) = Σ_cells e^{2πik·R} H(R).
  Same Bloch convention as crawfish's M(R) (section 2.3 of the audit), so R-resolution maps directly.
- `mlwfU`: the rotations U(k) (nBands × nCenters per k + pinned centres). **Written only if nIterations > 0**
  ("re-save only if any minimization has occured"). For projection-only runs, recover coefficients by
  diagonalizing H(k) instead (section 4.1).
- `mlwfBandContrib`, `mlwfBandRanges`; centres and spreads are printed in the `.out` ("Centers in ... coords").
- `Up`/`Dn` suffixes for z-spin.

---------------------------------------------------------------------------------------------------------------

## 4. Prototype: crawfish basis vs. Wannier-type bases on N2 and graphene

### 4.1 Setup

All in `~/Desktop/jdftx_hold/wannier_cohp/` (JDFTx `build_master`, `-c 8`):

| system | details | SCF time |
|---|---|---|
| `N2` | crawfish fixture geometry (24.6 bohr box), GBRV v1.2 USPP, PBE, no-spin, **Γ only**, 80 bands, 20 Ha | 22 s |
| `graphene` | fixture geometry, GBRV USPP, no-spin, 12×12×1 full mesh (`symmetries none`), 48 bands, Fermi 0.005 | 47 s |
| `N2_nc` | as N2 but SG15 ONCV norm-conserving, 30 Ha, 40 bands | 28 s |

Each has `totalE` (SCF, ortho projections), `nonortho` and `orthodo` (`dump-only` re-dumps of the same state
with `band-projection-params no no` / `yes no`), and `wannier` runs (≤ 5 s each). Fewer-band results are the
lowest N bands of the many-band run. That emulates a smaller `elec-n-bands`, as long as the cut doesn't split
a degenerate set.

Bases compared (all orthonormal; band j's coefficient on w_u is conj(U_ju)):

- **crawfish-Löwdin**: U = A(A†A)^(-1/2) from the ortho projections: crawfish's default (`los_orbs=True`).
- **FW-PWF** (frozen-window projected Wannier functions): `wcohp.basis_frozen_pwf`, a numpy port of JDFTx
  `initRotations` with nIterations 0, run on the *dumped* projections. Inner window: N2 e ≤ −0.06 Ha (5 occupied
  + bound π*, 7 bands) or e ≤ −0.09 (occupied only); graphene e ≤ μ + 0.05 Ha (4–5 bands per k).
- **PD**: FW-PWF plus projectability thresholds (outer set restricted to p > p_out; optionally p > p_in frozen).
- **JDFTx `wannier`**: `wproj_*` (nIterations 0), `wmlwf_pin` (`pinAtomicOrbitals yes`, minimized),
  `wmlwf_free` (minimized, unconstrained); outer window [−2, 10] Ha = all bands. COHP from H(k) interpolated on
  the DFT mesh and diagonalized (eigenvectors = coefficients, Fermi occupations at the DFT μ and smearing).

ICOHP is Σ_k w_k Σ_j f_j Re[U_ja H_ab(R) e^{2πik·R} U*_jb] over a ∈ atom A, b ∈ atom B: crawfish's
`image=R` formula, negative = bonding. Graphene bonds: R = (0,0,0), (−1,0,0), (0,1,0) from C1 to C2.

### 4.2 Validation

- `check_baseline.py`: the prototype's crawfish-Löwdin ICOHP equals crawfish `get_ipcohp` (HEAD snapshot) to
  all printed digits: N2 −0.89178; graphene −0.52876 for each of the three bonds.
- NC N2: FW-PWF from the dumped non-orthogonal projections reproduces JDFTx `wannier` (`wproj_occpi`) H to
  **2.1e-6 Ha**. So the crawfish-side construction is the same thing as the JDFTx run.
- USPP N2: they differ by 1.3e-2 Ha (in the σ block only; the π block matches exactly). Löwdin is invariant to a
  common scale but not to the s:p scale, and rescaling the s trial orbital by 1.077 cuts the difference to
  2e-3. Cause: the non-orthogonal dump uses a single-centre `OpsiRadial` for O·φ, while `wannier` applies the
  full O (section 4.8).

### 4.3 Full band set

N2 (80 bands; DFT band energy −5.5233 Ha; ICOHP N1–N2 in Ha, orbital decomposition along the bond axis x):

| basis | ICOHP | s–s | s–px (both) | px–px | π (py+pz) | spill_occ | sum rule |
|---|---|---|---|---|---|---|---|
| crawfish-Löwdin (ortho proj) | −0.8918 | −0.116 | −0.297 | −0.125 | −0.354 | 1.0e-2 | +0.106 |
| Löwdin of non-ortho proj | −0.9111 | −0.160 | −0.307 | −0.090 | −0.354 | | |
| FW-PWF (non-ortho, inner ≤ −0.06) | −0.9269 | −0.160 | −0.307 | −0.099 | −0.361 | 0 | 0 |
| JDFTx wproj_occpi | −0.9223 | −0.149 | −0.306 | −0.107 | −0.361 | 0 | 0 |
| JDFTx wproj_occ | −0.9259 | −0.149 | −0.306 | −0.107 | −0.364 | 0 | 0 |
| JDFTx wmlwf_pin | **−0.3637** | −0.000 | −0.002 | −0.001 | −0.361 | 0 | 0 |
| JDFTx wmlwf_free | **−0.3633** | −0.086 | −0.000 | −0.001 | −0.174 | 0 | 0 |

NC N2 (40 bands): crawfish −0.8762 (spill 0.73%), FW-PWF −0.9027 (spill 0).

Graphene (48 bands; band energy −2.8842 Ha; mean of the three C1–C2 bonds):

| basis | ICOHP | spread (3 bonds) | pz–pz (π) | σ (rest) |
|---|---|---|---|---|
| crawfish-Löwdin | −0.5288 | 0 | −0.0566 | −0.4721 |
| Löwdin of non-ortho proj | −0.5209 | 0 | −0.0567 | −0.4642 |
| FW-PWF (non-ortho, ≤ μ+0.05) | −0.5351 | 0 | −0.0593 | −0.4758 |
| JDFTx wproj_mu05 | −0.5423 | 0 | −0.0591 | −0.4832 |
| JDFTx wmlwf_pin | −0.5193 | 1e-5 | −0.0596 | −0.4597 |
| JDFTx wmlwf_free | −0.5186 | 0.0026 | −0.0596 | −0.4589 |

All Wannier variants reproduce N_e and the DFT band energy to 1e-6.

### 4.4 Band-count series (lowest N bands of the same run)

N2, bond ICOHP (Ha):

| bands | 8 | 10 | 12 | 15 | 20 | 30 | 40 | 60 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| crawfish-Löwdin | −0.8588 | −0.8562 | −0.8668 | −0.8655 | −0.8701 | −0.8744 | −0.8733 | −0.8841 | −0.8918 |
| FW-PWF (ortho proj) | −0.8548 | −0.8701 | −0.8713 | −0.8713 | −0.8784 | −0.8833 | −0.8870 | −0.9005 | −0.9098 |
| FW-PWF (non-ortho proj) | fails | −0.8911 | −0.8913 | −0.8913 | −0.8975 | −0.9021 | −0.9054 | −0.9180 | −0.9269 |
| PD p_out = 0.01 | fails | fails | −0.8913 | −0.8913 | −0.8975 | −0.9021 | −0.9021 | −0.9147 | −0.9236 |
| crawfish spill_occ | 0.06% | 0.22% | 0.28% | 0.34% | 0.49% | 0.53% | 0.75% | 0.89% | 1.0% |

FW-PWF spilling and sum-rule residual are ≤ 3e-12 at every band count (both windows; occupied-only and
occupied + π* windows agree to ≤ 0.004 Ha).

Graphene, mean bond ICOHP (Ha):

| bands | 8 | 10 | 12 | 16 | 20 | 30 | 40 | 48 |
|---|---|---|---|---|---|---|---|---|
| crawfish-Löwdin | −0.3369 | −0.3876 | −0.4214 | −0.4814 | −0.5013 | −0.5193 | −0.5239 | −0.5288 |
| FW-PWF (non-ortho) | fails | fails | fails | fails | fails | −0.5249 | −0.5301 | −0.5351 |
| FW-PWF (ortho) | fails | fails | fails | fails | fails | −0.5317 | −0.5370 | −0.5423 |
| PD p_out = 0.01 | fails | fails | fails | fails | fails | −0.5228 | −0.5255 | −0.5286 |
| PD p_out = 0.05 | fails | fails | fails | fails | fails | **−0.5181** | **−0.5181** | **−0.5181** |
| PD 0.01 / p_in 0.95 | fails | fails | fails | fails | fails | −0.5222 | −0.5248 | −0.5279 |

"fails" = at some k the partial bands cannot supply the missing trial directions ("trial orbitals do not cover
the subspace": the in-plane σ* combination lies above the band set at Γ, at 16 and 20 bands only at a few k).
That is exactly the condition F11 warns about, where crawfish returns numbers 4–36% off. A frozen-window basis
turns it into a hard error, which is the behaviour one wants.

### 4.5 What the projectability thresholds do

With p_out = 0.05 graphene's result is independent of the band count once every band with p > 0.05 is
included (30 bands here). The trial directions' tails in vacuum states no longer leak in. But the converged value
depends on the threshold: −0.5181 (0.05), −0.5286 at 48 bands (0.01), −0.5351 (none). That is ~3% for a
user-chosen parameter. N2's σ* has no band with p > 0.05 (the box continuum only), so p_out = 0.05 fails for
every band count. For a molecule in a box the σ* direction is intrinsically ill-defined.

### 4.6 Why MLWFs fail for N2 COHP (and not for graphene)

Centres and spreads from `wmlwf_*.out` (N atoms at x = 13.896 and 11.832 bohr):

- unpinned: three hybrids per atom at ~2.27 bohr², one function at x = 12.86 (the bond midpoint, 1.30 bohr²: the
  σ bond orbital), and one at x = 18.45 (4.5 bohr outside the molecule, 12.4 bohr²: a diffuse continuum σ*
  combination);
- pinned: centres stay within ~0.7 bohr of the atoms, but N1's "s" function moves 0.66 bohr outwards (a lone
  pair). N2's "s" and "px" sit between N2 and the bond centre (the σ bond). N1's "px" is a diffuse 22.5 bohr²
  function.

The σ bond ends up in a function counted as one atom's orbital, so it becomes an on-site term and the N–N
σ ICOHP goes to ~0. π survives because both π and π* are bound and inside the inner window. In graphene σ* is
present in the outer window with high projectability, so the hybrids keep bonding partners on the other atom
and ICOHP stays within 2% of the projection-only result. **Rule: spread minimization is only safe for COHP when
the antibonding partners of every bond are available with high projectability.** Check the centres and an
atomic-character measure before trusting MLWF COHP.

### 4.7 Atomic character

k-averaged |⟨w_u|O|φ̃_u⟩|² / ⟨φ̃_u|P_bands|φ̃_u⟩ with φ̃ the orthonormalized PAOs (1 = same function within the
band space); atom 1, orbitals s, py, pz, px (`results/atomic_character.txt`):

| | s | py | pz | px |
|---|---|---|---|---|
| N2 crawfish-Löwdin | 0.913 | 1.000 | 1.000 | 0.884 |
| N2 FW-PWF | 0.872 | 0.985 | 0.985 | 0.904 |
| N2 PD p_out = 0.01 | 0.860 | 0.985 | 0.985 | 0.890 |
| graphene crawfish-Löwdin | 1.000 | 1.000 | 1.000 | 1.000 |
| graphene FW-PWF | 0.996 | 0.993 | 0.993 | 0.993 |
| graphene PD p_out = 0.01 | 0.988 | 0.987 | 0.984 | 0.987 |

The adapted orbitals stay recognizably atomic (≥ 0.86). The adaptation lives mostly in the σ/σ* channel.

### 4.8 Exact AO overlap S(k) from two projection dumps, and the USPP inconsistency

JDFTx's ortho branch is `proj_o = invsqrt(psi^Opsi) * (Opsi^C)` and the non-ortho branch is
`getAtomicOrbitals(q, true)^C` (`DumpExcitationsMoments.cpp::dumpProjections`). If both used the same O·φ,
proj_o = S^(-1/2) proj_n per k, and S(k) = ⟨φ|O|φ⟩ (exact, independent of the band count: the overlap theory_audit
U1 wanted dumped) would follow from one least-squares fit per k.

- **Norm-conserving (N2_nc): works.** The linear-map residual is 1.4e-7 (print precision) and M is Hermitian to
  1e-6. diag S = 1.000; S(s1,s2) = 0.474, S(px1,px2) = −0.173 (ppσ negative, as expected). So for NC runs,
  crawfish can get the exact AO overlap, hence exact pCOOP (F7) and exact charge spilling, from one extra
  `dump-only` pass, with no JDFTx change. Both dumps must come from the same state (a `dump-only` re-dump
  reproduces the SCF projections bit for bit).
- **USPP: does not.** Residual 2.8e-3 (N2) and 3.9e-2 (graphene) on bands beyond the first nproj. The ortho
  branch applies the full operator O (all atoms' augmentation), while `getAtomicOrbitals(q, true)` →
  `SpeciesInfo::setAtomicOrbitals(..., applyO=true)` uses the precomputed single-centre radial function
  `OpsiRadial` (`SpeciesInfo_atomic.cpp`), which omits the augmentation of neighbouring atoms' projectors.
  (My reading of the source; consistent with the NC case being exact.)
  **Implication for crawfish:** USPP pCOOP from `band-projection-params no no` (the F7 recipe) and crawfish's
  `los_orbs=False` path use ⟨Õφ|ψ⟩ with an approximate O. Worth an entry in theory_audit.md (not added here,
  to stay out of the other session's files).
- The gauge-invariant alternative (solve M̄ X_n M̄ = X_o with X = Σ_j T T† over all bands) gives a Hermitian
  positive S, but for USPP it fails the per-band check at the 1e-3 level for the same reason.

---------------------------------------------------------------------------------------------------------------

## 5. Analysis

### 5.1 Why bond ICOHP depends on the antibonding part of any basis

For an orthonormal basis {w} whose span contains every occupied state, write H_ab = H^occ_ab + H^virt_ab
(contributions from occupied and unoccupied components of the w's). For occupied ψ_j,
Σ_ab c*_ja H^virt_ab c_jb = ⟨ψ_j|P_W H_virt P_W|ψ_j⟩ = 0, so the virtual part **cancels in the total** (that is
why the sum rule is exact for every frozen-window basis) **but not pair by pair**. It moves ICOHP between on-site
(aa') and inter-site (ab) pairs. So bond ICOHP depends on the virtual components of the basis functions, which
are set by whatever fixes the antibonding closure: band count (crawfish today, LOBSTER's NBANDS), outer window and
thresholds (PDWF), or localization (MLWF). The compact PAOs have high-kinetic-energy tails, so tiny admixtures of
high bands (ε of order 1 Ha) shift H_ab noticeably. Graphene at 48 bands has completeness 0.993–0.999 and still
drifts ~1% per 8 bands. None of the band-space constructions can remove this. Section 6.2 can.

### 5.2 Does the Wannier route give "better" results? Criterion by criterion

| criterion | crawfish-Löwdin | FW-PWF / JDFTx projection-only | PD thresholds | MLWF |
|---|---|---|---|---|
| zero occupied spilling, exact sum rule | no (1%, 0.1 Ha) | **yes** | **yes** | yes |
| atom-centred, orbital labels kept | yes | **yes** | yes | no (N2) / mostly (graphene) |
| fails loudly when antibonding directions are missing | no (warning only, since F11) | **yes (error)** | yes | — |
| converged in band count | no | no (better-defined) | **yes for solids** once p_out > tail | untested |
| free parameters | band count | inner window (robust: ≤ 0.004 Ha here) | p_out (3% effect) | many; gauge-dependent |
| cost | none | none (from existing data) or ~1 s JDFTx | none | seconds; needs `wfns` |
| fine k-mesh / LTI via interpolation | no | yes (from JDFTx `mlwfH`) | yes | yes |

The physical picture (which interactions bond, ppσ vs. sp etc.) is the same in every projection-based basis.
Differences are 1–4% in bond ICOHP and up to ~35% in individual s–s or px–px terms in N2 (−0.116 vs. −0.160 Ha).
The individual orbital terms are the most basis-sensitive numbers; the atom-pair totals are much more robust.

---------------------------------------------------------------------------------------------------------------

## 6. Other strategies (beyond the Wannier route)

### 6.1 Frozen-window PWF inside crawfish (no new data)

This is the Wannier route without running `wannier` (sections 4.2–4.4). It needs only `proj_tju`, `e_tj`
and μ. The ortho projections (crawfish's recommended `band-projection-params yes no`) are the PDWF convention
(Löwdin-orthonormalized PAO projectors) and avoid the USPP `OpsiRadial` issue, so they are the better input.
FW-PWF from ortho vs. non-ortho projections differ by ~1.3% in graphene and ~2% in N2 bond ICOHP (the Löwdin
metric). Equivalent to JDFTx's `wannier` only for NC with non-ortho input.

### 6.2 Exact AO-basis matrices (fixed PAOs done right; band-independent)

With S(k) = ⟨φ|O|φ⟩ and H(k) = ⟨φ|H|φ⟩ exact (upstream JDFTx dump, theory_audit U1: `psi ^ Opsi` and
`Opsi ^ (H psi)` in `dumpProjections`; a small patch, effort not checked):

- **LCAO-style pCOHP/pCOOP in the fixed PAO basis**: C = S⁻¹T for the occupied bands only, COHP = Re[C* H C].
  Independent of the empty-band count. Spilling (2% here) is reported, not removed. This is the original
  Dronskowski–Blöchl definition and the closest analogue of LOBSTER.
- **Complete-limit frozen-window basis (QO/IAO-like)**: the subspace span{ψ_frozen} ⊕ span{φ·V_free}, with
  V_free the null space of A_frozen (the trial combinations orthogonal to the frozen bands). This is the
  nb → ∞ limit of FW-PWF and needs **no empty bands at all**. H in that subspace is
  blockdiag(ε_frozen, V_free† H_AO V_free) (the cross terms vanish because Hψ = εOψ and the combinations are
  O-orthogonal), with metric V_free† S V_free. Zero spilling, exact sum rule, atom-centred, band-independent.
  **This is the strongest candidate for "adapted, complete and unique".** It needs H_AO, hence the upstream
  dump. S is already available for NC (section 4.8).

### 6.3 Processing `.wfns` with an extended localized basis (the user's "overcomplete molecular basis" idea)

Feasibility in JDFTx data:
- Wavefunctions: `dump End State` writes `wfns`. `dump End Gvectors` (exists; `Dump.cpp`) lists each k's G-vectors
  (`basis.iGarr` order), so C_j(k+G) can be mapped without reimplementing JDFTx's basis setup.
- For **NC**: ⟨g_μ|ψ_j⟩ = Ω^(-1/2) Σ_G g̃*_μ(k+G) C_j(G), with g̃ the analytic Fourier transform (Gaussians:
  closed form; numerical orbitals: a spherical Bessel transform) times Y_lm(k+G), (−i)^l and e^{−i(k+G)·τ}. The
  exact basis overlap S_g(k) and ⟨g|H|g⟩ are cheap analytically for the kinetic part, but ⟨g|V_loc + V_NL|g⟩
  needs the potential (`dump Vscloc`) and the projectors: real work.
- For **USPP**: also needs ⟨β|ψ⟩ and Q_ij (not dumped). That means reimplementing the augmentation, as the periodic
  NBO code does for PAW. Not attractive.

Assessment:
- All-electron Gaussian libraries (cc-pVXZ etc.) **do not fit pseudo-wavefunctions** (nodeless and smooth in the
  core). The basis must be pseudo-consistent: PAOs from the same pseudopotential plus extra radial functions, e.g.
  confined/ionized PAOs, split-valence and polarization, spillage-optimized as in SIESTA or ABACUS (Chen–Guo–He
  2010).
- An **overcomplete** non-orthogonal basis drives spilling toward 0 but makes pair populations basis-dependent
  (diffuse functions take weight from neighbours; Mulliken/COOP pathologies). LOBSTER deliberately uses minimal or
  near-minimal sets. For COHP you would contract back to a minimal adapted set (NAOs from the atom-block
  density matrix, as periodic NBO does; or IAO/QO). That leads back to section 6.2 at higher cost.
- Where it does add value: **missing channels**. Bonding that needs d or polarization functions the
  pseudopotential has no PAO for (e.g. hypervalent main-group, Si CBM per Qiao 2023, s/p-only TM
  pseudopotentials). Cheaper option first: JDFTx `wannier` already accepts `Gaussian` trial orbitals of any l on
  any centre (and `Numerical` ones), so a polarization channel can be added to a projection-only Wannier run
  without touching `wfns`. Its projections never appear in `bandProjections`, so this is Wannier-route only.

### 6.4 Not recommended for crawfish

- **SCDM**: automatic but not atom-centred.
- **Unconstrained MLWF**: section 4.6.
- **Band filtering only (PAOFLOW-style)**: improves the Hamiltonian, not the basis.

---------------------------------------------------------------------------------------------------------------

## 7. Proposed path (nothing implemented; each phase follows theory_audit section 4.0 rules)

**Phase A: frozen-window PWF as a crawfish basis option (crawfish-only, highest value per effort).**
- New `ElecData` settings, e.g. `basis = "lowdin" | "fw_pwf"` (default unchanged), `fw_inner_max` (relative to
  μ, default occupied + ~0.05 Ha, but never more than nproj bands at any k), optional
  `proj_thresholds = (p_out, p_in)`. All in the cache key and `get_edata_settings_str`; a `result_changes.py`
  entry only if the default changes.
- Implementation beside `core/operations/matrix.py::los_projs_for_orbs`: per state, the `basis_frozen_pwf`
  algorithm (~40 lines numpy; numba-able: SVDs of ≤ nproj columns). Output U[t] replaces the Löwdin projections.
  Then `coef_tju`, `h_t_uv` (= U†εU), `s_t_uv` (= I) and `d_t_uv` follow unchanged, and pCOHP/pCOBI/pDOS/images/LTI
  work as is. pCOOP stays zero (orthonormal basis, F7).
- Diagnostics to expose: per-k frozen count; singular values of the partial-band projection B (low values mean
  the antibonding direction is poorly represented, so add bands); the energy of the non-frozen directions;
  atomic character (4.7). Errors (not warnings) when frozen > nproj or the trial orbitals aren't covered.
- Tests: zero spilling and an exact sum rule on N2/graphene fixtures; matches JDFTx `wannier` H for a small NC
  fixture (store its `mlwfH`, `mlwfCellMap` and `mlwfCellWeights`; the `wfns` need not be committed); raises on
  `graphene_1x1` (8 bands).
- Before choosing the default, decide Q1 of the audit again (Löwdin vs. non-orthogonal vs. FW-PWF).

**Phase B: read JDFTx `wannier` output.**
- `ElecData` from `mlwfH` + `mlwfCellMap` + `mlwfCellWeights` (+ centres from the `.out`): diagonalize H(k) on
  any mesh → `user_proj_tju[t,j,u] = V[t,u,j]`, `e_tj` = eigenvalues, uniform weights. Then
  crawfish's Löwdin step is a no-op (S = VV† = I) and every spectrum is a WOHP/WOBI. Needs `ElecData.as_empty`
  fixed (known broken, deferred). Enables fine-mesh interpolation (LTI on a 50³ mesh costs nothing), MLWF with
  `pinAtomicOrbitals`, Gaussian polarization trial orbitals, and frozenCenters workflows.
- Map Wannier centres to atoms (by trial-orbital order for `addAtomicOrbitals`; nearest atom otherwise, as
  pengWann does) and keep lattice vectors from the cell map (no wrapping, consistent with F3).

**Phase C: upstream JDFTx (only if wanted; U1 extended).**
- Dump ⟨φ|O|φ⟩(k) and ⟨φ|H|φ⟩(k) → section 6.2 (band-independent LCAO pCOHP/pCOOP and the complete-limit adapted
  basis).
- Use the full O in the non-orthogonal projection branch for USPP, or document `OpsiRadial` (section 4.8).

**Phase D (research): extended pseudo-consistent basis from `wfns`** for missing channels (6.3). NC only.

---------------------------------------------------------------------------------------------------------------

## 8. Open questions and next experiments

1. Systems where fixed PAOs should be poor (cations, TM d, semicore, metals, adsorbates): rutile TiO2
   (pengWann's example, with a quoted LOBSTER spilling of 9.5e-3), ZnO (fixtures exist, but without `wfns`),
   fcc Al, NaCl/LiF, a Cu or Pt slab with CO. Measure exact PAO spilling and how far FW-PWF bond ICOHP moves from
   crawfish's.
2. Inner-window sensitivity in metals: the frozen set must include all fractionally occupied states. Check
   μ + n·σ choices on the Al fixtures.
3. PD threshold choice: p_out 0.01 vs. 0.05 changed graphene by 2%. Look for a physically motivated rule, e.g.
   p_out = the smallest p at which the chosen partial combinations' singular values stop growing.
4. Pinned MLWF band-count convergence (not run), and SLWF-style constraints (fix centres *and* forbid
   hybridization). JDFTx has no SLWF; check what `pinned` penalizes exactly (`WannierMinimizerFD.cpp`).
5. pCOBI in the Wannier basis vs. Müller 2021 (audit Q3), and spin-polarized Wannier (`Up`/`Dn` files).
6. Verify the LOBSTER facts (supported pseudopotentials, basis fitting, H reconstruction, NBANDS advice) from
   its manual. cohp.de failed TLS this session.
7. Section 4.8: confirm the `OpsiRadial` explanation by re-dumping USPP non-ortho projections with a patched
   `getAtomicOrbitals` (or compare isolated-atom cells, where neighbour augmentation vanishes).

---------------------------------------------------------------------------------------------------------------

## 9. Reproducing

```sh
cd ~/Desktop/jdftx_hold/wannier_cohp
J=~/Desktop/jdftx_hold/build_master/jdftx; WN=~/Desktop/jdftx_hold/build_master/wannier
(cd N2 && $J -c 8 -i totalE.in -o totalE.out && $J -c 8 -i nonortho.in -o nonortho.out && $J -c 8 -i orthodo.in -o orthodo.out)
(cd N2 && for n in wproj_occpi wproj_occ wmlwf_pin wmlwf_free; do $WN -c 8 -i $n.in -o $n.out; done)
# same for graphene (wproj_mu05 wproj_mu wmlwf_pin wmlwf_free) and N2_nc (wproj_occpi)
python band_series.py        # section 4.4  -> results/band_series_*.csv
python pdwf_series.py        # section 4.4/4.5 -> results/pdwf_series_*.csv
python analyze_wannier.py    # section 4.3  -> results/wannier_*.txt
python check_overlap_and_nc.py   # sections 4.2, 4.8 -> results/check_overlap_and_nc.txt
# baseline check against crawfish at a fixed commit:
git -C ~/vs/crawfish archive 8943d71 src | tar -x -C /tmp/crawfish_head
PYTHONPATH=/tmp/crawfish_head/src python check_baseline.py
```

`common.in` files carry absolute pseudopotential paths under `~/Desktop/jdftx_hold/build_master/pseudopotentials`.
The `wfns` files are kept so `wannier` can be rerun.

# Theory audit: crawfish vs. JDFTx source, USPP theory and the COHP formalism

Started 2026-10-27. Living document. Update the status column and add dated entries instead of rewriting history.

Scope: compare what crawfish computes (crawfish `master` at `3efe98f`) with (a) what JDFTx actually writes
(JDFTx source in `../jdftx/jdftx`), (b) ultrasoft-pseudopotential (USPP) theory, and (c) the COOP/COHP/COBI
formalism. The task prioritizes fixes that change only crawfish. JDFTx changes are listed separately as
lower-priority upstream options.

Code citations use `path::function` and avoid line numbers, which drift. Literature citations are listed under
References at the end. Every numbered "Check" can be rerun with the scripts in `notes/theory_audit_checks/`.

---------------------------------------------------------------------------------------------------------------

## 0. How to resume

1. Read section 1 (summary table) and section 2 (the target formalism, in crawfish's array conventions).
2. Rerun the checks (section 6) to confirm the evidence still holds against the current code.
3. Nothing in `src/` has been changed by this audit. The recommended implementation order is in section 4. Each
   item that changes numbers needs a `src/crawfish/result_changes.py` entry (see CLAUDE.md, "Invalidation").
4. Open questions are in section 5. Don't treat them as settled.

---------------------------------------------------------------------------------------------------------------

## 1. Summary of findings

Priority: **P0** = results are qualitatively wrong (wrong sign, or an interaction is silently zero).
**P1** = wrong magnitudes or settings-dependent inconsistencies. **P2** = crashes, edge cases, documentation.

| #  | Pri | Finding | Evidence | Fix needs JDFTx? | Status |
|----|-----|---------|----------|------------------|--------|
| F1 | P0 | JDFTx projections for orbitals with odd l carry a factor of i^l (relative to real orbitals). Crawfish takes `Re[T*_u T_v]`, which is identically ~0 for s–p, p–d (odd Δl) pairs. All s–p hybridization is dropped from pCOMO/pCOOP/pCOHP/pCOBI. | JDFTx source + Check 1 | No | **fixed** in `a22c558` (2026-10-27); legacy via `legacy_orbital_phase=True`; `result_changes.py` entry added |
| F2 | P0 | H/S/P "matrices" are built from `(Re[T*_u T_v])^2`, so they are elementwise non-negative (S, P) or non-positive (H, occupied). The sign of the true matrix element is lost. The bonding/antibonding call is wrong whenever the true H_uv or S_uv is positive (e.g. ppσ). In N2, px–px σ comes out antibonding. | Code + Check 2 | No | **fixed** on `theory_check` (uncommitted, 2026-10-27); legacy via `legacy_matrices=True`; `result_changes.py` entry to add after commit |
| F3 | P0 | Pair weights are the R=0 lattice-image element. k-point coordinates are never used, so there is no Bloch phase. The A–B result depends on how atoms are wrapped into the cell, and may describe a far-away image instead of the bond the user means. | Code + Check 4 | No | open |
| F4 | P1 | The `fillings` file is in [0,2] for `spintype no-spin`, but occupations crawfish computes itself are in [0,1]. k-weights already sum to 2. P/H (hence pCOHP/pCOBI) double when a fillings file exists for unpolarized runs. | JDFTx `Dump.cpp` | No | open |
| F5 | P1 | JDFTx's default `symmetries automatic` reduces the k-mesh by point group + time reversal. Atom-pair and m-resolved quantities from a reduced mesh are symmetry-averaged, and R=0 pairing gets scrambled. Crawfish only warns (and `README` never asks for `symmetries none`). Separately, `get_wk_t` silently replaces symmetry weights with uniform ones for spin-polarized reduced meshes. | JDFTx `Symmetries.cpp`, crawfish `data_parsing.py` | No | open |
| F6 | P1 | Spectrum normalization: the Gaussian kernel `exp(-(E-e)^2/sig)` is unnormalized (integral = sqrt(pi*sig)). For LTI, libtetrabz weights already include 1/N_k, and crawfish multiplies by w_k again. The two methods give different, arbitrary scales, and neither integrates to a state count. | Check 3 | No | open |
| F7 | P1 | pCOOP is basis-dependent in a way the docs don't say. With `band-projection-params yes ...` (the README recommendation) the true AO overlap is the identity, so physical COOP between different orbitals is zero, and anything non-zero is a band-truncation artifact. The README's claim that pCOMO's sign identifies bonding is false for pairs with negative overlap (ppσ) and for odd-Δl pairs, where pCOMO's sign depends on orbital orientation. | Check 2 | No | open |
| F8 | P2 | `elecdata.py::calculate_filling_nobroad` uses `np.heaviside(eig, efermi)`, which gives inverted fillings and ignores mu. `calculate_filling_gauss/mp1/cold` call `math.erf` on arrays and raise TypeError. Both paths are only reached when no `fillings` file exists. | Check 3 | No | open |
| F9 | P2 | Spinor modes (`vector-spin`, `spin-orbit`) are mis-handled. `data_parsing.py::spintype_nspin` maps them to nspin=2, but JDFTx has one state per k-point there. Orbital labels ignore JDFTx's per-m spin copies or j-resolved orbitals. | JDFTx `ElecInfo.cpp`, `SpeciesInfo_atomic.cpp` | No | open |
| F10 | P2 | Misc. `los_projs_for_orbs` uses an absolute eigenvalue cutoff of 1e-15 on complex64 data. `trim_excess_bands` selects bands globally by sum of \|T\| (not \|T\|^2, and not per k) and then Sinkhorn-normalizes. `p_uu_consistent` is a non-standard per-band renormalization. The README formula for S (\|T*T\|^2) doesn't match the code ((Re T*T)^2). In the pymatgen fork, the `mu` fallback path (`_mu_backup`) is Hartree while the main path is eV, and crawfish divides by Ha_to_eV unconditionally. | Code | No (pymatgen-fork fix for mu) | open |
| U1 | — | Optional upstream improvements: apply the i^l phase in `dumpProjections` (as the Wannier code does), dump the AO overlap S(k) = <phi\|O\|phi> (and optionally <phi\|H\|phi>), and use higher precision or binary projections. None of these is required for the crawfish-only fixes. | — | Yes | not started |

---------------------------------------------------------------------------------------------------------------

## 2. Target formalism (what crawfish should compute), in crawfish array conventions

### 2.1 What JDFTx writes (verified in source)

- `jdftx/electronic/DumpExcitationsMoments.cpp::dumpProjections`
  - `ortho == false`: `proj = iInfo.getAtomicOrbitals(q, true) ^ eVars.C[q]`, i.e. `proj[u,j] = <phi_u | O | psi_j>`.
    `getAtomicOrbitals(q, applyO=true)` returns O·phi.
  - `ortho == true`: `orthoMat = invsqrt(psi ^ Opsi)`, then `proj = orthoMat * (Opsi ^ C)`. This is the projection
    onto Löwdin-orthonormalized Bloch-sum AOs in the O metric, so the true overlap of the orbitals used is exactly I.
  - The header says `ortho-orbital-projections` when ortho is on, which crawfish can detect.
  - Values are written as `%9.7lf` (7 decimals, fixed point).
  - Layout: one line per band, orbitals along the line. `matrix proj` is nOrbitals x nBands, column-major, and
    `projData++` walks the orbitals of one band. This matches `data_parsing.py::_parse_bandfile_reader`.
- O (the USPP overlap) **includes the augmentation**. See `jdftx/electronic/SpeciesInfo.cpp::SpeciesInfo::setPsi`:
  `Opsi = psi + sum_{p1,p2} Q_{p1p2} |beta_p1><beta_p2|psi>`, with each orbital normalized so that `<psi|O|psi> = 1`.
  For norm-conserving pseudopotentials `Qint` is empty and O = 1.
- Bloch-sum phase convention: `jdftx/electronic/SpeciesInfo_internal.h::Vnl_calc` sets
  `Vnl = Ylm(qhat) f(|k+G|) cis(-2π pos·(k+G))`. In real space this is
  `phi_uk(r) = sum_R e^{ik·R} phi_u(r - tau_u - R)`, with tau_u the atom position exactly as JDFTx holds it.
  I found no wrapping of `atpos` in `IonInfo.cpp`, `SpeciesInfo.cpp` or `commands/ion.cpp`. The `floor` in
  `IonInfo.cpp` belongs to the minimum-image distance in the core-overlap check.
- **No i^l factor.** The radial transform `jdftx/core/RadialFunction.cpp::RadialFunctionR::transform` is a plain
  4π∫r²j_l(Gr)f(r)dr, and `Vnl_calc` multiplies by a real Ylm. The Rayleigh expansion
  e^{iG·r} = 4π Σ_l i^l j_l(Gr) Σ_m Y_lm(Ĝ)Y_lm(r̂) then gives real-space orbitals equal to i^l × (real orbital).
  JDFTx's own Wannier code corrects for this: `jdftx/wannier/WannierMinimizer.cpp` has
  `complex lPhase = cis(0.5*M_PI*od.l); //including this phase ensures odd l projectors are real (i^l term in spherical wave expansion)`.
  `dumpProjections` does not apply it.
- Orbital order within an atom (`SpeciesInfo_atomic.cpp::setAtomicOrbitals` and `::atomicOrbitalOffset`): l outer,
  then shell n, then m = -l..l, then spin copy (spinor modes only). The real Ylm order is in `jdftx/core/SphericalHarmonics.h`
  (`YlmInternal`): p = (y, z, x), d = (xy, yz, z², xz, x²-y²), f = (y(3x²-y²), xyz, yz², z³, xz², z(x²-y²), x(x²-3y²)).
  **`utils/indexing.py::orb_ref_list` matches this order** (verified). The docstring example in
  `get_atom_orb_labels_dict` lists p as px,py,pz, which is wrong, but the code is right.
- Weights and fillings: `ElecInfo.cpp::ElecInfo::kpointsReduce` multiplies k-weights by `spinWeight` (2 for
  no-spin, 1 for z-spin), so the weights in `kPts`/bandProjections headers sum to 2 in both modes (Check 3: 2.0).
  Internal fillings are in [0,1]. **The `fillings` dump multiplies them by 2 for no-spin**
  (`Dump.cpp`: `wInv = spinType==SpinNone ? 0.5 : 1.0; F *= 1/wInv; eInfo.write(F, ...)`). `eigenvals` is
  `eInfo.write(Hsub_eigs)`, binary float64 in Hartree.
- k-point order (`ElecInfo.cpp` folding loop, `i[0]` outermost, `i[2]` innermost; also seen directly in `kPts`) is
  spin-major, then a, b, c with c fastest. This matches crawfish's `reshape([nspin]+kfolding+[nbands])`.
- Symmetry reduction (`Symmetries.cpp::Symmetries::reduceKmesh`): `symmetries none` returns the full mesh, with no
  inversion reduction either. `automatic` is the default (`commands/symmetries.cpp`), and `kpoint-reduce-inversion`
  defaults to yes.
- Smearing (`ElecInfo.cpp::ElecInfo::smear`): x = (ε-μ)/(2σ). Fermi 0.5(1-tanh x), Gauss 0.5 erfc x, MP1, Cold.
  **crawfish's `calculate_filling_*` formulas match** (only the array/erf issue in F8 is wrong). The broadening
  value comes from the echoed `elec-smearing` line (`pymatgen ... jdftxoutfileslice.py::_get_broadeningvars`), which is
  in Hartree, consistent with `e_tj`.

### 2.2 USPP theory and why the crawfish-only route is sound

In USPP (Vanderbilt 1990) the KS problem is generalized: H|ψ⟩ = ε O|ψ⟩, with ⟨ψ_i|O|ψ_j⟩ = δ_ij. The resolution of
the identity is Σ_j |ψ_j⟩⟨ψ_j|O = 1 over a complete set, and the spectral form of H is H = Σ_j O|ψ_j⟩ε_j⟨ψ_j|O.
Therefore, with T_uj := ⟨φ_u|O|ψ_j⟩, which is exactly what JDFTx dumps:

- ⟨φ_u|O|φ_v⟩ ≈ Σ_j T_uj T*_vj (exact for a complete band set),
- ⟨φ_u|H|φ_v⟩ ≈ Σ_j T_uj ε_j T*_vj (same),
- ψ_j ≈ Σ_u C_uj φ_u with C = S⁻¹T (least squares in the O metric).

USPP needs no special treatment beyond using ⟨φ|O|ψ⟩ consistently, and JDFTx already does. The augmentation
Q_ij = ⟨φ_i|φ_j⟩_AE − ⟨φ̃_i|φ̃_j⟩_PS (generalized norm conservation) means ⟨φ̃|O|ψ̃⟩ reproduces the all-electron
overlap to the extent that the pseudo-atomic orbitals lie in the span of the partial waves. That is the USPP
analogue of LOBSTER's PAW reconstruction (Maintz 2013), and it answers the README's concern that non-PAW COHP is
"far from meaningless". The leftover error comes from truncating the band set and the orbital set. It is measurable
(section 2.4) and should be reported, as LOBSTER reports "spilling".

### 2.3 Target equations (crawfish layout: P[t,j,u] = ⟨φ_u|O|ψ_tj⟩ = T_uj at state t)

Step 0, phase fix (F1): `P[..., u] *= 1j**l_u`, where l_u comes from the bandProjections header. Apply it only to
JDFTx data, never to `user_proj_tju`.

Step 1, per state t (all matrices are complex Hermitian, shape (t,u,u)):

```
S_t[u,v] = Σ_j P[t,j,u] · conj(P[t,j,v])                 # overlap of band-projected orbitals
H_t[u,v] = Σ_j P[t,j,u] · e[t,j] · conj(P[t,j,v])        # Hamiltonian in the same basis
C_t      = P_t · inv(conj(S_t))                          # non-orthogonal basis (C = S^-1 T)
   or      P'_t = P_t · (P_t^† P_t)^(-1/2) and C_t = P'_t   # Löwdin (= los_projs_for_orbs, already correct);
                                                         # then rebuild S_t (= I) and H_t from P'_t
D_t[u,v] = Σ_j f[t,j] · C[t,j,u] · conj(C[t,j,v])        # density matrix (COBI, Mulliken/Löwdin bond orders)
```

Step 2, weights for pair sets A (orbitals u) and B (orbitals v), M ∈ {H, S, D, 1}:

```
all images: W^M[t,j] = w_t · Σ_{u∈A, v∈B} Re[ conj(C[t,j,u]) · M_t[u,v] · C[t,j,v] ]
one image : M_t[u,v] → M_uv(R) · exp(+2πi k_t·R),  M_uv(R) = Σ_{t∈spin} (w_t/Σw) M_t[u,v] exp(-2πi k_t·R)
```

Sign and convention checks: Check 4 verifies `exp(+2πi k·R)` with crawfish's layout, and Check 2 verifies the
COHP index order. R is the lattice vector of B's image relative to A's cell, using the positions JDFTx used. The
default should be the minimum image, |L(τ_B + R − τ_A)| minimal. Report R and the bond length with the spectrum,
as LOBSTER does.

Mapping to crawfish's modes: pCOHP: M=H. pCOOP: M=S. pCOMO: M=1 (Coulson/Mulliken-type bond order density,
orientation-dependent sign). pCOBI: M=D. Verify against Müller 2021 for the spin factor and sign before
implementing (open question Q3).

Sum rules, to use as tests:
- Σ_{all u,v} Σ_tj f_tj W^H_tj = Σ_tj w f (Π ε Π)_jj ≈ band energy Σ w f ε, where Π = T†S⁻¹T is the projector onto
  the orbital span. Check 2 gives −5.459 vs −5.476 Ha (99.7%).
- Σ_{u,v} W^S_tj = w_t Π_jj (Mulliken normalization; 1 − Π_jj is the band spilling).
- In the Löwdin basis, off-diagonal H is invariant to ε → ε + c, so `norm_eigs_for_h` is irrelevant there. In the
  non-orthogonal basis, ICOHP shifts by −c·ICOOP (Glassey & Hoffmann 2000). Check 2: shifting by μ moves the A–B
  total from −0.833 to −0.786 Ha.

### 2.4 Quality diagnostics crawfish should report

- Charge spilling of occupied states (Sánchez-Portal 1995): 1 − Σ_tj w f Σ_u |P|² / Σ_tj w f. This is only
  meaningful for ortho-orbital projections; for non-orthogonal ones use Π_jj. N2 test data: 2.06%.
- Orbital completeness ⟨φ_u|P_bands|φ_u⟩ = S_t[u,u] (ortho data). N2: s 0.80, py/pz 0.97, px (bond axis) 0.69, all
  with 15 bands. Low values mean too few empty bands (`elec-n-bands`). This directly limits COOP and COHP accuracy.
- Sum-rule residual from 2.3.

---------------------------------------------------------------------------------------------------------------

## 3. Findings in detail

### F1: the missing i^l phase zeroes odd-Δl interactions (P0)

- **What:** the real-space orbital behind each dumped projection is i^l φ_real, so
  T_raw = ⟨i^l φ_real|O|ψ⟩ = (−i)^l T_real and T_real = i^l T_raw. For an s–p pair, Re[T*_s T_p] is really
  ∓Im[T*_s,real T_p,real]. At Γ, KS states are real up to a per-band global phase, so the physical product is
  real and the crawfish quantity is exactly 0. At other k it is a meaningless mixture.
- **Where:** every crawfish spectrum except pDOS goes through `core/operations/matrix.py::_get_gen_tj_coef_uu_jit`
  and `::_get_gen_tj_coef_tuu_jit` (both take `(conj(proj_u) * proj_v).real`), and through
  `::_get_h_p_s_t_uu_and_uu_jit`. Löwdin (`los_projs_for_orbs`) is phase-covariant, so it neither causes nor fixes
  this. pDOS (|T|²) is unaffected.
- **Evidence:** Check 1 on N2 (USPP, GBRV): at Γ, Re(s_A* px_B) = 0.00000 for every band, while Im is ±0.23 for
  the σ bands. Over all k, Σ|Re| = 0.18 vs Σ|Im| = 114.2. In Check 2, crawfish IpCOHP(s,px) = 0.00000 while the
  reference gives −0.139 Ha, the third-largest inter-atomic term.
- **Fix (crawfish-only):** in `ElecData._set_proj_tju` (jdftx path only, after loading from cache or file,
  before band trimming and Löwdin), multiply column u by `1j**l_u`. Build l_u from the bandProjections header
  (`lMax`, `nShells(l)` per species; the same data `get_atom_orb_labels_dict` uses). Cache stage: "matrices", if
  applied after `proj_tju.npy` is loaded, so the parse cache stays valid. **Use the sign-aware formulas from F2 as
  well.** Those are phase-invariant (Re[C*_u M_uv C_v] is unchanged under φ_u → e^{iθ_u}φ_u), so after F2 the
  phase fix only matters for pCOMO and for sanity checks such as Re-at-Γ. Doing both is still cleaner.
- **Upstream option:** apply `lPhase` in `dumpProjections` behind a new flag. Old outputs would then differ, so
  crawfish would need a header marker. Not recommended as the primary fix.

**Implementation record (2026-10-27, branch `theory_check`, not yet committed):**
- `io/data_parsing.py::get_orb_l_u_from_bandfile_filepath` reads l per orbital from the bandProjections header
  (species lines: nAtoms, nOrbitalsPerAtom, lMax, nShells(l)), in JDFTx order (l, then n, then m). It allows the
  ×2 spinor multiplicity (spin copies or j-resolved orbitals are still grouped by l), and raises if the counts don't
  fit. Exposed as `ElecData.orb_l_u`.
- `core/operations/matrix.py::apply_jdftx_orbital_phase` multiplies orbital u by exact powers of i (lookup
  `[1, i, -1, -i][l % 4]`) and keeps complex64.
- `core/elecdata.py::ElecData._set_proj_tju` applies it after loading `proj_tju.npy` or the bandProjections file,
  and before band trimming or Löwdin (both are phase-covariant). It is skipped for `legacy_orbital_phase=True`,
  for `|projection|^2` (real) files, and for non-JDFTx data (`user_proj_tju` is never touched). The
  `proj_tju.npy` backup still holds the raw dump, so the "parse" cache stage is unaffected.
- **Legacy access (section 4.0):** `ElecData.legacy_orbital_phase` (default `False`) is a property whose setter
  calls `_init_caches()` then `alloc_elec_data()`. It is in `edata_setting_key_shorthand_dict` (`lgph`) and
  `get_edata_settings_str`, so legacy and corrected matrices, weight caches and spectra live in separate cache
  directories. A side effect: existing cache directories, which have no `lgph_` key, are orphaned rather than
  reused. That's harmless, just disk space.
- `result_changes.py`: **not added yet**. The entry needs the commit hash. Add it in a follow-up commit, with stage
  "matrices" and a description like "Apply i^l orbital phase to JDFTx band projections (F1)". With the new cache
  key, stale results can't be reused even without the entry, but CLAUDE.md requires one for every
  result-changing change.
- Docs: README "Projection modifications" item 0, a note on user projections in "Creating your own ElecData", the
  CLAUDE.md settings list, docstrings on the new functions and setting.
- Tests: `tests/core/test_orbital_phase.py` covers header parsing, the phase helper, Γ s–p products (real when
  corrected, imaginary in legacy), unchanged pDOS, the raw `proj_tju.npy` backup, legacy reproduction of the
  section-6 values, and a non-zero s–px IpCOMO when corrected. `legacy_orbital_phase` was added to
  `test_settings_change_updates_cache_sub_dir`. Full suite: 104 passed.
- Effect on N2 (default settings, everything else still legacy). Same-l pairs are unchanged in pCOMO. pCOHP moves
  because the per-band `p_uu_consistent` normalization sums over all pairs, including s–p:

  | pair | IpCOHP legacy → fixed | IpCOMO legacy → fixed |
  |------|-----------------------|-----------------------|
  | s-s  | −0.22098 → −0.12203   | +0.40719 → +0.40719   |
  | s-px | +0.00000 → −0.08856   | −0.00000 → +0.49230   |
  | px-px| +0.15313 → +0.08452   | −0.58770 → −0.58770   |
  | py-py| −0.11463 → −0.11462   | +0.99778 → +0.99778   |

  px–px is still antibonding: that is F2, which F1 alone can't fix.

### F2: sign-blind H/S/P construction (P0)

- **What:** `matrix.py::_get_h_p_s_t_uu_and_uu_jit` computes x = Re[T*_u T_v] and s = x², with
  p_coef = f·w (÷ Σ_uv s if `p_sc`) and h_coef = p_coef·(ε − μ). So S_uv ≥ 0, P_uv ≥ 0, and H_uv ≤ 0 for occupied
  states. pCOHP = H_uv · Σ Re[T*T] δ therefore has the sign of −Re[T*T], whatever the sign of the true H_uv.
  These are not the overlap, density or Hamiltonian matrices (compare section 2.3). In addition:
  - H is occupation-weighted. A Hamiltonian matrix element shouldn't depend on fillings. Crawfish's H is closer to
    an energy-weighted density matrix, which is a different object.
  - With `state_sep=False`, H is summed over k and paired with k-summed weights (see F3). With `state_sep=True`,
    `h_t_uu` is per k but still real and non-positive.
- **Evidence:** Check 2 on N2 (molecule along x; JDFTx px on both atoms points +x, so for σ bonding
  c_pxA = −c_pxB and the coefficient product is negative). Crawfish IpCOHP(px,px) is **+0.153** (antibonding)
  vs the reference **−0.118 Ha** (bonding). The π pairs keep the right sign (positive overlap) but have the wrong
  magnitude (−0.115 vs −0.179). s–s: −0.221 vs −0.111. At Γ band 4 (3σg), Re[pxA*pxB] = −0.32.
- **Fix (crawfish-only):** replace the matrix builder with section 2.3, step 1: complex (t,u,u) S_t, H_t, D_t,
  and coefficients C. Replace the weight kernels with
  `Re[conj(C[t,j,u]) * M[t,u,v] * C[t,j,v]]` (complex M). Memory: complex64 (t,u,u) is twice the current float32
  arrays, which is compatible with the `limited_memory` goals (no (t,j,u,u) tensors). `p_uu_consistent` and the
  squaring disappear. Keep `norm_eigs_for_h` only for the non-orthogonal basis, and document it with Glassey &
  Hoffmann 2000. The README sections "Matrix modifications" and the pCOHP/pCOOP/pCOBI equations need rewriting
  (done; see the implementation record below).
- **Theory references:** Dronskowski & Blöchl 1993 (COHP definition); Hughbanks & Hoffmann 1983 (COOP); Deringer
  2011 and Maintz 2013/2016 (projected COHP from plane waves: coefficients from the transfer matrix and overlap,
  H reconstructed from projections and eigenvalues); Müller 2021 (COBI uses the density matrix).

**Implementation record (2026-10-27, branch `theory_check`, not yet committed):**
- `core/operations/matrix.py`:
  - `get_coef_tju(proj_tju, orthonormal)`: C = P for Löwdin-orthonormalized projections; otherwise
    C_t = P_t pinv(P_t^† P_t) with `rtol=1e-6` (pinv, so orbitals the bands don't represent are dropped rather than
    amplified).
  - `get_h_s_d_t_uv(proj, coef, e, occ)`: one-pass numba kernel (`_get_h_s_d_t_uv_jit`) building complex Hermitian
    H_t, S_t, D_t (t,u,u) with complex128 accumulation. No (t,j,u,u) tensors, so it stays compatible with the
    `limited_memory` goals.
  - `get_r0_suv(mat_t_uv, wk_t, nspin)`: per-spin R=0 element Σ_{t∈s} w_t M_t / Σ w_t.
  - `get_spin_idx_t`.
  - `_get_gen_tj_coef_tuv(coef, mat_muv, m_t, wk, orbs_u, orbs_v)`: one weight kernel for both forms (m_t picks
    the matrix per state).
  - The legacy builder and kernels (`get_h_p_s_t_uu_and_uu`, `_get_gen_tj_coef_uu/_tuu`) are unchanged. Their
    docstrings now say "Legacy (pre-F2)".
- `core/elecdata.py`:
  - Setting `legacy_matrices` (default False, cache key `lgmt`).
  - Properties `coef_tju`, `h_t_uv`, `s_t_uv`, `d_t_uv`, method `get_mat_r0_suv(name)`, `set_mat_uv()`. The
    signed matrices are cached as `h_t_uv.npy`, `s_t_uv.npy` and `d_t_uv.npy` in the settings sub-directory.
  - `reval_mat_uu()` also resets the coefficients and re-evaluates the signed matrices, and `unnorm_projs` resets
    the coefficients.
  - `bandprojfile_is_ortho` reads the bandProjections header.
  - `norm_eigs_for_h` shifts e by −μ in H_t, which only matters in the non-orthogonal basis.
  - `p_uu_consistent` only affects the legacy path.
- `funcs/pcohp.py`: each `_compute_*_tj` dispatches on `legacy_matrices`, with the legacy branch unchanged.
  `_compute_signed_tj` maps pCOHP→H, pCOOP→S, pCOBI→D and pCOMO→1. pCOMO now uses C, which equals the
  projections when `los_orbs=True` and differs in the non-orthogonal basis.
- **Meaning of `state_sep`, new path:** `False` (default) uses the per-spin R=0 matrix, which keeps today's
  "same cell" pairing until F3 adds general R. `True` uses the state-resolved M_t, which sums the interaction over
  all lattice images. For N2 the two differ by less than 1e-4 Ha.
- **Basis (Q1):** not forced. `los_orbs=True` (default) gives the Löwdin basis and `los_orbs=False` gives the
  non-orthogonal basis (C = S⁻¹T, band-space S). Both are tested against an independent float64 implementation.
- **F7, partly addressed:** `_warn_if_coop_is_artifact` warns that pCOOP is identically zero when
  `los_orbs=True`, and that it only reflects band truncation for ortho-orbital projections with `los_orbs=False`.
  `tests/funcs/test_pcoop.py` now uses `los_orbs=False` and expects the warning (before, it passed only on float
  noise).
- **F10 setter ordering, fixed alongside:** the `los_orbs` and `trim_excess_bands` setters now call
  `_init_caches()` before `alloc_elec_data()`, as the other settings already did. Otherwise the signed matrices
  would be loaded from the previous settings' cache directory when toggling the basis.
- **Results (N2, defaults, IpCOHP, legacy matrices → fixed):** s-s −0.12203 → −0.11131, s-px −0.08856 → −0.13936,
  px-px **+0.08452 → −0.11815**, py-py −0.11462 → −0.17931, all-all −0.26678 → −0.86678. pCOBI px-px:
  −0.21373 → +0.17270. These match the reference to 1e-5, and the sum rule holds to 0.3%.
- Tests: `tests/core/test_signed_matrices.py` checks Hermiticity, S=I in the Löwdin basis, the band-energy sum
  rule, agreement with an independent reference for both bases and both energy zeros, that all N2 pairs are
  bonding, R=0 ≈ all images, the pCOOP warnings, and that the legacy matrices reproduce the recorded values. The
  F1 legacy-reproduction test now sets `legacy_matrices=True` too. Full suite: 117 passed.
- Docs: README "Matrix modifications" and "Provided analysis techniques" rewritten for the signed formalism, with
  the old equations kept under "Legacy formalism". The CLAUDE.md pipeline step 3 and settings list are updated.
  The F2 comparison notebook now runs legacy vs. fixed vs. reference, `examples/N2.ipynb` uses `los_orbs=False`
  for pCOOP (with a caveat), and `fixcompare.load_edata_at` keeps each notebook isolated to its own fix.
- `result_changes.py`: the F1 entry is added (`a22c558`, stage "matrices"). **The F2 entry still has to be added
  after this work is committed** (stage "matrices", e.g. "Signed projected H/S/D matrices for pair spectra (F2)").

### F3: lattice image (R) not resolved (P0 for periodic solids, irrelevant for isolated molecules)

- **What:** Σ_k w_k Re[T*_u(k) T_v(k)] with no phase is the R=0 element of the real-space density matrix, i.e.
  the interaction of u at τ_u with v at τ_v in the *same* cell. Nothing in crawfish uses k-point coordinates:
  `ElecData.ks_t` is parsed but never read by any spectrum function. So "atom A – atom B" means whichever image
  of B shares A's cell index. That depends on how the input happened to wrap coordinates, and is often not the
  bonded neighbor.
- **Evidence:** Check 4 runs crawfish's own kernel on a 1D SSH chain: the same crystal with B wrapped differently
  gives IpCOMO(A,B) = 0.977 (short bond) or 0.152 (long bond). The R-resolved formula recovers both bonds in
  either cell choice.
- **Fix (crawfish-only):** section 2.3, step 2. Provide `R=None` (sum over all images, k-resolved M_t; the closest
  analogue of today's `state_sep=True`) and `R="nearest"` (default), or an explicit integer 3-vector. The
  per-atom-pair R from the minimum image becomes a phase array on the coefficient matrix. Requirements: a full
  MP grid (see F5), atom positions exactly as JDFTx used them (verify that `JDFTXOutfile.structure` does not wrap
  them; open question Q2), and k-points from `kPts` or the bandProjections header (`ks_t`).

### F4: fillings-file factor 2 for no-spin (P1)

- JDFTx `Dump.cpp` writes `fillings` as F×2 for `SpinNone`. k-weights already include the factor 2. Crawfish
  `ElecData.occ_tj` reads the file as-is, but computes [0,1] when the file is absent. N_e = Σ w f is therefore 2×
  too large with a file for unpolarized runs. P, H and h_coef double. `get_generic_integrate(use_fillings=True)`
  divides by max(occ), which hides the problem for integrated spectra but not for matrices. (The test data is z-spin,
  so it doesn't exercise this: N_e = 10.000 there.)
- **Fix:** divide by 2 when `nspin == 1` and the run is not spinor (equivalently, divide by the JDFTx
  `spinWeight`). Sanity-check that Σ w f equals nElectrons from the outfile. Remove the `/ max(occ)` hack in
  `funcs/general.py::get_generic_integrate`.

### F5: symmetry-reduced k-meshes (P1)

- With the default `symmetries automatic`, k-points are reduced by the point group and by time reversal. For
  scalar, rotation-invariant quantities (total DOS, l-summed pDOS of a symmetry-complete atom set) the reduced
  weighted sum is fine. For one atom pair or one m, the IBZ sum equals the average over all symmetry images of that
  pair and orbital, and the R=0 pairing of the images is not preserved. LOBSTER similarly requires running without
  symmetry reduction (open question Q4).
- Crawfish today: `data_parsing.py::_get_lti_allowed` warns, `get_kfolding` invents `[nk,1,1]`, and
  `get_wk_t` (non-LTI branch) compares `len(wk)` (nspin·nk) with nk. **For spin-polarized reduced meshes it always
  discards the real weights and uses uniform 1/nk.** For nspin=1 the reduced weights are kept.
- **Fix:** document `symmetries none` as required (README "How to use", next to `band-projection-params`). Make
  pair and orbital-resolved functions raise on a reduced mesh, unless the user opts in and only full-shell,
  full-star sums are requested. Fix the `len(wk)` comparison (compare with nstates). Time-reversal-only reduction
  would be acceptable for Re[] quantities (C(−k) = C(k)* for real orbitals without SOC), but JDFTx can't produce it
  alone: `symmetries none` disables inversion reduction too. So "full mesh" is the practical requirement.

### F6: spectrum normalization (P1)

- `core/operations/scalar.py::gauss` returns `exp(-(x-mu)^2/sig)`. Its integral is sqrt(π·sig) (0.0056 for the
  default `SIGMA_DEFAULT=1e-5`). Docstrings call `sig` a "standard deviation", but the actual standard deviation is
  sqrt(sig/2) ≈ 0.061 eV.
- libtetrabz `dos()` weights integrate to 1 per band over the BZ (checked numerically: Σ_k ∫ wght dE = 1.000), so
  they already contain 1/N_k. `vector.py::get_lti_spectrum` multiplies by `weights_sabcj`, which already contain
  w_k. Check 3: Gaussian DOS integral 0.168, LTI 0.586, expected 30 (= Σw · nbands). The LTI number is also
  degraded here by flat molecular bands, so read it as "wrong scale", not as an exact factor.
- **Fix:** normalize the Gaussian by 1/sqrt(π·sig), keeping the exponent so peak shapes and the meaning of `sig`
  don't change. For LTI, divide the weights by w_k and multiply by the spin weight (or equivalently pass
  un-k-weighted weights). Add a test that ∫DOS = Σ_t w_t · nbands for both methods. Cache stage: "spectra".

### F7: COOP in an orthonormal basis; pCOMO's sign (P1, mostly documentation)

- With `band-projection-params yes no` (README requirement) the orbitals are O-orthonormal, so the physical
  ⟨φ_u|O|φ_v⟩ = δ_uv and COOP_uv = 0 for u ≠ v. After crawfish's Löwdin, the band-space S is exactly I as well
  (Check 2: max |S−I| = 1e-14). A non-orthogonal-basis "COOP" computed from ortho data only measures band
  truncation. Check 2 gives ICOOP(py,py) = +0.006 but (px,px) = +0.178, tracking the px completeness of 0.69. That
  is not chemistry.
- For meaningful COOP: run JDFTx with `band-projection-params no no`, use crawfish's non-orthogonal path
  (`los_orbs=False`, C = S⁻¹T) with band-space S, and use enough empty bands that S_uu ≈ 1 (JDFTx normalizes
  ⟨φ|O|φ⟩ = 1). The exact alternative is upstream U1: dump ⟨φ|O|φ⟩.
- pCOMO = Σ Re[C*_u C_v] δ: for ppσ with parallel p orbitals the bonding combination has a negative product (N2
  3σg: −0.32), so the README's claim that overlap "does not influence the identification of a peak as bonding" is
  false whenever S_uv < 0. Coulson's mobile order was defined for π systems, where all pz overlaps are positive.
  Keep pCOMO, but document it as orientation-dependent.

### F8: occupation helpers (P2)

- `elecdata.py::calculate_filling_nobroad`: `np.heaviside(eig, efermi)` gives 0 below 0 Ha and 1 above,
  ignoring mu. It should be `np.heaviside(efermi - eig, 0.5)`. Check 3 output: `[0,0,1,1]` for eig `[-.5,-.1,.1,.5]`.
- `calculate_filling_gauss/mp1/cold` use `from math import erf` on ndarrays and raise TypeError (Check 3). Use
  `scipy.special.erf`/`erfc`.
- These only run without a `fillings` file. For calculations without smearing, the outfile has no
  `FillingsUpdate: mu:` line, so the `mu` path is shaky too.

### F9: spinor modes (P2)

- JDFTx: `SpinVector` and `SpinOrbit` use spinor ColumnBundles with one state per k-point
  (`ElecInfo::kpointsReduce` comment). Orbitals come with `nSpinCopies = 2/qWeightSum` copies per m
  (`SpeciesInfo_atomic.cpp`), or are j-resolved for relativistic pseudopotentials (`getYlmToSpinAngleMatrix`).
- Crawfish: `data_parsing.py::spintype_nspin` gives 2. Labels from `orb_ref_list` ignore spin copies and j. The
  per-m spin pair also needs combining for COHP (sum over σ,σ' with the spinor components).
- **Fix:** refuse spinor runs with a clear error until the code supports them. Supporting them later is a
  self-contained project.

### F10: smaller items (P2)

- `matrix.py::los_projs_for_orbs`: the eigenvalue cutoff is an absolute `eigs > 1e-15` on complex64 input. Use a
  relative cutoff (e.g. 1e-6·max) and warn when an orbital is dropped, since that means it isn't represented by the
  bands.
- `elecdata.py::get_best_bands` sums |T| (not |T|²) over all k and picks the same band indices at every k, then
  `normalize_square_proj_tju` Sinkhorn-scales |T|² to be doubly stochastic. That changes the projections
  non-unitarily. Note that for square T, Löwdin already produces a unitary T', whose |T'|² is exactly doubly
  stochastic. Recommend deprecating `trim_excess_bands`, or selecting bands per k by energy window.
- `p_uu_consistent`: per-band renormalization Σ_uv P_tj,uv = f w. It becomes unnecessary under section 2.3. If
  charge-conserving rescaling is wanted, divide by Π_jj, and report the spilling either way.
- Setter ordering (found while fixing F1; **fixed with F2**): `ElecData.los_orbs` and `.trim_excess_bands` called
  `alloc_elec_data()` before `_init_caches()`. If matrices were already evaluated, `reval_mat_uu` → `set_mat_uu`
  then loads or saves them in the *old* settings directory. `p_uu_consistent`, `norm_eigs_for_h` and
  `legacy_orbital_phase` use the safe order. Also, `alloc_elec_data` reloads `proj_tju` but reuses the already
  trimmed `_e_tj`/`_occ_tj`, so toggling any setting while `trim_excess_bands=True` indexes trimmed arrays with
  untrimmed band indices.
- README "pCOHP" block: S is written |⟨μ|ψ⟩⟨ψ|ν⟩|², but the code uses (Re⟨μ|ψ⟩⟨ψ|ν⟩)².
- pymatgen fork (`jdftxoutfileslice.py`): `jstrucs.mu` is in eV (`jelstep.py::_set_mu` × Ha_to_eV), but
  `_mu_backup` is a raw Hartree float. `ElecData.mu` divides by Ha_to_eV unconditionally. This would be wrong by a
  factor of 27.2 if only the fallback is populated. `data_parsing.py::get_mu_from_outfile_filepath` (Hartree,
  crawfish-native) exists and is unused. Prefer it.
- The README's non-JDFTx instructions say weights should sum to nspin. JDFTx's convention is that they sum to 2
  (spin-unpolarized weights carry the spin factor). `_get_arbitrary_wk` gives a sum of nspin, which is 1 for
  unpolarized runs and undercounts by 2. `ElecData.as_empty` is broken anyway (see memory note).

---------------------------------------------------------------------------------------------------------------

## 4. Proposed implementation plan (crawfish-only)

Each step is independently testable. Add a `result_changes.py` entry per step that changes numbers.

### 4.0 Requirements for every fix (user instruction, 2026-10-27)

These apply to every item in this plan and to any later fix that changes results.

- **Document each fix well.** Record what changed, why (point to the finding number here and its evidence), and
  how results move. Put that in the code (docstrings or comments at the changed function), in the
  `result_changes.py` entry description, in the README wherever the formalism or the required JDFTx flags change,
  and in this file's status column.
- **Keep the legacy behaviour reachable from the current code.** It should be possible to measure how a fix changes
  results later without checking out old commits. Removing a legacy path is not a deprecation step. Suggested
  shape, to be settled in the first implementation:
  - Move the old implementations into one place (e.g. `crawfish/legacy/` or `_legacy`-suffixed functions next to
    their replacements). Don't edit them in place.
  - Add one switch that selects legacy or corrected behaviour. It could be a single `ElecData` setting such as
    `formalism="legacy" | "projected"`, or per-fix flags such as `legacy_phase`, `legacy_matrices`,
    `legacy_fillings` and `legacy_normalization`. Per-fix flags let you isolate the effect of each change.
  - The switch must be part of the cache key (`edata_setting_key_shorthand_dict`, `get_edata_settings_str`,
    `func_args_dict` / `metadata_dict`), so legacy and corrected results never share cache entries.
  - Add tests that run both paths on the test data. They should assert that the legacy path still reproduces the
    numbers recorded in section 6, and that the corrected path meets the new tests in step 1. That keeps the legacy
    code from rotting silently.
  - Consider a small comparison helper (e.g. `compare_formalisms(edata, func, **kwargs)`) that returns legacy and
    corrected spectra side by side. The scripts in `notes/theory_audit_checks/` are a starting point.
- Bug fixes with no scientific content (e.g. the `math.erf` crash in F8) don't need a legacy path, but still get
  documented.
- **Comparison notebook per fix** (user instruction, 2026-10-27). `examples/fix_comparisons/` holds one notebook per
  finding. Each plots legacy vs. corrected crawfish (and the reference from `fixcompare.Reference`), saves its
  figures to `figures/`, and ends with assert-based checks. Notebooks for fixes that aren't implemented yet already
  run: they compare current crawfish with the reference and carry "known-bug" checks. When a fix lands they switch
  over automatically through `fixcompare.LEGACY_FLAGS` / `fix_available()`. Use the flag names listed there, or
  update them, and add a notebook for any new result-changing fix. See `examples/fix_comparisons/README.md`.

1. **Tests first** (no behaviour change), in `tests/funcs/`:
   (a) after the phase fix, Im(T*_s T_p) ≈ 0 at Γ for N2;
   (b) band-energy sum rule within 1% for N2;
   (c) N2 px–px, s–px, py–py ICOHP all negative (bonding);
   (d) synthetic SSH chain (as in Check 4) for R-resolution and cell-choice invariance;
   (e) ∫DOS = Σw·nbands for Gaussian and LTI;
   (f) `fillings` file vs computed occupations agree (needs a no-spin fixture; ask the user for one, or generate
       one with JDFTx).
2. **Phase fix (F1):** l_u from the header, applied in `ElecData._set_proj_tju` for JDFTx data. Stage "matrices".
3. **New matrices (F2):** add `get_s_h_t_uv_complex(proj_tju, e_tj)` plus coefficients C (Löwdin or S⁻¹T) and
   D_t, as a one-pass numba kernel in `core/operations/matrix.py`, following the `_jit` pattern. Replace
   `_get_gen_tj_coef_uu/_tuu` with a complex-coefficient kernel taking `(C, M_t or M_R·phase, orbs_u, orbs_v, wk)`.
   Update `funcs/pcohp.py::_compute_*_tj`. Update `cache_mats`, `edata_setting_key_shorthand_dict` and
   `get_edata_settings_str` for any new setting (CLAUDE.md warns that missing these returns stale cache). Stage
   "matrices".
4. **R-resolution (F3):** a pair-to-R map (minimum image by default), M(R) by Fourier sum over the full grid, and
   `R` added to `func_args_dict`. Requires step 5's grid check.
5. **Inputs and weights (F4, F5, F8):** fillings /2 for no-spin; N_e check; raise for pair functions on reduced
   meshes; fix `get_wk_t`; fix `calculate_filling_*`. Stage "matrices" (occupations feed D).
6. **Normalization (F6):** Gaussian and LTI scale. Stage "spectra".
7. **Diagnostics (2.4):** `ElecData.charge_spilling`, `orbital_completeness`, `sum_rule_residual`, with a warning
   above a threshold (e.g. 5%).
8. **Docs:** README formalism and required JDFTx flags (`symmetries none`; `band-projection-params yes no` for
   COHP/COBI, `no no` for COOP; enough `elec-n-bands`); remove the false pCOMO claim.

Upstream JDFTx (only if wanted later, U1): flag-gated i^l phase in `dumpProjections`; optional dump of
`psi ^ Opsi` (AO overlap per k) and `Opsi ^ (H psi)`; binary or `%.15le` projections.

---------------------------------------------------------------------------------------------------------------

Additional evidence from the comparison notebooks (2026-10-27): with `rattle_eigenvals=True` the N2 LTI DOS
integral is 1.104 ≈ 30/27, exactly the extra 1/N_k of F6. In the current code (after F1), pCOBI px–px is
−0.214 vs. a reference of +0.173: F2 also flips COBI signs.

## 5. Open questions / not verified this session

- Q1. COHP basis choice: should the default be the Löwdin band-projected basis (energy-zero invariant, matches
  current `los_orbs=True`) or the non-orthogonal basis (closer to LOBSTER)? On N2 they agree in sign and roughly
  in size (A–B total −0.867 vs −0.833 Ha). This is the user's call.
- Q2. Does `pymatgen ... JDFTXOutfile.structure` preserve JDFTx's unwrapped ion coordinates? R bookkeeping (F3)
  needs the exact τ used in `Vnl_calc`. Alternatively, parse the `ion` lines from the outfile directly
  (`coords-type` matters).
- Q3. COBI details (spin factor, sign convention, whether LOBSTER evaluates it in an orthonormalized basis): check
  Müller et al. 2021 before implementing M=D.
- Q4. LOBSTER's exact requirements on VASP symmetry (ISYM) and its H(k) reconstruction formula for
  nbands > nAO: from memory, it reconstructs H from projected coefficients and eigenvalues, consistent with
  section 2.3. Re-read Deringer 2011 / Maintz 2013 to confirm.
- Q5. Whether JDFTx's `atpos` can be changed by symmetrization (`symmetries automatic` may snap positions). This
  is moot if `symmetries none` becomes required.

---------------------------------------------------------------------------------------------------------------

## 6. Reproducing the evidence

Scripts are in `notes/theory_audit_checks/`. They import the installed crawfish and run against the repo's test
data. ElecData writes `.crawfish_cache/` into the calc dir, which is gitignored.

```sh
python notes/theory_audit_checks/check_phase.py   tests/files/io/example_calc_dirs/N2_bare_min   # Check 1 (F1)
python notes/theory_audit_checks/compare_cohp.py  tests/files/io/example_calc_dirs/N2_bare_min   # Check 2 (F2, F7, sum rule, basis/energy-zero)
python notes/theory_audit_checks/check_misc.py    tests/files/io/example_calc_dirs/N2_bare_min   # Check 3 (F6, F8, spilling)
python notes/theory_audit_checks/check_image.py                                                  # Check 4 (F3)
```

Test system: N2 along x (atoms at x = 13.896 and 11.832 bohr, in a 24.6 bohr cubic box), GBRV v1.5 USPP, PBE,
z-spin, 3×3×3 k (`symmetries none`, 27 k per spin), 15 bands, 8 ortho-orbital projections (s, py, pz, px per atom),
Fermi smearing 0.001 Ha.

Recorded outputs (2026-10-27, crawfish 3efe98f, defaults los_orbs=True, p_uu_consistent=True, norm_eigs_for_h=True).
The crawfish columns predate the F1 fix. Reproduce them with `compare_cohp.py <dir> --legacy`
(sets `legacy_orbital_phase=True`); `tests/core/test_orbital_phase.py` pins them.

```
Check 1   Gamma, band 4 (3σg, e=-0.3695): Re s_A*px_B = 0.00000, Im = 0.22736;  all k: Σ|Re| 0.1781 vs Σ|Im| 114.2465
Check 2   pair    crawfish IpCOHP  crawfish IpCOMO  reference ICOHP (Löwdin, phase-fixed)
          s-s        -0.22098        0.40719          -0.11131
          s-px        0.00000       -0.00000          -0.13937
          px-px      +0.15313       -0.58770          -0.11823
          py-py      -0.11463        0.99778          -0.17931
          sum rule: Σ ICOHP = -5.45895 Ha vs band energy -5.47598 Ha
          non-orthogonal basis A-B total: -0.8329 (H from e), -0.7859 (H from e-mu); ICOOP A-B +0.4910 (artifact, ortho data)
Check 3   ∫DOS gauss 0.168, lti 0.586, expected 30;  nobroad fillings [0,0,1,1] (expected [1,1,0,0]);  gauss filling TypeError
          charge spilling 2.06%;  orbital completeness s 0.795, py/pz 0.974, px 0.688
Check 4   tau_B=+0.2: crawfish IpCOMO 0.9771 (short bond);  tau_B=-0.8: 0.1517 (long bond);  R-resolved values identical in both
```

---------------------------------------------------------------------------------------------------------------

## 7. References

### Code (paths relative to repo roots; cite by function, not line)

- JDFTx `jdftx/electronic/DumpExcitationsMoments.cpp::dumpProjections`: projection definition, ortho option, file format.
- JDFTx `jdftx/electronic/IonInfo.cpp::IonInfo::getAtomicOrbitals`; `jdftx/electronic/SpeciesInfo_atomic.cpp::SpeciesInfo::setAtomicOrbitals`, `::atomicOrbitalOffset`, `::nAtomicOrbitals`: orbital construction and order, spin copies.
- JDFTx `jdftx/electronic/SpeciesInfo.cpp::SpeciesInfo::setPsi`: O·psi with USPP augmentation, normalization ⟨ψ|O|ψ⟩=1.
- JDFTx `jdftx/electronic/SpeciesInfo_internal.h::Vnl_calc`: Bloch-sum phase, no i^l.
- JDFTx `jdftx/core/RadialFunction.cpp::RadialFunctionR::transform`: real Bessel transform.
- JDFTx `jdftx/core/SphericalHarmonics.h` (`YlmInternal::Ylm<lm>`): real Ylm order and signs.
- JDFTx `jdftx/wannier/WannierMinimizer.cpp` (atomic-orbital trial functions): `lPhase = cis(0.5*M_PI*l)`, i.e. JDFTx's own correction.
- JDFTx `jdftx/electronic/ElecInfo.cpp::ElecInfo::setup`, `::kpointsReduce`, `::smear`, `::printFillings`: spinWeight, weights, smearing, fillings convention.
- JDFTx `jdftx/electronic/Dump.cpp` (Fillings and BandProjections blocks): fillings ×2 for SpinNone.
- JDFTx `jdftx/electronic/Symmetries.cpp::Symmetries::reduceKmesh`; `jdftx/commands/symmetries.cpp`: defaults.
- crawfish `src/crawfish/core/operations/matrix.py` (`_get_h_p_s_t_uu_and_uu_jit`, `_get_gen_tj_coef_uu_jit`, `_get_gen_tj_coef_tuu_jit`, `los_projs_for_orbs`, `normalize_square_proj_tju`).
- crawfish `src/crawfish/core/elecdata.py` (`ElecData.occ_tj`, `alloc_elec_data`, `get_best_bands`, `calculate_filling_*`, `mu`).
- crawfish `src/crawfish/io/data_parsing.py` (`get_wk_t`, `get_kfolding`, `_get_lti_allowed`, `spintype_nspin`).
- crawfish `src/crawfish/funcs/general.py`, `core/operations/vector.py`, `core/operations/scalar.py::gauss`.
- libtetrabz `dos()` normalization: checked numerically in this session (tight-binding cosine band, 6³ grid).
- pymatgen fork `pymatgen/io/jdftx/jdftxoutfileslice.py` (`_get_broadeningvars`, `_get_mu`, `_mu_backup`), `jelstep.py::_set_mu`.

### Literature (cited from memory. Titles, venues and DOIs were not re-fetched this session, so verify before quoting externally)

- R. Dronskowski, P. E. Blöchl, "Crystal orbital Hamilton populations (COHP)...", J. Phys. Chem. 97, 8617 (1993). doi:10.1021/j100135a014
- T. Hughbanks, R. Hoffmann, "Chains of trans-edge-sharing molybdenum octahedra...", J. Am. Chem. Soc. 105, 3528 (1983). doi:10.1021/ja00349a027 (COOP)
- V. L. Deringer, A. L. Tchougréeff, R. Dronskowski, "Crystal Orbital Hamilton Population (COHP) Analysis As Projected from Plane-Wave Basis Sets", J. Phys. Chem. A 115, 5461 (2011). doi:10.1021/jp202489s
- S. Maintz, V. L. Deringer, A. L. Tchougréeff, R. Dronskowski, "Analytic projection from plane-wave and PAW wavefunctions and application to chemical-bonding analysis in solids", J. Comput. Chem. 34, 2557 (2013). doi:10.1002/jcc.23424
- S. Maintz, V. L. Deringer, A. L. Tchougréeff, R. Dronskowski, "LOBSTER: A tool to extract chemical bonding from plane-wave based DFT", J. Comput. Chem. 37, 1030 (2016). doi:10.1002/jcc.24300
- P. C. Müller, C. Ertural, J. Hempelmann, R. Dronskowski, "Crystal Orbital Bond Index: Covalent Bond Orders in Solids", J. Phys. Chem. C 125, 7959 (2021). doi:10.1021/acs.jpcc.1c00718
- W. V. Glassey, R. Hoffmann, "A comparative study of Hamilton and overlap population methods for the analysis of chemical bonding", J. Chem. Phys. 113, 1698 (2000). doi:10.1063/1.481971 (energy-zero dependence of COHP in non-orthogonal bases)
- D. Vanderbilt, "Soft self-consistent pseudopotentials in a generalized eigenvalue formalism", Phys. Rev. B 41, 7892 (1990). doi:10.1103/PhysRevB.41.7892
- K. Laasonen, A. Pasquarello, R. Car, C. Lee, D. Vanderbilt, Phys. Rev. B 47, 10142 (1993). doi:10.1103/PhysRevB.47.10142 (USPP orthonormality and overlap operator in practice)
- K. F. Garrity, J. W. Bennett, K. M. Rabe, D. Vanderbilt, "Pseudopotentials for high-throughput DFT calculations", Comput. Mater. Sci. 81, 446 (2014). doi:10.1016/j.commatsci.2013.08.053 (GBRV, used in the test data)
- D. Sánchez-Portal, E. Artacho, J. M. Soler, "Projection of plane-wave calculations into atomic orbitals", Solid State Commun. 95, 685 (1995). doi:10.1016/0038-1098(95)00341-X (spilling)
- P.-O. Löwdin, J. Chem. Phys. 18, 365 (1950). doi:10.1063/1.1747632; R. S. Mulliken, J. Chem. Phys. 23, 1833 (1955). doi:10.1063/1.1740588
- N. Marzari, A. A. Mostofi, J. R. Yates, I. Souza, D. Vanderbilt, Rev. Mod. Phys. 84, 1419 (2012). doi:10.1103/RevModPhys.84.1419 (projection + Löwdin initial guess A(A†A)^-1/2; same construction as `los_projs_for_orbs`)
- W. Ku, H. Rosner, W. E. Pickett, R. T. Scalettar, Phys. Rev. Lett. 89, 167204 (2002). doi:10.1103/PhysRevLett.89.167204 (projected Wannier functions)
- M. Kawamura, Y. Gohda, S. Tsuneyuki, Phys. Rev. B 89, 094515 (2014). doi:10.1103/PhysRevB.89.094515 (libtetrabz)
- C. A. Coulson, Proc. R. Soc. Lond. A 169, 413 (1939). doi:10.1098/rspa.1939.0006 (mobile order; crawfish's comment cites p. 419)
- R. Sundararaman et al., "JDFTx: software for joint density-functional theory", SoftwareX 6, 278 (2017). doi:10.1016/j.softx.2017.10.006

# Theory audit: crawfish vs. JDFTx source, USPP theory and the COHP formalism

Started 2026-10-27. Living document. Update the status column and add dated entries instead of rewriting history.

Scope: compare what crawfish computes (crawfish `master` at `3efe98f`) with (a) what JDFTx actually writes
(JDFTx source in `../jdftx/jdftx`), (b) ultrasoft-pseudopotential (USPP) theory, and (c) the COOP/COHP/COBI
formalism. The task prioritizes fixes that change only crawfish. JDFTx changes are listed separately as
lower-priority upstream options.

Code citations use `path::function` and avoid line numbers, which drift. Literature citations are listed under
References at the end. Every numbered "Check" can be rerun with the scripts in `notes/theory_audit_checks/`.

---------------------------------------------------------------------------------------------------------------

## 0a. Status (2026-10-27)

F1–F7 are committed (F5 includes default time-reversal unfolding; F7 = `2de60d2`). F8 (occupations without a
`fillings` file; bug fix, no legacy switch or `result_changes.py` entry) is done on `theory_check`, uncommitted. The
next findings are F9 (spinors), F10 (misc), F11 (too few bands; add the section 2.4 diagnostics), and F12
(point-group unfolding, future).

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
| F2 | P0 | H/S/P "matrices" are built from `(Re[T*_u T_v])^2`, so they are elementwise non-negative (S, P) or non-positive (H, occupied). The sign of the true matrix element is lost. The bonding/antibonding call is wrong whenever the true H_uv or S_uv is positive (e.g. ppσ). In N2, px–px σ comes out antibonding. | Code + Check 2 | No | **fixed** in `3d9ab1b` (2026-10-27); legacy via `legacy_matrices=True`; `result_changes.py` entry added |
| F3 | P0 | Pair weights are the R=0 lattice-image element. k-point coordinates are never used, so there is no Bloch phase. The A–B result depends on how atoms are wrapped into the cell, and may describe a far-away image instead of the bond the user means. | Code + Check 4 | No | **fixed** in `ceffa10` (2026-10-27); legacy via `legacy_lattice_image=True`; `result_changes.py` entry added |
| F4 | P1 | The `fillings` file is in [0,2] for `spintype no-spin`, but occupations crawfish computes itself are in [0,1]. k-weights already sum to 2. P/H (hence pCOHP/pCOBI) double when a fillings file exists for unpolarized runs. | JDFTx `Dump.cpp` | No | **fixed** in `756f9e0` (2026-10-27); legacy via `legacy_fillings=True`; `result_changes.py` entry added |
| F5 | P1 | JDFTx's default `symmetries automatic` reduces the k-mesh by point group + time reversal. Atom-pair and m-resolved quantities from a reduced mesh are symmetry-averaged, and R=0 pairing gets scrambled. Crawfish only warns (and `README` never asks for `symmetries none`). Separately, `get_wk_t` silently replaces symmetry weights with uniform ones for spin-polarized reduced meshes. | JDFTx `Symmetries.cpp`, crawfish `data_parsing.py` | No | **fixed** in `cc5f71e` (2026-10-27), with time-reversal unfolding; legacy via `legacy_reduced_kmesh=True`; `result_changes.py` entry added |
| F6 | P1 | Spectrum normalization: the Gaussian kernel `exp(-(E-e)^2/sig)` is unnormalized (integral = sqrt(pi*sig)). For LTI, libtetrabz weights already include 1/N_k, and crawfish multiplies by w_k again. The two methods give different, arbitrary scales, and neither integrates to a state count. | Check 3 | No | **fixed** in `44cf908` (2026-10-27); legacy via `legacy_normalization=True`; `result_changes.py` entry added |
| F7 | P1 | pCOOP is basis-dependent in a way the docs don't say. With `band-projection-params yes ...` (the README recommendation) the true AO overlap is the identity, so physical COOP between different orbitals is zero, and anything non-zero is a band-truncation artifact. The README's claim that pCOMO's sign identifies bonding is false for pairs with negative overlap (ppσ) and for odd-Δl pairs, where pCOMO's sign depends on orbital orientation. | Check 2 | No | **fixed** in `2de60d2`: docs, warnings, `orbital_completeness_u`; no numbers change, so no legacy switch or `result_changes` entry |
| F8 | P2 | `elecdata.py::calculate_filling_nobroad` uses `np.heaviside(eig, efermi)`, which gives inverted fillings and ignores mu. `calculate_filling_gauss/mp1/cold` call `math.erf` on arrays and raise TypeError. Both paths are only reached when no `fillings` file exists. Also found while fixing: JDFTx's spelling `Cold` was rejected, and runs without smearing print no mu, so that path crashed. | Check 3 | No | **fixed** on `theory_check` (uncommitted, 2026-10-27): vectorized helpers, constant fillings without smearing; bug fix, so no legacy switch or `result_changes` entry |
| F9 | P2 | Spinor modes (`vector-spin`, `spin-orbit`) are mis-handled. `data_parsing.py::spintype_nspin` maps them to nspin=2, but JDFTx has one state per k-point there. Orbital labels ignore JDFTx's per-m spin copies or j-resolved orbitals. | JDFTx `ElecInfo.cpp`, `SpeciesInfo_atomic.cpp` | No | open |
| F10 | P2 | Misc. `los_projs_for_orbs` uses an absolute eigenvalue cutoff of 1e-15 on complex64 data. `trim_excess_bands` selects bands globally by sum of \|T\| (not \|T\|^2, and not per k) and then Sinkhorn-normalizes. `p_uu_consistent` is a non-standard per-band renormalization. The README formula for S (\|T*T\|^2) doesn't match the code ((Re T*T)^2). In the pymatgen fork, the `mu` fallback path (`_mu_backup`) is Hartree while the main path is eV, and crawfish divides by Ha_to_eV unconditionally. | Code | No (pymatgen-fork fix for mu) | open |
| F11 | P1 | Too few bands (`elec-n-bands` ≈ number of orbitals) leave orbitals unspanned at some k-points. The band-space Löwdin step (and the non-orthogonal S⁻¹) is then ill-conditioned, and pair spectra become silently unreliable. In graphene with 8 bands, the three equivalent bonds differed by 5% and the non-orthogonal H reached ~79 Ha. No warning is given. | Graphene runs, 2026-10-27 | No | open (README now warns) |
| F12 | P2 | Meshes reduced by a point group (JDFTx `symmetries automatic`) can't give single-bond or m-resolved results; crawfish guards them (F5). Full unfolding with the symmetry operations would lift this. Time-reversal-only reductions are already unfolded (F5). | F5 work | No (`dump End Symmetries` exists) | open, future work |
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
  **crawfish's `calculate_filling_*` formulas match** (the array/erf issue in F8 is fixed). The broadening
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

#### F3 explained: the pCOHP equation before and after, and the misconceptions it clears up

Written 2026-10-27 after working through F3 with the user, so the reasoning isn't lost. The notation follows the
README:

- $T_{\mu,j}(t)=\langle\phi_\mu|\hat O|\psi_j(t)\rangle$ is `proj_tju[t,j,u]`. The README writes
  $\bra{\phi_\mu}\psi_j(t)\rangle$; $\hat O$ is the ultrasoft overlap operator, which JDFTx applies, and
  $\hat O=1$ for norm-conserving pseudopotentials.
- $\epsilon_j(t)$ is `e_tj`, $f_j(t)$ is `occ_tj` and $w_t$ is `wk_t`. The state index t is one (k-point, spin).
- μ runs over orbitals on atom A and ν over orbitals on atom B. $\mu_F$ is the Fermi level (`edata.mu`), written
  that way to avoid a clash with the orbital index.
- $C_{\mu,j}(t)$ are the expansion coefficients; they equal $T$ after Löwdin orthonormalization (`los_orbs=True`).

##### The old (wrong) equation: default `get_pcohp(..., state_sep=False)`, up to and including `3efe98f`

$$
pCOHP^{\,old}_{AB}(E)=\sum_{\mu\in A,\,\nu\in B} H^{old}_{\mu\nu}\sum_{j,t} w_t\,\mathrm{Re}\left[\langle\psi_j(t)|\hat O|\phi_\mu\rangle\langle\phi_\nu|\hat O|\psi_j(t)\rangle\right]\delta\big(E-\epsilon_j(t)\big)
$$

$$
H^{old}_{\mu\nu}=\sum_{t',\,j'} w_{t'}\,f_{j'}(t')\,\big(\epsilon_{j'}(t')-\mu_F\big)\,
\frac{\Big(\mathrm{Re}\left[\langle\psi_{j'}(t')|\hat O|\phi_\mu\rangle\langle\phi_\nu|\hat O|\psi_{j'}(t')\rangle\right]\Big)^2}
{\sum_{\mu',\nu'}\Big(\mathrm{Re}\left[\langle\psi_{j'}(t')|\hat O|\phi_{\mu'}\rangle\langle\phi_{\nu'}|\hat O|\psi_{j'}(t')\rangle\right]\Big)^2}
$$

The −μ_F comes from `norm_eigs_for_h=True` and the per-band denominator from `p_uu_consistent=True`; both were
defaults. The integrated value is $\sum_{j,t} f_j(t)\,(\ldots)$ without the δ.

Three separate defects sit in these two lines:

- **F1:** for odd l, JDFTx's $\langle\phi_\mu|$ is $i^l$ times a real orbital, so $\mathrm{Re}[\ldots]\approx 0$ for
  s–p and p–d pairs.
- **F2:** $H^{old}$ isn't $\langle\phi_\mu|\hat H|\phi_\nu\rangle$. It is squared (no sign), weighted by occupation,
  normalized per band, and already summed over every band and state.
- **F3:** $H^{old}_{\mu\nu}$ is the same number at every state t, so it factors out of the sum over k:
  $IpCOHP^{old}_{AB}=\sum_{\mu\nu}H^{old}_{\mu\nu}\cdot\big[\sum_{j,t}w_t f_j(t)\,\mathrm{Re}(T^*_{\mu,j}T_{\nu,j})\big]$.
  The bracket is the R = 0 element of the real-space density matrix: the bond order between μ and the copy of ν
  at $\tau_B$ as written in the input, whatever its distance. Any weight that is constant across states would do
  the same, including a correctly signed one, and `np.ones` in legacy pCOMO.

##### The old code (verbatim at `3efe98f`; still in the repo as the legacy path, `legacy_matrices=True`)

`src/crawfish/funcs/pcohp.py`:

```python
def _compute_pcohp_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        state_sep: bool,
        ):
    if state_sep:
        # h_t_uu already carries the k-point weight, so it is not applied again
        pcohp_tj = _get_gen_tj_coef_tuu(edata.proj_tju, edata.h_t_uu, orbs_u, orbs_v)
    else:
        h_uu = edata.h_uu
        pcohp_tj = _get_gen_tj_coef_uu(edata.proj_tju, edata.wk_t, h_uu, orbs_u, orbs_v)
    return pcohp_tj
```

`src/crawfish/core/operations/matrix.py`. This is the weight kernel; `coef_uu` is `edata.h_uu`, one number per
orbital pair, and the same at every t:

```python
@jit(nopython=True)
def _get_gen_tj_coef_uu_jit(gen_tj, coef_uu, proj_tju, wk_t, orbs_u, orbs_v):
    _t, _j, _ = np.shape(proj_tju)
    for t in range(_t):
        for j in range(_j):
            val = REAL_DTYPE(0)
            for u in orbs_u:
                for v in orbs_v:
                    if not u == v:
                        val += REAL_DTYPE(coef_uu[u, v]) * REAL_DTYPE((np.conj(proj_tju[t, j, u]) * proj_tju[t, j, v]).real)
            gen_tj[t, j] += val * REAL_DTYPE(wk_t[t])
```

`src/crawfish/core/operations/matrix.py` builds `h_uu`. The wrapper `get_h_p_s_t_uu_and_uu` first applies
`e_tj = e_tj - mu` when `norm_eigs_for_h` is True. Note the square at `s_tj_uu[u, v] = x * x`, the occupation in
`p_coef`, and the accumulation over every t into `h_uu`:

```python
# error_model="numpy" so a band with zero projections gives nan (as the numpy version does) instead of raising
@jit(nopython=True, error_model="numpy")
def _get_h_p_s_t_uu_and_uu_jit(h_t_uu, p_t_uu, s_t_uu, h_uu, p_uu, s_uu, proj_tju, e_tj, occ_tj, wk_t, p_sc):
    _t, _j, _u = np.shape(proj_tju)
    s_tj_uu = np.zeros((_u, _u), dtype=REAL_DTYPE)
    h_acc_uu = np.zeros((_u, _u), dtype=REAL_DTYPE)
    p_acc_uu = np.zeros((_u, _u), dtype=REAL_DTYPE)
    s_acc_uu = np.zeros((_u, _u), dtype=REAL_DTYPE)
    for t in range(_t):
        h_acc_uu[:, :] = 0.0
        p_acc_uu[:, :] = 0.0
        s_acc_uu[:, :] = 0.0
        for j in range(_j):
            # Only the upper triangle is filled and accumulated - s_tj_uu is symmetric
            s_sum = REAL_DTYPE(0)
            for u in range(_u):
                for v in range(u, _u):
                    # Matches get_s_tj_uu: the real part of the overlap is taken before squaring
                    x = REAL_DTYPE((np.conj(proj_tju[t, j, u]) * proj_tju[t, j, v]).real)
                    s_tj_uu[u, v] = x * x
                    s_sum += s_tj_uu[u, v] if u == v else REAL_DTYPE(2) * s_tj_uu[u, v]
            p_coef = REAL_DTYPE(occ_tj[t, j]) * REAL_DTYPE(wk_t[t])
            if p_sc:
                p_coef /= s_sum
            h_coef = p_coef * REAL_DTYPE(e_tj[t, j])
            for u in range(_u):
                for v in range(u, _u):
                    s_acc_uu[u, v] += s_tj_uu[u, v]
                    p_acc_uu[u, v] += p_coef * s_tj_uu[u, v]
                    h_acc_uu[u, v] += h_coef * s_tj_uu[u, v]
        for u in range(_u):
            for v in range(u, _u):
                h_t_uu[t, u, v] = h_acc_uu[u, v]
                h_t_uu[t, v, u] = h_acc_uu[u, v]
                p_t_uu[t, u, v] = p_acc_uu[u, v]
                p_t_uu[t, v, u] = p_acc_uu[u, v]
                s_t_uu[t, u, v] = s_acc_uu[u, v]
                s_t_uu[t, v, u] = s_acc_uu[u, v]
                h_uu[u, v] += h_acc_uu[u, v]
                p_uu[u, v] += p_acc_uu[u, v]
                s_uu[u, v] += s_acc_uu[u, v]
    for u in range(_u):
        for v in range(u + 1, _u):
            h_uu[v, u] = h_uu[u, v]
            p_uu[v, u] = p_uu[u, v]
            s_uu[v, u] = s_uu[u, v]
```

The old `state_sep=True` path used the per-state `h_t_uu[t]` (`h_acc_uu`), so its weight did vary with k. But it
is real and built from squares: it has no Bloch phases, and its variation over k doesn't encode hoppings to
particular copies.

##### The equation now used (F1 + F2 + F3; default settings)

The coefficients are phase-corrected, $T_{\mu,j}\to i^{l_\mu}T_{\mu,j}$ (F1), and Löwdin-orthonormalized by
default, so $C=T$. For each state t, the Hamiltonian of the orbitals sums over **all** bands j′ (occupied and
empty, with no occupations). This is the spectral form $\hat H=\sum_{j'}\hat O|\psi_{j'}\rangle\epsilon_{j'}\langle\psi_{j'}|\hat O$ of
the ultrasoft eigenproblem (F2):

$$
H_{\mu\nu}(t)=\sum_{j'}\langle\phi_\mu|\hat O|\psi_{j'}(t)\rangle\;\epsilon_{j'}(t)\;\langle\psi_{j'}(t)|\hat O|\phi_\nu\rangle\;\approx\;\langle\phi_{\mu,k_t}|\hat H|\phi_{\nu,k_t}\rangle
$$

Here $\phi_{\nu,k}=\sum_R e^{ik\cdot R}\phi_\nu(r-\tau_\nu-R)$ (JDFTx's Bloch sum), so
$H_{\mu\nu}(t)=\sum_R e^{2\pi i k_t\cdot R}H_{\mu\nu}(R)$. The hopping to the copy of ν at $\tau_\nu+R$ is the
Fourier component (F3):

$$
H_{\mu\nu}(R)=\frac{\sum_{t\in s} w_t\,e^{-2\pi i k_t\cdot R}\,H_{\mu\nu}(t)}{\sum_{t\in s} w_t}
$$

The pCOHP of A with **one** copy of B (R = nearest image per atom pair by default, or any lattice vector) is

$$
pCOHP_{AB}(E;R)=\sum_{\mu\in A,\,\nu\in B}\;\sum_{j,t} w_t\,\mathrm{Re}\left[C^*_{\mu,j}(t)\;H_{\mu\nu}(R)\,e^{2\pi i k_t\cdot R}\;C_{\nu,j}(t)\right]\delta\big(E-\epsilon_j(t)\big)
$$

The pCOHP of A with **every** copy of B (`image="all"`, or the old `state_sep=True`) is

$$
pCOHP_{AB}(E;\mathrm{all})=\sum_{\mu\in A,\,\nu\in B}\;\sum_{j,t} w_t\,\mathrm{Re}\left[C^*_{\mu,j}(t)\;H_{\mu\nu}(t)\;C_{\nu,j}(t)\right]\delta\big(E-\epsilon_j(t)\big)=\sum_R pCOHP_{AB}(E;R)
$$

The last equality holds exactly for one R per residue class of the k-mesh. The tests check it on N2's 3×3×3 mesh
and graphene's 23×23×1 mesh.

Code: `core/operations/matrix.py::get_h_s_d_t_uv` builds $H(t)$, `::get_mat_r_suv` gives $H(R)$, and
`funcs/pcohp.py::_compute_signed_tj` assembles $H_{\mu\nu}(R)e^{2\pi i k_t\cdot R}$ per atom pair (or $H(t)$ for
"all"). `::_get_gen_tj_block` evaluates $w_t\sum\mathrm{Re}[C^* M C]$. With `legacy_lattice_image=True`, R = 0
is used for every pair, which is still correctly signed: that is the F2-era behaviour.

Graphene, for scale (IpCOHP, Ha): each of the three nearest C1–C2 bonds is −0.146 (two of them inter-cell), each
2.91 Å copy is −0.026, and "all" is −0.578. Legacy (R = 0 only) gives −0.145. That is one bond, and right only
because the input happened to place a bonded copy in the same cell. With C2 written one cell over, legacy gives
−0.026.

##### Misconceptions worked through, and the corrections

1. **"JDFTx's projections didn't include each k-point's phase."** They do. `Vnl_calc` builds every orbital with
   `cis(-2π pos·(k+G))`, i.e. the Bloch sum above. $T_\nu(k)$ is the amplitude on the copy of ν at $\tau_\nu$,
   and the copy in cell R has amplitude $e^{ik\cdot R}T_\nu(k)$. Nothing was missing from the files; crawfish
   threw the information away by pairing $T^*_\mu T_\nu$ with a weight that doesn't vary with k. No JDFTx change
   is needed.
2. **"Integrating over k reduces the bond to the same cell."** The integral isn't the cause. At every single k,
   $T^*_\mu T_\nu$ is already the amplitude product for the same-cell pair; the pair with the copy at R is
   $T^*_\mu T_\nu e^{ik\cdot R}$. A sum over k of the bare phase $e^{ik\cdot R}$ vanishes for R ≠ 0, but a sum of
   $e^{ik\cdot R}$ times a k-dependent weight keeps that weight's −R component (convolution, or Parseval). Each
   pair therefore keeps its own nonzero bond order. The old code supplied a weight that doesn't depend on k, whose
   only Fourier component is R = 0.
3. **"The k-dependence of $H_{\mu\nu}(k)$ is a curiosity."** It is essential: $H_{\mu\nu}(k)=\sum_R e^{ik\cdot R}H_{\mu\nu}(R)$
   encodes the hopping to every copy. Averaging it over k (with correct signs) gives $H_{\mu\nu}(R{=}0)$, which is
   valid, but only for the same-cell pair.
4. **"To isolate one copy, rotate $H(k)$ by $e^{-ik\cdot R}$ at each k."** Not at fixed k:
   $H(k)e^{-ik\cdot R}=\sum_{R'}H(R')e^{ik\cdot(R'-R)}$ still contains every copy, and paired with
   $C^*C$ it mixes pairs. Do two steps instead. First Fourier-filter, $H(R)=\overline{e^{-ik\cdot R}H(k)}$, which
   no longer depends on k. Then attach $e^{+ik\cdot R}$ to the coefficient product at each k.
5. **"Then use a band-resolved $h[\mu,\nu,t,j]$."** No. The band index plays two roles. j (in $C_{\mu,j}$,
   $\delta(E-\epsilon_j)$ and $f_j$) labels the state being analysed. j′ (inside $H_{\mu\nu}(t)$) is the spectral
   representation of the operator: all bands, no occupations. Keeping only j′ = j gives, in the Löwdin basis,
   $\mathrm{Re}[C^*_{\mu j}(T_{\mu j}\epsilon_j T^*_{\nu j})C_{\nu j}]=\epsilon_j|T_{\mu j}|^2|T_{\nu j}|^2$.
   That has the sign of $\epsilon_j$ for every pair, so all bonding information is lost. The j′ ≠ j cross terms
   carry it; they cancel only in $\sum_{\mu\nu}$, which is the band-energy sum rule.

| you want | weight on $\mathrm{Re}[C^*_{\mu,j}(t)\ \cdot\ C_{\nu,j}(t)]$ |
|---|---|
| μ with every copy of ν (`image="all"`) | $H_{\mu\nu}(t)$ (`h_t_uv[t]`) |
| μ with the copy at R (`"nearest"` or a lattice vector) | $H_{\mu\nu}(R)\,e^{2\pi i k_t\cdot R}$ |
| μ with the same-cell copy only (`image=(0,0,0)`, or `legacy_lattice_image=True`) | $H_{\mu\nu}(R{=}0)$, constant over t |
| (old, wrong) | $H^{old}_{\mu\nu}$: constant over t, squared, occupation-weighted |

**Implementation record for F3 (2026-10-27, branch `theory_check`, not yet committed):**
- API: every pair function (`get_[i]pcohp`, `get_[i]pcoop`, `get_[i]pcobi`, `get_[i]pcomo`) takes
  `image="nearest" | "all" | (n1, n2, n3)`.
  - `state_sep=True` stays as an alias for `"all"` and takes precedence. `get_ipcohp` also gained `state_sep`.
  - pCOMO rejects `"all"`: Σ_R of a unit matrix isn't a bond quantity.
  - A new public `get_pair_images(edata, idcs1, idcs2, ..., image)` returns one dict per atom pair:
    atom labels, R, and distance in Å.
  - The normalized image (`"nearest"`, `"all"` or a list of ints) is stored in each spectrum's `func_args_dict`.
- `utils/lattice.py` (new):
  - `get_orb_atom_idx_u` (orbital → structure site through `kmap`/`orbs_idx_dict`).
  - `get_nearest_image`: searches ±2 around the minimum-image guess and returns the strictly nearest image (exact
    ties go to the smallest |R|). Images within `IMAGE_TIE_TOL` = 0.01 Å of it count as ties for the warning. The
    tolerance started at 1e-4 Å. In the graphene fixture, though, relaxation leaves the three equivalent C–C bonds
    0.003 Å apart, and with 1e-4 Å "nearest" silently picked one of them.
  - `get_pair_image_map`: one R per atom pair, plus a warning when the nearest image isn't unique (e.g. 4
    equivalent images in small cells).
  - `normalize_image`.
- `core/operations/matrix.py`:
  - `get_mat_r_suv(mat_t_uv, wk_t, ks_t, nspin, R)`: per-spin Fourier sum M_s(R) = Σ w_t e^{−2πik·R} M_t / Σ w_t.
    Its docstring covers aliasing by the k-mesh.
  - `_get_gen_tj_block`: weight kernel over an effective (t, |A|, |B|) matrix. That matrix is the state-resolved
    block for `"all"`, or M_ab(R_ab) e^{2πik·R_ab} assembled per atom pair in `funcs/pcohp._compute_signed_tj`.
  - The old `_get_gen_tj_coef_tuv` is no longer used by the pipeline.
- `core/elecdata.py`: setting `legacy_lattice_image` (cache key `lgri`). Its setter only re-points the caches,
  because the matrices don't change. `get_mat_r_suv(name, R)` added.
- Legacy combinations:
  - With `legacy_lattice_image=True`, "nearest" means R=(0,0,0) for every pair, i.e. exactly the F2-era
    behaviour (R=0 uses `get_mat_r0_suv`).
  - With `legacy_matrices=True`, image selection isn't possible: `"all"` maps to the legacy `state_sep=True`,
    `(0,0,0)` and `"nearest"` give the legacy same-cell result, and any other vector raises.
- Weight caches are used for `"nearest"` and `"all"`, and bypassed for explicit vectors (CachedFunction keys only
  on orbitals).
- Mesh check: an R ≠ 0 on a symmetry-reduced mesh (JDFTx `lti_allowed` False) raises with a pointer to
  `symmetries none`. R = 0 still works on reduced meshes, as before (F5 is still open).
- Validation, with no periodic JDFTx fixture available (JDFTx isn't built here):
  - **Wrapping test on real data:** atom 2 of N2 moved by a lattice vector L, with the Bloch-sum gauge change
    T → e^{2πik·L}T derived from `Vnl_calc`. "nearest" gives identical pCOHP/pCOBI/pCOMO, and
    `get_pair_images` reports R = −L. A negative control (R = +L or 0 on the shifted data) gives ≈0, so the sign
    convention is really being tested.
  - **Discrete-Fourier identity:** summing R-resolved weights over R ∈ {−1,0,1}³ reproduces the "all images"
    weights exactly on the 3×3×3 mesh.
  - Far image (R=(1,0,0), 13 Å) ≈ 0.
  - SSH chain: crawfish's kernels match an independent reference in both cell choices.
  - **Periodic fixture (added 2026-10-27):** `tests/files/io/example_calc_dirs/graphene_1x1`, about 2 MB, copied
    from the user's `PZC_gen/PBE_GBRV_20Ha/bulks/Gr_1x1/lat_opt/jdftx_run`: bandProjections, eigenvals,
    fillings, kPts, out and in, without the macOS `._*` files. It is a 2-C primitive cell, GBRV USPP, PBE, CANDLE
    solvation, lattice-relaxed, z-spin, 23×23×1 mesh, `symmetries none`, 8 bands and 8 ortho s/p projections.
    - Of C1's three nearest C2 images (1.4557, 1.4564 and 1.4584 Å), two are inter-cell:
      R = (−1,−1,0), (0,−1,0) and (0,0,0). Each bond is IpCOHP ≈ −0.146 Ha.
    - The 2.91 Å shell adds −0.026 Ha per image. The sum over all images is −0.578 Ha, of which the same-cell
      copy (legacy) is only −0.145.
    - Orbital-resolved values differ per bond: px–px is −0.040 for R=0 and −0.0015 for the other two, while pz–pz
      (π) is −0.031 for all three.
    - With C2 wrapped by (1,0,0), legacy gives −0.026 (a 2.91 Å copy) while fixed stays at −0.146.
    - Tests added: three equivalent bonds plus the tie warning, m-resolved differences, the 529-class Fourier
      identity, and wrapping by (1,0,0) and (2,−1,0).
    - Alternatives considered: Pb (`bulks/Pb/...`, 4-atom fcc, 8×8×8, s/p/d) has a 28 MB bandProjections, too
      large. Al (4-atom fcc, 5.6 MB) would also work.
    - Still missing: an end-to-end check against an actual JDFTx re-run with wrapped coordinates. The wrapping
      tests use the Bloch-sum gauge change instead.
- N2 numbers are unchanged by F3: its nearest image is the same-cell copy. Legacy vs. fixed only differ once atoms
  are wrapped; with atom 2 one cell over, legacy all-all IpCOHP is −0.00004 vs. fixed −0.86678.
- Tests: `tests/core/test_lattice_images.py` (19 tests, N2 and graphene). Full suite: 136 passed.
- Docs: the README section on which periodic image (with the M(R) formula and aliasing note), CLAUDE.md pipeline
  step 3 and settings list, the F3 notebook (chain + N2 wrapping, legacy vs. fixed vs. reference), and a
  `get_pair_images` cell in `examples/N2.ipynb`.
- `result_changes.py`: the F2 entry was added (`3d9ab1b`). **The F3 entry still has to be added after this work is
  committed** (stage "matrices", since cached weight arrays change).

### F4: fillings-file factor 2 for no-spin (P1)

- JDFTx `Dump.cpp` writes `fillings` as F×2 for `SpinNone`. k-weights already include the factor 2. Crawfish
  `ElecData.occ_tj` reads the file as-is, but computes [0,1] when the file is absent. N_e = Σ w f is therefore 2×
  too large with a file for unpolarized runs. P, H and h_coef double. `get_generic_integrate(use_fillings=True)`
  divides by max(occ), which hides the problem for integrated spectra but not for matrices. (The test data is z-spin,
  so it doesn't exercise this: N_e = 10.000 there.)
- **Fix:** divide by 2 when `nspin == 1` and the run is not spinor (equivalently, divide by the JDFTx
  `spinWeight`). Sanity-check that Σ w f equals nElectrons from the outfile. Remove the `/ max(occ)` hack in
  `funcs/general.py::get_generic_integrate`.

**Implementation record for F4 (2026-10-27, branch `theory_check`, not yet committed):**
- `ElecData.occ_tj`: when read from a `fillings` file (not legacy), divides by `ElecData.spin_weight`, which is
  JDFTx's `spinWeight`: 2 for `outfile.spintype == "no-spin"`, 1 otherwise; for non-JDFTx data, 2 if nspin == 1.
  It then calls `_check_electron_count`, which warns if Σ w f differs from `outfile.total_electrons` by more than
  1%. Occupations computed from mu and the smearing were already in [0, 1] and are untouched.
- `funcs/general.py::get_generic_integrate(use_fillings=True)` no longer divides occupations by `max(occ_tj)`;
  the legacy path keeps the division.
- Setting `legacy_fillings` (default False, cache key `lgfl`). Its setter resets `_occ_tj`, `_e_tj` and `_nbands`
  so that band trimming re-runs on fresh arrays, then calls `_init_caches()` and `alloc_elec_data()`.
- **Fixture:** `tests/files/io/example_calc_dirs/N2_nospin`, about 60 KB, with a README. The drive search found
  no no-spin runs: 1,126 bandProjections were checked on `/Volumes/Ext4TB/scratch_backup` before stopping. So the
  calculation was run locally with `~/Desktop/jdftx_hold/build_master/jdftx -c 8` in 38 s. It is gas-phase N2
  with the same geometry and box as `N2_bare_min`, GBRV v1.2 `n_pbe_v1.2.uspp` from that build, no fluid, a
  2×2×2 mesh and `symmetries none`.
- **Results:** the raw file peaks at 2.0 and the weights sum to 2. Legacy N_e is 20 (outfile: 10); fixed is 10.
  IpCOBI N1–N2 is 2.978 legacy vs 1.489 fixed. IpCOHP is −0.8668 in both, because H has no occupations and the
  old max(occ) rescaling hid the factor in integrated spectra. Fixed no-spin agrees with the z-spin `N2_bare_min`
  (IpCOBI 1.493, IpCOHP −0.8668), and z-spin results are identical in legacy and fixed.
- README: a new "Occupations and k-point weights" section. The non-JDFTx instructions now say weights sum to 2
  (they said nspin, which undercounts unpolarized data by half). CLAUDE.md conventions are updated too.
  `_get_arbitrary_wk` still gives Σw = nspin (F10, open).
- Tests: `tests/core/test_fillings.py` (8 tests: z-spin invariance, the electron-count warning, integration
  without max rescaling, no-spin halving, file vs. computed occupations, legacy pCOBI ×2, and no-spin vs.
  z-spin agreement). Full suite: 144 passed.
- Notebook: the F4 section of `F4_F8_inputs_and_normalization.ipynb` compares legacy and fixed for no-spin and
  z-spin (`figures/F4_nospin_fillings.png`).
- `result_changes.py`: the F3 entry was added (`ceffa10`). **The F4 entry still has to be added after this work is
  committed** (stage "matrices": occupations feed D and the cached weights).

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

**Implementation record for F5 (2026-10-27, branch `theory_check`, not yet committed):**
- Weights:
  - `io/data_parsing.py::get_wk_t(..., legacy=False)` compares the number of weights with nspin·nk (one per
    state). Legacy compared it with nk, which always failed for spin-polarized reduced meshes and replaced the
    weights with uniform ones.
  - `ElecData.wk_t` falls back to the per-state weights in the bandProjections header
    (`parse_kptsfrom_bandfile_header`) when kPts is missing. Legacy used uniform weights summing to nspin.
- Guard:
  - `utils/lattice.py::check_reduced_kmesh(edata, orbs, what, image)` is called from every
    `funcs/pcohp._get_*_tj` and from `funcs/dos.get_pdos`/`get_ipdos`.
  - On a reduced mesh (`ElecData.kmesh_is_reduced`, i.e. JDFTx `lti_allowed` False), single-image pair spectra
    raise, including the default `image="nearest"` and (0,0,0).
  - `image="all"` pair spectra, and pDOS of anything but every orbital, warn. The warning names incomplete
    orbital shells (`ElecData.orb_shell_u` from `io/data_parsing.get_orb_shell_u_from_bandfile_filepath`; one
    index per atom, l and shell n).
- Setting `legacy_reduced_kmesh` (default False, cache key `lgkm`) restores the old weights and turns the guard
  off. In fixed mode the generic "Internal inconsistency" warning from `_get_lti_allowed` is silenced, since the
  guard replaces it. `fixcompare.LEGACY_FLAGS["F5"]` was renamed from the proposed `legacy_kpoint_weights`.
- Why single images raise rather than warn: the reduced-mesh error for a single bond isn't small. A k-point
  stands in for its star, whose k-points belong to other, equivalent bonds, so a single bond's sum is biased.
  Only the average over the equivalent bonds is exact. For a single atom's l-complete pDOS the bias averages out
  over the atom's orbit (exact in graphene, where both C are equivalent and orbit-averaged), so pDOS only warns.
- Test data: `graphene_zspin_full` (`symmetries none`, 144 k per spin, 53 s) and `graphene_zspin_symm`
  (`symmetries automatic`, 19 k per spin with 5 distinct weights, 8 s). Both are ideal graphene, z-spin,
  12×12×1, 20 bands, GBRV v1.2 C, run locally; READMEs inside. A first attempt with 8 bands revealed F11.
- Results (IpCOHP in Ha; full / reduced-legacy / reduced-fixed):
  - C1–C2 all images: −1.50268 / −1.47588 / −1.50264
  - s–s all images: −0.13667 / −0.16354 / −0.13667
  - pz–pz all images: −0.16319 / −0.16941 / −0.16319
  - px–px all images: −0.14881 / −0.30163 / −0.31527 (warns)
  - one bond, R=0: −0.50127 / −1.12406 / raises
  - C1 occupied pDOS: 3.9669 / 3.9643 / 3.9668
  - On the full mesh the three bonds are identical (−0.50127 each) and sum with further shells to −1.5027.
- **Time-reversal unfolding (added 2026-10-27, on by default):**
  - `io/data_parsing.py::get_time_reversal_unfolding(ks_raw, w_raw, nspin, kfolding, k0)` rebuilds JDFTx's MP
    grid: k = (k0 mod 1 + i)/fold, i2 fastest, reduced to (−0.5, 0.5] (`ElecInfo.cpp`). It maps every grid point
    to a computed k (directly, or as its −k partner) and checks that the JDFTx weights equal multiplicity ×
    one-point weight in each spin channel. It returns the source state, conjugation flag, full-mesh k-points and
    weight factor, or None (already full; point-group reduced; explicit k-point list).
    `get_kpoint_offset_from_outfile_filepath` reads the single `kpoint` line of the last run's input echo.
  - `ElecData`: setting `unfold_time_reversal` (default True, cache key `utr`). It isn't applied for spinor runs
    or with `legacy_reduced_kmesh`.
    - `tr_unfolding` / `time_reversal_unfolded` report the map. `nstates_file` is the computed count, `nstates`
      the unfolded one.
    - `_unfold_states` re-indexes `e_tj` and the fillings-file occupations.
    - `_set_proj_tju` conjugates the partner projections after the F1 phase fix, using T(−k) = T(k)*. Under
      `legacy_orbital_phase` it uses T(−k) = (−1)^l T(k)*.
    - `wk_t` splits pair weights; `ks_t` and `kfolding` give the full mesh; `lti_allowed` is True.
    - `_reset_kpoint_arrays` clears every state-indexed array when the setting (or `legacy_reduced_kmesh`)
      changes.
  - Theory: without SOC each collinear spin channel has a real Hamiltonian, so ψ(−k) = ψ(k)* up to a band phase
    and ε(−k) = ε(k). JDFTx's Bloch sums satisfy φ(−k) = φ(k)* for real orbitals. The band phase cancels in every
    C*·M·C product.
  - Test data: `graphene_zspin_tr` (about 0.6 MB). It is the same input as `graphene_zspin_full`, but with
    `symmetries manual` and only the identity `symmetry-matrix`. JDFTx reduced it to 74 k per spin (4
    self-conjugate + 70 pairs at doubled weight) in 29 s.
  - Results vs. the full mesh: k-points identical, weights within 5e-10, eigenvalues within 3e-6 (separate SCF
    runs). Single bond R=0: −0.50126 vs −0.50127; px–px all images: −0.14880 vs −0.14881; px pDOS: 1.03889 vs
    1.03893; LTI allowed. With `unfold_time_reversal=False` the mesh is reduced and single images raise.
  - Full point-group unfolding stays future work. It needs `dump End Symmetries` (the `sym` file), rotation
    matrices for JDFTx's real Ylm (with the i^l convention), atom permutations and translation phases (tied to F3
    bookkeeping). JDFTx's `ColumnBundleTransform` is the reference for conventions. Logged as F12.
- Tests: `tests/core/test_reduced_kmesh.py` (15, including 6 for unfolding). Full suite: 159 passed.
- Docs: a README "JDFTx settings" table and reduced-mesh rules; CLAUDE.md settings list; the F5 section of
  `F4_F8_inputs_and_normalization.ipynb` (`figures/F5_reduced_mesh_weights.png`,
  `figures/F5_reduced_mesh_quantities.png`).
- `result_changes.py`: the F4 entry was added (`756f9e0`). **The F5 entry still has to be added after this work is
  committed** (stage "matrices": the weights feed every matrix and cached weight array).

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

**Implementation record for F6 (2026-10-27, branch `theory_check`, not yet committed):**
- Gaussian:
  - `core/operations/vector.py::get_gauss_smear_spectrum(..., normalize=False)` divides each spin channel's
    smeared spectrum by √(π·sig) when `normalize` is set. The kernel `scalar.gauss` (exp(−x²/sig)) is unchanged,
    and so is its unit test.
  - `funcs/general.py::get_generic_gsmear_spectrum` passes `normalize=not edata.legacy_normalization`.
  - Docstrings now say `sig` = 2·(standard deviation)²; they used to call it the standard deviation.
- LTI: `funcs/general.py::get_generic_lti_spectrum` multiplies the weights by N_k = prod(kfolding) (not legacy).
  libtetrabz `dos()` weights integrate to 1 per band over the zone (checked numerically in the audit), and
  crawfish weights carry w_t = spin weight / N_k on the full mesh LTI requires.
- Setting `legacy_normalization` (default False, cache key `lgnm`). Its setter only re-points the caches, because
  only the smeared spectra change. Integrated spectra (`get_i*`), `norm_max` and `norm_intg` are unaffected.
- Results:

  | system | Gaussian ∫DOS (legacy → fixed) | LTI ∫DOS (legacy → fixed) | expected |
  |---|---|---|---|
  | N2 | 0.168 → 30.000 | 1.1 → 29.5–30.2 (flat bands, rattled) | 30 |
  | graphene, full | 0.224 → 40.000 | 0.278 (= 40/144) → 40.005 | 40 |

  The unfolded time-reversal graphene also gives 40.000 / 39.9996; LTI works there because unfolding restores
  the full mesh.
- Tests: `tests/funcs/test_spectrum_normalization.py` (7):
  - DOS integrals for both methods, and for any `sig`;
  - Gaussian vs. LTI pDOS agreement;
  - the smeared pCOHP integrates to its weights;
  - the legacy scales √(π·sig) and 1/N_k;
  - `norm_max` is unchanged.

  `tests/funcs/test_general_funcs.py::test_get_generic_gsmear_spectrum` checks properties of the unnormalized
  kernel, so it is now pinned to `legacy_normalization=True`. Full suite: 166 passed.
- Docs: the README smearing paragraph (normalized kernel, `sig` meaning, legacy switch); CLAUDE.md; the F6
  section of `F4_F8_inputs_and_normalization.ipynb` (`figures/F6_dos_normalization.png`).
- `result_changes.py`: the F5 entry was added (`cc5f71e`). **The F6 entry still has to be added after this work is
  committed** (stage "spectra").

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

**Implementation record for F7 (2026-10-27, branch `theory_check`, not yet committed):**
- Already done with F2: the pCOOP warnings for the Löwdin basis and for orthonormalized-orbital projections, and
  the README notes on pCOOP and pCOMO. The README's false claim about pCOMO's sign was removed then.
- New:
  - `ElecData.orbital_completeness_u`: the k-weighted average of Σ_j |⟨φ_u|O|ψ_j⟩|² from the raw projections,
    so it doesn't depend on any setting. This is the first piece of the section 2.4 / F11 diagnostics.
  - `funcs/pcohp._warn_if_coop_is_artifact(edata, orbs)` also warns when a selected orbital's completeness is
    below `COOP_COMPLETENESS_WARN` = 0.85.
  - README: how to get meaningful pCOOP (`band-projection-params no no`, `los_orbs=False`, enough bands), plus
    a basis-dependence note.
- Test data: `N2_nospin_nonortho15` and `N2_nospin_nonortho40` (the `N2_nospin` input with
  `band-projection-params no no` and 15/40 bands; 38 s / 85 s locally; README inside).
- Results:
  - Completeness (s, py/pz, px): non-orthogonal 0.986, 0.976, 0.895 at 15 bands and 0.986, 0.982, 0.905 at 40;
    orthonormalized-orbital projections only 0.798, 0.974, 0.691.
  - N1–N2 overlaps (R=0, 40 bands): s–s +0.50, py–py +0.37, px–px −0.21. With orthonormalized-orbital data the
    band-space "overlap" py–py is 0.009, an artifact.
  - ICOOP (40 bands): all +0.507, px–px +0.203, py–py +0.271, s–s −0.208. The 15-band values are within 0.03.
  - **Basis dependence (input for Q1):** non-orthogonal ICOHP gives s–s +0.215 (net antibonding, the textbook
    N2 result with both 2σg and 2σu* filled) and all −0.305. The Löwdin basis on the same data gives s–s −0.160
    and all −0.896. pCOMO px–px is −0.95 while COOP/COHP say bonding.
- Tests: `tests/funcs/test_coop_basis.py` (6): no warning on the correct route, overlap signs, band convergence,
  s–s antibonding, pCOMO's orientation sign, all warnings (completeness via a raised threshold), and
  completeness independent of settings. Full suite: 172 passed.
- Notebook: `examples/fix_comparisons/F7_coop_basis.ipynb` (`figures/F7_completeness_overlap.png`,
  `figures/F7_coop_cohp_pairs.png`).

### F8: occupation helpers (P2)

- `elecdata.py::calculate_filling_nobroad`: `np.heaviside(eig, efermi)` gives 0 below 0 Ha and 1 above,
  ignoring mu. It should be `np.heaviside(efermi - eig, 0.5)`. Check 3 output: `[0,0,1,1]` for eig `[-.5,-.1,.1,.5]`.
- `calculate_filling_gauss/mp1/cold` use `from math import erf` on ndarrays and raise TypeError (Check 3). Use
  `scipy.special.erf`/`erfc`.
- These only run without a `fillings` file. For calculations without smearing, the outfile has no
  `FillingsUpdate: mu:` line, so the `mu` path is shaky too.

**Implementation record (2026-10-27, `theory_check`, uncommitted).**

- Two more defects turned up when the helpers were run on real data. JDFTx spells cold smearing `Cold`
  (`commands/elec_fillings.cpp`), but `occ_tj` compared against `"cold"` and raised "Unknown broadening type".
  Without `elec-smearing`, JDFTx never calls `ElecInfo::smearReport`, so it prints no mu. pymatgen's `outfile.mu`
  is then None, and `ElecData.mu` raised TypeError (`None / Ha_to_eV`) before the step function was ever reached.
- Without smearing, JDFTx doesn't use a step function at mu at all. It uses constant fillings (`FillingsConst`),
  and `ElecInfo.cpp::ElecInfo::setup` fills the lowest bands of every state with `nsElectrons[s] * wInv` electrons:
  N/2 for no-spin, and (N ± M)/2 for z-spin, where `qNet[s] = (Qinitial ± Minitial)/nSpins` and M is
  `elec-initial-magnetization`. Electrons aren't moved between spin channels or k-points, so a constrained
  magnetic run can have a spin-up HOMO above the spin-down LUMO. A step at any single mu (aufbau) would be wrong
  there.
- `elecdata.py`:
  - `calculate_filling_gauss/mp1/cold` use `scipy.special.erfc`, the same function `ElecInfo::smear` uses, and take
    scalars or arrays. `calculate_filling_nobroad` is `np.heaviside(efermi - eig, 0.5)`.
  - New: `SMEARING_FUNCS` (lower-case type → helper) and `calculate_fillings_constant(e_tj, nelec_per_state_s)`.
  - `occ_tj` calls the new `ElecData._calculate_occ_tj`, which matches the smearing type case-insensitively. For
    JDFTx data without smearing it uses `calculate_fillings_constant` with counts from
    `_constant_fillings_per_spin` (`outfile.total_electrons`, and `elec-initial-magnetization` from the echoed
    infile). It warns if the run read `elec-initial-fillings`, since those can't be reproduced. For non-JDFTx
    data without smearing, the step at the user's mu is used. The result goes through `_check_electron_count`.
  - The `broadening_type` setter accepts the JDFTx spellings in any case, plus `none`/None.
  - `ElecData.mu` raises a clear ValueError, suggesting `edata.mu` from the eigStats dump, when the outfile has no
    mu. The pymatgen `_mu_backup` unit issue (F10) is untouched.
- Why there's no legacy switch or `result_changes` entry: every old path either raised or (non-JDFTx data without
  smearing, which has no on-disk cache) gave occupations unrelated to mu. The Fermi path is numerically unchanged,
  so no cached result can change (section 4.0).
- Fixtures (each dumps `fillings` for comparison; README in `Al_fcc_Fermi/`): `Al_fcc_{Fermi,Gauss,MP1,Cold}` (fcc
  Al, 8x8x8 mesh reduced to 29 states, smearing 0.01 Ha) and `Si_nosmear_{nospin,zspin_M2}` (diamond Si, 4x4x4, no
  smearing, the second with z-spin and `elec-initial-magnetization 2 yes`). pymatgen can't parse
  `lattice face-centered Cubic`, so the inputs give the vectors explicitly.
- Results: occupations computed without the file match JDFTx's dump to 2.2e-5 (Fermi), 9e-6 (Gauss), 6e-6 (MP1)
  and 3e-6 (Cold), and exactly for both constant-filling runs. Every case gives N_e equal to the outfile's count.
  The remaining smeared difference comes from mu being printed to 1e-9 Ha and from float32 eigenvalues.
- Origin: these helpers were copied from the pymatgen JDFTx module (`jdftxoutfileslice.py::_calculate_filling`,
  used for `homo_filling`/`lumo_filling`/`is_metal`). That copy is scalar-only, spells `Cold` correctly and has no
  step function, so F8's bugs don't carry over. It has a different one: eV energies divided by a Hartree
  broadening, which makes the smearing 27× too narrow (fcc Al reads as `is_metal == False`). Logged in the
  pymatgen-core agent notes (`~/vs/claude-notes/pymatgen-jdftx.md`, "Open bugs", out-file parsing) and not fixed
  here, since crawfish doesn't use those attributes.
- Tests: `tests/core/test_occupation_helpers.py` (10). Full suite: 182 passed.
- Notebook: the F8 section of `F4_F8_inputs_and_normalization.ipynb` (`figures/F8_filling_helpers.png`,
  `figures/F8_fillings_vs_jdftx.png`).

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

- Q1. (2026-10-27 update: on N2 the non-orthogonal basis reproduces the textbook net-antibonding s–s interaction, while
  the Löwdin basis doesn't, and total ICOHP differs about 3×; see the F7 record.) COHP basis choice: should the default be the Löwdin band-projected basis (energy-zero invariant, matches
  current `los_orbs=True`) or the non-orthogonal basis (closer to LOBSTER)? On N2 they agree in sign and roughly
  in size (A–B total −0.867 vs −0.833 Ha). This is the user's call.
- Q2. **Answered (2026-10-27):** yes. `JDFTXOutfile.structure` (pymatgen fork) never calls `to_unit_cell`, and
  pymatgen `Structure` keeps unwrapped fractional coordinates. For N2 the Cartesian coordinates equal the outfile's
  final `ion` lines (13.895656 bohr = 7.3533 Å). F3 uses `edata.structure.frac_coords` as τ.
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

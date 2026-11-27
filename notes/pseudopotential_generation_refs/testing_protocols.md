# How a new norm-conserving pseudopotential is tested: sourced protocols

> **What this file is.** A research report written by a Claude subagent on 2026-11-25 from the sources tagged below.
> It read sources and ran no DFT. The instructions that use it are in `../pseudopotential_generation.md`; where the
> two disagree, that file is newer.
>
> **Settled after it was written:** item 11 of "Could not verify" (upstream SG15 1.2 files hold no orbitals:
> `number_of_wfc="0"` in all 219 files of the 2020-02-06 tarball, as in all 86 of the 2015-10-07 one; see
> `~/vs/research/crawfish/anl/pseudo_type_comparison/sg15_distribution/sg15_distribution.md`).

No project files were written or changed. Everything below was read from the primary text (arXiv HTML, PDFs, official repos) unless marked otherwise. Three caveats apply throughout:

- **arXiv versions.** Papers were read in their arXiv versions, which can differ in detail from the journal versions. The ACWF thresholds were also confirmed in the published SI.
- **"(page summary)".** The page was read through a summarising fetch, not verbatim. Treat those lines as lower confidence.
- **UNVERIFIED.** I could not confirm it. These are collected at the end.

## Sources (tags used after each claim)

**Delta test**
- [D1] Lejaeghere 2014, accepted manuscript: https://arxiv.org/abs/1204.2733
- [D2] DeltaCodesDFT package (README.md, calcDelta.py v3.0, eosfit.py, WIEN2k.txt, history.tar.gz, CIFs/, primCIFs/): https://github.com/molmod/DeltaCodesDFT
- [D3] Lejaeghere 2016 Science full text (NIST copy): https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=919278 (DOI https://doi.org/10.1126/science.aad3000)
- [D4] Old project page, now a "moved" notice: https://molmod.ugent.be/deltacodesdft
- [D5] Jollet, Torrent, Holzwarth (JTH): https://arxiv.org/abs/1309.7274

**ACWF**
- [A1] Bosoni 2024 main text: https://arxiv.org/abs/2305.17274 (DOI https://doi.org/10.1038/s42254-023-00655-3)
- [A2] Published SI: https://static-content.springer.com/esm/art%3A10.1038%2Fs42254-023-00655-3/MediaObjects/42254_2023_655_MOESM1_ESM.pdf
- [A3] Scripts and data: https://github.com/aiidateam/acwf-verification-scripts
- [A4] Data record: https://doi.org/10.24435/materialscloud:s4-3h ; viewer: https://acwf-verification.materialscloud.org

**GBRV, PseudoDojo, SSSP**
- [G1] GBRV paper: https://arxiv.org/abs/1305.5973 ; [G2] GBRV site: https://www.physics.rutgers.edu/gbrv/
- [P1] PseudoDojo paper: https://arxiv.org/abs/1710.10138
- [P2] https://github.com/abinit/pseudo_dojo/blob/master/pseudo_dojo/dojo/works.py
- [P3] https://github.com/abinit/pseudo_dojo/blob/master/pseudo_dojo/core/dojoreport.py
- [S1] SSSP paper: https://arxiv.org/abs/1806.05609
- [S2] https://github.com/aiidateam/aiida-sssp-workflow (src/aiida_sssp_workflow/protocol/*.yml, docs/spec.md)
- [S3] SSSP archive records: https://archive.materialscloud.org/doi/10.24435/materialscloud:rz-77 and https://archive.materialscloud.org/records/69mvv-cv520/latest
- [S4] https://aiida-sssp-workflow.readthedocs.io/en/latest/

**Atom-level tests and ONCV**
- [F1] Fuchs and Scheffler, fhi98PP: https://arxiv.org/abs/cond-mat/9807418
- [H1] Hamann 2013: https://doi.org/10.1103/PhysRevB.88.085117 (PDF shipped in https://github.com/oncvpsp/oncvpsp/tree/master/doc)
- [O1] ghosts.md, [O2] users_guide.md, [O3] program_overview.md, [O4] core_correction.md, [O5] release_notes.md, all under https://github.com/oncvpsp/oncvpsp/tree/master/doc
- [O6] https://github.com/oncvpsp/oncvpsp/blob/master/src/upfout.f90
- [C1] Abstracts, retrieved through https://api.openalex.org/works/doi:<DOI> : HSC https://doi.org/10.1103/PhysRevLett.43.1494 ; Teter https://doi.org/10.1103/PhysRevB.48.5031 ; Gonze 1990 https://doi.org/10.1103/PhysRevB.41.12264 ; Gonze 1991 https://doi.org/10.1103/PhysRevB.44.8503 ; Louie-Froyen-Cohen https://doi.org/10.1103/PhysRevB.26.1738
- [J1] Junquera ATOM tutorial: https://personales.unican.es/junqueraj/JavierJunquera_files/Metodos/Pseudos/Pseudo-test/Test-pseudo.pdf

**Other tests and functional consistency**
- [X1] Borlido 2020: https://arxiv.org/abs/2003.10177
- [X2] Yang 2018: https://doi.org/10.1103/PhysRevB.97.085130
- [X3] Range-separated hybrid pseudopotentials: https://arxiv.org/abs/2302.11676
- [X4] Yao and Kanai 2017: https://doi.org/10.1063/1.4984939
- [X5] Fuchs 1998: https://doi.org/10.1103/PhysRevB.57.2134
- [X6] Willand 2013: https://doi.org/10.1063/1.4793260
- [X7] SG15 paper: https://arxiv.org/abs/1502.00995 ; [X8] SG15 site: http://www.quantum-simulation.org/potentials/sg15_oncv/index.htm

**Pseudo-atomic orbitals**
- [Q1] https://www.quantum-espresso.org/Doc/INPUT_PW.html (QE 7.5) ; [Q2] https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html
- [Q3] QE source, develop branch: https://github.com/QEF/q-e/blob/develop/ + PP/src/projwfc.f90, PW/src/offset_atom_wfc.f90, upflib/atwfc_mod.f90, Modules/read_pseudo.f90, upflib/write_upf_new.f90
- [Q4] UPF format page: https://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format
- [L1] Sánchez-Portal, Artacho, Soler 1996: https://arxiv.org/abs/cond-mat/9509053 (the 1995 paper is https://doi.org/10.1016/0038-1098(95)00341-X)
- [L2] LOBSTER 2016: https://pmc.ncbi.nlm.nih.gov/articles/PMC5067632/ ; [L3] https://arxiv.org/abs/2304.02726 ; [L4] LOBSTER 2020: https://doi.org/10.1002/jcc.26353
- [U1] HP code paper: https://arxiv.org/abs/2203.15684 ; [U2] Timrov 2020: https://arxiv.org/abs/2010.13485
- [M1] QE list threads: https://www.mail-archive.com/users@lists.quantum-espresso.org/msg29931.html, msg29933.html, msg29934.html, msg40120.html, msg40121.html, msg40122.html
- [JX] https://github.com/shankar1729/jdftx/blob/master/jdftx/electronic/SpeciesInfo_readUPF.cpp ; https://jdftx.org/Pseudopotentials.html

---

## 1. Delta test (Lejaeghere et al. 2014 and 2016)

**Test set**
- 71 elemental crystals in their ground-state structure, at the PBE equilibrium geometry, without spin-orbit. Mn is replaced by antiferromagnetic fcc (space group 225) and S by the β-Po phase (space group 166). [D1][D2]
- `calcDelta.py` covers H to Rn, excluding La to Yb and At. [D2]
- CIFs are in `CIFs/` (71 files). `primCIFs/` holds primitive cells, which are "not the ones reported in the original article ... so k-points need to be modified". [D2]
- The old project page now says its content moved to https://molmodstatic.ugent.be/deltacodesdft/. [D4] That static copy returned HTTP 403 to me.

**Procedure**
- Compute the energy at 94, 96, 98, 100, 102, 104 and 106 % of the CIF volume, seven points per crystal. [D2]
- Geometry is frozen: no cell-shape or internal relaxation ("Force optimizations should not be employed"). [D1][D2]
- Spin polarisation: antiferromagnetic for O, Cr, Mn; ferromagnetic for Fe, Co, Ni. For oxygen "The antiferromagnetism ... should take place between the O2 molecules". [D2]
- If the fitted V0 is below 96 % or above 104 % of the WIEN2k V0, add volumes so there are again seven points roughly symmetric about V0. [D2]
- Valence and semicore electrons are scalar-relativistic (Koelling-Harmon in the 2014 work). [D1][D3]
- Fit a third-order Birch-Murnaghan curve. `eosfit.py` does this with `np.polyfit(V**(-2/3), E, 3)` and derives V0, B0, B1 from the polynomial. Input is Å³/atom and eV/atom. [D2]
- Birch-Murnaghan form: E(V) = E0 + (9 V0 B0/16) { [(V0/V)^(2/3) − 1]³ B1 + [(V0/V)^(2/3) − 1]² [6 − 4 (V0/V)^(2/3)] }. [A1]

**Formulas**
- 2014 definition: Δ = ⟨ sqrt( ∫ ΔE²(V) dV / ΔV ) ⟩, integrated over V0,WIEN2k ± 6 % (ΔV = 0.12 V0,WIEN2k), with all EOS zeroed at their minimum and averaged over crystals. [D1]
- 2016 definition: Δi(a,b) = sqrt( ∫ from 0.94 V0,i to 1.06 V0,i of [E_b,i(V) − E_a,i(V)]² dV / (0.12 V0,i) ), in "an interval that is symmetrical around the average equilibrium volume". [D3]
- `calcDelta.py` v3.0 implements the symmetric form: Vi = 0.94 (V0w + V0f)/2 and Vf = 1.06 (V0w + V0f)/2. The old reference-based form is `--asymm`. [D2]
- Analytic integral: F(V) = Σ over n = −2..4 of x_n V^(−(2n+1)/3), with x_n = −3/(2n+1) Σ over i+j = n+2 of (a_i^c − a_i^w)(a_j^c − a_j^w). The coefficients are a3 = 9 V0³ B0 (B1 − 4)/16, a2 = 9 V0^(7/3) B0 (14 − 3 B1)/16, a1 = 9 V0^(5/3) B0 (3 B1 − 16)/16, a0 = 9 V0 B0 (6 − B1)/16. [D1]
- Unit conversion in the script: B0[GPa] × 1e9 / 1.602176565e-19 / 1e30 gives eV/Å³. [D2]
- Δ1 (JTH): Δ1 = (V_ref B_ref)/(V_AE B_AE) × Δ. The JTH paper states "V_ref = 30 Bohrs³ and B_ref = 100 GPa". [D5]
- Δ1 in the package (symmetric): Δ × 4 × vref × bref / ((V0w + V0f)(B0w + B0f)), with `vref = 30.`, `bref = 100 GPa`, and volumes in Å³/atom. [D2]
- PseudoDojo codes Δ1 as `dfact * (30 * 100) / (v0 * b0_GPa)`. [P3] So the implementations use 30 in Å³, while the JTH text says Bohr³.
- Δrel in the package: 100 × sqrt( ∫ (E_f − E_w)² dV / ∫ ((E_f + E_w)/2)² dV ), in percent. [D2]

**Numerical settings**
- The Δ protocol itself does not prescribe k-points or smearing. "Appropriate numerical settings were determined separately for each method, ensuring converged results"; details are in SI tables S3 to S42. [D3]
- Science warns that "the use of sometimes very small electronic smearing widths requires much higher numbers of k-points than routine DFT calculations warrant". [D3]
- SSSP (QE): 200 Ry wavefunction cutoff, 20×20×20 Monkhorst-Pack, Marzari-Vanderbilt smearing 2 mRy. [S1]
- PseudoDojo (ABINIT) code defaults: `kppa=6750` k-points per reciprocal atom, `smearing="fermi_dirac:0.1 eV"`, `toldfe=1.e-9`, volumes `v0 * np.arange(94, 108, 2) / 100`. [P2]
- The 2014 structure relaxations (VASP 5.2.2) used the tetrahedron method with Blöchl corrections and a 0.01 eV/Å force criterion. [D1]

**All-electron reference**
- The 2014 paper used WIEN2k 11.1 (APW+lo). [D1]
- The package's reference file `WIEN2k.txt` is headed "WIEN2k version 13.1, calcDelta package version 3.0". Columns are element, V0 [Å³/atom], B0 [GPa], B1. [D2]
- Science Table 2 gives ⟨Δ⟩ "as an average over the all-electron methods". [D3]

**What counts as good**
- "a mutual Δ of 1 or even 2 meV per atom can be deemed to yield indistinguishable EOS for all practical purposes". [D3]
- The maximum Δ among all-electron codes, 1.0 meV/atom, corresponds to an average volume deviation of 0.14 Å³/atom (0.38 %) and 1.6 GPa (4.0 %) in B0. [D3]
- Experiment against experiment for Cu, Ag, Au gives Δexp = 1.0 meV/atom. [D3]
- Current potentials reach "Δ values of about 1 meV per atom, with a low approaching 0.3 meV per atom". [D3]
- SSSP: two EOS are "undistinguishable if the Δ-factor is smaller than 1 meV/atom ... or if the Δ′-factor is smaller than 3 meV/atom", the latter meaning under 0.5 % in V0. [S1]
- PseudoDojo: for Δ′ "a value less than 2 in general indicates an accurate potential". [P1]

**Reported values**
- Science Table 2, ⟨Δ⟩ against all-electron, meV/atom: JTH01 1.1 and JTH02 0.6; Vdb/CASTEP 6.5, OTFG7 2.6, OTFG9 0.7; GPAW06 3.6, GPAW09 1.6; PSlib031 1.7, PSlib100 1.0; VASP2007 2.0, VASP2012 0.8, VASPGW2015 0.6. [D3]
- Same recipe across codes: Schlipf-Gygi 2015-01-24 ONCVPSP agrees to 0.3 meV/atom between QE and CASTEP. GBRV 1.4 agrees to 0.3 between QE and CASTEP. GBRV 1.2 agrees to 0.7 between ABINIT PAW and QE ultrasoft. [D3]
- SSSP 1.1: efficiency 0.44 and precision 0.33 meV, against WIEN2k, excluding the rare-earth nitrides. [S1]
- PseudoDojo: norm-conserving tables average 1.4 against WIEN2k, excluding the old FHI table at 13.4. PD-PBE means are 0.70 (standard) and 0.64 (stringent) with Cr, Mn, Fe excluded. Standard-table outliers are Cr, Mn, Fe at about 10.4, 7.7, 5.9. [P1]
- The per-method Fig. 4 numbers in Science for SG15, PD0.1, GBRV and SSSP are in a figure whose text I could not extract. **UNVERIFIED as quoted from the paper.**

In their place, I recomputed Δ against `WIEN2k.txt` (13.1) from the package's own `history.tar.gz`, using the v3.0 symmetric formulas. This is against one all-electron code, not the Science all-electron average. The source is [D2]; the script is listed at the end.

| Dataset in history.tar.gz | Δ (meV/atom) | Δ1 | Elements | Worst element |
|---|---|---|---|---|
| QE, SSSP Accuracy | 0.31 | 0.87 | 70 | Cr 2.5 |
| QE, SSSP Efficiency | 0.46 | 1.19 | 70 | Cr 3.6 |
| QE, GBRV 1.4 | 0.99 | 2.45 | 62 | Mn 6.1 |
| QE, GBRV 1.2 | 0.99 | 2.43 | 62 | Mn 6.1 |
| CASTEP, GBRV 1.4 | 1.06 | 2.76 | 63 | Mn 5.5 |
| ABINIT PAW, GBRV 1.2 | 0.83 | 2.05 | 63 | P 3.8 |
| QE, SG15_ONCV | 1.35 | 2.69 | 67 | Cr 21.1 |
| CASTEP, Schlipf-Gygi 2015-01-24 | 1.46 | 2.98 | 68 | Cr 20.8 |
| CASTEP, Schlipf-Gygi 2015-05-20 | 1.44 | 2.96 | 68 | Cr 20.8 |
| ABINIT, PseudoDojo ONCVPSP 0.1 | 0.66 | 1.63 | 70 | Rh 2.0 |
| QE, pslibrary 1.0.0 PAW | 0.83 | 2.06 | 69 | O 11.0 |
| ABINIT, JTH 0.2 | 0.58 | 1.31 | 71 | Lu 4.9 |
| ABINIT, FHI (7.6.4) | 13.46 | 20.26 | 69 | W 310.9 |
| Elk 3.1.5 | 0.26 | 0.99 | 71 | O 1.1 |
| FLEUR 0.26 | 0.49 | 1.59 | 71 | Cd 2.6 |

**Known weaknesses**
- Δ scales with stiffness. The JTH example: a 0.76 % V0 deviation gives Δ = 0.39 meV for Cs and 9.14 meV for Os. [D5]
- "very soft materials, as for example the noble-gas solids, are generally associated with small values of the Δ-factor, even for significant volume differences". [S1]
- "The noble gas solids for instance always have a low Δ-Gauge." [P1]
- Δ "has the shortcoming of being too sensitive to the value of the bulk modulus". [A1]
- Only single-element crystals and ground-state properties are tested, and the "prescribed computational settings ... are rather stringent making it unsuitable for fast pre-testing". [P1]
- Good Δ does not imply transferability. 5d-only transition-metal potentials had "good results obtained for the Δ-Gauge" yet "are not transferable" and had ghost states just above the Fermi level. [P1]
- The 960-crystal set exposes failures the 71-crystal set misses, for example lanthanides and oxides. [A2 S15]

## 2. ACWF verification (Bosoni et al. 2024)

**Sets**
- Z = 1 to 96, ten cubic prototypes per element, 960 equations of state. [A1]
- All structures are cubic with forces zero by symmetry, so only total energies are needed. [A1]
- Unaries (384 systems) [A1][A2 Table S1.1]:
  - FCC: Fm-3m, site 4a, coordination 12.
  - BCC: Im-3m, 2a, coordination 8.
  - SC: Pm-3m, 1a, coordination 6.
  - Diamond: Fd-3m, 8a, coordination 4.
- Oxides (576 systems), formal oxidation states +1 to +6 [A1][A2 Table S1.2]:
  - X2O: Na2O type, Fm-3m, X on 8c, O on 4a.
  - XO: NaCl type, Fm-3m, 4a and 4b.
  - X2O3: Ag2O3 type, Pn-3m, X on 4b, O on 6d.
  - XO2: ZrO2 fluorite type, Fm-3m, 4a and 8c.
  - X2O5: no prototype listed, Pn-3m, X on 4b, O on 4c + 6d.
  - XO3: ReO3 type, Pm-3m, 1a and 3d.
- Atoms per formula unit: FCC 1, BCC 1, SC 1, Diamond 2, X2O 3, XO 2, X2O3 5, XO2 3, X2O5 7, XO3 4. The primitive cells of X2O3 and X2O5 hold 10 and 14 atoms. [A2 S1]
- "XO3 could be a proxy for systems with very short bond lengths, such as in high-pressure studies". [A1]

**EOS procedure**
- Seven equidistant volumes between 94 % and 106 % of a tabulated central volume Ṽ0, with a Birch-Murnaghan fit. [A1] The central volumes are in SI Table S1.3. [A2]
- The fitted quantity is the free energy E − TS, not the internal energy and not E − TS/2. [A1 Box 3]
- The fit is the non-iterative polynomial fit `np.polyfit(V**(-2/3), E, 3)` in `eosfit_31_adapted.BM`. [A3 3-analyze/eos_utils/eosfit_31_adapted.py]
- The SI warns that `scipy.optimize.curve_fit` is less stable for B1. [A2 S3]

**Prescribed settings (Box 3)**
- PBE, non-spin-polarised, scalar-relativistic. [A1]
- Fermi-Dirac smearing, 0.0045 Ry (about 61.2 meV). [A1]
- Γ-including Monkhorst-Pack grid: "the smallest integration mesh guaranteeing a linear spacing of at most 0.06 Å⁻¹ in each of the three reciprocal-space directions for the smallest volume, and the same set of k-points (in scaled units) for all other volumes". [A1]
- The SI gives the spacing as "0.06 Å⁻¹ (i.e., 0.00954929 2πÅ⁻¹)", so the reciprocal vectors include the 2π. [A2 S13]
- Cutoffs are code-specific:
  - QE 7.0 with SSSP-prec 1.3 used "plane-wave cutoffs corresponding to the largest recommended value from the elements in each structure" and an energy threshold of "0.1·10⁻⁹ Ry per atom". [A2 S13.6]
  - ABINIT 9.6.2 with PseudoDojo 0.5 used `tolvrs = 1e-10`, `occopt = 3`, `tsmear = 0.00225`, `fband = 2`, and the "high" stringency cutoffs. [A2 S13.1]
- The aiida-sssp-workflow `eos.yml` `standard` ("acwf") protocol encodes the same: `degauss: 0.0045`, `smearing: fd`, `kpoints_distance: 0.06`, `conv_thr_per_atom: 1.0e-9`, `scale_count: 7`, `scale_increment: 0.02`. [S2]

**Metrics**
- Notation: ⟨f⟩ = (1/(V_M − V_m)) ∫ f dV, with V_m = 0.94 Ṽ0 and V_M = 1.06 Ṽ0. [A1]
- ε(a,b) = sqrt( ⟨[E_a − E_b]²⟩ / sqrt( ⟨[E_a − ⟨E_a⟩]²⟩ ⟨[E_b − ⟨E_b⟩]²⟩ ) ). 1 − ε² approximates R². It is independent of per-atom or per-formula-unit normalisation and insensitive to B0. [A1]
- ν(a,b) = 100 sqrt( Σ over Y = V0, B0, B1 of [ w_Y (Y_a − Y_b) / ((Y_a + Y_b)/2) ]² ), with w_V0 = 1, w_B0 = 1/20, w_B1 = 1/400. [A1]
- The weights come from error propagation of random energy noise: errors on B0 and B1 are about 20 and 400 times those on V0 for the 94 to 106 % range. A 90 to 110 % range would give about 15 and 200. [A2 S3]
- ν ≈ 1.65 ε for ν ≲ 1. [A1]

**Thresholds**
- Excellent: ε ≤ 0.06 or ν ≤ 0.10. Good: 0.06 < ε ≤ 0.20 or 0.10 < ν ≤ 0.33. Noticeably different: up to ε = 1.0 or ν = 1.65. Clearly different: above that. [A1 Box 3][A2 S7, S9]
- ε = 0.06 corresponds to R² ≈ 0.9964 and ε = 0.20 to R² ≈ 0.96. [A1]
- The "excellent" level is set by twice the standard deviation of the FLEUR and WIEN2k discrepancies (0.12 % V0, 0.7 % B0, 4 % B1). [A2 S7]
- The thresholds hold only for this volume range. [A1]

**Reference data**
- The "reference average all-electron dataset" is the average of V0, B0, B1 from FLEUR (LAPW+LO) and WIEN2k ((L)APW+lo+LO). [A1]
- 883 of 960 systems are excellent in both metrics between the two codes. Four exceed "good": Cs2O5, Fr2O5, Ra2O5, RbO3. [A1]
- Files: `acwf_paper_plots/code-data/results-{unaries,oxides}-verification-PBE-v1-AE-average.json`. [A3]
- The full archive, including AiiDA databases, is `ACWF-verification-data-and-scripts.zip` in the Materials Cloud record (page summary). [A4]
- JSON format, as inspected: top-level keys are `BM_fit_data`, `num_atoms_in_sim_cell`, `script_version` ("0.0.3"). [A3]
  - Keys look like `"Si-X/Diamond"` and `"Ac-X2O3"`.
  - Each entry is `{"E0": 0, "min_volume": <Å³ per simulation cell>, "bulk_modulus_ev_ang3": <eV/Å³>, "bulk_deriv": <B1>, "residuals": 0}`.
  - Example: Si diamond has min_volume 40.9149, B0 0.55244 eV/Å³, B1 4.3118, 2 atoms.
  - Per-code result files add `eos_data` (lists of [V, E]), `stress_data`, `uuid_mapping`, `failed_wfs`, `completely_off`, `missing_outputs`.

**Libraries assessed** (counts are excellent / good / noticeably different / clearly different on ε, against the all-electron average) [A2 Figs. S9.1 to S9.11]:

| Approach | Computed | Counts |
|---|---|---|
| ABINIT, PseudoDojo 0.5 | 720 | 232 / 377 / 111 / 0 |
| QE, SSSP-prec 1.3 | 960 | 388 / 300 / 199 / 73 |
| SIRIUS/CP2K, SSSP-prec 1.2 | 700 | 363 / 251 / 81 / 5 |
| VASP, GW-PAW54* | 960 | 403 / 348 / 200 / 9 |
| CASTEP, C19MK2 | 960 | 197 / 410 / 277 / 76 |
| GPAW, PAW 0.9.20000 | 670 | 130 / 156 / 350 / 34 |
| BigDFT, HGH-K | 402 | 45 / 97 / 173 / 87 |
| CP2K, TZV2P GTH | 709 | 57 / 171 / 317 / 164 |
| SIESTA, PseudoDojo 0.4 | 698 | 30 / 117 / 444 / 107 |
| FLEUR or WIEN2k against the average | 960 | 936 / 23 / 1 / 0 |

- PseudoDojo 0.4 to 0.5: for Te to Ba and Tl to Rn, accuracy "is significantly improved by including a projector for the unbound f state". [A1]
- Same library across codes (ABINIT, QE, CASTEP with PseudoDojo 0.4): initial discrepancies for Cu, Zn, Ne were traced to the treatment of the long-range radial mesh. The UPF mesh was truncated from 10 to 6 Bohr, "as it is done in the psp8 format", after which agreement was better than between the two all-electron codes. [A2 S14]

**Running one element without AiiDA** (assembled from the above)
1. Take the structure from `0-preliminary-do-not-run/unaries/xsfs-unaries-verification-PBE-v1/` or `.../oxides/xsfs-oxides-verification-PBE-v1/`. Lattice parameters are in `lattice_parameters_unaries-verification-PBE-v1.json`; central volumes in the oxides JSON and in SI Table S1.3. [A3][A2]
2. Scale to seven volumes, 0.94 to 1.06 in steps of 0.02. Fix the k-mesh from the 0.06 Å⁻¹ rule at the smallest volume. Use Fermi-Dirac 0.0045 Ry, PBE, no spin. [A1]
3. Record E − TS per simulation cell. Fit with the `BM` function. [A3]
4. Divide V0 by (atoms in cell / atoms per formula unit). Call `nu(V0, B0, B1, V0c, B0c, B1c, 100, 1/20, 1/400)` and `epsilon(..., prefact=1, ...)` from `3-analyze/outputs/quantities_for_comparison.py`, as `generate_periodic_tables.py` does. [A3]

Two things I noticed reading that file; neither is a documented issue. First, `epsilon()` and `delta()` integrate over 0.94 to 1.06 times the mean of the two fitted V0, not over the tabulated Ṽ0 the paper describes. Second, `epsilon()` computes `Eavg2` from the first curve's parameters (`intEdV(v0w,b0w,b1w,...)`), the same as `Eavg1`. This looks like a typo and has negligible effect for similar curves.

## 3. GBRV tests (Garrity et al. 2014)

**Principle**
- Compare with all-electron WIEN2k under "exactly the same choice of DFT exchange-correlation functional, the same k-point sets, and the same smearing temperature", not with experiment. [G1]

**Settings**
- PBE, scalar-relativistic. WIEN2k treats the core fully relativistically. [G1]
- 8×8×8 k-points and 0.002 Ry Fermi-Dirac smearing, explicitly not converged but identical for both methods. [G1]
- GBRV run at 40 Ry / 200 Ry. WIEN2k at R_MT·K_MAX of 8 to 10. [G1]
- Non-spin-polarised, except the magnetic moments. [G1]

**Procedure**
- Variable-cell relaxation for an initial guess, then "nine points from −1% to 1% of the initial guess and fitting the results to a parabola". [G1]
- PseudoDojo's reimplementation uses `ngkpt=(8,8,8)`, `smearing="fermi_dirac:0.001 Ha"`, `ecutsm=0.05`, and volumes `V * np.arange(99, 101.25, 0.25)/100`. [P2]

**Four sets** (all cubic, no internal parameters) [G1]
1. Each element in fcc and bcc: metallic bonding and per-element diagnosis. WIEN2k could not be converged for N, P, Hg.
2. Rocksalt: ionic bonding. Main-group elements are paired in preferred oxidation states; all transition metals are paired with O.
3. Perovskites: higher oxidation states.
4. Half-Heuslers (MgAgAs type, F-43m): 138 compounds, covering all elements except H and the halogens.

**Properties and statistics**
- Lattice constant, bulk modulus, fcc-bcc energy difference. [G1]
- Magnetic moments of the rocksalt transition-metal oxides, computed at the all-electron non-spin-polarised lattice constants. [G1]
- Statistics: RMS error against all-electron, and the percentage of lattice constants outside ±0.2 %. The stated reason is that "reliable results require every calculation to be accurate". [G1]
- GBRV column of Table 1 [G1]:
  - RMS lattice constant: fcc 0.14 %, bcc 0.15 %, rocksalt 0.13 %, perovskite 0.08 %, half-Heusler 0.11 %.
  - RMS bulk modulus: 3.6, 5.3, 5.0, 5.5, 5.4 %.
  - fcc-bcc energy 3.7 meV. Rocksalt magnetic moment 0.08 μB.
  - Outside ±0.2 %: 9.8, 9.8, 7.8, 0.0, 3.6 %.
- General level: "most lattice constants within 0.2% ... and most bulk moduli within 5%". [G1]
- fcc and bcc errors correlate at 0.95. [G1]
- Single-projector norm-conserving libraries gave 2.3 % (TM) and 2.8 % (HGH) RMS on half-Heusler lattice constants, and 0.76 % for Bennett-Rappe. [G1]
- SG15 adopted the same target, δ0 = 0.2 % on the lattice constant against FLEUR, with 8×8×8 k-points, Fermi smearing 0.001 Ha, and an 11-point Murnaghan fit. Its training set is bcc, fcc and rocksalt; its test set is sc, diamond and zincblende; 602 materials in all. [X7]
- The GBRV site offers the full testing data (`psp_supp.pdf` and a spreadsheet) and a zincblende WIEN2k file (page summary). [G2]

## 4. PseudoDojo (van Setten et al. 2018)

**Tests in a DojoReport**
- "a series of 7 tests in crystalline environments: Δ-Gauge, Δ′-Gauge, GBRV-FCC, GBRV-BCC, GBRV-compound, ghost state detection, and phonons at Γ, all executed using ABINIT". Convergence studies and cutoff hints are added. The report is JSON and stores results as a function of cutoff. [P1]
- Trial keys in code: `deltafactor`, `gbrv_bcc`, `gbrv_fcc`, `phgamma`, `ghosts`, each with a `_soc` variant, plus `hints` and `ppgen_hints`. [P3]

**Code defaults for each test** [P2]
- Δ: `kppa=6750`, Fermi-Dirac 0.1 eV.
- Ghosts: band structure of the Δ-test crystal with `kppa=3000`, `maxene=250` eV, Fermi-Dirac 0.1 eV.
- Γ phonons: `kppa=1000`. Frequencies are extracted with `asr=2` and with `asr=0` (both `chneut=1, dipdip=1`) and both are stored.

**Ghost test**
- "scanning the band structure and the density of states for dispersionless states, up to energies around 200 eV above the Fermi level". [P1]
- Ghosts listed in the paper's Table 8 are high enough that "for ground state calculations they do not cause any problem"; they matter for GW and optics. [P1]

**Acoustic-sum-rule test**
- Model core charges with strong oscillations in high-order derivatives "have disastrous effects for density functional perturbation theory calculations, in particular for the fulfillment of the acoustic sum rule". [P1]
- "Large deviations from zero (when the ASR is not enforced by the code) usually indicate that the model core charge and its derivatives cannot be correctly described with a sufficiently small cutoff energy". [P1]
- Observed: within 2 cm⁻¹ except for outliers Ne (about 14), Mg (about 11), Na (about 6), Ge (about 4), Cu (about 4). [P1]
- Optical modes "are by far not converged at the low Ec hint". [P1]

**Hints** (paper Table 3; low / normal / high) [P1]
- ε − ε_AE, the ONCVPSP atomic estimate: not used / < 1 / < 1 mHa per electron.
- Δ1 − Δ1(converged): < 2 / < 1 / < 0.5 meV.
- Total energy − converged: < 10 / < 5 / < 2 meV/atom.
- Cutoff mesh: 2 Ha steps near the ONCVPSP hint, then 10 Ha steps. The converged value is taken from the largest cutoff, initially 22 Ha above the high hint. Curves are inspected by hand, because the automatic evaluation "turned out to be too optimistic". [P1]
- B and B1 converge 10 to 100 times more slowly than V0 and Δ. [P1]
- The current code holds `ATOLS = (0.5, 0.3, 0.1)`, commented as tolerances on the deltafactor prime. [P3] This differs from the paper table and I did not resolve it.

**Generation choices**
- Two projectors per angular channel, with listed exceptions (H, He, O, F and others), giving log-derivative agreement "up to at least 3-5 Ha. In many cases ... up to 10 Ha". [P1]
- Continuity of the pseudised potential to fourth order (`ncon=4`) "to avoid possible problems in the computation of elastic properties". This costs about 5 Ha against `ncon=3`. [P1]
- Semicore versions are provided where partitioning is unclear, since "NCPPs with semi-core states are more accurate and transferable" and semicore "may be needed for accurate GW calculations". [P1]
- Standard and stringent tables. High-accuracy versions with small core radii are for high pressure and are "recommended for calculations in magnetic systems". [P1]
- Nonlinear core correction: "included in all the PSPs with electrons frozen in the core except for the third row semi-core PSPs (Na-sp – Cl-sp) and Ne". It uses Teter's two-parameter model charge, smooth in real and reciprocal space. [P1]
  - Reasons given: transferability, and removing "nonphysical oscillations of the local part close to the origin" that appear with GGA. [P1]
  - No clear correlation was found between hardness matching and solid-state quality. [P1]
- Unbound states: for positive-energy reference states an artificial bound state is made by adding a barrier, V_AEB(r) = V_AE(r) + v∞ θ(x) x³/(1 + x³), with x = (r − rc)/rb. [P1]

**Which failure shows in which test**
- Magnetic structure: Cr, Mn, Fe are Δ outliers in the standard table because "the structures used in the Δ-Gauge test are magnetic. To resolve the magnetic structure a harder PSP is needed". GBRV has no magnetic systems, so the high table does not improve it. [P1]
- Soft crystals: Δ′ exposes Hg, Sr, Ba (7.2, 6.1, 5.0) which Δ hides. Ru and Rh have barely acceptable Δ only because of their high bulk modulus. [P1]
- Correlations: fcc and bcc correlate strongly. Δ and Δ′ correlate partly. GBRV and Δ "hardly show any correlation". The acoustic-sum-rule error is "completely decoupled from all other tests". [P1]
- Ghosts appear only in the band-structure scan. Including 4d alone for fifth-row main-group elements gives ghosts 20 to 80 eV above the Fermi level. 5d-only transition metals have ghosts just above the Fermi level despite good Δ. [P1]
- Core correction: with 1s frozen and 2s, 2p filled it does not help and "Often they even become unstable". For magnetic 3d metals, well-balanced core corrections "dramatically" improve magnetic results. Very localised model charges are hard to converge "especially in DFPT". [P1]
- GBRV compound test: 63 rocksalts, 54 half-Heuslers, 138 perovskites; outliers are errors above 0.25 %. [P1] This count conflicts with the GBRV paper's 138 half-Heuslers. [G1]
- Norm-conserving potentials tend to underestimate GBRV lattice parameters against all-electron. [P1]

## 5. SSSP (Prandini et al. 2018; aiida-sssp-workflow)

**Protocol**
- Verification by Δ at 200 Ry, plus convergence against wavefunction cutoff Ec of four quantities, each measured against its value at Ec_ref = 200 Ry. [S1]
- Convergence runs use the elemental ground-state crystals, a 6×6×6 mesh (10×10×10 for O and the lanthanides) and no spin polarisation. SiF4 replaces F; rare-earth nitrides replace the lanthanides. 85 elements in all. [S1]

**Definitions**
- Phonons, by DFPT at q = (1/2, 1/2, 1/2): δω̄ = sqrt( (1/N) Σ_i |(ω_i(Ec) − ω_i(ref)) / ω_i(ref)|² ). The error bar is the maximum over i of the same ratio. If ω_max < 100 cm⁻¹, absolute deviations are used. [S1]
- Cohesive energy: δE_coh = |E_coh(Ec) − E_coh(ref)| in meV/atom, with the isolated atom in a 12 Å cell. [S1]
- Pressure: P = (1/3) Tr σ, with δP = P(Ec) − P(ref). Solve P_BM(V′) = δP on the all-electron Birch-Murnaghan curve, where P_BM(V) = (3 B0/2) [ (V0/V)^(7/3) − (V0/V)^(5/3) ] { 1 + (3/4)(B0′ − 4) [ (V0/V)^(2/3) − 1 ] }. Then δV_press = (V′ − V0)/V0. [S1]
- Band distance: η_v(A,B) = min over ω of sqrt( Σ_nk f̃_nk (ε^A_nk − ε^B_nk + ω)² / Σ_nk f̃_nk ), with f̃_nk = sqrt( f_nk(ε_F^A, σ) f_nk(ε_F^B, σ) ) and Fermi-Dirac f. [S1]
  - η10 is the same "with a Fermi level up shift of 10 eV".
  - max η = max over n,k of |ε^A − ε^B + ω|.
  - Smearing is 0.3 eV Fermi-Dirac (none for η_v of insulators), on a uniform 6×6×6 mesh over the full Brillouin zone without symmetry reduction.
  - When valence counts differ, compare only the common bands, dropping the extra low-lying ones.

**Thresholds** (paper Table 2) [S1]

| Quantity | Efficiency | Precision |
|---|---|---|
| δω̄ | < 2 % | < 1 % (< 1 cm⁻¹ if ω_max < 100 cm⁻¹) |
| δE_coh | < 2 meV/atom | < 2 meV/atom |
| δV_press | < 1 % | < 0.5 % |
| η10 | < 10 meV | < 10 meV |
| max η10 | < 20 meV | < 20 meV |
| Δ | < 1 meV/atom "if possible" | smallest available |

- The text says "within 2 cm⁻¹" for efficiency when ω_max < 100 cm⁻¹. [S1]
- 1 % in δV_press is 0.33 % in a cubic lattice parameter. [S1]
- The criteria are "general guidelines and not ... strict rules"; noble-gas solids are noisy. [S1]

**Duals and cutoffs**
- Norm-conserving always uses dual 4, with ρ(G) reaching twice the wavefunction G_max. Ultrasoft and PAW use dual 8. Duals 12 and 16 were checked for Mn, Fe, Co, Hf, O; dual 12 is suggested for Fe and Mn. [S1]
- Average suggested cutoffs are 44 Ry (efficiency) and 56 Ry (precision). [S1]
- The library JSON keys are `cutoff_wfc` and `cutoff_rho`, where cutoff_rho is the old cutoff times dual (page summary). [S3]

**Ghost detection and independence of tests**
- Ghosts are "signalled by extremely large values (of the order of eV or more) of the bands distances when computed with respect to other ghost-free PSPs". [S1]
- The paper's Table 3 lists SG15 1.0 potentials with ghosts within 10 eV of the Fermi level: Ar, Cs, In, S, Se, Sn, Te, Sb. "The latest release of the SG15 library (SG15-1.1) has no ghost states." [S1]
- For Na, all potentials agree to Δ < 0.3 meV yet one differs by η10 = 65 meV. [S1]
- "there is no strong correlation between pairs of tested quantities". [S1]

**Bands chessboard**
- I confirmed only that the SSSP element pages and archive ship "eta function chessboards" among the plots (search-result level). **Its exact definition (axes, colour scale) is UNVERIFIED.** It is presumably the matrix of pairwise η between all tested potentials of an element. The site https://sssp.materialscloud.org/ is JavaScript-rendered and returned no text.

**Workflow repo (newer protocol)** [S2]
- `criteria.yml` has `efficiency` and `precision` blocks matching the paper for cohesive energy (2 meV/atom), phonons (2 % and 1 %) and pressure (1 % and 0.5 %). Both list bands at 20 meV. A `v2024.1001` block lists phonons 1 %, pressure 0.5 %, bands 10 meV and an EOS bound of 0.02.
- `convergence.yml` `balanced`: cold smearing 0.02 Ry, k-distance 0.15 Å⁻¹, `conv_thr_per_atom` 2e-10, q = (0.5, 0.5, 0.5), `tr2_ph` 1e-18, 12 Å vacuum.
- `control.yml` `standard` wavefunction scan: 30 to 80 in steps of 5, then 90, 100, 120, 150, 200 Ry. Dual scans are 2.0 to 4.0 for norm-conserving, 6 to 8 otherwise, and 8 to 18 for high-dual elements.
- `docs/spec.md`: dual starts at 4 for norm-conserving and 8 for PAW/ultrasoft, "but 18 for high dual elements ... O, Fe, Hf, Mn, Co, Ni, Cr". The precision measure uses the ACWF configurations and the ν metric.
- CLI: `aiida-sssp-workflow launch --property ... --protocol ... --cutoff-control ... --criteria efficiency <upf>` (page summary). [S4]

**Versions**
- The latest archive record I found is v11 (April 2023) with SSSP 1.3.0 (page summary). [S3] **An "SSSP 2.0" release or paper is UNVERIFIED.** I found none.

## 6. Atom-level tests before any solid-state test

**Norm conservation and first-order transferability**
- HSC: "logarithmic derivatives of real and pseudo wave functions and their first energy derivatives agree for r > rc guaranteeing excellent transferability". [C1]
- Identity: −(1/2) r² |ψ(ε; r)|² d/dε { d/dr ln ψ(ε; r) } at r = rc, ε = ε′ equals ∫∫ from 0 to rc of |ψ(ε′; r)|² r² dr dΩ. So equal norm inside rc means log derivatives agree to first order in energy around each reference level. [F1 Eq. 1]
- Construction conditions: same eigenvalue; same log derivative and amplitude for r > rc; ∫ from 0 to r′ of |u_ps|² dr = ∫ from 0 to r′ of |u_AE|² dr for r′ ≥ rc. [F1]
- ONCV: with two projectors and the generalised condition ⟨φ_i|φ_j⟩_rc = ⟨ψ_i|ψ_j⟩_rc, the energy derivatives of the log derivatives match at each projector energy ε_i, "extending this property ... to several energies". [H1]

**Log-derivative test**
- D_l(ε, r_diag) = d/dr ln u_l(ε; r) at r_diag, taken outside the core, "say half of a typical interatomic distance". Pseudo and all-electron should "closely agree over the whole range of energies where the valence states are expected to form Bloch bands or molecular orbitals, say ±1 hartree around the atomic valence eigenvalues". [F1]
- ONCVPSP plots atan(r (dψ/dr)/ψ) at the largest rc (max(rc) + 1 when shallow cores are included), with the energy range set in the input. [O3]

**Eigenvalue and excitation-energy tests**
- Compare E_ba = E_tot(f_b) − E_tot(f_a) for pseudo, all-electron and frozen-core all-electron. A transferable potential performs like the frozen core, |ΔE_PS| ≃ |ΔE_FC|. "Typically, say for the first ionization potential, the errors do not exceed a few ten meV". [F1]
- ATOM tutorial: "The typical difference should be of around 1 mRyd for a 'good' pseudopotential", followed by "The real proof of good transferability ... can only come from a molecular or solid-state calculation". Its example configurations for N are 2s² 2p³, 2s^1.8 2p^3.2, 2s^1.6 2p^3.4, and the ion 2s² 2p^3.7. [J1]
- ONCVPSP `run_config`: up to four extra valence configurations; eigenvalues and excitation energies are compared with all-electron, but "its utility is not clear". [O3]
- Hamann: rms excitation error for one- and two-electron ionised states is 0.012 Ha (KB) against 0.003 Ha (ONCV), yet "the correlation between these results and results for solids was at best difficult to discern". [H1]
- ONCVPSP per-state diagnostics: "e in and e test should closely match, and the norm and slope values should be close to one". For kinetic and potential tests "the ratio should be close to one". [O2]

**Chemical hardness**
- Teter: hardness is "one-half the second derivative of the total energy ... with respect to the number of electrons". Requiring atom and pseudo-atom energies to "match to second order with arbitrary changes in valence-state occupancy yields major decreases in the errors". "The concept also clarifies the role of core density". [C1]
- Hardness-matrix test: H_ij = ∂ε_j/∂f_i = ∂²E_tot/∂f_i ∂f_j. Pseudo eigenvalues should track all-electron ones under occupation changes to within the frozen-core error. [F1]
- Caveats: "no systematic evidence that 'Teter hardness metric' accuracy correlates with improved agreement between all-electron and pseudopotential results". [O4] PseudoDojo "did not observe a clear correlation". [P1]

**Ghost states: single Kleinman-Bylander projector**
- Gonze 1990: "a theorem which tells if a 'ghost' state occurs below the zero-node state of the atom". Gonze 1991: spectral and log-derivative analyses. [C1]
- Criterion as implemented in fhi98PP `pswatch`. Let ε̃_i(0) be the bound levels of the local potential alone. Then:
  - ε̃_0(0) < ε̃_0(E_l) < ε̃_1(0) < ... if E_l^KB > 0.
  - ε̃_0(E_l) < ε̃_0(0) < ε̃_1(E_l) < ... if E_l^KB < 0.
  - No ghost lies below the reference level ε_l^ps when (E^KB > 0 and ε̃_0(0) < ε_l^ps < ε̃_1(0)), or (E^KB < 0 and ε_l^ps < ε̃_0(0)). [F1 Eqs. 40 to 43]
- Two further signs: marked deviation of the KB log derivative from the semilocal or all-electron one, and extra levels in the KB bound-state spectrum. [F1]

**Ghost states: two-projector ONCV**
- The KB prescription "does not generalize to the multiprojector case, but we can test a potential by scanning the logarithmic derivative it produces outside rc over a sufficiently wide range of energies below the lowest desired eigenvalue. A spurious step ... signals the occurrence of a ghost"; "the second projector ... is very effective at suppressing ghosts". [H1]
- "no such test exists for multiple-projector potentials". [O1]
- ONCVPSP 3.3.0 added a "ghost detector": the pseudo-atom in a hard-wall sphere is diagonalised in a radial basis, and eigenvalues below the wanted bound states are flagged ("Testing for bound ghosts" in the output). It is "beta" and not implemented for fully relativistic runs. [O1]
  - It found a −208 Ha f ghost in the 60_Nd example, removed by changing `lpopt` from 3 to 5.
- Log-derivative plots "remain the only way to assess the general quality of the potential and identify shallow bound ghosts [or] ghost resonances above 0". Sign-change jumps of π can mimic ghosts and are cancelled by an algorithm since 2.0.1. [O1]
- Remedy: "adjusting the local potential to be closer to the semi-local potential of the ghost l often works". [O1]

**Atom-level convergence estimate**
- ΔE_kin,l(k_pw) = ∫ from 0 to k_pw of (k²/2)|u_l(k)|² dk − ⟨u_l| −(1/2) d²/dr² + l(l+1)/(2r²) |u_l⟩. A converged solid calculation "typically corresponds to a plane wave cutoff that meets the criterion ΔE_kin ≤ 100 meV for the pseudo atom". [F1]

## 7. Other tests expected for a general-purpose potential

**Cutoff convergence**
- Each property must be tested separately: "convergence of one quantity does not imply convergence of the other". Stress and phonons "can ... display a slower convergence". [S1]
- Quantified targets are the SSSP thresholds (section 5) and the PseudoDojo hints (section 4).
- I found no protocol with an explicit threshold on forces. Phonon frequencies and pressure play that role in SSSP and PseudoDojo. **A force-convergence threshold is UNVERIFIED.**

**Magnetic systems and nonlinear core correction**
- Louie-Froyen-Cohen: treating nonlinear core-valence exchange-correlation explicitly gives "significant improvement in the transferability"; "spin-polarized configurations are well described with a single potential". [C1]
- It is needed "in studies involving alkali metal atoms or within spin-density functional theory" and matters more to the left of the periodic table and for heavier atoms. Choose r_nlc "about the radius where the full core density drops below the valence electron density". Decide by "a comparison of two calculations with and without a partial core density". [F1]
- The Δ set contains magnetic O, Cr, Mn, Fe, Co, Ni. [D2] GBRV tests rocksalt transition-metal-oxide moments. [G1]
- ACWF v1 is non-magnetic only. [A1] Its repo also holds "PBE-magnetic" unary structure sets, which I did not analyse. [A3]
- SG15 "does not have non-linear core corrections", according to PseudoDojo. [P1]

**Molecules**
- G2-1 atomisation energies against all-electron, with errors of "only half a kcal/mol", for core-corrected HGH-type potentials. [X6]
- SG15 added the H and N dimers to its test set because earlier-generation potentials did not describe the bond length. [X7]
- GBRV updated B, C, N, O to v1.01 in August 2013 "to fix poor performance in small molecules" (page summary). [G2]
- "It is ... good practice to also cross check ... bond lengths in molecules or [in] bulk crystals ... with all-electron results" using the same functional. [F1]

**High pressure and short bonds**
- The XO3 oxides act as a short-bond proxy. [A1]
- Use small-core-radius versions "for the study of systems under high pressure". [P1]
- Willand et al. report reliability "for high pressure phases of crystalline solids". [X6]

**Phonons**
- Γ-point DFPT with the acoustic-sum-rule check (section 4). Zone-boundary phonon convergence (section 5).
- "Model core charges often produce convergence problems, especially with phonon calculations and violations of the acoustic sum rule at Gamma". [O4]
- GBRV 1.4 was issued to improve DFPT phonon convergence, and Na 1.5 to fix phonon problems (page summary). [G2]

**Band structures, conduction bands, GW**
- η10 and max η10 against other potentials. The bands distance "could in principle also be used for verification ... with reference all-electron band structures". [S1]
- Ghost scan to about 200 eV for GW and optics. [P1]
- Semicore states for the exchange self-energy. [P1]
- Consistent potentials reproduce all-electron band gaps with a mean absolute error of about 0.02 eV over 473 solids. [X1]

**Cross-code check**
- Run the same file in two codes or formats. Radial-mesh truncation (UPF 10 Bohr against psp8 6 Bohr) changed Cu, Zn, Ne results. [A2 S14]
- SG15 notes that ONCVPSP 2.1.1 echoes slightly modified core radii, so exact reproduction needs the original `.dat` inputs. [X8]

**Transferability across functionals**
- LDA potentials in GGA calculations give results that "differ significantly" from consistent ones, mainly through core-valence exchange-correlation. [X5]
- Band gaps with inconsistent potentials: the mean absolute error rises "by a factor of three or more", to about 0.1 eV, and "in few specific (and unpredictable) cases can be quite large ... several eV". Magnetic materials were not tested, and preliminary results were as bad or worse. [X1]
- Hybrids: PBE0-consistent potentials improve agreement with all-electron. [X2] Consistency errors with PBE-based potentials for hybrid and range-separated functionals average 0.4 %, reach −1.4 %, and are mostly systematic. [X3]
- Meta-GGA: SCAN "shows more significant dependence on the fast Fourier transform grid"; potentials from a lower-level functional "could introduce noticeable errors ... such as phase transition energetics". [X4]
- SSSP: convergence patterns transfer between PBE, PBEsol and PZ when generation parameters are identical, but for SCAN or HSE the PBE-potential approach "would not ensure the transferability". [S1]
- The GBRV LDA and PBEsol sets "were not specifically tested" (page summary). [G2]

## 8. Pseudo-atomic orbitals in the file (UPF PP_PSWFC / PP_CHI)

**What the format holds**
- PP_PSWFC is optional. Each PP_CHI is the "radial atomic (pseudo-)orbital (radial part of the KS equation, multiplied by r)". `number_of_wfc` "May not coincide with the number of atomic states used in PP generation". Units are Rydberg atomic units (page summary). [Q4]
- Attributes QE writes per PP_CHI: `size`, `index`, `label`, `l`, `occupation`, `n` (only if n > l), `pseudo_energy` (only if > 0), `cutoff_radius`, `ultrasoft_cutoff_radius`; `nn` and `jchi` for spin-orbit files. [Q3 write_upf_new.f90]
- ONCVPSP has written PP_PSWFC since 3.2.1: "The upf output now includes pseudo wave funtions for the occupied states. This will allow LDA+U calculations in some codes. Please note that ONCVPSP potentials have not been tested or benchmarked for this use." [O5]
  - It writes one PP_CHI per valence state with `occupation`, `pseudo_energy = 2*ea` (Ha to Ry), `label` (n and S/P/D/F) and `l`. [O6]
  - It extends the linear output mesh until each orbital's norm is "accurate to 10^-6". [O6]
- SG15 was generated with ONCVPSP 2.1.1, which predates that feature. A v1.2 UPF set was regenerated on 2020-02-06. [X8]

**Required properties**
No published checklist exists that I could find. The following is my synthesis from how the codes consume the data.
- **Normalised to 1 on the radial grid.** QE offers `norm-atomic` only as "a 'quick and dirty' trick to be used when atomic orbitals from the pseudopotential are not normalized (and thus produce occupation whose value exceeds unity)". [Q1]
- **Decayed inside the grid, in practice inside 10 Bohr.** QE integrates atomic functions by Simpson only up to `rcut = 10.d0` a.u. [Q3 read_pseudo.f90, atwfc_mod.f90] ONCVPSP uses a 1e-6 norm-tail criterion. [O6]
- **Labelled, with correct l and occupation ≥ 0.** QE uses only orbitals with `oc >= 0`. [Q3 atwfc_mod.f90] QE stops with "does not contain labels for atomic orbitals! Please add them by hand in the pseudo." or "Mismatch between the requested and available manifolds" when the Hubbard manifold label (for example 3d) is absent. [Q3 offset_atom_wfc.f90]
- **A bound-state solution of the pseudo-atom in the reference configuration**, with eigenvalue equal to `pseudo_energy` and the right node count. These follow from the format's definition as a radial Kohn-Sham solution. [Q4] JDFTx reads `pseudo_energy` (×0.5 to Hartree) as atomic eigenvalues and files orbitals by l in file order. [JX]
- **Orthonormal within one atom.** Timrov et al. assume this: orbitals are "orthonormal within each atom ... but not between different atoms". [U2] This matters when both semicore and valence functions of the same l are stored.

**Quantum ESPRESSO dependence**
- `startingwfc='atomic'`: "If not enough atomic orbitals are available, fill with random numbers the remaining wfcs". [Q1]
- `projwfc.x` "projects wavefunctions onto orthogonalized atomic wavefunctions, calculates Lowdin charges, spilling parameter, projected DOS". [Q2]
  - In code: `psum = SUM(charges)/nelec`, then `Spilling Parameter = 1 − psum`, citing Sánchez-Portal 1995. [Q3 projwfc.f90]
- Spilling definition: S_Ω = (1/N_k)(1/N_α) Σ_k Σ_α ⟨ψ_α(k)| (1 − P(k)) |ψ_α(k)⟩, with P(k) = Σ_μν |φ_μ(k)⟩ S⁻¹_μν(k) ⟨φ_ν(k)|. It lies between 0 and 1. Restricted to occupied states it is the charge spilling S_Q. [L1]
- Without PP_PSWFC, `projwfc.x` stops with "Cannot project on zero atomic wavefunctions!" (reported for SG15, 2016). [M1 msg29931, msg29933]
- With a mix, species lacking orbitals are silently skipped. In a 2021 Pb/I case, I had PSWFC and `Pb_ONCV_PBE_FR-1.0.upf` did not. Grafting PSWFC in after the SCF run gave "wrong # of atomic wfcs". [M1 msg40120, msg40122]
- Workarounds posted: regenerate from the `PP_INPUTFILE` block with ONCVPSP 3.2.3 or later. [M1 msg29934] Or copy PP_PSWFC from an ld1.x potential on the same grid, called "quick ... and dirty". [M1 msg29933] Paulatto warns that regenerating SG15 with a different ONCVPSP version "can produce ghost states". [M1 msg40121]

**DFT+U**
- Projector types in the HUBBARD card [Q1]:
  - `atomic`: "use atomic orbitals (read from pseudopotential) to build the Hubbard projectors".
  - `ortho-atomic`: "use Lowdin orthogonalized atomic orbitals. This option is recommended to be used whenever possible instead of atomic because it allows to avoid applying Hubbard corrections twice in the orbital overlap regions."
  - `norm-atomic`: Löwdin normalisation only, not orthogonalised.
  - `wf`: Wannier functions from `pmw.x`.
  - `pseudo`: beta projectors; "a pseudopotential with the all-electron atomic orbitals are required (i.e., as generated by ld1.x with lsave_wfc flag)".
  - Forces and stress exist only for `atomic`, `ortho-atomic` and `pseudo`.
- With `ortho-atomic` or `norm-atomic`, any species without PP_PSWFC triggers "no atomic wavefunctions in pseudopotential file ... use a pseudopotential file with atomic wavefunctions!". [Q3 offset_atom_wfc.f90]
- Sensitivity: U and V "depend strongly on the choice of projector functions ... as well as on the type of PPs, oxidation state". Long orbital tails make "formal occupation numbers ... substantially overestimated". [U2]
- "the value of U obtained for the same structure but with differing pseudopotentials may easily differ by as much as 2–3 eV, in particular if the pseudopotentials were generated from different oxidation states". Hubbard parameters "are not transferable"; keep "the same Hubbard projectors, pseudopotentials, oxidation states, functionals". [U1]

**LOBSTER**
- It does not use PP_PSWFC. It projects PAW wavefunctions onto its own contracted Slater-type local basis, fitted to atomic functions and supplemented by functions from free-atom VASP PBE calculations (page summary). [L2]
- Quality metric: "absolute charge spilling", the k-weighted average over occupied bands of |1 − O_jj(k)| for the projected-band overlap. [L2][L3]
- Example: Be went from roughly 19 %, "unacceptable", to 1.73 % after adding 2p (page summary). [L2]
- In a 1520-compound VASP + LOBSTER database, "Only a very few compounds show a charge spilling of > 5 %"; nine outliers range from 5.5 to 46.7 % (BaO2). [L3]
- LOBSTER is described as PAW-based. [L4] **Norm-conserving support and the QE interface are UNVERIFIED.**

---

## Could not verify

1. The per-method Δ values in Science 2016 Fig. 4 (SG15, PD0.1, GBRV, SSSP against the all-electron average). The figure text could not be extracted, the SI returned 403, and the static project page returned 403. The table in section 1 is my recomputation from the package's own data against WIEN2k 13.1 only.
2. The WIEN2k numerical settings behind the Δ reference (R·Kmax, k-mesh, smearing) and any k-point or smearing recipe recommended by the Δ authors. These are in SI tables S3 to S42, which I could not access.
3. The exact definition of the SSSP "bands chessboard". Only its existence as "eta function chessboards" is confirmed.
4. An SSSP 2.0 release or paper. I found only the workflow repo with a `v2024.1001` criteria block; the newest archive record found is 1.3.0.
5. The PseudoDojo hint tolerances differ between the paper table (Δ1: 2, 1, 0.5 meV) and the code constant `ATOLS = (0.5, 0.3, 0.1)`. Not resolved.
6. The GBRV set sizes for rocksalt and perovskite. The GBRV paper gives only 138 half-Heuslers; PseudoDojo states "63 rocksalt, 54 half-heuslers, 138 perovskites".
7. An explicit force-convergence threshold in any of these protocols.
8. LOBSTER: the basis name "pbeVaspFit2015", QE or norm-conserving support, and any official acceptance threshold for charge spilling. The verbatim spilling equation was not captured.
9. The original text of Gonze 1990 and 1991, Teter 1993, HSC 1979, Louie-Froyen-Cohen 1982 and Sánchez-Portal 1995 (APS returned 403). I used abstracts plus the Fuchs-Scheffler, Hamann, ONCVPSP and 1996 follow-up accounts.
10. The remark attributed to P. Giannozzi in a January 2019 QE thread, "ONCV pseudopotential files do not contain atomic orbitals", was seen only in a search summary. lists.quantum-espresso.org has an expired certificate, so I read threads through mail-archive.com summaries.
11. Whether SG15 v1.2 UPF files (2020-02-06) all contain PP_PSWFC. Mailing-list reports say some SG15 files do and others do not. I did not download the tarball.
12. No published checklist for PP_PSWFC validity was found (norm, nodes, eigenvalue equal to pseudo_energy, decay). The list in section 8 is inferred from code behaviour and is not a cited standard.
13. Lines marked "(page summary)" were not read verbatim.

## Files

The Delta recomputation is kept in `~/vs/research/crawfish/anl/pseudo_type_comparison/delta_history/`:
`delta_hist.py` (written new, because the downloaded `calcDelta.py` uses the removed `np.float`; it copies that
script's v3.0 formulas and asserts Δ = 0 for identical inputs), its data (`WIEN2k.txt`, `history/`) and its output
`delta_hist.txt`. The ONCVPSP documentation quoted here is in `~/Desktop/jdftx_hold/oncvpsp/src/doc/`.

Skipped: I did not run any DFT calculation, and I did not check the paywalled originals, the Science SI or the SSSP web app. The main risk is the items read through summarising fetches (GBRV site, LOBSTER paper, mailing-list threads, archive records), which could be paraphrased inaccurately.
# SG15: generation method, validation, release history, comparable ONCV libraries

> **What this file is.** A research report written by a Claude subagent on 2026-11-25. The first agent was stopped
> early (token budget). A second agent finished it the same day: it re-ran every [computed] item from the upstream
> files (all are now [verified]; the description of version 1.2 was wrong and is corrected in section 3), re-read
> the arXiv sources, the SG15 index page and the GitHub items behind sections 1 to 3, and worked through the open
> list (section 5). Not
> re-checked: most of section 4, and the items marked "not re-checked". The instructions that use this report are
> in `../pseudopotential_generation.md`.
>
> **2026-11-26:** the published paper (Comput. Phys. Commun. 196, 36 (2015), open access under CC BY 4.0; the
> user's copy is `~/Desktop/Crawfish/schlipf2015.pdf`) was read in full by the main session. What it adds or
> settles is tagged **[read, CPC]**. Its supplementary material (`~/Desktop/Crawfish/schlipf2015-si/`: the
> construction inputs and an 11-page table of every material) was checked the same day with
> `sg15_history_checks/check_si.py`; those items are tagged **[verified 2026-11-26, check_si.py]**.
>
> Tags: **[read]** = read in the named source ("re-read" = checked again in the second pass);
> **[verified 2026-11-25, sg15_history_checks/SCRIPT]** = the agent's own analysis of the upstream SG15 files,
> re-run with a script kept, with its output, in
> `~/vs/research/crawfish/anl/pseudo_type_comparison/sg15_history_checks/` (README there says how to rerun).
> The upstream files are in `~/vs/research/crawfish/anl/pseudo_type_comparison/sg15_distribution/upstream/`,
> `.../sg15_history_checks/downloads/` (1.2 XML tarball, index page, arXiv source of the SG15 paper) and
> `~/Desktop/jdftx_hold/oncvpsp/smoke_test/sg15_full/`.

## 1. How SG15 was generated

- **Code:** ONCVPSP scalar-relativistic 2.1.1 (03/26/2014). [read, re-read]
  http://www.quantum-simulation.org/potentials/sg15_oncv/index.htm and UPF `PP_INFO`.
- **Modifications:** the full tarball `sg15_oncv_2015-10-07.tar.gz` ships Hamann's source plus four files in
  `src/`: three replace his (`check_data.f90`, `oncvpsp.f90`, `upfout.f90`) and one is new (`fpmdout.f90`, the
  XML writer). The changes are I/O only: `psfile` may be `upf+fpmd`, the input echo prints 5 decimals, the UPF
  header gets the right version string (Hamann's 2.1.1 `upfout.f90` still printed "1.1.1, 08/28/2013") and
  `<UPF version="2.0.1">`. [verified 2026-11-25, sg15_history_checks/check_inputs.py]
- **Fixed in all 86 `.dat` inputs** (69 at 1.0, 17 at 1.1): `iexc=4` (PBE), `ncon=5`, `nbas=8`, `nproj=2`,
  `lloc=4`, `lpopt=5`, `dvloc0=0`, `icmod=0` (no NLCC), `rlmax=6.0`, `drl=0.01`; also `epsh = -5.0 3.0 0.02`
  and `ncnf=0`. [verified 2026-11-25, sg15_history_checks/check_inputs.py]
- **Varied:** `rc(l)` 1.13–3.92 bohr, `qcut(l)` 2.53–11.35, local `rc(5)` 0.49–2.89 bohr, `debl` 0–3.84 Ha, and
  `ep`. `ep` is 0 for all 222 channels whose l has a valence state in the reference configuration, and non-zero
  (−0.008 to 0.277 Ha) for the 28 channels whose l has none.
  [verified 2026-11-25, sg15_history_checks/check_inputs.py]
- **lmax:** 0 for H/He, 1 for Li–Ar (no d projectors), 2 for K–Cs, 3 for Ba, La and Hf–Bi; the same in an
  element's 1.0 and 1.1 input. [verified 2026-11-25, sg15_history_checks/check_inputs.py]
- **Algorithm** [read, arXiv:1502.00995 TeX source, re-read]:
  - Start from the ONCVPSP example inputs ("where available, or generated our own PP otherwise") and randomly
    perturb all N parameters to make N+1 potentials.
  - Run Nelder–Mead for 80–200 iterations, with GBRV partners in the compounds.
  - Restart with the first-generation potentials as partners for about 100 more iterations.
  - The score is the geometric mean of the quality function over the training materials.
- **Algorithm, details in the published paper** [read, CPC sections 2 and 3.4]:
  - The paper marks three kinds of input as optimised: the upper-limit wave vector `qc`, the projector radius
    `rc`, and the energy shift for unoccupied states. Fixed: M = 5 continuity constraints, N = 8 basis functions,
    two projectors per l ≤ lmax. This matches the `.dat` analysis above.
  - The starting potentials are for the neutral atom, with possible semicore states in the valence. The initial
    simplex changes any input parameter by up to 20 %, keeping necessary conditions such as projector radius ≥
    local-potential radius.
  - Convergence is judged by eye on the quality, plus input parameters converged to two significant digits.
  - **Rejection step:** a candidate "is rejected if it cannot be created by the ONCVPSP code or if it exhibits
    ghost states in the vicinity of the Fermi energy. The latter is evidenced by changes in the logarithmic
    derivative as compared to the AE result."
  - **That step is new in the published version.** The arXiv v1 source does not contain the word "ghost"
    [verified 2026-11-26, grep of `sg15_history_checks/downloads/arxiv_src/oncv_paper.tex`]. The published
    acknowledgments thank "I. Castelli and N. Marzari for pointing out to us the importance of rejecting
    pseudopotentials with ghost states". Dates: received 13 February 2015, revised 22 April 2015, accepted 13
    May 2015.
  - The valence choice per element, the construction inputs and every lattice constant and bulk modulus are in
    the supplementary material at the DOI (10.1016/j.cpc.2015.05.011).
- **The supplement documents version 1.0, for every element** [verified 2026-11-26, check_si.py]:
  - Its `input.dat` (dated 2015-05-27) has 69 inputs. Each equals the tarball's 1.0 `.dat` number for number,
    including the 14 elements that got a 1.1 file on 2015-05-20 and Hf, Ta, W with 4f in valence.
  - Its Table I (valence states) agrees with those inputs for all 69 elements.
  - So the paper's tables and inputs describe 1.0. **The authors published no lattice-constant validation of the
    1.1 files.** The only independent numbers for 1.1 are the Δ-test entries of the 2015-05-20 release
    (section 2).
- **Quality function** [read, re-read]: δ = a_PP/a_AE − 1, target δ0 = 0.2 %.
  - For δ < 2δ0: q = A + Cδ² + Dδ³ + Eδ⁴ + Fδ⁵, with A = 1 + 1280/Ecut and y0 = 1 + 680/Ecut.
  - For δ ≥ 2δ0: q = (2δ0/δ)².
  - δ is made monotonic in cutoff, from 160 Ry down to 40 Ry in 10 Ry steps. Published Eq. 11:
    δ_corr(E_i) = |δ(E_max)| + Σ_{k=1..i} |δ(E_k) − δ(E_{k−1})|, with E_i = E_max − 10·i Ry, so the value at a
    cutoff is an upper bound for every larger cutoff. [read, CPC]
  - Neither the arXiv nor the published text says how q is combined over those 13 cutoffs. [read, CPC]
  - q(δ0) = y0, q(2δ0) = 1 and a continuous slope at 2δ0 hold only if the powers of δ0 printed for C and D are
    swapped (C ∝ δ0⁻², D ∝ δ0⁻³); as printed, q(δ0) is off by orders of magnitude.
    [verified 2026-11-25, sg15_history_checks/check_qfunc.py] The published Eq. 10 prints the same powers
    (C over 4δ0³, D over 4δ0²). [read, CPC]
- **Cutoff target:** there is no explicit 60 Ry target. The abstract reports good agreement "with a plane-wave
  energy cutoff of approximately 60 Ry" as an outcome. [read, re-read]
- **Reference data** [read, re-read]: FLAPW (FLEUR), 8×8×8 k-mesh, Fermi smearing 0.001 Ha (315.8 K), Murnaghan
  fit through 11 points. Inside the loop a parabola through a_AE and ±1 % is used, run with Quantum ESPRESSO.
  - Published additions [read, CPC section 3.3]: the all-electron lattice constant is converged to 0.1 %
    relative, which is why δ0 was set to 0.2 %. Stated FLAPW limits: "only core states are treated fully
    relativistically", non-spherical potential terms only for valence states, and basis-set linear dependence.
  - The paper says nothing about the relativistic treatment of the pseudopotentials themselves, and never
    mentions spin polarisation or magnetism. [read, CPC, whole text]
- **Materials** [read, re-read]: 602 in total. Training = bcc, fcc, rock-salt; test = sc, diamond, zincblende.
  sc was added to training for Sc–Mn. The H2 and N2 dimers were added too; the arXiv sentence says "into the
  test set", inside the subsection "Refining the training set" (the published text is the same).
  - Counts for SG15 in the published tables: 69 each for bcc, fcc, sc and diamond, 163 each for rock-salt and
    zincblende, so 301 training and 301 test materials. Compounds pair groups by common oxidation state, using
    the three smallest elements of each group; transition metals are paired with C, N and O; noble gases appear
    only in the four elemental structures. [read, CPC section 3.2 and Tables 1–6]

## 2. Validation

- **Authors** [read, arXiv:1502.00995 TeX source, re-read]:
  - Lattice constant rms error 0.08 %; fraction with |δa| > 0.2 % is 30.07 % at 40 Ry, 4.65 % at 60 Ry,
    1.99 % at 160 Ry.
  - Bulk modulus rms error 2.40 %.
  - Named outliers: C, Ca (bcc); Cd (fcc); Ca, La (sc); B, Cl, Sc, Ni, Rb, Y (diamond); BeO (zincblende).
  - No Delta value, relativity statement, magnetism, band-structure or ghost tests appear in arXiv v1. arXiv has
    only v1 (checked 2026-11-25).
  - **The published paper agrees with all of the numbers above** (Table 8: average δa 0.01 %, rms 0.08 %;
    30.07 / 4.65 / 1.99 %; average δB −0.14 %, rms 2.40 %; |δB| > 5 % for 54.49 / 13.62 / 3.82 % at 40 / 60 /
    160 Ry; 602 materials) and the same outliers. Rms δa per structure: bcc 0.08, fcc 0.07, rock-salt 0.06,
    sc 0.09, diamond 0.12, zincblende 0.07 %. [read, CPC Tables 1–8]
  - Beyond the two sets (section 4.3, Table 7): bond lengths of H2, N2, O2, F2, Cl2, Br2 against the NIST CCCBDB
    are within 0.2 % except O2 (0.25 %) and F2 (0.35 %); six perovskites, worst LaAlO3 at −0.13 %; six
    half-Heuslers, all within 0.2 %. [read, CPC]
  - The published paper still has no Delta value for SG15, no magnetic or spin-polarised test, no band-structure
    test and no phonon test. Ghost states appear only as the rejection step of section 1. [read, CPC]
- **Per-material results from the supplement's Table II** (FLAPW and SG15 lattice constant and bulk modulus of
  all 602 materials, converged cutoff, version 1.0) [verified 2026-11-26, check_si.py; output `check_si.txt`]:
  - Recomputed from the table: mean δa +0.007 %, rms 0.081 % (paper 0.01, 0.08); |δB| > 5 % for 3.82 % (paper
    3.82). The table prints 3 decimals, so 15 materials come out above 0.2 % against the paper's 12, and the
    rms δB is 2.82 % against 2.40 % (bulk moduli below 1 GPa are printed to one digit).
  - Above 0.2 %: Sc diamond −0.43, C bcc +0.34, Y diamond +0.29, Cl diamond +0.25, Ca sc +0.23, Cd fcc +0.22,
    B diamond −0.22, Ni diamond +0.22, La sc −0.21, Rb diamond −0.21, BeO −0.21; borderline at 0.20–0.21: Ga bcc,
    Sr sc, Hf sc, Ca bcc.
  - Largest bulk-modulus errors with a good lattice constant: He sc −29 %, Xe sc −25 %, Kr fcc +17 %, Hg sc
    +16 %, S bcc +14 %, Hg diamond −13 %, Sb bcc +13 %.
  - **The 3d magnets are unremarkable in these non-magnetic structures:** largest |δa| over bcc, fcc, sc and
    diamond is Fe 0.06 %, Mn 0.12 %, Co 0.14 %, Cr 0.18 %, Ni 0.22 %. Their Δ-test errors (Cr 21, Mn 13,
    Fe about 5 meV/atom, below) come from the magnetic ground states, which the paper never ran.
- **Delta test:**
  - SG15 2015-01-24 agrees between Quantum ESPRESSO and CASTEP to 0.3 meV/atom. [read, Science 351, aad3000;
    not re-checked]
  - Recomputation from the archived EOS data (github.com/molmod/DeltaCodesDFT, vs WIEN2k 13.1) gives
    1.35 meV/atom (QE, 67 elements, no Tc) and 1.46/1.44 (CASTEP, 01-24/05-20 releases, 68 elements).
    [verified 2026-11-25, sg15_history_checks/check_delta.py; `delta_hist.py` rerun gives the stored
    `delta_hist.txt` exactly]
  - Worst elements (QE / CASTEP 01-24): Cr 21.1 / 20.8, Mn 13.0 / 13.0, Fe 4.7 / 5.2, Cd 3.1 / 3.5, B 3.1 / 3.1,
    Co 3.0 / 3.4, Pt 2.4 / 3.5, Ni 2.4 / 3.1. Without Cr, Mn and Fe the mean is 0.80 (QE) and 0.92 (CASTEP).
    [verified 2026-11-25, sg15_history_checks/check_delta.py]
  - The same functions give GBRV-1.4 0.99 (QE) and 1.06 (CASTEP), and PseudoDojo-0.1 0.66 (Abinit).
    [verified 2026-11-25, sg15_history_checks/check_delta.py]
- **QuantumATK paper (third-party, GBRV-type test)** [read, arXiv:1905.02794 TeX source, Table "delta"]: with
  its plane-wave engine, Δ is 1.32 meV for SG15 and 1.04 for PseudoDojo; the RMS error of lattice constant and
  bulk modulus on the GBRV rock-salt set is 0.16 % against 0.09 %, and on the perovskite set 0.13 % against
  0.06 %. "PseudoDojo PPs are slightly more accurate than the SG15 ones." The SG15 version is not stated.
- **PseudoDojo paper:** "The SG15 table was designed for efficiency and does not have non-linear core
  corrections." [read, arXiv:1710.10138 TeX source, re-read]
- **SSSP paper** [read, arXiv:1806.05609 TeX source, re-read]:
  - Its ghost table lists SG15 (1.0) for Ar, Cs, In, S, Se, Sn, Te, Sb: "ghost states in the conduction bands up
    to 10 eV above the Fermi level". "The latest release of the SG15 library (SG15-1.1) has no ghost states."
    Ghosts were detected as eV-size band distances to other pseudopotentials.
  - All eight are among the 14 elements the SG15 authors updated to 1.1 on 2015-05-20.
  - The paper tests "SG15, versions 1.0 and 1.1" and does not say whether the files were the original 2.1.1
    files or regenerated ones (section 5).
  - Its protocol is itself a third-party test of SG15 beyond Δ: cutoff convergence of phonon frequencies,
    cohesive energy, pressure and bands, per element. SSSP 1.1 chose SG15 for 15 elements in "efficiency"
    (He, Ne, Ar, Sc, Kr, Rb, Mo, Tc, Ru, Rh, Pd, Ag, Xe, Au, Hg) and 18 in "precision" (H, He, Be, Ne, Na, Ar,
    Cu, Kr, Rb, Mo, Tc, Ru, Rh, Pd, Ag, Xe, Au, Hg); Ar and Xe at 1.1.
  - For fcc Pb, "the SG15 bands deviate from the other bands around 7-10 eV over the Fermi level".
- **SSSP 1.2/1.3 precision** [read, arXiv:2305.17274 SI TeX source, re-read]:
  - Dropped SG15 for the noble gases (the SI lists Ne, Ar, Kr, Xe, Rn) in favour of pslibrary PAW, which
    "provide significantly better agreement with the all-electron reference" EOS curves.
  - The SI says the SG15 Hg and I potentials "are less precise" and "lead to the electronic step convergence
    issue". Its own table has Hg as SG15 in precision 1.1.2, but I as pslibrary PAW.
  - That table labels the SG15 entries of precision 1.1.2 `NC|SG15|v0|oncvpsp3`: generated with ONCVPSP 3.
  - The SSSP changelog is at https://archive.materialscloud.org/records/rcyfm-68h65. [read; not re-checked]
- **ACWF (Bosoni et al.):** SG15 is not benchmarked as a library. [read, acwf-verification-scripts
  `labels.json`; not re-checked]
- **MC3D paper:** "suboptimal precision for the Hg and Au pseudopotentials provided by the SSSP v1.3
  efficiency" [read, arXiv:2508.19223 TeX source, re-read]; both are SG15 1.0 files. [read; not re-checked]
- **Regenerating with ONCVPSP ≥ 3 changes the potentials and can give ghosts:**
  - Hamann: Fe was "on the 'hairy edge' of ghost behavior". [read, mail-archive QE-users msg39068, msg40121;
    not re-checked]
  - github.com/oncvpsp/oncvpsp issue #4, "Problems with Fe ONCV pseudopotential": the original 2.1.1 Fe file
    "works well", the same `.dat` run through ONCVPSP 3 does not. Issue #28, "SG-15 ONCV Psps generated w/
    oncvpsp-4.0.1 don't match V1.2 equivalents near the core" (Na). [read, re-read title and opening text; the
    Se report of the first session was not re-checked]
- **SPMS paper:** its table is "softer than the current PseudoDojo table of comparable accuracy and more accurate
  than the current SG15 table of comparable softness". [read, arXiv:2209.09806 TeX source, re-read]

## 3. Release history and licence

Dates are from `index.htm` [read, re-read]:

| Date | Change |
|---|---|
| 2015-01-24 | Version 1.0 |
| 2015-05-20 | Version 1.1 for Ar, As, Cl, Cs, I, In, P, S, Sb, Se, Si, Sn, Te, Xe |
| 2015-08-25 | Reference to the SG15 paper added to the files |
| 2015-10-07 | Hf, Ta, W 1.1 without 4f in valence; full paper reference in the files |
| 2018-06-25 | Version 1.2 in XML ("smooth local potential near r=0") |
| 2020-02-06 | Version 1.2 in UPF |

- **What 1.1 changes:** for the 14 elements of 2015-05-20, `rc`, `qcut`, `rc(5)` and `debl` all change (and `ep`
  for As, Cs, Se); the valence is the same. For Hf, Ta and W, 4f¹⁴ moves to the core (Z_val 26/27/28 →
  12/13/14) and all parameters change. No file or page states a reason for the 1.1 updates.
  [verified 2026-11-25, sg15_history_checks/check_inputs.py]
  - **Likely reason: ghost states** [inferred 2026-11-26 from the published paper]. The ghost-rejection step
    entered the paper in its revision of 22 April 2015, credited to Castelli and Marzari (section 1). Version
    1.1 came out on 20 May 2015, one week after acceptance, and its 14 elements include all eight for which the
    SSSP paper (Castelli and Marzari are co-authors) lists ghosts in 1.0. Nothing states the link outright.
  - **Against reading too much into it:** the paper's supplement, dated a week after the 1.1 release, still
    holds the 1.0 inputs for all 14 elements (section 1), and the published tables equal the arXiv v1 ones. So
    the paper describes the ghost-rejection step but documents and validates the set made before it.
- **What 1.2 changes:** each 1.2 UPF has the header and every block of an earlier file, except `PP_LOCAL` on the
  first mesh points. [verified 2026-11-25, sg15_history_checks/check_releases.py]
  - **Base version:** 1.1 for the 14 elements of 2015-05-20; 1.0 for the other 55, Hf, Ta and W included. (The
    first session gave no base version.)
  - **Points changed:** the first 10 (r ≤ 0.09 bohr) in 58 files; 15 for Cs, I, N, Os, Te; 20 for Br, Pb, Sb,
    Sn, Xe; 25 (r ≤ 0.24 bohr) for Bi. (The first session had 10 for all.)
  - **Size:** max |ΔV_loc| runs from 1e-6 Ry (As) to 35 Ry (Ge, Nb); In 21, S 16, Si 7.7, B 5.2, C 5.1. The
    large values are a jagged first few points in the older file.
  - There is no 1.2 `.dat`: it is post-processing. Issue #28 above is consistent with that.
- **Hf/Ta/W trap:** the 1.2 UPF derives from 1.0 (4f in valence, Z_val 26/27/28), but the 1.2 XML derives from
  1.1 (Z_val 12/13/14). [verified 2026-11-25, sg15_history_checks/check_releases.py]
- **Tarball contents:** UPF 2015-10-07 has 86 files; UPF 2020-02-06 has 219 (69 at 1.0, 17 at 1.1, 69 at 1.2, 64
  FR); XML 2018-06-25 has 155 (1.0, 1.1 and 1.2). The 86 files common to the two UPF tarballs have the same
  header and numbers; only two `PP_INFO` lines (the file name) were added.
  [verified 2026-11-25, sg15_history_checks/check_releases.py]
- **Orbitals:** every upstream UPF (all 86 and all 219) has `number_of_wfc="0"` and no `PP_CHI`.
  [verified 2026-11-25, sg15_history_checks/check_releases.py]
- **Atomic density is cut at 6 bohr:** `PP_RHOATOM` ends at `rlmax` (6.01 bohr), so 73 of the 86 files hold more
  than 0.01 e less than Z_val. Worst: Ba and Cs 0.39 e, Rb and La 0.30, Sr 0.28, K 0.26, Y and Ca 0.20; Na
  0.135 (issue #28 reports the same Na number). [verified 2026-11-25, sg15_history_checks/check_releases.py]
- **Input echo:** `PP_INPUTFILE` is not the true input. rc is shifted to a mesh node and ep/debl are overwritten;
  use the `.dat` files. [read, index.htm and ONCVPSP release notes, re-read]
- **Fully relativistic set:** 64 files (52 at 1.0, 12 at 1.1), by P. Scherpelz, made with ONCVPSP FR 2.1.1 "with
  the SG15 1.0 and/or 1.1 parameter sets"; the header says the set "does not employ the above self-consistent
  procedure at the fully-relativistic level". None for Ba, Be, Bi, Li, Ne. Cs, In, Hf, Ta and W have an FR file
  at 1.0 only (so Hf, Ta, W keep 4f in valence, and Cs and In are 1.0 files of SSSP's ghost list).
  [verified 2026-11-25, sg15_history_checks/check_releases.py; index.htm]
- **Licence:** CC BY-SA 4.0; the files carry "Copyright 2015 The Regents of the University of California".
  [read, re-read]
- **What is public:** the inputs, the 2.1.1 source, SG15's four source files, a Makefile, two awk scripts and a
  binary are in the full tarball. No optimiser is in it (no file mentions Nelder–Mead or a simplex).
  [verified 2026-11-25, sg15_history_checks/check_inputs.py] A `2.1.1` branch is on github.com/oncvpsp/oncvpsp.
  [read, re-read] A GitHub repository search for "sg15 oncv" finds one repository (a psp8 conversion), and a web
  search finds only third-party regenerations with ONCVPSP 3; the Nelder–Mead scripts were not found anywhere.
  [read, 2026-11-25]

## 4. Comparable ONCV libraries

Not re-checked in the second pass, except where marked.

- **ONCVPSP itself:**
  - UPF pseudo-wavefunctions were added in 3.2.1 "for the occupied states" (credit R. Sundararaman).
    [read, release notes; the quoted words re-read in `~/Desktop/jdftx_hold/oncvpsp/src/doc/release_notes.md`]
  - Reference occupancies "MUST be >0", so stock output has no empty-state orbitals.
    [read, `32_Ge_annotated.dat`]
- **PseudoDojo 0.4.1** (github.com/PseudoDojo/ONCVPSP-PBE-SR) [read]:
  - ONCVPSP 3.3.0; NLCC for most elements (a noCC variant repo exists); CC BY 4.0; 72 elements in the standard
    table.
  - `PP_CHI` covers occupied valence and semicore states only (Fe: 3S 3P 3D 4S; K: 3S 3P 4S).
  - Version 0.5 adds 11 potentials with an f projector. [read, ACWF SI]
- **SPMS** (github.com/SPARC-X/SPMS-psps) [read]: ONCVPSP 4.0.1; NLCC for most; GPL-3; 69 elements (H–La,
  Hf–Bi; element list re-read); `PP_CHI` for occupied valence states only.
- **SSSP 1.3 copies of SG15** [read; the local-file part was not re-run, no SSSP file is in the local folders]:
  regenerated with ONCVPSP 3.3.1, so they carry `PP_CHI` (Ag: 4S 4P 5S 4D) but are not the 2.1.1 potentials.
  Licence CC BY-SA 4.0.
- **pslibrary** [read, dalcorso.github.io/pslibrary]: US/PAW only, GPL, no norm-conserving files. Its ld1.x
  convention also writes bound empty states (K 4P, Cu 4P).

## 5. Not found or unverified

Still open after the second pass (2026-11-25):

- **The published CPC text: read on 2026-11-26** (it is open access; the earlier failure was blocked automated
  access, not a paywall). It has no Delta value, no statement on the relativistic treatment of the potentials,
  and no rule for combining q over cutoffs, so those three are not answerable from the paper. Its supplementary
  material was obtained and checked on 2026-11-26 (sections 1 and 2); it does not answer them either.
- **The Science Fig. 4 SG15 Delta numbers.** No text table found (one search; the DeltaCodesDFT page is rendered
  by script). The recomputation against WIEN2k in section 2 stands in for them.
- **The reason for the 1.1 updates.** Not stated on the index page, in `PP_INFO` or in the tarball's
  `util/description`. Ghost states are the likely reason (section 3, inferred from the published paper).
- **Whether SSSP's ghost list was measured on the original 2.1.1 files.** The SSSP paper does not say. By
  precision 1.1.2 SSSP's SG15 files were ONCVPSP 3 regenerations (section 2). The SG15 acknowledgment shows that
  two SSSP authors had reported ghost states to the SG15 authors by April 2015, when only the original 1.0 files
  existed, so the 1.0 ghosts were probably first seen in the original files [inferred]. Nobody has scanned the
  original 1.0 files for positive-energy ghosts here.
- **The Nelder–Mead scripts.** Not public as far as two searches show (section 3).
- **Third-party tests of magnetic systems or phonons with SG15 specifically.** None found (one search). What
  exists: the QuantumATK rock-salt/perovskite numbers and the SSSP convergence protocol (section 2).
- **The Scherpelz JCTC methods.** Paywalled; no arXiv version found by title.
- **The FLEUR inputs** the paper says are on the web page ("all input files can be obtained from our web page",
  also in the published text): not on the index page and not in the full tarball.
- **What changed between PseudoDojo 0.4 and 0.4.1.** The repository has no tags or changelog; its commits are
  2018-06-25 "Stable version of scalar-relativistic PBE-PD v0.4", then format additions (2021-12), Eu (2022-08)
  and an "s-sp" potential (2022-09).
- **Items of the first session not re-read:** the Science QE–CASTEP agreement, the QE mail-archive quotations,
  the ACWF `labels.json`, the SSSP changelog, "both are SG15 1.0 files" for SSSP-efficiency Hg and Au, and
  section 4.

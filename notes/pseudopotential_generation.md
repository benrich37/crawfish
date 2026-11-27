# Generating and testing a pseudopotential: instructions for Claude sessions

Written 2026-11-25 for the repair of JDFTx's SG15 norm-conserving set. Read sections 0 and 1 before doing
anything; the rest is reference for the step you are on. Sourced detail is in three research reports in
`pseudopotential_generation_refs/` (indexed in section 9).

Every statement here carries one of these tags where it matters:
**[run]** verified by running on 2026-11-25 · **[read]** read in source code or a primary document ·
**[lit]** taken from the literature reports in the refs folder · **[inferred]** a conclusion, not a measurement.

## 0. State of play

- **What is broken.** JDFTx's copies of SG15 carry pseudo-atomic orbitals (`PP_CHI`) that Sundararaman added in
  2015 with a private patched ONCVPSP 2.1.1. Upstream SG15 files have no orbitals at all. In 20 shipped files
  (19 at version 1.0, plus Sb 1.1) the orbitals are wrong. The potentials in JDFTx's files equal upstream's
  (≤ 8e-5 Ry) [run]. Details: memory `sg15-pslibrary-file-defects`, and
  `~/vs/research/crawfish/anl/pseudo_type_comparison/pseudo_type_comparison.md` section 5.
- **The orbitals-only repair is done and tested** (2026-11-25/26, section 4). The fixed files are in
  `~/vs/research/crawfish/anl/pseudo_type_comparison/sg15_orbitals/sg15_chi_fix/fixed_all_v2/` (`fixed_all/` is
  the first version, with 12 shorter grids). The test campaign is in `.../sg15_orbitals/route_a_tests/`. A JDFTx
  pull request is drafted but not sent; memory `sg15-pslibrary-file-defects` has its current state (section 7).
- **Nobody has shown that an SG15 potential must be regenerated.** Section 5.1 lists the known weaknesses that
  could justify it. A regenerated potential is a new pseudopotential and needs the whole test ladder of section 6.
- **No solid-state test of a regenerated potential has been run by us.** Sections 5 and 6 are built from the
  literature and from atom-level runs only.

## 1. Rules

1. **Pick the route first.** Route A = replace orbitals, keep `PP_LOCAL` / `PP_BETA` / `PP_DIJ` byte-identical.
   Route B = anything that changes the potential. Route A is the default. Take route B only for a measured defect
   of the potential, and say which measurement.
2. **A changed potential never keeps an existing file name.** New name, new `.pulay` file, full validation.
3. **Reproduce before you modify.** Regenerate the shipped file first (ONCVPSP 2.1.1 + SG15's `.dat`), confirm
   it matches, then change one thing at a time.
4. **The ONCVPSP version and the build are part of the pseudopotential.** The same input gives a different
   potential in a different major version [run], and 2.1.1 itself gives different potentials on another machine
   (section 5.2). Record version, commit, compiler, flags and math library with every generated file.
5. **Use SG15's real `.dat` inputs, never the `PP_INPUTFILE` echo** (section 5.2).
6. **A norm of 1 does not prove an orbital is right.** 13 of the bad files had normalised, wrong orbitals. Check
   the density against `PP_RHOATOM` (section 4).
7. **Check for an existing validated potential before generating one** (section 5.1, PseudoDojo and SPMS).
8. **Reuse the existing code** (section 2). Do not write a second UPF reader, atom solver or Δ calculator.
9. **Cheap tests first.** Tiers 0 and 1 of section 6 take seconds on the Mac. Tiers 2 and up are cluster
   campaigns: propose the plan and its cost to the user before launching.
10. **Nothing leaves the machine without the user.** No JDFTx pull request, no Zenodo deposit, no push. SG15 is
    CC BY-SA 4.0, so every derived file carries that licence, the attribution and a statement of changes.
11. **Tag what you report** as run, read or inferred, as this file does.

## 2. Where things are

| What | Where |
|---|---|
| Fixed orbital files, solver, validation, README | `~/vs/research/crawfish/anl/pseudo_type_comparison/sg15_orbitals/sg15_chi_fix/` |
| Origin of the defect: tests, instrumented and fixed ONCVPSP 2.1.1 patches | `.../sg15_orbitals/origin_of_defect/smal/` (README, `patches/`), `.../sg15_orbitals/rhoatom_crosscheck.py` |
| Upstream SG15 tarballs, provenance check, distribution plan | `.../pseudo_type_comparison/sg15_distribution/` (`upstream/`, `sg15_provenance.py`, `sg15_distribution.md`) |
| Orbital norm scan of any UPF folder | `.../pseudo_type_comparison/pao_scan.py` |
| Δ recomputation and WIEN2k reference data | `.../pseudo_type_comparison/delta_history/` (`delta_hist.py`, `WIEN2k.txt`) |
| Checks of SG15's inputs, releases and Δ numbers (README inside) | `.../pseudo_type_comparison/sg15_history_checks/` |
| ONCVPSP sources, builds, regeneration check | `~/Desktop/jdftx_hold/oncvpsp/` (README inside) |
| Full SG15 distribution unpacked (`.dat`, `.out`, patched sources) | `~/Desktop/jdftx_hold/oncvpsp/smoke_test/sg15_full/` |
| PseudoDojo NC SR 0.5 PBE standard set | `~/Desktop/jdftx_hold/pseudopotentials/nc-sr-05_pbe_standard/` |
| JDFTx source (reader, `calcPulay`, `SG15.tgz`, `SG15-pulay.tgz`) | `~/Desktop/jdftx_hold/jdftx_master/jdftx/` |
| JDFTx binary and installed SG15 files | `~/Desktop/jdftx_hold/build_master/` (memory `jdftx-local-runs`) |
| Quantum ESPRESSO 7.6 (`pw.x`, `projwfc.x`, `ld1.x`) | conda env `qe` (memory `qe-local-install`) |

Code to reuse (read each script's docstring or the folder README before calling it):

- `sg15_chi_fix/scripts/upf.py`: UPF v2 reader that replaces blocks byte-exactly. `atom.py`: self-consistent
  pseudo-atom solver for a UPF file's own potentials. `check_rhoatom.py`: Σ f·χ² against `PP_RHOATOM`.
  `write_fixed.py`: writes and self-checks the fixed files. `make_ab.py` / `analyze_ab.py`: JDFTx A/B runs.
- `~/Desktop/jdftx_hold/oncvpsp/smoke_test/regen_check.py`: runs an input through each ONCVPSP build and
  compares the result with a reference UPF (`summarize.py` tabulates a full run).
- `delta_hist.py`: the Δ formulas of calcDelta 3.0.

## 3. What JDFTx does with a UPF file

Read in `electronic/SpeciesInfo_readUPF.cpp`, `SpeciesInfo.cpp` (`setPsi`, `setupPulay`) and
`SpeciesInfo_atomic.cpp` at JDFTx `e305ab5d` [read].

- Needs UPF ≥ 2.0.1. `PP_HEADER` must have `element`, `functional`, `is_paw`, `z_valence`, `l_max`, `mesh_size`,
  `number_of_proj`, `number_of_wfc`.
- **Ignores `PP_INFO` and `PP_RHOATOM`.** A wrong `PP_RHOATOM` changes nothing in JDFTx (it does in QE).
- Every block is read with `mesh_size` points. A longer orbital tail means extending every block of the file.
- Units are converted from Rydberg: `PP_LOCAL` → 0.5·V + Z/r; `PP_BETA.i` ÷ r (needs `angular_momentum`);
  `PP_DIJ` × 0.5; `PP_NLCC` as is.
- `PP_CHI.i` needs `l`, `occupation` and `label`; `occupation` is only printed. Optional `pseudo_energy` × 0.5.
  - If every orbital has a nonzero `pseudo_energy`, those are the atomic eigenvalues. Otherwise they are
    estimated from the orbital tails, and an orbital judged unbound is never occupied.
  - The LCAO starting atom is filled aufbau by these eigenvalues, not by the file's occupations.
  - `setPsi` warns "large normalization error in atomic orbital" when |∫χ² − 1| > 1e-2, and rescales any orbital
    with ∫χ² > 1e-10 to norm 1. Garbage therefore becomes a full basis function.
- Orbitals affect the LCAO guess, `bandProjections` (so everything crawfish computes) and DFT+U. They do not
  enter the converged energy, but a bad guess can change which state the SCF finds [run: Ni converged to M = 0
  and Cs never converged with the shipped files; `sg15_chi_fix/README.md`].
- **Pulay file.** `<basename>.pulay` is used when readable; if missing, the correction is silently off. Format:
  a count line, then `Ecut dE_dnG` lines. Ecut above the table gives 0; Ecut below it stops the run. The shipped
  SG15 tables run from 5 to 50 Ha in steps of 5.
- **Default symlinks.** `pseudopotentials/update.sh` points `<El>_ONCV_PBE.upf` and `.pulay` at 1.1 when it
  exists, otherwise 1.0. JDFTx ships 1.1 for Ar As Cl Cs I In P S Sb Se Si Sn Te Xe. A file with a new name gets
  neither a symlink nor a `.pulay` unless you make them.
- Only collinear spin matters here: crawfish refuses spinor runs, so fully relativistic potentials are out of scope.

## 4. Route A: orbitals only

**Done for all 20 files.** Method and results are in `sg15_chi_fix/README.md`; do not redo it. For new orbital
work (another file, another library, extra orbitals), follow the same steps:

1. Solve the self-consistent pseudo-atom with the file's own `PP_LOCAL`, `PP_BETA`, `PP_DIJ` in the reference
   configuration (`scripts/atom.py`).
2. Validate the solver on files known to be good before trusting it on bad ones.
3. Accept an orbital only if all of these hold. The numbers are what the method reached on the 50 good SG15 1.0
   files [run, SMAL session]:
   - norm 1, and the node count expected for that state of the pseudo-atom;
   - eigenvalue within 2e-4 Ha of `pseudo_energy` (1e-4 for all but three orbitals; `pseudo_energy` is the
     all-electron eigenvalue, so a small offset is the potential's, not the solver's);
   - overlap with the shipped orbital ≥ 0.99999996 where the shipped one is good;
   - density Σ f·χ² equal to `PP_RHOATOM` to about 3e-5 e (the shipped good files agree to 5e-6 e). **This is the
     test that catches normalised but wrong orbitals**, and it does not depend on the solver;
   - SCF total energy within 2e-5 Ha of the header's `total_psenergy`;
   - the tail has decayed inside the grid; extend the mesh (all blocks) if not. In `fixed_all`, 14 orbitals in
     12 files still lost up to 2.5e-4 of their norm past the grid end (clean files: ≤ 1.1e-6), so `fixed_all_v2`
     extends those grids to 30 bohr [run, other session; `route_a_tests/README.md`].
4. Replace the `PP_CHI` blocks and nothing else (`scripts/write_fixed.py`), and confirm the other blocks are
   byte-identical.
5. JDFTx A/B with the original file (`scripts/make_ab.py`): no normalisation warning; converged energy unchanged
   (reached: ≤ 3.3e-6 Ha/atom, largest for the extended meshes); first LCAO energy close to the final one;
   spilling down.

Facts worth knowing:

- **Why they were bad** [run, SMAL session; `origin_of_defect/smal/README.md`]: ONCVPSP 2.1.1 prints
  `psatom: lschvkbb convergence error` and `run_config: ERROR for fully non-local PS atom` for exactly 21 of the
  86 SG15 inputs: the 19 bad 1.0 files, Sb 1.1 and Hf 1.1 (not shipped by JDFTx). JDFTx's patch wrote whatever
  array the pseudo-atom self-consistency test (`psatom`) left for each state, without checking the error flag.
  The garbage orbitals equal the unnormalised output of the failed call (18 of 20 exactly). The normalised wrong
  ones are that state's last successful iterate (31 of 31). The patch itself is not public.
- **Root cause** [run, same source]: `psatom` starts from the all-electron valence density, so the lower energy
  bound it gives the solver is 10 to 130 Ha too deep. With a nonlocal potential the node count is not monotonic
  in energy, and the bisection walks down to that bound. `patches/22-fix-both.diff` (start from the pseudo
  density, step down gradually) makes `psatom` converge for all 21 with bit-identical potentials. Its orbitals
  equal `fixed_all` (overlap 1.000000000, eigenvalues within 2.1e-6 Ha). So the fixed orbitals have two
  independent sources: the linear-grid solver of `sg15_chi_fix` and ONCVPSP's own solver.
- **Detector for any ONCVPSP ≤ 3.3.1 output:** grep the `.out` for `lschvkbb convergence error` or `run_config:
  ERROR` (a WARNING in 3.3.1). If present, do not trust that file's `PP_CHI`. ONCVPSP 3.3.1 has the same
  unguarded copy [read, `src/psatom.f90` line 153] but does not fail for these 21 inputs [run, same source].
- ONCVPSP 4.x takes the orbitals from the main valence loop and stops on error; all 263 orbitals it wrote for
  the SG15 inputs have norm 1 [run]. Their density was not checked against `PP_RHOATOM`.
- Stock ONCVPSP writes orbitals for occupied reference states only (occupations must be > 0) [lit]. So Be has
  only s orbitals and its spilling stays near 0.18 [run]. Orbitals for empty states are not part of any shipped
  set; adding them is a new feature to raise with the user, not a repair.

## 5. Route B: regenerating a potential

### 5.1 Decide whether it is warranted

Known weaknesses of SG15 that could justify it:

- **No nonlinear core correction** in any file (`icmod=0`) [lit]. The Δ outliers are the magnetic 3d metals:
  Cr 21, Mn 13, Fe about 5 meV/atom, against 1.35 for the library [lit, recomputed]. That the missing core
  correction causes them is plausible and not shown.
- **Never trained or tested on a magnetic state.** The SG15 paper mentions neither spin polarisation nor
  magnetism; its 602 materials are judged on lattice constants only [read, paper]. The Δ test runs Cr, Mn, Fe,
  Co and Ni in their magnetic states. In the paper's non-magnetic structures these elements are fine: the
  largest lattice-constant error over bcc, fcc, sc and diamond is Fe 0.06 %, Mn 0.12 %, Cr 0.18 %
  [run, `sg15_history_checks/check_si.py`]. So the Δ outliers arise in the magnetic states. This is the best
  current candidate for a potential worth regenerating, and the first thing to test is a core correction.
- **Ghost states** within 10 eV above the Fermi level in the 1.0 files of Ar, Cs, In, S, Se, Sn, Te, Sb
  [lit, SSSP]. JDFTx defaults to 1.1 for all eight. Our own solver found no bound ghosts in the 69 shipped files
  [run]; positive-energy ghosts were not scanned.
  - SG15's published procedure rejects any candidate whose log derivative shows a ghost near the Fermi energy.
    That step was added in the paper's revision of April 2015, and version 1.1 followed a month later, so 1.1 is
    probably the ghost fix [inferred from read, paper; `sg15_history.md` section 3].
  - **The paper validates 1.0 only.** Its supplement holds the 1.0 inputs for all 69 elements, number for number
    [run, `check_si.py`]. The 1.1 files, which JDFTx uses by default for 14 elements, have no published
    lattice-constant validation by the authors; only the Δ test covers them.
  - bcc Cs with the 1.0 potential has a spurious lower state that 1.1 and 1.2 do not have [run, other session;
    `sg15_orbitals/cs_ghost_check/`]. Prefer the 1.1 file wherever one exists.
- SSSP precision 1.2/1.3 dropped SG15 for the noble gases and calls its Hg potential less precise. Those SSSP
  files are ONCVPSP 3 regenerations, not the 2.1.1 potentials. The SG15 paper's own outliers are C, Ca, Cd, La,
  B, Cl, Sc, Ni, Rb, Y, BeO [lit]. Every material's lattice constant and bulk modulus, and each element's error
  in its elemental structures, are in `sg15_history_checks/check_si.txt` (from the paper's supplement; worst:
  Sc diamond −0.43 %, C bcc +0.34 %, Y diamond +0.29 %).
- A third-party GBRV-type test (QuantumATK) finds SG15 less accurate than PseudoDojo: RMS error 0.16 % against
  0.09 % on rock-salt compounds and 0.13 % against 0.06 % on perovskites [lit]. No magnetic or phonon benchmark
  of SG15 was found.
- Li to Ar have no d projector (`lmax=1`) [lit].
- The set is PBE. The user also runs it with r2SCAN; potentials from a lower-rung functional can add noticeable
  error for meta-GGAs [lit, testing report section 7].

Reasons not to:

- Every existing SG15 result, including the user's 30 Ha production data, stops being comparable.
- SG15's quality came from a Nelder–Mead search over `rc`, `qcut`, `rc(5)`, `debl` and `ep`. Each candidate was
  scored on the lattice constants of 301 training materials against FLAPW (target 0.2 %) and on cutoff, then
  checked on 301 test materials [read, paper]. Those scripts are not public. A hand-edited input loses that
  optimisation unless the tests show otherwise.
- **An existing library may already solve it.** PseudoDojo 0.4.1/0.5 (mostly ONCVPSP 3.3.0, core-corrected, CC BY 4.0;
  all orbitals of the 72 local files have norm 1 [run]) and SPMS (ONCVPSP 4.0.1, GPL-3) are validated. They are
  harder than SG15, so compare at converged cutoffs.

The three ways to regenerate, in order of preference:

1. **2.1.1 with one parameter changed** from SG15's `.dat`. The baseline is reproducible, so the effect of the
   change is isolated.
2. **Another library's potential for that element.** Nothing to generate; still run tiers 0 to 2 in JDFTx.
3. **SG15 inputs in ONCVPSP ≥ 3.** This is not SG15: the nonlocal kernel differs by about 50 % (median) [run].
   Master writes no file for 2 of 86 inputs and its ghost detector flags 40 [run]. ONCVPSP 3.3.1 differs too,
   by 1e-3 to 0.55 Ry in V_loc for the 21 inputs tried [run, SMAL session]. Treat the inputs as a starting
   point only.

### 5.2 Running ONCVPSP

Builds are in `~/Desktop/jdftx_hold/oncvpsp/`: `build_2.1.1/bin/` and `build_master/bin/` (4.0.1 + 95 commits,
`dc150ab`), each with `oncvpsp.x` (scalar-relativistic), `oncvpspnr.x`, `oncvpspr.x`. They were built with CMake
and Homebrew gfortran, without libxc, so only the built-in functionals work (`iexc` 1 to 4; 4 is PBE).

```sh
cd ~/Desktop/jdftx_hold/oncvpsp/smoke_test
../build_2.1.1/bin/oncvpsp.x < in.dat > out.out            # stdin must be a file
python regen_check.py sg15_full/elements Sb_ONCV_PBE-1.1   # run both builds, compare with upstream/<stem>.upf
```

Gotchas, all met in practice [run]:

- **Stdin must be a file.** The code backspaces on unit 5; a pipe gives `Fortran runtime error: Illegal seek`.
- **Use the `.dat` from `sg15_full/elements/NN_El/`.** Change `upf+fpmd` to `upf` on the first data line; only
  SG15's own build knows `upf+fpmd`.
- **The `PP_INPUTFILE` echo is not the input.** 2.1.1 snaps each `rc` up to a log-mesh point (mesh ratio 1.012)
  and echoes the snapped value, rounded; `ep` and `debl` are overwritten too [lit]. Re-reading the echo can land one
  mesh point further, and V_loc then differs by up to 0.7 Ry. If only the echo exists, subtract 1e-4 bohr from
  every `rc` and `rc(5)`: the result agrees to 3e-5 Ry instead of 2e-8.
- **The UPF sits inside the `.out`.** It starts after `Begin PSP_UPF` and ends at `</UPF>`. 3.x and 4.x add an
  `END_PSP` line; 2.1.1 does not.
- **Compare potentials gauge-invariantly.** Projectors can differ by a sign or a rotation. Compare V_loc at r > 0
  and, per l, the kernel Σ_ij β_i(r) D_ij β_j(r′) (`kernels` in `regen_check.py`).
- **Reproduction baseline:** 2.1.1 with the `.dat` files reproduces all 86 upstream files: V_loc ≤ 2.3e-8 Ry,
  β ≤ 2.5e-7, D_ij ≤ 6.1e-5 Ry, kernel ≤ 2.0e-7 relative. A new run that misses these numbers has a setup error.
- **That baseline holds on this Mac build only** (arm64, Homebrew gfortran 16.2, `-O3`, Apple Accelerate), which
  also matches the 2015 logs. On SMAL (x86_64, gfortran 15.2, OpenBLAS or netlib, with and without optimisation)
  the same source and inputs reproduce only Pt, Au, Hg and Sb 1.1 of 24 files tried. The others differ from the
  projector optimisation onward, by up to 0.44 Ry in V_loc (Ge) [run, SMAL session; memory
  `pseudopotential-generation-notes`]. The cause is not known. Generate on the Mac build, and rerun the baseline
  on any other build before trusting it.
- ONCVPSP 3.3.1 needs two commas added to `upfout` format strings before gfortran 15 compiles it
  (`origin_of_defect/smal/patches/41`).
- Version 1.2 has no input. Each 1.2 UPF is an earlier file with `PP_LOCAL` smoothed on the first 10 to 25 mesh
  points (r ≤ 0.24 bohr) and nothing else changed. The earlier file is 1.1 for the 14 elements that have one
  from 2015-05-20, and 1.0 for the other 55 [run, `sg15_history_checks/check_releases.py`]. For Hf, Ta and W the
  1.2 UPF comes from 1.0 (4f in valence) but the 1.2 XML from 1.1.
- Upstream files end at `rlmax` = 6 bohr, so their `PP_RHOATOM` is cut there and misses up to 0.39 e (Ba, Cs)
  [run, same script]. Compare a density over the file's own grid, not against Z_val. JDFTx ignores the block.
- Version differences that matter: 2.1.1 has no orbital output and no ghost detector; 3.2.1 adds orbitals;
  3.3.0 adds the bound-ghost detector and 3.3.1 the positive-energy one; 4.x changes how negative
  second-projector energies and localised well states are handled (26 SG15 inputs are affected) [lit].

### 5.3 The input, and the order to tune it

Layout (full description: `oncvpsp_reference.md` section 3, and `src/doc/32_Ge_annotated.dat`):

```
atsym z nc nv iexc psfile        n l f  (nc+nv lines)       lmax
l rc ep ncon nbas qcut  (per l)  lloc lpopt rc(5) dvloc0    l nproj debl  (per l)
icmod fcfact (rcfact)            epsh1 epsh2 depsh          rlmax drl
ncnf, then the test configurations
```

SG15 fixes `ncon=5`, `nbas=8`, `nproj=2`, `lloc=4`, `lpopt=5`, `dvloc0=0`, `icmod=0`, `rlmax=6.0`, `drl=0.01`
and varies only `rc`, `qcut`, `rc(5)`, `debl`, `ep` [lit].

Order of adjustment [lit, assembled from Hamann's guide and paper; he gives no numbered recipe]:

1. Reference configuration: the neutral ground state. Keep SG15's valence and semicore choice unless that is the
   thing being fixed.
2. `rc` per l, above the printed outermost-peak radius. Smaller is more transferable and harder.
3. `qcut` per l from the "Energy error per electron / Cutoff" table, balanced across l. "First pseudo wave
   function has node" means raise `qcut` a little.
4. `ep` (0.1 to 0.25 Ha for an empty l) and `debl` (about 1 Ha) so the well states converge. Keep `ep + debl`
   positive.
5. Ghosts: fix through the local potential (`lpopt`, `rc(5)`, `dvloc0`), moving it toward the semilocal
   potential of the ghost's l. More projectors invite ghosts.
6. Model core charge last (`icmod` 1 to 4). It often hurts phonon convergence and the acoustic sum rule, so test
   those when it is on.

### 5.4 What to check in the output

Tier 1 of section 6. Strings and meanings: `oncvpsp_reference.md` section 4.

## 6. Test ladder

Each tier must pass before the next is worth its cost. Thresholds come from the protocols in
`testing_protocols.md`; **the acceptance bar itself is the user's decision**. A sensible default: the new file
must not do worse than the SG15 file it replaces in the same test, and must beat it in the test that motivated
the change. Run the shipped file through the same tests as the control.

**Tier 0: the file** (seconds)
- JDFTx reads it and its `out` shows the right valence charge, projector count and orbitals, with no
  normalisation warning.
- Orbital checks of section 4 step 3; `pao_scan.py` for norms.
- Header and blocks consistent: `mesh_size`, `number_of_proj`, `number_of_wfc`, Ry units, V_loc·r → −2·Z_val (Ry)
  at the last point.

**Tier 1: the atom** (seconds; from the ONCVPSP `.out`)
- All-electron reference converged. No `ERROR`, no `run_config: ERROR`, no `lschvkbb convergence error`.
- Constraint consistency about 1e-13; kinetic and potential consistency ratios close to 1; B-matrix
  Hermiticity error 1e-4 to 1e-5 (scalar-relativistic).
- "Diagnostic tests": `e in` equals `e test`; norm and slope ratios close to 1.
- No `WARNING - GHOST(-)`. Look at the log-derivative plot around every `GHOST(+)`: the detector is "beta",
  version-dependent, and can flag a real resonance, mostly for l = 3.
- Log derivatives (arctan plots) of pseudo and all-electron agree from about 1 Ha below to 1 Ha above the
  valence eigenvalues, and further up if conduction states matter. A step in the pseudo curve with no
  all-electron partner is a ghost. Set the scanned range with `epsh1 epsh2 depsh`.
- Test configurations (`ncnf` up to 4; ionised and promoted states): excitation errors of about 1 mRy mark a good
  potential. Hamann found little correlation with solids, so this test cannot pass a potential, only fail it.
- Solve the written file's pseudo-atom independently (`sg15_chi_fix/scripts/atom.py`): eigenvalues, density and
  bound-ghost check as in section 4.

**Tier 2: cutoff convergence in JDFTx** (minutes to hours per element)
- Elemental crystal and one compound; total energy, forces, stress and, where it matters, a phonon, each against
  its value at a 100 Ha reference. Convergence of one quantity does not imply the others.
- SSSP thresholds for the recommended cutoff: cohesive energy within 2 meV/atom; pressure error equivalent to
  < 1 % in volume (0.5 % for precision); phonon frequencies within 2 % (1 %); band distance η10 < 10 meV,
  max 20 meV. No protocol gives a force threshold.
- The SG15 paper reports good agreement at about 60 Ry (30 Ha). That is an outcome, not a design target: 4.65 %
  of its lattice constants are still off by more than 0.2 % there [lit]. Norm-conserving density cutoff is 4×
  the wavefunction cutoff.
- Make the `.pulay` file here (section 7).

**Tier 3: equation of state against all-electron**
- Δ (Lejaeghere): elemental crystals, Birch–Murnaghan fit, against WIEN2k (`delta_history/`). SG15 as a library
  is at 1.35 meV/atom; SSSP asks < 1 "if possible".
- ACWF (Bosoni 2024): 4 unaries and 6 oxides per element, 7 volumes from 94 to 106 %, Fermi–Dirac 0.0045 Ry,
  k-spacing 0.06 Å⁻¹. Metrics ε and ν: excellent ε ≤ 0.06, ν ≤ 0.10; good ε ≤ 0.20, ν ≤ 0.33. The oxides probe
  short bonds and a second chemical environment, which Δ does not. Reference data are in the
  `acwf-verification-scripts` repository (not downloaded here).

**Tier 4: compounds, magnetism, molecules**
- GBRV-style lattice constants (fcc, bcc, rocksalt, perovskite, half-Heusler) within 0.2 % of all-electron, which
  is also SG15's own design target.
- Magnetic moments and magnetic ordering energies for 3d elements, with and without a core correction if one was
  added.
- Bond lengths of small molecules (H2, N2 were in SG15's training set).

**Tier 5: phonons, bands, high-energy ghosts**
- Γ phonons with the acoustic sum rule; a zone-boundary phonon against cutoff.
- Band distance to a trusted potential. A ghost in the solid shows as a band distance of order eV.
- Ghost scan to about 200 eV above the Fermi level if the potential may be used for GW or optics.

**Tier 6: what crawfish needs**
- Spilling of `bandProjections` in a few solids (reached with the fixed SG15 orbitals: 0 to 0.055).
- pDOS and pCOHP of a reference system against the shipped file and against LOBSTER or PseudoDojo results where
  those exist (`examples/benchmarking/`).

Cross-code check: run the same UPF in QE (env `qe`) for at least one tier 2 or 3 system. `ld1.x` cannot read
SG15 files (linear mesh), and QE cannot read the Hf, Ta and W files at all: their mesh is longer than QE's
limit of 3500 points [run, other session; `route_a_tests/README.md`, test R6].

The SG15 authors' own validation (of the 1.0 set) is the yardstick for a replacement [read, paper]: lattice constants within
0.2 % of FLAPW for 95 % of 602 materials at 60 Ry (rms 0.08 % converged), bulk moduli rms 2.4 %, bond lengths
of six diatomic molecules within 0.35 %, six perovskites and six half-Heuslers within 0.2 %.

## 7. Packaging and release

- **Name.** Route A files are `<El>_ONCV_PBE-<ver>-chifix-all.upf`. A regenerated potential needs a name that
  cannot be taken for an upstream SG15 version; agree it with the user.
- **`PP_INFO`.** State what changed, the date, the tool with version and commit, the original file, the licence.
  For a regenerated file put the true `.dat` in `PP_INPUTFILE`, not the echo.
- **`.pulay`.** Route A: reuse the original's file under the new basename. That was tested (R1 in
  `route_a_tests/`: agreement to 1.7e-7 for Au, Re, Cs). The same campaign made new Hg and Ne tables and found
  the shipped Fe table gives a large spurious pressure at 40 Ha, so read its README before relying on a shipped
  table [run, other session]. Route B: `scripts/calcPulay <psp> [Estart:dE:Estop] [out]`
  (defaults 5:1:50 Ha, box 20 bohr, PBE), covering at least 5 to 50 Ha. The script uses GNU `expr`; it has not
  been tried on macOS, so run it on Linux.
- **Symlinks and tarballs.** `update.sh` and `SG15.tgz` / `SG15-pulay.tgz` in `jdftx/pseudopotentials/`.
- **Licence.** CC BY-SA 4.0: credit Schlipf and Gygi, Comput. Phys. Commun. 196, 36 (2015) and Hamann, Phys. Rev.
  B 88, 085117 (2013); say what was changed; release under the same licence.
- **Channel.** The plan (a JDFTx pull request to `SG15.tgz` plus a Zenodo deposit) is in `sg15_distribution.md`.
  The user has not decided. Do not act on it unasked.
- **Keep with every generated file:** the `.dat`, the full `.out`, the ONCVPSP version, commit and compiler, the
  comparison against the file it replaces, and each tier's results with the control.

## 8. Not verified

- The text of JDFTx's 2015 ONCVPSP patch. Its behaviour is reproduced, except for 21 orbital slots of Au, Hg, Re,
  Hf, Ta and W that match nothing (probably uninitialised memory).
- Why ONCVPSP 2.1.1 gives different potentials on SMAL than on the Mac.
- Whether SG15 1.0 potentials have positive-energy ghosts beyond the eight SSSP lists, and whether SSSP measured
  the original 2.1.1 files or regenerated copies. The SSSP paper does not say; SSSP's SG15 files were ONCVPSP 3
  regenerations by precision 1.1.2. Two SSSP authors had reported ghosts to the SG15 authors by April 2015, when
  only the original files existed, so the originals are the likely source [inferred].
- Why the SG15 authors issued the 1.1 updates: ghost states are the likely reason, but nothing states it. Their
  optimisation scripts are not public.
- Anything in tiers 2 to 6 for a regenerated potential: none has been run.
- `calcPulay` on macOS.
- Density consistency of the orbitals ONCVPSP master writes.
- Three things the SG15 paper does not state (it was read in full on 2026-11-26): a Δ value, the relativistic
  treatment of the potentials, and how the quality function is combined over cutoffs. Its supplementary
  material (checked the same day) does not state them either.
- Any solid-state validation of the 1.1 files beyond the Δ test.
- Each refs report ends with its own "could not verify" list. Read it before relying on a number from that report.

## 9. Reference index

| For | Read |
|---|---|
| ONCVPSP versions, build, input variables, output strings, UPF and psp8 writers, tuning guidance | `pseudopotential_generation_refs/oncvpsp_reference.md` |
| Δ, ACWF, GBRV, PseudoDojo and SSSP protocols with formulas and thresholds; atom-level tests; orbital requirements | `pseudopotential_generation_refs/testing_protocols.md` |
| How SG15 was made and validated, its releases, comparable libraries | `pseudopotential_generation_refs/sg15_history.md` |
| The orbital repair and its validation | `~/vs/research/crawfish/anl/pseudo_type_comparison/sg15_orbitals/sg15_chi_fix/README.md` |
| Hamann's own documentation and papers | `~/Desktop/jdftx_hold/oncvpsp/src/doc/` |
| The SG15 paper (Schlipf and Gygi 2015) and other reference papers | `~/Desktop/Crawfish/` (`schlipf2015.pdf`) |
| Tests of the fixed orbital files (R0 to R6) | `~/vs/research/crawfish/anl/pseudo_type_comparison/sg15_orbitals/route_a_tests/README.md` |

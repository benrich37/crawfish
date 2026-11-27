# ONCVPSP technical reference (for repairing JDFTx's SG15 set)

**Headline finding.** JDFTx's SG15 orbitals were added by a patched ONCVPSP 2.1.1 whose `PP_CHI` data comes from the pseudo-atom self-consistency test ("Test configuration 0"). That test fails in 2.1.1 for 21 SG15 inputs: 19 elements at 1.0, plus Sb-1.1 and Hf-1.1. The failing state is exactly the one with a garbage orbital in JDFTx's shipped files. Official 4.0.1 takes the orbitals from a different, robust place (Q5).

> **What this file is.** A research report written by a Claude subagent on 2026-11-25 from the sources keyed below.
> It read sources and ran nothing. The instructions that use it are in `../pseudopotential_generation.md`; where
> the two disagree, that file is newer. The sources it downloaded now live in `~/Desktop/jdftx_hold/oncvpsp/`
> (`src/` = official git clone, `tarballs/` = 3.3.1 and 4.0.1, `smoke_test/sg15_full/` = the SG15 2015-10-07 tarball).
>
> **Settled after it was written** (runs of 2026-11-25, `~/Desktop/jdftx_hold/oncvpsp/smoke_test/`), which closes
> four items of "Could not verify" at the end:
> - gfortran builds of 2.1.1 and master work on macOS arm64 with CMake;
> - 2.1.1 plus SG15's `.dat` inputs reproduces all 86 shipped UPF files (V_loc to 2e-8 Ry);
> - the 21 inputs with the `run_config` error map one-to-one onto the files with bad orbitals (19 at 1.0, Sb 1.1,
>   and Hf 1.1, which JDFTx does not ship);
> - master ran on all 86 inputs: 84 give a UPF, and none of them is the SG15 potential.

## Source keys

- **[GH]** https://github.com/oncvpsp/oncvpsp — raw files at `https://raw.githubusercontent.com/oncvpsp/oncvpsp/master/<path>`; tag `4.0.1` and branch `2.1.1` at `.../oncvpsp/4.0.1/<path>` and `.../oncvpsp/2.1.1/<path>`
- **[T401]** http://www.mat-simresearch.com/oncvpsp-4.0.1.tar.gz
- **[T331]** http://www.mat-simresearch.com/oncvpsp-3.3.1.tar.gz
- **[MS]** http://www.mat-simresearch.com/ (http only; https is refused)
- **[RN]** `doc/release_notes.md` in [GH] (same text as `doc/release_notes.txt` in [T401])
- **[UG]** `doc/users_guide.md`; **[ANN]** `doc/32_Ge_annotated.dat`; **[PO]** `doc/program_overview.md`; **[GHO]** `doc/ghosts.md`; **[CC]** `doc/core_correction.md`; **[REL]** `doc/relativistic.md`; **[LXC]** `doc/libxc_use.md`; **[PWX]** `doc/pwscf_exc.md` — all in [GH]
- **[PRB]** `doc/PhysRevB.88.085117.pdf` in [GH]; **[ERR]** `doc/Erratum_PRB.95.239906.pdf` in [GH] (DOI 10.1103/PhysRevB.95.239906)
- **[SG]** http://www.quantum-simulation.org/potentials/sg15_oncv/index.htm
- **[SGT]** http://www.quantum-simulation.org/potentials/sg15_oncv/sg15_oncv_2015-10-07.tar.gz
- **[JC]** https://github.com/shankar1729/jdftx/commit/6cd5db7d7f9c047e2f9a5350881817ce37c54a35
- **[JCL]** https://github.com/shankar1729/jdftx/blob/master/jdftx/doc/ChangeLog.dox
- **[I#n]** `https://github.com/oncvpsp/oncvpsp/issues/<n>` (PRs: `/pull/<n>`)
- **[PIPI]** https://github.com/pipidog/ONCVPSP
- **[QE]** https://raw.githubusercontent.com/QEF/q-e/develop/atomic/src/import_upf.f90
- **[UPF]** https://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format
- **[PSP8]** https://docs.abinit.org/developers/psp8_info/
- **[ABI]** https://abinit.github.io/abipy/scripts/oncv.html
- **[LOCAL]** your own `~/vs/jdftx/jdftx/pseudopotentials/SG15.tgz`, read only

Statements marked "(source reading)" are my reading of the cited Fortran file, not a documented statement.

## 1. Distribution

- **Canonical source.** github.com/oncvpsp/oncvpsp, described as "The official repository of the oncvpsp code". It was created 2019-05-23 with Hamann's commit "Import version 4.0.1". Docs are at https://oncvpsp.github.io/oncvpsp/. [GH]
- **Licence.** GPL-3.0 (`COPYING`, and the header of every source file). [GH]
- **Tags and branches.** There are no GitHub releases. The only tag is `4.0.1`; branches are `master`, `2.1.1` and `gmatteo`. [GH API]
- **Hamann's site.** It still serves `oncvpsp-3.3.1.tar.gz`, `oncvpsp-4.0.1.tar.gz` and `metapsp-1.0.1.tar.gz` ("alpha release"). [MS] The 2.1.1, 2.1.2, 3.0.0, 3.1.1, 3.2.1, 3.2.2, 3.2.3 and 3.3.0 tarballs return 404 there (checked today).
- **Dates from version strings in the sources.**

| Version | Date | Source |
|---|---|---|
| 1.1.1 | 08/28/2013 | stale string in 2.1.1's `upfout.f90` |
| 2.1.1 | 03/26/2014 | [SGT] `3rd_party/oncv_src/oncvpsp.f90` |
| 3.3.0 | about June 2017 | [ERR], received 5 June 2017, cites "Release 3.3.0" |
| 3.3.1 | 12/12/2017 | [T331] |
| 4.0.1 | 06/04/2019 | [T401] |

- **3.2.1, 3.2.2, 3.2.3 dates.** UNVERIFIED. File times inside [T331] suggest about Aug 2015 and about Sep 2016.
- **Tag versus tarball.** The GitHub tag `4.0.1` differs from the 4.0.1 tarball only in version-date strings, in 7 files (my diff).
- **Nothing newer than 4.0.1 is released.** `master` still calls itself 4.0.1 (CMake `VERSION 4.0.1`) but carries 2024–2026 work:
  - libxc 5/6/7 interfaces and PSML output (`psfile=psml`);
  - CMake/CTest build and TOML input;
  - an optional 4th value `rxpsh` on the log-derivative line, and an optional explicit scan grid on the `icmod=4` line;
  - UPF fixes: PR 24 (integer formats), PR 29 (projectors for `nproj>2`), PR 30 (`PP_CHI` closing tag). [GH commit log, `src/input_text_m.f90`]
- **Where to get 2.1.1.** Genuine 2.1.1 source is in the SG15 full tarball under `3rd_party/oncv_src`. [SGT] It is also the GitHub branch `2.1.1` ("ONCVPSP v2.1.1 with some commas added to format strings", 2025-12-15), uploaded in response to I#28. [GH]
- **SG15's patch to 2.1.1.** SG15 built its binary from 2.1.1 plus four replacement files in `src/`:
  - `oncvpsp.f90` and `check_data.f90`: `psfile` may be a string such as `upf+fpmd`;
  - `upfout.f90`: writes `<UPF version="2.0.1">` and the correct version string (genuine 2.1.1 writes `<UPF version="2.1.2">` and "version 1.1.1, 08/28/2013"), echoes 5 decimals, and adds an `End PSP_UPF` marker;
  - `fpmdout.f90`: Qbox XML output.
  - It was built with `ifort -O3` and MKL. [SGT `Makefile`, `src/`]
- **Fork named abinit/oncvpsp.** It does not exist (API 404). `gmatteo/oncvpsp` exists, last pushed 2019-05-24.

**What changed between versions that matters for reproducing 2015 files:**

- **Input format.** The line template is unchanged from 2.1.1 to 4.0.1, and 4.0.0 says: "Input data from the two presently published oncvpsp periodic tables runs without error." [RN] Additions:
  - `psfile` accepts `psp8|upf|both` from 3.2.3; [RN]
  - `icmod` 2–4 and the optional third datum `rcfact` (read only when `icmod==3`) from 3.2.1; [RN, CC]
  - negative `iexc` (libxc) from 3.0.0; [RN]
  - `nproj` up to 5 from 4.0.0. [RN]
  - 2.1.1 accepts only `iexc` 1–4, `icmod` 0–1, `nproj` 1–2 and `psfile` psp8|upf. [SGT `3rd_party/oncv_src/check_data.f90`]
- **rc echo bug (pre-3.2.3).** "Previous versions erroneously reported rcs as shifted to the next-larger log mesh point ... The original input rcs are now truncated to 5 decimals before use". [RN 3.2.3] SG15 confirms: "make sure to use the actual input files (*.dat)". [SG] The code actually computes `int(rc*10.0d5)/10.0d5`, which is 6 decimals, and prints with f10.5. [T401 `src/oncvpsp.f90` l.264-268]
- **Optimizer (3.3.0).** A math error in Eq. (16) was fixed. "The 3.2.3 results can be reproduced by commenting out lines 177-180 and 293-296 of run_optimize"; 3.3.1 keeps `old_optimize.f90` for that purpose. [RN 3.3.0, 3.3.1]
- **Radial mesh.** [`oncvpsp.f90` in SGT, T331, T401]

| | 2.1.1 and 3.3.1 | 4.0.1 |
|---|---|---|
| `amesh` | 1.012 | 1.006 |
| `rr1` | 0.0005/Z | min(0.0005/Z, 0.00005) |
| outer radius | 45 a0 | 45 a0 |

- **Linear-mesh interpolation.** 2.1.1 uses cubic `dp3int`, with r=0 values by 4-point extrapolation. 3.1.1 and later use 7th-order `dpnint`. [RN 3.1.1] From 4.0.0 the local potential at r=0 is set to the value at the first log-mesh point. [RN 4.0.0 item 10]
- **Number of linear points.** 2.1.1: `nrl=int(rlmax/drl)+1`, made even only for upf. 3.3.1 and later: `int(rlmax/drl-0.5)+1`, always made even.
- **4.0.0 rewrite.** Multi-projector loops; a new well-state algorithm (asymptote fixed at ep+0.5 Ha, width varied, extra nodes allowed); a blend of a relativistic potential into the last 5% of [0, rc]; new excited-configuration tests. "The tests/refs *.out files ... show small differences from previous releases." [RN 4.0.0]
- **Second projector with `ep+debl<0` (source reading).** 2.1.1 and 3.3.1 use the actual bound state at n+1 and print "WARNING: negative energy of ... specified for second projector / Actual bound-state energy of ... will be used / Action: debl should probably be increased". [T331 `src/oncvpsp.f90`] 4.0.1's `wellstate` instead substitutes a localized bound state, or +0.05 Ha, or +0.25 Ha ("WARNING wellstate: negative energy specified ..."). [T401 `src/wellstate.f90`] In the SG15 distribution, 26 `.out` files carry the 2.1.1 warning (Mg, Na, Cl, Ar, Cr, Mn, Cu, Zn, As-1.1, Ag, Cd, Pd, Rh, Ta, Hf, Re, Os, W, Au, Hg, Ir, Pt ...), so 4.x cannot give the same second projector for those. [SGT `elements/*/*.out`]
- **Maintainers' view.** "major versions of ONCVPSP generally made breaking changes to the numerical solvers such that the same inputs will likely lead to significantly different pseudopotentials". The reporter confirmed that 2.1.1 reproduces SG15 and 4.0.1 does not. [I#28]
- **SG15 v1.2 is post-processed.** It has smoothing of the local potential near r=0 that is not in the inputs. [I#28; SG revision history 2018-06-25] JDFTx ships only 1.0 and 1.1. [LOCAL]
- **Built-in PBE.** `exc_off_pbe.f` is byte-identical in 2.1.1, 3.3.1 and 4.0.1 (my diff). The other built-in routines got more significant figures in 3.0.0: "Changes in the ref/*.out files are in the 5-6th decimal place". [RN 3.0.0]
- **Erratum.** PRB 95, 239906 concerns only Eqs. (15) and (16) of the optimization: "This error was inconsequential for the accuracy ... small sacrifice in convergence optimization". [ERR] It says nothing about fully relativistic or PBE bugs. I found no erratum on those topics.

## 2. Build and run

- **Requirements.** "a Linux or Unix system with a Fortran 95 compiler", LAPACK and BLAS, gnuplot (4.6 or later for the 3.1.1 graphics), and perl for `fldiff.pl`. [GH `README.md`; T401 `INSTALL`]
- **`make.inc`.** It sets `F77/F90/CC/FCCPP`, `FCCPPFLAGS`, `FFLAGS` (shipped as `-O3 -ffast-math -funroll-loops`), `LIBS` and `OBJS_LIBXC`. [T401 `make.inc`]
- **Building without libxc.** The shipped 4.0.1 `make.inc` has the libxc lines active. To build without libxc, leave only `OBJS_LIBXC = exc_libxc_stub.o`; otherwise the link fails with `undefined reference to exc_libxc_`. [I#5] The built-in `iexc` 1–4 do not need libxc. [I#8]
- **libxc versions.**
  - 3.2.3: "tested with libxc-2.2.1". [RN]
  - 3.3.1 `make.inc`: `-DLIBXC_VERSION=203`. [T331]
  - 4.0.x: "extended to libxc versions 4.0.0 ... and above", via `-DLIBXC_VERSION=400` (or `301`). [RN 4.0.0; T401 `make.inc`]
  - master: CMake needs 3.0 or later and refuses 5.0.0 ("affected by serious bugs"). 4.3.4, 5.2.3, 6.2.2 and 7.0.0 were tested. [GH `CMakeLists.txt`; I#8; I#3]
- **Make build.** `make all >& make.log` builds in `src`, runs `./set_path` (rewrites `PREFIX=` in the scripts and copies them into `tests/`), then runs `tests/data/TEST.sh`. [T401 `Makefile`, `set_path`]
- **CMake build (master).** `cmake -S . -B build [-DLibxc_ROOT=..] [-DLAPACK_ROOT=..]`, then `cmake --build build`, then `ctest --test-dir build`. Options are `ONCVPSP_WITH_Libxc` and `ONCVPSP_WITH_TOML`. Binaries land in `build/bin`. [GH `README.md`, `CMakeLists.txt`]
- **Executables.** `src/Makefile` builds `oncvpsp.x` (scalar-relativistic), `oncvpspnr.x` (non-relativistic) and `oncvpspr.x` (fully relativistic). The INSTALL text's `oncvpsp_nr.x` and `oncvpsp_r.x` are a doc inconsistency. [T401 `src/Makefile`]
- **Purpose of the non-relativistic binary.** "The only reason for including this feature is to support the discussion following Eq.(24) of the paper." [PO]
- **Running.** `oncvpsp.x < prefix.dat > prefix.out`. `scripts/run.sh <prefix> [-np]` does that, then `grep GHOST`, then cuts the plot data and gnuplot script out of the output and starts gnuplot; `-np` skips gnuplot. `run_nr.sh` writes `<prefix>_nr.out`; `run_r.sh` writes `<prefix>_r.out`. [T401 `scripts/`]
- **Other scripts.** `replot.sh <prefix>`; `multirun.sh *.dat` (4 jobs at a time, no plots); `extract.sh <prefix> <dir>` writes `~/<dir>/<prefix>.oncvpsp.psp8` and `.upf`. [T401 `scripts/`]
- **Tests.** `TEST.sh` runs 26 inputs (three with `run_r.sh`), then `compare.sh`. That script strips everything up to "ATOM AND REFERENCE CONFIGURATION", masks lines containing `pspd` or `date`, and runs `fldiff.pl -easy` against `tests/refs`. [T401]
- **Tolerance.** `-easy` is 1.01e-5, and "BOTH the absolute difference and the relative difference ... must be bigger than the tolerance". [T401 `scripts/fldiff.pl`] Summary lines are collected in `TEST.report`: "If a reported summary error appears to be non-trivial, look at the `<prefix>.diff`." [README] CTest passes only on "no significant difference" or zero diffs. [GH `tests/CMakeLists.txt`]
- **Tests needing libxc.** 38_Sr_lxc, 79_Au_lxc and 83_Bi. [RN 4.0.0; README]
- **macOS and recent gfortran.** I found no documented macOS-specific failure. Evidence that it builds:
  - master's own `make.inc` is an Apple-silicon Homebrew setup (`/opt/homebrew/bin/gfortran`, `cpp-13`, `-fallow-argument-mismatch`, libxc 6.0.0 with `-lxcf03`); [GH `make.inc`]
  - maintainers report the full suite passing "on macOS Release and RelWithDebInfo builds" with references generated "on macOS/arm64 with gfortran 15". [I#39]
  - My own check on this machine (gfortran 16.2, `-fsyntax-only`, default flags): every `.f90`/`.f` file of 2.1.1, 3.3.1 and 4.0.1 passes with no errors. `functionals.F90` and `exc_libxc.f90` were not checked, and nothing was linked or run.
- **Reproducibility caveat.** Semilocal l>0 potentials at the innermost grid point differ between compilers and flags because of a cancellation in `pspot.f90`. psp8 and UPF are affected only when `lloc>0`. [I#39]

## 3. Input file

All from [ANN], with bounds from [T401 `src/check_data.f90`] and reading order from [T401 `src/oncvpsp.f90` l.108-180]. Lines starting with `#` are skipped. "There are no defaults." [UG]

**Line 1 — `atsym z nc nv iexc psfile`.**
- `nc`, `nv`: number of core and valence states; `nc+nv` ≤ 30.
- `iexc`: "1-Wigner, 2-Hedin-Lundquist, 3-Perdew-Zunger-Ceperly-Alder, 4-Perdew-Burke-Enzerhof"; negative values select libxc.
- `psfile`: `psp8`, `upf` or `both`.

**Next `nc+nv` lines — `n l f`.**
- "f occupancy (MUST be >0)"; l ≤ 3. Total charge above Z is an error ("reference configuration is negative ion").
- Consequence: unoccupied or unbound states cannot be listed as valence.

**`lmax`.** "maximum angular momentum for which psp is calculated (<=3)".

**`lmax+1` lines — `l rc ep ncon nbas qcut`.**
- `rc`: must be positive. Guidance: "rc should be > that value (usually >1.5 * r_peak, see examples)". [UG]
- `ep`: "eigenvalue inserted for occupied state in reference configuration, positive energy must be specified for barrier-confined "scattering" state for unoccupied l <=lmax. A small positive energy is usually good (0.1-0.25 Ha)."
- `ncon`: "number of constraints for pseudo wave function to match all-electron wave function at rc, value + ncon-1 derivatives, must be between 3 and 5". "I prefer to use ncon = 4 or 5." [UG]
- `nbas`: "Must be between ncon+2 and ncon+5"; the code requires `ncon+3` or more when `nproj=2`. "minimal improvements beyond N = M + 3 or N = M + 4". [PRB Sec. II]
- `qcut`: "wave vector defining "residual energy" in the RRKJ method". Guidance:
  - "typically inversely proportional to rc. For too small a qc, the ... convergence of Er(q) will flatten off for qs larger than qc, while for too large a value, useful convergence is unnecessarily sacrificed"; [PRB]
  - "Increasing qcut by a small amount, say ~0.5, can usually give more uniform convergence". [UG]

**`lloc lpopt rc(5) dvloc0`.**
- `lloc`: "angular momentum whose semi-local psp is taken as local. lloc=4 denotes a smooth polynomial continuation of the all-electron potential ... The rc corresponding to lloc MUST BE THE MINIMUM rc for lloc<=lmax, or <= the minumum for lloc=4 (usually the best choice)". Also "lloc=0, which is always a bad idea anyway". [UG]
- `lpopt` (1–5):
  - "1) match 2 derivatives, r^0,2,4"
  - "2) match 2 derivatives, r^0,4,6"
  - "3) match 3 derivatives, r^0,4,5,6"
  - "4) match 3 derivatives, r^0,4,6,8"
  - "5) match 3 derivatives, r^0,2,4,6"
- `dvloc0`: "shift of r=0 potential from basic continuation (Ha)", added as `dvloc0*(1-x^2)^3` for lpopt 1, `(1-x^4)^3` for lpopt 2, and `(1-x^4)^4` for lpopt 3–5, with x = r/rc.

**`lmax+1` lines — `l nproj debl`.**
- `nproj`: "1 or 2. automatically set to 0 for l=lloc" (up to 5 in 4.x).
- `debl`: "energy added to basic psp energy ep for 2nd projector automatically reset to match 2nd projector with 2nd bound state at this l when it is occupied". The code rejects `debl<0`.
- Guidance: "a spread of ∼1 Ha between ε1 and ε2 works well when the choice is not dictated by the use of two bound functions". [PRB Sec. III]
- 4.x: `debl<=0` together with two or more scattering states is a fatal error. [T401 `oncvpsp.f90` l.409]

**`icmod fcfact (rcfact)`.**
- `icmod` 0: no core correction.
- `icmod` 1: "smooth monotonic polynomial model core charge fit at "matching" rc"; `fcfact` is the "radius ... determined by rho_core(r)=fcfact*rho_pseudo_valence(r) values 0.25-0.5 are usually good".
- `icmod` 2–4: Teter function, see Q9.

**`epsh1 epsh2 depsh`.** Energy window for the log-derivative plots.
- `epsh1`: "should be below the nearest core level for this l to make sure there are no ghosts, but -2.0 Ha usually is OK".
- `epsh2`: "2.0 usually good". `depsh`: "0.02 usually good enough".

**`rlmax drl`.**
- `rlmax`: "must be greater than maximum rc ... ~2-3*rcmax is usually good".
- `drl`: "0.02 is good for "softer" psps, 0.01 is probably better with 1st row, 3d's, or semi-core psps".

**`ncnf`, then per configuration `nvcnf` and `n l f` lines.**
- `ncnf` ≤ 4. "The reference config is always run first ... core always is the reference core".
- Here f may be zero or positive.

**SG15 inputs.** All 86 `.dat` files use `iexc=4`, `lloc=4`, `lpopt=5`, `dvloc0=0`, `icmod=0`, `rlmax=6.0`, `drl=0.01`, `ncnf=0`, `ep=0` (auto), and `psfile=upf+fpmd`. That `psfile` value must be changed to `upf` for official builds. [SGT `elements/*/*.dat`]

## 4. Output file

Layout from [UG, PO], with exact strings from [T401 `tests/refs/32_Ge.out`] and the 4.0.1 sources.

1. **Input echo.** It includes "energy (Ha)" all-electron eigenvalues and is "formatted so that it can be copied and re-used as an input file". [UG] Next comes "Reference configufation results" (sic) with the iteration count and all-electron total energy. `ERROR all-electron reference atom not converged` appears at 100 iterations.

2. **Per l and projector.**
   - In 4.x, a "Wellstate for l = ..." block (eigenvalue, asymptotic potential, half-point radius).
   - In 2.x/3.x, "First projector wave function outermost peak radius" and "All-electron projector functions [0,rc] norms and overlap ... qq(1,1) ...". [SGT `.out` files]
   - "Constraint consistency test": "should be very well satisfied, or there is an error in the code". [UG] The reference shows differences around 1e-13.
   - "Fraction of norm inside rc": a very small value signals an unphysically small rc. [UG]
   - "Residual kinetic energy error".
   - "Total kinetic energy consistency test" and "Potential consistency test at r_c": "the ratio should be close to one, and the difference should be small". [UG]
   - The table "Energy error per electron / Cutoff" gives the cutoffs (Ha) for 1e-2 to 1e-5 Ha. It "correlates quite well with the actual convergence behavior of the total energy". [PRB]

3. **"Construct Vanderbilt / Kleinmman-Bylander projectors".**
   - "B matrix Hermiticity error, ll=" should be about 1e-4 to 1e-5 scalar-relativistically. [PRB Sec. III] It "will be greatly reduced in a non-relativistic run". [UG]
   - "Orthonormal projector coefficients" are the diagonal strengths in Ha: "if the coefficients are huge or the overlap is very close to one, it probably pays to change some parameters". [UG]
   - B-matrix eigenvalues are not printed by default.

4. **"Pseudoatom total energy".**

5. **"Diagnostic tests using Vanderbilt-Kleinman-Bylander pseudopotentials".**
   - 4.x columns: `l rcore rmatch e in delta e norm test slope test`.
   - 2.x/3.x print `e test` instead of `delta e`, and also a semi-local table. [SGT `.out`]
   - Judging: "e in and e test should closely match, and the norm and slope values should be close to one. For two projectors and a non-relativistic calculation, the agreement should be within numerical convergence error". [UG]
   - For positive energies the pseudo energy is the one matching the all-electron log derivative; value and slope ratios are then "equal by construction". [PO]

6. **Ghost test (3.3.0 and later).**
   - Headers: " Testing for bound ghosts" (columns `n l E NL Schr. Eq E Basis Diag. E Cutoff`) and " Testing for highly-localized positive-energy ghosts" (columns `l <radius>/rc E Basis Diag. E Cutoff`).
   - Warning strings: `  WARNING - GHOST(-)` and `  WARNING - GHOST(+)`.
   - History: introduced in 3.3.0; positive energies added in 3.3.1; convergence fixes and fully relativistic support in 4.0.0. [RN, GHO]
   - Method: the pseudo-atom is enclosed in a hard wall at 3·rc. A basis of local-potential-plus-wall eigenstates is built, the projectors are added, and the matrix is diagonalized.
   - GHOST(-): "Eigenvalues lying below the wanted bound states ... are tagged as ghosts". The threshold is 1e-4.
   - GHOST(+): positive-energy states with ⟨r⟩/rc < 1. In 4.x the state must also lie more than 0.05 Ha above the reference. [GHO; T401 `src/run_ghosts.f90`]
   - Basis cutoff: 1.25·½·qmax² in 4.x; in 3.3.1 it is fixed by the basis size (nproj+2+l)·3+1. [T331, T401 `run_ghosts.f90`]
   - Caveats: "This test should be considered only "beta"". "a legitimate low-energy scattering resonance may be reported as a positive-energy ghost, especially for L=3". [RN 4.0.0] GHOST(+) means "the log-derivative plot should be examined carefully around the reported energy". [`run_ghosts.f90` header]
   - Version dependence: SG15 Fe-1.0 gives GHOST(-) at −38.8 Ha with a 3.x build but only GHOST(+) near 20 Ha with 4.0.1. One user found results "numerically equivalent" and asked "Are ghosts real?", which is unanswered. [I#4]

7. **"Test configuration N".**
   - Table `n l f eae eps diff`, then "Total energies and differences" (AE_ref, AE_tst, PS_ref, PS_tst) and `PSP excitation error=`.
   - Configuration 0 is "primarily a consistency check". [PO]
   - "the correlation between these results and results for solids was at best difficult to discern". [PRB Sec. V]
   - A failed configuration is skipped with a WARNING or ERROR line and the run continues. [UG item 7]

8. **"DATA FOR PLOTTING".** Line prefixes [T401 `src/run_plot.f90`, `run_phsft.f90`]:

| Prefix | Content |
|---|---|
| `!p` | r, valence charge, unscreened potential per l |
| ` !L` | local potential |
| `!r` | r, valence, core, model core charge |
| `&` | projector index, l, r, all-electron wf, pseudo wf |
| `!J` | projectors |
| `!C` | convergence profile |
| `! ` | l, energy, all-electron atan(log-derivative), pseudo atan(log-derivative) |

   - The log-derivative blocks are headed "log derivativve data for plotting, l=" (sic) and are computed at the mesh point rc+2 (4.x).
   - Then `GNUSCRIPT` ... `END_GNU`.

9. **Log-derivative plots.** "A step in the pseudopotential plot which is not matched by one in the all-electron plot indicates a ghost". [PO] Since 2.0.1, spurious "sign-change" jumps of π are removed, but "the absence of an expected semi-core step might still occur" and "a perversely sharp ghost step could be mistakenly canceled". [GHO]

10. **Pseudopotential files.**
   - `Begin PSPCODE8` ... `END_PSP` and `Begin PSP_UPF` ... `END_PSP`.
   - `extract.sh` uses awk between `/PSPCODE8/` or `/PSP_UPF/` and `/END_PSP/`. [T401 `scripts/extract.sh`]
   - With `psfile=both`, a line "Updating nrl = ... for uurcut = ..." sits between the two blocks. [T401 `tests/refs/32_Ge.out`]
   - SG15's build ends the UPF with `End PSP_UPF` and uses its own awk scripts. [SGT `util/`]

**Stops you will meet** [UG]: "pspot: first pseudo wave function has node" (fix: "increase qcut a bit"); "optimize: norm constraint violated"; "wellstate: well potential iteration failed".

## 5. UPF writer

**Does any official version write pseudo-atomic wavefunctions? Yes, from 3.2.1.**
"The upf output now includes pseudo wave funtions for the occupied states. This will allow LDA+U calculations in some codes. Please note that ONCVPSP potentials have not been tested or benchmarked for this use ... (Thanks to R. Sundararaman for providing and testing these code modifications)". [RN 3.2.1]

**Blocks by version.**

| Block | Genuine 2.1.1 [SGT `3rd_party/oncv_src/upfout.f90`] | 3.3.1 / 4.0.1 [T331, T401 `src/upfout.f90`] |
|---|---|---|
| `PP_INFO` + `PP_INPUTFILE` | yes; echoes the log-mesh-shifted rc | yes; echoes the input rc (f10.5) |
| `PP_HEADER` | `number_of_wfc="0"`; functional "LDA" or "PBE" only | `number_of_wfc=nv`; functional PZ, PBE, PW91, PBESOL, REVPBE, BP, BLYP, WC |
| `PP_MESH` | linear; `PP_R = (i-1)*drl` (f10.4); `PP_RAB = drl` | same |
| `PP_LOCAL` | 2·Vloc (Ry) | same |
| `PP_BETA.i` | full length | values to `cutoff_radius_index`, then zeros |
| `PP_DIJ` | diagonal, 2·coefficient (Ry) | same |
| `PP_PSWFC` | empty `<PP_PSWFC></PP_PSWFC>` | one `PP_CHI.i` per valence state |
| `PP_NLCC` | only if `icmod=1`; model charge/4π | if `icmod>=1`; same factor |
| `PP_RHOATOM` | r²·rho | same |

- **`PP_HEADER` attributes in both.** generated, author="anonymous", date, element, `pseudo_type="NC"`, `relativistic="scalar"`, `is_ultrasoft/is_paw/is_coulomb/has_so/has_wfc/has_gipaw="F"`, core_correction, functional, z_valence, `total_psenergy` (2·E, Ry), `rho_cutoff`, l_max, `l_local` ("-1" when lloc=4), mesh_size, number_of_wfc, number_of_proj.
- **`has_wfc="F"` is correct.** In UPF it refers to all-electron orbitals in `PP_FULL_WFC`. [UPF]
- **`rho_cutoff` is misused.** ONCVPSP writes the last mesh radius; UPF defines it as a suggested density cutoff. [UPF]

**Which states.** Every valence state of the reference configuration (`do ii=1,nv`). They are bound and occupied by construction, since check_data rejects f ≤ 0. No unbound or unoccupied states are written by any official version.

**Attributes per `PP_CHI`.**
- `occupation` = f (format f6.3).
- `pseudo_energy` = 2·(all-electron eigenvalue), in Ry. It is not the pseudo eigenvalue.
- `label` = all-electron principal quantum number plus one of S, P, D, F, for example "3P".
- `l`.
- Data: the pseudo wavefunction r·χ(r), normalised, interpolated to the linear mesh, with no truncation or smoothing.
- [T401 `src/upfout.f90` l.461-491]

**Radius.** The mesh is extended beyond `rlmax` until the tail norm of every orbital is below 1e-6 ("Cutoff radius such that uu norm accurate to 10^-6"), and "Updating nrl" is printed. All blocks share the longer mesh. [l.95-112]

**How the orbitals are computed (source reading).**
- **3.2.1 to 3.3.1.** They are a by-product of the "Test configuration 0" pseudo-atom calculation. `psatom.f90` stores `uua(:,ii) = uu` after each `lschvkbb` call, before the error check. The main program copies it out only for the first configuration (`if(jj==1) uua = uuatmp`). If that calculation fails, the run continues ("no output for this configuration") and whatever is in the array is written. [T331 `src/psatom.f90` l.150-158, `src/oncvpsp.f90` l.588-608]
- **4.0.x.** They are saved in the main program's valence loop, which builds the pseudo-charge: `lschvkbb` on the screened local potential plus final projectors, with the energy window 0.75·e to 1.25·e around the all-electron eigenvalue; then `uupsa(:,kk)=uu(:)`. A failure there stops the run. These orbitals are by construction consistent with `PP_RHOATOM`. [T401 `src/oncvpsp.f90` l.475-515]
- **Genuine 2.1.1** already runs the equivalent `lschvkbb` loop to build the charge but does not keep the result. [SGT `3rd_party/oncv_src/oncvpsp.f90`]

**JDFTx's SG15 files.**
- The commit message reads "Updating SG15 pseudopotentials with longer atomic orbital tails (generated from my patched version of ONCVPSP using SG15 input files)". [JC]
- The shipped files say "version 2.1.1, 03/26/2014", have `number_of_wfc>0`, and extended meshes (Si-1.1 has 1510 points against SG15's 602). [LOCAL]
- The patch itself is not public as far as I found. That it equals the 3.2.1 code is an inference from the release-note credit and the identical attribute formatting.
- **Reference outputs that fail.** 21 SG15 2.1.1 `.out` files contain `psatom: lschvkbb convergence error` followed by `run_config: ERROR for fully non-local PS atom`: Li, Be, Ne, Sc, Ti, Cr, Mn, Fe, Ni, Ge, Cs, Hf, Ta, W, Re, Au, Hg, Pb, Bi (all 1.0), plus Sb-1.1 and Hf-1.1. [SGT]
- **Spot check of JDFTx's files** (Σχ²·dr): [LOCAL]

| File | Bad orbital | Norm | Failing state in SG15 `.out` |
|---|---|---|---|
| Fe-1.0 | 3S | 0.0000 | n=1, l=0 |
| Li-1.0 | 1S | 0.0000 | n=1, l=0 |
| Ne-1.0 | 2S | 0.0000 | n=1, l=0 |
| Sb-1.1 | 4D | 0.0813 | n=3, l=2 |
| Hf-1.0 | 4F, 5S, 5P, 6S, 5D | 0.0175, 0.0010, 0.0000, 0.9886, 1.9867 | not looked up |

  Si-1.1, O-1.0, Cu-1.0 and Sb-1.0 are all 1.0000. I did not check the other affected files.
- JDFTx 1.4.1 added "normalization of atomic orbitals on radial grid to fix DFT+U for bad pseudopotentials (... previously an issue for some SG15 pseudopotentials)". [JCL]

**Bugs in the official writer.**
- 4.0.1 interpolates only projectors 1 and 2, so `nproj>2` UPFs are wrong (fixed in master, PR 29).
- `number_of_wfc` and `PP_CHI` indices use format i1, giving `*` at 10 or more (I#22, fixed in PR 24).
- The 4.0.1 tarball dropped PW-LDA (`iexc=-001012`) support that 3.3.1 had, with a typo; it is restored in master.

**Other sources of UPFs with `PP_PSWFC`.** pipidog/ONCVPSP re-ran SG15 and ABINIT inputs with "v3.x" unmodified and notes "Some elements in scalar relativistic using sg15's inputs failed ... Although those inputs work well in v2.x, they fail in v3.x". [PIPI] I found no JDFTx mailing-list or doc text on the patch beyond [JC] and [JCL].

**Units and conventions.**
- UPF is in Rydberg units. `PP_BETA` is r·β(r) ("note the factor r!"), `PP_PSWFC` is r·χ(r), `PP_RHOATOM` is 4πr²ρ, `PP_NLCC` is the true density. [UPF]
- ONCVPSP's projectors are r times the radial function. [PRB Sec. III] They are normalised, and the strength sits in the coefficient. [T401 `src/run_vkb.f90`] Hence β is in bohr^-1/2 and `PP_DIJ` = 2·coefficient in Ry.
- The internal densities are 4π·n(r). [T401 `oncvpsp.f90`: `rhops=(uu/rr)**2`]

**Why QE's `ld1.x` cannot read them.** `import_upf.f90` reconstructs a logarithmic mesh from the file, as either exp(xmin+i·dx)/Z or exp(xmin)/Z·(exp(i·dx)−1), and otherwise calls `errore('read_pseudoup','mesh not supported',1)`. A linear mesh starting at r=0 fails that check. [QE l.67-92] Hamann notes: "While all the psp files presently linked to PWSCF use a log mesh, the linear mesh works fine". [PO]

## 6. psp8

- **Header.** [T401 `src/linout.f90`; PSP8]
  - Line 1: title with `ONCVPSP-4.0.1 r_core=` and the log-mesh rc values.
  - Line 2: `zatom, zion, pspd`.
  - Line 3: `pspcod=8, pspxc, lmax, lloc, mmax, r2well`. `pspxc` maps iexc 1→4, 2→5, 3→2, 4→11; a negative iexc is passed through.
  - Line 4: `rchrg, fchrg, qchrg`.
  - Line 5: `nproj`.
  - Line 6: `extension_switch`.
- **`extension_switch`.** 1 (scalar or non-relativistic) or 3 (fully relativistic) since 3.2.3, "to indicate the presence of the valence pseudo-charge density"; use 0 or 2 for ABINIT before 7.10.5. A second integer flags the extra all-electron columns. [RN 3.2.3]
- **Body.** In Hartree, on the same linear mesh:
  - per l, a line `l, ekb(1..nproj)` followed by `i, r, projectors`;
  - the local potential block;
  - if `fchrg>0`, the model core charge and four derivatives (rhoc = 4π × density); [PSP8]
  - `i, r, pseudo valence charge, all-electron valence charge, all-electron core charge`;
  - an `<INPUT>` ... `</INPUT>` echo (3.2.3 and later).
- **No wavefunctions.** There are no pseudo wavefunctions in psp8 in 4.0.1; the format inputs projectors directly "rather than wave functions and semi-local potentials". [PSP8] An HDF5 output PR (I#19) is open and unmerged. Whether master's `linout` adds anything is UNVERIFIED.
- **Radius.** psp8 stops at `rlmax`; only the UPF mesh is extended.

## 7. Exchange-correlation

- **Codes.** `iexc` 1 Wigner, 2 Hedin-Lundqvist, 3 Perdew-Zunger-Ceperley-Alder, 4 PBE. [ANN; T401 `src/vout.f90`]
- **libxc.** Negative `iexc = -XXXCCC`, for example "-001009 (Perdew-Zunger-Ceperly-Alder)" and "-101130 (Perdew-Burke-Ernzerhof)". [LXC]
- **Built-in versus libxc PBE.** "The built-in exc*.f90 rouitines for iexc=1-4 have been tweaked, essentially adding more significant figures to constants, so that the output they generate and that using libxc look essentially identical to the compare.sh test." [LXC] No other difference is documented.
- **Limits.** "Meta-ggas and hybrid exc functions are not implemented ... HF and hybrid functionals are incompatible with norm and generalized-norm conservation." [LXC]
- **UPF functional names.** 3/−001009 → "PZ"; 4/−101130 → "PBE"; −109134 PW91; −116133 PBESOL; −102130 REVPBE; −106132 BP; −106131 BLYP; −118130 WC. Anything else stops the UPF output. [PWX]
- **Mixing functionals.** "when a pseudopotential generated with PBE was (unintentionally) used to compare local-density AE and OV calculations, differences increased noticeably". [PRB Sec. V]

## 8. Relativity

- **`oncvpsp.x`.** Solves the scalar-relativistic radial Schrödinger equation (`srel=.true.`), "since by including the mass-velocity, Darwin, and other higher-order terms it gives a better description of heavier atoms". This breaks exact B-matrix symmetry at about 1e-4 to 1e-5, "so we simply symmetrize". [PRB Sec. III]
- **`oncvpspnr.x`.** The same code with `srel=.false.`. [PO]
- **`oncvpspr.x`.** A separate main program using the radial Dirac equations, with every l loop gaining "ikap=1,2 ... kappa (kap) =l, -(l+1) for j=l -/+ 1/2". Norms use the "renormalized large component only". [REL] "The same input files drive the non- scalar- and fully-relativistic codes." [RN 2.0.1]
- **Fully relativistic UPF.** "The j = l +/- 1/2 non-local pseusopotentials V(r,r') can be used directly for input to PWSCF". [REL] It is written with `relativistic="full"`, `has_so="T"`, and a `PP_SPIN_ORB` block; `PP_CHI` occupations are split as (l+1)/(2l+1)·f and l/(2l+1)·f. [T401 `src/upfout_r.f90`]
- **Fully relativistic psp8.** The two j potentials are decomposed into "the weighted-average and simple difference for scalar (sr) and so", each re-diagonalised; small coefficients are dropped ("eprmin ... 2E-5 Ha"). [REL]
- **Relation between the two.** The output of `oncvpsp.x` is not a j-average of the fully relativistic potentials. It is an independent scalar-relativistic construction from the same input. The j-averaged potential exists only inside the fully relativistic psp8 output. (My reading of [REL] and the two main programs; no doc states this explicitly.)
- **4.0.1 change.** The well found for J=L+1/2 is reused for J=L−1/2. [RN 4.0.1]

## 9. Design guidance

**Reading order.** Read the paper, then INSTALL and the program overview, then the plot walk-through (`73_Ta_plots.pdf`), then the annotated input, then a reference `.out`. "You should always look at these diagnostics, even if the automated graphics all look good." [UG]

**Reference configuration.** "I strongly believe that neutral ground-state reference configurations provide the best pseudopotentials for solids". [PO]

**Semicores.** Treat the outermost core shell as valence for groups 1 and 2 and for transition metals: "including semicores in the same shell added little computational effort". [PRB Sec. IV-V]

**What Hamann tuned.** "Some experimentation with the optimization parameters qc and N as well as the projector energies ε1 and ε2 (where not fixed by bound states) was done to improve convergence and balance it among the ℓs ... with no reference to results for solids." [PRB Sec. V]

**Practical order of adjustment (my assembly of [UG] and [PRB]; Hamann gives no numbered recipe):**
1. Choose rc above the printed outermost-peak radius. Smaller rc gives better scattering but worse convergence.
2. Set `qcut` per l using the "Energy error per electron" table and plot, balancing the l channels.
3. Set `ep` (0.1–0.25 Ha for an empty l) and `debl` (about 1 Ha) so that the well states converge.
4. Inspect the log-derivative plots and the ghost test; fix problems through the local potential.
5. Add or tune the model core charge last.

**Usual suspects for bad behaviour.** Nodes, a large l=0 potential at r=0, or oscillations: "qcut, nbas, rc, ep". [UG items 2-4]

**Ghosts.**
- Cause: "its eigenstates are not necessarily ordered in energy by numbers of nodes". [PRB]
- Remedies:
  - "the second projector of the OV method is very effective at suppressing ghosts compared to KB. In either case, adjustment of the local potential will fix the problem"; [PRB]
  - "There is no real theory for getting rid of ghosts, but adjusting the local potential to be closer to the semi-local potential of the ghost l often works"; [GHO]
  - the Nd example's −208 Ha f ghost "goes away if lpopt is set to 5 instead of 3"; [GHO]
  - 3.3.1 has a commented debug hook that scans `dvloc0` over ±1 Ha and writes B-matrix eigenvalues to `fort.20`. [T331 `src/oncvpsp.f90`]
- "More projectors are an invitation to ghost states." [RN 4.0.0]

**Positive-energy projectors.**
- The barrier is V + v∞·x³/(1+x³), with x=(r−rc)/rb. The well state "should have one more node than the highest-lying core state with the same ℓ"; a second projector adds a node. [PRB Sec. II]
- "the optimized pseudo-wave-function is relatively insensitive to the barrier parameters". [PRB]
- 4.x: "If a highly-localized bound state is found when searching for the first specified scattering state, it is substituted for that state since ghosts and poor performance would otherwise invariably result." [RN 4.0.0]
- A negative `ep+debl` is "probably inadvertent, and it is probably better to adjust debl for a small positive energy". [RN 2.0.1]

**Model core charge.** [CC]
- "Since ONCVPSP can include shallow core states as valence ... the model-core corrections are often unnecessary."
- "Model core charges often produce convergence problems, especially with phonon calculations".
- `icmod=1`: polynomial continuation from the radius where core = `fcfact` × valence.
- `icmod=2`: Teter function fitted in value and slope at that radius.
- `icmod=3`: Teter amplitude and scale given as `fcfact`, `rcfact`. "fcfact" then "has a DIFFERENT MEANING".
- `icmod=4`: `rhomod(rr)=(fcfact*rhocmatch)*F_Teter(rr/(rcfact*rmatch))`, with a coarse-grid scan and Nelder-Mead minimisation of the rms d²Exc error. This "may converge to yield a wildly implausalbe model, or may not converge".
- Since 3.2.3 the Teter function is used at all radii, with no blending to the all-electron tail. [RN 3.2.3]
- **How to read the test.** The output section "Model core correction analysis / based on d2Exc/dn_idn_j" prints `d2excae`, `d2excps` without and with the core, and "rms 2nd derivative error".
- "there is no systematic evidence that "Teter hardness metric" accuracy correlates with improved agreement ... should be regarded as experimental".

**Workflow tooling.** abipy's `oncv.py` wraps the run and plots, and has a `ghost` sub-command that scans output files. [ABI] The PseudoDojo README says nothing about ONCVPSP versions or patches (checked).

## Could not verify

- Release dates of 3.2.1, 3.2.2 and 3.2.3, and the exact date of 3.3.0.
- The text of Sundararaman's 2.1.1 patch. It is not public as far as I found; that it equals the 3.2.1 code is an inference.
- Whether all 21 failing SG15 runs correspond one-to-one to bad JDFTx files; I spot-checked Fe, Li, Ne, Sb-1.1 and Hf-1.0 only.
- That the SG15 2.1.1 source plus `.dat` reproduces the shipped UPFs to the digit with gfortran. SG15 used `ifort -O3`; I#28 reports agreement with 2.1.1 only in words.
- How 4.0.1 treats each of the 26 SG15 inputs with a negative second-projector energy. This is source reading; I did not run it.
- A real macOS build. I ran a syntax check only; the LAPACK link line on macOS is untested.
- Whether master's psp8 writer differs from 4.0.1.
- Release notes for 2.0.1 items 3–4, which are truncated in the original file itself.
- The APS online pages. The paper and erratum were read from the PDFs shipped in the repository.

Skipped: nothing was compiled or run, and the fully relativistic path was only skimmed. Risk: the `PP_CHI` mechanism and the 4.x behaviour for negative second-projector energies rest on my reading of the Fortran, so confirm with one test run (for example Fe-1.0) before building the manual around them.

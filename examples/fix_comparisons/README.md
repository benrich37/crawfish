# Fix comparisons

One notebook per finding in `notes/theory_audit.md`. Each notebook shows how crawfish's results change because
of that fix, saves its figures to `figures/`, and ends with a cell of `assert` checks. A "Before and after" cell near
the top gives, in the README's bra-ket notation, what crawfish did before, what was wrong with it, what changed and
what it does now, with a table pointing each issue to an example spectrum further down.

| notebook | finding(s) | status | compares |
|---|---|---|---|
| `F1_orbital_phase.ipynb` | F1: i^l phase of JDFTx projections | implemented | `legacy_orbital_phase=True` vs. default, plus the reference: N2 (s–p), and wurtzite ZnO pCOMO for p–d (zeroed) and s–d (sign flipped) pairs, with the ZnO pCOHP spectra where every s–d peak flips sign (axial and basal Zn–O) |
| `F2_signed_matrices.ipynb` | F2: signed H/S/P matrices | implemented | legacy matrices vs. fixed vs. reference: pCOHP and pCOBI spectra, integrals, the band-energy sum rule |
| `F3_lattice_images.ipynb` | F3: lattice-image (R) resolution | implemented | legacy (R=0) vs. fixed (nearest image) vs. reference on a 1D chain; N2 with atom 2 wrapped into the next cell; graphene, where two of the three nearest C–C bonds are inter-cell (image shells, orbital-resolved bonds, wrapping, and pCOHP spectra of the wrapped cell and of each bond) |
| `F7_coop_basis.ipynb` | F7: pCOOP and the orbital basis; pCOMO's sign | implemented (docs + warnings; no numbers change) | orthonormalized vs. non-orthogonal projections (15/40 bands), Löwdin vs. non-orthogonal basis, pCOMO vs. COOP/COHP signs; pCOOP spectra for all four choices, and pCOMO vs. pCOOP vs. −pCOHP spectra |
| `F4_F8_inputs_and_normalization.ipynb` | F4, F5, F6, F8 | all implemented | F4: legacy vs. fixed on `N2_nospin` and z-spin N2, including pCOBI spectra; F5: graphene on full, point-group-reduced and k/−k-reduced (unfolded) meshes, including DOS, single-bond and px–px spectra; F6: DOS integrals, Gaussian and LTI, legacy vs. fixed, and two smearing widths; F8: filling helpers vs. `ElecInfo::smear`, and occupations computed without the `fillings` file vs. the file for Al (four smearings) and Si (no smearing), with occupied-DOS spectra |
| `F10_lowdin_cutoff_and_misc.ipynb` | F10: Löwdin cutoff, mu units, `trim_excess_bands`, fallback weights | implemented | overlap eigenvalues vs. the cutoff (N2, graphene with 20 and 8 bands); the old cutoff's dependence on noise sign; graphene's three equivalent bonds, `legacy_lowdin_cutoff=True` vs. default; N2 unchanged; mu vs. pymatgen, and the Si DOS on the old mu axis; re-trimming and weight sums, with N2 pCOHP and DOS spectra |
| `F11_band_span.ipynb` | F11: too few bands | implemented (diagnostic + warning; no numbers change) | `orbital_span_t` per state for fixture pairs that differ only in `elec-n-bands` (N2 10/15, graphene 20/30) and graphene_1x1; IpCOHP band convergence vs. the warning, with pCOHP spectra and running integrals of the fixture pairs |
| `F12_point_group_unfolding.ipynb` | F12: unfolding k-meshes reduced by the point group | implemented | which k-points a 3x3 graphene mesh lumps together (point group vs. time reversal alone); `unfold_point_group=False` (the F5 guard) vs. default vs. full-mesh runs, for graphene (single bonds, px–px, m-resolved pDOS) and wurtzite ZnO (axial and basal Zn–O bonds, Zn d pDOS), with m-resolved pDOS spectra; little-group invariance and rotated vs. directly computed states |
| `F16_F20_cobi_scale_grid_and_inputs.ipynb` | F16: pCOBI scale; F17: default energy grid; F18: `\|projection\|^2` files; F19: an atom with its own periodic images; F20: runs without smearing; F15: its warning | all implemented (F16 with a switch; F17 changed a default; F18–F20 without one) | F16: `legacy_cobi_scale=True` vs. default vs. a float64 Wiberg/Mayer index for N2 no-spin/z-spin and a graphene C–C bond, plus an analytic H₂; F17: one peak's area on the grid vs. its offset, N2 DOS and pCOHP on the old (`res=0.01`) and new default grids, DOS integrals vs. `res`; F18: pDOS from a \|T\|² file as the old code computed it, now, and from the complex file; F19: graphene C #1 with its six second-neighbour images, crawfish vs. float64 and the μ = ν share; F20: runs without a printed μ_F (N2 with it removed, the two Si runs without smearing): the HOMO–LUMO midpoint as μ_F vs. `eigStats`, Löwdin and non-orthogonal bases; F15: the warning text |

## How the notebooks adapt as fixes land

- `fixcompare.py::LEGACY_FLAGS` maps each finding to the `ElecData` setting that restores the old behaviour (notes
  section 4.0). `fix_available("F2")` is True once that setting exists.
- Before a fix: the notebook plots current crawfish against the reference implementation (`fixcompare.Reference`,
  section 2.3 of the notes). Its tests include "known-bug" checks that document the current wrong behaviour.
- After a fix: the notebook loads one `ElecData` with the legacy flag and one with defaults, plots both, and its
  tests require the fixed result to match the reference. The known-bug checks are no longer reached.
- If an implementation names its legacy flag differently, update `LEGACY_FLAGS`. If it picks a different default
  basis (open question Q1), update `REF_BASIS` in the F2 notebook.
- F8 has no legacy switch (crash and bug fixes don't need one), so its notebook checks the fixed code against JDFTx directly.
  Of F10's items only the Löwdin cutoff changes results, so `legacy_lowdin_cutoff` is the F10 entry in `LEGACY_FLAGS`.
- F12's switch is `unfold_point_group`, on by default, so its pre-fix value is False (`fixcompare.LEGACY_VALUES`).
- F16's switch is `legacy_cobi_scale`, the last entry of `LEGACY_FLAGS`, so `load_edata_at` holds it at legacy in
  every earlier notebook: their pCOBI stays on the old (half) scale. `fixcompare.Reference` matches that by default;
  `Reference(..., wiberg_cobi=True)` gives the Wiberg/Mayer scale. `Reference` also takes a lattice vector for
  `images` (μ = ν included for R ≠ 0, as since F19) and `lowdin_rtol` (pass crawfish's `LOWDIN_RTOL` where the bands
  leave orbitals unspanned).
- F17 has no switch: it changed the default `res`, and `res=0.01` gives the old grid and spectra exactly. Notebooks
  that pass an explicit `erange` don't change; F12's spectra without one are drawn on the finer grid since F17.
- F18 (`|projection|^2` files), F19 (self-images, a new capability) and F20 (runs without smearing) are bug fixes
  without a switch; their notebook section rebuilds the "before" from the functions the old code called, or quotes it.
- Add a notebook in the same style for every new result-changing fix.

## Running

Run each notebook with this directory as the working directory, which is VS Code's default, so that
`fixcompare.py` can be imported. The notebooks use the repo's test data and write `.crawfish_cache/` there
(gitignored).

Headless, from the repo root (needs `nbclient`/`nbconvert`; add `nbmake` for the pytest route):

```sh
jupyter nbconvert --to notebook --execute --inplace examples/fix_comparisons/*.ipynb examples/N2.ipynb
pytest --nbmake examples/fix_comparisons examples/N2.ipynb
```

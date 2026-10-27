# Fix comparisons

One notebook per finding in `notes/theory_audit.md`. Each notebook shows how crawfish's results change because
of that fix, saves its figures to `figures/`, and ends with a cell of `assert` checks.

| notebook | finding(s) | status | compares |
|---|---|---|---|
| `F1_orbital_phase.ipynb` | F1: i^l phase of JDFTx projections | implemented | `legacy_orbital_phase=True` vs. default, plus the reference |
| `F2_signed_matrices.ipynb` | F2: signed H/S/P matrices | implemented | legacy matrices vs. fixed vs. reference |
| `F3_lattice_images.ipynb` | F3: lattice-image (R) resolution | implemented | legacy (R=0) vs. fixed (nearest image) vs. reference on a 1D chain; N2 with atom 2 wrapped into the next cell; graphene, where two of the three nearest C–C bonds are inter-cell (image shells, orbital-resolved bonds, wrapping) |
| `F4_F8_inputs_and_normalization.ipynb` | F4, F5, F6, F8 | F4 implemented; F5, F6, F8 upcoming | F4: legacy vs. fixed on `N2_nospin` and z-spin N2; others: current helpers vs. JDFTx conventions and normalized references |

## How the notebooks adapt as fixes land

- `fixcompare.py::LEGACY_FLAGS` maps each finding to the `ElecData` setting that restores the old behaviour (notes
  section 4.0). `fix_available("F2")` is True once that setting exists.
- Before a fix: the notebook plots current crawfish against the reference implementation (`fixcompare.Reference`,
  section 2.3 of the notes). Its tests include "known-bug" checks that document the current wrong behaviour.
- After a fix: the notebook loads one `ElecData` with the legacy flag and one with defaults, plots both, and its
  tests require the fixed result to match the reference. The known-bug checks are no longer reached.
- If an implementation names its legacy flag differently, update `LEGACY_FLAGS`. If it picks a different default
  basis (open question Q1), update `REF_BASIS` in the F2 notebook.
- F8 has no legacy switch (crash and bug fixes don't need one), so its notebook detects the fix from behaviour.
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

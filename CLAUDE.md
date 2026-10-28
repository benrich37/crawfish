# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`crawfish` is a Python library for bonding analysis (pDOS, pCOHP, pCOOP, pCOBI, pCOMO) on JDFTx DFT output, aimed at non-PAW pseudopotential calculations that LOBSTER cannot handle. The README covers the physics, the equations for each spectrum, and how to build an `ElecData` by hand for non-JDFTx data.

## Commands

```sh
pip install .                                   # install (pymatgen comes from a fork: github.com/benrich37/pymatgen, for pymatgen.io.jdftx)
python -m pytest                                # full suite (~15s)
python -m pytest tests/funcs/test_pcohp.py      # one file
python -m pytest tests/utils/test_caching.py::test_name   # one test
pre-commit run --all-files                      # only yaml/eof/whitespace hooks are on; ruff/mypy/pyright are commented out
```

Ruff settings live in `pyproject.toml` (line length 120, py310), but the ruff hook is not enabled.

## Conventions

- **Array naming encodes shape.** `name_indices`, e.g. `proj_tju`, `h_uu`, `p_t_uu`, `e_sabcj`. Indices: `t` = state (spin and k-point collapsed), `s` = spin, `a/b/c` = k-steps along each reciprocal vector (`kfolding`), `j` = band, `u`/`v` = orbital. Underscores inside the index part mean nothing (`s_tj_uu` is the same as `s_tjuu`). New arrays should follow this scheme.
- `tj` arrays are converted to `sabcj` with `reshape([nspin] + list(kfolding) + [nbands])`. This depends on states being ordered spin-major and then by k-grid. A mesh JDFTx reduced only by k ↔ −k is unfolded into that order on load (`ElecData.tr_unfolding`, `unfold_time_reversal`); `nstates_file` is the number of states JDFTx actually wrote, and arrays parsed from files are re-indexed with `_unfold_states`.
- Use `REAL_DTYPE` (float32) / `COMPLEX_DTYPE` (complex64) from `crawfish.utils.typing`, not bare numpy dtypes.
- Energies are in **Hartree** and are **not** shifted to the Fermi level.
- `ElecData.orbital_completeness_u` (how much of each orbital the bands span, from the raw projections) backs the pCOOP warnings; low values mean too few `elec-n-bands`.
- Occupations `occ_tj` are per state in [0, 1], and `wk_t` carries the spin factor (sums to 2), as in JDFTx. JDFTx's `fillings` file is in [0, 2] for no-spin, so `ElecData.occ_tj` divides it by `spin_weight` (unless `legacy_fillings`).
- Hot loops are numba `@jit(nopython=True)` helpers (suffix `_jit`) that sit next to a plain-Python wrapper (`core/operations/`, `io/data_parsing.py`).

## Architecture

**`core/elecdata.py` — `ElecData`** holds all electronic data for one calculation and lazily derives everything else. `ElecData.from_calc_dir(path)` (or `ElecData(path)`) finds JDFTx output files by suffix (`bandProjections`, `eigenvals`, `out`, plus optional `kPts`, `fillings`, ...). `ElecData.as_empty()` plus setters (`user_proj_tju`, `atom_orb_labels_dict`, `kfolding`, `wk_t`, `mu`, ...) is the non-JDFTx path. Most attributes are properties backed by `_private` fields that parse files on first access.

Settings that change derived tensors (`legacy_orbital_phase`, `legacy_matrices`, `legacy_lattice_image`, `legacy_fillings`, `legacy_reduced_kmesh`, `unfold_time_reversal`, `legacy_normalization`, `trim_excess_bands`, `los_orbs`, `p_uu_consistent`, `norm_eigs_for_h`) are properties. `legacy_*` settings restore behaviour from before a documented fix (see `notes/theory_audit.md`, section 4.0), so its effect on results can be measured; they default to `False`. Their setters call `alloc_elec_data()` (except `legacy_lattice_image` and `legacy_normalization`, which only re-point the caches), which re-processes projections (band trimming/normalization, Löwdin orthogonalization) and rebuilds the `h/p/s` orbital matrices through `reval_mat_uu()`. Setting the `_private` field skips that work, so you have to call `alloc_elec_data()` yourself afterwards.

**Spectrum pipeline** (`funcs/`): each public `get_<mode>` / `get_i<mode>` in `funcs/pcohp.py` and `funcs/dos.py` does the following:
1. Resolves `ElecData | path` through `utils/arg_correction.edata_input_to_edata`.
2. Maps atom indices, elements and orbital labels to orbital indices with `utils/indexing.get_orb_idcs`.
3. Builds a `(t, j)` weight array `w_t Re[conj(C_u) M_uv C_v]` from the coefficients `edata.coef_tju` and a signed complex matrix (`edata.h_t_uv`/`s_t_uv`/`d_t_uv` for pCOHP/pCOOP/pCOBI). The `image` argument picks the lattice image: `"all"` (or `state_sep=True`) uses the state-resolved matrices; `"nearest"` (default) or a lattice vector uses `edata.get_mat_r_suv` times `exp(2πi k·R)` per atom pair (pairs chosen in `utils/lattice.get_pair_image_map`). This goes through `funcs/pcohp._compute_signed_tj` and `core/operations/matrix._get_gen_tj_block`. With `legacy_matrices=True` it uses the pre-F2 unsigned `h_uu`/`s_uu`/`p_uu` (or `*_t_uu`) and `_get_gen_tj_coef_uu`/`_tuu` instead. These compute only the requested `(u, v)` overlaps. Don't build full `(t, j, u, u)` tensors in the pipeline (`get_h_p_s_t_uu_and_uu` builds the h/p/s matrices in one pass for this reason). The old 4D functions remain only as test references.
4. Hands that to `funcs/general.evaluate_or_retrieve_generic_spectrum`, which smears it onto `erange` (Gaussian via `core/operations/vector.get_gauss_smear_spectrum`, or linear tetrahedron via `libtetrabz` when `lti=True`). Integrated variants use `get_generic_integrate`.

All spectrum functions return `(erange, spectrum)`. To add a new analysis mode, copy this `get_X` → `_get_X_tj` → `_compute_X_tj` structure.

**Caching (two layers, both on disk under `<calc_dir>/.crawfish_cache/`)**:
- *Weight caches:* `CachedFunction` objects on `ElecData` (`pcohp_tj_cache`, `pdos_tj_cache`, ... listed in `cache_mats`). They map sorted orbital-index pairs to `tj` arrays and are saved as `.npz` in a per-settings subdirectory (`cache_sub_dir`, named by `get_edata_settings_str`, which uses short keys from `edata_setting_key_shorthand_dict`). They are only used when `use_cache` (or `edata.use_cache_default`) is true.
- *Spectrum caches:* `utils/caching.py` builds nested `erange_<i>/` and `<func_name>_<i>/` directories. Each holds a `metadata.pickle`, and a lookup matches by comparing metadata values with type-aware tolerance (`is_matching_value`). Results are written to `spec.npy`. This layer is on by default (`use_cached_spectrum`, `save_spectrum`).
- When you add a parameter that changes the output, include it in `func_args_dict` / `metadata_dict` and in the settings string. If you don't, stale cached results get returned without any error.
- *Invalidation:* on init, `utils/caching.validate_cache_dir` checks `<calc_dir>/.crawfish_cache/cache_stamp.json`. The cache is wiped if a source file changed (size, then POSIX mtime, then sha256 if only the mtime moved) or if the stamp is missing. The stamp also lists the `src/crawfish/result_changes.py` entries (by commit) known to the code that wrote it (`utils/versioning.py`). For any current entry missing from that list, that entry's stage and the stages after it are cleared (`parse` → `matrices` → `spectra`). **Any change that alters results needs an entry there.** Never edit an existing entry's commit, because it is the entry's ID.
- `ElecData.reset_cache()` wipes the cache directory. Tests run against `tests/files/io/example_calc_dirs/*` and write `.crawfish_cache/` there (it's gitignored). Delete it when testing changes to the spectrum code, or the cache can hide them.

**`io/`**: `data_parsing.py` parses JDFTx text outputs (bandProjections, eigenvals, kPts, outfile slices). After the first parse, projections are cached as `proj_tju.npy` plus a `bandProjections_header` in the cache dir. Structure, infile and outfile objects come from `pymatgen.io.jdftx`.

**Tests**: Fixture paths come from `crawfish.utils.testing` (`EXAMPLE_FILES_DIR`, `EXAMPLE_CALC_DIRS_DIR`, `TMP_FILES_DIR`). Spectrum tests currently check only for "runs, no NaN, not all zero". Many stricter numerical tests are commented out.

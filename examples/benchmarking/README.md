# Benchmarking

Notebooks that show how crawfish's results (pCOHP, pCOOP, pCOBI, pDOS and their integrals) converge with the
parameters a user picks when setting up the JDFTx calculation, for example `elec-n-bands`, the cutoff and the
k-mesh. They also show how other choices change that convergence and the overall accuracy:
- the pseudopotential family (ultrasoft GBRV vs norm-conserving SG15);
- orthonormalized vs non-orthogonal projections (`band-projection-params`) and crawfish's Löwdin step;
- alternative orbital bases (Wannier-type, LOBSTER, bases built from exact atomic-orbital matrices).

Where `examples/fix_comparisons/` asks "did this fix change the numbers correctly?", this directory asks "how
much should I trust a number, and which settings make it converge?". Background and derivations are in
`notes/wannier_and_adaptive_basis.md`.

| notebook | shows |
|---|---|
| `01_ipcohp_band_convergence.ipynb` | Bond IpCOHP vs the number of bands for N2 and graphene, with GBRV and SG15 pseudopotentials. Compares eight approaches: crawfish's default; exact-Löwdin and complete frozen-window bases (band-independent; need the patched JDFTx); LOBSTER; frozen-window projected Wannier functions in band space; PDWF; JDFTx `wannier` (projection-only and MLWF); crawfish on non-orthogonal projections, including the ultrasoft `OpsiRadial` issue. Also crawfish's energy-zero dependence when too few bands drop an orbital direction. Section 10: arrays of pCOHP spectra (figures 5–7) and per-peak weights, showing which peaks grow and shrink with the band count, the projection type and the pseudopotential, and where each basis puts the antibonding weight. |

## Contents

- `benchmark.py`: shared helpers. It has the systems (`SYSTEMS`), parsers and float64 implementations of every
  basis, and `crawfish_ipcohp`, which runs crawfish itself on band-truncated copies of a calculation (a run with
  `elec-n-bands nb` computes the same lowest nb states), and the matching spectra (`bond_pcohp`, `crawfish_pcohp`,
  `lobster_bond_pcohp`). It also holds the plotting palette and axis style.
- `data/<system>/`: JDFTx inputs (`common.in`, `*.in`) and the small outputs of each calculation: eigenvalues,
  fillings, k-points, out file, projection dumps, exact AO matrices, JDFTx `wannier` outputs, and LOBSTER outputs
  for the SG15 runs. Wavefunctions are not kept, so to regenerate, rerun `totalE.in` first and then the
  `dump-only` inputs. Pseudopotentials are named relative to JDFTx's pseudopotential library (`GBRV/...`,
  `SG15/...`).
- `jdftx_patch/crawfish_ao_dumps.patch`: the JDFTx change behind the extra `band-projection-params` fields
  `fullO`, `aoMatrices` and `precise`, used by `aom.in` and `aomo.in`. It is not in released JDFTx (notes, section
  10.1).
- `figures/`: PNGs written by the notebooks.

## Running

Run each notebook with this directory as the working directory, which is VS Code's default, so that
`benchmark.py` can be imported. Headless, from the repo root (needs `nbconvert`):

```sh
jupyter nbconvert --to notebook --execute --inplace examples/benchmarking/*.ipynb
```

Each notebook ends with a cell of `assert` checks, one per claim made in its text.

## Adding a notebook

- Number it (`02_...`) and add a row to the table above.
- Put each new calculation in its own `data/<system>/` folder with its inputs, and register it in
  `benchmark.SYSTEMS`.
- Keep wavefunctions and other large files out.
- Run crawfish itself for any curve labelled "crawfish", and keep the float64 reimplementations as independent
  checks.
- Keep the colors of `benchmark.COLORS` per entity across figures.

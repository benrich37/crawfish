# CI setup: test workflow for Linux, macOS and Windows

Written 2026-11-21 against `theory_check` at `57ca953`. New, untracked files:
`.github/workflows/test.yml`, `.gitattributes` and this note.

## Summary

The workflow is modelled on pymatgen-core's `test.yml`. Its install command works on the unedited repo: from a
clean `git archive HEAD` export and a fresh environment on this Mac, 243 tests pass. It will still be red on
GitHub until three test lines change (next section). Nothing in `pyproject.toml` blocks CI. Nothing has run on
GitHub, on real Linux or on Windows; those results are predictions from the local stand-ins described below.

## Changes needed before CI is green

Most important first. Line numbers are for `57ca953`.

1. **`tests/io/test_data_parsing.py`, lines 37 and 47: wrong letter case in two file names.** The fixtures are
   `bandProjections` and `bandProjections_corrupted`. macOS and Windows ignore case, Linux does not, so
   `test_bandprojections_parsing` fails on both Ubuntu jobs.
   ```python
   bandfile_filepath = exfdir / "bandProjections"                                         # line 37
   _parse_bandfile_complex(bandfile_filepath=exfdir / "bandProjections_corrupted")        # line 47
   ```
2. **`tests/core/test_misc_f10.py`, line 135: a pinned number that depends on the LAPACK build.** Delete the line:
   ```python
   assert legacy == pytest.approx([-0.14569, -0.14604, -0.14539], abs=2e-5)  # values before F10
   ```
   It pins the output of the legacy (ill-conditioned) Löwdin path to 2e-5. Those values hold only with Apple's
   Accelerate on arm64. numpy's Linux and Windows wheels use OpenBLAS, so `test_graphene_equivalent_bonds_agree_better`
   fails on the Ubuntu and Windows jobs. Lines 136 and 137 hold everywhere I tried, and line 137 already states
   the legacy-versus-fixed claim. (A tolerance of `abs=5e-3` would cover the three builds below, if you want to
   keep a rough pin; I did not run that variant.)

   | numpy build | legacy values (line 135) | fixed values (line 136) |
   |---|---|---|
   | arm64, Accelerate (this Mac, macOS runner) | -0.145688, -0.146043, -0.145390 | -0.143520, -0.143459, -0.143475 |
   | arm64, OpenBLAS (as Linux/Windows wheels) | -0.146836, -0.146493, -0.145338 | identical to 6 decimals |
   | x86-64, Accelerate (Rosetta) | -0.147783, -0.146914, -0.148038 | identical to 6 decimals |
3. **Commit the two new files and open a pull request into `master`.** The workflow runs on pull requests into
   `master` and on pushes to `master`, so a (draft) PR from `theory_check` is what starts the first run. The
   manual "Run workflow" button appears only once the file is on `master`.

With changes 1 and 2 applied to a scratch copy, all 243 tests pass on the Linux stand-in for Python 3.12 and 3.14.

### `pyproject.toml` (not needed for CI, needed for the release)

The workflow passes with the file as it is, and the same install line still resolves after these edits (checked
with `uv pip install --dry-run` on a scratch copy).

```toml
requires-python = ">=3.12"     # was ">=3.10". pymatgen-core main and misc_fixes_D require 3.12
    "pymatgen-core",           # was "pymatgen@git+https://github.com/benrich37/pymatgen.git",
    "scipy>=1.13.0",           # new. Imported in core/elecdata.py and funcs/general.py; pymatgen-core's own floor
[dependency-groups]            # was [tool.pdm.dev-dependencies]. uv and pip read this table, not the pdm one
```

- **Do not give `pymatgen-core` a version floor while CI installs the fork.** The fork has no upstream tags after
  `v2026.4.7`, so its build reports version `2026.4.7.post1.dev580+g2fa1184dd`. With `pymatgen-core>=2026.10.31`
  the fork jobs fail with "No solution found" (checked). Add the floor at the switch (below). Pushing upstream's
  tags to the fork would also lift the reported version, to about `2026.10.31.post1.dev144` (not tried).
- After the edits, delete the `echo pymatgen > uv-excludes.txt` line and the `--excludes uv-excludes.txt` flag in
  the workflow, and replace `pytest` with `--group test`.
- `numba>=0.55` and `numpy>=1.25.0` predate Python 3.12. The oldest versions with 3.12 wheels are numba 0.59.0
  and numpy 1.26.0.
- `requirements.txt` is stale (numpy 2.0.2, numba 0.60.0, no pymatgen). No workflow reads it.

## Matrix

| Job | Python | pymatgen-core | Must pass |
|---|---|---|---|
| `windows-latest` | 3.13 | fork, `misc_fixes_D` | yes |
| `macos-latest` | 3.13 | fork, `misc_fixes_D` | yes |
| `ubuntu-latest` | 3.12 | fork, `misc_fixes_D` | yes |
| `ubuntu-latest` | 3.14 | fork, `misc_fixes_D` | yes |
| `ubuntu-latest` | 3.13 | latest release on PyPI | no (`continue-on-error`) |

The first four are the platforms and Python versions pymatgen-core tests: its lowest and highest supported
Python on Ubuntu, one version each on macOS and Windows. crawfish cannot go below 3.12 because the pymatgen-core
it needs (upstream main, `misc_fixes_D`) requires 3.12. The suite takes under a minute, so nothing is split
across runners.

## pymatgen-core

**Now.** The four required jobs install
`pymatgen-core @ git+https://github.com/benrich37/pymatgen-core@misc_fixes_D`, the branch you named on
2026-11-21. It is on the public fork at `2fa1184d`, identical to the local clone, and contains every commit of
`misc_fixes_A`. All 243 tests pass on it. Each runner builds it from source (Cython); uv caches the build per
commit.

- `pyproject.toml` still names the old `pymatgen` fork, which would install a second `pymatgen` package over
  pymatgen-core. The workflow leaves it out with uv's `--excludes`.
- The branch is not pinned to a commit, so CI follows it. A broken or force-pushed commit there breaks CI. To
  pin, replace `@misc_fixes_D` with `@<sha>`.

**Switching to stock pymatgen-core.** The fifth job installs the latest pymatgen-core release from PyPI. Today
that is 2026.10.31 and 76 of 243 tests fail locally: 70 for the missing `JDFTXOutfile.spintype`, 5 from the
rejected `elec-n-bands 8 1` line and warnings that then do not fire, plus the file-name test of change 1. Its
red mark is expected and does not fail the run. It installs whatever is newest, so it turns green by itself once
upstream releases #319-#324. When it does:

1. In `pyproject.toml`, set `"pymatgen-core>=<that release>"`.
2. In `test.yml`, delete the `stock: true` entry, the `PYMATGEN` variable and its `"$PYMATGEN"` argument, the
   `continue-on-error` line and the name suffix. The install line becomes `uv pip install -e . --group test`.

If upstream has not released by your cut-off, CI can keep testing the fork, but a PyPI release cannot depend on
it. The stock job then shows how far crawfish is from working on the released pymatgen-core.

## What was checked

Run on this Mac (arm64), each from a `git archive HEAD` export with a fresh uv environment and the workflow's
exact install command. Caches were deleted between runs.

| Check | Result |
|---|---|
| Python 3.13, as the macOS job | 243 passed |
| Python 3.12 and 3.14 on a case-sensitive volume (Linux file names) | 242 passed, 1 failed (change 1) |
| The same with OpenBLAS numpy and scipy (Linux/Windows numerics) | 1 more failed (change 2) |
| The same with changes 1 and 2 applied to the scratch copy | 243 passed on both |
| x86-64 Python 3.13 under Rosetta, with changes 1 and 2 (numpy 2.3.5, numba 0.62.1: newest with Intel-mac wheels) | 242 passed, 1 skipped |
| Stock job's install (pymatgen-core 2026.10.31), Python 3.13 | 76 failed, 167 passed, as expected |
| Wheels exist for every job's OS and Python (wheels-only `uv pip compile`) | yes, except the two source builds below |
| `test.yml`: PyYAML, actionlint 1.7.12 | no findings |
| Tests need no file outside git (`~/Desktop/jdftx_hold`, `examples/`, ignored files) | none needed |

Predicted only:

- **GitHub itself.** Triggers, `setup-uv`'s venv activation and the cache have not run.
- **Source builds on every runner.** `libtetrabz` publishes no wheels and has a C extension; the pymatgen-core
  fork needs Cython. Both built here on Python 3.12, 3.13 and 3.14. On Windows they need MSVC, which the runner
  has. `libtetrabz.c` is plain C without variable-length arrays, and its 0.0.4 release shipped a Windows wheel.
  If the fork fails to build on Windows, add `actions/setup-python@v6` before `setup-uv`, which is what
  pymatgen-core does there.
- **x86-64 with OpenBLAS**, the actual Linux and Windows combination. I tested OpenBLAS on arm64 and Accelerate
  on x86-64 separately; both show only the failure of change 2.
- **Windows** (read from the code, not run). Expect 242 passed and 1 skipped after change 2; the skip is
  `test_reader_time_zone_does_not_matter`, which needs `time.tzset`.
  - Line endings: `.gitattributes` keeps `tests/files/**` byte-identical on every platform. This is insurance,
    not a known failure: all text parsers open files in text mode and git already detects the binary dumps.
  - Path length: a test run writes three cache files whose full path is 268 characters under
    `D:\a\crawfish\crawfish`, above the classic 260 limit. GitHub's Windows image enables long paths, so CI
    should pass. A Windows user without that setting would get errors from the 143-character settings
    directory name.
  - File locking: I found no handle left open on cache files (`np.load` on `.npy`; the `.npz` object is local to
    `load_cache`). `reset_cache` uses `rmtree(ignore_errors=True)`, which would hide a failed delete.
  - Encoding: the text fixtures are pure ASCII, and the only non-ASCII character in `src` is "ö", which the
    Windows default code page can write.

To repeat the local checks:

```sh
mkdir /tmp/crawfish_ci && git archive HEAD | tar -x -C /tmp/crawfish_ci && cd /tmp/crawfish_ci
uv venv --python 3.13 && source .venv/bin/activate
echo pymatgen > uv-excludes.txt
uv pip install -e . pytest "pymatgen-core @ git+https://github.com/benrich37/pymatgen-core@misc_fixes_D" --excludes uv-excludes.txt
pytest tests
# Linux/Windows numerics: swap in the OpenBLAS builds of numpy and scipy, then rerun pytest
uv pip install numpy scipy --python-platform aarch64-apple-darwin --reinstall-package numpy --reinstall-package scipy
# Linux file names: do the same inside a case-sensitive volume
hdiutil create -size 4g -fs "Case-sensitive APFS" -volname cs -type SPARSE cs && hdiutil attach cs.sparseimage
```

## Kept and dropped from pymatgen-core's workflow

Kept: the triggers (push and pull request with a path filter, manual run, reusable by a release workflow), the
`config` matrix style with `fail-fast: false`, `actions/checkout@v6`, `astral-sh/setup-uv@v8.1.0`,
`uv pip install`, `pytest tests --color=yes`, and a `.gitattributes` line that protects the fixtures.

Dropped, with the reason:

- micromamba, conda packages, Enumlib and the cached optional binaries: crawfish has no such dependencies. One
  `setup-uv` step provides Python on all three systems.
- Building a wheel and testing that: `crawfish.utils.testing` finds `tests/files` relative to the source tree,
  so the install must be editable. Moving those paths to `tests/conftest.py` would allow a wheel test.
- `--group test`: there is no `[dependency-groups]` table yet, so the workflow names `pytest`.
- `--resolution=lowest-direct` and optional extras: the lower bounds predate Python 3.12 (see above).
- Coverage and the Codecov upload, with its `id-token` permission: nothing is set up for it.
- The `github.repository` guard, submodules, and the `MPLBACKEND`, `PYTHONWARNDEFAULTENCODING` and `PMG_*`
  variables: not needed here.

Changed: the path filter is `src/**` and `tests/**` instead of `src/**/*.*` and `tests/**/*.*`, because most
fixtures have no file extension, and it also watches `.gitattributes`. A 30-minute job timeout is added.

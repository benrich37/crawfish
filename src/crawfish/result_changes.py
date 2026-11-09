"""Registry of crawfish changes that alter analysis results.

Every change that makes crawfish produce different numbers from the same output files (parsing,
matrix construction, weights, smearing, ...) gets an entry in `RESULT_CHANGES`. Cached data is stamped
with the commits of the entries the writing code knew about. When an ElecData opens a calculation's
cache, any entry missing from that stamp means the cache predates it, and the affected stages are
cleared and recomputed (see `crawfish.utils.caching.validate_cache_dir`).

Adding an entry
---------------
commit : str
    Hash of the commit that introduced the change. It is the entry's ID, so it must be unique and
    must not be edited afterwards: editing it makes every cache look like it predates the entry.
    The hash only exists after committing, so the entry usually goes in a follow-up commit. Caches
    written in between are cleared too, which is harmless.
version : str
    crawfish version the change was made under. Recorded for reference, not compared.
stage : str
    Earliest cache stage the change affects. That stage and everything downstream of it is cleared.
        "parse"    - parsed JDFTx outputs (proj_tju.npy, bandProjections_header)
        "matrices" - h/p/s orbital matrices and cached (t, j) weight arrays
        "spectra"  - cached spectra (erange_*/ directories)
description : str
    One line on what changed.

Example::

    ResultChange(
        commit="0123abcd",
        version="1.0.0",
        stage="matrices",
        description="Fix sign of off-diagonal h_uu elements when norm_eigs_for_h is False",
    ),
"""

from __future__ import annotations

from dataclasses import dataclass

# Ordered upstream -> downstream. Clearing a stage also clears every stage after it.
CACHE_STAGES = ("parse", "matrices", "spectra")


@dataclass(frozen=True)
class ResultChange:
    commit: str
    version: str
    stage: str
    description: str

    def __post_init__(self):
        if self.stage not in CACHE_STAGES:
            raise ValueError(f"Unknown cache stage {self.stage!r} (must be one of {CACHE_STAGES})")


# Caches written before cache stamping existed have no stamp and are cleared in full,
# so changes that predate stamping don't need entries here.
RESULT_CHANGES: list[ResultChange] = [
    ResultChange(
        commit="6bb3bfa660781070fe9380e7440572f5f596e08e",
        version="0.0.0",
        stage="parse",
        description="Initial commit (example entry)",
    ),
    ResultChange(
        commit="a22c558c9fb0616f5610d15522358b3c5faa4b59",
        version="1.0.0",
        stage="matrices",
        description="Apply the i^l orbital phase to JDFTx band projections (notes/theory_audit.md F1; "
        "legacy_orbital_phase restores the old behaviour)",
    ),
    ResultChange(
        commit="3d9ab1b39456dbd322f104ee796bcc6490706f56",
        version="1.0.0",
        stage="matrices",
        description="Signed projected H/S/D matrices for pair spectra (notes/theory_audit.md F2; "
        "legacy_matrices restores the old behaviour)",
    ),
    ResultChange(
        commit="ceffa108c91fd4acbe971d8818a7216b2a76a2c0",
        version="1.0.0",
        stage="matrices",
        description="Pair spectra use the nearest lattice image per atom pair (notes/theory_audit.md F3; "
        "legacy_lattice_image restores same-cell pairing)",
    ),
    ResultChange(
        commit="756f9e04d601eee978726455b1ebefbc82a65457",
        version="1.0.0",
        stage="matrices",
        description="Divide no-spin JDFTx fillings files by the spin weight (notes/theory_audit.md F4; "
        "legacy_fillings restores the raw values)",
    ),
    ResultChange(
        commit="cc5f71ec26a5d65f990351654000606eb3f2e570",
        version="1.0.0",
        stage="matrices",
        description="JDFTx k-point weights on reduced meshes, reduced-mesh guards, time-reversal unfolding "
        "(notes/theory_audit.md F5; legacy_reduced_kmesh / unfold_time_reversal)",
    ),
    ResultChange(
        commit="44cf9081d9bcab99c2acfc478d789823ad261009",
        version="1.0.0",
        stage="spectra",
        description="Normalized Gaussian kernel and single N_k in LTI spectra (notes/theory_audit.md F6; "
        "legacy_normalization restores the old scales)",
    ),
    ResultChange(
        commit="131fd7a689e220ee46cff85f4bd02d97bfc4bcc0",
        version="1.0.0",
        stage="matrices",
        description="Löwdin step in float64 with a relative eigenvalue cutoff (notes/theory_audit.md F10; "
        "legacy_lowdin_cutoff restores the old one)",
    ),
]

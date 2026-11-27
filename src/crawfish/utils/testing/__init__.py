"""This module contains utility functions for testing the crawfish package."""

from crawfish.core import ROOT
from pathlib import Path
import numpy as np
from crawfish.utils.typing import REAL_DTYPE

TEST_FILES_DIR = Path(ROOT) / ".." / "tests" / "files"
EXAMPLE_FILES_DIR = TEST_FILES_DIR / "io" / "example_files"
EXAMPLE_CALC_DIRS_DIR = TEST_FILES_DIR / "io" / "example_calc_dirs"
TMP_FILES_DIR = TEST_FILES_DIR / "io" / "tmp_files"
# This directory should be empty before committing, so creation is required.
TMP_FILES_DIR.mkdir(parents=True, exist_ok=True)


def approx_idx(arr: np.ndarray[REAL_DTYPE], val: REAL_DTYPE) -> int:
    """Return index of value in array closest to input value.

    Return index of value in array closest to input value.

    Parameters
    ----------
    arr : np.ndarray[REAL_DTYPE]
        The (1D) array of interest.
    val : REAL_DTYPE
        The value of interest.
    """
    difs = np.abs(arr - val)
    return np.argsort(difs)[0]


def get_pocket_idx(e_sabcj: np.ndarray[REAL_DTYPE], erange: np.ndarray[REAL_DTYPE]) -> int:
    """Return the index of greatest gap in eigenvalues within energy range.

    Return the index of greatest gap in eigenvalues within energy range.

    Parameters
    ----------
    e_sabcj : np.ndarray[REAL_DTYPE]
        The eigenvalues of interest.
    erange : np.ndarray[REAL_DTYPE]
        The energy range of interest.
    """
    spaces = np.diff(e_sabcj.flatten())
    pocket_mean = np.mean(e_sabcj.flatten()[np.argmax(spaces) : np.argmax(spaces) + 2])
    return approx_idx(erange, pocket_mean)


def write_norm_bandprojections(src: str | Path, dst: str | Path) -> None:
    """Write a copy of a complex bandProjections file as JDFTx writes it with `band-projection-params <ortho> yes`.

    Each state's header line ends "with |projection|^2 per orbital:" and each projection pair (Re, Im) becomes
    |T|^2 printed "%9.7lf " (DumpExcitationsMoments.cpp::dumpProjections with norm=true). No fixture has such a
    file, so tests of finding F18 build one from a complex fixture.
    """
    import re

    lines = Path(src).read_text().splitlines()
    nspecies = int(re.match(r".*, (\d+) species", lines[0])[1])
    out = lines[: 2 + nspecies]
    for line in lines[2 + nspecies:]:
        if line.startswith("#"):
            out.append(line.replace("projectionRe, projectionIm per orbital:", "|projection|^2 per orbital:"))
        else:
            vals = np.array(line.split(), float)
            out.append("".join(f"{x:9.7f} " for x in vals[0::2] ** 2 + vals[1::2] ** 2))
    Path(dst).write_text("\n".join(out) + "\n")


def write_band_truncated_copy(src_dir: str | Path, nbands: int, dst_dir: str | Path) -> Path:
    """Copy a JDFTx calculation directory keeping only its lowest `nbands` bands, as `elec-n-bands nbands` would.

    bandProjections, eigenvals and fillings are cut to the lowest bands of every state; other files are copied as
    they are (any `.crawfish_cache` is left out). For tests with fewer bands than orbitals, or with no empty band.
    """
    import re
    import shutil

    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(".crawfish_cache"))
    lines = (src_dir / "bandProjections").read_text().splitlines()
    m = re.match(r"(\d+) states, (\d+) bands, (.*)", lines[0])
    nstates, nb_all = int(m[1]), int(m[2])
    nspecies = int(lines[0].split(",")[-1].split()[0])
    out = [f"{nstates} states, {nbands} bands, {m[3]}"] + lines[1:2 + nspecies]
    for t in range(nstates):
        start = 2 + nspecies + t * (1 + nb_all)
        out += lines[start:start + 1 + nbands]
    (dst_dir / "bandProjections").write_text("\n".join(out) + "\n")
    for fname in ("eigenvals", "fillings"):
        if (src_dir / fname).exists():
            np.fromfile(src_dir / fname).reshape(nstates, nb_all)[:, :nbands].tofile(dst_dir / fname)
    return dst_dir

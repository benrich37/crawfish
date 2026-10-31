"""Effect of JDFTx's single-centre OpsiRadial (non-ortho USPP projections) on crawfish's non-orthogonal pCOOP/pCOHP.

Compares `band-projection-params no no` (OpsiRadial) with the patched `no no yes` (full O) dumps of the same state,
made by the Wannier/adaptive-basis session in ~/Desktop/jdftx_hold/wannier_cohp/{N2,graphene}.
Usage: python opsiradial_effect.py <wannier_cohp dir> <work dir>
"""
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

from crawfish.core.elecdata import ElecData
from crawfish.funcs.pcohp import get_ipcohp, get_ipcoop

src_root, work = Path(sys.argv[1]), Path(sys.argv[2])
warnings.simplefilter("ignore")


def truncate(src: Path, dst: Path, nb_new: int):
    lines = src.read_text().splitlines()
    nst, nb = int(lines[0].split()[0]), int(lines[0].split()[2])
    out = [lines[0].replace(f"{nb} bands", f"{nb_new} bands")]
    i = 1
    while not lines[i].startswith("#     0"):
        out.append(lines[i]); i += 1
    for _ in range(nst):
        out.append(lines[i]); out.extend(lines[i + 1:i + 1 + nb_new]); i += 1 + nb
    dst.write_text("\n".join(out) + "\n")
    return nst, nb


def make_dir(system, proj_name, nb_new):
    d = work / f"{system}_{proj_name}_{nb_new}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    s = src_root / system
    nst, nb = truncate(s / f"{proj_name}.bandProjections", d / "bandProjections", nb_new)
    for f in ("eigenvals", "fillings"):
        a = np.fromfile(s / f"totalE.{f}", dtype=np.float64).reshape(nst, nb)[:, :nb_new]
        a.tofile(d / f)
    shutil.copy(s / "totalE.kPts", d / "kPts")
    shutil.copy(s / "totalE.out", d / "out")
    return d


cases = {"N2": [15, 80], "graphene": [20, 48]}
print(f"{'system':9s} {'bands':>5s} {'quantity':22s} {'OpsiRadial':>11s} {'full O':>11s} {'diff':>9s}")
for system, nbs in cases.items():
    for nb in nbs:
        res = {}
        for proj in ("nonortho", "aom"):
            ed = ElecData(make_dir(system, proj, nb))
            ed.los_orbs = False
            r = {}
            r["ICOOP all"] = get_ipcoop(ed, idcs1=0, idcs2=1, use_cache=False)[1][-1]
            r["ICOHP all"] = get_ipcohp(ed, idcs1=0, idcs2=1, use_cache=False)[1][-1]
            r["ICOOP s-s"] = get_ipcoop(ed, idcs1=0, idcs2=1, orbs1="s", orbs2="s", use_cache=False)[1][-1]
            r["ICOHP s-s"] = get_ipcohp(ed, idcs1=0, idcs2=1, orbs1="s", orbs2="s", use_cache=False)[1][-1]
            res[proj] = r
        for q in res["aom"]:
            a, b = res["nonortho"][q], res["aom"][q]
            print(f"{system:9s} {nb:5d} {q:22s} {a:+11.5f} {b:+11.5f} {a - b:+9.5f}")

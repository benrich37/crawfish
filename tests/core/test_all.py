# Write a basic call of all spectrum-generating functions to just make sure they run without error

from crawfish.core.elecdata import ElecData
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
from crawfish.funcs import get_pcohp, get_ipcohp, get_dos, get_idos, get_pdos, get_ipdos
#from crawfish.funcs.pcohp import get_pcohp, get_ipcohp
n2_calcdir = EXAMPLE_CALC_DIRS_DIR / "N2_bare_min"
import pytest

@pytest.mark.parametrize("calcdir", [n2_calcdir])
def test_all(calcdir: Path):
    edata = ElecData(calcdir)
    _test_all_tester(edata)
    edata = ElecData.from_calc_dir(calcdir)
    _test_all_tester(edata)


def _test_all_tester(edata: ElecData):
    erange, pcohp = get_pcohp(edata, idcs1=0, idcs2=1, lti=False, res=0.001, norm_max=True)
    erange, ipcohp = get_ipcohp(edata, idcs1=0, idcs2=1)
    erange, dos = get_dos(edata, lti=False, res=0.001, norm_max=True)
    erange, idos = get_idos(edata)
    erange, pdos = get_pdos(edata, elements="N", lti=False, res=0.001, norm_max=True)
    erange, ipdos = get_ipdos(edata, elements="N")

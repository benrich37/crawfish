import sys, warnings
warnings.simplefilter("ignore")
from crawfish.core.elecdata import ElecData
from crawfish.funcs.pcohp import get_ipcohp
for d in sys.argv[1:]:
    out = []
    for flag in (True, False):
        ed = ElecData(d)
        ed.norm_eigs_for_h = flag
        out.append(get_ipcohp(ed, idcs1=0, idcs2=1, use_cache=False)[1][-1])
    print(d.split("/")[-1], f"norm_eigs_for_h=True {out[0]:+.5f}  False {out[1]:+.5f}")

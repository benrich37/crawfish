from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from crawfish.funcs.pcohp import get_pcohp
from crawfish.core.operations.vector import get_uneven_integrated_array
from crawfish.core.operations.matrix import _get_gen_tj, get_h_uu_p_uu_s_uu
from crawfish.core.elecdata import ElecData
import matplotlib.pyplot as plt

n2_calcdir = EXAMPLE_CALC_DIRS_DIR / "N2_no_fillings"
assert n2_calcdir.exists()

# edata = ElecData(n2_calcdir)
# erange, pdos = get_pdos(edata, elements="N", lti=False, res=0.001, norm_max=True)
# plt.plot(erange, pdos)
# erange, pdos = get_pdos(edata, erange=erange, elements="N", lti=True, rattle_eigenvals=True, norm_max=True)
# plt.plot(erange, pdos)
# plt.ylim(0,0.01)
# plt.xlim(0,0.1)


fig, ax = plt.subplots(nrows=2)

edata = ElecData(n2_calcdir)
erange, pdos = get_pcohp(edata, idcs1=0, idcs2=1, lti=False, res=0.001, norm_max=True)
ax[0].plot(erange, pdos)

h_uu, p_uu, s_uu = get_h_uu_p_uu_s_uu(edata.proj_tju, edata.e_tj, edata.occ_tj, edata.wk_t)
pcohp_tj = _get_gen_tj(edata.proj_tju, h_uu, edata.wk_t, [0, 1, 2, 3], [4, 5, 6, 7])
pcohp_sabcj = pcohp_tj.reshape([edata.nspin] + list(edata.kfolding) + [edata.nbands])
e_sabcj = edata.e_sabcj
es, integs = get_uneven_integrated_array(e_sabcj, pcohp_sabcj)
ax[1].plot(es, integs)

print("gf")
plt.show()
print("gd")
print("gfg")

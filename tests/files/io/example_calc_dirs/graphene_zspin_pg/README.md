# graphene_zspin_pg

Made 2026-10-28 to test crawfish's point-group unfolding (notes/theory_audit.md, finding F12). It is the
`graphene_zspin_symm` input (ideal graphene, z-spin, 12x12x1 mesh, 20 bands, `symmetries automatic`) with
`Symmetries` added to the `dump End` line, run with `~/Desktop/jdftx_hold/build_master/jdftx -c 6 -i in -o out -d`
(10 s). JDFTx found 24 space-group operations and reduced the mesh to 19 k-points per spin.

`dump End Symmetries` writes the `sym` file: JDFTx's operations, which crawfish uses to unfold the mesh onto the
full 144 k-points per spin (`ElecData.unfold_point_group`). The `Kpoints` dump also wrote `kMap`, JDFTx's own map
from each full-mesh k-point to the computed one (used by the tests as a cross-check).

`graphene_zspin_full` is the full-mesh reference (`symmetries none`, a separate SCF run on the same FFT grids).
After unfolding, single-bond IpCOHP (−0.50126 Ha vs −0.50126) and m-resolved pDOS (C1 px 1.03889 vs 1.03889)
agree with it to about 1e-5, the agreement between separate SCF runs. `graphene_zspin_symm` (the same run without
the `sym` file) stays reduced and guarded.

Like the other 20-band graphene runs, this one misses an in-plane σ* combination at K, so crawfish warns about too
few bands (finding F11).

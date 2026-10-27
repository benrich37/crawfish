# N2_nospin

Gas-phase N2 with `spintype no-spin`, generated 2026-10-27 to test how crawfish reads JDFTx `fillings` files
(notes/theory_audit.md, finding F4). It uses the same geometry and box as `N2_bare_min`, with no fluid and a
2x2x2 k-mesh. It ran in 38 s on 8 threads with JDFTx `build_master` (`~/Desktop/jdftx_hold/build_master/jdftx -c 8 -i in -o out`).

Key property: JDFTx writes `fillings` in [0, 2] for no-spin, and the k-point weights in `kPts` already sum to 2,
so the raw file counts 20 electrons. crawfish divides by the spin weight and gets 10.

# graphene_ao_full / graphene_ao_pg / graphene_ao_full_nonortho

Made 2026-11-16 to test crawfish's reading of JDFTx's exact atomic-orbital matrices (`dump End AO`: files
`aoOverlap` = <phi|O|phi> and `aoHamiltonian` = <phi|H|phi>, one column-major complex128 nOrbitals x nOrbitals
matrix per state; `ElecData.use_ao_matrices`). The dump comes from the JDFTx branch `lowdin_dumps` at `e81d53b0`,
built in `~/Desktop/jdftx_hold/build_lowdin_dumps_claude`. Each run was
`~/Desktop/jdftx_hold/build_lowdin_dumps_claude/jdftx -c 8 -i in -o out -d` (3 to 12 s).

All three are ideal graphene (the geometry of `graphene_zspin_*`) with the SG15 norm-conserving
`C_ONCV_PBE-1.0.upf` pseudopotential at 30 Ha, PBE, no-spin, a 6x6x1 mesh, 16 bands and Fermi smearing of
0.005 Ha. Norm-conserving, so JDFTx's non-orthogonal projections are exactly <phi|psi> (ultrasoft ones are not;
notes/theory_audit.md, F13) and ortho = S^-1/2 x non-ortho holds to the 7 printed decimals.

- `graphene_ao_full`: `symmetries none`, ortho projections (`band-projection-params yes no`), 36 states.
- `graphene_ao_pg`: `symmetries automatic` plus `dump End Symmetries` (24 operations, 7 states), so crawfish unfolds
  the AO matrices onto the full mesh like the projections. They match `graphene_ao_full` to the agreement between
  separate SCF runs.
- `graphene_ao_full_nonortho`: the files of `graphene_ao_full`, except `bandProjections`, which comes from a
  dump-only restart of its wavefunctions with `band-projection-params no no` (`dumponly.in`). That restart's own
  outfile has no mu, and it recomputes H and the eigenvalues to about 1e-6, so those files were taken from the
  SCF run.

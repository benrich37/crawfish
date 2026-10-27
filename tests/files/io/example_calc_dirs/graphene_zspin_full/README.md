# graphene_zspin_full / graphene_zspin_symm / graphene_zspin_tr

Three calculations made 2026-10-27 to test crawfish on symmetry-reduced k-meshes (notes/theory_audit.md,
finding F5). Both are ideal graphene (a = 2.46 Å, 20 bohr cell height), GBRV v1.2 `c_pbe_v1.2.uspp`, PBE,
z-spin with zero magnetization, a 12x12x1 Γ-centred mesh, 20 bands, Fermi smearing 0.005 Ha, and ortho-orbital
projections. They were run with `~/Desktop/jdftx_hold/build_master/jdftx -c 8 -i in -o out -d`.

- `graphene_zspin_full`: `symmetries none`, 144 k-points per spin (53 s).
- `graphene_zspin_symm`: `symmetries automatic`, reduced to 19 k-points per spin with non-uniform weights (8 s).
- `graphene_zspin_tr`: `symmetries manual` with only the identity `symmetry-matrix`. JDFTx then reduces by time
  reversal alone (`kpoint-reduce-inversion yes` is the default), giving 74 k-points per spin: 4 self-conjugate
  plus 70 k/−k pairs with doubled weight (29 s). crawfish unfolds this onto the full mesh by default
  (`ElecData.unfold_time_reversal`), and it then matches `graphene_zspin_full` to within ~1e-5 Ha.

`graphene_zspin_full` and `graphene_zspin_symm` agree on symmetry-invariant quantities: the C1–C2 pCOHP summed over all images, total-shell pDOS,
pz. They differ for single bonds and for m-resolved px/py.

20 bands are needed. With 8 bands (= number of orbitals), the σ* states fall outside the band set at some
k-points, so the s/p orbitals aren't spanned there and the Löwdin step becomes ill-conditioned.

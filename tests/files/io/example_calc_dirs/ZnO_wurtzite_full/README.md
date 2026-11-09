# ZnO_wurtzite_pg and ZnO_wurtzite_full

Made 2026-10-28 to test crawfish's point-group unfolding (notes/theory_audit.md, finding F12). Wurtzite ZnO (a =
3.25 Å, c = 5.207 Å, u = 0.382; space group P6_3mc), GBRV `zn_pbe_v1.uspp` and `o_pbe_v1.2.uspp`, PBE, no-spin, a
4x4x3 Γ-centred mesh, 56 bands, Fermi smearing 0.005 Ha, ortho-orbital projections, `energyDiffThreshold 1e-12`.
Run with `~/Desktop/jdftx_hold/build_master/jdftx -i in -o out -d` (`-c 8`, then `-c 12`).

Wurtzite exercises everything the unfolding has to get right:

- 12 operations, half with a screw translation of c/2 (`sym` file), so atoms map onto images in other cells;
- no inversion, so JDFTx adds k <-> -k and time reversal combines with the rotations;
- Zn d orbitals and two s and two p shells on Zn (13 orbitals per Zn, 4 per O);
- two species, and two inequivalent Zn–O bonds (axial 1.989 Å, three basal 1.974 Å).

- `ZnO_wurtzite_pg`: `symmetries automatic` with `dump End ... Symmetries ElecDensity` (10 s). Reduced to 8 k-points.
  `sym` holds JDFTx's operations, `kMap` its own k-point map, and `n` the converged density.
- `ZnO_wurtzite_full`: `symmetries none`, non-self-consistent at that density (`fix-electron-density`, 22 s; the run
  read it from a scratch copy, as its `out` shows). It is the full-mesh reference with the same Hamiltonian, so it
  doesn't carry the ~1e-3 differences that two separate SCF runs show in individual band subspaces.

After unfolding, every rotated state's band-space overlap and Hamiltonian match the directly computed state to
~1e-6 (file precision). Single-bond IpCOHP matches: axial −0.144380 vs −0.144380 Ha, basal −0.14943 vs −0.14941
(the three basal bonds themselves differ by 1.5e-5 in the full run). Zn d m-resolved occupied pDOS matches to 1e-6.
Γ is the exception for individual subspaces: JDFTx's two runs differ there by up to 2e-3 even though no
transformation is involved, and the 56th band at Γ is half of a degenerate pair.

# N2_vectorspin

Gas-phase N2 with `spintype vector-spin` (spinor wavefunctions), made 2026-10-27 to test that crawfish refuses
spinor calculations with a clear error (notes/theory_audit.md, finding F9). It is a 15 bohr cubic box, GBRV v1.2
`n_pbe_v1.2.uspp`, PBE, Γ only, 16 bands, Fermi smearing 0.001 Ha and ortho-orbital projections, run with
`~/Desktop/jdftx_hold/build_master/jdftx -c 8 -i in -o out -d` (5 s).

What it shows about JDFTx's spinor layout: one state (one k-point, weight 1, no spin index), and 16 projections
for 2 atoms × 4 spatial orbitals. The bandProjections header lists `nOrbitalsPerAtom` = 8 for N (s, p), because each
spatial orbital comes as a spin-up and a spin-down copy (`SpeciesInfo_atomic.cpp`, `nSpinCopies = 2`).

There is no spin-orbit fixture: no fully relativistic pseudopotentials are installed locally. The tests cover
`spin-orbit` with a copy of this outfile whose `spintype` line is edited.

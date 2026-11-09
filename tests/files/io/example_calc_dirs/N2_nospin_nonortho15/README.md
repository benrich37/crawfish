# N2_nospin_nonortho15 / N2_nospin_nonortho40

Gas-phase N2 (same input as `N2_nospin`: no-spin, 2x2x2, no fluid), but with `band-projection-params no no`
(non-orthogonal atomic orbitals), and 15 or 40 bands. Made 2026-10-27 to test pCOOP (notes/theory_audit.md,
finding F7). They ran in 38 s and 85 s with `~/Desktop/jdftx_hold/build_master/jdftx -c 8`.

With non-orthogonal orbitals, the band-space overlap S = T T^dag approximates <phi|O|phi>. Orbital completeness
(`ElecData.orbital_completeness_u`) is s 0.99, py/pz 0.98, px 0.90 (15 bands) and 0.91 (40 bands). The N1–N2
overlaps have the textbook signs: S(s,s) = 0.50, S(py,py) = +0.37 (π), S(px,px) = −0.21 (σ, parallel p).

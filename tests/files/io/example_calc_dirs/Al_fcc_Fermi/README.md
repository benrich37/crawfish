# Al_fcc_{Fermi,Gauss,MP1,Cold} / Si_nosmear_{nospin,zspin_M2}

Six small calculations made 2026-10-27 to test crawfish's occupation helpers, which compute fillings from the
eigenvalues when no `fillings` file was dumped (notes/theory_audit.md, finding F8). Each was run with
`~/Desktop/jdftx_hold/build_master/jdftx -c 8 -i in -o out -d` in a few seconds, and dumps `fillings` so the
helpers can be compared against JDFTx. This README is shared by all six directories (it lives in `Al_fcc_Fermi`).

- `Al_fcc_<type>`: fcc Al (a = 7.653 bohr, lattice vectors written out because pymatgen doesn't parse
  `lattice face-centered Cubic`), GBRV v1 `al_pbe_v1.uspp`, PBE, no-spin, 8x8x8 mesh reduced by symmetry to 29
  states, 8 bands, `elec-smearing <type> 0.01`. The metal gives many fractional fillings for each smearing type.
- `Si_nosmear_nospin`: diamond Si (a = 10.26 bohr), GBRV v1 `si_pbe_v1.uspp`, no-spin, 4x4x4 mesh (8 states),
  8 bands, no `elec-smearing`, so JDFTx uses constant fillings (4 bands full in every state) and prints no mu.
- `Si_nosmear_zspin_M2`: the same with `spintype z-spin` and `elec-initial-magnetization 2 yes`. Constant
  fillings then put 5 electrons in every spin-up state and 3 in every spin-down state (ElecInfo::setup), so the
  spin-up HOMO lies above the spin-down LUMO (eigStats reports a gap of -0.26 Ha). Filling the lowest levels of
  both channels together would get this wrong.

# N2_nospin_10bands and graphene_zspin_tr_30bands

Band-count companions to existing fixtures, made 2026-10-28 to test crawfish's too-few-bands warning
(notes/theory_audit.md, finding F11). Each is the original input with only `elec-n-bands` changed, run with
`~/Desktop/jdftx_hold/build_master/jdftx -c 8 -i in -o out -d`. The first comment line of each `in` is the
original fixture's.

- `N2_nospin_10bands`: `N2_nospin` (15 bands) with 10 bands (36 s). At 5 of 8 k-points an N s/px combination
  along the bond (σ*) is missing from the bands, so crawfish warns. N1–N2 IpCOHP: −0.8497 Ha at 10 bands,
  −0.8668 at 15, −0.8757 at 40 (not kept).
- `graphene_zspin_tr_30bands`: `graphene_zspin_tr` (20 bands) with 30 bands (30 s). At 30 bands every orbital
  combination is spanned at every k-point (smallest band-space overlap eigenvalue 0.93), so crawfish doesn't warn.
  At 20 bands an in-plane σ* combination is missing at K and crawfish warns. Mean IpCOHP of C1's three nearest
  C2 images: −0.5013 Ha at 20 bands, −0.5193 at 30, −0.5239 at 40 (not kept).

Other band counts (graphene 8, 10, 12, 16, 40; N2 8, 12, 25, 40; non-orthogonal graphene 8, 12, 20, 40) were run
for the F11 notes but not kept.

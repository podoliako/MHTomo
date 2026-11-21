# The Geysers withheld-block test (article Section 4.7)

Files for repeating the real-data evaluation of hard and soft MHTomo and tomoDD on P picks from
The Geysers, 1–20 April 2011. The numerical code is the code of this repository, unchanged; the
article runs used a frozen copy whose numerical modules are byte-identical to it.

## Contents

| File | Content |
|---|---|
| `data/training/arrivals.npz`, `metadata.json` | Training archive of all article runs: 3471 events × 31 stations, 42,973 picks. Times are relative to each event's earliest training pick. |
| `data/external_holdout_picks.csv` | The 4775 picks (event, station) removed from `../../data/geysers_2011/real_picks_example` by an earlier pick-level split; they are not used in Section 4.7. The example archive minus these picks has exactly the training pick pattern. |
| `blocks.json` | Four withheld spheres A–D (radius 1.4 km; centers in local box coordinates, *z* down from the box top at 1166 m above sea level) and the IDs of their events. The catalog defines the blocks only. Block A is the block of the article figures; B–D were chosen from the catalog alone, disjoint from A and from each other, before any of their inversions. |
| `configs.json` | `InversionConfig` values of the hard and soft runs (identical for all blocks; 15 updates, no automatic stopping). Start: `../../data/geysers_2011/initial_models/linear_vp_3p5_6p0`. |
| `evaluation_splits.csv` | Role of every withheld pick: `relocation` (about 80% per event, at least five) or `prediction` (the rest, at least one). Drawn per event with NumPy `default_rng(42)`, `count = min(max(1, round(0.2 n)), n − 5)`. `evaluate --split-seed K` draws the same rule with seed K; K = 42 reproduces this file exactly, and K = 43, 44 are the two further splits of the article (Appendix table of the Section 4.7 split seeds). |
| `tomodd/<block>/` | tomoDD-SE inputs of each block: training events only, unperturbed catalog start locations, the same linear start sampled at the `MOD` nodes, three joint iterations, damping 40, smoothing 5/5/5 (`tomoDD-SE.inp`). `dt.ct` pairs training events within 3 km, at most four neighbours per event, using common training P stations; `dt.cc` is empty. |
| `tomodd/tomoDD-SE.patch`, `tomodd/build.json` | All changes of the solver used for the article to tomoDD-SE v1.0 of the USTC training 2020 distribution (array dimensions, coordinate round-off, minimum ray segment 0.05 km, horizontal grid guard) and the compiler command. tomoDD-SE itself is not redistributed. |
| `reproduce.py` | `prepare`, `invert`, `evaluate`, `score`; see its docstring. |

## Commands

From `projects/EMTomo`:

```sh
D=reproduce/geysers_withheld_blocks
for B in A B C D; do
  python $D/reproduce.py prepare --block $B --out work/$B
  python $D/reproduce.py invert  --block $B --mode hard --out work/$B --workers 12
  python $D/reproduce.py invert  --block $B --mode soft --out work/$B --workers 12
done
# tomoDD: run tomoDD-SE in a copy of tomodd/<block>/ (outputs MOD-node Vp_model.dat)
python $D/reproduce.py evaluate --block A --out work/A --workers 12 \
  --model initial=data/geysers_2011/initial_models/linear_vp_3p5_6p0/initial_velocity_m_s.npy \
  --model hard=work/A/runs/<hard run>/final_model.npy \
  --model soft=work/A/runs/<soft run>/final_model.npy \
  --model tomodd=tomodd:<tomoDD work directory>
# ... the same for B, C, D
python $D/reproduce.py score --work work
# further pick splits: prepare each block into another --out (e.g. work43/<block>), run the same
# evaluate commands there with --split-seed 43 (or 44), then score that directory
```

`evaluate` keeps each velocity model fixed. Every withheld event is located from its relocation
picks with the same grid likelihood (relative 1%, absolute 0.05 s, model 0.02 s), 50 seed cells
with a minimum separation of 7 cells on the 125-m grid and bounded Powell refinement, without the
catalog; its origin time is the precision-weighted mean offset on the relocation picks, and its
prediction picks are then predicted. `score` reports the pick-weighted prediction RMSE per model
and paired event-bootstrap intervals of relative differences (5000 draws, seed 20261118, drawn
for blocks in the order A, B, C, D), per block and pooled over the blocks (events resampled within
blocks). Evaluating the article's saved models with these commands reproduces the article values
(Table 7 and the appendix table of split seeds 43 and 44, including the pooled soft–hard intervals),
for example 131.5, 118.5, 115.7 and 118.7 ms for the start, hard, soft and tomoDD models of block A
and a pooled soft–hard difference of −1.0% (95% −3.2, +1.6). The inversions were not rerun
from this package; their inputs (training archive, configuration, start model) are byte-identical
to those of the article runs.

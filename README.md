# MHTomo — multi-hypothesis local-earthquake traveltime tomography

MHTomo (developed under the working name *EMTomo*; module and directory names keep that name)
jointly estimates a 3-D P-wave velocity model and earthquake locations from arrival-time picks.
It alternates an event-location step and a velocity-update step. Each event enters the velocity
update either through its most likely candidate hypocenter (**hard** assignment) or through several
likelihood-weighted candidates (**soft** assignment), with all other settings shared.

* Velocity model: box of cubic cells with one P velocity each; traveltimes by fast marching
  (`scikit-fmm`) on a finer computational grid, one field per station.
* Location step: likelihood with the origin time integrated out, evaluated on the whole grid;
  candidate cells selected with a minimum separation and refined by a local Powell search.
* Velocity step: rays traced through the station fields, damped weighted least squares for the
  slowness increment, optional per-cell step limit.

## Installation

Python ≥ 3.11.

```sh
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

## Tests

```sh
cd projects/EMTomo && python -m pytest -q tests          # method
cd ../.. && python -m pytest -q projects/forward_modeling/tests   # synthetic generator
```

## Your own picks

Observations are a sparse arrival matrix in `arrivals.npz` with a `metadata.json`
(see `projects/EMTomo/data/geysers_2011/real_picks_example/emtomo/` and
`experiment_data.load_real_pick_experiment`):

* `event_ids`, `station_ids` (unique strings), `station_coordinates_m` (stations × 3, local box
  coordinates in metres, *z* positive down from the box top), `arrival_times_s` (events × stations,
  `NaN` for missing picks; each event referenced to its earliest pick);
* `metadata.json`: `schema="emtomo.real_picks.v1"`, `time_reference`, `units`, `domain_extent_m`,
  `minimum_training_picks`, and the SHA-256 of `arrivals.npz`.

The domain is defined by `domain_extent_m` and the cell size; a starting model is an array of
cell velocities (m/s) on that grid. Run from `projects/EMTomo`:

```python
import numpy as np
from config import InversionConfig
from experiment_runner import run_real_experiment
from velocity_model import VelocityModel

data = "data/geysers_2011/real_picks_example/emtomo"
start = np.load("data/geysers_2011/initial_models/linear_vp_3p5_6p0/initial_velocity_m_s.npy")
config = InversionConfig(cell_size=1000., subdivision=8, n_cycles=15, candidate_mode="soft",
                         n_candidates=50, weights_top_n=50, weights_min_distance=7,
                         weight_noise_relative_sigma=.01, weight_noise_absolute_sigma_s=.05,
                         control_fraction=0., n_workers=12, runs_dir="runs")
run_real_experiment(data, config, initial_model=VelocityModel(start, config.cell_size))
```

Use `candidate_mode="hard"` (one candidate) for hard assignment. All settings and their meaning
are documented in `config.py`. Results (models after each update, candidate weights, rays,
diagnostics) are written to `runs/`.

## Example data

`projects/EMTomo/data/geysers_2011/real_picks_example/` contains all 47,748 P-wave picks of 3471 events
(at least six picks each) recorded at 31 stations of The Geysers geothermal field, 1–20 April 2011, converted to the input
format above, with station metadata, the event catalog (for comparison only; depths in km below
mean sea level) and the box geometry (23 × 11 × 7 km, top 1166 m above sea level).
Picks and catalog are from the NCEDC Enhanced Geothermal Systems (BG/LBNL) catalog; station
metadata from the NCEDC FDSN Station service. Please cite:
NCEDC (2014), Northern California Earthquake Data Center. UC Berkeley Seismological Laboratory.
Dataset. doi:10.7932/NCEDC.

## Synthetic tests

`projects/forward_modeling` generates synthetic first arrivals with PyKonal from a velocity model,
stations and events, with optional relative + absolute Gaussian pick noise
(`python -m projects.forward_modeling EXPERIMENT_ID --noise` from the repository root, after
saving inputs with `projects.forward_modeling.experiments.save_inputs`). `projects/EMTomo/main.py`
inverts such a saved experiment.

## License

MIT, see `LICENSE`.

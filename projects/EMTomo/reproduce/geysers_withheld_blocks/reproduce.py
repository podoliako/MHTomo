"""Reproduce the withheld-block test of The Geysers (Section 4.7 of the MHTomo article).

Run from projects/EMTomo (Python >= 3.11, requirements.txt installed):

    D=reproduce/geysers_withheld_blocks
    python $D/reproduce.py prepare  --block A --out work/A
    python $D/reproduce.py invert   --block A --mode hard --out work/A --workers 12
    python $D/reproduce.py invert   --block A --mode soft --out work/A --workers 12
    python $D/reproduce.py evaluate --block A --out work/A --workers 12 \
        --model initial=data/geysers_2011/initial_models/linear_vp_3p5_6p0/initial_velocity_m_s.npy \
        --model hard=work/A/runs/<hard run>/final_model.npy \
        --model soft=work/A/runs/<soft run>/final_model.npy \
        --model tomodd=tomodd:<tomoDD work directory with MOD and Vp_model.dat>
    python $D/reproduce.py score --work work      # all prepared blocks, A..D

prepare  training archive (data/training, the 4775 external holdout picks already removed) without
         the withheld events of the block -> <out>/emtomo; withheld events -> <out>/withheld.npz.
invert   15 updates with the article configuration (configs.json) from the linear 3.5-6.0 km/s start;
         run directories in <out>/runs.
evaluate every model fixed: each withheld event is located from its relocation picks
         (evaluation_splits.csv) by the grid likelihood, 50 seed cells with minimum separation 7
         cells on the 125-m grid and Powell refinement, without the catalog; its origin time is the
         precision-weighted mean offset on the relocation picks; its prediction picks are then
         predicted. Predictions -> <out>/evaluation/<name>.npz.
score    pick-weighted prediction RMSE and paired event bootstrap (5000 draws, seed 20261118;
         draws are generated for blocks in the order A, B, C, D, as in the article).
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import numpy as np

HERE = Path(__file__).resolve().parent
EMTOMO = HERE.parents[1]
sys.path.insert(0, str(EMTOMO))
INITIAL = EMTOMO / 'data/geysers_2011/initial_models/linear_vp_3p5_6p0/initial_velocity_m_s.npy'
GEOMETRY = EMTOMO / 'data/geysers_2011/real_picks_example/geometry.json'
H, SHAPE = 125., (184, 88, 56)
NOISE = (.01, .05, .02)
SHORTLIST, DMIN = 50, 7
N_BOOT, BOOT_SEED = 5000, 20261118
ORDER = ('initial', 'hard', 'soft', 'tomodd')
_STATE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocks():
    return json.loads((HERE / 'blocks.json').read_text())


def training():
    with np.load(HERE / 'data/training/arrivals.npz', allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def prepare(block, out):
    b = blocks()[block]
    arrays = training()
    withheld = np.isin(arrays['event_ids'].astype(str), b['withheld_event_ids'])
    assert withheld.sum() == b['n_withheld_events']
    (out / 'emtomo').mkdir(parents=True, exist_ok=False)
    with (out / 'emtomo/arrivals.npz').open('xb') as stream:
        np.savez(stream, **dict(arrays, event_ids=arrays['event_ids'][~withheld],
                                arrival_times_s=arrays['arrival_times_s'][~withheld]))
    metadata = json.loads((HERE / 'data/training/metadata.json').read_text())
    metadata['arrays_sha256'] = sha(out / 'emtomo/arrivals.npz')
    metadata['spatial_holdout'] = dict(centre_m=b['centre_m'], radius_m=b['radius_m'],
                                       removed_events=int(withheld.sum()))
    (out / 'emtomo/metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.savez(out / 'withheld.npz', event_ids=arrays['event_ids'][withheld], station_ids=arrays['station_ids'],
             station_coordinates_m=arrays['station_coordinates_m'],
             arrival_times_s=arrays['arrival_times_s'][withheld])
    print(f'block {block}: {int((~withheld).sum())} training events, '
          f'{int(np.isfinite(arrays["arrival_times_s"][~withheld]).sum())} training picks; '
          f'{int(withheld.sum())} withheld events')


def invert(block, mode, out, workers):
    from config import InversionConfig
    from experiment_runner import run_real_experiment
    from velocity_model import VelocityModel
    config = dict(json.loads((HERE / 'configs.json').read_text())[mode])
    config.update(n_workers=workers, runs_dir=str(out / 'runs'), save_runs=True,
                  run_name=f'geysers-withheld-{block}-{mode}')
    config = InversionConfig(**config)
    start = VelocityModel(np.load(INITIAL, allow_pickle=False), config.cell_size)
    run_real_experiment(out / 'emtomo', config, initial_model=start)


def fine_velocity(spec):
    kind, _, path = spec.partition(':') if spec.startswith('tomodd:') else ('cells', '', spec)
    path = Path(path)
    if kind == 'cells':
        coarse = np.load(path, allow_pickle=False)
        if coarse.shape != (23, 11, 7) or not np.isfinite(coarse).all() or np.any(coarse <= 0):
            raise ValueError(f'Expected a 23x11x7 model of 1-km cells in m/s: {path}')
        return coarse.repeat(8, 0).repeat(8, 1).repeat(8, 2)
    # tomoDD: nodal Vp (MOD axes, Vp_model.dat), linear interpolation at the 125-m cell centres.
    from scipy.interpolate import RegularGridInterpolator
    lines = (path / 'MOD').read_text().splitlines()
    nx, ny, nz = map(int, lines[0].split()[1:])
    lon, lat, depth = (np.array(line.split(), float) for line in lines[1:4])
    native = np.array((path / 'Vp_model.dat').read_text().split(), float).reshape(nz, ny, nx)
    g = json.loads(GEOMETRY.read_text())
    x, y, z = np.meshgrid(*[(np.arange(n) + .5) * H / 1000 for n in SHAPE], indexing='ij')
    ox, oy = np.asarray(g['geographic_reference_in_new_xy_m']) / 1000
    angle = np.deg2rad(g['x_azimuth_degrees'])
    north = (x - ox) * np.cos(angle) - (y - oy) * np.sin(angle)
    east = (x - ox) * np.sin(angle) + (y - oy) * np.cos(angle)
    latitude = g['reference_latitude'] + north / 110.574
    longitude = g['reference_longitude'] + east / (111.320 * np.cos(np.deg2rad(g['reference_latitude'])))
    points = np.stack((z - g['top_elevation_m'] / 1000, latitude, longitude), axis=-1)
    return RegularGridInterpolator((depth, lat, lon), native, bounds_error=True)(points) * 1000


def safe_refine(fields, observed, cell, noise, selected):
    """Powell refinement; bounded coordinate descent if Powell fails on a degenerate direction."""
    from tomography.tomography_math import refine_hypocentre_in_cell
    try:
        position, value = refine_hypocentre_in_cell(fields, observed, cell, noise, station_indices=selected)
        return position, value, False
    except ValueError as error:
        if str(error) != 'zero-size array to reduction operation maximum which has no identity':
            raise
    from scipy.optimize import minimize_scalar
    from instruments.hypocentre_objective import make_refinement_objective
    from instruments.instruments_coords import cell_coord_bounds
    objective = make_refinement_objective(fields, observed, noise, station_indices=selected)
    bounds = cell_coord_bounds(tuple(int(v) for v in cell), fields.shape[1:])
    position = np.asarray(cell, dtype=float)
    value = float(objective(position))
    for _ in range(4):
        previous = value
        for axis, (lo, hi) in enumerate(bounds):
            if hi <= lo:
                continue

            def scalar(x):
                point = position.copy()
                point[axis] = x
                return float(objective(point))
            result = minimize_scalar(scalar, bounds=(lo, hi), method='bounded', options={'xatol': 1e-4})
            for x in (lo, hi, result.x):
                candidate = scalar(x)
                if np.isfinite(candidate) and candidate < value:
                    position[axis], value = x, candidate
        if previous - value <= 1e-10 * max(1, abs(previous)):
            break
    return position, value, True


def fit_event(index):
    from instruments.likelihood import negative_log_likelihood
    from instruments.instruments_coords import sample_cell_centered_trilinear_batch
    from tomography.tomography_math import select_candidate_cells
    fields, observed, noise = _STATE['fields'], _STATE['train'][index], _STATE['noise']
    selected = np.flatnonzero(np.isfinite(observed))
    cost = negative_log_likelihood(fields, observed[selected], noise, station_indices=selected)
    candidates = select_candidate_cells(cost, SHORTLIST, DMIN)
    del cost
    refined = [safe_refine(fields, observed[selected], cell, noise, selected) for cell in candidates]
    position, nll, _ = min(refined, key=lambda item: item[1])
    times = sample_cell_centered_trilinear_batch(fields, position)
    sigma = noise.sigmas(times[selected])
    origin = float(np.average(observed[selected] - times[selected], weights=1 / sigma ** 2))
    return index, position, times + origin


def split_mask(block, event_ids, station_ids):
    roles = {}
    with (HERE / 'evaluation_splits.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            if r['block'] == block:
                roles[(r['event_id'], r['station_id'])] = r['role'] == 'prediction'
    mask = np.zeros((len(event_ids), len(station_ids)), bool)
    for i, e in enumerate(event_ids.astype(str)):
        for j, s in enumerate(station_ids.astype(str)):
            mask[i, j] = roles.get((e, s), False)
    return mask, len(roles)


def evaluate(block, out, models, workers):
    from eikonal import station_travel_time_fields
    from velocity_model import VelocityModel
    from instruments.likelihood import PickNoise, negative_log_likelihood
    from tomography.tomography_math import refine_hypocentre_in_cell
    with np.load(out / 'withheld.npz', allow_pickle=False) as a:
        observed, event_ids, station_ids = a['arrival_times_s'], a['event_ids'], a['station_ids']
        stations = a['station_coordinates_m']
    prediction, n_roles = split_mask(block, event_ids, station_ids)
    assert n_roles == int(np.isfinite(observed).sum()), 'Split does not cover the withheld picks'
    train = observed.copy()
    train[prediction] = np.nan
    noise = PickNoise(*NOISE)
    toy = np.zeros((3, 2, 2, 2), dtype=np.float32)
    negative_log_likelihood(toy, np.zeros(3), noise, station_indices=np.arange(3))
    refine_hypocentre_in_cell(toy, np.zeros(3), (0, 0, 0), noise, station_indices=np.arange(3))
    (out / 'evaluation').mkdir(exist_ok=True)
    for spec in models:
        name, _, source = spec.partition('=')
        started = time.monotonic()
        fields = station_travel_time_fields(VelocityModel(fine_velocity(source), H), stations, workers)
        _STATE.update(fields=fields, train=train, noise=noise)
        predicted = np.empty_like(train)
        positions = np.empty((len(event_ids), 3))
        with mp.get_context('fork').Pool(workers) as pool:
            for i, position, times in pool.imap_unordered(fit_event, range(len(event_ids))):
                predicted[i], positions[i] = times, position
        _STATE.clear()
        np.savez_compressed(out / 'evaluation' / f'{name}.npz', predicted_relative_s=predicted,
                            position_m=(positions + .5) * H, prediction_mask=prediction, observed_s=observed,
                            event_ids=event_ids, station_ids=station_ids, source=source)
        r = (predicted - observed)[prediction]
        print(f'block {block} {name}: prediction RMSE {1000 * np.sqrt(np.mean(r ** 2)):.2f} ms '
              f'({prediction.sum()} picks, {time.monotonic() - started:.0f} s)', flush=True)


def score(work):
    rng = np.random.default_rng(BOOT_SEED)
    pooled = {}
    for block in 'ABCD':
        directory = work / block / 'evaluation'
        if not directory.is_dir():
            continue
        sq, counts = {}, None
        for path in sorted(directory.glob('*.npz')):
            with np.load(path, allow_pickle=False) as a:
                mask = a['prediction_mask']
                sq[path.stem] = np.where(mask, (a['predicted_relative_s'] - a['observed_s']) ** 2, 0).sum(1)
                counts = mask.sum(1)
        draws = rng.integers(0, len(counts), size=(N_BOOT, len(counts)))
        n = counts[draws].sum(1)
        rmse = {m: np.sqrt(v.sum() / counts.sum()) for m, v in sq.items()}
        boot = {m: np.sqrt(v[draws].sum(1) / n) for m, v in sq.items()}
        names = sorted(sq, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))
        print(f'block {block}: ' + ', '.join(f'{m} {1000 * rmse[m]:.1f} ms' for m in names))
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ci = np.quantile(100 * (boot[b] / boot[a] - 1), [.025, .975])
                print(f'   {b} vs {a}: {100 * (rmse[b] / rmse[a] - 1):+.1f}% (95% {ci[0]:+.1f}, {ci[1]:+.1f})')
        pooled[block] = (sq, counts, draws)
    if len(pooled) > 1:
        names = set.intersection(*(set(v[0]) for v in pooled.values()))
        offset, draws, counts, sq = 0, [], [], {m: [] for m in names}
        for s, c, d in pooled.values():
            draws.append(d + offset)
            offset += len(c)
            counts.append(c)
            for m in names:
                sq[m].append(s[m])
        draws, counts = np.concatenate(draws, 1), np.concatenate(counts)
        sq = {m: np.concatenate(v) for m, v in sq.items()}
        n = counts[draws].sum(1)
        rmse = {m: np.sqrt(v.sum() / counts.sum()) for m, v in sq.items()}
        print('pooled: ' + ', '.join(f'{m} {1000 * v:.1f} ms' for m, v in sorted(rmse.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--block', choices='ABCD', required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('invert')
    p.add_argument('--block', choices='ABCD', required=True)
    p.add_argument('--mode', choices=('hard', 'soft'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    p = sub.add_parser('evaluate')
    p.add_argument('--block', choices='ABCD', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', action='append', required=True, help='NAME=PATH.npy or NAME=tomodd:DIR')
    p.add_argument('--workers', type=int, default=12)
    p = sub.add_parser('score')
    p.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.block, args.out.resolve())
    elif args.command == 'invert':
        invert(args.block, args.mode, args.out.resolve(), args.workers)
    elif args.command == 'evaluate':
        evaluate(args.block, args.out.resolve(), args.model, args.workers)
    else:
        score(args.work.resolve())


if __name__ == '__main__':
    main()

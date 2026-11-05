"""Subset an official MintPy test dataset to a short nearest-neighbour chain. The source is read-only."""
import argparse
import os
import shutil

import h5py
import numpy as np


def copy_subset(src, dst, idx, n_expected, drop_prefix=('unwrapPhase_',)):
    with h5py.File(src, 'r') as fi, h5py.File(dst, 'w') as fo:
        for key, val in fi.attrs.items():
            if key.startswith('mintpy.') or key.startswith('REF_'):
                continue
            fo.attrs[key] = val
        for name, ds in fi.items():
            if name.startswith(drop_prefix):
                continue
            data = ds[idx] if ds.shape and ds.shape[0] == n_expected else ds[()]
            out = fo.create_dataset(name, data=data)
            for key, val in ds.attrs.items():
                out.attrs[key] = val
        if 'dropIfgram' in fo:
            fo['dropIfgram'][...] = True


parser = argparse.ArgumentParser()
parser.add_argument('src')
parser.add_argument('dst')
parser.add_argument('-n', type=int, required=True)
args = parser.parse_args()

src_stack = f'{args.src}/inputs/ifgramStack.h5'
with h5py.File(src_stack, 'r') as f:
    date12 = f['date'][:].astype(str)
    all_dates = sorted(set(date12.ravel().tolist()))
    next_date = {d: all_dates[i + 1] for i, d in enumerate(all_dates[:-1])}
    nn = [i for i, (m, s) in enumerate(date12) if next_date.get(m) == s]
    coh = {i: float(np.nanmean(f['coherence'][i, ::8, ::8])) for i in nn}

# best chain of n consecutive nearest-neighbour pairs
by_ref = {date12[i][0]: i for i in nn}
best, best_score = None, -1.0
for start in all_dates:
    chain, ref = [], start
    while len(chain) < args.n and ref in by_ref:
        chain.append(by_ref[ref])
        ref = date12[by_ref[ref]][1]
    if len(chain) == args.n:
        score = sum(coh[i] for i in chain)
        if score > best_score:
            best, best_score = chain, score
if best is None:
    raise SystemExit(f'no nearest-neighbour chain of {args.n} pairs')

idx = sorted(best)
os.makedirs(f'{args.dst}/inputs', exist_ok=True)
copy_subset(src_stack, f'{args.dst}/inputs/ifgramStack.h5', idx, date12.shape[0])
for name in os.listdir(f'{args.src}/inputs'):
    if name.startswith('geometry') and name.endswith('.h5'):
        shutil.copy2(f'{args.src}/inputs/{name}', f'{args.dst}/inputs/{name}')

keep_dates = sorted(set(date12[idx].ravel().tolist()))
era5 = f'{args.src}/inputs/ERA5.h5'
if os.path.isfile(era5):
    with h5py.File(era5, 'r') as f:
        era5_dates = f['date'][:].astype(str)
        eidx = [i for i, d in enumerate(era5_dates) if d in keep_dates]
    if len(eidx) == len(keep_dates):
        copy_subset(era5, f'{args.dst}/inputs/ERA5.h5', eidx, len(era5_dates))
print('pairs', date12[idx].tolist(), 'mean coh', round(best_score / args.n, 3))

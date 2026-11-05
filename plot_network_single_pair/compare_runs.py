"""Compare HDF5 datasets and text products between two run directories."""
import glob
import os
import sys

import h5py
import numpy as np


def datasets(path):
    out = {}
    with h5py.File(path, 'r') as f:
        f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    return out


def compare(a, b):
    n_same, diffs = 0, []
    h5s = sorted(os.path.relpath(p, a) for p in glob.glob(f'{a}/**/*.h5', recursive=True))
    for rel in h5s:
        if rel.startswith('inputs/ERA5') or not os.path.exists(f'{b}/{rel}'):
            if not os.path.exists(f'{b}/{rel}'):
                diffs.append(f'missing {rel}')
            continue
        da, db = datasets(f'{a}/{rel}'), datasets(f'{b}/{rel}')
        same = set(da) == set(db) and all(
            np.array_equal(da[k], db[k], equal_nan=da[k].dtype.kind == 'f') for k in da)
        if same:
            n_same += 1
        else:
            diffs.append(rel)
    n_txt = 0
    for p in sorted(glob.glob(f'{a}/*.txt')):
        rel = os.path.basename(p)
        if rel in ('case.txt',) or not os.path.exists(f'{b}/{rel}'):
            continue
        if open(p).read() == open(f'{b}/{rel}').read():
            n_txt += 1
        else:
            diffs.append(rel)
    return n_same, n_txt, diffs


for pair in sys.argv[1:]:
    a, b = pair.split(':')
    n, t, d = compare(a, b)
    print(f'{os.path.relpath(a, os.path.dirname(os.path.dirname(a)))} vs {os.path.basename(b)}: h5 identical={n} txt identical={t} differ={d}')

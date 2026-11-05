"""Compare one-pair products against closed-form expectations, and P3 against upstream for n >= 2."""
import glob
import os
import sys

import h5py
import numpy as np

R = sys.argv[1]


def rd(path, name):
    with h5py.File(path) as f:
        return f[name][:], dict(f.attrs)


for run in ('K1_P3', 'B1_P3'):
    d = f'{R}/{run}'
    print(f'===== {run}')
    unw, sa = rd(f'{d}/inputs/ifgramStack.h5', 'unwrapPhase')
    unw = unw[0]
    with h5py.File(f'{d}/inputs/ifgramStack.h5') as f:
        d12 = f['date'][0].astype(str)
    ry, rx = int(sa['REF_Y']), int(sa['REF_X'])
    wvl = float(sa['WAVELENGTH'])
    dt = (np.datetime64(f'{d12[1][:4]}-{d12[1][4:6]}-{d12[1][6:]}') - np.datetime64(f'{d12[0][:4]}-{d12[0][4:6]}-{d12[0][6:]}')).astype(int) / 365.25
    ts_raw, _ = rd(f'{d}/timeseries.h5', 'timeseries')
    disp = ts_raw[1] - ts_raw[0]
    print('timeseries.h5 shape', ts_raw.shape, 'dt[yr]', round(dt, 4))
    exp = (unw - unw[ry, rx]) * (-wvl / (4 * np.pi))
    mask = (disp != 0) & np.isfinite(disp)
    print(f'  ts_raw vs expected: valid px {mask.sum()}/{mask.size}, max abs diff {np.abs(disp-exp)[mask].max():.3e} m, rms disp {np.sqrt(np.mean(exp[mask]**2)):.4f} m')
    tc, _ = rd(f'{d}/temporalCoherence.h5', 'temporalCoherence')
    print('  temporalCoherence unique on valid:', np.unique(np.round(tc[mask], 6))[:5])
    with h5py.File(f'{d}/velocity.h5') as f:
        vel = f['velocity'][:]
        vstd = f['velocityStd'][:] if 'velocityStd' in f else None
        res = f['residue'][:] if 'residue' in f else None
        print('  velocity.h5 attrs: timeseries src', f.attrs.get('DATE12'), 'datasets', list(f.keys()))
    tsfiles = sorted(glob.glob(f'{d}/timeseries*.h5'), key=os.path.getmtime)
    print('  ts files:', [os.path.basename(x) for x in tsfiles])
    final = sorted(glob.glob(f'{d}/timeseries*demErr*.h5') or tsfiles, key=os.path.getmtime)[-1]
    ts_final, a = rd(final, 'timeseries')
    v = (ts_final[1] - ts_final[0]) / dt
    m2 = np.isfinite(vel) & (vel != 0)
    print(f'  velocity vs (d1-d0)/dt from {os.path.basename(final)}: max abs diff {np.abs(vel - v)[m2].max():.3e} m/yr')
    if vstd is not None:
        fin = np.isfinite(vstd[m2])
        print(f'  velocityStd on valid px: finite {fin.sum()}/{fin.size}, unique(first 4) {np.unique(vstd[m2][fin])[:4]}, nan {np.isnan(vstd[m2]).sum()}, inf {np.isinf(vstd[m2]).sum()}')
    if res is not None:
        print(f'  residue on valid px: max {np.nanmax(np.abs(res[m2])):.3e}')
    if os.path.isfile(f'{d}/demErr.h5'):
        de, _ = rd(f'{d}/demErr.h5', 'dem')
        m3 = np.isfinite(de) & (de != 0)
        print(f'  demErr.h5: valid {m3.sum()}/{m3.size}, abs p50/p95/max = {np.percentile(np.abs(de[m3]),50):.3f}/{np.percentile(np.abs(de[m3]),95):.3f}/{np.abs(de[m3]).max():.3f} m')
        d1 = ts_final[1]-ts_final[0]; print(f'  disp rms: raw {np.sqrt(np.nanmean(disp[m3]**2)):.4f} m -> final {np.sqrt(np.nanmean(d1[m3]**2)):.4f} m, rms(final-raw) {np.sqrt(np.nanmean((d1-disp)[m3]**2)):.4f} m')
    else:
        print('  demErr.h5 absent')

print('===== P3 vs upstream on multi-pair stacks')
for ds in 'KB':
    for n in (2, 3):
        a, b = f'{R}/{ds}{n}_up', f'{R}/{ds}{n}_P3'
        for name, key in (('timeseries.h5', 'timeseries'), ('velocity.h5', 'velocity'), ('velocity.h5', 'velocityStd'), ('temporalCoherence.h5', 'temporalCoherence')):
            try:
                x, _ = rd(f'{a}/{name}', key)
                y, _ = rd(f'{b}/{name}', key)
                print(f'{ds}{n} {name}:{key:<18} identical={np.array_equal(x, y, equal_nan=True)} maxdiff={np.nanmax(np.abs(x-y)):.2e}')
            except Exception as e:
                print(f'{ds}{n} {name}:{key} ERR {e}')

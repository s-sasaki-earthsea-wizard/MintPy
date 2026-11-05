"""Non-regression on the complete official run, and scientific sanity of the one-pair products."""
import glob
import os
import sys

import h5py
import numpy as np

S = sys.argv[1]
fc, tr, op = f'{S}/runs/fullctl', f'{S}/runs/triage', f'{S}/runs/options'


def arrays(path):
    out = {}
    with h5py.File(path) as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in 'fiub':
                out[name] = obj[()]
        f.visititems(visit)
    return out


print('=== Ffull: upstream vs all three changes (every root-level and geo product)')
a, b = f'{fc}/Ffull_up', f'{tr}/Ffull_P3'
names = sorted({os.path.relpath(p, a) for p in glob.glob(f'{a}/*.h5') + glob.glob(f'{a}/geo/*.h5') + glob.glob(f'{a}/*.he5')})
same = diff = 0
for n in names:
    if not os.path.isfile(f'{b}/{n}'):
        print('  only in up:', n)
        continue
    x, y = arrays(f'{a}/{n}'), arrays(f'{b}/{n}')
    keys_ok = set(x) == set(y)
    eq = keys_ok and all(np.array_equal(x[k], y[k], equal_nan=True) for k in x)
    if eq:
        same += 1
    else:
        diff += 1
        bad = [k for k in x if k in y and not np.array_equal(x[k], y[k], equal_nan=True)]
        print('  DIFF', n, bad[:4], [float(np.nanmax(np.abs(x[k].astype(float) - y[k].astype(float)))) for k in bad[:4]])
print(f'  products compared {len(names)}: identical {same}, different {diff}')
for t in ('coherenceSpatialAvg.txt',):
    print(' ', t, 'identical:', open(f'{a}/{t}').read() == open(f'{b}/{t}').read())

print('=== DEM error: one pair vs complete network, same dataset (Fernandina)')
de_full = arrays(f'{a}/demErr.h5')['dem']
mf = np.isfinite(de_full) & (de_full != 0)
print(f'  full network: |dz| p50/p95 = {np.percentile(np.abs(de_full[mf]),50):.2f}/{np.percentile(np.abs(de_full[mf]),95):.2f} m')
for case, base in (('F1_topo_default', op), ('F1_topo_poly1', op), ('B1_topo_default', op)):
    de = arrays(f'{base}/{case}/demErr.h5')['dem']
    m = np.isfinite(de) & (de != 0)
    with h5py.File(f'{base}/{case}/timeseries.h5') as f:
        t0 = f['timeseries'][:]
    with h5py.File(f'{base}/{case}/timeseries_demErr.h5') as f:
        t1 = f['timeseries'][:]
    d0, d1 = t0[1] - t0[0], t1[1] - t1[0]
    print(f'  {case}: |dz| p50/p95/max = {np.percentile(np.abs(de[m]),50):.4f}/{np.percentile(np.abs(de[m]),95):.4f}/{np.abs(de[m]).max():.4f} m; rms(disp change) = {np.sqrt(np.nanmean((d1-d0)[m]**2)):.2e} m on disp rms {np.sqrt(np.nanmean(d0[m]**2)):.4f} m')
for case in ('F2_off_up', 'F3_off_up', 'F5_off_up'):
    p = f'{op}/{case}/demErr.h5'
    if os.path.isfile(p):
        de = arrays(p)['dem']
        m = np.isfinite(de) & (de != 0)
        print(f'  {case} (upstream): |dz| p50/p95 = {np.percentile(np.abs(de[m]),50):.3f}/{np.percentile(np.abs(de[m]),95):.3f} m')

print('=== closed-form check on the workflow outputs')
for case, base in ((f'F1_baseline', op), ('K1_baseline', op), ('B1_baseline', op), ('Sx1_P3', f'{S}/runs/sanfran'), ('raw1_P3', tr), ('K5_keep1_P3', op), ('F5_keep1_P3', op)):
    d = f'{base}/{case}'
    with h5py.File(f'{d}/inputs/ifgramStack.h5') as f:
        drop = f['dropIfgram'][:]
        i = int(np.where(drop)[0][0])
        ds_name = 'unwrapPhase_bridging' if 'unwrapPhase_bridging' in f else 'unwrapPhase'
        unw = f[ds_name][i]
        d12 = f['date'][i].astype(str)
        ry, rx, wvl = int(f.attrs['REF_Y']), int(f.attrs['REF_X']), float(f.attrs['WAVELENGTH'])
    exp = (unw - unw[ry, rx]) * (-wvl / (4 * np.pi))
    with h5py.File(f'{d}/timeseries.h5') as f:
        ts, tdates = f['timeseries'][:], f['date'][:].astype(str).tolist()
    disp = ts[1] - ts[0]
    m = (disp != 0) & np.isfinite(disp)
    dt = (np.datetime64(f'{d12[1][:4]}-{d12[1][4:6]}-{d12[1][6:]}') - np.datetime64(f'{d12[0][:4]}-{d12[0][4:6]}-{d12[0][6:]}')).astype(int) / 365.25
    with h5py.File(f'{d}/temporalCoherence.h5') as f:
        tc = f['temporalCoherence'][:]
    line = f'  {case:<14} kept {d12.tolist()} ts dates {len(tdates)} valid {m.sum()}/{m.size} |ts-exp|max {np.abs(disp-exp)[m].max():.1e} m; tempCoh on valid = {np.unique(np.round(tc[m],5))[:3]}'
    if os.path.isfile(f'{d}/velocity.h5'):
        with h5py.File(f'{d}/velocity.h5') as f:
            v, vs = f['velocity'][:], f['velocityStd'][:]
        finals = sorted(glob.glob(f'{d}/timeseries*.h5'), key=os.path.getmtime)
        finals = [x for x in finals if 'Residual' not in x]
        with h5py.File(finals[-1]) as f:
            t2 = f['timeseries'][:]
        line += f'; |v-(d/dt)|max {np.nanmax(np.abs(v - (t2[1]-t2[0])/dt)[m]):.1e} m/yr; velocityStd NaN {np.isnan(vs[m]).mean():.0%}'
    print(line)
print('=== plots written on one pair'); 
for c in ('F1_plot', 'K1_plot', 'B1_plot'):
    print(' ', c, len(glob.glob(f'{op}/{c}/pic/*')), 'files; Traceback in log:', open(f'{op}/{c}/out.log').read().count('Traceback'))

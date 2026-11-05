"""Export an official Kuju (ROI_PAC) sub-stack back to ROI_PAC binary products for a real load_data run."""
import os
import sys

import h5py
import numpy as np

base, out = sys.argv[1], sys.argv[2]
src_stack = f'{base}/inputs/ifgramStack.h5'
with h5py.File(src_stack) as f:
    meta = dict(f.attrs)
    date12 = f['date'][:].astype(str)
    unw, cor, bperp = f['unwrapPhase'][:], f['coherence'][:], f['bperp'][:]
length, width = unw.shape[1:]

def rsc(path, extra):
    keep = ['WAVELENGTH', 'RANGE_PIXEL_SIZE', 'AZIMUTH_PIXEL_SIZE', 'EARTH_RADIUS', 'HEIGHT', 'STARTING_RANGE',
            'ALOOKS', 'RLOOKS', 'PLATFORM', 'ORBIT_DIRECTION', 'CENTER_LINE_UTC', 'HEADING', 'LAT_REF1', 'LON_REF1',
            'LAT_REF2', 'LON_REF2', 'LAT_REF3', 'LON_REF3', 'LAT_REF4', 'LON_REF4', 'PRF', 'ANTENNA_SIDE']
    d = {k: meta[k] for k in keep if k in meta}
    d.update(WIDTH=width, FILE_LENGTH=length, XMIN=0, XMAX=width - 1, YMIN=0, YMAX=length - 1, **extra)
    with open(path, 'w') as fp:
        fp.writelines(f'{k:<40}{v}\n' for k, v in d.items())

for i, (m, s) in enumerate(date12):
    d6 = f'{m[2:]}-{s[2:]}'
    idir = f'{out}/interferograms/{d6}'
    os.makedirs(idir, exist_ok=True)
    stem = f'{idir}/filt_{d6}-sim_HDR_4rlks_c10'
    for ext, arr in (('.unw', unw[i]), ('.cor', cor[i])):
        rmg = np.zeros((length * 2, width), dtype=np.float32)
        rmg[0::2] = np.abs(arr)
        rmg[1::2] = arr
        rmg.tofile(stem + ext)
        rsc(stem + ext + '.rsc', dict(DATE12=d6))
    with open(f'{idir}/{m[2:]}_{s[2:]}_baseline.rsc', 'w') as fp:
        fp.write(f"{'P_BASELINE_TOP_HDR':<40}{bperp[i]}\n{'P_BASELINE_BOTTOM_HDR':<40}{bperp[i]}\n")

# radar-coded DEM + lookup table from the official geometry files
os.makedirs(f'{out}/geom', exist_ok=True)
with h5py.File(f'{base}/inputs/geometryRadar.h5') as f:
    hgt = f['height'][:].astype(np.float32)
rmg = np.zeros((length * 2, width), dtype=np.float32)
rmg[0::2], rmg[1::2] = hgt, hgt
rmg.tofile(f'{out}/geom/radar_4rlks.hgt')
rsc(f'{out}/geom/radar_4rlks.hgt.rsc', {})
with h5py.File(f'{base}/inputs/geometryGeo.h5') as f:
    rc, ac = f['rangeCoord'][:].astype(np.float32), f['azimuthCoord'][:].astype(np.float32)
    gmeta = dict(f.attrs)
gl, gw = rc.shape
trans = np.zeros((gl * 2, gw), dtype=np.float32)
trans[0::2], trans[1::2] = rc, ac
trans.tofile(f'{out}/geom/geomap_4rlks.trans')
with open(f'{out}/geom/geomap_4rlks.trans.rsc', 'w') as fp:
    for k, v in dict(WIDTH=gw, FILE_LENGTH=gl, X_FIRST=gmeta['X_FIRST'], Y_FIRST=gmeta['Y_FIRST'], X_STEP=gmeta['X_STEP'],
                     Y_STEP=gmeta['Y_STEP'], X_UNIT=gmeta['X_UNIT'], Y_UNIT=gmeta['Y_UNIT'],
                     RANGE_PIXEL_SIZE=meta['RANGE_PIXEL_SIZE'], AZIMUTH_PIXEL_SIZE=meta['AZIMUTH_PIXEL_SIZE'],
                     Y_REF1=0, X_REF1=0).items():
        fp.write(f'{k:<40}{v}\n')
cfg = f'''mintpy.load.processor = roipac
mintpy.load.unwFile = {out}/interferograms/*/filt_*.unw
mintpy.load.corFile = {out}/interferograms/*/filt_*.cor
mintpy.load.demFile = {out}/geom/radar_4rlks.hgt
mintpy.load.lookupYFile = {out}/geom/geomap_4rlks.trans
mintpy.load.lookupXFile = {out}/geom/geomap_4rlks.trans
mintpy.reference.lalo = auto
'''
open(f'{out}/load.txt', 'w').write(cfg)

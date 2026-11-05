"""Geocoded 1-pair stack (40x40) with a mild ramp, for pipeline-depth checks."""
import os

import h5py
import numpy as np

length, width = 40, 40
geo = dict(X_FIRST=140.0, Y_FIRST=35.0, X_STEP=0.001, Y_STEP=-0.001, X_UNIT='degrees', Y_UNIT='degrees',
           WAVELENGTH=0.238, PLATFORM='NISAR', ORBIT_DIRECTION='ASCENDING', ALOOKS=1, RLOOKS=1,
           CENTER_LINE_UTC=43200.0, HEADING=-10.0, EARTH_RADIUS=6371000., HEIGHT=747000.,
           STARTING_RANGE=800000., RANGE_PIXEL_SIZE=10., AZIMUTH_PIXEL_SIZE=10.)
yy, xx = np.mgrid[0:length, 0:width]
phase = (0.02 * xx + 0.01 * yy).astype(np.float32)
os.makedirs('inputs', exist_ok=True)
with h5py.File('inputs/ifgramStack.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='ifgramStack', LENGTH=length, WIDTH=width, **geo)
    f['date'] = np.array([['20260101', '20260113']], dtype='S8')
    f['bperp'] = np.array([10.], dtype=np.float32)
    f['dropIfgram'] = np.ones(1, dtype=np.bool_)
    f['unwrapPhase'] = phase[None]
    f['coherence'] = np.full((1, length, width), 0.9, dtype=np.float32)
    f['connectComponent'] = np.ones((1, length, width), dtype=np.int16)
with h5py.File('inputs/geometryGeo.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='geometry', LENGTH=length, WIDTH=width, **geo)
    for name, val in dict(height=100., incidenceAngle=35., slantRangeDistance=8e5, azimuthAngle=-100.).items():
        f[name] = np.full((length, width), val, dtype=np.float32)

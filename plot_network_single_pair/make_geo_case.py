"""Write a geocoded 1-pair stack + geometry under inputs/, mimicking NISAR GUNW layout."""
import os

import h5py
import numpy as np

length, width = 5, 5
geo = dict(X_FIRST=140.0, Y_FIRST=35.0, X_STEP=0.001, Y_STEP=-0.001, X_UNIT='degrees', Y_UNIT='degrees')
os.makedirs('inputs', exist_ok=True)
with h5py.File('inputs/ifgramStack.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='ifgramStack', LENGTH=length, WIDTH=width, **geo)
    f['date'] = np.array([['20260101', '20260113']], dtype='S8')
    f['bperp'] = np.array([10.], dtype=np.float32)
    f['dropIfgram'] = np.ones(1, dtype=np.bool_)
    f['unwrapPhase'] = np.zeros((1, length, width), dtype=np.float32)
    f['coherence'] = np.full((1, length, width), 0.8, dtype=np.float32)
    f['connectComponent'] = np.ones((1, length, width), dtype=np.int16)
with h5py.File('inputs/geometryGeo.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='geometry', LENGTH=length, WIDTH=width, **geo)
    for name, val in dict(height=100., incidenceAngle=35., slantRangeDistance=8e5).items():
        f[name] = np.full((length, width), val, dtype=np.float32)

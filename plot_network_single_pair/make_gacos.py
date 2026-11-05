"""Write smooth synthetic GACOS zenith delays for the dates in inputs/ifgramStack.h5."""
import os

import h5py
import numpy as np

with h5py.File('inputs/ifgramStack.h5') as f:
    dates = sorted(set(f['date'][:].astype(str).ravel().tolist()))
os.makedirs('GACOS', exist_ok=True)
y0, x0, dy, dx, n_row, n_col = -0.10, -91.80, -0.01, 0.01, 60, 60
yy, xx = np.mgrid[0:n_row, 0:n_col]
for i, d in enumerate(dates):
    ztd = (2.30 + 0.0005 * yy + 0.0003 * xx + 0.01 * i).astype(np.float32)
    ztd.tofile(f'GACOS/{d}.ztd')
    with open(f'GACOS/{d}.ztd.rsc', 'w') as fp:
        for k, v in dict(WIDTH=n_col, FILE_LENGTH=n_row, X_FIRST=x0, Y_FIRST=y0, X_STEP=dx, Y_STEP=dy, X_UNIT='degrees', Y_UNIT='degrees').items():
            fp.write(f'{k:<20}{v}\n')

"""Write a small synthetic ifgramStack.h5 with N sequential pairs."""
import datetime as dt
import sys

import h5py
import numpy as np

n_pair = int(sys.argv[1])
with_coh = (len(sys.argv) < 3 or sys.argv[2] != 'nocoh')
length, width = 5, 5

dates = [(dt.date(2026, 1, 1) + dt.timedelta(days=12 * i)).strftime('%Y%m%d') for i in range(n_pair + 1)]
date12 = [[dates[i], dates[i + 1]] for i in range(n_pair)]

with h5py.File('ifgramStack.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='ifgramStack', LENGTH=length, WIDTH=width)
    f['date'] = np.array(date12, dtype='S8')
    f['bperp'] = np.linspace(10, 10 + 5 * (n_pair - 1), n_pair, dtype=np.float32)
    f['dropIfgram'] = np.ones(n_pair, dtype=np.bool_)
    f['unwrapPhase'] = np.zeros((n_pair, length, width), dtype=np.float32)
    if with_coh:
        f['coherence'] = np.full((n_pair, length, width), 0.8, dtype=np.float32)

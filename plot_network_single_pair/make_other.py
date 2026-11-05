"""Write a 2D velocity.h5 and a 1-date timeseries.h5 to exercise other file types."""
import h5py
import numpy as np

length, width = 5, 5
with h5py.File('velocity.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='velocity', LENGTH=length, WIDTH=width)
    f['velocity'] = np.full((length, width), 0.01, dtype=np.float32)
with h5py.File('timeseries1.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='timeseries', LENGTH=length, WIDTH=width, REF_DATE='20260101')
    f['date'] = np.array(['20260101'], dtype='S8')
    f['bperp'] = np.array([0.], dtype=np.float32)
    f['timeseries'] = np.full((1, length, width), 0.01, dtype=np.float32)

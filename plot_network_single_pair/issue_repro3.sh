mkdir repro && cd repro

python - <<'PY'
import h5py, numpy as np

n = 1  # number of interferograms; n = 2 runs fine
dates = ['20260101', '20260113', '20260125'][:n + 1]
atr = dict(LENGTH=5, WIDTH=5, WAVELENGTH=0.056, REF_Y=0, REF_X=0)
phase = np.arange(1, 26, dtype=np.float32).reshape(5, 5)

with h5py.File('ifgramStack.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='ifgramStack', **atr)
    f['date'] = np.array(list(zip(dates[:-1], dates[1:])), dtype='S8')
    f['bperp'] = np.arange(n, dtype=np.float32) + 10
    f['dropIfgram'] = np.ones(n, dtype=np.bool_)
    f['unwrapPhase'] = np.repeat(phase[None], n, axis=0)
    f['coherence'] = np.full((n, 5, 5), 0.8, dtype=np.float32)

with h5py.File('timeseries.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='timeseries', REF_DATE=dates[0], **atr)
    f['date'] = np.array(dates, dtype='S8')
    f['bperp'] = np.zeros(n + 1, dtype=np.float32)
    f['timeseries'] = np.array([phase * i for i in range(n + 1)], dtype=np.float32)
PY

plot_network.py ifgramStack.h5 --nodisplay; echo "plot_network exit=$?"
ifgram_inversion.py ifgramStack.h5 -w no -o ts_inv.h5 tcoh_inv.h5 num_inv.h5; echo "ifgram_inversion exit=$?"
timeseries2velocity.py timeseries.h5 -o vel.h5; echo "timeseries2velocity exit=$?"

mkdir repro && cd repro

python - <<'PY'
import h5py, numpy as np

n = 1  # number of interferograms; with n = 2 this runs fine
dates = [['20260101', '20260113'], ['20260113', '20260125']][:n]
with h5py.File('ifgramStack.h5', 'w') as f:
    f.attrs.update(FILE_TYPE='ifgramStack', LENGTH=5, WIDTH=5)
    f['date'] = np.array(dates, dtype='S8')
    f['bperp'] = np.arange(n, dtype=np.float32) + 10
    f['dropIfgram'] = np.ones(n, dtype=np.bool_)
    f['unwrapPhase'] = np.zeros((n, 5, 5), dtype=np.float32)
    f['coherence'] = np.full((n, 5, 5), 0.8, dtype=np.float32)
PY

plot_network.py ifgramStack.h5 --nodisplay; echo "exit=$?"

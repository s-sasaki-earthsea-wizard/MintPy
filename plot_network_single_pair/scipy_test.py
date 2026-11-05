import numpy as np
import scipy
from scipy import linalg
print('scipy', scipy.__version__, 'numpy', np.__version__)
t = 12 / 365.25
for name, G in [
    ('M=N=2 full rank', [[1, 0], [1, t]]),
    ('M=N=3 full rank (poly2)', [[1, 0, 0], [1, t, t**2], [1, 2*t, (2*t)**2]]),
    ('M=3,N=2 full rank', [[1, 0], [1, t], [1, 2*t]]),
    ('M=3,N=2 rank 1 (same date 3x)', [[1, 0], [1, 0], [1, 0]]),
    ('M=N=2 rank 1 (same date 2x)', [[1, 0], [1, 0]]),
]:
    G = np.array(G, float)
    b = np.ones((G.shape[0], 4))
    e2 = linalg.lstsq(G, b, cond=None)[1]
    rank = linalg.lstsq(G, b, cond=None)[2]
    print(f'  {name:<32} rank={rank} e2.size={np.size(e2)} -> upstream {"RAISES rank-deficient" if np.size(e2) == 0 else "accepts"}')

"""Compare a saved kite container with the MintPy input it was built from."""
import sys

import h5py
import numpy as np
from kite.scene import Scene


def main(container, data_file, dset, mask_file, geom_file):
    scene = Scene.load(container)
    with h5py.File(data_file, 'r') as f:
        if dset == 'velocity':
            expected = f['velocity'][()]
        else:
            d1, d2 = dset.split('_')
            dates = [d.decode() for d in f['date'][()]]
            ts = f['timeseries']
            expected = ts[dates.index(d2)] - ts[dates.index(d1)]
    with h5py.File(mask_file, 'r') as f:
        mask = f['mask'][()].astype(bool)
    with h5py.File(geom_file, 'r') as f:
        inc = f['incidenceAngle'][()]
        azi = f['azimuthAngle'][()]
    expected = np.array(expected, dtype=np.float64)
    expected[~mask] = np.nan
    expected = np.flipud(expected)
    got = np.array(scene.displacement, dtype=np.float64)
    same_nan = np.array_equal(np.isnan(got), np.isnan(expected))
    max_diff = np.nanmax(np.abs(got - expected))
    theta_ref = np.flipud(np.pi / 2 - np.deg2rad(inc))
    phi_ref = np.flipud(np.deg2rad(azi)) + np.pi / 2
    fin = np.isfinite(got)
    nleaves = scene.quadtree.nleaves
    print(f'shape={got.shape} valid={int(fin.sum())} same_nan={same_nan} max_diff={max_diff:.3e} '
          f'theta_diff={np.nanmax(np.abs(scene.theta[fin] - theta_ref[fin])):.3e} '
          f'phi_diff={np.nanmax(np.abs(scene.phi[fin] - phi_ref[fin])):.3e} '
          f'd_range=[{np.nanmin(got):.4f},{np.nanmax(got):.4f}] quadtree_leaves={nleaves} '
          f'times={scene.meta.time_master}..{scene.meta.time_slave} node={scene.meta.orbital_node}')


if __name__ == '__main__':
    main(*sys.argv[1:])

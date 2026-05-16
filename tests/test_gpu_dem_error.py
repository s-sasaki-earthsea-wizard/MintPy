"""Numerical comparison tests for src/mintpy/gpu/dem_error.py.

Compare the batched-Cholesky GPU solver against the per-pixel CPU
reference (``scipy.linalg.lstsq`` via
``mintpy.dem_error.estimate_dem_error``) on small synthetic fixtures.
This file is intentionally lightweight: real-data validation on
FernandinaSenDT128 is a follow-up task. We assert finiteness and shape
only; the RMS / max-abs differences between CPU and GPU outputs are
printed for inspection (use ``pytest -s`` to see them).

Tests are skipped automatically when PyTorch / CUDA are unavailable.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mintpy.dem_error import estimate_dem_error
from mintpy.gpu.dem_error import estimate_dem_error_pixelwise_batch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-capable GPU required for mintpy.gpu.dem_error tests",
)


# ---------------------------------------------------------------------------
# synthetic fixture builders
# ---------------------------------------------------------------------------

def _build_design_matrices(num_date, num_param_defo, seed):
    """Return (G_defo, tbase, pbase_scene) at the scale of a real SBAS run.

    G_defo columns: [vel * t, 0.5 * acc * t^2, ...] truncated at
    num_param_defo. tbase is cumulative random year fractions starting
    at 0. pbase_scene is a (D, 1) perpendicular-baseline column in
    metres in roughly Sentinel-1 range.
    """
    rng = np.random.default_rng(seed)
    dt = rng.uniform(0.05, 0.2, size=num_date - 1).astype(np.float32)
    tbase = np.concatenate([[0.0], np.cumsum(dt)]).astype(np.float32)  # (D,)
    cols = []
    for k in range(num_param_defo):
        # column = t^(k+1) / (k+1)! — polynomial deformation basis
        from math import factorial
        cols.append((tbase ** (k + 1) / factorial(k + 1)).astype(np.float32))
    G_defo = np.stack(cols, axis=1)                                    # (D, P_defo)
    pbase_scene = rng.uniform(-200.0, 200.0, size=(num_date, 1)).astype(np.float32)
    return G_defo, tbase.reshape(-1, 1), pbase_scene


def _build_geometry(num_pixel, seed):
    """Return per-pixel slant range and sin(inc) in roughly Sentinel-1
    swath range (R ~ 700 km, inc ~ 30-45 deg)."""
    rng = np.random.default_rng(seed)
    range_dist = rng.uniform(7.0e5, 7.5e5, size=num_pixel).astype(np.float32)
    inc_deg = rng.uniform(30.0, 45.0, size=num_pixel).astype(np.float32)
    sin_inc = np.sin(np.deg2rad(inc_deg)).astype(np.float32)
    return range_dist, sin_inc


def _synthesize_ts(G_defo, pbase_valid, range_dist, sin_inc_angle,
                   num_pixel, noise_std, seed):
    """Forward-model ts_valid = G0_k @ X_true_k + Gaussian noise."""
    rng = np.random.default_rng(seed)
    num_date, num_param_defo = G_defo.shape
    num_param = 1 + num_param_defo

    # X_true: column 0 is delta_z (metres of DEM error), rest are
    # polynomial deformation coefficients. Scale them so the geometric
    # term and deformation term are comparable in radians.
    X_true = np.zeros((num_param, num_pixel), dtype=np.float32)
    X_true[0, :] = rng.normal(0.0, 5.0, size=num_pixel)        # ~5 m DEM error
    for p in range(1, num_param):
        X_true[p, :] = rng.normal(0.0, 0.01, size=num_pixel)   # small defo amps

    ts = np.zeros((num_date, num_pixel), dtype=np.float32)
    pbase_per_pixel = pbase_valid.shape[1] != 1
    for k in range(num_pixel):
        pbase_k = pbase_valid if not pbase_per_pixel else pbase_valid[:, k:k + 1]
        G_geom = pbase_k / (range_dist[k] * sin_inc_angle[k])  # (D, 1)
        G0 = np.hstack((G_geom, G_defo))                       # (D, P)
        ts[:, k] = (G0 @ X_true[:, k]).astype(np.float32)
    ts += rng.normal(0.0, noise_std, size=ts.shape).astype(np.float32)
    return ts, X_true


def _cpu_reference(ts_valid, pbase_valid, range_dist, sin_inc_angle,
                   G_defo, tbase, date_flag=None):
    """Per-pixel CPU reference mirroring correct_dem_error_patch's
    pixelwise branch (phase_velocity=False).
    """
    num_date, num_pixel = ts_valid.shape
    delta_z = np.zeros(num_pixel, dtype=np.float32)
    ts_cor = np.zeros((num_date, num_pixel), dtype=np.float32)
    ts_res = np.zeros((num_date, num_pixel), dtype=np.float32)
    pbase_per_pixel = pbase_valid.shape[1] != 1
    for k in range(num_pixel):
        pbase_k = pbase_valid if not pbase_per_pixel else pbase_valid[:, k:k + 1]
        G_geom = pbase_k / (range_dist[k] * sin_inc_angle[k])
        G0 = np.hstack((G_geom, G_defo)).astype(np.float32)
        dz, tc, tr = estimate_dem_error(
            ts0=ts_valid[:, k], G0=G0, tbase=tbase,
            date_flag=date_flag, phase_velocity=False,
        )
        delta_z[k] = np.asarray(dz).reshape(-1).item()
        ts_cor[:, k] = np.asarray(tc).flatten()
        ts_res[:, k] = np.asarray(tr).flatten()
    return delta_z, ts_cor, ts_res


# ---------------------------------------------------------------------------
# assertion / comparison helpers
# ---------------------------------------------------------------------------

def _assert_finite_and_shaped(delta_z, ts_cor, ts_res, num_date, num_pixel):
    assert delta_z.shape == (num_pixel,), \
        f'delta_z shape {delta_z.shape} != ({num_pixel},)'
    assert ts_cor.shape == (num_date, num_pixel), \
        f'ts_cor shape {ts_cor.shape} != ({num_date}, {num_pixel})'
    assert ts_res.shape == (num_date, num_pixel), \
        f'ts_res shape {ts_res.shape} != ({num_date}, {num_pixel})'
    assert np.all(np.isfinite(delta_z)), 'delta_z has non-finite entries'
    assert np.all(np.isfinite(ts_cor)), 'ts_cor has non-finite entries'
    assert np.all(np.isfinite(ts_res)), 'ts_res has non-finite entries'


def _print_diff(name, cpu, gpu):
    """Print rms / max-abs diff between CPU and GPU output for ``name``.

    The point is qualitative inspection, not gating; numerical
    equivalence is not asserted in this session.
    """
    diff = (cpu.astype(np.float64) - gpu.astype(np.float64))
    scale = float(np.abs(cpu).max()) if cpu.size else 0.0
    scale_safe = max(scale, 1e-12)
    rms = float(np.sqrt(np.mean(diff ** 2)))
    mx = float(np.abs(diff).max()) if diff.size else 0.0
    print(
        f'  {name:8s}: rms={rms:.3e}  max|diff|={mx:.3e}  '
        f'|cpu|max={scale:.3e}  rms/|cpu|max={rms / scale_safe:.3e}'
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_inputs():
    """Single synthetic fixture shared across tests.

    Sized small (num_pixel=1000) to keep the test fast; FernandinaSenDT128-
    scale validation is a follow-up task.
    """
    num_date = 98
    num_pixel = 1000
    num_param_defo = 2          # vel + acc polynomial basis
    noise_std = 1e-3
    G_defo, tbase, pbase_scene = _build_design_matrices(
        num_date=num_date, num_param_defo=num_param_defo, seed=0,
    )
    range_dist, sin_inc = _build_geometry(num_pixel=num_pixel, seed=1)
    return dict(
        num_date=num_date,
        num_pixel=num_pixel,
        G_defo=G_defo,
        tbase=tbase,
        pbase_scene=pbase_scene,
        range_dist=range_dist,
        sin_inc=sin_inc,
        noise_std=noise_std,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@requires_cuda
def test_pixelwise_scene_mean_pbase(synthetic_inputs, capsys):
    """Pixelwise GPU path with scene-mean baseline (pbase shape (D, 1))."""
    s = synthetic_inputs
    ts, _ = _synthesize_ts(
        G_defo=s['G_defo'], pbase_valid=s['pbase_scene'],
        range_dist=s['range_dist'], sin_inc_angle=s['sin_inc'],
        num_pixel=s['num_pixel'], noise_std=s['noise_std'], seed=2,
    )
    cpu = _cpu_reference(
        ts_valid=ts, pbase_valid=s['pbase_scene'],
        range_dist=s['range_dist'], sin_inc_angle=s['sin_inc'],
        G_defo=s['G_defo'], tbase=s['tbase'],
    )
    gpu = estimate_dem_error_pixelwise_batch(
        ts_valid=ts, pbase_valid=s['pbase_scene'],
        range_dist_valid=s['range_dist'], sin_inc_angle_valid=s['sin_inc'],
        G_defo=s['G_defo'], tbase=s['tbase'],
        phase_velocity=False, solver='torch', print_msg=False,
    )
    _assert_finite_and_shaped(*gpu, num_date=s['num_date'], num_pixel=s['num_pixel'])

    with capsys.disabled():
        print(f"\n[scene-mean pbase (D, 1), num_date={s['num_date']}, "
              f"num_pixel={s['num_pixel']}]")
        _print_diff('delta_z', cpu[0], gpu[0])
        _print_diff('ts_cor',  cpu[1], gpu[1])
        _print_diff('ts_res',  cpu[2], gpu[2])


@requires_cuda
def test_pixelwise_per_pixel_pbase(synthetic_inputs, capsys):
    """Pixelwise GPU path with per-pixel baseline (pbase shape (D, K))."""
    s = synthetic_inputs
    rng = np.random.default_rng(3)
    # Perturb the scene-mean pbase per-pixel to simulate
    # geometryRadar.h5's bperp dataset.
    pbase_per_pixel = (
        s['pbase_scene']
        + rng.uniform(-20.0, 20.0, size=(s['num_date'], s['num_pixel'])).astype(np.float32)
    )
    ts, _ = _synthesize_ts(
        G_defo=s['G_defo'], pbase_valid=pbase_per_pixel,
        range_dist=s['range_dist'], sin_inc_angle=s['sin_inc'],
        num_pixel=s['num_pixel'], noise_std=s['noise_std'], seed=4,
    )
    cpu = _cpu_reference(
        ts_valid=ts, pbase_valid=pbase_per_pixel,
        range_dist=s['range_dist'], sin_inc_angle=s['sin_inc'],
        G_defo=s['G_defo'], tbase=s['tbase'],
    )
    gpu = estimate_dem_error_pixelwise_batch(
        ts_valid=ts, pbase_valid=pbase_per_pixel,
        range_dist_valid=s['range_dist'], sin_inc_angle_valid=s['sin_inc'],
        G_defo=s['G_defo'], tbase=s['tbase'],
        phase_velocity=False, solver='torch', print_msg=False,
    )
    _assert_finite_and_shaped(*gpu, num_date=s['num_date'], num_pixel=s['num_pixel'])

    with capsys.disabled():
        print(f"\n[per-pixel pbase (D, K), num_date={s['num_date']}, "
              f"num_pixel={s['num_pixel']}]")
        _print_diff('delta_z', cpu[0], gpu[0])
        _print_diff('ts_cor',  cpu[1], gpu[1])
        _print_diff('ts_res',  cpu[2], gpu[2])


@requires_cuda
def test_phase_velocity_true_raises(synthetic_inputs):
    """phase_velocity=True is not yet implemented and must raise."""
    s = synthetic_inputs
    ts, _ = _synthesize_ts(
        G_defo=s['G_defo'], pbase_valid=s['pbase_scene'],
        range_dist=s['range_dist'], sin_inc_angle=s['sin_inc'],
        num_pixel=8, noise_std=0.0, seed=5,
    )
    with pytest.raises(NotImplementedError, match='phase_velocity=True'):
        estimate_dem_error_pixelwise_batch(
            ts_valid=ts,
            pbase_valid=s['pbase_scene'],
            range_dist_valid=s['range_dist'][:8],
            sin_inc_angle_valid=s['sin_inc'][:8],
            G_defo=s['G_defo'], tbase=s['tbase'],
            phase_velocity=True, solver='torch', print_msg=False,
        )


@requires_cuda
def test_unsupported_solver_raises(synthetic_inputs):
    s = synthetic_inputs
    ts, _ = _synthesize_ts(
        G_defo=s['G_defo'], pbase_valid=s['pbase_scene'],
        range_dist=s['range_dist'], sin_inc_angle=s['sin_inc'],
        num_pixel=8, noise_std=0.0, seed=6,
    )
    with pytest.raises(ValueError, match='unsupported solver'):
        estimate_dem_error_pixelwise_batch(
            ts_valid=ts,
            pbase_valid=s['pbase_scene'],
            range_dist_valid=s['range_dist'][:8],
            sin_inc_angle_valid=s['sin_inc'][:8],
            G_defo=s['G_defo'], tbase=s['tbase'],
            phase_velocity=False, solver='cupy', print_msg=False,
        )

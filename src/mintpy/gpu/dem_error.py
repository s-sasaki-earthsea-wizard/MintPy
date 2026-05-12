############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# GPU-accelerated DEM error correction                     #
############################################################
# Recommend import:
#     from mintpy.gpu import dem_error as gpu_demerr


"""GPU-batched solver for the pixelwise DEM-error correction.

This module provides ``estimate_dem_error_pixelwise_batch``, an opt-in
replacement for the per-pixel CPU loop in ``correct_dem_error_patch``
(``mintpy.dem_error``). All pixels in a chunk are solved on CUDA via
batched normal equations + cuSolver-batched Cholesky; rank-deficient
pixels are detected via ``cholesky_ex`` info codes and zeroed so NaN/Inf
never propagate.

Compared to the SBAS network inversion solver in
``mintpy.gpu.ifgram_inversion``, the pixelwise DEM-error design matrix
differs per pixel only in its first column (the geometric scaling
``pbase / (range_dist * sin_inc_angle)``). The Phase-1 implementation
materializes the full ``(K, D, P)`` ``G_batch`` for clarity and code
symmetry with the SBAS solver; a Phase-2 optimisation that exploits the
shared structure is left for a follow-up if benchmarks show the naive
form is leaving performance on the table.

Currently ``phase_velocity=True`` is not implemented and will raise. The
production default is ``phase_velocity=False`` (see
``smallbaselineApp.cfg``: ``mintpy.topographicResidual.phaseVelocity =
no``).
"""


import numpy as np

from mintpy.gpu._common import (
    HAS_TORCH,
    SUPPORTED_SOLVERS,
    auto_chunk_size,
    get_torch_device,
    is_solver_available,
    solve_normal_equations_batched,
)

if HAS_TORCH:
    import torch


def estimate_dem_error_pixelwise_batch(
    ts_valid,
    pbase_valid,
    range_dist_valid,
    sin_inc_angle_valid,
    G_defo,
    tbase=None,
    date_flag=None,
    phase_velocity=False,
    chunk_size=None,
    solver='torch',
    print_msg=True,
):
    """Batched GPU solver for the pixelwise DEM-error correction.

    Mirrors the CPU per-pixel loop in ``correct_dem_error_patch`` over
    the ``mask`` of invertable pixels: builds, per pixel ``k``,
        ``G0_k = [pbase_k / (R_k * sinθ_k) | G_defo]``     (D, P)
    where ``P = 1 + G_defo.shape[1]``, then solves
    ``lstsq(G0_k[date_flag], ts_k[date_flag])`` and assembles
    ``ts_cor``/``ts_res`` from the full (un-filtered) ``G0_k``.

    Args:
        ts_valid:           (num_date, num_pixel) float. Time-series for
                            the pixels to invert (already mask-filtered
                            by caller). NaN-free.
        pbase_valid:        (num_date, 1) for scene-mean baseline, or
                            (num_date, num_pixel) for per-pixel baseline
                            (column-aligned with ``ts_valid``).
        range_dist_valid:   (num_pixel,) slant range distance in metres.
        sin_inc_angle_valid:(num_pixel,) sin of incidence angle.
        G_defo:             (num_date, num_param_defo) shared deformation
                            design matrix (polynomial / step / etc).
        tbase:              (num_date,) or (num_date, 1) temporal
                            baseline in years. Only consumed when
                            ``phase_velocity=True`` (currently
                            unsupported). Kept for API parity with the
                            CPU path.
        date_flag:          (num_date,) bool. Dates used for the fit.
                            ``None`` means all dates.
        phase_velocity:     bool. Only ``False`` is currently
                            implemented; ``True`` raises
                            ``NotImplementedError``.
        chunk_size:         int or None. Pixels per GPU chunk. ``None``
                            => auto from free VRAM.
        solver:             str. Only ``'torch'`` is implemented.
        print_msg:          bool.

    Returns:
        delta_z:    (num_pixel,) float32. Estimated DEM residual per
                    pixel (first parameter of X).
        ts_cor:     (num_date, num_pixel) float32. Topography-corrected
                    time-series, ``ts - G0[:, 0] * delta_z``.
        ts_res:     (num_date, num_pixel) float32. Fit residual,
                    ``ts - G0 @ X``.
    """
    if solver != 'torch':
        raise ValueError(
            f"unsupported solver={solver!r}; choose from {SUPPORTED_SOLVERS}"
        )
    if phase_velocity:
        raise NotImplementedError(
            "phase_velocity=True is not yet supported on the torch solver; "
            "fall back to solver='cpu'"
        )
    device = get_torch_device(solver)

    ts_valid = np.asarray(ts_valid, dtype=np.float32)
    if ts_valid.ndim != 2:
        raise ValueError(f'ts_valid must be 2D (num_date, num_pixel); got {ts_valid.shape}')
    num_date, num_pixel = ts_valid.shape

    G_defo_np = np.asarray(G_defo, dtype=np.float32)
    if G_defo_np.shape[0] != num_date:
        raise ValueError(
            f'G_defo first dim {G_defo_np.shape[0]} must match num_date={num_date}'
        )
    num_param_defo = G_defo_np.shape[1]
    num_param = 1 + num_param_defo

    pbase_np = np.asarray(pbase_valid, dtype=np.float32)
    if pbase_np.shape == (num_date,):
        pbase_np = pbase_np.reshape(-1, 1)
    if pbase_np.ndim != 2 or pbase_np.shape[0] != num_date:
        raise ValueError(
            f'pbase_valid must be (num_date, 1) or (num_date, num_pixel); '
            f'got {pbase_np.shape}'
        )
    pbase_per_pixel = pbase_np.shape[1] != 1
    if pbase_per_pixel and pbase_np.shape[1] != num_pixel:
        raise ValueError(
            f'pbase_valid column count {pbase_np.shape[1]} must be 1 or '
            f'num_pixel={num_pixel}'
        )

    range_dist_np = np.asarray(range_dist_valid, dtype=np.float32).reshape(-1)
    sin_inc_angle_np = np.asarray(sin_inc_angle_valid, dtype=np.float32).reshape(-1)
    if range_dist_np.shape != (num_pixel,) or sin_inc_angle_np.shape != (num_pixel,):
        raise ValueError(
            f'range_dist_valid / sin_inc_angle_valid must be 1D of length num_pixel='
            f'{num_pixel}; got {range_dist_np.shape} / {sin_inc_angle_np.shape}'
        )

    if date_flag is None:
        date_flag_np = np.ones(num_date, dtype=np.bool_)
    else:
        date_flag_np = np.asarray(date_flag, dtype=np.bool_).reshape(-1)
        if date_flag_np.shape != (num_date,):
            raise ValueError(
                f'date_flag must be of length num_date={num_date}; '
                f'got {date_flag_np.shape}'
            )

    delta_z = np.zeros(num_pixel, dtype=np.float32)
    ts_cor = np.zeros((num_date, num_pixel), dtype=np.float32)
    ts_res = np.zeros((num_date, num_pixel), dtype=np.float32)

    if num_pixel == 0:
        return delta_z, ts_cor, ts_res

    if chunk_size is None or chunk_size <= 0:
        chunk_size = auto_chunk_size(num_date, num_param)
        if print_msg:
            free_gib = torch.cuda.mem_get_info()[0] / 2**30
            print(f'GPU auto chunk_size = {chunk_size} pixels '
                  f'(free VRAM {free_gib:.1f} GiB)')
    else:
        chunk_size = int(chunk_size)

    num_chunk = (num_pixel + chunk_size - 1) // chunk_size
    if print_msg:
        print(f'estimating DEM error via {solver} batched LSQ '
              f'in {num_chunk} chunk(s) of up to {chunk_size} pixels ...')

    # move shared design tensors to GPU once
    G_defo_dev = torch.as_tensor(G_defo_np, device=device)               # (D, P_defo)
    date_flag_dev = torch.as_tensor(date_flag_np, dtype=torch.bool, device=device)
    pbase_shared_dev = None
    if not pbase_per_pixel:
        pbase_shared_dev = torch.as_tensor(pbase_np.reshape(-1), device=device)  # (D,)

    for ci in range(num_chunk):
        c0 = ci * chunk_size
        c1 = min(c0 + chunk_size, num_pixel)
        n = c1 - c0

        # per-pixel scaling c_k = 1 / (R_k * sin θ_k)
        c_chunk = 1.0 / (range_dist_np[c0:c1] * sin_inc_angle_np[c0:c1])
        c_dev = torch.as_tensor(c_chunk, device=device)                  # (n,)

        # G_geom: (n, D, 1)
        if pbase_per_pixel:
            pbase_chunk = pbase_np[:, c0:c1]                             # (D, n)
            pbase_chunk_dev = torch.as_tensor(pbase_chunk, device=device)
            G_geom = (pbase_chunk_dev * c_dev.unsqueeze(0)).t().unsqueeze(-1)
        else:
            G_geom = (pbase_shared_dev.unsqueeze(0) * c_dev.unsqueeze(-1)).unsqueeze(-1)

        # broadcast G_defo across pixels via expand (view, no copy)
        G_defo_b = G_defo_dev.unsqueeze(0).expand(n, -1, -1)             # (n, D, P_defo)
        G0_batch = torch.cat([G_geom, G_defo_b], dim=-1).contiguous()    # (n, D, P)

        ts_chunk_dev = torch.as_tensor(ts_valid[:, c0:c1], device=device).t()  # (n, D)

        # apply date_flag for the fit; keep full-D for ts_cor / ts_res assembly
        G_fit = G0_batch[:, date_flag_dev, :]                            # (n, D', P)
        y_fit = ts_chunk_dev[:, date_flag_dev]                           # (n, D')

        X = solve_normal_equations_batched(G_fit, y_fit, print_msg=print_msg)  # (n, P)

        delta_z_chunk = X[:, 0]                                          # (n,)
        G0_col0 = G0_batch[:, :, 0]                                      # (n, D)
        ts_cor_chunk = ts_chunk_dev - G0_col0 * delta_z_chunk.unsqueeze(-1)
        ts_pred = (G0_batch @ X.unsqueeze(-1)).squeeze(-1)               # (n, D)
        ts_res_chunk = ts_chunk_dev - ts_pred

        delta_z[c0:c1] = delta_z_chunk.detach().cpu().numpy().astype(np.float32)
        ts_cor[:, c0:c1] = ts_cor_chunk.t().detach().cpu().numpy().astype(np.float32)
        ts_res[:, c0:c1] = ts_res_chunk.t().detach().cpu().numpy().astype(np.float32)

        del G0_batch, G_fit, y_fit, G_geom, G_defo_b, X, ts_chunk_dev
        del ts_cor_chunk, ts_res_chunk, ts_pred, G0_col0

        if print_msg:
            chunk_step = max(1, num_chunk // 5)
            if (ci + 1) % chunk_step == 0 or ci == num_chunk - 1:
                print(f'chunk {ci + 1} / {num_chunk}')

    return delta_z, ts_cor, ts_res

############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# GPU-accelerated network inversion                        #
############################################################
# Recommend import:
#     from mintpy.gpu import ifgram_inversion as gpu_ifginv


"""GPU-batched solver for the SBAS network inversion.

This module provides ``estimate_timeseries_batch``, an opt-in replacement for
the per-pixel CPU loop in ``run_ifgram_inversion_patch``. Pixels are solved
in batches on CUDA via the normal equations + cuSolver-batched Cholesky:
this collapses the per-pixel Householder iterations of a QR path into ~5
kernel launches per chunk on the FernandinaSenDT128 fixture. Rank-deficient
pixels are detected via ``cholesky_ex`` info codes and zeroed so NaN/Inf
never propagate.

Per-pixel NaN observations are masked by zeroing the corresponding row
weights, which is mathematically equivalent to dropping them from the WLS
system. The CPU code path is unchanged when ``solver='cpu'`` and remains
the numerical reference.
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


def _solve_cholesky(G_dev, w_dev, y_dev):
    """Per-pixel weighted least-squares via normal equations + batched Cholesky.

    Assembles the weighted system
        ``Gw_k = diag(w_k) @ G``,  ``yw_k = w_k * y_k``
    in one broadcast and dispatches to the shared
    ``solve_normal_equations_batched`` core.

    Args:
        G_dev: (num_pair, num_unknown) design matrix on GPU.
        w_dev: (num_pair, n) per-pixel sqrt-weights with NaN rows zeroed.
        y_dev: (num_pair, n) observations with NaN rows zeroed.

    Returns:
        (n, num_unknown) per-pixel solution.
    """
    Gw = w_dev.t().unsqueeze(-1) * G_dev.unsqueeze(0)
    yw = (w_dev * y_dev).t()
    return solve_normal_equations_batched(Gw, yw)


def estimate_timeseries_batch(
    A, B, y, tbase_diff,
    weight_sqrt=None,
    min_norm_velocity=True,
    rcond=1e-5,
    min_redundancy=1.0,
    inv_quality_name='temporalCoherence',
    chunk_size=None,
    solver='torch',
    print_msg=True,
):
    """Batch GPU least-squares solver for the SBAS network inversion.

    Solves, in batch over pixels k:
        (G * w_k) X_k = (y_k * w_k)         if weight_sqrt is not None (WLS)
        G X_k = y_k                          otherwise (OLS)
    where G = B if min_norm_velocity else A.

    Per-pixel NaN observations are handled by zeroing the corresponding row
    weight, which is equivalent to dropping the row from the (W)LS system.

    Args:
        A:                np.ndarray (num_pair, num_date-1). Design matrix,
                          phase formulation (used when min_norm_velocity=False).
        B:                np.ndarray (num_pair, num_date-1). Design matrix,
                          velocity formulation (used when min_norm_velocity=True).
        y:                np.ndarray (num_pair, num_pixel). Observations,
                          NaN-tolerant.
        tbase_diff:       np.ndarray (num_date-1, 1). Differential temporal
                          baseline in years.
        weight_sqrt:      np.ndarray (num_pair, num_pixel) or None. Square
                          root of per-(ifgram, pixel) weight (WLS), or None
                          for OLS.
        min_norm_velocity: bool. Solve for velocity (True) or phase (False).
        rcond:            float. Unused on CUDA: the Cholesky solver does
                          not consume an rcond cutoff. Kept for API parity
                          with the CPU path.
        min_redundancy:   float. Network-level redundancy check; if the design
                          matrix has any column with fewer non-zeros than
                          this, return zeros for all pixels.
        inv_quality_name: str. 'temporalCoherence' | 'residual' | 'no'.
        chunk_size:       int or None. Pixels per GPU chunk. None => auto.
        solver:           str. 'torch' (only one implemented).
        print_msg:        bool.

    Returns:
        ts:               np.ndarray (num_date, num_pixel) float32.
        inv_quality:      np.ndarray (num_pixel,) float32.
        num_inv_obs:      np.ndarray (num_pixel,) int16.
    """
    if solver != 'torch':
        raise ValueError(
            f"unsupported solver={solver!r}; choose from {SUPPORTED_SOLVERS}"
        )
    device = get_torch_device(solver)

    G = B if min_norm_velocity else A
    num_pair, num_unknown = G.shape
    num_pixel = y.shape[1]
    num_date = num_unknown + 1

    ts = np.zeros((num_date, num_pixel), dtype=np.float32)
    inv_quality = np.zeros(num_pixel, dtype=np.float32)
    num_inv_obs = np.zeros(num_pixel, dtype=np.int16)

    # network-level redundancy check (matches estimate_timeseries L162)
    if np.min(np.sum(A != 0., axis=0)) < min_redundancy:
        if print_msg:
            print(f'network redundancy < {min_redundancy}; skip inversion')
        return ts, inv_quality, num_inv_obs

    # decide chunk size
    if chunk_size is None or chunk_size <= 0:
        chunk_size = auto_chunk_size(num_pair, num_unknown)
        if print_msg:
            free_gib = torch.cuda.mem_get_info()[0] / 2**30
            print(f'GPU auto chunk_size = {chunk_size} pixels '
                  f'(free VRAM {free_gib:.1f} GiB)')
    else:
        chunk_size = int(chunk_size)

    num_chunk = (num_pixel + chunk_size - 1) // chunk_size
    if print_msg:
        mode = 'WLS' if weight_sqrt is not None else 'OLS'
        print(f'estimating time-series via {solver} batched {mode} '
              f'in {num_chunk} chunk(s) of up to {chunk_size} pixels ...')

    # move design matrix and tbase to GPU once (re-used across chunks)
    G_dev = torch.as_tensor(G, dtype=torch.float32, device=device)
    tbase_dev = torch.as_tensor(np.asarray(tbase_diff).flatten(),
                                dtype=torch.float32, device=device)

    use_wls = weight_sqrt is not None

    for ci in range(num_chunk):
        c0 = ci * chunk_size
        c1 = min(c0 + chunk_size, num_pixel)
        n = c1 - c0

        # prepare per-chunk inputs (host side)
        y_chunk = np.asarray(y[:, c0:c1], dtype=np.float32)
        nan_mask = np.isnan(y_chunk)             # (num_pair, n)
        y_chunk = np.where(nan_mask, 0.0, y_chunk)

        if use_wls:
            w_chunk = np.asarray(weight_sqrt[:, c0:c1], dtype=np.float32)
            w_chunk = np.where(nan_mask, 0.0, w_chunk)
        else:
            w_chunk = (~nan_mask).astype(np.float32)

        # to GPU
        y_dev = torch.as_tensor(y_chunk, device=device)         # (num_pair, n)
        w_dev = torch.as_tensor(w_chunk, device=device)         # (num_pair, n)
        valid_dev = torch.as_tensor(~nan_mask, device=device)   # (num_pair, n)

        # solve per-pixel WLS via batched Cholesky on the normal equations
        X_batch = _solve_cholesky(G_dev, w_dev, y_dev)          # (n, num_unknown)

        # inversion-quality: |sum_i exp(j * e_i)| / N (phase coherence)
        # N is the per-pixel count of valid (non-NaN) ifgrams, matching the
        # CPU reference path which drops NaN rows via skip_invalid_obs before
        # entering calc_inv_quality.
        if inv_quality_name == 'temporalCoherence':
            y_pred = G_dev @ X_batch.t()                        # (num_pair, n)
            e_dev = (y_dev - y_pred) * valid_dev                # zero-out NaN rows
            cos_sum = e_dev.cos().sum(dim=0)
            sin_sum = e_dev.sin().sum(dim=0)
            # cos(0) = 1 contributes from masked rows; subtract their count
            n_invalid = (~valid_dev).sum(dim=0).to(cos_sum.dtype)
            cos_sum = cos_sum - n_invalid
            n_valid = valid_dev.sum(dim=0).to(cos_sum.dtype).clamp(min=1.0)
            tcoh = torch.sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / n_valid
            inv_quality[c0:c1] = tcoh.detach().cpu().numpy().astype(np.float32)

        elif inv_quality_name == 'residual':
            y_pred = G_dev @ X_batch.t()
            e_dev = (y_dev - y_pred) * valid_dev
            inv_quality[c0:c1] = (
                e_dev.pow(2).sum(dim=0).sqrt().detach().cpu().numpy().astype(np.float32)
            )
        # else 'no': leave zeros

        # assemble timeseries
        if min_norm_velocity:
            # X is per-interval velocity; ts[i+1] = ts[i] + v_i * dt_i
            ts_diff = X_batch * tbase_dev.unsqueeze(0)          # (n, num_unknown)
            ts_chunk = ts_diff.cumsum(dim=1).t()                # (num_unknown, n)
        else:
            # X is per-date phase
            ts_chunk = X_batch.t()                              # (num_unknown, n)
        ts[1:, c0:c1] = ts_chunk.detach().cpu().numpy().astype(np.float32)

        num_inv_obs[c0:c1] = (~nan_mask).sum(axis=0).astype(np.int16)

        # free per-chunk tensors before next iteration
        del X_batch, y_dev, w_dev, valid_dev

        if print_msg:
            chunk_step = max(1, num_chunk // 5)
            if (ci + 1) % chunk_step == 0 or ci == num_chunk - 1:
                print(f'chunk {ci + 1} / {num_chunk}')

    return ts, inv_quality, num_inv_obs

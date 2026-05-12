############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Shared helpers for the mintpy.gpu batched solvers        #
############################################################
# Recommend import:
#     from mintpy.gpu import _common as gpu_common


"""Shared device/solver helpers for the mintpy.gpu batched solvers.

Both ``mintpy.gpu.ifgram_inversion`` and ``mintpy.gpu.dem_error`` reduce
a per-pixel weighted (or unweighted) least-squares problem to batched
normal equations + cuSolver-batched Cholesky. This module factors out
the pieces that do not depend on the specific solver:

* PyTorch / CUDA availability probing (``HAS_TORCH``,
  ``is_solver_available``, ``get_torch_device``).
* VRAM-aware chunk sizing (``auto_chunk_size``).
* The normal-equations + rank-deficient fallback core
  (``solve_normal_equations_batched``).

Callers are still responsible for assembling the per-pixel design
matrix and right-hand-side (e.g. applying sqrt-weights, broadcasting a
shared geometric column, ...) before passing them in.
"""


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


SUPPORTED_SOLVERS = ('cpu', 'torch')

# default chunk size when caller does not provide one and VRAM probing fails
DEFAULT_CHUNK_SIZE = 20000

# safety factor applied to free VRAM when auto-sizing the chunk
VRAM_SAFETY = 0.4


def is_solver_available(solver):
    """Return True if the named solver is importable and usable."""
    if solver == 'cpu':
        return True
    if solver == 'torch':
        return HAS_TORCH and torch.cuda.is_available()
    return False


def get_torch_device(solver):
    """Return ``torch.device('cuda')`` for a GPU solver, else raise."""
    if not HAS_TORCH:
        raise ImportError(
            f"solver='{solver}' requires PyTorch. "
            "Install with `pip install -e \".[gpu]\" "
            "--extra-index-url https://download.pytorch.org/whl/cu128 "
            "--index-strategy unsafe-best-match`."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"solver='{solver}' requires CUDA, but torch.cuda.is_available() is False."
        )
    return torch.device('cuda')


def auto_chunk_size(num_rows, num_cols, dtype_bytes=4):
    """Pick chunk size from free VRAM.

    The dominant per-pixel allocation in both batched solvers is the
    design matrix tensor of shape ``(n, num_rows, num_cols)``; ~3x of
    that empirically covers temporaries (normal matrix, Cholesky
    factor, residual). Returns ``DEFAULT_CHUNK_SIZE`` when CUDA is
    unavailable.
    """
    if not (HAS_TORCH and torch.cuda.is_available()):
        return DEFAULT_CHUNK_SIZE
    free_b, _ = torch.cuda.mem_get_info()
    per_pixel_bytes = 3 * num_rows * num_cols * dtype_bytes
    n = int(VRAM_SAFETY * free_b / max(per_pixel_bytes, 1))
    return max(1, n)


def solve_normal_equations_batched(G_batch, y_batch, print_msg=True):
    """Per-pixel least-squares via normal equations + batched Cholesky.

    For each pixel k with system ``G_k @ x_k = y_k`` (shapes
    ``(num_row, num_col)`` and ``(num_row,)``), solve the normal
    equations
        ``(G_k^T @ G_k) x_k = G_k^T @ y_k``
    via cuSolver-batched Cholesky. Compared to a per-pixel QR path this
    collapses ~num_col Householder iterations into a single batched
    factorization.

    Rank-deficient pixels are detected via ``cholesky_ex`` info codes;
    their factor is replaced with identity and right-hand-side with
    zeros so the downstream ``cholesky_solve`` produces an all-zero
    solution for those pixels and never propagates NaN/Inf.

    Args:
        G_batch:   (n, num_row, num_col) per-pixel design matrix.
                   Callers solving a weighted system must pre-apply
                   sqrt-weights to ``G_batch`` and ``y_batch``.
        y_batch:   (n, num_row) per-pixel observation vector.
        print_msg: bool. Warn on rank-deficient pixels.

    Returns:
        (n, num_col) per-pixel solution.
    """
    G_T = G_batch.transpose(-1, -2)
    N = G_T @ G_batch
    r = G_T @ y_batch.unsqueeze(-1)

    L, info = torch.linalg.cholesky_ex(N)
    fail_mask = info != 0
    if fail_mask.any():
        n_fail = int(fail_mask.sum().item())
        if print_msg:
            print(f'WARNING: {n_fail} rank-deficient pixel(s) in chunk; '
                  'setting solution to zero')
        eye = torch.eye(N.shape[-1], device=L.device, dtype=L.dtype)
        L = torch.where(fail_mask.view(-1, 1, 1), eye, L)
        r = torch.where(fail_mask.view(-1, 1, 1), torch.zeros_like(r), r)

    X = torch.cholesky_solve(r, L)
    return X.squeeze(-1)

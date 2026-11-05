# Single-interferogram network verification

This is the verification record for a crash report against `insarlab/MintPy`. The branch is an orphan on purpose: it shares no history with MintPy, holds no MintPy code, and is not meant to be merged or opened as a pull request.

Everything was tested against upstream `main` at `cc441486e3ca68212dec5deb5b0adf29abaa04f7`.

## Result

A one-interferogram network stops `smallbaselineApp.py` at three separate places.

| Order | Failure on `main` | Cause |
|---|---|---|
| 1 | `utils1.py:199`, in `plot_network` | `np.loadtxt` returns a 1-D array for a one-row file |
| 2 | `ifgram_inversion.py:430`, in `invert_network` | `get_reference_phase()` returns a 0-D array |
| 3 | `time_func.py:70`, in `velocity` | `estimate_time_func()` treats empty `lstsq` residues as rank deficiency, but they are also empty at full rank when M = N |

`fix-candidate.patch` (5 files, +11/-8) fixes all three. With it, one pair completes through `hdfeos5` on all four official test datasets. Two-pair and three-pair chains, and the full Fernandina stack, give the same output as `main`.

## Reproducing

`plot_network_single_pair/issue_repro3.sh` is the standalone reproduction. It builds a small synthetic stack in a new `repro` directory.

The other files record the wider runs and are not a turnkey test suite. They expect the official test data and the scratch directory layout used at the time.

## Reading the logs

Run names ending in `_up` use `main`, names ending in `_P4` use the patch, and `_P1` to `_P3` use earlier versions of it. A `FAIL` line does not always mean a problem with the patch:

- One-pair runs on `main` are expected to fail.
- Some logs include mistakes in the harness that a later run corrects. For example, `rawfix` and `rawsub2` follow `rawload`, and `cov4` follows earlier covariance attempts that are not kept here.
- Some failures are separate bugs that also occur with two pairs.

| Files in `plot_network_single_pair/` | What they cover |
|---|---|
| `p4final.*`, `p4np1.*`, `p4last.*` | The final patch on the official datasets, including the numpy 1.26.4 environment |
| `official.*`, `final.*`, `fullctl.*`, `options.*`, `sanfran.*` | Earlier suite runs and the option sweep |
| `rawload.*`, `rawfix.*`, `rawsub2.*` | Real `load_data` from raw products, cut to one or two pairs |
| `opera.*`, `cov4.*`, `offset3.*` | OPERA data, covariance uncertainty, and a synthetic offset stack |
| `cfgaudit.*`, `cfgctl.*` | Configuration keys after loading |
| `extras*` | Standalone tools |
| `save_kite_real.sh`, `check_container.py`, `kite_real.log`, `kite_venv_freeze.txt` | `save_kite.py` with the real `kite` 1.5.8, on Python 3.11 |
| `scipy_test.py` | `lstsq` residues across scipy versions |

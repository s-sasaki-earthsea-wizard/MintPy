# Configure GPU acceleration #

Two `smallbaselineApp.py` steps ship an opt-in GPU solver that batches their per-pixel inversion as normal-equations + Cholesky on a CUDA device via PyTorch:

- **`invert_network`** (`ifgram_inversion.py`) — weighted least-squares network inversion. Toggle: `mintpy.networkInversion.solver = torch` / `--solver torch`.
- **`correct_topography`** (`dem_error.py`) — pixelwise DEM-error fit. Toggle: `mintpy.topographicResidual.solver = torch` / `--solver torch`. Only the pixelwise-geometry branch (`pixelwiseGeometry = yes`, the production default) is GPU-dispatched; the mean-geometry branch is already pixel-batched on CPU and stays there.

Every other step in `smallbaselineApp.py` continues to run on the CPU. Both solvers are opt-in — `mintpy.*.solver = auto` resolves to `cpu`, so existing setups are unaffected.

The `torch` solver is orthogonal to Dask parallel processing (see [dask.md](./dask.md)): the former replaces the per-pixel CPU loop with a single batched Cholesky on one CUDA device, the latter distributes that same per-pixel loop across multiple worker processes. The two paths are not currently combined; pick one.

## 1. Setup ##

See [installation.md](./installation.md) section 2.4 for installing the `[gpu]` extras with the matching CUDA wheel index. Selecting `solver = torch` on a host without a visible CUDA device is a hard error (no silent CPU fallback).

## 2. Enable on `invert_network` ##

#### 2.1 via command line ####

Run the following in the terminal:

```bash
ifgram_inversion.py inputs/ifgramStack.h5 --solver torch
ifgram_inversion.py inputs/ifgramStack.h5 --solver torch --gpu-chunk-size 20000
```

`--gpu-chunk-size 0` (the default) auto-sizes the per-chunk pixel count from free VRAM; pass a positive integer to override.

#### 2.2 via template file ####

Adjust options in the template file:

```cfg
mintpy.networkInversion.solver       = torch  #[cpu / torch], auto for cpu
mintpy.networkInversion.gpuChunkSize = auto   #[int >= 0], auto for 0 (auto-size from free VRAM)
```

and feed the template file to the script:

```bash
ifgram_inversion.py inputs/ifgramStack.h5 -t smallbaselineApp.cfg
smallbaselineApp.py smallbaselineApp.cfg
```

#### 2.3 Testing using example data ####

Download and run the FernandinaSenDT128 example data; then run with and without the GPU solver:

```bash
cd FernandinaSenDT128/mintpy
ifgram_inversion.py inputs/ifgramStack.h5 -w no --solver cpu
ifgram_inversion.py inputs/ifgramStack.h5 -w no --solver torch
```

The two outputs should agree to float32 round-off (RMS on the order of 1e-5).

## 3. Enable on `correct_topography` ##

The same `--solver torch` opt-in is exposed on `dem_error.py`, controlled by `mintpy.topographicResidual.solver`. Only the pixelwise-geometry branch (`pixelwiseGeometry = yes`) is GPU-dispatched.

#### 3.1 via command line ####

```bash
dem_error.py timeseries_ERA5_ramp.h5 -g inputs/geometryRadar.h5 --solver torch
dem_error.py timeseries_ERA5_ramp.h5 -g inputs/geometryRadar.h5 --solver torch --gpu-chunk-size 200000
```

#### 3.2 via template file ####

```cfg
mintpy.topographicResidual.solver       = torch  #[cpu / torch], auto for cpu
mintpy.topographicResidual.gpuChunkSize = auto   #[int >= 0], auto for 0 (auto-size from free VRAM)
```

then:

```bash
smallbaselineApp.py smallbaselineApp.cfg
```

#### 3.3 Testing using example data ####

```bash
cd FernandinaSenDT128/mintpy
dem_error.py timeseries_ERA5_ramp.h5 -g inputs/geometryRadar.h5 --solver cpu -o ts_demErr_cpu.h5
dem_error.py timeseries_ERA5_ramp.h5 -g inputs/geometryRadar.h5 --solver torch -o ts_demErr_gpu.h5
```

CPU and GPU outputs should agree to float32 round-off (rms / |cpu|.max on the order of 1e-6 for `delta_z`, 1e-8 for `ts_cor`).

## 4. Behavior notes ##

+ **VRAM auto-sizing.** `gpuChunkSize = 0` (auto) probes free VRAM at runtime and chooses a per-chunk pixel count with a fixed headroom factor. Set an explicit integer to override (e.g. for reproducible chunking across hosts with different VRAM).

+ **Rank-deficient pixels.** Detected via `torch.linalg.cholesky_ex` info codes; their solution is set to zero so NaN/Inf never propagate downstream. A warning line reports the count per chunk.

+ **Per-pixel NaN observations.** Handled by zeroing the corresponding row weight, which is mathematically equivalent to dropping that row from the WLS system.

+ **No silent CPU fallback.** Selecting `solver = torch` on a host without a visible CUDA device raises immediately rather than silently falling back to CPU; this keeps performance regressions visible.

## 5. Performance ##

Benchmarks on RTX 5080 (Blackwell sm_120, CUDA 12.8, PyTorch 2.11) are tracked in the sibling [`mintpy-benchmark`](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark) repository.

### `invert_network`

Tutorial-scale on FernandinaSenDT128 (270k pixels, 288 ifgs):

+ [report_torch.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/report_torch.md) — `cpu` vs `torch` end-to-end
+ [report_solver_comparison.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/report_solver_comparison.md) — lstsq vs Cholesky, numerical equivalence (RMS ~1e-5) and per-step speedup
+ [report_chunk_sweep.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/report_chunk_sweep.md) — chunk-size sensitivity
+ [report_profile.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/report_profile.md) — `torch.profiler` GPU kernel breakdown

Large-scene on GalapagosSenDT128 (3.4M pixels, 475 kept ifgs; ~12.6× pixels and 1.65× ifgs vs Fernandina):

+ [report_large_scene.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/report_large_scene.md) — `solver=torch` reaches **36.4× step wall** / **44.4× internal** speedup on `invert_network` (cpu 6189 s → torch 170 s on RTX 5080 / SSD), confirming the speedup grows at scale; output equivalence preserved at float32 round-off (abs RMS max ~16 µm)

### `correct_topography`

Five-scene survey (Fernandina + Galapagos + three Zenodo Tier 1 scenes covering ISCE2 / GMTSAR / ARIA / ROI_PAC ingest pipelines, two wavelengths, and D ∈ [24, 333]):

+ [report_bench_survey.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/dem_error/report_bench_survey.md) — combined speedup-vs-K curve, numeric gate validation, axes coverage analysis
+ Per-scene: [report_fernandina.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/dem_error/report_fernandina.md), [report_galapagos.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/dem_error/report_galapagos.md), [report_sanfranbay.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/dem_error/report_sanfranbay.md), [report_sanfran_aria.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/dem_error/report_sanfran_aria.md), [report_kuju.md](https://github.com/s-sasaki-earthsea-wizard/mintpy-benchmark/blob/main/reports/dem_error/report_kuju.md)

Headline: on local SSD, tutorial-scale scenes (~250–325k pixels) typically see **2–2.5×** speedup; production-scale (K = 3.4M Galapagos) reaches **6.15×**. CPU/GPU numeric agreement at `rms / |cpu|.max < 1e-5` on all 5 scenes. Note that the GPU path is **more I/O sensitive** than CPU because its compute portion is sub-second — networked storage (CIFS / NAS) can erase the speedup at small scenes (Fernandina drops from 2.56× on SSD to 0.97× on NAS).

The closed-form speedup-vs-(K,D,P) model fitted against the survey lives in the wiki: [GPU Speedup Scaling Model](https://github.com/s-sasaki-earthsea-wizard/MintPy/wiki/GPU-Speedup-Scaling-Model). Short version: K is the GPU-parallelisable batch dimension; D and P are per-pixel internal dimensions that scale CPU and GPU walls similarly. Expect ≳ 4× speedup once K ≳ 1M.

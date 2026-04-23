# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository context

This is a **fork** of [meta-pytorch/torchcomms](https://github.com/meta-pytorch/torchcomms) with added sparse/compressed collective primitives inside the bundled NCCLX backend. The sparse work is the point of this fork — do not assume unfamiliar `sparse`/`ccd`/`Ccd*` code came from upstream. It's part of an ongoing research project ("CCD" = the sparse chunk protocol; SPOP / COO1D / DENSE / ADAPTIVE are its compression formats).

The upstream project is an experimental PyTorch communications API with multiple backends (NCCL, NCCLX, Gloo, RCCL, RCCLX, XCCL). Upstream docs live at https://meta-pytorch.org/torchcomms/main/index.html and in `README.md`.

## Building on Perlmutter (primary dev environment)

Builds run inside SLURM on a compute node, not the login node, because they need CUDA/nvcc and a lot of CPU.

```bash
# First-time / full build (builds third-party deps + NCCLX)
sbatch build_submit.sh          # runs build_perlmutter.sh -> build_ncclx_pm.sh

# Incremental NCCLX rebuild (skips third-party deps)
bash build_perlmutter.sh        # sets NCCL_BUILD_SKIP_DEPS=1
```

Key environment knobs read by `build_ncclx_pm.sh`:

- `INSTALL_PREFIX` / `CONDA_PREFIX` — where third-party deps (folly, fbthrift, …) get installed. On Perlmutter, `build_perlmutter.sh` pins this to `/pscratch/sd/e/egencer/sparsecomms/nccl-prefix`.
- `NCCL_HOME` — source tree for NCCLX. Perlmutter build points this at `comms/ncclx/stable` (not `v2_27` / `v2_28`).
- `NCCL_BUILD_SKIP_DEPS=1` — reuse already-built third-party libs. Default for `build_perlmutter.sh`.
- `NVCC_ARCH` — defaults to `a100` (Perlmutter). Auto-adds `b200` when CUDA ≥ 12.8.
- `CLEAN_BUILD=1` — wipe `$BUILDDIR` (= `build/ncclx`) before rebuilding NCCLX.
- `NCCL_BUILD_AND_INSTALL=1` — run `make src.install` instead of `make src.build`.
- `USE_SYSTEM_LIBS=1` — consume third-party libs from conda/system instead of building from source (upstream default path).

The `build_ncclx_pm.sh` script also auto-generates `comms/utils/cvars/nccl_cvars.{h,cc}` from `comms/utils/cvars/nccl_cvars.yaml` via `extractcvars.py`. These two files are **generated, not checked in** — don't hand-edit them. Add new knobs to `nccl_cvars.yaml`.

For the torchcomms Python package itself (not just NCCLX), upstream uses:

```bash
pip install -r requirements.txt
pip install --no-build-isolation -v .
```

Backend selection is via env vars (`USE_NCCL`, `USE_NCCLX`, `USE_GLOO`, `USE_RCCL`, `USE_RCCLX`, `USE_XCCL`, `USE_TRANSPORT`). See `setup.py` and `CMakeLists.txt` for defaults.

## Running tests

- Python integration tests for the collectives API live in `comms/torchcomms/tests/integration/py/` (e.g. `AllReduceTest.py`, `AllGatherTest.py`). They run under `torchrun --nproc_per_node=N <TestFile>.py`.
- Python unit tests (non-collective helpers) live in `comms/torchcomms/tests/unit/py/` — plain `pytest`.
- C++ tests live under `comms/torchcomms/tests/{unit,integration}/cpp/` and are gated behind `-DBUILD_TESTS=ON` in CMake.
- A smoke test script is at `scripts/smoke_test.py`.

No repo-level `make test` / `pytest` entry point — pick the individual test file relevant to what you changed.

## High-level code architecture

```
comms/
├── torchcomms/        # The user-facing PyTorch comms API (C++ + pybind)
│   ├── TorchComm*.{hpp,cpp}     # Core TorchComm / TorchWork / TorchCommBatch abstractions
│   ├── TorchCommBackend.hpp     # Virtual interface every backend implements
│   ├── TorchCommFactory.cpp     # new_comm("nccl"|"ncclx"|"gloo"|"rccl"|"rcclx"|"xccl", ...)
│   ├── nccl/ ncclx/ gloo/ rccl/ rcclx/  # per-backend adapters → TorchCommBackend
│   └── tests/ {unit,integration}/{cpp,py}/
│
├── ncclx/             # Vendored NCCLX (Meta's NCCL fork). Multiple versions coexist:
│   ├── stable/        # THE active tree. build_ncclx_pm.sh points NCCL_HOME here.
│   ├── v2_27/ v2_28/  # Snapshots, not actively built by build_perlmutter.sh
│   ├── headers/ stable/
│   └── stable/src/
│       ├── collectives.cc        # Public ncclAllReduce / ncclReduceScatterSparse / …
│       ├── enqueue.cc            # Schedules kernels, per-op tuning
│       ├── device/               # CUDA device code (kernels + primitives)
│       │   ├── prims_simple.h    # Core ring primitives — modified for CCD
│       │   ├── ccd.cuh           # Sparse chunk header + format enum (SPOP/COO1D/DENSE/ADAPTIVE)
│       │   ├── all_reduce_sparse_block.cu{,h}
│       │   └── all_reduce.h / reduce_scatter.h / all_gather.h / …
│       ├── transport/            # net_ib, p2p, shm, nvls, coll_net, gdaki
│       ├── graph/                # Topology + ring/tree construction
│       └── include/              # NCCLX headers; info.h carries isSparse + ccd* fields
│
├── utils/             # Shared utility library (logging, env parsing, CUDA RAII, …)
│   └── cvars/         # NCCLX configuration variable system (see below)
│
├── common/            # Cross-backend primitives (IpcGpuBarrier, BitOps, …)
├── ctran/             # CTran (collective transport) subsystem
├── analyzer/ pipes/   # Tracing / instrumentation helpers
└── rcclx/             # RCCL-X (AMD equivalent), parallel to ncclx/

python/                # Out-of-tree JIT helper that links against a prebuilt NCCLX
                       # (see below) — NOT part of the main torchcomms Python package.
```

### The sparse ("CCD") path

Upstream torchcomms/NCCLX does not ship with sparse collectives. This fork adds three new public APIs in `comms/ncclx/stable/src/nccl.h.in`:

- `ncclReduceScatterSparse`
- `ncclAllGatherSparse`
- `ncclAllReduceSparse`

(plus the pre-existing `ncclAllReduceSparseBlock`.) They are implemented in `collectives.cc` as thin wrappers that populate a regular `ncclInfo` but set `info.isSparse = 1` plus a handful of `ccd*` fields (format mask, dense thresholds, intra-node threshold). From there, the kernels in `device/prims_simple.h` + `device/ccd.cuh` take over: each FIFO slot gets a `CcdSparseChunkHeader` that tells the receiver which compression format (`DENSE`, `COO1D`, `SPOP`, `ADAPTIVE`) was used and where to find the bitvector / indices / values. Each hop re-decides format based on tracked density.

CCD tuning flows through env vars (read at collective-launch time in `collectives.cc`):

- `NCCL_CCD_FORMAT_MASK` (default `0b0111` — all formats enabled)
- `NCCL_CCD_DENSE_THRESHOLD` (default `0.3`) — inter-node RS/AR cutoff
- `NCCL_CCD_AG_DENSE_THRESHOLD` (default `0.1`) — AG phase cutoff
- `NCCL_CCD_DENSE_INTRA_THRESHOLD` (default `0.9`) — intra-node cutoff

Two *channel*-count CVARs also exist (`NCCL_ALL_REDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS`, `NCCL_ALL_REDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE`). These go through the CVAR system (see below), not `getenv`.

### `python/sparse_comms.py` — the out-of-tree Python shim

`python/sparse_comms.py` + `python/sparse_comms_ext.cpp` are a **separate** JIT-compiled PyTorch C++ extension (not part of the main torchcomms wheel). They grab the raw `ncclComm_t` out of a PyTorch process group via `pg._comm_ptr()` and call the new `ncclAllReduceSparse` / etc. directly against `libnccl_static.a` from `build/ncclx/`. Set `NCCLX_BUILD_DIR` if that lib lives elsewhere; `SPARSE_COMMS_BUILD_DIR` caches the JIT build.

This is how sparse collectives are exercised end-to-end without rebuilding the entire torchcomms Python package.

### The CVAR system

`comms/utils/cvars/nccl_cvars.yaml` is the single source of truth for NCCLX knobs. `extractcvars.py` generates `nccl_cvars.h` / `nccl_cvars.cc` from it. At runtime CVARs are initialized once during `ncclInit`; do not read them before init. Values can be set via environment variable or `/etc/nccl.conf` (env wins). To add a knob, edit `nccl_cvars.yaml`, rebuild, and `#include "comms/utils/cvars/nccl_cvars.h"`.

### Backend selection at the torchcomms layer

`comms/torchcomms/TorchCommFactory.cpp` dispatches `new_comm("<backend>", device, name=...)` to the right `TorchComm*` implementation. Each backend lives in its own subdir under `comms/torchcomms/` and implements `TorchCommBackend.hpp`. The NCCLX backend adapter (`comms/torchcomms/ncclx/`) is what wraps the vendored NCCLX library under `comms/ncclx/stable/`.

## MSCCL engine kernel debugging

The MSCCL engine kernel (`msccl_engine.cuh`) is instantiated 30 times (3
redops × 10 datatypes). Each instantiation pulls in the heavy `Primitives<>`
template. **Never add `printf` calls to device-side kernel code** — the
variadic call machinery blows up register pressure across all 30
instantiations, causing ptxas register allocation to take hours instead of
minutes.

For debugging device-side crashes (e.g. Warp Illegal Address), use:
- `cuda-memcheck` / `compute-sanitizer` for memory error detection
- `cuda-gdb` with `set cuda break_on_launch application` to catch the exact
  faulting instruction
- Host-side validation: dump kernel arguments (comm pointers, algo struct
  fields, work params) to stderr from the host launch path before launching
- Write diagnostic values to a device-memory scratch buffer and read back on
  host after kernel completion, instead of inline printf

## Conventions (from CONTRIBUTING.md)

- 2 spaces for indentation (not tabs).
- 80-character line length.
- `nccl_cvars.h` / `nccl_cvars.cc` are generated — never commit edits to them.

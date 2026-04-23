#!/bin/bash
# Build a standalone MSCCL algo .so from a generated .cu data file.
# No nvcc or CUDA headers needed — the generated files are pure host-side
# struct data; this script compiles them with g++ (C++20 for designated init).
#
# Usage:
#   bash build_msccl_algo.sh [path/to/algo.cu] [-o output.so]
#
#   path/to/algo.cu  — generated algo file (default: the naive ring 8-GPU 1KB
#                      file already in the device directory)
#   -o output.so     — where to write the .so (default: basename of .cu in
#                      the repo root, e.g. msccl_algo_naive_ring_8gpus_1KB.so)
#
# Examples:
#   bash build_msccl_algo.sh
#   bash build_msccl_algo.sh comms/ncclx/stable/src/device/msccl_algo_naive_ring_8gpus_1KB.cu
#   bash build_msccl_algo.sh /path/to/my_new_algo.cu -o my_new_algo.so

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
NCCL_SRC="${BASE_DIR}/comms/ncclx/stable/src"
DEVICE_SRC="${NCCL_SRC}/device"

# ── argument parsing ─────────────────────────────────────────────────────────
ALGO_CU="${DEVICE_SRC}/msccl_algo_naive_ring_8gpus_1KB.cu"
OUTPUT_SO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUTPUT_SO="$2"; shift 2 ;;
    *)  ALGO_CU="$1"; shift ;;
  esac
done

if [[ ! -f "${ALGO_CU}" ]]; then
  echo "ERROR: algo file not found: ${ALGO_CU}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_SO}" ]]; then
  BASENAME="$(basename "${ALGO_CU}" .cu)"
  OUTPUT_SO="${BASE_DIR}/${BASENAME}.so"
fi

# ── build ────────────────────────────────────────────────────────────────────
g++ -shared -fPIC -std=c++20 -x c++ \
    -DMSCCL_STANDALONE \
    -I "${NCCL_SRC}" \
    -I "${DEVICE_SRC}" \
    -o "${OUTPUT_SO}" \
    "${ALGO_CU}"

echo "Built: ${OUTPUT_SO}"
echo ""
echo "Use with:"
echo "  MSCCL_ALGO_PATH=${OUTPUT_SO} \\"
echo "  LD_PRELOAD=${BASE_DIR}/ncclmsccl_preload.so \\"
echo "  LD_LIBRARY_PATH=${BASE_DIR}/build/ncclx/lib:\$LD_LIBRARY_PATH \\"
echo "  srun ... ./all_reduce_perf ..."

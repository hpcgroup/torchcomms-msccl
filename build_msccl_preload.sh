#!/bin/bash
# Build ncclmsccl_preload.so — the LD_PRELOAD shim that intercepts
# ncclAllReduce -> ncclAllReduceMsccl and loads algo .so files at comm-init
# time via MSCCL_ALGO_PATH.
#
# Run from the repo root:
#   bash build_msccl_preload.sh
#
# Output: ncclmsccl_preload.so  (in the repo root)

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILDDIR="${BUILDDIR:-${BASE_DIR}/build/ncclx}"
NCCL_SRC="${BASE_DIR}/comms/ncclx/stable/src"

if [[ ! -d "${BUILDDIR}/include" ]]; then
  echo "ERROR: ${BUILDDIR}/include not found — build libnccl first" >&2
  exit 1
fi

g++ -shared -fPIC \
    -o "${BASE_DIR}/ncclmsccl_preload.so" \
    "${BASE_DIR}/ncclmsccl_preload.cpp" \
    -I "${NCCL_SRC}" \
    -I "${BUILDDIR}/include" \
    -ldl

echo "Built: ${BASE_DIR}/ncclmsccl_preload.so"

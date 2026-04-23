#!/bin/bash
# Incremental NCCLX build for Perlmutter.
# Assumes deps were already built via a full build_ncclx_pm.sh run.
module load python

INSTALL_PREFIX=/pscratch/sd/e/egencer/sparsecomms/nccl-prefix \
NCCL_BUILD_SKIP_DEPS=1 \
NCCL_HOME=/pscratch/sd/e/egencer/coll_evolve/ir_compile/torchcomms-sparse/comms/ncclx/stable \
bash "$(dirname "$0")/build_ncclx_pm.sh" "$@"

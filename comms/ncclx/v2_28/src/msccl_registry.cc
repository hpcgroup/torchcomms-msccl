/*
 * MSCCL algo registry implementation.
 */

#include "msccl_registry.h"
#include "msccl_engine_host.h"
#include "nccl_common.h"
#include "debug.h"
#include "comm.h"
#include "checks.h"

#define MSCCL_REGISTRY_MAX 256

struct mscclRegistryEntry {
  int    nRanks;
  int    coll;
  size_t minBytes;
  size_t maxBytes;
  const struct mscclEngineAlgo *algos;
};

static struct mscclRegistryEntry gRegistry[MSCCL_REGISTRY_MAX];
static int gRegistrySize = 0;

void ncclMscclRegisterAlgo(int nRanks, int coll,
                           size_t minBytes, size_t maxBytes,
                           const struct mscclEngineAlgo *algos) {
  if (gRegistrySize >= MSCCL_REGISTRY_MAX) {
    WARN("MSCCL registry full (%d entries), dropping algo nRanks=%d",
         MSCCL_REGISTRY_MAX, nRanks);
    return;
  }
  struct mscclRegistryEntry *e = &gRegistry[gRegistrySize++];
  e->nRanks   = nRanks;
  e->coll     = coll;
  e->minBytes = minBytes;
  e->maxBytes = maxBytes;
  e->algos    = algos;
  TRACE(NCCL_INIT, "MSCCL registered algo nRanks=%d coll=%d bytes=[%zu,%zu]",
        nRanks, coll, minBytes, maxBytes);
}

ncclResult_t ncclMscclAutoLoad(ncclComm_t comm) {
  int nLoaded = 0;
  for (int i = 0; i < gRegistrySize; i++) {
    const struct mscclRegistryEntry *e = &gRegistry[i];
    if (e->nRanks != comm->nRanks) continue;
    /* For now only auto-load allreduce (coll == ncclFuncAllReduce). */
    if (e->coll != ncclFuncAllReduce) continue;
    NCCLCHECK(ncclMscclEngineAddAlgo(comm, e->algos, e->nRanks,
                                     e->minBytes, e->maxBytes));
    nLoaded++;
  }
  if (nLoaded)
    INFO(NCCL_INIT, "MSCCL auto-loaded %d algo(s) for nRanks=%d",
         nLoaded, comm->nRanks);
  return ncclSuccess;
}

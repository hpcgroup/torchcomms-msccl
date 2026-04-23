/*
 * MSCCL algo registry.
 *
 * Generated algo files call MSCCL_REGISTER_ALGO() via a C++ constructor
 * attribute to self-register at startup.  At communicator init,
 * ncclMscclAutoLoad() scans the registry and loads all matching entries.
 */

#ifndef MSCCL_REGISTRY_H_
#define MSCCL_REGISTRY_H_

#include <stddef.h>
#include "nccl.h"
#include "nccl_common.h"
#include "msccl_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

void ncclMscclRegisterAlgo(int nRanks, int coll,
                           size_t minBytes, size_t maxBytes,
                           const struct mscclEngineAlgo *algos);

/* Called at ncclCommInitRank time; loads all registered algos matching comm. */
ncclResult_t ncclMscclAutoLoad(ncclComm_t comm);

#ifdef __cplusplus
}
#endif

/*
 * Macro for use in generated algo .cu files.
 * coll: ncclFuncAllReduce / ncclFuncReduceScatter / etc.
 * sym:  the extern array symbol name (e.g. msccl_algo_naive_ring_8gpus_1KB).
 */
#define MSCCL_REGISTER_ALGO(sym, nranks, coll, minB, maxB)              \
  static void __attribute__((constructor))                              \
  __msccl_register_##sym(void) {                                        \
    ncclMscclRegisterAlgo((nranks), (coll), (minB), (maxB), (sym));     \
  }

#endif /* MSCCL_REGISTRY_H_ */

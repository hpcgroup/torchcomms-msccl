/*
 * Host-side state for the MSCCL interpreter engine.
 *
 * One mscclEngineState lives per ncclComm (comm->mscclEngine).
 * It holds up to MSCCL_ENGINE_MAX_SIZE_ENTRIES size-ranged algos so that
 * ncclAllReduceMsccl can pick the right one at launch time.
 *
 * Usage:
 *   ncclMscclEngineAddAlgo(comm, algos, nRanks, minBytes, maxBytes);
 *   // repeat for each size class
 *   ncclAllReduceMsccl(sendbuff, recvbuff, count, dtype, op, comm, stream);
 */

#ifndef MSCCL_ENGINE_HOST_H_
#define MSCCL_ENGINE_HOST_H_

#include "nccl.h"
#include "msccl_engine.h"
#include <cuda_runtime.h>

/* Forward declarations of CUDA types used by the launch wrapper. */
struct ncclKernelCommAndChannels;
struct mscclEngineWork;

typedef void (*mscclKernelFn)(struct ncclKernelCommAndChannels *,
                               const struct mscclEngineAlgo *,
                               struct mscclEngineWork);

#ifdef __cplusplus
extern "C" {
#endif
void mscclLaunchKernel(mscclKernelFn kfn, int nBlocks, int nThreads, int smem,
                       cudaStream_t stream,
                       struct ncclKernelCommAndChannels *commAndChans,
                       const struct mscclEngineAlgo *algo,
                       struct mscclEngineWork work);
/* devOp: ncclDevRedOp_t cast to int; dt: ncclDataType_t cast to int; proto: NCCL_PROTO_* */
mscclKernelFn mscclSelectKernel(int devOp, int dt, int proto);
#ifdef __cplusplus
}
#endif

#define MSCCL_ENGINE_MAX_SIZE_ENTRIES 32

struct mscclEngineSizeEntry {
  size_t  minBytes;
  size_t  maxBytes;            /* inclusive upper bound on total message bytes */
  struct mscclEngineAlgo *devAlgo;
  const struct mscclEngineAlgo *hostAlgo;
  int32_t nBlocks;
  int32_t nChunksPerLoop;
  int32_t nScratchChunks;
  int32_t protocol;
  /* Per-TB connection-slot indices (mirror what was written to devAlgo). The
   * host-side proxy-save loop must dispatch ops to the same peers[peer]->
   * {recv,send}[ci] FIFO that the kernel TB will read/write, otherwise the
   * proxy never advances slot 1 and any TB on slot 1 hangs. */
  uint8_t connIndexRecv[MSCCL_ENGINE_MAX_TBS];
  uint8_t connIndexSend[MSCCL_ENGINE_MAX_TBS];
  /* FIFO-stripe count chosen at AddAlgo time. >=1; 1 means "no striping"
   * (today's behavior). >1 when some TB's pre-recv send pressure on any
   * connection exceeds NCCL_STEPS — both ranks would otherwise queue more
   * outbound slots than the FIFO holds before either reaches its recv,
   * causing a symmetric circular wait. The kernel splits each step's count
   * across this many stripes per outer iter so the per-stripe-per-conn slot
   * pressure stays <= NCCL_STEPS. */
  uint16_t nStripes;
  /* runtime_v3 (R4a/R4b). See msccl_engine_host.cc. */
  uint8_t  stripeFlags;        /* NCCL_MSCCL_STRIPE_FLAGS: 0 step, 1 stripe, 2 auto    */
  uint8_t  stripeCountLaunch;  /* NCCL_MSCCL_STRIPE_COUNT: 0 addalgo, 1 launch         */
  /* R4a: nStripes for maxAllowedCount = m (index m, 1..MSCCL_ENGINE_MAX_COUNT-1), max over all ranks of
   * the IR. 0 = more than MSCCL_ENGINE_MAX_STRIPES (the launch is rejected). */
  uint16_t nStripesForCount[MSCCL_ENGINE_MAX_COUNT];
  uint16_t nStripesAddAlgoMax; /* AddAlgo formula, max over all ranks (cross-check of the per-rank value) */
  /* R4b: one device algo per distinct nStripes > 1, with MSCCL_ENGINE_DEP_ALIGNED marks for that count.
   * Unused when stripeFlags == 0, or when no dependency is aligned (devAlgo is used then). */
  int      nStripeVariants;
  struct {
    uint16_t nStripes;
    int32_t  depsTotal, depsCandidate, depsAligned, demoted, intraHazards, unordered;
    int32_t  skipped;            /* 1 = analysis skipped (budget/unresolved dep/cycle): all non-aligned */
    int32_t  cycleStep, cycle;   /* flag-graph cycle with step order / with the final marks (= deadlock) */
    struct mscclEngineAlgo *devAlgo;  /* NULL when depsAligned == 0 (use the entry's devAlgo)  */
  } stripeVariant[MSCCL_ENGINE_MAX_STRIPES];
  /* Once-per-size INFO log of the per-launch stripe decision. */
  int      nLoggedSizes;
  size_t   loggedSizes[8];
};

struct mscclEngineState {
  struct mscclEngineSizeEntry sizes[MSCCL_ENGINE_MAX_SIZE_ENTRIES];
  int     nSizes;
  int     maxNBlocks;          /* max nBlocks across all size entries          */
  struct mscclEngineFlag *devSyncFlags;  /* maxNBlocks entries, zero-init     */
  void   *devScratch;
  size_t  scratchBytes;
  uint32_t workIndex;
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Add one size-ranged algo to a communicator's engine.
 *
 * hostAlgos : array[nRanks] of mscclEngineAlgo (one per rank, in rank order).
 * nRanks    : must equal comm->nRanks.
 * minBytes/maxBytes : total message size range (bytes) this algo handles.
 *                     Use 0/SIZE_MAX to match all sizes.
 *
 * May be called multiple times for different size classes before the first
 * ncclAllReduceMsccl(); each call adds one entry to the internal table.
 */
ncclResult_t ncclMscclEngineAddAlgo(ncclComm_t comm,
                                    const struct mscclEngineAlgo *hostAlgos,
                                    int nRanks,
                                    size_t minBytes, size_t maxBytes);

/* Release all device memory; called automatically at comm destroy. */
ncclResult_t ncclMscclEngineFree(ncclComm_t comm);

/* Internal: kernel dispatch; called from collectives.cc public wrapper. */
ncclResult_t ncclAllReduceMscclImpl(const void *sendbuff, void *recvbuff,
                                    size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op,
                                    ncclComm_t comm, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* MSCCL_ENGINE_HOST_H_ */

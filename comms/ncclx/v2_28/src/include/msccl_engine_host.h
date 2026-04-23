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
/* devOp: ncclDevRedOp_t cast to int; dt: ncclDataType_t cast to int */
mscclKernelFn mscclSelectKernel(int devOp, int dt);
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

/*
 * MSCCL engine — host-side launch glue.
 *
 * ncclMscclEngineAddAlgo : copy one size-ranged per-rank algo to device.
 * ncclMscclEngineFree    : release device allocations.
 * ncclAllReduceMscclImpl : select algo by message size, build work, launch.
 */

#include "core.h"          /* NCCL_API */
#include "comm.h"
#include "collectives.h"   /* ncclTypeSize */
#include "alloc.h"         /* ncclCudaCalloc, ncclCalloc */
#include "device.h"        /* ncclShmemDynamicSize, NCCL_MAX_NTHREADS */
#include "msccl_engine.cuh"
#include "msccl_engine_host.h"

/* ── redOpArg (mirrors enqueue.cc::hostToDevRedOp for Sum/Prod/Min/Max) ─── */
static ncclResult_t mscclGetRedOpArg(ncclRedOp_t op, ncclDataType_t dt,
                                     ncclDevRedOp_t *devOp, uint64_t *arg) {
  int nbits = 8 * ncclTypeSize(dt);
  if (nbits <= 0) return ncclInvalidArgument;
  uint64_t allBits = uint64_t(-1) >> (64 - nbits);
  uint64_t signBit = allBits ^ (allBits >> 1);
  *arg = 0;
  switch ((int)op) {
    case ncclSum:  *devOp = ncclDevSum;    break;
    case ncclProd: *devOp = ncclDevProd;   break;
    case ncclMin:
      *devOp = ncclDevMinMax;
      if (dt==ncclInt8 || dt==ncclInt32 || dt==ncclInt64) *arg ^= signBit;
      break;
    case ncclMax:
      *devOp = ncclDevMinMax;
      if (dt==ncclInt8 || dt==ncclInt32 || dt==ncclInt64) *arg ^= signBit;
      *arg ^= allBits;
      break;
    default: return ncclInvalidArgument;
  }
  return ncclSuccess;
}

/* ── kernel function pointer type and dispatch table ────────────────────── */
typedef void (*mscclKernelFn)(struct ncclKernelCommAndChannels *,
                              const struct mscclEngineAlgo *,
                              struct mscclEngineWork);

#define KFUNC(redop, type) ((mscclKernelFn)mscclEngineKernel_##redop##_##type)

static mscclKernelFn selectKernel(ncclDevRedOp_t devOp, ncclDataType_t dt) {
#define CASE_DT(redop) \
  switch (dt) { \
    case ncclInt8:    return KFUNC(redop, int8_t); \
    case ncclUint8:   return KFUNC(redop, uint8_t); \
    case ncclInt32:   return KFUNC(redop, int32_t); \
    case ncclUint32:  return KFUNC(redop, uint32_t); \
    case ncclInt64:   return KFUNC(redop, int64_t); \
    case ncclUint64:  return KFUNC(redop, uint64_t); \
    case ncclFloat16: return KFUNC(redop, half); \
    case ncclFloat32: return KFUNC(redop, float); \
    case ncclFloat64: return KFUNC(redop, double); \
    case ncclBfloat16:return KFUNC(redop, __nv_bfloat16); \
    default:          return nullptr; \
  }
  switch (devOp) {
    case ncclDevSum:    { CASE_DT(Sum);    break; }
    case ncclDevProd:   { CASE_DT(Prod);   break; }
    case ncclDevMinMax: { CASE_DT(MinMax); break; }
    default: return nullptr;
  }
#undef CASE_DT
  return nullptr;
}

/* ── find the best size entry for a given message size ──────────────────── */
static const struct mscclEngineSizeEntry *
selectSizeEntry(const struct mscclEngineState *s, size_t msgBytes) {
  /* Prefer exact-range match; fall back to the entry whose maxBytes is
   * smallest but still >= msgBytes (closest upper bound). */
  const struct mscclEngineSizeEntry *best = nullptr;
  for (int i = 0; i < s->nSizes; i++) {
    const struct mscclEngineSizeEntry *e = &s->sizes[i];
    if (msgBytes >= e->minBytes && msgBytes <= e->maxBytes) {
      if (!best || e->maxBytes < best->maxBytes)
        best = e;
    }
  }
  if (best) return best;
  /* No range match: pick entry with smallest maxBytes > msgBytes. */
  for (int i = 0; i < s->nSizes; i++) {
    const struct mscclEngineSizeEntry *e = &s->sizes[i];
    if (e->maxBytes >= msgBytes)
      if (!best || e->maxBytes < best->maxBytes)
        best = e;
  }
  /* Last resort: largest algo. */
  if (!best && s->nSizes > 0) best = &s->sizes[s->nSizes - 1];
  return best;
}

/* ── ensure scratch buffer >= bytes ─────────────────────────────────────── */
static ncclResult_t ensureScratch(struct mscclEngineState *s, size_t bytes) {
  if (bytes <= s->scratchBytes) return ncclSuccess;
  if (s->devScratch) CUDACHECK(cudaFree(s->devScratch));
  s->devScratch   = nullptr;
  s->scratchBytes = 0;
  NCCLCHECK(ncclCudaCalloc((char **)&s->devScratch, bytes));
  s->scratchBytes = bytes;
  return ncclSuccess;
}

/* ── ensure syncFlags array is >= newMaxBlocks entries ──────────────────── */
static ncclResult_t ensureSyncFlags(struct mscclEngineState *s, int nBlocks) {
  if (nBlocks <= s->maxNBlocks) return ncclSuccess;
  if (s->devSyncFlags) CUDACHECK(cudaFree(s->devSyncFlags));
  s->devSyncFlags = nullptr;
  s->maxNBlocks   = 0;
  NCCLCHECK(ncclCudaCalloc(&s->devSyncFlags, nBlocks));
  s->maxNBlocks = nBlocks;
  return ncclSuccess;
}

/* ── internal helpers ────────────────────────────────────────────────────── */

ncclResult_t ncclMscclEngineAddAlgo(ncclComm_t comm,
                                    const struct mscclEngineAlgo *hostAlgos,
                                    int nRanks,
                                    size_t minBytes, size_t maxBytes) {
  if (nRanks != comm->nRanks)  return ncclInvalidArgument;
  if (minBytes > maxBytes)     return ncclInvalidArgument;

  /* Lazily allocate the state struct. */
  if (!comm->mscclEngine) {
    NCCLCHECK(ncclCalloc(&comm->mscclEngine, 1));
  }
  struct mscclEngineState *s = comm->mscclEngine;

  if (s->nSizes >= MSCCL_ENGINE_MAX_SIZE_ENTRIES) return ncclInvalidUsage;

  const struct mscclEngineAlgo *myAlgo = &hostAlgos[comm->rank];
  struct mscclEngineSizeEntry *e = &s->sizes[s->nSizes];

  e->minBytes      = minBytes;
  e->maxBytes      = maxBytes;
  e->nBlocks       = myAlgo->nBlocks;
  e->nChunksPerLoop= myAlgo->nChunksPerLoop;
  e->nScratchChunks= myAlgo->nScratchChunks;

  ncclResult_t ret = ncclSuccess;
  NCCLCHECKGOTO(ncclCudaCalloc(&e->devAlgo, 1), ret, fail);
  CUDACHECKGOTO(cudaMemcpy(e->devAlgo, myAlgo, sizeof(*myAlgo),
                           cudaMemcpyHostToDevice), ret, fail);
  NCCLCHECKGOTO(ensureSyncFlags(s, e->nBlocks), ret, fail);

  s->nSizes++;
  INFO(NCCL_INIT,
       "MSCCL algo added: rank %d/%d nBlocks=%d nChunks=%d bytes=[%zu,%zu]",
       comm->rank, nRanks, e->nBlocks, e->nChunksPerLoop, minBytes, maxBytes);
  return ncclSuccess;

fail:
  if (e->devAlgo) { cudaFree(e->devAlgo); e->devAlgo = nullptr; }
  return ret;
}

ncclResult_t ncclMscclEngineFree(ncclComm_t comm) {
  struct mscclEngineState *s = comm->mscclEngine;
  if (!s) return ncclSuccess;
  for (int i = 0; i < s->nSizes; i++)
    if (s->sizes[i].devAlgo) cudaFree(s->sizes[i].devAlgo);
  if (s->devSyncFlags) cudaFree(s->devSyncFlags);
  if (s->devScratch)   cudaFree(s->devScratch);
  free(s);
  comm->mscclEngine = nullptr;
  return ncclSuccess;
}

ncclResult_t ncclAllReduceMscclImpl(const void *sendbuff, void *recvbuff,
                                size_t count, ncclDataType_t datatype,
                                ncclRedOp_t op,
                                ncclComm_t comm, cudaStream_t stream) {
  struct mscclEngineState *s = comm->mscclEngine;
  if (!s || s->nSizes == 0) return ncclInvalidUsage;

  ncclDevRedOp_t devOp;
  uint64_t redOpArg;
  NCCLCHECK(mscclGetRedOpArg(op, datatype, &devOp, &redOpArg));

  mscclKernelFn kfn = selectKernel(devOp, datatype);
  if (!kfn) return ncclInvalidArgument;

  size_t msgBytes = count * (size_t)ncclTypeSize(datatype);
  const struct mscclEngineSizeEntry *e = selectSizeEntry(s, msgBytes);
  if (!e) return ncclInvalidUsage;

  if (e->nScratchChunks > 0) {
    size_t scratchBytes = (size_t)e->nScratchChunks
                        * (count / e->nChunksPerLoop)
                        * ncclTypeSize(datatype);
    NCCLCHECK(ensureScratch(s, scratchBytes));
  }

  struct mscclEngineWork work;
  work.syncFlags        = s->devSyncFlags;
  work.scratchBuffer    = s->devScratch;
  work.sendBuff         = sendbuff;
  work.recvBuff         = recvbuff;
  work.sizePerMscclChunk= count / e->nChunksPerLoop;
  work.redOpArg         = redOpArg;
  work.redOpArgIsPtr    = 0;
  work.workIndex        = s->workIndex++;
  work.maxAllowedCount  = 1;
  work.nChunksPerLoop   = e->nChunksPerLoop;
  work.hasReduce        = 0;

  struct ncclKernelCommAndChannels *commAndChans =
    (struct ncclKernelCommAndChannels *)comm->devComm;

  INFO(NCCL_ALL, "MSCCL launch: rank=%d msgBytes=%zu nBlocks=%d nChunks=%d "
       "devAlgo=%p commAndChans=%p sendBuff=%p recvBuff=%p "
       "scratch=%p syncFlags=%p smem=%d",
       comm->rank, msgBytes, e->nBlocks, e->nChunksPerLoop,
       e->devAlgo, commAndChans, sendbuff, recvbuff,
       (void*)work.scratchBuffer, (void*)work.syncFlags,
       ncclShmemDynamicSize(comm->cudaArch));

  {
    struct mscclEngineAlgo hostAlgo;
    CUDACHECK(cudaMemcpy(&hostAlgo, e->devAlgo, sizeof(hostAlgo),
                          cudaMemcpyDeviceToHost));
    INFO(NCCL_ALL, "MSCCL algo: nBlocks=%d nTBs=%d nChunksPerLoop=%d "
         "nScratchChunks=%d nranks=%d",
         hostAlgo.nBlocks, hostAlgo.nTBs, hostAlgo.nChunksPerLoop,
         hostAlgo.nScratchChunks, hostAlgo.nranks);
    for (int b = 0; b < hostAlgo.nBlocks && b < 4; b++) {
      struct mscclEngineThreadBlock tb;
      CUDACHECK(cudaMemcpy(&tb, &e->devAlgo->mscclTBs[b], sizeof(tb),
                            cudaMemcpyDeviceToHost));
      INFO(NCCL_ALL, "MSCCL tb[%d]: channelId=%d sendPeer=%d recvPeer=%d "
           "nSteps=%d", b, tb.channelId, tb.sendPeer, tb.recvPeer, tb.nSteps);
      if (tb.channelId < 0 || tb.channelId >= comm->nChannels) {
        WARN("MSCCL tb[%d] channelId=%d out of range [0,%d)",
             b, tb.channelId, comm->nChannels);
      }
    }
  }

  int smem = ncclShmemDynamicSize(comm->cudaArch);
  mscclLaunchKernel(kfn, e->nBlocks, NCCL_MAX_NTHREADS, smem, stream,
                    commAndChans, e->devAlgo, work);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

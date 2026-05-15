/*
 * MSCCL engine — host-side launch glue (compiled by g++, not NVCC).
 */

#include "core.h"
#include "comm.h"
#include "collectives.h"   /* ncclTypeSize */
#include "alloc.h"         /* ncclCudaCalloc, ncclCalloc */
#include "device.h"        /* ncclShmemDynamicSize, NCCL_MAX_NTHREADS, NCCL_MAX_CONNS */
#include "proxy.h"
#include "msccl_engine_host.h"

#include <cstring>         /* memcpy */
#include <vector>

/* ── redOpArg ──────────────────────────────────────────────────────────── */
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

/* ── size entry selection ───────────────────────────────────────────────── */
static const struct mscclEngineSizeEntry *
selectSizeEntry(const struct mscclEngineState *s, size_t msgBytes) {
  const struct mscclEngineSizeEntry *best = nullptr;
  for (int i = 0; i < s->nSizes; i++) {
    const struct mscclEngineSizeEntry *e = &s->sizes[i];
    if (msgBytes >= e->minBytes && msgBytes <= e->maxBytes) {
      if (!best || e->maxBytes < best->maxBytes)
        best = e;
    }
  }
  if (best) return best;
  for (int i = 0; i < s->nSizes; i++) {
    const struct mscclEngineSizeEntry *e = &s->sizes[i];
    if (msgBytes <= e->maxBytes) {
      if (!best || e->maxBytes < best->maxBytes)
        best = e;
    }
  }
  if (best) return best;
  return &s->sizes[s->nSizes-1];
}

static int mscclProtocolChunkSteps(int protocol) {
  return protocol == NCCL_PROTO_SIMPLE ? MSCCL_ENGINE_CHUNKSTEPS : 1;
}

static int mscclProtocolSliceSteps(int protocol) {
  return protocol == NCCL_PROTO_SIMPLE ? MSCCL_ENGINE_SLICESTEPS : 1;
}

static int mscclProtocolStepSizeBytes(const struct ncclComm *comm, int protocol) {
  return comm->buffSizes[protocol] / NCCL_STEPS;
}

static int mscclProtocolDataStepSizeBytes(const struct ncclComm *comm, int protocol) {
  int stepSize = mscclProtocolStepSizeBytes(comm, protocol);
  if (protocol == NCCL_PROTO_LL) stepSize /= 2;
  if (protocol == NCCL_PROTO_LL128)
    stepSize = (stepSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  return stepSize;
}

static int mscclProtocolChunkSizeBytes(const struct ncclComm *comm, int protocol) {
  return mscclProtocolDataStepSizeBytes(comm, protocol) *
         mscclProtocolChunkSteps(protocol);
}

/* ── scratch/flag helpers ───────────────────────────────────────────────── */
static ncclResult_t ensureScratch(struct mscclEngineState *s, size_t bytes) {
  if (bytes <= s->scratchBytes) return ncclSuccess;
  if (s->devScratch) CUDACHECK(cudaFree(s->devScratch));
  s->devScratch   = nullptr;
  s->scratchBytes = 0;
  NCCLCHECK(ncclCudaCalloc((char **)&s->devScratch, bytes));
  s->scratchBytes = bytes;
  return ncclSuccess;
}

static ncclResult_t ensureSyncFlags(struct mscclEngineState *s, int nBlocks) {
  if (nBlocks <= s->maxNBlocks) return ncclSuccess;
  if (s->devSyncFlags) CUDACHECK(cudaFree(s->devSyncFlags));
  s->devSyncFlags = nullptr;
  s->maxNBlocks   = 0;
  NCCLCHECK(ncclCudaCalloc(&s->devSyncFlags, nBlocks));
  s->maxNBlocks = nBlocks;
  return ncclSuccess;
}

/* ── public functions ───────────────────────────────────────────────────── */

NCCL_API(ncclResult_t, ncclMscclEngineAddAlgo,
         ncclComm_t comm, const struct mscclEngineAlgo *hostAlgos,
         int nRanks, size_t minBytes, size_t maxBytes);
ncclResult_t ncclMscclEngineAddAlgo(ncclComm_t comm,
                                    const struct mscclEngineAlgo *hostAlgos,
                                    int nRanks,
                                    size_t minBytes, size_t maxBytes) {
  if (nRanks != comm->nRanks)  return ncclInvalidArgument;
  if (minBytes > maxBytes)     return ncclInvalidArgument;

  if (!comm->mscclEngine) {
    NCCLCHECK(ncclCalloc(&comm->mscclEngine, 1));
    comm->mscclEngine->workIndex = 1;  // start at 1 (flag 0 is the zero-init value)
  }
  struct mscclEngineState *s = comm->mscclEngine;

  if (s->nSizes >= MSCCL_ENGINE_MAX_SIZE_ENTRIES) return ncclInvalidUsage;

  const struct mscclEngineAlgo *myAlgo = &hostAlgos[comm->rank];
  if (myAlgo->protocol != NCCL_PROTO_SIMPLE &&
      myAlgo->protocol != NCCL_PROTO_LL &&
      myAlgo->protocol != NCCL_PROTO_LL128) {
    WARN("MSCCL algo has invalid protocol %d", myAlgo->protocol);
    return ncclInvalidArgument;
  }

  // Validate channel requirements: find the max channelId across all TBs.
  int maxChanId = -1;
  for (int b = 0; b < myAlgo->nBlocks; b++) {
    int cid = myAlgo->mscclTBs[b].channelId;
    if (cid > maxChanId) maxChanId = cid;
  }
  // Channels 0..maxChanId must have devPeers != NULL (initialized by
  // ncclTransportMscclConnect which extends beyond nChannels).
  if (maxChanId >= 0 && comm->channels[maxChanId].devPeers == nullptr) {
    WARN("MSCCL algo requires channelId %d but only %d channels are initialized. "
         "Set NCCL_MSCCL_MAX_CHANNELS=%d and rebuild, or reduce algo nchannels.",
         maxChanId, comm->nChannels, maxChanId + 1);
    return ncclInvalidArgument;
  }

  struct mscclEngineSizeEntry *e = &s->sizes[s->nSizes];

  e->minBytes      = minBytes;
  e->maxBytes      = maxBytes;
  e->nBlocks       = myAlgo->nBlocks;
  e->nChunksPerLoop= myAlgo->nChunksPerLoop;
  e->nScratchChunks= myAlgo->nScratchChunks;
  e->protocol      = myAlgo->protocol;
  e->hostAlgo      = myAlgo;
  e->nStripes      = 1;
  memset(e->connIndexRecv, 0, sizeof(e->connIndexRecv));
  memset(e->connIndexSend, 0, sizeof(e->connIndexSend));

  /* FIFO-stripe trigger: walk each TB's IR steps in order; for each TB,
   * accumulate per-step send-slot pressure on the (recvPeer-or-sendPeer,
   * direction) connection it uses, and stop counting when the TB hits its
   * first recv-side step. If both ranks' mirror TBs are "send-first" (the
   * common case in symmetric algos) and the pre-recv pressure exceeds
   * NCCL_STEPS on any connection, neither side reaches its recv before its
   * outbound FIFO fills — symmetric circular wait. Set nStripes large enough
   * that ceil(maxPressure / nStripes) <= NCCL_STEPS.
   *
   * Per-step slot pressure for a SEND-side step = count * chunkSteps,
   * independent of maxAllowedCount (which only affects how many separate
   * prims.send calls are issued, not total slot consumption since
   * Primitives' genericOp slices internally). */
  {
    int chunkSteps = mscclProtocolChunkSteps(myAlgo->protocol);
    int maxPressure = 0;
    for (int b = 0; b < myAlgo->nBlocks; b++) {
      const struct mscclEngineThreadBlock *tb = &myAlgo->mscclTBs[b];
      if (tb->channelId < 0) continue;
      int sendPressure = 0;
      for (int s = 0; s < tb->nSteps; s++) {
        uint8_t typ = tb->transmissions[s].type;
        bool isSend = (typ == MSCCL_ENGINE_SEND ||
                       typ == MSCCL_ENGINE_RECV_COPY_SEND ||
                       typ == MSCCL_ENGINE_RECV_REDUCE_SEND ||
                       typ == MSCCL_ENGINE_RECV_REDUCE_COPY_SEND);
        bool isRecv = (typ == MSCCL_ENGINE_RECV ||
                       typ == MSCCL_ENGINE_RECV_COPY_SEND ||
                       typ == MSCCL_ENGINE_RECV_REDUCE_SEND ||
                       typ == MSCCL_ENGINE_RECV_REDUCE_COPY ||
                       typ == MSCCL_ENGINE_RECV_REDUCE_COPY_SEND);
        if (isSend) sendPressure += (int)tb->transmissions[s].count * chunkSteps;
        if (isRecv) break;  /* first recv reached: peer's mirror starts to drain */
      }
      if (sendPressure > maxPressure) maxPressure = sendPressure;
    }
    if (maxPressure > NCCL_STEPS) {
      int stripes = (maxPressure + NCCL_STEPS - 1) / NCCL_STEPS;
      if (stripes > MSCCL_ENGINE_MAX_STRIPES) {
        WARN("MSCCL algo: nStripes=%d exceeds MSCCL_ENGINE_MAX_STRIPES=%d "
             "(maxPressure=%d, NCCL_STEPS=%d). Bump MSCCL_ENGINE_MAX_STRIPES "
             "and check flag bit budget in msccl_engine.cuh.",
             stripes, MSCCL_ENGINE_MAX_STRIPES, maxPressure, NCCL_STEPS);
        return ncclInvalidArgument;
      }
      e->nStripes = (uint16_t)stripes;
    }
    INFO(NCCL_INIT,
         "MSCCL algo stripe analysis: rank=%d maxSendPressure=%d NCCL_STEPS=%d "
         "chunkSteps=%d nStripes=%d",
         comm->rank, maxPressure, NCCL_STEPS, chunkSteps, (int)e->nStripes);
  }

  ncclResult_t ret = ncclSuccess;

  /* Compute per-TB connIndex{Recv,Send} so multiple TBs sharing the same
   * (channelId, peer, dir) tuple address distinct peers[peer]->{recv,send}[ci]
   * FIFOs. Without this, two TBs racing on the same connection slot deadlock
   * the producer/consumer step counters. We mutate a local copy of myAlgo
   * (the caller's struct is const) and ship that to the device. */
  {
    struct mscclEngineAlgo algoCopy;
    memcpy(&algoCopy, myAlgo, sizeof(algoCopy));

    /* nextXxxCi[channelId * nRanks + peer] = next free connIndex for that
     * bucket. channelId is bounded by MSCCL_ENGINE_MAX_TBS; peer by
     * comm->nRanks. Previously this was a [MSCCL_ENGINE_MAX_TBS][MSCCL_ENGINE_MAX_TBS]
     * stack array, which silently overflowed when nRanks > 64 (recvPeer could
     * be 64..nRanks-1 while the second dim was only 64). Now heap-allocated
     * and sized correctly by nRanks. */
    if (myAlgo->nBlocks > MSCCL_ENGINE_MAX_TBS) {
      WARN("MSCCL algo nBlocks=%d exceeds MSCCL_ENGINE_MAX_TBS=%d",
           myAlgo->nBlocks, MSCCL_ENGINE_MAX_TBS);
      return ncclInvalidArgument;
    }
    const int ciRanks = comm->nRanks;
    std::vector<int> nextRecvCi((size_t)MSCCL_ENGINE_MAX_TBS * ciRanks, 0);
    std::vector<int> nextSendCi((size_t)MSCCL_ENGINE_MAX_TBS * ciRanks, 0);

    int maxConnSeen = 0;
    for (int b = 0; b < myAlgo->nBlocks; b++) {
      const struct mscclEngineThreadBlock *src = &myAlgo->mscclTBs[b];
      struct mscclEngineThreadBlock *dst = &algoCopy.mscclTBs[b];
      int ch = src->channelId;
      if (ch < 0) {
        dst->connIndexRecv = 0;
        dst->connIndexSend = 0;
        continue;
      }
      if (ch >= MSCCL_ENGINE_MAX_TBS || src->recvPeer >= comm->nRanks
          || src->sendPeer >= comm->nRanks) {
        WARN("MSCCL algo TB[%d] has out-of-range channel=%d or peer (recv=%d send=%d nRanks=%d)",
             b, ch, src->recvPeer, src->sendPeer, comm->nRanks);
        return ncclInvalidArgument;
      }

      /* Only consume a connIndex slot if the TB actually issues network ops
       * in that direction. Pure NOP transmissions don't touch the FIFO. */
      bool usesRecv = false, usesSend = false;
      for (int t = 0; t < src->nSteps; t++) {
        uint8_t typ = src->transmissions[t].type;
        if (typ == MSCCL_ENGINE_RECV ||
            typ == MSCCL_ENGINE_RECV_COPY_SEND ||
            typ == MSCCL_ENGINE_RECV_REDUCE_SEND ||
            typ == MSCCL_ENGINE_RECV_REDUCE_COPY ||
            typ == MSCCL_ENGINE_RECV_REDUCE_COPY_SEND) usesRecv = true;
        if (typ == MSCCL_ENGINE_SEND ||
            typ == MSCCL_ENGINE_RECV_COPY_SEND ||
            typ == MSCCL_ENGINE_RECV_REDUCE_SEND ||
            typ == MSCCL_ENGINE_RECV_REDUCE_COPY_SEND) usesSend = true;
      }

      if (usesRecv && src->recvPeer >= 0) {
        int *cnt = &nextRecvCi[(size_t)ch * ciRanks + src->recvPeer];
        if (*cnt >= NCCL_MAX_CONNS) {
          WARN("MSCCL algo: too many TBs share (ch=%d, recvPeer=%d): need >%d "
               "connection slots but NCCL_MAX_CONNS=%d. Reduce TB sharing in "
               "the algorithm or bump NCCL_MAX_CONNS.",
               ch, src->recvPeer, *cnt, NCCL_MAX_CONNS);
          return ncclInvalidArgument;
        }
        dst->connIndexRecv = (uint8_t)(*cnt);
        if (*cnt + 1 > maxConnSeen) maxConnSeen = *cnt + 1;
        (*cnt)++;
      } else {
        dst->connIndexRecv = 0;
      }
      if (usesSend && src->sendPeer >= 0) {
        int *cnt = &nextSendCi[(size_t)ch * ciRanks + src->sendPeer];
        if (*cnt >= NCCL_MAX_CONNS) {
          WARN("MSCCL algo: too many TBs share (ch=%d, sendPeer=%d): need >%d "
               "connection slots but NCCL_MAX_CONNS=%d.",
               ch, src->sendPeer, *cnt, NCCL_MAX_CONNS);
          return ncclInvalidArgument;
        }
        dst->connIndexSend = (uint8_t)(*cnt);
        if (*cnt + 1 > maxConnSeen) maxConnSeen = *cnt + 1;
        (*cnt)++;
      } else {
        dst->connIndexSend = 0;
      }
      /* Mirror into side arrays so the host-side proxy-save loop (which sees
       * only the const e->hostAlgo == myAlgo) can dispatch to the correct
       * peers[peer]->{recv,send}[ci] FIFO. Without this, all proxy ops land on
       * slot 0 and any TB on slot 1 hangs waiting for its proxy. */
      e->connIndexRecv[b] = dst->connIndexRecv;
      e->connIndexSend[b] = dst->connIndexSend;
    }

    INFO(NCCL_INIT, "MSCCL algo connIndex assigned: rank %d nBlocks=%d maxConnIndex+1=%d",
         comm->rank, myAlgo->nBlocks, maxConnSeen);

    NCCLCHECKGOTO(ncclCudaCalloc(&e->devAlgo, 1), ret, fail);
    CUDACHECKGOTO(cudaMemcpy(e->devAlgo, &algoCopy, sizeof(algoCopy),
                             cudaMemcpyHostToDevice), ret, fail);
  }
  NCCLCHECKGOTO(ensureSyncFlags(s, e->nBlocks), ret, fail);

  s->nSizes++;
  INFO(NCCL_INIT,
       "MSCCL algo added: rank %d/%d nBlocks=%d nChunks=%d proto=%s bytes=[%zu,%zu]",
       comm->rank, nRanks, e->nBlocks, e->nChunksPerLoop,
       ncclProtoToString(e->protocol), minBytes, maxBytes);
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

  size_t msgBytes = count * (size_t)ncclTypeSize(datatype);
  const struct mscclEngineSizeEntry *e = selectSizeEntry(s, msgBytes);
  if (!e) return ncclInvalidUsage;
  if (e->nChunksPerLoop <= 0 || msgBytes % (size_t)e->nChunksPerLoop != 0)
    return ncclInvalidArgument;

  mscclKernelFn kfn = mscclSelectKernel((int)devOp, (int)datatype, e->protocol);
  if (!kfn) return ncclInvalidArgument;

  int stepSizeBytes = mscclProtocolStepSizeBytes(comm, e->protocol);
  int dataStepSizeBytes = mscclProtocolDataStepSizeBytes(comm, e->protocol);
  int chunkSizeBytes = mscclProtocolChunkSizeBytes(comm, e->protocol);
  int chunkSteps = mscclProtocolChunkSteps(e->protocol);
  if (stepSizeBytes <= 0 || dataStepSizeBytes <= 0 || chunkSizeBytes <= 0)
    return ncclInvalidArgument;
  size_t bytesPerMscclChunk = msgBytes / (size_t)e->nChunksPerLoop;
  uint32_t maxAllowedCount = 1;
  if (bytesPerMscclChunk > 0) {
    size_t chunkBytes = (size_t)chunkSizeBytes;
    size_t allowed = chunkBytes / bytesPerMscclChunk;
    if (allowed < 1) allowed = 1;
    if (allowed >= MSCCL_ENGINE_MAX_COUNT) allowed = MSCCL_ENGINE_MAX_COUNT - 1;
    maxAllowedCount = (uint32_t)allowed;
  }

  if (e->nScratchChunks > 0) {
    size_t scratchBytes = (size_t)e->nScratchChunks
                        * (count / e->nChunksPerLoop)
                        * ncclTypeSize(datatype);
    NCCLCHECK(ensureScratch(s, scratchBytes));
  }

  struct mscclEngineWork work;
  /* Zero sync flags before every launch so stale values from previous
   * algos/launches never alias into the current workIndex. */
  if (s->devSyncFlags) {
    CUDACHECK(cudaMemsetAsync(s->devSyncFlags, 0,
                              sizeof(struct mscclEngineFlag) * s->maxNBlocks,
                              stream));
  }

  work.syncFlags        = s->devSyncFlags;
  work.scratchBuffer    = s->devScratch;
  work.sendBuff         = sendbuff;
  work.recvBuff         = recvbuff;
  work.sizePerMscclChunk= count / e->nChunksPerLoop;
  work.redOpArg         = redOpArg;
  work.redOpArgIsPtr    = 0;
  work.workIndex        = s->workIndex++;
  work.maxAllowedCount  = maxAllowedCount;
  work.nChunksPerLoop   = e->nChunksPerLoop;
  work.nStripes         = e->nStripes ? e->nStripes : 1;
  work.hasReduce        = 0;

  struct ncclKernelCommAndChannels *commAndChans =
    (struct ncclKernelCommAndChannels *)comm->devComm;

  if (e->hostAlgo) {
    const struct mscclEngineAlgo *hostAlgo = e->hostAlgo;
    int stepSizeElts = dataStepSizeBytes / ncclTypeSize(datatype);
    ssize_t sizePerChunk = count / e->nChunksPerLoop;
    ssize_t chunkSizeElts = (ssize_t)(chunkSizeBytes / ncclTypeSize(datatype));
    ssize_t realChunkSize = (sizePerChunk < chunkSizeElts) ? sizePerChunk : chunkSizeElts;
    // Kernel iterates `for (gridOff=0; gridOff<sizePerChunk; gridOff+=chunkSize)`
    // with chunkSize == chunkSizeElts, so outer iters must be measured in chunk-
    // size units, not stepSize units.
    int outerIters = (chunkSizeElts > 0)
                     ? (int)((sizePerChunk + chunkSizeElts - 1) / chunkSizeElts)
                     : 1;
    if (outerIters < 1) outerIters = 1;

    uint64_t proxyOpCount = uint64_t(comm->sharedRes->collOpCount++) << 1;

    INFO(NCCL_ALL, "MSCCL proxy: rank=%d msgBytes=%zu nBlocks=%d nChunksPerLoop=%d "
         "proto=%s stepSizeElts=%d sizePerChunk=%ld outerIters=%d chunkSizeElts=%ld realChunkSize=%ld chunkSteps=%d maxAllowedCount=%u "
         "buffSize[proto]=%d NCCL_STEPS=%d nStripes=%d",
         comm->rank, msgBytes, hostAlgo->nBlocks, e->nChunksPerLoop,
         ncclProtoToString(e->protocol),
         stepSizeElts, (long)sizePerChunk, outerIters, (long)chunkSizeElts, (long)realChunkSize, chunkSteps,
         maxAllowedCount,
         comm->buffSizes[e->protocol], NCCL_STEPS, (int)work.nStripes);
    for (int b = 0; b < hostAlgo->nBlocks; b++) {
      const struct mscclEngineThreadBlock *tb = &hostAlgo->mscclTBs[b];
      INFO(NCCL_ALL, "MSCCL TB[%d]: ch=%d sendPeer=%d recvPeer=%d nSteps=%d",
           b, tb->channelId, tb->sendPeer, tb->recvPeer, tb->nSteps);
    }

    /* Bucket by (channel, peer, direction, connIndex). Multiple TBs that share
     * (ch,peer,dir) but use distinct connIndex slots must produce SEPARATE
     * proxy ops — slot 1's proxy thread won't advance unless we post work to
     * peers[peer]->{recv,send}[1]'s proxyConn directly. */
    struct { int ch; int peer; int dir; int ci; int nsteps; } seen[MSCCL_ENGINE_MAX_TBS * 2 * NCCL_MAX_CONNS];
    int nSeen = 0;

    for (int b = 0; b < hostAlgo->nBlocks; b++) {
      const struct mscclEngineThreadBlock *tb = &hostAlgo->mscclTBs[b];
      if (tb->channelId < 0) continue;
      for (int dir = 0; dir < 2; dir++) {
        int p = (dir == 0) ? tb->recvPeer : tb->sendPeer;
        if (p < 0) continue;

        int netSteps = 0;
        for (int s = 0; s < tb->nSteps; s++) {
          uint8_t type = tb->transmissions[s].type;
          bool isSend = (type == MSCCL_ENGINE_SEND ||
                         type == MSCCL_ENGINE_RECV_COPY_SEND ||
                         type == MSCCL_ENGINE_RECV_REDUCE_SEND ||
                         type == MSCCL_ENGINE_RECV_REDUCE_COPY_SEND);
          bool isRecv = (type == MSCCL_ENGINE_RECV ||
                         type == MSCCL_ENGINE_RECV_COPY_SEND ||
                         type == MSCCL_ENGINE_RECV_REDUCE_SEND ||
                         type == MSCCL_ENGINE_RECV_REDUCE_COPY ||
                         type == MSCCL_ENGINE_RECV_REDUCE_COPY_SEND);
           if ((dir == 1 && isSend) || (dir == 0 && isRecv)) {
             // The kernel splits each IR step's count into `nStripes` contiguous
             // slices (see msccl_engine.cuh) and issues one prims call per
             // non-empty stripe. Each call advances the proxy by chunkSteps
             // FIFO slots (4 for Simple, 1 for LL/LL128). The proxy must post
             // one irecv per slot the kernel will produce, so accumulate the
             // per-stripe slot count: sum_k DIVUP(span_k, maxAllowedCount) *
             // chunkSteps where span_k = (cnt*(k+1))/nS - (cnt*k)/nS.
             // For nStripes=1 this collapses to the un-striped formula.
             int nS = (int)(e->nStripes ? e->nStripes : 1);
             int cnt = (int)tb->transmissions[s].count;
             for (int k = 0; k < nS; k++) {
               int sStart = (cnt * k) / nS;
               int sEnd   = (cnt * (k + 1)) / nS;
               int span   = sEnd - sStart;
               if (span > 0) {
                 netSteps += (int)DIVUP(span, (int)maxAllowedCount) * chunkSteps;
               }
             }
           }
        }
        if (netSteps == 0) continue;

        int ci = (dir == 0) ? (int)e->connIndexRecv[b] : (int)e->connIndexSend[b];

        int idx = -1;
        for (int i = 0; i < nSeen; i++) {
          if (seen[i].ch == tb->channelId && seen[i].peer == p &&
              seen[i].dir == dir && seen[i].ci == ci) {
            idx = i;
            break;
          }
        }
        if (idx >= 0) {
          seen[idx].nsteps += netSteps * outerIters;
        } else {
          seen[nSeen].ch = tb->channelId;
          seen[nSeen].peer = p;
          seen[nSeen].dir = dir;
          seen[nSeen].ci = ci;
          seen[nSeen].nsteps = netSteps * outerIters;
          nSeen++;
        }
      }
    }

    for (int i = 0; i < nSeen; i++) {
      INFO(NCCL_ALL, "MSCCL proxy[%d]: ch=%d peer=%d dir=%s ci=%d nsteps=%d opCount=%lu "
	           "stepSizeElts=%d outerIters=%d chunkSteps=%d sizePerChunk=%ld chunkSizeElts=%ld realChunkSize=%ld",
	           i, seen[i].ch, seen[i].peer, seen[i].dir==0?"recv":"send", seen[i].ci,
	           seen[i].nsteps, (unsigned long)proxyOpCount,
	           stepSizeElts, outerIters, chunkSteps,
	           (long)sizePerChunk, (long)chunkSizeElts, (long)realChunkSize);
	      NCCLCHECK(ncclMscclProxySaveOp(comm, seen[i].ch, seen[i].peer, seen[i].dir,
	                                        seen[i].ci,
	                                        seen[i].nsteps, proxyOpCount, datatype, op,
	                                        e->protocol));
    }
    INFO(NCCL_ALL, "MSCCL proxy: posting %d ops, then ncclProxyStart", nSeen);
    NCCLCHECK(ncclProxyStart(comm));
  }

  int smem = ncclShmemDynamicSize(comm->cudaArch);

  {
    cudaError_t preErr = cudaGetLastError();
    if (preErr != cudaSuccess) {
      INFO(NCCL_ALL, "MSCCL WARNING: pre-launch cuda error cleared: %s", cudaGetErrorString(preErr));
    }
  }

  // MSCCL_DIAG: per-collective sync cudaMemcpy(D2H) of ncclConnInfo for every
  // (block, dir, connIndex). Useful when debugging stuck connections, but adds
  // hundreds of µs of host overhead per collective. Disabled by default.
#ifdef MSCCL_DIAG_CONNINFO
  if (e->hostAlgo) {
    for (int b = 0; b < e->nBlocks; b++) {
      const struct mscclEngineThreadBlock *tb = &e->hostAlgo->mscclTBs[b];
      int channelId = tb->channelId;
      if (channelId < 0) continue;
      struct ncclChannel *ch = &comm->channels[channelId];
      for (int dir = 0; dir < 2; dir++) {
        int peer = (dir == 0) ? tb->recvPeer : tb->sendPeer;
        if (peer < 0 || peer >= comm->nRanks) continue;
        if (!ch->devPeersHostPtr) {
          INFO(NCCL_ALL, "MSCCL DIAG TB[%d] ch=%d dir=%s peer=%d: devPeersHostPtr NULL!", b, channelId, dir==0?"recv":"send", peer);
          continue;
        }
        if (!ch->devPeersHostPtr[peer]) {
          INFO(NCCL_ALL, "MSCCL DIAG TB[%d] ch=%d dir=%s peer=%d: devPeersHostPtr[peer] NULL!", b, channelId, dir==0?"recv":"send", peer);
          continue;
        }
        struct ncclConnInfo hostConn[2];
        struct ncclDevChannelPeer *devPeer = ch->devPeersHostPtr[peer];
        struct ncclConnInfo *devConn0 = (dir == 0) ? &devPeer->recv[0] : &devPeer->send[0];
        struct ncclConnInfo *devConn1 = (dir == 0) ? &devPeer->recv[1] : &devPeer->send[1];
        cudaError_t cpyErr0 = cudaMemcpy(&hostConn[0], devConn0, sizeof(struct ncclConnInfo), cudaMemcpyDeviceToHost);
        cudaError_t cpyErr1 = cudaMemcpy(&hostConn[1], devConn1, sizeof(struct ncclConnInfo), cudaMemcpyDeviceToHost);
        for (int ci = 0; ci < 2; ci++) {
          cudaError_t cpyErr = (ci == 0) ? cpyErr0 : cpyErr1;
          if (cpyErr != cudaSuccess) {
            INFO(NCCL_ALL, "MSCCL DIAG TB[%d] ch=%d dir=%s peer=%d connIndex=%d: cudaMemcpy failed: %s", b, channelId, dir==0?"recv":"send", peer, ci, cudaGetErrorString(cpyErr));
          } else {
            INFO(NCCL_ALL, "MSCCL DIAG TB[%d] ch=%d dir=%s peer=%d connIndex=%d: step=%lu head=%p tail=%p buffs[SIMPLE]=%p stepSize=%d flags=%d connected=%d",
                 b, channelId, dir==0?"recv":"send", peer, ci,
                 (unsigned long)hostConn[ci].step,
                 (void*)hostConn[ci].head, (void*)hostConn[ci].tail,
                hostConn[ci].buffs[e->protocol],
                 hostConn[ci].stepSize, hostConn[ci].flags,
                 hostConn[ci].head != nullptr && hostConn[ci].tail != nullptr && hostConn[ci].buffs[e->protocol] != nullptr);
          }
        }
      }
    }
  }
#endif

  INFO(NCCL_ALL, "MSCCL launching kernel: nBlocks=%d nThreads=%d smem=%d kfn=%p "
       "commAndChans=%p devAlgo=%p workIndex=%u sizePerChunk=%zu",
       e->nBlocks, NCCL_MAX_NTHREADS, smem, (void*)kfn,
       (void*)commAndChans, (void*)e->devAlgo,
       work.workIndex, work.sizePerMscclChunk);

  mscclLaunchKernel(kfn, e->nBlocks, NCCL_MAX_NTHREADS, smem, stream,
                    commAndChans, e->devAlgo, work);
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    WARN("MSCCL kernel launch failed: %s", cudaGetErrorString(launchErr));
    return ncclUnhandledCudaError;
  }
  INFO(NCCL_ALL, "MSCCL kernel launched OK");
  return ncclSuccess;
}

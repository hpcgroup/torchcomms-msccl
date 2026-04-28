/*
 * MSCCL engine — host-side launch glue (compiled by g++, not NVCC).
 */

#include "core.h"
#include "comm.h"
#include "collectives.h"   /* ncclTypeSize */
#include "alloc.h"         /* ncclCudaCalloc, ncclCalloc */
#include "device.h"        /* ncclShmemDynamicSize, NCCL_MAX_NTHREADS */
#include "proxy.h"
#include "msccl_engine_host.h"

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
    if (e->maxBytes >= msgBytes)
      if (!best || e->maxBytes < best->maxBytes)
        best = e;
  }
  if (!best && s->nSizes > 0) best = &s->sizes[s->nSizes - 1];
  return best;
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
  }
  struct mscclEngineState *s = comm->mscclEngine;

  if (s->nSizes >= MSCCL_ENGINE_MAX_SIZE_ENTRIES) return ncclInvalidUsage;

  const struct mscclEngineAlgo *myAlgo = &hostAlgos[comm->rank];

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
  e->hostAlgo      = myAlgo;

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

  mscclKernelFn kfn = mscclSelectKernel((int)devOp, (int)datatype);
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

  if (e->hostAlgo) {
    const struct mscclEngineAlgo *hostAlgo = e->hostAlgo;
    int stepSizeElts = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / ncclTypeSize(datatype);
    ssize_t sizePerChunk = count / e->nChunksPerLoop;
    ssize_t chunkSizeElts = (ssize_t)stepSizeElts * MSCCL_ENGINE_CHUNKSTEPS;
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
         "stepSizeElts=%d sizePerChunk=%ld outerIters=%d chunkSizeElts=%ld realChunkSize=%ld chunkSteps=%d "
         "buffSize[SIMPLE]=%d NCCL_STEPS=%d",
         comm->rank, msgBytes, hostAlgo->nBlocks, e->nChunksPerLoop,
         stepSizeElts, (long)sizePerChunk, outerIters, (long)chunkSizeElts, (long)realChunkSize, MSCCL_ENGINE_CHUNKSTEPS,
         comm->buffSizes[NCCL_PROTO_SIMPLE], NCCL_STEPS);
    for (int b = 0; b < hostAlgo->nBlocks; b++) {
      const struct mscclEngineThreadBlock *tb = &hostAlgo->mscclTBs[b];
      INFO(NCCL_ALL, "MSCCL TB[%d]: ch=%d sendPeer=%d recvPeer=%d nSteps=%d",
           b, tb->channelId, tb->sendPeer, tb->recvPeer, tb->nSteps);
    }

    struct { int ch; int peer; int dir; int nsteps; } seen[MSCCL_ENGINE_MAX_TBS * 2];
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
             // Each prims.send/recv of one chunk-sized chunk advances the proxy
             // step counter by MSCCL_ENGINE_CHUNKSTEPS (= SlicePerChunk *
             // StepPerSlice). The IR's per-transmission `count` says how many
             // chunk-sized calls the kernel makes for this op.
             netSteps += (int)tb->transmissions[s].count * MSCCL_ENGINE_CHUNKSTEPS;
           }
        }
        if (netSteps == 0) continue;

        int idx = -1;
        for (int i = 0; i < nSeen; i++) {
          if (seen[i].ch == tb->channelId && seen[i].peer == p && seen[i].dir == dir) {
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
          seen[nSeen].nsteps = netSteps * outerIters;
          nSeen++;
        }
      }
    }

    for (int i = 0; i < nSeen; i++) {
      INFO(NCCL_ALL, "MSCCL proxy[%d]: ch=%d peer=%d dir=%s nsteps=%d opCount=%lu "
           "stepSizeElts=%d outerIters=%d chunkSteps=%d sizePerChunk=%ld chunkSizeElts=%ld realChunkSize=%ld",
           i, seen[i].ch, seen[i].peer, seen[i].dir==0?"recv":"send",
           seen[i].nsteps, (unsigned long)proxyOpCount,
           stepSizeElts, outerIters, MSCCL_ENGINE_CHUNKSTEPS,
           (long)sizePerChunk, (long)chunkSizeElts, (long)realChunkSize);
      NCCLCHECK(ncclMscclProxySaveOp(comm, seen[i].ch, seen[i].peer, seen[i].dir,
                                        seen[i].nsteps, proxyOpCount, datatype, op));
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
                 hostConn[ci].buffs[NCCL_PROTO_SIMPLE],
                 hostConn[ci].stepSize, hostConn[ci].flags,
                 hostConn[ci].head != nullptr && hostConn[ci].tail != nullptr && hostConn[ci].buffs[NCCL_PROTO_SIMPLE] != nullptr);
          }
        }
      }
    }
  }

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

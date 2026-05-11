/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "transport.h"
#include "bootstrap.h"
#include "channel.h"
#include "param.h"
#include <algorithm>

#include "meta/transport/transportConnect.h"

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  if (comm && comm->lazySetupChannels) {
    return ncclx::transportRingConnect(
        comm, comm->planner.algoMaxChannelsNeedConnect[NCCL_ALGO_RING]);
  }
  struct ringConnInfo {
    bool useNetPXN;
    bool useGdr;
  };
  struct ringConnInfo* ringInfo = NULL;
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    comm->useGdr = true;
    comm->useNetPXN = false;
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    if (ncclParamLocalRegister() || ncclParamGraphRegister()) {
      NCCLCHECK(ncclCalloc(&ringInfo, comm->nRanks));
      ringInfo[comm->rank].useGdr = comm->useGdr;
      ringInfo[comm->rank].useNetPXN = comm->useNetPXN;
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ringInfo, sizeof(struct ringConnInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (!ringInfo[i].useGdr) comm->useGdr = false;
        if (ringInfo[i].useNetPXN) comm->useNetPXN = true;
        if (comm->useGdr == false && comm->useNetPXN == true) break;
      }
    }
    INFO(NCCL_INIT, "Connected all rings, use ring PXN %d GDR %d", comm->useNetPXN, comm->useGdr);
  }
exit:
  free(ringInfo);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  if (comm && comm->lazySetupChannels) {
    return ncclx::transportTreeConnect(
        comm, comm->planner.algoMaxChannelsNeedConnect[NCCL_ALGO_TREE]);
  }
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  if (comm && comm->lazySetupChannels) {
    return ncclx::transportPatConnect(comm, comm->planner.algoMaxChannelsNeedConnect[NCCL_ALGO_PAT]);
  }
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int mask=1; mask<comm->nRanks; mask<<=1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    INFO(NCCL_INIT, "Connected binomial trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

NCCL_PARAM(MscclMaxChannels, "MSCCL_MAX_CHANNELS", 16);
// Number of distinct connection slots (peers[peer]->{recv,send}[connIndex])
// to allocate per (channel, peer, direction). Algorithms whose IR maps
// multiple thread blocks to the same (channel, peer, dir) — e.g. the
// node-aware deepseekflash medium/large algos — need >1 so each TB owns
// its own FIFO. Hard-capped by NCCL_MAX_CONNS (=2). Default 2: matches the
// hard cap and unblocks all currently-shipped MSCCL algos at the cost of
// allocating an extra (often unused) FIFO per pair.
NCCL_PARAM(MscclNumConnsPerPeer, "MSCCL_NUM_CONNS_PER_PEER", 2);

// Connect all rank pairs on all channels so any MSCCL algorithm (arbitrary
// send/recv peer topology) works without per-launch transport init.
// Also initializes channels beyond nChannels up to NCCL_MSCCL_MAX_CHANNELS
// so that MSCCL algos requiring more channels than the auto-tuned ring/tree
// count don't dereference NULL devPeers pointers.
ncclResult_t ncclTransportMscclConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    int maxCh = std::max(comm->nChannels, (int)ncclParamMscclMaxChannels());
    if (maxCh > MAXCHANNELS) maxCh = MAXCHANNELS;

    int numConns = (int)ncclParamMscclNumConnsPerPeer();
    if (numConns < 1) numConns = 1;
    if (numConns > NCCL_MAX_CONNS) numConns = NCCL_MAX_CONNS;

    // Initialize channels beyond nChannels that MSCCL algos may need.
    for (int c = comm->nChannels; c < maxCh; c++) {
      if (comm->channels[c].id == -1) {
        NCCLCHECKGOTO(initChannel(comm, c), ret, fail);
      }
    }

    /* Pass the ring graph so NET transport sets shared=0 even at connIndex>=1.
     * net.cc::sendSetup forces shared=1 when (graph==NULL && connIndex!=0)
     * which routes the connection through the PXN shared-buffer protocol —
     * incompatible with our raw per-conn FIFO usage and segfaults the proxy
     * progress thread on the first cross-node op. With graph non-null, the
     * `graph || connIndex == 0` guard short-circuits and shared=0 for every
     * connIndex, making slot 1 behave identically to slot 0. */
    struct ncclTopoGraph* graph = &comm->graphs[NCCL_ALGO_RING];
    for (int ci = 0; ci < numConns; ci++) {
      for (int peer = 0; peer < comm->nRanks; peer++) {
        if (peer == comm->rank) continue;
        for (int c = 0; c < maxCh; c++) {
          NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &peer, 1, &peer, ci), ret, fail);
        }
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, graph, ci), ret, fail);
    }
    INFO(NCCL_INIT, "Connected MSCCL all-to-all channels (%d ranks x %d channels x %d conns, nChannels=%d)",
         comm->nRanks, maxCh, numConns, comm->nChannels);
  }
exit:
  return ret;
fail:
  goto exit;
}

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
#include "msccl_engine.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

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
// 1: MSCCL NET connections use each GPU's own closest NIC. 0: use the NIC the
// ring graph assigns to the channel (previous behavior).
NCCL_PARAM(MscclNetLocalNic, "MSCCL_NET_LOCAL_NIC", 1);
// 1: with no MSCCL algo descriptor (MSCCL_ALGO_PATH unset or no algo for nRanks), still connect every
// peer on every channel x NCCL_MSCCL_NUM_CONNS_PER_PEER slots (previous behaviour). 0 (default): skip it.
// The all-to-all connect claims connIndex 1 with shared=0 buffers, which plain NCCL p2p (it uses
// connIndex 1 with shared buffers) then reuses: sendrecv_perf segfaults on the first p2p op.
NCCL_PARAM(MscclConnectAll, "MSCCL_CONNECT_ALL", 0);

/* Sparse used-peer set computed by dlopening MSCCL_ALGO_PATH .so's at
 * MsccConnect time and walking algos[myRank].mscclTBs. Avoids the
 * O(nRanks × maxChannels × numConns) all-to-all that breaks the libfabric
 * QP / pinned-mem budget past 64 GPUs.
 *
 * Symmetry is required: rank A's recv[c, B] must match rank B's send[c, A]
 * (else P2pSetup hangs in bootstrap exchange at iteration i = (B-A) mod N).
 * IR codegen produces symmetric algos so we trust this. */
struct MscclUsedPeers {
  bool valid;
  int  maxChNeeded;      /* one past the highest channelId referenced */
  std::vector<uint8_t> chanUsed;   /* size MAXCHANNELS */
  std::vector<uint8_t> recv;       /* size MAXCHANNELS * nRanks */
  std::vector<uint8_t> send;       /* size MAXCHANNELS * nRanks */
};

static MscclUsedPeers mscclLoadUsedPeerSet(struct ncclComm* comm) {
  MscclUsedPeers up;
  up.valid = false;
  up.maxChNeeded = 0;

  const char* pathEnv = getenv("MSCCL_ALGO_PATH");
  if (!pathEnv || !pathEnv[0]) return up;

  up.chanUsed.assign(MAXCHANNELS, 0);
  up.recv.assign((size_t)MAXCHANNELS * comm->nRanks, 0);
  up.send.assign((size_t)MAXCHANNELS * comm->nRanks, 0);

  /* tokenize a local copy; strtok_r mutates */
  std::vector<char> buf(strlen(pathEnv) + 1);
  memcpy(buf.data(), pathEnv, buf.size());

  int loaded = 0, skippedMismatched = 0;
  char* save = nullptr;
  for (char* tok = strtok_r(buf.data(), ":", &save); tok;
            tok = strtok_r(nullptr, ":", &save)) {
    void* h = dlopen(tok, RTLD_NOW | RTLD_LOCAL);
    if (!h) continue;
    const struct mscclAlgoDescriptor* desc =
        (const struct mscclAlgoDescriptor*)dlsym(h, "msccl_algo_desc");
    if (!desc || !desc->algos) {
      dlclose(h);
      continue;
    }
    if (desc->nranks != comm->nRanks) {
      skippedMismatched++;
      dlclose(h);
      continue;
    }
    const struct mscclEngineAlgo* me = &desc->algos[comm->rank];
    for (int b = 0; b < me->nBlocks; b++) {
      const struct mscclEngineThreadBlock* tb = &me->mscclTBs[b];
      int c = tb->channelId;
      if (c < 0 || c >= MAXCHANNELS) continue;
      up.chanUsed[c] = 1;
      if (c + 1 > up.maxChNeeded) up.maxChNeeded = c + 1;
      int rp = tb->recvPeer;
      int sp = tb->sendPeer;
      if (rp >= 0 && rp < comm->nRanks && rp != comm->rank)
        up.recv[(size_t)c * comm->nRanks + rp] = 1;
      if (sp >= 0 && sp < comm->nRanks && sp != comm->rank)
        up.send[(size_t)c * comm->nRanks + sp] = 1;
    }
    dlclose(h);
    loaded++;
  }
  if (loaded == 0) {
    up.recv.clear(); up.send.clear(); up.chanUsed.clear();
    if (skippedMismatched > 0) {
      INFO(NCCL_INIT, "MSCCL sparse-connect: %d algo(s) skipped (nranks mismatch); "
           "falling back to all-to-all", skippedMismatched);
    }
    return up;
  }
  up.valid = true;
  return up;
}

// Connect rank pairs on all channels needed by MSCCL algorithms. When
// MSCCL_ALGO_PATH is set and contains at least one descriptor matching
// comm->nRanks, only the (channel, peer, dir) tuples actually referenced
// by the algo IR are connected — saving up to (nRanks-1) × MAXCHANNELS ×
// numConns wasted connections per rank at 128+ GPU scales. Falls back to
// all-to-all (capped by NCCL_MSCCL_MAX_CHANNELS) when no descriptor is
// available.
ncclResult_t ncclTransportMscclConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    MscclUsedPeers used = mscclLoadUsedPeerSet(comm);

    int maxCh;
    if (used.valid) {
      maxCh = std::max(comm->nChannels, used.maxChNeeded);
    } else {
      maxCh = std::max(comm->nChannels, (int)ncclParamMscclMaxChannels());
    }
    if (maxCh > MAXCHANNELS) maxCh = MAXCHANNELS;

    int numConns = (int)ncclParamMscclNumConnsPerPeer();
    if (numConns < 1) numConns = 1;
    if (numConns > NCCL_MAX_CONNS) numConns = NCCL_MAX_CONNS;

    // Initialize channels beyond nChannels that MSCCL algos may need.
    // In sparse mode, only init channels that some TB references — others
    // can stay uninitialized since we won't connect any peers on them.
    for (int c = comm->nChannels; c < maxCh; c++) {
      if (used.valid && !used.chanUsed[c]) continue;
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
     * connIndex, making slot 1 behave identically to slot 0.
     *
     * Side effect of passing the graph: ncclTopoGetNetDev then picks the NIC
     * from graph->inter[] by channel, not by GPU, so every local GPU sending
     * on the same MSCCL channel shares one NIC, usually not its own.
     * mscclNetLocalNic makes net.cc pick the GPU's own closest NIC (no PXN)
     * instead, as plain p2p would. NCCL_MSCCL_NET_LOCAL_NIC=0 restores the
     * old channel-based NIC choice. */
    struct ncclTopoGraph* graph = &comm->graphs[NCCL_ALGO_RING];
    comm->mscclNetLocalNic = ncclParamMscclNetLocalNic() != 0;
    INFO(NCCL_INIT|NCCL_NET, "MSCCL connect: rank=%d NET NIC selection=%s",
         comm->rank, comm->mscclNetLocalNic ? "per-GPU local NIC" : "ring-graph channel NIC");
    if (used.valid) {
      /* Accumulate intra/inter-node tallies via connectionSummary so we can
       * report per-rank net-fanout at INFO level. The per-(channel,peer,dir)
       * setup is also logged by net.cc::{sendSetup,recvSetup} at NCCL_NET,
       * but that's too verbose to enable in normal runs — this summary gives
       * a one-line picture per rank to verify sparse-connect is doing what
       * we expect (counts should match the algo IR's peer fan-out). */
      connectionSummary cumSummary;
      /* Per-channel usage counters: how many distinct peers this rank
       * connects per channel, split intra/inter. Helps spot channel
       * imbalance — a node-aware algo should show intra > 0 only on
       * local-stage channels. */
      std::vector<int> perChIntra(maxCh, 0), perChInter(maxCh, 0);
      for (int ci = 0; ci < numConns; ci++) {
        for (int peer = 0; peer < comm->nRanks; peer++) {
          if (peer == comm->rank) continue;
          bool sameNode = (comm->rankToNode[comm->rank] == comm->rankToNode[peer]);
          for (int c = 0; c < maxCh; c++) {
            if (!used.chanUsed[c]) continue;
            int nrecv = used.recv[(size_t)c * comm->nRanks + peer] ? 1 : 0;
            int nsend = used.send[(size_t)c * comm->nRanks + peer] ? 1 : 0;
            if (!nrecv && !nsend) continue;
            NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, nrecv, &peer, nsend, &peer, ci, &cumSummary), ret, fail);
            if (ci == 0) {
              if (sameNode) perChIntra[c] += (nrecv + nsend);
              else          perChInter[c] += (nrecv + nsend);
            }
          }
        }
        NCCLCHECKGOTO(ncclTransportP2pSetup(comm, graph, ci), ret, fail);
      }
      INFO(NCCL_INIT|NCCL_NET,
           "MSCCL sparse-connect summary: rank=%d nRanks=%d nChannels=%d maxCh=%d "
           "numConns=%d intraSend=%d intraRecv=%d interSend=%d interRecv=%d",
           comm->rank, comm->nRanks, comm->nChannels, maxCh, numConns,
           cumSummary.intraSend, cumSummary.intraRecv,
           cumSummary.interSend, cumSummary.interRecv);
      /* Channel-by-channel breakdown — only log channels that actually have
       * connections, to keep the line manageable at maxCh=32+. */
      char chBuf[1024]; chBuf[0] = '\0';
      size_t off = 0;
      for (int c = 0; c < maxCh; c++) {
        if (perChIntra[c] == 0 && perChInter[c] == 0) continue;
        int w = snprintf(chBuf + off, sizeof(chBuf) - off,
                         " ch%d=intra%d/inter%d", c, perChIntra[c], perChInter[c]);
        if (w < 0 || (size_t)w >= sizeof(chBuf) - off) break;
        off += (size_t)w;
      }
      INFO(NCCL_INIT|NCCL_NET,
           "MSCCL sparse-connect per-channel (rank=%d, ci=0):%s",
           comm->rank, chBuf);
    } else if (ncclParamMscclConnectAll() == 0) {
      INFO(NCCL_INIT, "MSCCL connect: rank=%d no MSCCL algo descriptor; skipping all-to-all connect "
           "(NCCL_MSCCL_CONNECT_ALL=1 restores it)", comm->rank);
    } else {
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
  }
exit:
  if (comm) comm->mscclNetLocalNic = false;
  return ret;
fail:
  goto exit;
}

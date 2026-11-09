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

#include <algorithm>
#include <cstdio>          /* snprintf */
#include <cstring>         /* memcpy, strcmp */
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

/* ── size entry selection ─────────────────────────────────────────────────
 * Exact match: the registered entry whose [minBytes, maxBytes] holds msgBytes (the one with the smallest
 * maxBytes when several do), else nullptr. This loop is the whole matched path and is the same code as
 * before runtime_v3b.
 *
 * Until runtime_v3b an unmatched size fell back to the entry with the smallest maxBytes >= msgBytes, and
 * else to the last entry. A `-b 256 -e 1G` sweep with one exact-size algo then ran that algo's schedule
 * at every size without a word, and a timing could belong to another size's IR
 * (collectives_evolve docs/runtime_issues/README.md, "Smaller items" and "Pool data that this engine
 * cannot have produced"). Now an unmatched size is an error (ncclAllReduceMscclImpl below).
 * NCCL_MSCCL_SIZE_FALLBACK=1 restores the old choice, with a WARN. */
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
  return best;
}

/* The pre-v3b fallback for a size that no entry matches. Used only with NCCL_MSCCL_SIZE_FALLBACK=1. */
static const struct mscclEngineSizeEntry *
selectSizeEntryFallback(const struct mscclEngineState *s, size_t msgBytes) {
  const struct mscclEngineSizeEntry *best = nullptr;
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

NCCL_PARAM(MscclMaxStripes, "MSCCL_MAX_STRIPES", MSCCL_ENGINE_MAX_STRIPES);
/* 0 (default) = a message size that no registered entry matches is an error. 1 = pre-v3b fallback. */
NCCL_PARAM(MscclSizeFallback, "MSCCL_SIZE_FALLBACK", 0);

/* Unmatched message size (cold path: never runs when an entry matches). Names the size and every
 * registered entry, then either rejects the call or, with NCCL_MSCCL_SIZE_FALLBACK=1, returns the
 * pre-v3b fallback entry. */
static const struct mscclEngineSizeEntry *
mscclUnmatchedSize(const struct ncclComm *comm, const struct mscclEngineState *s, size_t msgBytes) {
  char list[1024];
  int off = 0;
  list[0] = '\0';
  for (int i = 0; i < s->nSizes && off < (int)sizeof(list) - 1; i++) {
    const struct mscclEngineSizeEntry *e = &s->sizes[i];
    int n = snprintf(list + off, sizeof(list) - off, "%s#%d [%zu,%zu] %s nChunksPerLoop=%d",
                     i ? "; " : "", i, e->minBytes, e->maxBytes, ncclProtoToString(e->protocol),
                     e->nChunksPerLoop);
    if (n < 0) break;
    off += n;
  }
  if (off >= (int)sizeof(list)) memcpy(list + sizeof(list) - 5, " ...", 5);
  if (ncclParamMscclSizeFallback() != 1) {
    WARN("MSCCL: no registered algo entry matches msgBytes=%zu (rank %d, %d registered: %s); rejecting. "
         "The engine does not run another size's schedule (NCCL_MSCCL_SIZE_FALLBACK=1 restores that).",
         msgBytes, comm->rank, s->nSizes, list);
    return nullptr;
  }
  const struct mscclEngineSizeEntry *e = selectSizeEntryFallback(s, msgBytes);
  WARN("MSCCL: no registered algo entry matches msgBytes=%zu (rank %d); NCCL_MSCCL_SIZE_FALLBACK=1, so "
       "running entry #%d [%zu,%zu], a schedule built for another size (%d registered: %s)",
       msgBytes, comm->rank, (int)(e - s->sizes), e->minBytes, e->maxBytes, s->nSizes, list);
  return e;
}

/* ── runtime_v3: stripe count (R4a) and per-stripe flags (R4b) ─────────────
 *
 * NCCL_MSCCL_STRIPE_COUNT = addalgo | launch (default launch)
 *   addalgo: nStripes from this rank's algo at AddAlgo time, P = sum(count * chunkSteps) over the send
 *            steps before a TB's first recv (runtime_v2b, R4).
 *   launch:  nStripes per launch from the merged-call pressure P = sum(DIVUP(count, maxAllowedCount) *
 *            chunkSteps) over the same steps, i.e. the FIFO slots the kernel really takes (it issues one
 *            prims call per maxAllowedCount chunks, each call takes chunkSteps slots). Max over ALL ranks
 *            of the IR, so every rank uses the same value (the per-connection call counts must match).
 * NCCL_MSCCL_STRIPE_FLAGS = step | stripe | auto (default auto)
 *   step:    flags per (iter, step); wait at stripe 0, publish at the last stripe (runtime_v2b).
 *   stripe:  flags per (iter * nStripes + stripe, step); publish every stripe; aligned dependencies
 *            (MSCCL_ENGINE_DEP_ALIGNED, mscclStripeAnalyze below) wait at every stripe for the
 *            producer's same stripe, the others at stripe 0 for the producer's last stripe.
 *   auto:    per nStripes, stripe flags only where step flags deadlock (the step-order flag graph has a
 *            cycle) and the aligned marks remove the cycle; step flags everywhere else. Keeps runtime_v2b's
 *            schedule (and timing) on every launch that runs on runtime_v2b. */
static int mscclStripeFlagsMode() {
  static int mode = -1;
  if (mode < 0) {
    const char *v = ncclGetEnv("NCCL_MSCCL_STRIPE_FLAGS");
    if (v == nullptr || v[0] == '\0' || strcmp(v, "auto") == 0 || strcmp(v, "2") == 0) mode = 2;
    else if (strcmp(v, "stripe") == 0 || strcmp(v, "1") == 0) mode = 1;
    else if (strcmp(v, "step") == 0 || strcmp(v, "0") == 0) mode = 0;
    else { WARN("NCCL_MSCCL_STRIPE_FLAGS=%s is not step|stripe|auto; using auto", v); mode = 2; }
  }
  return mode;
}

static const char *mscclStripeFlagsName(int mode) {
  return mode == 0 ? "step" : (mode == 1 ? "stripe" : "auto");
}

static int mscclStripeCountLaunchMode() {
  static int mode = -1;
  if (mode < 0) {
    const char *v = ncclGetEnv("NCCL_MSCCL_STRIPE_COUNT");
    if (v == nullptr || v[0] == '\0' || strcmp(v, "launch") == 0 || strcmp(v, "1") == 0) mode = 1;
    else if (strcmp(v, "addalgo") == 0 || strcmp(v, "0") == 0) mode = 0;
    else { WARN("NCCL_MSCCL_STRIPE_COUNT=%s is not addalgo|launch; using launch", v); mode = 1; }
  }
  return mode;
}

static bool mscclIsSendType(uint8_t typ) {
  return typ == MSCCL_ENGINE_SEND || typ == MSCCL_ENGINE_RECV_COPY_SEND ||
         typ == MSCCL_ENGINE_RECV_REDUCE_SEND || typ == MSCCL_ENGINE_RECV_REDUCE_COPY_SEND;
}
static bool mscclIsRecvType(uint8_t typ) {
  return typ == MSCCL_ENGINE_RECV || typ == MSCCL_ENGINE_RECV_COPY_SEND ||
         typ == MSCCL_ENGINE_RECV_REDUCE_SEND || typ == MSCCL_ENGINE_RECV_REDUCE_COPY ||
         typ == MSCCL_ENGINE_RECV_REDUCE_COPY_SEND;
}

/* FIFO-stripe trigger pressure of one rank's algo: max over TBs (channelId >= 0) of the send-slot
 * pressure accumulated over the TB's steps in order, up to and including its first recv-type step
 * (whose send part, if any, still counts). merge = 1 gives the AddAlgo formula (count * chunkSteps);
 * merge = maxAllowedCount gives the merged-call formula DIVUP(count, merge) * chunkSteps. */
static int mscclSendPressure(const struct mscclEngineAlgo *a, int chunkSteps, int merge) {
  int maxPressure = 0;
  for (int b = 0; b < a->nBlocks; b++) {
    const struct mscclEngineThreadBlock *tb = &a->mscclTBs[b];
    if (tb->channelId < 0) continue;
    int p = 0;
    for (int s = 0; s < tb->nSteps; s++) {
      uint8_t typ = tb->transmissions[s].type;
      if (mscclIsSendType(typ)) p += (int)DIVUP((int)tb->transmissions[s].count, merge) * chunkSteps;
      if (mscclIsRecvType(typ)) break;
    }
    if (p > maxPressure) maxPressure = p;
  }
  return maxPressure;
}

static int mscclStripesFromPressure(int pressure) {
  int stripes = pressure > NCCL_STEPS ? (pressure + NCCL_STEPS - 1) / NCCL_STEPS : 1;
  int stripeCap = (int)ncclParamMscclMaxStripes();   /* diagnostic cap (9ba302f) */
  if (stripeCap >= 1 && stripes > stripeCap) stripes = stripeCap;
  return stripes;
}

/* One local buffer access of a step. Input and output are one buffer (the HW runs are in place,
 * sendbuff == recvbuff); scratch is separate. The remote side of a send/recv is not a local access. */
struct mscclAccess { uint8_t buf; uint8_t write; int32_t off; int32_t cnt; };

static int mscclStepAccesses(const struct mscclEngineTransmission *t, struct mscclAccess out[2]) {
  int c = (int)t->count;
  if (c <= 0) return 0;
  struct mscclAccess rd = { (uint8_t)(t->srcBuffer == MSCCL_ENGINE_SCRATCH_BUF), 0, t->srcOffset, c };
  struct mscclAccess wr = { (uint8_t)(t->dstBuffer == MSCCL_ENGINE_SCRATCH_BUF), 1, t->dstOffset, c };
  switch (t->type) {
    case MSCCL_ENGINE_SEND:
    case MSCCL_ENGINE_RECV_REDUCE_SEND:      out[0] = rd; return 1;
    case MSCCL_ENGINE_RECV:
    case MSCCL_ENGINE_RECV_COPY_SEND:        out[0] = wr; return 1;
    case MSCCL_ENGINE_RECV_REDUCE_COPY:
    case MSCCL_ENGINE_RECV_REDUCE_COPY_SEND:
    case MSCCL_ENGINE_LOCAL_COPY:
    case MSCCL_ENGINE_REDUCE:                /* dst is read and written; the write covers both */
      out[0] = rd; out[1] = wr; return 2;
    default: return 0;                       /* NOP */
  }
}

/* Stripe that moves position q (0-based) of a step with n chunks: the kernel's stripe k covers
 * [floor(n*k/S), floor(n*(k+1)/S)) (msccl_engine.cuh), so k = ceil((q+1)*S/n) - 1. */
static inline int mscclStripeOf(int q, int n, int S) {
  return (int)(((int64_t)(q + 1) * S + n - 1) / n) - 1;
}

struct mscclStripeResult {
  int depsTotal = 0, depsCandidate = 0, depsAligned = 0, demoted = 0, intraHazards = 0, unordered = 0;
  int skipped = 0;
  int cycleStep = 0;   /* the flag graph with every dependency non-aligned (runtime_v2b order) has a cycle */
  int cycle = 0;       /* the final flag graph has a cycle: every launch with this nStripes deadlocks */
  const char *reason = "";
  /* aligned[b][depPtr + d] */
  std::vector<std::vector<uint8_t>> aligned;
};

/* R4b alignment analysis for one rank's algo and one stripe count S > 1.
 *
 * Model. A TB runs its events (stripe, step) in lexicographic order. A dependency (consumer TB C,
 * step i) on (producer TB P, kernel step d) is satisfied by the first publish of P with (subIter,
 * step) >= the goal; P publishes at every dep-source step j (hasDependence) with step value pub(j).
 * So the edge's producer event is step j* = min{ j : hasDependence(j), pub(j) >= d } of P:
 *   aligned edge:     (P, s, j*)   -> (C, s, i) for every stripe s
 *   non-aligned edge: (P, S-1, j*) -> (C, 0, i)
 * Unstriped (S = 1) happens-before (HB) is program order + edges (P, j*) -> (C, i).
 *
 * Rule. Every pair of accesses of one chunk position (same buffer, at least one write) by two steps
 * of different TBs that is ordered in the unstriped HB must stay ordered in the striped HB, at the
 * stripes that move that position. This covers the edges msccl-tools drops because an earlier step
 * of the consumer TB already implies them (msccl-tools ir.py:247-255): the dropped edge's hazard is
 * still a pair ordered by unstriped HB, so it is checked against whatever striped path remains.
 *
 * Algorithm. Candidates: edges whose own hazard is aligned (the producer step and the consumer's next
 * non-nop step share a chunk position with a write, and every shared position is in producer stripe
 * <= consumer stripe; no shared position = not aligned). Then repeat: build striped HB (vector clocks),
 * check every ordered conflicting pair; for a violated pair (A = (a, i) HB B = (b, j)) demote one
 * aligned edge into TB b at a consumer step <= j whose producer event is HB-after-or-equal A (an edge
 * from A's own TB if there is one, else the one nearest to j; one per pass for the first 256 passes,
 * then at most one per consumer TB per pass).
 * Demoting any such edge fixes that pair, by the lemma below.
 * Lemma: unstriped (X, x) HB= (Y, y) implies striped (X, s, x) HB (Y, S-1, y) for every s (by
 * induction over program order, aligned and non-aligned edges). So once all those edges are
 * non-aligned, every unstriped path A -> B enters b through a non-aligned edge and the pair holds. So
 * while a violation exists such an edge is still aligned: each pass demotes at least one edge, and the
 * loop ends within (number of edges) passes.
 * Same-TB pairs that the stripe replay reorders (an earlier step's position in a later stripe than a
 * later step's) cannot be fixed by flags; they are counted (intraHazards) and reported. runtime_v2b
 * has the same reordering. Unordered conflicting pairs (races in the unstriped IR) are counted only.
 * Any doubt (unresolvable dependency, cycle, work budget) = no aligned edges (runtime_v2b order). */
static void mscclStripeAnalyze(const struct mscclEngineAlgo *a, int S, struct mscclStripeResult *res) {
  const int nB = a->nBlocks;
  res->aligned.assign(nB, std::vector<uint8_t>(MSCCL_ENGINE_MAX_STEPS, 0));
  std::vector<int> nSt(nB);
  for (int b = 0; b < nB; b++) nSt[b] = a->mscclTBs[b].nSteps;

  /* kernel step values at publish time (msccl_engine.cuh: step += numDeps-1, REDUCE += numRed-1) */
  std::vector<std::vector<int>> pub(nB);
  for (int b = 0; b < nB; b++) {
    const struct mscclEngineThreadBlock *tb = &a->mscclTBs[b];
    pub[b].resize(nSt[b]);
    int step = 0;
    for (int i = 0; i < nSt[b]; i++) {
      const struct mscclEngineTransmission *t = &tb->transmissions[i];
      if (t->numDependencies > 0) step += t->numDependencies - 1;
      if (t->type == MSCCL_ENGINE_REDUCE) step += (int)t->numReductions - 1;
      pub[b][i] = step;
      step++;
    }
  }

  /* dependency edges */
  struct Edge { int c, i, slot, p, j; uint8_t al; };
  std::vector<Edge> edges;
  std::vector<std::vector<std::vector<int>>> inEdges(nB);   /* inEdges[b][i] = edge ids */
  for (int b = 0; b < nB; b++) inEdges[b].resize(nSt[b]);
  for (int b = 0; b < nB; b++) {
    const struct mscclEngineThreadBlock *tb = &a->mscclTBs[b];
    for (int i = 0; i < nSt[b]; i++) {
      const struct mscclEngineTransmission *t = &tb->transmissions[i];
      for (int d = 0; d < t->numDependencies; d++) {
        int slot = t->dependencePointer + d;
        res->depsTotal++;
        if (slot < 0 || slot >= MSCCL_ENGINE_MAX_STEPS) { res->skipped = 1; res->reason = "dep slot out of range"; continue; }
        int p = tb->dependentBid[slot];
        int ds = tb->dependentStep[slot];
        if (p < 0 || p >= nB || p == b) { res->skipped = 1; res->reason = "dep block out of range"; continue; }
        int jstar = -1;
        for (int j = 0; j < nSt[p]; j++)
          if (a->mscclTBs[p].transmissions[j].hasDependence && pub[p][j] >= ds) { jstar = j; break; }
        if (jstar < 0) { res->skipped = 1; res->reason = "dep never published"; continue; }
        inEdges[b][i].push_back((int)edges.size());
        edges.push_back({b, i, slot, p, jstar, 0});
      }
    }
  }
  if (res->skipped) return;

  /* work budget: striped vector clocks hold sum(S * nSteps) * nB entries */
  int64_t evU = 0;
  for (int b = 0; b < nB; b++) evU += nSt[b];
  if (evU * S * nB > (int64_t)32 * 1024 * 1024) { res->skipped = 1; res->reason = "vector-clock budget"; return; }

  /* unstriped HB: vcU[b][i * nB + x] = highest step of TB x that HB-or-equals (b, i), -1 none */
  std::vector<std::vector<int16_t>> vcU(nB);
  for (int b = 0; b < nB; b++) vcU[b].assign((size_t)nSt[b] * nB, -1);
  {
    std::vector<int> ptr(nB, 0);
    int64_t done = 0;
    bool progress = true;
    while (progress) {
      progress = false;
      for (int b = 0; b < nB; b++) {
        while (ptr[b] < nSt[b]) {
          int i = ptr[b];
          bool ready = true;
          for (int e : inEdges[b][i]) if (ptr[edges[e].p] <= edges[e].j) { ready = false; break; }
          if (!ready) break;
          int16_t *v = &vcU[b][(size_t)i * nB];
          if (i > 0) memcpy(v, &vcU[b][(size_t)(i - 1) * nB], sizeof(int16_t) * nB);
          for (int e : inEdges[b][i]) {
            const int16_t *u = &vcU[edges[e].p][(size_t)edges[e].j * nB];
            for (int x = 0; x < nB; x++) v[x] = std::max(v[x], u[x]);
          }
          v[b] = (int16_t)i;
          ptr[b]++; done++; progress = true;
        }
      }
    }
    if (done != evU) { res->skipped = 1; res->reason = "dependency cycle"; return; }
  }
  auto hbU = [&](int xa, int ia, int xb, int ib) -> bool {   /* (xa, ia) HB-or-equal (xb, ib) */
    return vcU[xb][(size_t)ib * nB + xa] >= ia;
  };

  /* accesses: one record per chunk position */
  struct Rec { int64_t key; int16_t b, i; int16_t q, n; uint8_t w; };
  std::vector<Rec> recs;
  for (int b = 0; b < nB; b++) {
    for (int i = 0; i < nSt[b]; i++) {
      struct mscclAccess acc[2];
      int na = mscclStepAccesses(&a->mscclTBs[b].transmissions[i], acc);
      for (int k = 0; k < na; k++)
        for (int q = 0; q < acc[k].cnt; q++)
          recs.push_back({ ((int64_t)acc[k].buf << 32) | (uint32_t)(acc[k].off + q),
                           (int16_t)b, (int16_t)i, (int16_t)q, (int16_t)acc[k].cnt, acc[k].write });
      if ((int64_t)recs.size() > (int64_t)16 * 1024 * 1024) { res->skipped = 1; res->reason = "access budget"; return; }
    }
  }
  std::sort(recs.begin(), recs.end(), [](const Rec &x, const Rec &y) { return x.key < y.key; });

  /* candidates: each edge's own hazard (producer step j* vs the consumer's next non-nop step) */
  for (Edge &e : edges) {
    const struct mscclEngineThreadBlock *ctb = &a->mscclTBs[e.c];
    int k = e.i;
    while (k < nSt[e.c] && ctb->transmissions[k].type == MSCCL_ENGINE_NOP) k++;
    if (k >= nSt[e.c]) continue;
    struct mscclAccess pa[2], ca[2];
    int np = mscclStepAccesses(&a->mscclTBs[e.p].transmissions[e.j], pa);
    int nc = mscclStepAccesses(&ctb->transmissions[k], ca);
    bool conflict = false, ok = true;
    for (int x = 0; x < np && ok; x++)
      for (int y = 0; y < nc && ok; y++) {
        if (pa[x].buf != ca[y].buf || !(pa[x].write || ca[y].write)) continue;
        int lo = std::max(pa[x].off, ca[y].off), hi = std::min(pa[x].off + pa[x].cnt, ca[y].off + ca[y].cnt);
        for (int pos = lo; pos < hi; pos++) {
          conflict = true;
          if (mscclStripeOf(pos - pa[x].off, pa[x].cnt, S) > mscclStripeOf(pos - ca[y].off, ca[y].cnt, S)) { ok = false; break; }
        }
      }
    if (conflict && ok) { e.al = 1; res->depsCandidate++; }
  }

  /* verify / demote, until a pass finds no violation. */
  std::vector<std::vector<int32_t>> vcS(nB);
  int64_t pairWork = 0;
  /* Striped HB: event (b, s, i) at position s * nSt[b] + i; vcS[b][pos * nB + x] = highest position of
   * TB x that HB-or-equals it. Returns false if the striped flag graph has a cycle (a certain deadlock:
   * every edge is a flag wait the kernel really does). allNonAligned = ignore the marks. */
  auto buildStriped = [&](bool allNonAligned, bool withVC) -> bool {
    if (withVC) for (int b = 0; b < nB; b++) vcS[b].assign((size_t)S * nSt[b] * nB, -1);
    std::vector<int> ptr(nB, 0);
    int64_t done = 0;
    bool progress = true;
    while (progress) {
      progress = false;
      for (int b = 0; b < nB; b++) {
        while (ptr[b] < S * nSt[b]) {
          int pos = ptr[b], s = pos / nSt[b], i = pos % nSt[b];
          bool ready = true;
          for (int e : inEdges[b][i]) {
            const Edge &E = edges[e];
            bool al = E.al && !allNonAligned;
            int need = al ? s * nSt[E.p] + E.j : (s == 0 ? (S - 1) * nSt[E.p] + E.j : -1);
            if (need >= 0 && ptr[E.p] <= need) { ready = false; break; }
          }
          if (!ready) break;
          if (withVC) {
            int32_t *v = &vcS[b][(size_t)pos * nB];
            if (pos > 0) memcpy(v, &vcS[b][(size_t)(pos - 1) * nB], sizeof(int32_t) * nB);
            for (int e : inEdges[b][i]) {
              const Edge &E = edges[e];
              bool al = E.al && !allNonAligned;
              int need = al ? s * nSt[E.p] + E.j : (s == 0 ? (S - 1) * nSt[E.p] + E.j : -1);
              if (need < 0) continue;
              const int32_t *u = &vcS[E.p][(size_t)need * nB];
              for (int x = 0; x < nB; x++) v[x] = std::max(v[x], u[x]);
            }
            v[b] = pos;
          }
          ptr[b]++; done++; progress = true;
        }
      }
    }
    return done == evU * S;
  };
  res->cycleStep = buildStriped(true, false) ? 0 : 1;

  for (int round = 0; ; round++) {
    if (round > (int)edges.size() + 1) { res->skipped = 1; res->reason = "demotion did not converge"; break; }
    if (!buildStriped(false, true)) {
      res->cycle = 1; res->skipped = 1; res->reason = "striped flag graph has a cycle (certain deadlock)";
      break;
    }

    int nDemote = 0, intra = 0, unordered = 0;
    bool stuck = false;   /* a violation with no aligned edge left to demote */
    std::vector<uint8_t> demotedTb(nB, 0);
    for (size_t g0 = 0; g0 < recs.size(); ) {
      size_t g1 = g0;
      while (g1 < recs.size() && recs[g1].key == recs[g0].key) g1++;
      for (size_t x = g0; x < g1; x++) {
        for (size_t y = x + 1; y < g1; y++) {
          const Rec &X = recs[x], &Y = recs[y];
          if (!(X.w || Y.w)) continue;
          if (X.b == Y.b && X.i == Y.i) continue;
          if (++pairWork > (int64_t)256 * 1024 * 1024) { res->skipped = 1; res->reason = "pair budget"; goto out; }
          const Rec *A, *B;
          if (X.b == Y.b) { A = X.i < Y.i ? &X : &Y; B = X.i < Y.i ? &Y : &X; }
          else if (hbU(X.b, X.i, Y.b, Y.i)) { A = &X; B = &Y; }
          else if (hbU(Y.b, Y.i, X.b, X.i)) { A = &Y; B = &X; }
          else { unordered++; continue; }
          int kA = mscclStripeOf(A->q, A->n, S), kB = mscclStripeOf(B->q, B->n, S);
          if (A->b == B->b) {
            if (kA > kB) intra++;
            continue;
          }
          int posA = kA * nSt[A->b] + A->i, posB = kB * nSt[B->b] + B->i;
          if (vcS[B->b][(size_t)posB * nB + A->b] >= posA) continue;
          /* violated. Demote ONE aligned edge into B's TB: the one at the highest consumer step <= B.i whose
           * producer event is HB-after-or-equal A (one of these edges carries the lost order). The pass's
           * striped HB is stale after a demotion, so the first 256 passes demote one edge each (fewest
           * demotions: on pool IR 35fb12ce, 10 instead of 16 with batching), later passes at most one per
           * consumer TB (bounded time). The lemma guarantees a qualifying edge while the violation exists. */
          if (round < 256 ? nDemote > 0 : demotedTb[B->b] != 0) continue;
          int before = nDemote;
          /* Prefer an edge straight from A's TB (the dropped-edge case), then any qualifying edge. */
          for (int pass = 0; pass < 2 && nDemote == before; pass++)
            for (int ci = B->i; ci >= 0 && nDemote == before; ci--)
              for (int e : inEdges[B->b][ci]) {
                Edge &E = edges[e];
                if (!E.al) continue;
                if (pass == 0 && E.p != A->b) continue;
                if (E.p == A->b ? E.j >= A->i : hbU(A->b, A->i, E.p, E.j)) {
                  E.al = 0; nDemote++; demotedTb[B->b] = 1; break;
                }
              }
          if (nDemote == before) stuck = true;   /* cannot happen (lemma); fall back to no marks */
        }
      }
      g0 = g1;
    }
    res->intraHazards = intra;
    res->unordered = unordered;
    res->demoted += nDemote;
    if (nDemote == 0 && stuck) { res->skipped = 1; res->reason = "violation without a demotable edge"; break; }
    if (nDemote == 0) break;
  }
out:
  if (res->skipped) {
    for (Edge &e : edges) e.al = 0;
    res->cycle = res->cycleStep;   /* the final flags are then runtime_v2b's order */
  }
  res->depsAligned = 0;
  for (const Edge &e : edges)
    if (e.al) { res->aligned[e.c][e.slot] = 1; res->depsAligned++; }
}

/* ── scratch/flag helpers ───────────────────────────────────────────────── */

/* Free a device allocation made with ncclCudaCalloc, and clear the pointer. With cuMem on (the default
 * where the driver supports it) ncclCudaCalloc maps cuMemCreate memory, which cudaFree cannot release:
 * the call fails and the memory stays mapped until the process ends. 1a44eab fixed this for the regrow
 * paths (ensureScratch / ensureSyncFlags); this is the same fix for the AddAlgo error path and for comm
 * teardown. A failure is logged and the caller goes on, so one bad free cannot stop the teardown. */
template <typename T>
static void mscclDevFree(T *&ptr, const char *what) {
  if (ptr == nullptr) return;
  ncclResult_t r = ncclCudaFree(ptr);
  if (r != ncclSuccess) WARN("MSCCL: ncclCudaFree(%s %p) failed with %d", what, (void *)ptr, (int)r);
  ptr = nullptr;
}

static ncclResult_t ensureScratch(struct mscclEngineState *s, size_t bytes) {
  if (bytes <= s->scratchBytes) return ncclSuccess;
  if (s->devScratch) NCCLCHECK(ncclCudaFree(s->devScratch));
  s->devScratch   = nullptr;
  s->scratchBytes = 0;
  NCCLCHECK(ncclCudaCalloc((char **)&s->devScratch, bytes));
  s->scratchBytes = bytes;
  return ncclSuccess;
}

static ncclResult_t ensureSyncFlags(struct mscclEngineState *s, int nBlocks) {
  if (nBlocks <= s->maxNBlocks) return ncclSuccess;
  if (s->devSyncFlags) NCCLCHECK(ncclCudaFree(s->devSyncFlags));
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
  /* Every rejection below says why at NCCL_DEBUG=WARN (v3b): the caller (ncclmsccl_preload) only sees the
   * return code, and before v3b three of these returns were silent. */
  if (nRanks != comm->nRanks) {
    WARN("MSCCL algo is for %d ranks but the communicator has %d; rejecting the algo", nRanks, comm->nRanks);
    return ncclInvalidArgument;
  }
  if (minBytes > maxBytes) {
    WARN("MSCCL algo has minBytes=%zu > maxBytes=%zu; rejecting the algo", minBytes, maxBytes);
    return ncclInvalidArgument;
  }
  /* Array bounds of EVERY rank's algo, before any loop walks mscclTBs[] or transmissions[]: the stripe
   * table below reads all ranks. Before v3b only this rank's nBlocks was checked, and only after the
   * channel and stripe loops had already walked mscclTBs[0..nBlocks). */
  for (int r = 0; r < nRanks; r++) {
    const struct mscclEngineAlgo *a = &hostAlgos[r];
    if (a->nBlocks < 0 || a->nBlocks > MSCCL_ENGINE_MAX_TBS) {
      WARN("MSCCL algo: rank %d has nBlocks=%d, outside [0, MSCCL_ENGINE_MAX_TBS=%d]; rejecting the algo",
           r, a->nBlocks, MSCCL_ENGINE_MAX_TBS);
      return ncclInvalidArgument;
    }
    for (int b = 0; b < a->nBlocks; b++) {
      if (a->mscclTBs[b].nSteps > MSCCL_ENGINE_MAX_STEPS) {
        WARN("MSCCL algo: rank %d TB %d has nSteps=%d > MSCCL_ENGINE_MAX_STEPS=%d; rejecting the algo",
             r, b, (int)a->mscclTBs[b].nSteps, MSCCL_ENGINE_MAX_STEPS);
        return ncclInvalidArgument;
      }
    }
  }

  if (!comm->mscclEngine) {
    NCCLCHECK(ncclCalloc(&comm->mscclEngine, 1));
    comm->mscclEngine->workIndex = 1;  // start at 1 (flag 0 is the zero-init value)
  }
  struct mscclEngineState *s = comm->mscclEngine;

  if (s->nSizes >= MSCCL_ENGINE_MAX_SIZE_ENTRIES) {
    WARN("MSCCL engine already holds MSCCL_ENGINE_MAX_SIZE_ENTRIES=%d algo entries; rejecting the algo",
         MSCCL_ENGINE_MAX_SIZE_ENTRIES);
    return ncclInvalidUsage;
  }

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
   * Slot pressure of a SEND-side step. The kernel issues one prims call per
   * maxAllowedCount chunks of a stripe's slice (msccl_engine.cuh, the `for c`
   * loop), and every call takes chunkSteps FIFO slots whatever its size
   * (Simple: SlicePerChunk=2 slices of StepPerSlice=2 steps; LL: 1 step). So
   * the real pressure of one step is DIVUP(count, maxAllowedCount) *
   * chunkSteps, and it DOES depend on maxAllowedCount, which is known only per
   * launch. (runtime_v2b's comment here said the opposite: that the pressure
   * is count * chunkSteps, independent of maxAllowedCount. That is the
   * AddAlgo formula below; it overestimates the pressure whenever
   * maxAllowedCount > 1, R4a.)
   *
   * addalgo (NCCL_MSCCL_STRIPE_COUNT=addalgo, runtime_v2b): count * chunkSteps,
   *   this rank's algo only, fixed at AddAlgo.
   * launch (default): DIVUP(count, m) * chunkSteps for every possible
   *   maxAllowedCount m, max over ALL ranks of the IR (hostAlgos[0..nRanks)),
   *   so every rank picks the same value for a given size. */
  {
    e->stripeFlags = (uint8_t)mscclStripeFlagsMode();
    e->stripeCountLaunch = (uint8_t)mscclStripeCountLaunchMode();
    e->nStripeVariants = 0;
    e->nLoggedSizes = 0;
    int chunkSteps = mscclProtocolChunkSteps(myAlgo->protocol);
    int maxPressure = mscclSendPressure(myAlgo, chunkSteps, 1);
    if (maxPressure > NCCL_STEPS) {
      /* Diagnostic cap (NCCL_MSCCL_MAX_STRIPES, default MSCCL_ENGINE_MAX_STRIPES = no change).
       * 1 disables striping, to test whether a hang comes from the IR's FIFO pressure or from the
       * stripe replay itself (stripe-0 dependency waits need the producer's last-stripe publish). */
      int stripes = mscclStripesFromPressure(maxPressure);
      if (stripes > MSCCL_ENGINE_MAX_STRIPES) {
        if (!e->stripeCountLaunch) {
          WARN("MSCCL algo: nStripes=%d exceeds MSCCL_ENGINE_MAX_STRIPES=%d "
               "(maxPressure=%d, NCCL_STEPS=%d). Bump MSCCL_ENGINE_MAX_STRIPES "
               "and check flag bit budget in msccl_engine.cuh.",
               stripes, MSCCL_ENGINE_MAX_STRIPES, maxPressure, NCCL_STEPS);
          return ncclInvalidArgument;
        }
        stripes = 0;   /* unused with STRIPE_COUNT=launch */
      }
      e->nStripes = (uint16_t)stripes;
    }
    INFO(NCCL_INIT,
         "MSCCL algo stripe analysis: rank=%d maxSendPressure=%d NCCL_STEPS=%d "
         "chunkSteps=%d nStripes=%d",
         comm->rank, maxPressure, NCCL_STEPS, chunkSteps, (int)e->nStripes);

    /* R4a table, and the AddAlgo formula over all ranks as a cross-check of the per-rank value. */
    int addAlgoMax = 1;
    for (int r = 0; r < nRanks; r++) {
      int st = mscclStripesFromPressure(mscclSendPressure(&hostAlgos[r], chunkSteps, 1));
      addAlgoMax = std::max(addAlgoMax, st);
    }
    e->nStripesAddAlgoMax = (uint16_t)std::min(addAlgoMax, 65535);
    e->nStripesForCount[0] = 0;
    int launchMin = 1 << 30, launchMax = 0;
    for (int m = 1; m < MSCCL_ENGINE_MAX_COUNT; m++) {
      int ns = 1;
      for (int r = 0; r < nRanks; r++)
        ns = std::max(ns, mscclStripesFromPressure(mscclSendPressure(&hostAlgos[r], chunkSteps, m)));
      e->nStripesForCount[m] = (uint16_t)(ns > MSCCL_ENGINE_MAX_STRIPES ? 0 : ns);
      launchMin = std::min(launchMin, ns);
      launchMax = std::max(launchMax, ns);
    }
    INFO(NCCL_INIT,
         "MSCCL v3 stripes: rank=%d NCCL_MSCCL_STRIPE_FLAGS=%s NCCL_MSCCL_STRIPE_COUNT=%s "
         "nStripes addalgo(this rank)=%d addalgo(max over %d ranks)=%d launch(m=1)=%d launch(m=%d)=%d "
         "launch range over m=1..%d [%d,%d]",
         comm->rank, mscclStripeFlagsName(e->stripeFlags), e->stripeCountLaunch ? "launch" : "addalgo",
         (int)e->nStripes, nRanks, addAlgoMax, (int)e->nStripesForCount[1],
         MSCCL_ENGINE_MAX_COUNT - 1, (int)e->nStripesForCount[MSCCL_ENGINE_MAX_COUNT - 1],
         MSCCL_ENGINE_MAX_COUNT - 1, launchMin, launchMax);
    if (!e->stripeCountLaunch && addAlgoMax != (int)e->nStripes)
      WARN("MSCCL algo: rank %d nStripes=%d (AddAlgo formula, this rank) differs from the max over the IR's "
           "ranks (%d); peers then split steps differently (NCCL_MSCCL_STRIPE_COUNT=launch uses the max)",
           comm->rank, (int)e->nStripes, addAlgoMax);
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

    /* R4b: alignment marks for every stripe count this entry can launch with. The marks are used only
     * with stripe flags; with step flags the analysis only logs (and predicts an R4 deadlock). */
    {
      std::vector<int> counts;
      if (e->stripeCountLaunch) {
        for (int m = 1; m < MSCCL_ENGINE_MAX_COUNT; m++) counts.push_back(e->nStripesForCount[m]);
      } else {
        counts.push_back(e->nStripes);
      }
      std::sort(counts.begin(), counts.end());
      counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
      for (int S : counts) {
        if (S <= 1 || e->nStripeVariants >= MSCCL_ENGINE_MAX_STRIPES) continue;
        struct mscclStripeResult r;
        mscclStripeAnalyze(&algoCopy, S, &r);
        auto &v = e->stripeVariant[e->nStripeVariants++];
        v.nStripes = (uint16_t)S;
        v.depsTotal = r.depsTotal; v.depsCandidate = r.depsCandidate; v.depsAligned = r.depsAligned;
        v.demoted = r.demoted; v.intraHazards = r.intraHazards; v.unordered = r.unordered;
        v.skipped = r.skipped;
        v.devAlgo = nullptr;
        v.cycleStep = r.cycleStep; v.cycle = r.cycle;
        INFO(NCCL_INIT,
             "MSCCL v3 R4b: rank=%d nStripes=%d flags=%s deps aligned=%d/%d (candidates %d, demoted %d) "
             "sameTbReorderPairs=%d unorderedPairs=%d flagCycle step=%d stripe=%d%s%s",
             comm->rank, S, mscclStripeFlagsName(e->stripeFlags), r.depsAligned, r.depsTotal, r.depsCandidate,
             r.demoted, r.intraHazards, r.unordered, r.cycleStep, r.cycle,
             r.skipped ? " analysis skipped: " : "", r.skipped ? r.reason : "");
        if (e->stripeFlags ? r.cycle : r.cycleStep)
          WARN("MSCCL algo: rank %d nStripes=%d: the cross-block flag graph (%s flags) has a cycle, so every "
               "launch that uses this stripe count deadlocks (R4)", comm->rank, S,
               mscclStripeFlagsName(e->stripeFlags));
        if (r.intraHazards > 0)
          WARN("MSCCL algo: rank %d nStripes=%d: the stripe replay reorders %d same-TB access pairs "
               "(a later step touches a chunk before an earlier step's stripe does); data may be wrong. "
               "runtime_v2b does the same; flags cannot fix it.", comm->rank, S, r.intraHazards);
        if (r.depsAligned == 0 || !e->stripeFlags) continue;
        std::vector<char> buf(sizeof(struct mscclEngineAlgo));
        struct mscclEngineAlgo *marked = (struct mscclEngineAlgo *)buf.data();
        memcpy(marked, &algoCopy, sizeof(algoCopy));
        for (int b = 0; b < marked->nBlocks; b++)
          for (int k = 0; k < MSCCL_ENGINE_MAX_STEPS; k++)
            if (r.aligned[b][k])
              marked->mscclTBs[b].dependentBid[k] =
                  (int8_t)(marked->mscclTBs[b].dependentBid[k] | MSCCL_ENGINE_DEP_ALIGNED);
        NCCLCHECKGOTO(ncclCudaCalloc(&v.devAlgo, 1), ret, fail);
        CUDACHECKGOTO(cudaMemcpy(v.devAlgo, marked, sizeof(*marked), cudaMemcpyHostToDevice), ret, fail);
      }
    }
  }
  NCCLCHECKGOTO(ensureSyncFlags(s, e->nBlocks), ret, fail);

  s->nSizes++;
  INFO(NCCL_INIT,
       "MSCCL algo added: rank %d/%d nBlocks=%d nChunks=%d proto=%s bytes=[%zu,%zu]",
       comm->rank, nRanks, e->nBlocks, e->nChunksPerLoop,
       ncclProtoToString(e->protocol), minBytes, maxBytes);
  return ncclSuccess;

fail:
  mscclDevFree(e->devAlgo, "devAlgo");
  for (int v = 0; v < e->nStripeVariants; v++)
    mscclDevFree(e->stripeVariant[v].devAlgo, "stripe variant devAlgo");
  e->nStripeVariants = 0;
  return ret;
}

ncclResult_t ncclMscclEngineFree(ncclComm_t comm) {
  struct mscclEngineState *s = comm->mscclEngine;
  if (!s) return ncclSuccess;
  /* Runs once per communicator, from commFree (init.cc), after the last collective has finished. */
  for (int i = 0; i < s->nSizes; i++) {
    mscclDevFree(s->sizes[i].devAlgo, "devAlgo");
    for (int v = 0; v < s->sizes[i].nStripeVariants; v++)
      mscclDevFree(s->sizes[i].stripeVariant[v].devAlgo, "stripe variant devAlgo");
  }
  mscclDevFree(s->devSyncFlags, "devSyncFlags");
  mscclDevFree(s->devScratch, "devScratch");
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
  if (!e) {   /* same test as before v3b; the body is new */
    e = mscclUnmatchedSize(comm, s, msgBytes);
    if (!e) return ncclInvalidArgument;
  }
  /* The kernel splits the ELEMENT count into nChunksPerLoop chunks (sizePerMscclChunk below) and has no
   * remainder path, so the rule is on elements. Checking bytes let e.g. 12300 B (3075 floats) with
   * nChunksPerLoop=12 run and leave the last 3 elements unreduced, and count < nChunksPerLoop ran an
   * empty kernel. */
  if (e->nChunksPerLoop <= 0 || count % (size_t)e->nChunksPerLoop != 0 || count < (size_t)e->nChunksPerLoop) {
    WARN("MSCCL: element count %zu is not a positive multiple of nChunksPerLoop=%d (msgBytes=%zu); rejecting",
         count, e->nChunksPerLoop, msgBytes);
    return ncclInvalidArgument;
  }

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

  /* Stripe count and device algo for this launch (R4a / R4b). */
  int nStripes = 1;
  if (e->stripeCountLaunch) {
    nStripes = e->nStripesForCount[maxAllowedCount];
    if (nStripes == 0) {
      WARN("MSCCL: nStripes for maxAllowedCount=%u exceeds MSCCL_ENGINE_MAX_STRIPES=%d (msgBytes=%zu); rejecting",
           maxAllowedCount, MSCCL_ENGINE_MAX_STRIPES, msgBytes);
      return ncclInvalidArgument;
    }
  } else {
    nStripes = e->nStripes ? e->nStripes : 1;
  }
  const struct mscclEngineAlgo *launchAlgo = e->devAlgo;
  int variant = -1;
  bool useStripeFlags = false;   /* with one stripe both orders are the same: take the step path */
  if (e->stripeFlags && nStripes > 1) {
    for (int v = 0; v < e->nStripeVariants; v++)
      if (e->stripeVariant[v].nStripes == nStripes) { variant = v; break; }
    useStripeFlags = e->stripeFlags == 1 ||
        (variant >= 0 && e->stripeVariant[variant].cycleStep && !e->stripeVariant[variant].cycle);
    if (useStripeFlags && variant >= 0 && e->stripeVariant[variant].devAlgo)
      launchAlgo = e->stripeVariant[variant].devAlgo;
  }
  {
    /* Flag sub-iteration budget (MSCCL_ENGINE_MAX_ITER, include/msccl_engine.h). */
    size_t perChunk = count / (size_t)e->nChunksPerLoop;
    size_t chunkElts = (size_t)chunkSizeBytes / ncclTypeSize(datatype);
    size_t iters = chunkElts ? (perChunk + chunkElts - 1) / chunkElts : 1;
    size_t subIters = iters * (e->stripeFlags ? (size_t)nStripes : 1);
    if (subIters > (size_t)MSCCL_ENGINE_MAX_ITER) {
      WARN("MSCCL: outerIters=%zu x nStripes=%d = %zu flag sub-iterations exceed MSCCL_ENGINE_MAX_ITER=%u "
           "(msgBytes=%zu); rejecting", iters, nStripes, subIters, (unsigned)MSCCL_ENGINE_MAX_ITER, msgBytes);
      return ncclInvalidArgument;
    }
  }
  {
    /* Once per rank, algo entry and size: the stripe decision. */
    struct mscclEngineSizeEntry *em = const_cast<struct mscclEngineSizeEntry *>(e);
    bool logged = false;
    for (int k = 0; k < em->nLoggedSizes; k++) if (em->loggedSizes[k] == msgBytes) logged = true;
    if (!logged) {
      if (em->nLoggedSizes < 8) em->loggedSizes[em->nLoggedSizes++] = msgBytes;
      int addalgoNs = e->nStripes ? e->nStripes : 1;
      int launchNs = e->nStripesForCount[maxAllowedCount];
      if (variant >= 0) {
        const auto &v = e->stripeVariant[variant];
        INFO(NCCL_INIT,
             "MSCCL v3 launch: rank=%d entry=%d msgBytes=%zu maxAllowedCount=%u nStripes addalgo=%d launch=%d "
             "used=%d flags=%s(used %s) alignedDeps=%d/%d demoted=%d sameTbReorderPairs=%d flagCycle step=%d stripe=%d%s",
             comm->rank, (int)(e - s->sizes), msgBytes, maxAllowedCount, addalgoNs, launchNs, nStripes,
             mscclStripeFlagsName(e->stripeFlags), useStripeFlags ? "stripe" : "step", v.depsAligned, v.depsTotal,
             v.demoted, v.intraHazards, v.cycleStep, v.cycle,
             v.skipped ? " (analysis skipped)" : "");
      } else {
        INFO(NCCL_INIT,
             "MSCCL v3 launch: rank=%d entry=%d msgBytes=%zu maxAllowedCount=%u nStripes addalgo=%d launch=%d "
             "used=%d flags=%s alignedDeps=0 (%s)",
             comm->rank, (int)(e - s->sizes), msgBytes, maxAllowedCount, addalgoNs, launchNs, nStripes,
             mscclStripeFlagsName(e->stripeFlags),
             nStripes <= 1 ? "nStripes=1" : (e->stripeFlags ? "no analysis for this nStripes" : "step flags"));
      }
    }
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
  work.nStripes         = (uint16_t)nStripes;
  work.hasReduce        = 0;
  /* With one stripe, stripe flags and step flags are the same order (subIter = iter, every stripe is the
   * last), so take the kernel's step-flag path: the stripe path's extra per-dependency work cost 0.6-2% on
   * nStripes=1 LL cells (fe685aea LL 1 MiB 1.020x runtime_v2b on a86998d, job 60949733). */
  work.stripeFlags      = useStripeFlags ? 1 : 0;
  memset(work._pad, 0, sizeof(work._pad));

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
             int nS = nStripes;   /* this launch's value (R4a): kernel and proxy must agree */
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
       (void*)commAndChans, (void*)launchAlgo,
       work.workIndex, work.sizePerMscclChunk);

  mscclLaunchKernel(kfn, e->nBlocks, NCCL_MAX_NTHREADS, smem, stream,
                    commAndChans, launchAlgo, work);
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    WARN("MSCCL kernel launch failed: %s", cudaGetErrorString(launchErr));
    return ncclUnhandledCudaError;
  }
  INFO(NCCL_ALL, "MSCCL kernel launched OK");
  return ncclSuccess;
}

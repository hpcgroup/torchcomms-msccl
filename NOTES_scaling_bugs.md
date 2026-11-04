# ncclx+msccl scaling bugs (found 2026-05-11)

Two scaling issues at `nRanks > 64`, both in the custom ncclx fork (not in the LD_PRELOAD shim). Symptom observed: 64 GPU sweep completes cleanly; 128 GPU sweep crashes with ExitCode 139 (SIGSEGV) inside `ncclCommInitRankConfig` (nccl-tests `common.cu:210`), CUDA error string suppressed because `NCCL_DEBUG=WARN`. Then `malloc_consolidate(): invalid chunk size` from heap-corruption in cleanup.

---

## Issue 1 — `ncclTransportMscclConnect` does O(nRanks × MSCCL_MAX_CHANNELS × MSCCL_NUM_CONNS_PER_PEER) all-to-all P2P connect at init time

**File:** `comms/ncclx/stable/src/transport/generic.cc:103-159`

```c
NCCL_PARAM(MscclMaxChannels,      "MSCCL_MAX_CHANNELS",      16);
NCCL_PARAM(MscclNumConnsPerPeer,  "MSCCL_NUM_CONNS_PER_PEER", 2);

ncclResult_t ncclTransportMscclConnect(struct ncclComm* comm) {
  ...
  int maxCh = std::max(comm->nChannels, (int)ncclParamMscclMaxChannels());
  if (maxCh > MAXCHANNELS) maxCh = MAXCHANNELS;
  int numConns = (int)ncclParamMscclNumConnsPerPeer();
  ...
  for (int c = comm->nChannels; c < maxCh; c++) {
    if (comm->channels[c].id == -1) {
      NCCLCHECKGOTO(initChannel(comm, c), ret, fail);
    }
  }
  ...
  for (int ci = 0; ci < numConns; ci++) {
    for (int peer = 0; peer < comm->nRanks; peer++) {
      if (peer == comm->rank) continue;
      for (int c = 0; c < maxCh; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &peer, 1, &peer, ci), ret, fail);
      }
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, graph, ci), ret, fail);
  }
}
```

Called from `comms/ncclx/stable/src/init.cc:1400` and `:1418` — both inside `ncclCommInitRankConfig`. Per-rank connection count = `(nRanks - 1) × maxCh × numConns`.

| nRanks | conns/rank w/ defaults (maxCh=16, numConns=2) |
|-------:|----:|
| 64     | 2016 |
| 128    | **4064** |
| 256    | 8160 |

**Why it bites at 128 and not 64:** each connection allocates send/recv FIFO buffers (host-side pinned mem because `NCCL_NET="AWS Libfabric"` on Perlmutter) plus IB QPs on the CXI provider. At ~4k conns/rank × 32 nodes the per-NIC QP / pinned-memory budget tips over. CUDA error is masked at `NCCL_DEBUG=WARN`.

**Why the work is wasted:** the shipped deepseekflash algos use only **1/2/4** channels (`best_program_128gpus_tiny/medium/large_*.xml` headers: `nchannels="1"`, `"2"`, `"4"`). The 16-channel default sets up 4-16× more transport than any algo will use.

**Fix in this repo (two layers):**

1. **Proper fix — sparse connect (in `ncclTransportMscclConnect`).** libnccl now dlopens the .so's listed in `MSCCL_ALGO_PATH` at MsccConnect time, reads each `mscclAlgoDescriptor->algos[myRank].mscclTBs`, and builds per-rank `(channel, peer, dir)` used-set bitmaps. The connect loop then only calls `ncclTransportP2pConnect` for tuples actually referenced by some TB, with asymmetric `nrecv`/`nsend` if needed. Falls back to all-to-all if `MSCCL_ALGO_PATH` is unset or no descriptor matches `comm->nRanks`. Symmetry of the per-rank used-set across ranks (rank A's recv[c,B] ↔ rank B's send[c,A]) is required and guaranteed by the IR codegen.

2. **Fallback layer — adaptive `NCCL_MSCCL_MAX_CHANNELS` in the preload** from the loaded algo descriptors' `nChannels`, before the real `ncclCommInitRankConfig` runs. Still does all-to-all but only over the channels the algo declares. Useful when libnccl's sparse path is disabled or when running against an older libnccl. See `ncclmsccl_preload.cpp` after the 2026-05-11 patch.

---

## Issue 2 — Stack-array overflow in `ncclMscclEngineAddAlgo` at `nRanks > MSCCL_ENGINE_MAX_TBS (=64)`

**File:** `comms/ncclx/stable/src/msccl_engine_host.cc:228-310`

```c
int nextRecvCi[MSCCL_ENGINE_MAX_TBS][MSCCL_ENGINE_MAX_TBS] = {{0}};  // [64][64]
int nextSendCi[MSCCL_ENGINE_MAX_TBS][MSCCL_ENGINE_MAX_TBS] = {{0}};
/* nextXxxCi[channelId][peer] = next free connIndex for that bucket.
 * channelId is bounded by maxChanId <= MSCCL_ENGINE_MAX_TBS in practice;
 * peer < comm->nRanks <= MSCCL_ENGINE_MAX_TBS for currently-shipped algos. */
if (myAlgo->nBlocks > MSCCL_ENGINE_MAX_TBS) { ... return ncclInvalidArgument; }
...
if (ch >= MSCCL_ENGINE_MAX_TBS || src->recvPeer >= comm->nRanks
    || src->sendPeer >= comm->nRanks) { ... return ncclInvalidArgument; }
...
int *cnt = &nextRecvCi[ch][src->recvPeer];   // <-- recvPeer can be 0..nRanks-1
...
int *cnt = &nextSendCi[ch][src->sendPeer];
```

The guard checks `recvPeer < comm->nRanks`, but the second dimension of `nextRecvCi`/`nextSendCi` is bounded by `MSCCL_ENGINE_MAX_TBS = 64` (`include/msccl_engine.h:30`). At 128 ranks `recvPeer` can be 64..127 — every write past index 63 corrupts adjacent stack frames.

This fires inside `ncclMscclEngineAddAlgo`, which the preload (`ncclmsccl_preload.cpp:130-153`) calls at the first `ncclAllReduce` per comm. Even after Issue 1 is fixed, the first allreduce at 128 ranks will hit this.

**Fix:** size the second dimension by `comm->nRanks` (heap-allocated, since `nRanks` is dynamic). See `msccl_engine_host.cc` after the 2026-05-11 patch.

---

## Useful invariants worth knowing

- `mscclEngineAlgo.nChannels` is per-rank but identical across ranks (set from XML root attribute `nchannels` during codegen — `ir_codegen_interp.py:386,450`).
- `mscclAlgoDescriptor` exported by each algo .so exposes `algos[nranks]`. The preload already dlopens and reads `nranks`, `min_bytes`, `max_bytes` from it — extending to read `algos[0].nChannels` requires no ABI changes.
- `NCCL_PARAM(...)` reads its env var on first use only (caches the value); the preload must `setenv()` **before** the real `ncclCommInitRankConfig` runs.
- The preload's existing `ncclCommInitRankConfig` interceptor (`ncclmsccl_preload.cpp:168-176`) is the natural place to do that — it's already pass-through and runs before any libnccl init work.

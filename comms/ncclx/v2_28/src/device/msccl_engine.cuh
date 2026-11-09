/*
 * MSCCL interpreter engine — device-side kernel implementation.
 *
 * Ported from the RCCL/MSCCL runtime (msccl_kernel_impl.h) to the ncclx-stable
 * (NCCL 2.28) Primitives API. Key differences from the original:
 *
 *   · Primitives<> template has 6 args (not 8): no Direct flag for MSCCL,
 *     no RCCL_METADATA_MSCCL tag. We use Direct=0.
 *   · ncclDevComm → ncclKernelCommAndChannels. The engine kernel manually
 *     loads ncclShmem.comm and ncclShmem.channel from the passed pointer.
 *   · setDataPtrs() takes 6 args (not 2). We always pass ipcReg=0 so the
 *     extra args are not dereferenced.
 *   · barrier() → __syncthreads() (CUDA; no HIP asm).
 *   · localCopy uses a manual per-thread loop (prims.localCopy() removed in
 *     newer NCCL).
 *   · MSCCL_MAX_NUM_STEPS 64 → MSCCL_ENGINE_MAX_STEPS 512 (see msccl_engine.h).
 *
 * Shared memory layout per block:
 *   · Static:  __shared__ mscclEngineThreadBlock engineTB  (~10 KB)
 *              (declared inside the kernel via the macro below)
 *   · Dynamic: ncclShmemData ncclShmem (extern __shared__)
 *              + per-warp LL128 scratch (ncclShmemPerWarp, same base address)
 *   The two regions do not overlap: static precedes dynamic in the SM's
 *   shared memory space.
 */

#ifndef MSCCL_ENGINE_CUH_
#define MSCCL_ENGINE_CUH_

#include "device.h"       /* ncclShmemData, ncclShmem, NCCL_MAX_NTHREADS, … */
#include "primitives.h"   /* Primitives<>, FanAsymmetric<>, ProtoSimple/LL/LL128 */
#include "collectives.h"  /* reduce_kernel helpers (BytePack, applyReduce, …) */
#include "include/msccl_engine.h"

/* MSCCL_ENGINE_MAX_ITER (flag sub-iteration budget) is in include/msccl_engine.h. */

/*
 * Flag = lexicographic (workIndex, subIter, step) packed into a uint64.
 * A TB is "ahead" of another iff its flag is numerically larger.
 *
 * work.stripeFlags == 0 (NCCL_MSCCL_STRIPE_FLAGS=step, runtime_v2b): subIter = iter. When nStripes > 1
 * the flag is published only after the LAST stripe of a step and waited on only at the FIRST stripe.
 * A consumer that needs the producer's step at stripe 0 then waits for all of the producer's stripes,
 * which deadlocks a send -> (dep) -> reduce -> send-back chain (R4, docs/runtime_issues).
 *
 * work.stripeFlags == 1 (NCCL_MSCCL_STRIPE_FLAGS=stripe, runtime_v3): subIter = iter * nStripes + stripe
 * (stripe is more significant than step, which matches the order a TB executes in). Every stripe of a
 * dep-source step publishes. A dependency marked MSCCL_ENGINE_DEP_ALIGNED in dependentBid is waited on
 * at every stripe s for the producer's (iter * nStripes + s, step); the host marks a dependency only
 * when it has proven that stripe s of the consumer needs only work the producer has finished by then.
 * An unmarked dependency is waited on at stripe 0 for the producer's last stripe, as before.
 */
#define MSCCL_ENGINE_FLAG(wi, iter, step) \
  ((uint64_t)(wi) * (MSCCL_ENGINE_MAX_ITER * (uint64_t)MSCCL_ENGINE_MAX_STEPS) \
   + (uint64_t)(iter) * MSCCL_ENGINE_MAX_STEPS \
   + (uint64_t)(step))

#define MSCCL_ENGINE_WORKINDEX_FROM_FLAG(f) \
  ((f) / ((uint64_t)MSCCL_ENGINE_MAX_ITER * MSCCL_ENGINE_MAX_STEPS))

/* PTX volatile load/store for cross-block flags — bypasses L1 cache.
 * Mirrors the upstream MSCCL load()/store() helpers. */
__device__ __forceinline__ static uint64_t
mscclEngineFlagLoad(volatile struct mscclEngineFlag *flags, int bid) {
  uint64_t v;
  asm volatile("ld.volatile.global.b64 %0,[%1];"
               : "=l"(v) : "l"((uint64_t*)&flags[bid].flag));
  return v;
}

__device__ __forceinline__ static void
mscclEngineFlagStore(volatile struct mscclEngineFlag *flags, int bid, uint64_t val) {
  asm volatile("st.volatile.global.b64 [%0],%1;"
               :: "l"((uint64_t*)&flags[bid].flag), "l"(val));
}

template<typename Prim, typename Proto>
__device__ __forceinline__ static void
mscclEngineSendWithBarrier(Prim& prims, intptr_t srcOff, int nelem, int nthreads) {
  prims.send(srcOff, nelem);
  if constexpr (Proto::Id == NCCL_PROTO_LL) {
    barrier_sync(15, nthreads);
  }
}

/* (removed — replaced by msccl_tbCopy for all shmem loads) */

/* Copy `size` uint32_t words from src to dst using all `nthreads` threads.
 * Uses 128-bit (uint4) vectorised stores where possible (4× shmem bandwidth),
 * falling back to uint32_t for any trailing words. Both pointers must be
 * 16-byte aligned for the vectorised path — all our callers satisfy this. */
__device__ __forceinline__ static void
msccl_tbCopy(uint32_t *__restrict__ dst, const uint32_t *__restrict__ src,
             uint64_t size, int tid, int nthreads) {
  const int vec = 4; /* uint4 = 4 × uint32_t */
  const int nvec = (int)(size / vec);
  /* Vectorised 128-bit path */
  for (int i = tid; i < nvec; i += nthreads)
    ((uint4 *)dst)[i] = ((const uint4 *)src)[i];
  /* Scalar tail (at most 3 words) */
  for (int i = nvec * vec + tid; i < (int)size; i += nthreads)
    dst[i] = src[i];
}

/* ── REDUCE helper (same logic as original mscclReduce) ─────────────────── */
#define MSCCL_REDUCE_LOOP(n, BytePerPack)                                       \
  for (int r = 0; r < (n); r++) {                                               \
    srcOff = srcBase + (ssize_t)sharedTB.reductionSrcOffsets[t->reductionPointer+r] \
             * sizePerChunk;                                                     \
    reduce_in = ld_volatile_global<BytePerPack>((uintptr_t)(srcPtr + srcOff)); \
    acc = applyReduce(redFn, reduce_in, acc);                                   \
  }

template<typename T, typename RedOp, int BytePerPack>
__device__ __forceinline__ static void
msccl_reduce(int c, int numRed, int curr,
             ssize_t sizePerChunk, RedOp redFn,
             const struct mscclEngineTransmission *t,
             ssize_t gridOff, ssize_t &srcOff, ssize_t dstOff,
             const T *srcPtr, T *dstPtr,
             const struct mscclEngineThreadBlock &sharedTB) {
  const int elPerPack = BytePerPack / (int)sizeof(T);
  T *dstIdx = dstPtr + dstOff + curr * elPerPack;
  BytePack<BytePerPack> reduce_in;
  BytePack<BytePerPack> acc = ld_volatile_global<BytePerPack>((uintptr_t)dstIdx);
  ssize_t srcBase = gridOff + (ssize_t)c * sizePerChunk + curr * elPerPack;
  switch (numRed) {
    case 7:  { MSCCL_REDUCE_LOOP(7, BytePerPack); break; }
    default: { MSCCL_REDUCE_LOOP(numRed, BytePerPack); break; }
  }
  st_global<BytePerPack>((uintptr_t)dstIdx, acc);
}

/* ── Main interpreter template ───────────────────────────────────────────── */
/*
 * mscclRunEngine<T, RedOp, Proto>
 *
 * commAndChans  : ncclKernelCommAndChannels* for this rank (device pointer)
 * algo          : mscclEngineAlgo* for this rank (device pointer)
 * work          : mscclEngineWork — all per-launch arguments inline
 *
 * All opcodes (including compound rcs/rrs/rrc/rrcs) are supported.
 */
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void
mscclRunEngine(struct ncclKernelCommAndChannels *commAndChans,
               const struct mscclEngineAlgo     *algo,
               struct mscclEngineWork            work) {
  /* ── shmem layout ──────────────────────────────────────────────────────
   * Static shared: the TB descriptor for this block (loaded at startup).
   * Dynamic shared (ncclShmem): populated below; used by Primitives<>.   */
  __shared__ struct mscclEngineThreadBlock sharedTB;

  const int tid      = threadIdx.x;
  const int bid      = blockIdx.x;
  const int nthreads = NCCL_MAX_NTHREADS;

  /* Step 1: load this TB's descriptor into static shmem. */
  msccl_tbCopy(
    (uint32_t *)&sharedTB,
    (const uint32_t *)(algo->mscclTBs + bid),
    sizeof(struct mscclEngineThreadBlock) / sizeof(uint32_t),
    tid, nthreads);
  __syncthreads(); /* sharedTB.channelId now visible to all threads */

  /* Step 2: populate ncclShmem so that Primitives<> can set up connections.
   * We use all nthreads (not just one warp) to copy comm and channel since
   * ncclDevChannel can be 400+ bytes in modern NCCL (NVLS fields etc.).   */
  {
    int channelId = sharedTB.channelId;

    /* Copy ncclKernelComm into ncclShmem.comm */
    msccl_tbCopy(
      (uint32_t *)&ncclShmem.comm,
      (const uint32_t *)&commAndChans->comm,
      sizeof(struct ncclKernelComm) / sizeof(uint32_t),
      tid, nthreads);

    /* Copy the relevant ncclDevChannel into ncclShmem.channel */
    msccl_tbCopy(
      (uint32_t *)&ncclShmem.channel,
      (const uint32_t *)&commAndChans->channels[channelId],
      sizeof(struct ncclDevChannel) / sizeof(uint32_t),
      tid, nthreads);

    if (tid == 0) {
      ncclShmem.channelId = channelId;
      ncclShmem.aborted   = 0;
    }
  }
  __syncthreads(); /* ncclShmem now populated */

  /* ── user buffers and per-block topology ─────────────────────────────── */
  const T *__restrict__ thisInput   = (const T *)work.sendBuff;
  T       *__restrict__ thisOutput  = (T       *)work.recvBuff;
  T       *__restrict__ thisScratch = (T       *)work.scratchBuffer;

  int recvPeer = sharedTB.recvPeer;
  int sendPeer = sharedTB.sendPeer;

  /* ── Sanity-check peers before Primitives tries to dereference them ──── */
  if (tid == 0) {
    if (ncclShmem.channel.peers == nullptr) {
      printf("MSCCL KERNEL ERROR rank=%d bid=%d: ncclShmem.channel.peers is NULL\n",
             ncclShmem.comm.rank, bid);
      ncclShmem.aborted = 1;
    } else {
      if (recvPeer >= 0 && ncclShmem.channel.peers[recvPeer] == nullptr) {
        printf("MSCCL KERNEL ERROR rank=%d bid=%d ch=%d recvPeer=%d: peers[recvPeer] is NULL\n",
               ncclShmem.comm.rank, bid, sharedTB.channelId, recvPeer);
        ncclShmem.aborted = 1;
      }
      if (sendPeer >= 0 && ncclShmem.channel.peers[sendPeer] == nullptr) {
        printf("MSCCL KERNEL ERROR rank=%d bid=%d ch=%d sendPeer=%d: peers[sendPeer] is NULL\n",
               ncclShmem.comm.rank, bid, sharedTB.channelId, sendPeer);
        ncclShmem.aborted = 1;
      }
    }
  }
  __syncthreads();
  if (ncclShmem.aborted) return;

  /* ── chunk / slice sizing (identical to original MSCCL interpreter) ─── */
  const ssize_t chunkSize =
    (ssize_t)(Proto::calcBytePerStep() / sizeof(T))
    * (Proto::Id == NCCL_PROTO_SIMPLE ? MSCCL_ENGINE_CHUNKSTEPS : 1);

  ssize_t minChunkSize = 0;
  if (Proto::Id == NCCL_PROTO_LL)
    minChunkSize = nthreads * (Proto::calcBytePerGrain() / (ssize_t)sizeof(T));
  if (Proto::Id == NCCL_PROTO_LL128)
    minChunkSize = nthreads * (Proto::calcBytePerGrain() / (ssize_t)sizeof(T)) / 2;

  /* ── Primitives instance (FanAsymmetric<1,1>, Direct=0, no P2p) ──────── */
  /* connIndex{Recv,Send} address peers[peer]->{recv,send}[connIndex]; multiple
   * TBs sharing the same (channel, peer, dir) tuple must use distinct slots
   * (host-side ncclMscclEngineAddAlgo computes them). When all four (algos
   * with no sharing, e.g. tiny) values are zero this collapses to the default
   * single-FIFO-per-pair path. */
  RedOp redFn(work.redOpArg);
  Primitives<T, RedOp, FanAsymmetric<1,1>, 0, Proto, 0>
    prims(tid, nthreads, &recvPeer, &sendPeer,
          thisInput, thisOutput, work.redOpArg,
          /*group=*/0,
          sharedTB.connIndexRecv, sharedTB.connIndexSend);

  /* ── synchronisation flags ────────────────────────────────────────────── */
  const int64_t workIndex = (int64_t)work.workIndex;
  volatile struct mscclEngineFlag *mscclFlags = work.syncFlags;

  const ssize_t sizePerChunk  = (ssize_t)work.sizePerMscclChunk;
  const uint32_t maxCount     = work.maxAllowedCount;
  /* nStripes >=1: per outer iter we walk the IR `nStripes` times and on the
   * k-th pass each step processes only count-chunks [k*count/nStripes,
   * (k+1)*count/nStripes). Caps per-stripe-per-conn slot pressure to
   * ceil(count/nStripes)*chunkSteps <= NCCL_STEPS, breaking the symmetric
   * send-first deadlock. nStripes=1 is the fast path (host-side trigger
   * computes pre-recv send pressure per TB and only sets >1 when needed). */
  const int nStripes = max(1, (int)work.nStripes);
  const bool stripeFlags = (work.stripeFlags != 0);

  /* ── outer loop: iterate over the full message in chunkSize pieces ─────── */
  for (ssize_t gridOff = 0, iter = 0;
       gridOff < sizePerChunk;
       gridOff += chunkSize, iter++) {

    ssize_t realChunkSize;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      realChunkSize = min(chunkSize, sizePerChunk - gridOff);
      realChunkSize = roundUp(realChunkSize,
                              (ssize_t)(nthreads * sizeof(uint64_t) / sizeof(T)));
      // Keep slices within one buffer slot; recv proxy posts irecv of size
      // stepSize*sliceSteps (== chunkSize / SlicePerChunk * SlicePerChunk),
      // and roundUp can overshoot chunkSize when granularity doesn't divide it.
      realChunkSize = min(realChunkSize, chunkSize);
    } else {
      realChunkSize = min(chunkSize,
                          divUp(sizePerChunk - gridOff, minChunkSize) * minChunkSize);
    }
    realChunkSize = (ssize_t)(int)realChunkSize;
    const int nelem = (int)min(realChunkSize, sizePerChunk - gridOff);

    /* ── stripe loop: each stripe replays the IR walk on its own slice of
     * every step's count. The intra-stripe interleave of send-then-recv-
     * then-send-... is what breaks the symmetric send-first cycle. Cross-block
     * dep waits and flag publishes: see MSCCL_ENGINE_FLAG (step or stripe
     * flags, work.stripeFlags). */
    const int lastStripe = nStripes - 1;
    for (int stripe = 0; stripe < nStripes; stripe++) {
      const bool isFirstStripe = (stripe == 0);
      const bool isLastStripe  = (stripe == lastStripe);
      /* Flag sub-iteration of this stripe and of this iter's last stripe (see MSCCL_ENGINE_FLAG). */
      const int64_t subIter     = stripeFlags ? (int64_t)iter * nStripes + stripe     : (int64_t)iter;
      const int64_t lastSubIter = stripeFlags ? (int64_t)iter * nStripes + lastStripe : (int64_t)iter;

      /* ── inner loop: execute each IR step. The count-split loop stays
       * inside the step, restricted to this stripe's contiguous slice. */
      int step = 0;
      for (int i = 0; i < (int)sharedTB.nSteps; i++) {
        const struct mscclEngineTransmission *t = &sharedTB.transmissions[i];

        /* cross-block dependency wait. Step flags: only on the first stripe, for the producer's
         * full step (published at its last stripe). Stripe flags: an aligned dependency waits at
         * every stripe for the producer's same stripe; any other waits at stripe 0 for the
         * producer's last stripe. */
        int16_t numDeps = t->numDependencies;
        if (numDeps > 0) {
          int16_t depPtr = t->dependencePointer;
          bool anyAligned = false;   /* uniform across the block (shared memory) */
          if (stripeFlags) {
            for (int d = 0; d < numDeps; d++)
              anyAligned |= (sharedTB.dependentBid[depPtr + d] & MSCCL_ENGINE_DEP_ALIGNED) != 0;
          }
          if (tid < numDeps) {
            int8_t  depRaw  = sharedTB.dependentBid[depPtr + tid];
            int8_t  depBid  = depRaw & MSCCL_ENGINE_DEP_BID_MASK;
            int16_t depStep = sharedTB.dependentStep[depPtr + tid];
            const bool aligned = stripeFlags && (depRaw & MSCCL_ENGINE_DEP_ALIGNED);
            if (aligned || isFirstStripe) {
              uint64_t goal = MSCCL_ENGINE_FLAG(workIndex, aligned ? subIter : lastSubIter, depStep);
              while (true) {
                uint64_t cur = mscclEngineFlagLoad(mscclFlags, depBid);
                if (cur >= goal &&
                    MSCCL_ENGINE_WORKINDEX_FROM_FLAG(cur) == (uint64_t)workIndex)
                  break;
              }
            }
          }
          /* step counter advance is uniform across stripes (the IR's step
           * encoding is fixed by the producer-side flag layout). */
          step += numDeps - 1;
          if (isFirstStripe || anyAligned) __syncthreads();
        }

        /* buffer pointer selection */
        const T *srcPtr =
          (t->srcBuffer == MSCCL_ENGINE_INPUT_BUF)   ? thisInput  :
          (t->srcBuffer == MSCCL_ENGINE_OUTPUT_BUF)   ? thisOutput : thisScratch;
        T *dstPtr =
          (t->dstBuffer == MSCCL_ENGINE_INPUT_BUF)   ? const_cast<T*>(thisInput)  :
          (t->dstBuffer == MSCCL_ENGINE_OUTPUT_BUF)   ? thisOutput : thisScratch;

        /* This stripe's slice of count: contiguous range
         * [stripeStart, stripeEnd). Even split with remainder absorbed by
         * later stripes via floor of (stripe+1)*count. */
        const int count       = (int)t->count;
        const int stripeStart = (count * stripe) / nStripes;
        const int stripeEnd   = (count * (stripe + 1)) / nStripes;
        const int stripeSpan  = stripeEnd - stripeStart;

        if (stripeSpan > 0) {
          /* update Primitives' data pointers for this step */
          if constexpr (Proto::Id == NCCL_PROTO_SIMPLE) {
            prims.setDataPtrs(srcPtr, dstPtr, work.redOpArg, nullptr, 0, sendPeer);
          } else {
            prims.setDataPtrs(srcPtr, dstPtr);
          }

          for (int c = stripeStart; c < stripeEnd; c += (int)maxCount) {
            ssize_t srcOff = gridOff + (ssize_t)(t->srcOffset + c) * sizePerChunk;
            ssize_t dstOff = gridOff + (ssize_t)(t->dstOffset + c) * sizePerChunk;
            const int thisCount = min((int)maxCount, stripeEnd - c);
            const int thisNelem = nelem * thisCount;

            switch (t->type) {
              case MSCCL_ENGINE_SEND:
                mscclEngineSendWithBarrier<decltype(prims), Proto>(
                  prims, srcOff, thisNelem, nthreads);
                break;

              case MSCCL_ENGINE_RECV:
                prims.recv(dstOff, thisNelem);
                break;

              case MSCCL_ENGINE_REDUCE: {
                /* manual per-element reduce (no network) */
                int numRed = (int)t->numReductions;
                ssize_t dstOff2 = gridOff + (ssize_t)(t->dstOffset + c) * sizePerChunk;
                int curr = tid;
                const int elPerPack16 = 16 / (int)sizeof(T);
                while (curr < thisNelem / elPerPack16) {
                  msccl_reduce<T, RedOp, 16>(c, numRed, curr, sizePerChunk, redFn,
                                             t, gridOff, srcOff, dstOff2,
                                             srcPtr, dstPtr, sharedTB);
                  curr += nthreads;
                }
                curr = tid + (thisNelem / elPerPack16) * elPerPack16;
                if (curr < thisNelem)
                  msccl_reduce<T, RedOp, (int)sizeof(T)>(
                    c, numRed, curr, sizePerChunk, redFn,
                    t, gridOff, srcOff, dstOff2, srcPtr, dstPtr, sharedTB);
                __syncthreads();
                break;
              }

              case MSCCL_ENGINE_LOCAL_COPY:
                /* intra-GPU copy — no network primitives needed */
                for (int j = tid; j < thisNelem; j += nthreads)
                  dstPtr[dstOff + j] = srcPtr[srcOff + j];
                __syncthreads();
                break;

              case MSCCL_ENGINE_RECV_COPY_SEND:
                prims.recvCopySend(dstOff, thisNelem);
                break;
              case MSCCL_ENGINE_RECV_REDUCE_SEND:
                prims.recvReduceSend(srcOff, thisNelem);
                break;
              case MSCCL_ENGINE_RECV_REDUCE_COPY:
                prims.recvReduceCopy(srcOff, dstOff, thisNelem);
                break;
              case MSCCL_ENGINE_RECV_REDUCE_COPY_SEND:
                prims.recvReduceCopySend(srcOff, dstOff, thisNelem);
                break;
              case MSCCL_ENGINE_NOP:
                break;
              default:
                return; /* unknown opcode — bail */
            } /* switch type */
          } /* for c */
        } /* if stripeSpan > 0 */

        /* Step counter advance for REDUCE's reduction-fan: applied uniformly
         * across stripes (must match producer-side encoding). The original
         * `if (c == 0) step += numRed - 1` fired once per IR step regardless
         * of count, so we replicate that — once per (stripe, step). */
        if (t->type == MSCCL_ENGINE_REDUCE) {
          step += (int)t->numReductions - 1;
        }

        /* Raise cross-block flag for dep-source steps. Step flags: only on the LAST stripe, after
         * the full count for this step has been processed. Stripe flags: on every stripe, with that
         * stripe's sub-iteration; a consumer that needs the full step waits for lastSubIter. */
        if ((stripeFlags || isLastStripe) && t->hasDependence && tid == nthreads - 1) {
          __threadfence();
          mscclEngineFlagStore(mscclFlags, bid,
                               MSCCL_ENGINE_FLAG(workIndex, subIter, step));
        }
        step++;
      } /* for i (steps) */
    } /* for stripe */
  } /* for gridOff (chunks) */
}

/* ── Kernel entry-point macro ─────────────────────────────────────────────
 *
 * Generates one __global__ kernel per (RedOp, T, Proto) triple.
 * The kernel packs all per-launch parameters into mscclEngineWork to keep
 * the argument list short.
 */
using MscclProtoSimple =
  ProtoSimple<MSCCL_ENGINE_CHUNKSTEPS/MSCCL_ENGINE_SLICESTEPS,
              MSCCL_ENGINE_SLICESTEPS>;

#define MSCCL_ENGINE_KERNEL_PROTO(devredop, type, protoName, protoType)         \
__global__ void mscclEngineKernel_##devredop##_##type##_##protoName(            \
    struct ncclKernelCommAndChannels *commAndChans,                             \
    const struct mscclEngineAlgo     *algo,                                    \
    struct mscclEngineWork            work)                                    \
{                                                                               \
  mscclRunEngine<type, Func##devredop<type>, protoType>                         \
    (commAndChans, algo, work);                                                 \
}

#define MSCCL_ENGINE_KERNEL(devredop, type)                                    \
  MSCCL_ENGINE_KERNEL_PROTO(devredop, type, Simple, MscclProtoSimple)          \
  MSCCL_ENGINE_KERNEL_PROTO(devredop, type, LL, ProtoLL)                       \
  MSCCL_ENGINE_KERNEL_PROTO(devredop, type, LL128, ProtoLL128)

#define MSCCL_ENGINE_KERNELS_FOR_DEVREDOP(devredop)   \
  MSCCL_ENGINE_KERNEL(devredop, int8_t)               \
  MSCCL_ENGINE_KERNEL(devredop, uint8_t)              \
  MSCCL_ENGINE_KERNEL(devredop, int32_t)              \
  MSCCL_ENGINE_KERNEL(devredop, uint32_t)             \
  MSCCL_ENGINE_KERNEL(devredop, int64_t)              \
  MSCCL_ENGINE_KERNEL(devredop, uint64_t)             \
  MSCCL_ENGINE_KERNEL(devredop, half)                 \
  MSCCL_ENGINE_KERNEL(devredop, float)                \
  MSCCL_ENGINE_KERNEL(devredop, double)               \
  MSCCL_ENGINE_KERNEL(devredop, __nv_bfloat16)

#define MSCCL_ENGINE_ALL_KERNELS()                    \
  MSCCL_ENGINE_KERNELS_FOR_DEVREDOP(Sum)              \
  MSCCL_ENGINE_KERNELS_FOR_DEVREDOP(Prod)             \
  MSCCL_ENGINE_KERNELS_FOR_DEVREDOP(MinMax)

/* Declarations (usable from host-side launch code): */
#define MSCCL_ENGINE_DECL_KERNEL_PROTO(devredop, type, protoName)              \
__global__ void mscclEngineKernel_##devredop##_##type##_##protoName(           \
    struct ncclKernelCommAndChannels *commAndChans,                             \
    const struct mscclEngineAlgo     *algo,                                    \
    struct mscclEngineWork            work);

#define MSCCL_ENGINE_DECL_KERNEL(devredop, type)                               \
  MSCCL_ENGINE_DECL_KERNEL_PROTO(devredop, type, Simple)                       \
  MSCCL_ENGINE_DECL_KERNEL_PROTO(devredop, type, LL)                           \
  MSCCL_ENGINE_DECL_KERNEL_PROTO(devredop, type, LL128)

#define MSCCL_ENGINE_DECL_KERNELS_FOR_DEVREDOP(devredop) \
  MSCCL_ENGINE_DECL_KERNEL(devredop, int8_t)             \
  MSCCL_ENGINE_DECL_KERNEL(devredop, uint8_t)            \
  MSCCL_ENGINE_DECL_KERNEL(devredop, int32_t)            \
  MSCCL_ENGINE_DECL_KERNEL(devredop, uint32_t)           \
  MSCCL_ENGINE_DECL_KERNEL(devredop, int64_t)            \
  MSCCL_ENGINE_DECL_KERNEL(devredop, uint64_t)           \
  MSCCL_ENGINE_DECL_KERNEL(devredop, half)               \
  MSCCL_ENGINE_DECL_KERNEL(devredop, float)              \
  MSCCL_ENGINE_DECL_KERNEL(devredop, double)             \
  MSCCL_ENGINE_DECL_KERNEL(devredop, __nv_bfloat16)

#define MSCCL_ENGINE_ALL_DECLS()                          \
  MSCCL_ENGINE_DECL_KERNELS_FOR_DEVREDOP(Sum)             \
  MSCCL_ENGINE_DECL_KERNELS_FOR_DEVREDOP(Prod)            \
  MSCCL_ENGINE_DECL_KERNELS_FOR_DEVREDOP(MinMax)

MSCCL_ENGINE_ALL_DECLS()

#endif /* MSCCL_ENGINE_CUH_ */

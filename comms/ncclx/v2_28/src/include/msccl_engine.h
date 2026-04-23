/*
 * MSCCL interpreter engine — struct definitions for the IR-as-data engine.
 *
 * This header is included by both host (setup) and device (kernel) code.
 * No C++ STL — only plain C structs and macros.
 *
 * The design mirrors the original MSCCL runtime structs (msccl_struct.h from
 * the RCCL/MSCCL tree) with two changes:
 *   1. MSCCL_MAX_NUM_STEPS bumped from 64 → 512 (covers 256-GPU ring algorithms
 *      that need up to 510 steps per thread block).
 *   2. Host-only STL fields (std::vector, std::map, …) removed — this header
 *      must compile cleanly under nvcc for device code.
 */

#ifndef MSCCL_ENGINE_H_
#define MSCCL_ENGINE_H_

#include <stdint.h>
#include <stddef.h>  /* size_t */

/* ── protocol constants (mirror nccl_tuner.h; guarded for dual inclusion) ── */
#ifndef NCCL_PROTO_LL
#define NCCL_PROTO_LL     0
#define NCCL_PROTO_LL128  1
#define NCCL_PROTO_SIMPLE 2
#endif

/* ── limits ──────────────────────────────────────────────────────────────── */
#define MSCCL_ENGINE_MAX_STEPS       512   /* must be >= max steps in any IR TB */
#define MSCCL_ENGINE_MAX_TBS         64    /* max thread blocks per rank */
#define MSCCL_ENGINE_MAX_COUNT       72    /* max chunks per transmission */
#define MSCCL_ENGINE_MAX_REDUCE_FUSION 16

#define MSCCL_ENGINE_SLICESTEPS      2     /* NCCL_STEPS/4 = 8/4 */
#define MSCCL_ENGINE_CHUNKSTEPS      4     /* NCCL_STEPS/2 = 8/2 */

/* buffer selector values */
#define MSCCL_ENGINE_INPUT_BUF       0
#define MSCCL_ENGINE_OUTPUT_BUF      1
#define MSCCL_ENGINE_SCRATCH_BUF     2

/* opcode values (match XML type codes) */
#define MSCCL_ENGINE_SEND            0
#define MSCCL_ENGINE_RECV            1
#define MSCCL_ENGINE_RECV_COPY_SEND  2
#define MSCCL_ENGINE_RECV_REDUCE_SEND 3
#define MSCCL_ENGINE_RECV_REDUCE_COPY 4
#define MSCCL_ENGINE_RECV_REDUCE_COPY_SEND 5
#define MSCCL_ENGINE_LOCAL_COPY      6
#define MSCCL_ENGINE_REDUCE          7
#define MSCCL_ENGINE_NOP             8

/* ── per-step transmission descriptor ───────────────────────────────────── */
struct mscclEngineTransmission {
  int16_t dependencePointer;  /* index into dependentBid/dependentStep arrays */
  int16_t numDependencies;    /* how many cross-block deps to wait on */
  int16_t reductionPointer;   /* index into reductionSrcOffsets */
  int16_t numReductions;      /* number of source offsets for reduce */
  int16_t srcOffset;          /* source chunk offset (× sizePerMscclChunk) */
  int16_t dstOffset;          /* dest chunk offset */
  uint8_t srcBuffer;          /* MSCCL_ENGINE_{INPUT,OUTPUT,SCRATCH}_BUF */
  uint8_t dstBuffer;
  int8_t  hasDependence;      /* this step raises a flag for others to wait on */
  uint8_t type;               /* MSCCL_ENGINE_* opcode */
  uint8_t count;              /* number of chunks in this transmission */
  uint8_t _pad;
}; /* 18 bytes */

static_assert(sizeof(struct mscclEngineTransmission) == 18,
              "mscclEngineTransmission must be 18 bytes");

/* ── per-thread-block algorithm state ───────────────────────────────────── */
struct alignas(16) mscclEngineThreadBlock {
  struct mscclEngineTransmission transmissions[MSCCL_ENGINE_MAX_STEPS];

  /* cross-block dependency info indexed by dependencePointer+i */
  int8_t  dependentBid[MSCCL_ENGINE_MAX_STEPS];
  int16_t dependentStep[MSCCL_ENGINE_MAX_STEPS];

  /* source offsets for multi-source reductions */
  int16_t reductionSrcOffsets[MSCCL_ENGINE_MAX_STEPS];

  int16_t sendPeer;   /* -1 = no send for this TB */
  int16_t recvPeer;   /* -1 = no recv for this TB */
  uint16_t nSteps;
  int16_t  channelId; /* NCCL channel this TB is bound to; -1 for local-copy TBs */
  uint8_t  _pad[8];   /* pad to 16-byte alignment for uint4 copies */
};

static_assert(sizeof(struct mscclEngineThreadBlock) % 16 == 0,
              "mscclEngineThreadBlock must be 16-byte aligned for uint4 copies");

/* ── cross-block synchronisation flag (one per TB) ───────────────────────── */
struct mscclEngineFlag {
  uint64_t flag;
  uint64_t _pad[3]; /* cache-line pad to avoid false sharing */
};

/* ── per-rank algorithm descriptor (lives in device global memory) ──────── */
struct mscclEngineAlgo {
  int32_t nChunksPerLoop;
  int32_t protocol;       /* NCCL_PROTO_SIMPLE / LL / LL128 */
  int32_t nChannels;
  int32_t nRanks;
  int32_t nBlocks;        /* thread blocks launched for this rank */
  int32_t nScratchChunks;
  int32_t sizeMultiplier; /* 1 for allreduce, nRanks for allgather/rs */
  int32_t chunkSteps;     /* = MSCCL_ENGINE_CHUNKSTEPS */
  int32_t sliceSteps;     /* = MSCCL_ENGINE_SLICESTEPS */
  int8_t  hasReduce;
  int8_t  _pad[7];

  /* indexed by blockIdx.x (bid) for this rank */
  alignas(16) struct mscclEngineThreadBlock mscclTBs[MSCCL_ENGINE_MAX_TBS];
};

/* ── per-launch work descriptor (passed as kernel argument) ──────────────── */
struct alignas(16) mscclEngineWork {
  volatile struct mscclEngineFlag *syncFlags; /* one per TB; NULL if no cross-block deps */
  void       *scratchBuffer;                  /* NULL if nScratchChunks == 0 */
  const void *sendBuff;
  void       *recvBuff;
  size_t      sizePerMscclChunk;
  uint64_t    redOpArg;
  uint32_t    workIndex;  /* used to construct unique flag values */
  uint32_t    maxAllowedCount;
  int         nChunksPerLoop;
  int8_t      hasReduce;
  int8_t      redOpArgIsPtr;
  uint8_t     _pad[6];
};
static_assert(sizeof(struct mscclEngineWork) % 16 == 0,
              "mscclEngineWork must be 16-byte aligned");

/* Descriptor exported by a standalone algo .so for use with ncclmsccl_preload.
 * Each algo .so exports the symbol "msccl_algo_desc" of this type. */
struct mscclAlgoDescriptor {
  const struct mscclEngineAlgo *algos; /* array[nranks] of per-rank algo data */
  int    nranks;
  size_t min_bytes;
  size_t max_bytes;
};

#endif /* MSCCL_ENGINE_H_ */

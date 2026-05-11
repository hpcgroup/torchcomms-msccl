/*
 * MSCCL engine kernel instantiations.
 *
 * This file is the ONLY .cu file needed for the engine itself.  It is small
 * (~150 lines of template instantiation boilerplate) and compiles quickly
 * regardless of algorithm count.  All algorithm-specific data lives in
 * separate generated files (msccl_algo_<name>.cu) that contain only
 * mscclEngineAlgo struct initialisers — no CUDA code.
 *
 * To add a new algorithm: generate its data file with ir_codegen_interp.py
 * and add it to the build.  No changes here.
 */

#include "device.h"
#include "msccl_engine.cuh"

typedef void (*mscclKernelFn)(struct ncclKernelCommAndChannels *,
                               const struct mscclEngineAlgo *,
                               struct mscclEngineWork);

/* Instantiate all kernels: Sum/Prod/MinMax × 10 types × 3 protocols */
MSCCL_ENGINE_ALL_KERNELS()

/* Thin CUDA launch wrapper — lets host-.cc code avoid <<<>>> syntax. */
extern "C"
void mscclLaunchKernel(mscclKernelFn kfn, int nBlocks, int nThreads, int smem,
                       cudaStream_t stream,
                       struct ncclKernelCommAndChannels *commAndChans,
                       const struct mscclEngineAlgo *algo,
                       struct mscclEngineWork work) {
  if (smem > 0) {
    cudaFuncSetAttribute((const void*)kfn,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  kfn<<<nBlocks, nThreads, smem, stream>>>(commAndChans, algo, work);
}

/* Kernel selector — returns function pointer for (devOp, datatype, protocol) triple.
 * Compiled by NVCC so __global__ symbols are referenceable. */
#define KFUNC(redop, type, protoName) ((mscclKernelFn)mscclEngineKernel_##redop##_##type##_##protoName)
#define CASE_PROTO(redop, type) \
  switch (proto) { \
    case NCCL_PROTO_SIMPLE: return KFUNC(redop, type, Simple); \
    case NCCL_PROTO_LL:     return KFUNC(redop, type, LL); \
    case NCCL_PROTO_LL128:  return KFUNC(redop, type, LL128); \
    default:                return nullptr; \
  }
extern "C"
mscclKernelFn mscclSelectKernel(int devOp, int dt, int proto) {
#define CASE_DT(redop) \
  switch (dt) { \
    case ncclInt8:    { CASE_PROTO(redop, int8_t); } \
    case ncclUint8:   { CASE_PROTO(redop, uint8_t); } \
    case ncclInt32:   { CASE_PROTO(redop, int32_t); } \
    case ncclUint32:  { CASE_PROTO(redop, uint32_t); } \
    case ncclInt64:   { CASE_PROTO(redop, int64_t); } \
    case ncclUint64:  { CASE_PROTO(redop, uint64_t); } \
    case ncclFloat16: { CASE_PROTO(redop, half); } \
    case ncclFloat32: { CASE_PROTO(redop, float); } \
    case ncclFloat64: { CASE_PROTO(redop, double); } \
    case ncclBfloat16:{ CASE_PROTO(redop, __nv_bfloat16); } \
    default:          return nullptr; \
  }
  switch (devOp) {
    case ncclDevSum:    { CASE_DT(Sum);    break; }
    case ncclDevProd:   { CASE_DT(Prod);   break; }
    case ncclDevMinMax: { CASE_DT(MinMax); break; }
    default: return nullptr;
  }
#undef CASE_DT
#undef CASE_PROTO
  return nullptr;
}

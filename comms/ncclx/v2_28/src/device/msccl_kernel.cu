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

/* Instantiate all kernels: Sum/Prod/MinMax × 10 types × Simple proto */
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

/* Kernel selector — returns function pointer for (devOp, datatype) pair.
 * Compiled by NVCC so __global__ symbols are referenceable. */
#define KFUNC(redop, type) ((mscclKernelFn)mscclEngineKernel_##redop##_##type)
extern "C"
mscclKernelFn mscclSelectKernel(int devOp, int dt) {
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

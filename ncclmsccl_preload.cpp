/*
 * LD_PRELOAD shim: intercepts ncclAllReduce -> ncclAllReduceMsccl, and loads
 * MSCCL algorithm data from standalone .so files lazily at first allreduce
 * time so that the comm is guaranteed to be fully initialized.
 *
 * Build:
 *   bash build_msccl_preload.sh
 *
 * Use:
 *   MSCCL_ALGO_PATH=/path/to/algo1.so:/path/to/algo2.so \
 *   LD_PRELOAD=$PWD/ncclmsccl_preload.so \
 *   LD_LIBRARY_PATH=build/ncclx/lib:$LD_LIBRARY_PATH \
 *   srun ... ./your_app
 *
 * Algo registration is deferred to the first ncclAllReduce call per comm,
 * because ncclCommInitRankConfig may return before the background init
 * thread has finished (comm->nRanks still 0). By the time the user calls
 * ncclAllReduce the comm is guaranteed ready.
 *
 * ncclMscclEngineAddAlgo must be exported from libnccl.so (add NCCL_API
 * wrapping in msccl_engine_host.cc and rebuild once).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nccl.h"
#include "include/msccl_engine.h"

typedef ncclResult_t (*ncclAllReduceFn_t)(
    const void*, void*, size_t, ncclDataType_t, ncclRedOp_t,
    ncclComm_t, cudaStream_t);

typedef ncclResult_t (*ncclCommInitRankFn_t)(
    ncclComm_t*, int, ncclUniqueId, int);

typedef ncclResult_t (*ncclCommInitRankConfigFn_t)(
    ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);

typedef ncclResult_t (*ncclCommInitAllFn_t)(
    ncclComm_t*, int, const int*);

typedef ncclResult_t (*ncclMscclEngineAddAlgoFn_t)(
    ncclComm_t, const struct mscclEngineAlgo*, int, size_t, size_t);

/* ── algo loading ───────────────────────────────────────────────────────── */

#define MAX_ALGOS 32

struct LoadedAlgo {
    void                                     *handle;
    const struct mscclAlgoDescriptor         *desc;
};

static struct LoadedAlgo g_algos[MAX_ALGOS];
static int               g_nalgos = -1;

static void load_algos_once(void) {
    if (g_nalgos >= 0) return;
    g_nalgos = 0;

    const char *path_env = getenv("MSCCL_ALGO_PATH");
    if (!path_env || !path_env[0]) {
        fprintf(stderr, "[msccl_preload] MSCCL_ALGO_PATH not set\n");
        return;
    }
    fprintf(stderr, "[msccl_preload] MSCCL_ALGO_PATH=%s\n", path_env);

    char buf[4096];
    strncpy(buf, path_env, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *saveptr = NULL;
    for (char *tok = strtok_r(buf, ":", &saveptr); tok;
              tok = strtok_r(NULL, ":", &saveptr)) {
        if (g_nalgos >= MAX_ALGOS) break;
        void *h = dlopen(tok, RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            fprintf(stderr, "[msccl_preload] dlopen(%s): %s\n", tok, dlerror());
            continue;
        }
        const struct mscclAlgoDescriptor *desc =
            (const struct mscclAlgoDescriptor *)dlsym(h, "msccl_algo_desc");
        if (!desc) {
            fprintf(stderr, "[msccl_preload] %s: no msccl_algo_desc symbol\n", tok);
            dlclose(h);
            continue;
        }
        g_algos[g_nalgos].handle = h;
        g_algos[g_nalgos].desc   = desc;
        g_nalgos++;
        fprintf(stderr, "[msccl_preload] loaded %s  nranks=%d bytes=[%zu,%zu]\n",
                tok, desc->nranks, desc->min_bytes, desc->max_bytes);
    }
    fprintf(stderr, "[msccl_preload] loaded %d algo(s)\n", g_nalgos);
}

/* ── per-comm lazy registration ─────────────────────────────────────────── */

#include <set>
#include <mutex>

static std::set<ncclComm_t>   g_registered_comms;
static std::mutex             g_reg_mutex;

static ncclMscclEngineAddAlgoFn_t get_add_fn(void) {
    static ncclMscclEngineAddAlgoFn_t fn = nullptr;
    if (fn) return fn;

    fn = (ncclMscclEngineAddAlgoFn_t)dlsym(RTLD_NEXT, "ncclMscclEngineAddAlgo");
    if (!fn) {
        void *h = dlopen("libnccl.so", RTLD_NOW | RTLD_NOLOAD);
        if (h) fn = (ncclMscclEngineAddAlgoFn_t)dlsym(h, "ncclMscclEngineAddAlgo");
    }
    if (!fn) {
        void *h = dlopen("libnccl.so.2", RTLD_NOW | RTLD_NOLOAD);
        if (h) fn = (ncclMscclEngineAddAlgoFn_t)dlsym(h, "ncclMscclEngineAddAlgo");
    }
    if (!fn)
        fprintf(stderr, "[msccl_preload] ERROR: ncclMscclEngineAddAlgo not found\n");
    else
        fprintf(stderr, "[msccl_preload] ncclMscclEngineAddAlgo = %p\n", (void*)fn);
    return fn;
}

static void ensure_algos_registered(ncclComm_t comm) {
    {
        std::lock_guard<std::mutex> lock(g_reg_mutex);
        if (g_registered_comms.count(comm)) return;
    }

    load_algos_once();
    if (g_nalgos <= 0) return;

    ncclMscclEngineAddAlgoFn_t add_fn = get_add_fn();
    if (!add_fn) return;

    for (int i = 0; i < g_nalgos; i++) {
        const struct mscclAlgoDescriptor *d = g_algos[i].desc;
        fprintf(stderr,
            "[msccl_preload] register comm=%p algo[%d] nranks=%d bytes=[%zu,%zu]\n",
            (void*)comm, i, d->nranks, d->min_bytes, d->max_bytes);
        ncclResult_t r = add_fn(comm, d->algos, d->nranks, d->min_bytes, d->max_bytes);
        fprintf(stderr, "[msccl_preload] add_algo[%d] -> %d\n", i, (int)r);
    }

    std::lock_guard<std::mutex> lock(g_reg_mutex);
    g_registered_comms.insert(comm);
}

/* ── interceptors ───────────────────────────────────────────────────────── */

extern "C" {

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks,
                               ncclUniqueId id, int rank) {
    typedef ncclResult_t (*fn_t)(ncclComm_t*, int, ncclUniqueId, int);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ncclCommInitRank");
    return real_fn(comm, nranks, id, rank);
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks,
                                     ncclUniqueId id, int rank,
                                     ncclConfig_t *config) {
    typedef ncclResult_t (*fn_t)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ncclCommInitRankConfig");
    return real_fn(comm, nranks, id, rank, config);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist) {
    typedef ncclResult_t (*fn_t)(ncclComm_t*, int, const int*);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ncclCommInitAll");
    return real_fn(comms, ndev, devlist);
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream) {
    ensure_algos_registered(comm);

    static ncclAllReduceFn_t msccl_fn = NULL;
    if (!msccl_fn) {
        msccl_fn = (ncclAllReduceFn_t)dlsym(RTLD_NEXT, "ncclAllReduceMsccl");
        if (!msccl_fn) {
            fprintf(stderr, "[msccl_preload] ERROR: ncclAllReduceMsccl not found\n");
            return ncclInternalError;
        }
        fprintf(stderr, "[msccl_preload] ncclAllReduce -> ncclAllReduceMsccl (fn=%p)\n",
                (void*)msccl_fn);
    }
    return msccl_fn(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

} /* extern "C" */

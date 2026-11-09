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
 *
 * Load failures are fatal (runtime_v3b). When MSCCL_ALGO_PATH is set and an
 * algo cannot be loaded or registered (dlopen fails, no msccl_algo_desc, more
 * than MAX_ALGOS entries, ncclMscclEngineAddAlgo or ncclAllReduceMsccl
 * missing, AddAlgo returns an error, e.g. more than 64 thread blocks per
 * rank, or no algo for the communicator's rank count), the shim prints the
 * reason ("[msccl_preload] ERROR: ..."), then one line that starts with
 * "[msccl_preload] FATAL:", and ends the process with _exit(99). Before v3b it
 * printed the return code and went on, so a run could continue without the
 * algo it was meant to time. _exit, not exit: the proxy threads of the comm
 * are running, and atexit handlers / static destructors could hang or hide
 * the message. MSCCL_PRELOAD_STRICT=0 restores the old behaviour.
 * An algo built for another rank count than the communicator's is skipped
 * (not an error), so one MSCCL_ALGO_PATH can serve communicators of several
 * sizes; a communicator that ends with no registered algo is an error.
 * All of this is on the first-call path; the per-call path is unchanged.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
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

/* ── strict mode ────────────────────────────────────────────────────────── */

#define MSCCL_PRELOAD_FATAL_EXIT 99   /* hw_bridge maps 99 to "msccl_not_loaded" */

/* MSCCL_PRELOAD_STRICT: unset or anything but "0" = strict (default). */
static int preload_strict(void) {
    static int strict = -1;
    if (strict < 0) {
        const char *v = getenv("MSCCL_PRELOAD_STRICT");
        strict = (v && v[0] == '0' && v[1] == '\0') ? 0 : 1;
    }
    return strict;
}

/* A load / registration failure that would leave the run without its MSCCL algo. The caller has printed
 * the reason ("[msccl_preload] ERROR: ..."). Strict: print the FATAL line and end the process now.
 * Not strict: return, as before v3b. */
static void preload_failure(void) {
    if (!preload_strict()) return;
    fprintf(stderr, "[msccl_preload] FATAL: the MSCCL algo is not usable, so this run cannot be an MSCCL "
                    "run; exiting with code %d (MSCCL_PRELOAD_STRICT=0 disables this check; "
                    "NCCL_DEBUG=WARN shows the engine's reason)\n", MSCCL_PRELOAD_FATAL_EXIT);
    fflush(stderr);
    _exit(MSCCL_PRELOAD_FATAL_EXIT);
}

/* ── algo loading ───────────────────────────────────────────────────────── */

#define MAX_ALGOS 32

struct LoadedAlgo {
    void                                     *handle;
    const struct mscclAlgoDescriptor         *desc;
    char                                      path[512];   /* for error messages */
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
    if (strlen(path_env) >= sizeof(buf)) {
        fprintf(stderr, "[msccl_preload] ERROR: MSCCL_ALGO_PATH is %zu bytes, longer than the %zu-byte "
                        "buffer; the list would be cut\n", strlen(path_env), sizeof(buf) - 1);
        preload_failure();
    }

    for (char *tok = strtok_r(buf, ":", &saveptr); tok;
              tok = strtok_r(NULL, ":", &saveptr)) {
        if (g_nalgos >= MAX_ALGOS) {
            fprintf(stderr, "[msccl_preload] ERROR: MSCCL_ALGO_PATH has more than %d entries; %s and "
                            "the entries after it are not loaded\n", MAX_ALGOS, tok);
            preload_failure();
            break;
        }
        void *h = dlopen(tok, RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            fprintf(stderr, "[msccl_preload] ERROR: dlopen(%s): %s\n", tok, dlerror());
            preload_failure();
            continue;
        }
        const struct mscclAlgoDescriptor *desc =
            (const struct mscclAlgoDescriptor *)dlsym(h, "msccl_algo_desc");
        if (!desc) {
            fprintf(stderr, "[msccl_preload] ERROR: %s: no msccl_algo_desc symbol\n", tok);
            preload_failure();
            dlclose(h);
            continue;
        }
        g_algos[g_nalgos].handle = h;
        g_algos[g_nalgos].desc   = desc;
        snprintf(g_algos[g_nalgos].path, sizeof(g_algos[g_nalgos].path), "%s", tok);
        g_nalgos++;
        fprintf(stderr, "[msccl_preload] loaded %s  nranks=%d bytes=[%zu,%zu]\n",
                tok, desc->nranks, desc->min_bytes, desc->max_bytes);
    }
    fprintf(stderr, "[msccl_preload] loaded %d algo(s)\n", g_nalgos);
}

/* Inspect loaded algo descriptors and set NCCL_MSCCL_MAX_CHANNELS to the max
 * nChannels actually used by any algo. ncclTransportMscclConnect (called from
 * ncclCommInitRankConfig) does an O(nRanks × maxChannels × numConns) all-to-all
 * P2P setup at init time, defaulting to 16 channels — wildly more than any
 * shipped algo needs. Setting this env var down to the real value drops init
 * cost by up to 16× and is what unblocks runs at nRanks > 64.
 *
 * NCCL_PARAM caches the value on first read inside libnccl, so this must run
 * BEFORE the real ncclCommInitRank* is called. Uses setenv overwrite=0 so a
 * user-provided NCCL_MSCCL_MAX_CHANNELS still wins. */
static void apply_adaptive_msccl_env(void) {
    if (g_nalgos <= 0) return;
    int maxCh = 0;
    for (int i = 0; i < g_nalgos; i++) {
        const struct mscclAlgoDescriptor *d = g_algos[i].desc;
        if (!d || !d->algos || d->nranks <= 0) continue;
        /* All ranks in a given IR share the same nChannels (codegen emits the
         * XML root's nchannels attribute into every per-rank entry). Scanning
         * rank 0 is sufficient; scan all anyway for robustness. */
        for (int r = 0; r < d->nranks; r++) {
            int c = d->algos[r].nChannels;
            if (c > maxCh) maxCh = c;
        }
    }
    if (maxCh <= 0) return;
    char val[32];
    snprintf(val, sizeof(val), "%d", maxCh);
    if (setenv("NCCL_MSCCL_MAX_CHANNELS", val, 0) == 0) {
        const char *eff = getenv("NCCL_MSCCL_MAX_CHANNELS");
        fprintf(stderr,
                "[msccl_preload] adaptive NCCL_MSCCL_MAX_CHANNELS: algo-max=%d, effective=%s\n",
                maxCh, eff ? eff : "(unset)");
    }
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
    if (!fn) {
        fprintf(stderr, "[msccl_preload] ERROR: ncclMscclEngineAddAlgo not found in libnccl "
                        "(is LD_LIBRARY_PATH the MSCCL runtime?)\n");
        preload_failure();
    } else
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

    /* Rank count of this communicator (public API), to tell "algo for another communicator size" from
     * a rejected algo. -1 = unknown: every algo is then tried, as before v3b. */
    int comm_nranks = -1;
    {
        typedef ncclResult_t (*count_fn_t)(const ncclComm_t, int*);
        count_fn_t count_fn = (count_fn_t)dlsym(RTLD_NEXT, "ncclCommCount");
        if (!count_fn || count_fn(comm, &comm_nranks) != ncclSuccess) comm_nranks = -1;
    }

    int nfailed = 0, nskipped = 0;
    for (int i = 0; i < g_nalgos; i++) {
        const struct mscclAlgoDescriptor *d = g_algos[i].desc;
        if (preload_strict() && comm_nranks > 0 && d->nranks != comm_nranks) {
            nskipped++;
            fprintf(stderr,
                "[msccl_preload] skip algo[%d] %s on comm=%p: built for %d ranks, the communicator has %d\n",
                i, g_algos[i].path, (void*)comm, d->nranks, comm_nranks);
            continue;
        }
        fprintf(stderr,
            "[msccl_preload] register comm=%p algo[%d] nranks=%d bytes=[%zu,%zu]\n",
            (void*)comm, i, d->nranks, d->min_bytes, d->max_bytes);
        ncclResult_t r = add_fn(comm, d->algos, d->nranks, d->min_bytes, d->max_bytes);
        fprintf(stderr, "[msccl_preload] add_algo[%d] -> %d\n", i, (int)r);
        if (r != ncclSuccess) {
            nfailed++;
            fprintf(stderr,
                "[msccl_preload] ERROR: ncclMscclEngineAddAlgo rejected algo[%d] %s (nranks=%d "
                "bytes=[%zu,%zu]) with code %d on comm=%p\n",
                i, g_algos[i].path, d->nranks, d->min_bytes, d->max_bytes, (int)r, (void*)comm);
        }
    }
    int nregistered = g_nalgos - nfailed - nskipped;
    if (nfailed == 0 && nregistered == 0)
        fprintf(stderr,
            "[msccl_preload] ERROR: none of the %d loaded algo(s) is for a %d-rank communicator (comm=%p)\n",
            g_nalgos, comm_nranks, (void*)comm);
    if (nfailed || nregistered == 0) preload_failure();
    fprintf(stderr, "[msccl_preload] registered %d of %d algo(s) on comm=%p (%d rejected, %d skipped)\n",
            nregistered, g_nalgos, (void*)comm, nfailed, nskipped);

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
    load_algos_once();
    apply_adaptive_msccl_env();
    return real_fn(comm, nranks, id, rank);
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks,
                                     ncclUniqueId id, int rank,
                                     ncclConfig_t *config) {
    typedef ncclResult_t (*fn_t)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ncclCommInitRankConfig");
    load_algos_once();
    apply_adaptive_msccl_env();
    return real_fn(comm, nranks, id, rank, config);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist) {
    typedef ncclResult_t (*fn_t)(ncclComm_t*, int, const int*);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ncclCommInitAll");
    load_algos_once();
    apply_adaptive_msccl_env();
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
            fprintf(stderr, "[msccl_preload] ERROR: ncclAllReduceMsccl not found in libnccl "
                            "(is LD_LIBRARY_PATH the MSCCL runtime?)\n");
            preload_failure();
            return ncclInternalError;
        }
        fprintf(stderr, "[msccl_preload] ncclAllReduce -> ncclAllReduceMsccl (fn=%p)\n",
                (void*)msccl_fn);
    }
    return msccl_fn(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

} /* extern "C" */

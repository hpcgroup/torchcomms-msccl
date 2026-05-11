#!/usr/bin/env python3
"""
ir_codegen_interp.py — MSCCL IR → mscclEngineAlgo struct data generator.

Instead of the previous approach (emitting unrolled CUDA kernels that OOM nvcc
at 256-GPU scale), this script emits C struct initialiser data for
mscclEngineAlgo.  The generated file is just a const array — no CUDA kernel
code — so it compiles in seconds regardless of algorithm size.

The companion engine kernel (msccl_kernel.cu + msccl_engine.cuh) is fixed and
small; only this data file changes per algorithm.

Usage
-----
    python3 ir_codegen_interp.py <output_dir> <ir1.xml> [<ir2.xml> ...]
        [--min-bytes N] [--max-bytes N]

Output (per IR file named <stem>.xml):
    <output_dir>/msccl_algo_<stem>.cu   — C struct initialisers
    <output_dir>/msccl_algo_<stem>.h    — extern declarations + metadata macros

The .cu files include "include/msccl_engine.h" and define one symbol:
    const struct mscclEngineAlgo msccl_algo_<stem>[<nranks>]
indexed by rank (0 .. nranks-1).  At runtime, copy the appropriate entry to
device memory and pass the device pointer to the engine kernel.

Struct limits (must match msccl_engine.h):
    MSCCL_ENGINE_MAX_STEPS = 512
    MSCCL_ENGINE_MAX_TBS   = 64
"""

import sys
import os
import re
import xml.etree.ElementTree as ET

MAX_STEPS = 512
MAX_TBS   = 64
SIZE_MAX  = (1 << 64) - 1   # used as "no upper bound" in registry

_SIZE_UNITS = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}

def _size_from_stem(stem):
    """Return target size in bytes inferred from stem like foo_8gpus_512MB."""
    m = re.search(r'_(\d+)(KB|MB|GB)(?:_|$)', stem, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)) * _SIZE_UNITS[m.group(2).upper()]

def _size_range(target_bytes):
    """Return (minBytes, maxBytes) for a target size.
    Uses [target//4, target*4] so adjacent size classes don't overlap."""
    if target_bytes is None:
        return 0, SIZE_MAX
    return max(0, target_bytes // 4), target_bytes * 4

# ── XML type code → MSCCL_ENGINE_* opcode ──────────────────────────────────
OPCODE = {
    's':    'MSCCL_ENGINE_SEND',
    'r':    'MSCCL_ENGINE_RECV',
    'rcs':  'MSCCL_ENGINE_RECV_COPY_SEND',
    'rrs':  'MSCCL_ENGINE_RECV_REDUCE_SEND',
    'rrc':  'MSCCL_ENGINE_RECV_REDUCE_COPY',
    'rrcs': 'MSCCL_ENGINE_RECV_REDUCE_COPY_SEND',
    'cpy':  'MSCCL_ENGINE_LOCAL_COPY',
    're':   'MSCCL_ENGINE_REDUCE',
    'nop':  'MSCCL_ENGINE_NOP',
}

BUF = {'i': 'MSCCL_ENGINE_INPUT_BUF',
       'o': 'MSCCL_ENGINE_OUTPUT_BUF',
       's': 'MSCCL_ENGINE_SCRATCH_BUF'}

PROTO = {'Simple': 'NCCL_PROTO_SIMPLE',
         'LL':     'NCCL_PROTO_LL',
         'LL128':  'NCCL_PROTO_LL128'}


def _ident(path: str) -> str:
    """Derive a C identifier from an XML filename stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'[^A-Za-z0-9]', '_', stem)


def _parse_ir(xml_path: str):
    """
    Parse an MSCCL IR XML file.

    Returns a dict:
        {
          'name': str,
          'proto': str,           # 'Simple'|'LL'|'LL128'
          'nchannels': int,
          'nchunksperloop': int,
          'ngpus': int,
          'coll': str,
          'inplace': bool,
          'gpus': [                # indexed by gpu id
            {
              'id': int,
              'i_chunks': int,
              'o_chunks': int,
              's_chunks': int,
              'tbs': [             # indexed by tb id
                {
                  'id': int,
                  'send': int,    # -1 if absent
                  'recv': int,    # -1 if absent
                  'chan': int,
                  'steps': [
                    {
                      's': int,
                      'type': str,
                      'srcbuf': str,
                      'dstbuf': str,
                      'srcoff': int,
                      'dstoff': int,
                      'cnt': int,
                      'depid': int,  # -1 if none
                      'deps': int,   # -1 if none
                      'hasdep': int,
                    }, ...
                  ]
                }, ...
              ]
            }, ...
          ]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    algo = {
        'name':           root.get('name', ''),
        'proto':          root.get('proto', 'Simple'),
        'nchannels':      int(root.get('nchannels', 1)),
        'nchunksperloop': int(root.get('nchunksperloop', 1)),
        'ngpus':          int(root.get('ngpus', 1)),
        'coll':           root.get('coll', 'allreduce'),
        'inplace':        root.get('inplace', '0') == '1',
        'gpus':           [],
    }

    for gpu_el in sorted(root.findall('gpu'), key=lambda e: int(e.get('id', 0))):
        gpu = {
            'id':       int(gpu_el.get('id', 0)),
            'i_chunks': int(gpu_el.get('i_chunks', 0)),
            'o_chunks': int(gpu_el.get('o_chunks', 0)),
            's_chunks': int(gpu_el.get('s_chunks', 0)),
            'tbs':      [],
        }
        for tb_el in sorted(gpu_el.findall('tb'), key=lambda e: int(e.get('id', 0))):
            tb = {
                'id':    int(tb_el.get('id', 0)),
                'send':  int(tb_el.get('send', -1)),
                'recv':  int(tb_el.get('recv', -1)),
                'chan':  int(tb_el.get('chan', -1)),
                'steps': [],
            }
            for step_el in sorted(tb_el.findall('step'),
                                   key=lambda e: int(e.get('s', 0))):
                tb['steps'].append({
                    's':      int(step_el.get('s', 0)),
                    'type':   step_el.get('type', 'nop'),
                    'srcbuf': step_el.get('srcbuf', 'i'),
                    'dstbuf': step_el.get('dstbuf', 'i'),
                    'srcoff': int(step_el.get('srcoff', 0)),
                    'dstoff': int(step_el.get('dstoff', 0)),
                    'cnt':    int(step_el.get('cnt', 1)),
                    'depid':  int(step_el.get('depid', -1)),
                    'deps':   int(step_el.get('deps', -1)),
                    'hasdep': int(step_el.get('hasdep', 0)),
                })
            gpu['tbs'].append(tb)
        algo['gpus'].append(gpu)

    return algo


def _build_dep_tables(tbs_for_rank, max_steps):
    """
    Build per-TB dependentBid, dependentStep, reductionSrcOffsets arrays.

    For each TB that has steps with deps, the TB that the step depends on
    (depid) and which step (deps) are recorded in sequential order.
    The transmissions[i].dependencePointer/numDependencies index into these.

    Returns a list of dicts, one per TB:
        {
          'dependentBid':   list[int],   length max_steps, unused entries = -1/0
          'dependentStep':  list[int],
          'reductionSrcOffsets': list[int],
          'transmissions': list[dict],   augmented with dependencePointer, numDependencies, etc.
        }
    """
    result = []
    for tb in tbs_for_rank:
        dep_bids  = [-1] * max_steps
        dep_steps = [-1] * max_steps
        red_srcs  = [0]  * max_steps
        dep_ptr   = 0   # next free slot in dep_bids/dep_steps

        transmissions = []
        for step in tb['steps']:
            depid  = step['depid']
            deps   = step['deps']
            hasdep = step['hasdep']

            if depid >= 0 and deps >= 0:
                dep_bids[dep_ptr]  = depid
                dep_steps[dep_ptr] = deps
                t = {
                    'dependencePointer': dep_ptr,
                    'numDependencies':   1,
                    'reductionPointer':  0,
                    'numReductions':     0,
                    'srcOffset':  step['srcoff'],
                    'dstOffset':  step['dstoff'],
                    'srcBuffer':  step['srcbuf'],
                    'dstBuffer':  step['dstbuf'],
                    'hasDependence': hasdep,
                    'type':  step['type'],
                    'count': step['cnt'],
                }
                dep_ptr += 1
            else:
                t = {
                    'dependencePointer': 0,
                    'numDependencies':   0,
                    'reductionPointer':  0,
                    'numReductions':     0,
                    'srcOffset':  step['srcoff'],
                    'dstOffset':  step['dstoff'],
                    'srcBuffer':  step['srcbuf'],
                    'dstBuffer':  step['dstbuf'],
                    'hasDependence': hasdep,
                    'type':  step['type'],
                    'count': step['cnt'],
                }
            transmissions.append(t)

        result.append({
            'dependentBid':        dep_bids,
            'dependentStep':       dep_steps,
            'reductionSrcOffsets': red_srcs,
            'transmissions':       transmissions,
        })
    return result


def _emit_transmission(t, indent='          '):
    """Emit one mscclEngineTransmission initialiser."""
    return (
        f"{indent}{{"
        f".dependencePointer={t['dependencePointer']}, "
        f".numDependencies={t['numDependencies']}, "
        f".reductionPointer={t['reductionPointer']}, "
        f".numReductions={t['numReductions']}, "
        f".srcOffset={t['srcOffset']}, "
        f".dstOffset={t['dstOffset']}, "
        f".srcBuffer={BUF[t['srcBuffer']]}, "
        f".dstBuffer={BUF[t['dstBuffer']]}, "
        f".hasDependence={t['hasDependence']}, "
        f".type={OPCODE.get(t['type'], 'MSCCL_ENGINE_NOP')}, "
        f".count={t['count']}, "
        f"._pad=0"
        f"}}"
    )


def _emit_tb(tb_data, steps, max_steps, send_peer, recv_peer, chan_id, indent='      '):
    """
    Emit one mscclEngineThreadBlock initialiser.

    Only emits the entries that are actually used; C99 zero-initialises the
    rest.  For arrays that should be -1 (dependentBid/dependentStep), entries
    beyond dep_ptr are never accessed when numDependencies==0, so zero is safe.

    tb_data : dict from _build_dep_tables
    steps   : list of raw step dicts (for nSteps count)
    """
    ind2 = indent + '  '
    ind3 = ind2   + '  '

    # Transmissions — only emit used steps; remainder are zero-init (NOP).
    trans_lines = []
    for idx, t in enumerate(tb_data['transmissions']):
        trans_lines.append(f"[{idx}] = " + _emit_transmission(t, '').lstrip())

    trans_body = (',\n' + ind3).join(trans_lines)

    # Dependency arrays — only emit the slots that were actually written.
    dep_bids  = tb_data['dependentBid']
    dep_steps = tb_data['dependentStep']
    red_srcs  = tb_data['reductionSrcOffsets']

    # Count how many dep slots were used (first run of non-(-1) values).
    # n_dep: entries are filled from index 0; first -1 marks the end.
    n_dep = next((i for i, v in enumerate(dep_bids) if v == -1), len(dep_bids))
    # n_red: emit through the last non-zero entry (0 is a valid offset, so find last non-zero).
    n_red = max((i + 1 for i, v in enumerate(red_srcs) if v != 0), default=0)

    lines = [
        f"{indent}{{",
        f"{ind2}.transmissions = {{",
        f"{ind3}{trans_body}",
        f"{ind2}}},",
    ]

    if n_dep > 0:
        dep_bid_vals  = ', '.join(str(v) for v in dep_bids[:n_dep])
        dep_step_vals = ', '.join(str(v) for v in dep_steps[:n_dep])
        lines.append(f"{ind2}.dependentBid  = {{{dep_bid_vals}}},")
        lines.append(f"{ind2}.dependentStep = {{{dep_step_vals}}},")

    if n_red > 0:
        red_vals = ', '.join(str(v) for v in red_srcs[:n_red])
        lines.append(f"{ind2}.reductionSrcOffsets = {{{red_vals}}},")

    lines += [
        f"{ind2}.sendPeer  = {send_peer},",
        f"{ind2}.recvPeer  = {recv_peer},",
        f"{ind2}.nSteps    = {len(steps)},",
        f"{ind2}.channelId = {chan_id}",
        f"{indent}}}",
    ]
    return '\n'.join(lines)


def codegen_interp(xml_path: str, out_dir: str,
                   min_bytes: int = None, max_bytes: int = None):
    """Generate msccl_algo_<stem>.cu and msccl_algo_<stem>.h."""
    algo  = _parse_ir(xml_path)
    ident = _ident(xml_path)
    ngpus = algo['ngpus']

    # Size range for the registry: use explicit args, then filename inference.
    _target = _size_from_stem(ident)
    _auto_min, _auto_max = _size_range(_target)
    reg_min = min_bytes if min_bytes is not None else _auto_min
    reg_max = max_bytes if max_bytes is not None else _auto_max

    # Determine max TBs per rank and max steps per TB across all ranks.
    max_tbs_per_rank  = max(len(g['tbs']) for g in algo['gpus']) if algo['gpus'] else 0
    max_steps_per_tb  = max(
        (len(tb['steps']) for g in algo['gpus'] for tb in g['tbs']),
        default=0)
    assert max_tbs_per_rank <= MAX_TBS, \
        f"max TBs per rank {max_tbs_per_rank} exceeds MSCCL_ENGINE_MAX_TBS={MAX_TBS}"
    assert max_steps_per_tb <= MAX_STEPS, \
        f"max steps {max_steps_per_tb} exceeds MSCCL_ENGINE_MAX_STEPS={MAX_STEPS}"

    has_scratch = any(g['s_chunks'] > 0 for g in algo['gpus'])
    has_reduce  = any(
        s['type'] in ('rrc', 'rrs', 'rrcs', 're')
        for g in algo['gpus'] for tb in g['tbs'] for s in tb['steps'])

    proto_c  = PROTO.get(algo['proto'], 'NCCL_PROTO_SIMPLE')
    nchunks  = algo['nchunksperloop']

    # ── .cu file ────────────────────────────────────────────────────────────
    cu_lines = [
        f'/* Auto-generated by ir_codegen_interp.py from {os.path.basename(xml_path)} */',
        f'/* DO NOT EDIT — regenerate with ir_codegen_interp.py                        */',
        '',
        '#include "include/msccl_engine.h"',
        '#ifndef MSCCL_STANDALONE',
        '#include "msccl_registry.h"',
        '#endif',
        f'#include "msccl_algo_{ident}.h"',
        '',
        f'const struct mscclEngineAlgo msccl_algo_{ident}[{ngpus}] = {{',
    ]

    for rank, gpu in enumerate(algo['gpus']):
        assert gpu['id'] == rank, \
            f"GPU ids must be contiguous and sorted; got {gpu['id']} at position {rank}"

        tbs = gpu['tbs']
        nblocks = len(tbs)
        tb_tables = _build_dep_tables(tbs, MAX_STEPS)

        cu_lines.append(f'  /* rank {rank} */')
        cu_lines.append(f'  [{rank}] = {{')
        cu_lines.append(f'    .nChunksPerLoop = {nchunks},')
        cu_lines.append(f'    .protocol       = {proto_c},')
        cu_lines.append(f'    .nChannels      = {algo["nchannels"]},')
        cu_lines.append(f'    .nRanks         = {ngpus},')
        cu_lines.append(f'    .nBlocks        = {nblocks},')
        cu_lines.append(f'    .nScratchChunks = {gpu["s_chunks"]},')
        cu_lines.append(f'    .sizeMultiplier = 1,')
        cu_lines.append(f'    .chunkSteps     = MSCCL_ENGINE_CHUNKSTEPS,')
        cu_lines.append(f'    .sliceSteps     = MSCCL_ENGINE_SLICESTEPS,')
        cu_lines.append(f'    .hasReduce      = {1 if has_reduce else 0},')
        cu_lines.append(f'    ._pad           = {{0}},')
        cu_lines.append(f'    .mscclTBs = {{')

        for tb_idx, (tb, tb_data) in enumerate(zip(tbs, tb_tables)):
            assert tb['id'] == tb_idx, \
                f"TB ids must be contiguous; rank {rank} tb {tb['id']} at {tb_idx}"
            tb_str = _emit_tb(
                tb_data, tb['steps'], MAX_STEPS,
                send_peer=tb['send'], recv_peer=tb['recv'], chan_id=tb['chan'],
                indent='      ')
            cu_lines.append(f'      [{tb_idx}] = {tb_str.lstrip()},')

        cu_lines.append('    }')
        cu_lines.append(f'  }}{"," if rank < ngpus - 1 else ""}')

    cu_lines.append('};')
    cu_lines.append('')
    cu_lines.append('/* Standalone descriptor: dlsym target for ncclmsccl_preload. */')
    cu_lines.append('#ifdef __cplusplus')
    cu_lines.append('extern "C" {')
    cu_lines.append('#endif')
    cu_lines.append(f'extern const struct mscclAlgoDescriptor msccl_algo_desc = {{')
    cu_lines.append(f'  msccl_algo_{ident},')
    cu_lines.append(f'  {ngpus},')
    cu_lines.append(f'  {reg_min}ULL,')
    cu_lines.append(f'  {reg_max}ULL,')
    cu_lines.append('};')
    cu_lines.append('#ifdef __cplusplus')
    cu_lines.append('}')
    cu_lines.append('#endif')
    cu_lines.append('')
    cu_lines.append('#ifndef MSCCL_STANDALONE')
    cu_lines.append(
        f'MSCCL_REGISTER_ALGO(msccl_algo_{ident}, {ngpus}, '
        f'ncclFuncAllReduce, {reg_min}ULL, {reg_max}ULL);')
    cu_lines.append('#endif')
    cu_lines.append('')

    cu_path = os.path.join(out_dir, f'msccl_algo_{ident}.cu')
    with open(cu_path, 'w') as f:
        f.write('\n'.join(cu_lines))
    print(f'Wrote {cu_path}')

    # ── .h file ─────────────────────────────────────────────────────────────
    guard  = f'MSCCL_ALGO_{ident.upper()}_H_'
    h_lines = [
        f'/* Auto-generated by ir_codegen_interp.py from {os.path.basename(xml_path)} */',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include "include/msccl_engine.h"',
        '',
        f'/* Number of ranks (GPU count) this algorithm targets. */',
        f'#define MSCCL_ALGO_{ident.upper()}_NGPUS         {ngpus}',
        f'#define MSCCL_ALGO_{ident.upper()}_NCHUNKS       {nchunks}',
        f'#define MSCCL_ALGO_{ident.upper()}_NCHANNELS     {algo["nchannels"]}',
        f'#define MSCCL_ALGO_{ident.upper()}_MAX_NBLOCKS   {max_tbs_per_rank}',
        f'#define MSCCL_ALGO_{ident.upper()}_MAX_STEPS     {max_steps_per_tb}',
        f'#define MSCCL_ALGO_{ident.upper()}_HAS_SCRATCH   {1 if has_scratch else 0}',
        f'#define MSCCL_ALGO_{ident.upper()}_HAS_REDUCE    {1 if has_reduce else 0}',
        '',
        f'/* One entry per rank; copy entry [myRank] to device memory before launch. */',
        f'extern const struct mscclEngineAlgo msccl_algo_{ident}[{ngpus}];',
        '',
        f'#endif /* {guard} */',
        '',
    ]

    h_path = os.path.join(out_dir, f'msccl_algo_{ident}.h')
    with open(h_path, 'w') as f:
        f.write('\n'.join(h_lines))
    print(f'Wrote {h_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate mscclEngineAlgo struct data from MSCCL IR XMLs.')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('xml_files', nargs='+', help='IR XML files')
    parser.add_argument('--min-bytes', type=int, default=None,
                        help='Registry minBytes override (default: derived from filename)')
    parser.add_argument('--max-bytes', type=int, default=None,
                        help='Registry maxBytes override (default: derived from filename)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for xml_path in args.xml_files:
        try:
            codegen_interp(xml_path, args.out_dir,
                           min_bytes=args.min_bytes,
                           max_bytes=args.max_bytes)
        except Exception as e:
            print(f'ERROR processing {xml_path}: {e}', file=sys.stderr)
            raise


if __name__ == '__main__':
    main()

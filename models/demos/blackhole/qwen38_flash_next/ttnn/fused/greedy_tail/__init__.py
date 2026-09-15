# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``greedy_tail``: the greedy epilogue's candidates and resolve as three data-movement programs around one gather.

Candidates (``Qwen38TTNNLMHead.greedy_candidates``, 9 programs: untilize, argmax, pad, reshape, max, reshape, max):
``scan`` on 40 cores reads row 0 of the local bf16 logits tiles and keeps each core's maximum with its lowest id;
``merge`` on one core picks the local maximum (lowest id on ties, ttnn.argmax's rule) and writes ``local_values``
(bf16 TILE ``[1,1,1,1]``), ``local_indices`` (uint32 ``[1,1,1]``) and the packed fp32 ROW_MAJOR row ``[value | id]``.
Resolve (``resolve_greedy_on_device``, 15 programs: two all_gathers with their relayouts, tie-break, argmax, rebase,
gather, splat, tilize): ONE ``all_gather`` of the packed rows, then ``resolve`` on one core: ``value_d -
owner_tie_break[d]`` in fp32, the first maximum in owner order, ``id + lm_head_vocab_starts[owner]`` in fp32, written
as lane 0 of the fp32 TILE token row.  Integer / exact-fp32 work on the RISC (IEEE round-to-nearest, as the SFPU's
fp32 ops): tolerance class BITWISE on the token id and on both candidate tensors.  Rows = 1 (the decode step); the
MTP rows path keeps the chain.
"""

from __future__ import annotations

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "greedy_tail"
TILE = fp.TILE
KERNELS = {name: fp.kernel_source(NAME, f"{name}.cpp") for name in ("scan", "merge", "resolve")}
CB_STAGE = 0
SCAN_CORES = 40
SCAN_CORES_ENV = (
    "QWEN38_FUSED_GREEDY_TAIL_SCAN_CORES"  # dev knob: cores of the scan program (40 default; 80 / 130 to sweep)
)


def scan_cores(environ=None) -> int:
    import os

    value = (os.environ if environ is None else environ).get(SCAN_CORES_ENV, "")
    return int(value) if value.strip() else SCAN_CORES


SCAN_STAGE_PAGES = 4  # 49 tiles x 128 bytes of face rows + the 16-byte pair
MERGE_STAGE_PAGES = 2  # zero tile + pairs + out
RESOLVE_STAGE_PAGES = 3  # fp32 zero tile (4 KB) + three 64-byte reads
SCAN_ARGS = ("logits_addr", "pairs_addr", "first_tile", "tile_count", "core_index")
MERGE_ARGS = ("pairs_addr", "zero_tile_addr", "values_addr", "indices_addr", "packed_addr")
RESOLVE_ARGS = ("gathered_addr", "tie_break_addr", "vocab_starts_addr", "zero_tile_addr", "token_row_addr")
_ZERO_TILES: dict[int, tuple] = {}


def _embedding():
    from models.demos.blackhole.qwen38_flash_next.ttnn import embedding

    return embedding


def prepare(mesh):
    """One zero bf16 tile and one zero fp32 tile ``[1,1,32,32]`` per mesh (the merge's value tile and the resolve's token
    tile start from them), uploaded once (before trace capture)."""

    import torch

    key = id(mesh)
    if key not in _ZERO_TILES:
        zeros = torch.zeros(1, 1, TILE, TILE)
        _ZERO_TILES[key] = tuple(
            ttnn.from_torch(
                zeros, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for dtype in (ttnn.bfloat16, ttnn.float32)
        )
    return _ZERO_TILES[key]


def _one_core(mesh):
    core = ttnn.CoreCoord(0, 0)
    return core, ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _rows_of(logits) -> int:
    shape = tuple(int(v) for v in logits.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or logits.dtype != ttnn.bfloat16 or logits.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            f"greedy_tail logits must be bf16 TILE [1,1,rows,vocab], got {logits.dtype} {logits.layout} {shape}"
        )
    if shape[3] % TILE:
        raise ValueError(f"greedy_tail logits width must be whole tiles, got {shape[3]}")
    return shape[2]


def greedy_candidates(logits, *, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> tuple:
    """``(local_values bf16 TILE [1,1,1,1], local_indices uint32 [1,1,1], packed fp32 ROW_MAJOR [1,1,1,2])`` of one row."""

    if _rows_of(logits) != 1:
        raise ValueError("greedy_tail candidates take one row (the decode step)")
    mesh = logits.device()
    tiles = int(logits.shape[3]) // TILE
    zero_bf16, _zero_fp32 = prepare(mesh)
    grid = mesh.compute_with_storage_grid_size()
    work = fp.split_work(tiles, mesh, cores=min(scan_cores(), tiles, grid.x * grid.y))
    grid = fp.core_rectangle(work, mesh)
    pairs_lanes = -(-4 * len(work) // 16) * 16  # 16 bytes per core, the row padded to the 64-byte DRAM read grain
    pairs = fp.allocate((1, 1, 1, pairs_lanes), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    scan = fp.reader_kernel(
        KERNELS["scan"],
        grid,
        fp.accessor_args(logits) + fp.accessor_args(pairs),
        [(w.core, [logits.buffer_address(), pairs.buffer_address(), w.start, w.count, i]) for i, w in enumerate(work)],
        named={"cb_stage": CB_STAGE, "lanes_per_tile": TILE},
    )
    fp.run_program(
        [logits, pairs],
        fp.program_descriptor([scan], cbs=[fp.cb_descriptor(CB_STAGE, ttnn.bfloat16, 2048, SCAN_STAGE_PAGES, grid)]),
    )
    values = fp.allocate((1, 1, 1, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, memory_config)
    indices = fp.allocate((1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    packed = fp.allocate((1, 1, 1, 2), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    core, one = _one_core(mesh)
    tensors = [pairs, zero_bf16, values, indices, packed]
    merge = fp.reader_kernel(
        KERNELS["merge"],
        one,
        [a for t in tensors for a in fp.accessor_args(t)],
        [(core, [t.buffer_address() for t in tensors])],
        named={"cb_stage": CB_STAGE, "cores": len(work)},
    )
    fp.run_program(
        tensors,
        fp.program_descriptor([merge], cbs=[fp.cb_descriptor(CB_STAGE, ttnn.bfloat16, 2048, MERGE_STAGE_PAGES, one)]),
    )
    ttnn.deallocate(pairs)
    for tensor in (values, indices, packed):
        tensor.update_tensor_topology(logits.tensor_topology())  # local partials, as the chain's argmax / max outputs
    return values, indices, packed


def greedy_candidates_composed(logits, *, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> tuple:
    """The chain op for op (the grid-reduce form of ``Qwen38TTNNLMHead.greedy_candidates``); ``packed`` is built from
    its two outputs with the chain's own widening ops so the resolve's input can be compared too."""

    rows = _rows_of(logits)
    row_major = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    local_indices = ttnn.argmax(row_major, dim=-1, keepdim=False)
    ttnn.deallocate(row_major)
    columns = 32
    width = int(logits.shape[3])
    grid_rows = ((width + columns - 1) // columns + 31) // 32 * 32
    padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, grid_rows * columns - width)], value=-1e30)
    grid = ttnn.reshape(padded, (1, rows, grid_rows, columns))
    partial = ttnn.max(grid, dim=-1)
    partial_rows = ttnn.reshape(partial, (1, 1, rows, grid_rows))
    local_values = ttnn.max(partial_rows, dim=-1, keepdim=True)
    for tensor in (padded, grid, partial, partial_rows):
        ttnn.deallocate(tensor)
    values_fp32 = ttnn.typecast(
        ttnn.to_layout(local_values, ttnn.ROW_MAJOR_LAYOUT), ttnn.float32, memory_config=memory_config
    )
    index_fp32 = ttnn.typecast(ttnn.reshape(local_indices, (1, 1, 1, 1)), ttnn.float32, memory_config=memory_config)
    packed = ttnn.concat([values_fp32, index_fp32], dim=3, memory_config=memory_config)
    ttnn.deallocate(values_fp32)
    ttnn.deallocate(index_fp32)
    return local_values, local_indices, packed


def resolve(gathered, tie_break, vocab_starts, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The token row (fp32 TILE ``[1,1,1,32]``, lane 0 = the global id) from the gathered packed rows ``[1,1,1,2*devices]``."""

    embedding = _embedding()
    devices = int(gathered.shape[3]) // 2
    if (
        gathered.dtype != ttnn.float32
        or gathered.layout != ttnn.ROW_MAJOR_LAYOUT
        or tuple(gathered.shape)[:3] != (1, 1, 1)
    ):
        raise ValueError("greedy_tail resolve takes the fp32 ROW_MAJOR gathered packed row [1,1,1,2*devices]")
    mesh = gathered.device()
    _zero_bf16, zero_fp32 = prepare(mesh)
    token_row = fp.allocate(embedding.TOKEN_ROW_SHAPE, ttnn.float32, ttnn.TILE_LAYOUT, mesh, memory_config)
    core, one = _one_core(mesh)
    tensors = [gathered, tie_break, vocab_starts, zero_fp32, token_row]
    kernel = fp.reader_kernel(
        KERNELS["resolve"],
        one,
        [a for t in tensors for a in fp.accessor_args(t)],
        [(core, [t.buffer_address() for t in tensors])],
        named={"cb_stage": CB_STAGE, "devices": devices},
    )
    fp.run_program(
        tensors,
        fp.program_descriptor([kernel], cbs=[fp.cb_descriptor(CB_STAGE, ttnn.float32, 4096, RESOLVE_STAGE_PAGES, one)]),
    )
    token_row.update_tensor_topology(tie_break.tensor_topology())  # replicated, as the chain's token row
    return token_row


def resolve_composed(
    gathered_values, gathered_indices, tie_break, vocab_starts, unit_column, *, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """The chain's post-gather ops of ``resolve_greedy_on_device`` on the gathered bf16 TILE values ``[1,1,1,devices]`` and
    uint32 ROW_MAJOR indices ``[1,1,1,devices]``."""

    values_row_major = ttnn.to_layout(gathered_values, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    values_fp32 = ttnn.typecast(values_row_major, ttnn.float32, memory_config=memory_config)
    ranked = ttnn.subtract(values_fp32, tie_break, memory_config=memory_config)
    owner = ttnn.argmax(ranked, dim=-1, keepdim=True)
    index_fp32 = ttnn.typecast(gathered_indices, ttnn.float32, memory_config=memory_config)
    candidate_tokens = ttnn.add(index_fp32, vocab_starts, memory_config=memory_config)
    token = ttnn.gather(candidate_tokens, 3, owner, memory_config=memory_config)
    token_wide = ttnn.multiply(token, unit_column, memory_config=memory_config)
    token_row = ttnn.to_layout(token_wide, ttnn.TILE_LAYOUT, memory_config=memory_config)
    for tensor in (values_row_major, values_fp32, ranked, owner, index_fp32, candidate_tokens, token, token_wide):
        ttnn.deallocate(tensor)
    return token_row


def greedy_candidates_fused(lm_head, logits, *, values_by_gather: bool = False):
    """``Qwen38TTNNLMHead.greedy_candidates`` on the fused programs for one row; other rows keep the chain."""

    embedding = _embedding()
    rows = lm_head._validate_logits(logits)
    if rows != 1:
        return type(lm_head).greedy_candidates(lm_head, logits, values_by_gather=values_by_gather)
    values, indices, packed = greedy_candidates(logits.tensor)
    for tensor, shape in ((indices, (1, 1, rows)), (values, (1, 1, rows, 1)), (packed, (1, 1, 1, 2))):
        lm_head.mesh_contract.mark_local_partial(
            tensor, replicated_reference=lm_head.weights.replicated_anchor, expected_shape=shape
        )
    return embedding.Qwen38GreedyCandidates(
        local_indices=indices, local_values=values, rows=rows, vocab_ranges=lm_head.weights.vocab_ranges, packed=packed
    )


def resolve_greedy_on_device_fused(lm_head, candidates):
    """``Qwen38TTNNLMHead.resolve_greedy_on_device`` with one gather of the packed rows and the resolve program; candidates
    without a packed row (the chain's) take the chain."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    embedding = _embedding()
    if getattr(candidates, "packed", None) is None:
        return type(lm_head).resolve_greedy_on_device(lm_head, candidates)
    if candidates.rows != 1:
        raise TypeError("on-device greedy resolve requires single-token Qwen38GreedyCandidates")
    if candidates.vocab_ranges != lm_head.weights.vocab_ranges:
        raise ValueError("greedy candidates have different vocabulary ownership")
    if lm_head.collective_topology != ttnn.Topology.Linear:
        raise RuntimeError("on-device greedy resolve requires Linear topology")
    lm_head.mesh_contract.validate_tensor(candidates.packed, placement=TensorPlacement.LOCAL_PARTIAL)
    constants = lm_head.weights.token_row
    gathered = ttnn.all_gather(
        candidates.packed, dim=3, cluster_axis=embedding.TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    lm_head.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
    token_row = resolve(gathered, constants.owner_tie_break, constants.lm_head_vocab_starts)
    ttnn.deallocate(gathered)
    lm_head.mesh_contract.validate_tensor(token_row, placement=TensorPlacement.REPLICATED)
    return token_row


def greedy_candidates_chain(lm_head, logits, *, values_by_gather: bool = False):
    return type(lm_head).greedy_candidates(lm_head, logits, values_by_gather=values_by_gather)


register(
    FusedKernel(
        name=NAME,
        replaces="greedy_candidates (9 programs) and resolve_greedy_on_device (15 programs) of the decode tail",
        tolerance=BITWISE,
        fused=greedy_candidates_fused,
        composed=greedy_candidates_chain,
        gate=None,  # the single-chip device test is the component gate (random logits with ties vs the chain's ops)
    )
)

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``final_mixer``: ``Qwen38TTNNFinalMixer.__call__`` (21 programs: rms_norm_pre, reshape, all_gather, rms_norm_post,
gamma multiply, down linear, S2I, the fp32 all_reduce composite's 7 programs, typecast, silu, up linear, S2I, sigmoid,
gate multiply, fast_reduce_nc, reshape) on F2's gr_read programs: ``stats`` and ``normalize`` as they are (the mixer
is the GR read pattern without the injection), ``down`` and ``gate`` with the mixer's shapes (K = 2560 -> 320 tiles
10, spill 8; K = 320 -> 2560, spill 5), and a new ``low_rank`` that sums the four gathered fp32 partials the way the
chain's composite does: ``local_sum_float32`` reshapes the gathered rows to a leading device dim and ``ttnn::sum``
transposes that dim into H (zero pad) and reduces with ReduceOpDim::H; for fp32 with the reduce's default
fp32_dest_acc_en that is the ACCURATE SFPU path (input unpacked to the fp32 dest, rows folded by the fp32 SFPU;
``reduce_op.cpp`` fp32_sfpu_eligible), which the kernel calls through the same ``compute_kernel_lib::reduce`` helper
on a tile whose rows 0..3 are the four partial rows; then the chain's typecast (RNE in the dest) and bf16 silu.  The two collectives stay (``all_gather`` of the stats, ``all_gather`` of
the partials along a new dim 0 = the composite's own all_broadcast + concat bytes).  5 programs + 2 collectives; the
tolerance class is BITWISE: every op is the chain's op or its LLK sequence.  Rows = 1 (the decode step)."""

from __future__ import annotations

import torch

import ttnn

from .. import gr_read as gr
from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "final_mixer"
TP_SIZE, TP_AXIS, BRANCHES = gr.TP_SIZE, gr.TP_AXIS, gr.BRANCHES
LOCAL_HIDDEN, FLAT_WIDTH, FLAT_TILES, HIDDEN_TILES = gr.LOCAL_HIDDEN, gr.FLAT_WIDTH, gr.FLAT_TILES, gr.HIDDEN_TILES
RANK = 320
RANK_TILES = RANK // fp.TILE  # 10
DOWN_SPILL = 8  # dram_sharded_matmul_configs(K=2560, N=320, num_cores=5): 16 K tiles per core -> in0_block_w 8 (gr_read's down too)
UP_SPILL = (
    5  # dram_sharded_matmul_configs(K=320, N=2560, num_cores=2): 5 K tiles per core -> in0_block_w 5 (gr_read's up: 6)
)
BF16, FP32, TILE_BF16, TILE_FP32 = gr.BF16, gr.FP32, gr.TILE_BF16, gr.TILE_FP32
LOWRANK_COMPUTE = fp.kernel_source(NAME, "lowrank_reduce_compute.cpp")
LOWRANK_READER = fp.kernel_source(NAME, "lowrank_reduce_reader.cpp")
LOWRANK_WRITER = fp.kernel_source(NAME, "lowrank_reduce_writer.cpp")
_ZERO_FP32: dict[int, object] = {}
_NORM_ROWS: dict[int, object] = {}


def _mixer_module():
    from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer

    return final_mixer


def zero_tile(mesh):
    """One fp32 zero tile per mesh (the stacked tiles' padding rows), uploaded once."""

    key = id(mesh)
    if key not in _ZERO_FP32:
        _ZERO_FP32[key] = ttnn.from_torch(
            torch.zeros(1, 1, fp.TILE, fp.TILE),
            dtype=FP32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return _ZERO_FP32[key]


def norm_scale_rows(mesh, norm_scale):
    """The mixer's fp32 gamma/4 ``[1, 4, 1, 640]`` (per device) repeated over the 32 tile rows -> ``[1, 4, 32, 640]``
    fp32, the row-repeated operand F2's ``normalize`` streams (the chain row-broadcasts the one-row tensor in its
    multiply; the values are the same).  Built once from the resident weight; a 1x1 mesh takes the plain upload."""

    hosts = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(norm_scale)]
    rows = [h.expand(-1, -1, fp.TILE, -1).contiguous() for h in hosts]
    if len(rows) == 1:
        return ttnn.from_torch(
            rows[0], dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=_mixer_module().MESH_SHAPE, dims=(None, 3))
    return ttnn.from_torch(
        torch.cat(rows, dim=3),
        dtype=FP32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def module_norm_scale_rows(module):
    key = id(module)
    if key not in _NORM_ROWS:
        _NORM_ROWS[key] = norm_scale_rows(module.mesh_device, module.weights.norm_scale)
    return _NORM_ROWS[key]


def down(normalized, weight):
    """Stage 2b: the flat row times the mixer's down weight ``[1, 1, 2560, 320]`` -> the fp32 partial
    ``[1, 1, rows, 320]`` (F2's DOWN kernel: fp32-dest accumulation in K order with the DRAM-sharded matmul's spill)."""

    rows = fp.rows_of(normalized)
    gr._expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "final mixer normalized row")
    gr._expect(weight, (1, 1, FLAT_WIDTH, RANK), BF16, "final mixer down")
    mesh = normalized.device()
    out = fp.allocate((1, 1, rows, RANK), FP32, ttnn.TILE_LAYOUT, mesh)
    n_tiles = gr._tiles_wide(weight)
    blk = 8
    work = fp.split_work(RANK_TILES, mesh)
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, FLAT_TILES, cores),
        fp.cb_descriptor(1, BF16, TILE_BF16, 2 * blk, cores),
        fp.cb_descriptor(2, FP32, TILE_FP32, 1, cores),
        fp.cb_descriptor(16, FP32, TILE_FP32, 1, cores),
    ]
    reader = gr._reader(
        cores,
        [(normalized, 0), (weight, 1)],
        [],
        [
            (
                w.core,
                (
                    [
                        gr._stream(normalized, 1, FLAT_TILES, 0, 1, 0, blk),
                        gr._stream(weight, 1, FLAT_TILES, w.start, n_tiles, 0, blk),
                    ],
                    [],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(gr.DOWN, cores, [FLAT_TILES, blk, 0, 1, 16, DOWN_SPILL, 2], fp32_dest=True)
    writer = gr._writer(cores, [(out, 16)], [(w.core, [(1, w.start, 1, 1)]) for w in work])
    return fp.run_program([normalized, weight, out], fp.program_descriptor([reader, compute, writer], cbs=cbs))


def low_rank(gathered_partials):
    """Stage 3a: the four gathered fp32 partial rows ``[4, 1, 1, 320]`` summed as the chain's composite sums them,
    typecast bf16, silu -> ``[1, 1, 1, 320]`` bf16 (one core)."""

    shape = tuple(int(v) for v in gathered_partials.shape)
    if (
        shape != (TP_SIZE, 1, 1, RANK)
        or gathered_partials.dtype != FP32
        or gathered_partials.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(f"final mixer gathered partials must be fp32 TILE [4, 1, 1, {RANK}], got {shape}")
    mesh = gathered_partials.device()
    zero = zero_tile(mesh)
    out = fp.allocate((1, 1, 1, RANK), BF16, ttnn.TILE_LAYOUT, mesh)
    core = ttnn.CoreCoord(0, 0)
    one = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    cbs = [
        fp.cb_descriptor(0, FP32, TILE_FP32, RANK_TILES, one),
        fp.cb_descriptor(1, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(2, FP32, TILE_FP32, RANK_TILES, one),
        fp.cb_descriptor(3, BF16, TILE_BF16, RANK_TILES, one),
        fp.cb_descriptor(16, BF16, TILE_BF16, RANK_TILES, one),
    ]
    reader = fp.reader_kernel(
        LOWRANK_READER,
        one,
        [RANK_TILES, TP_SIZE] + fp.accessor_args(gathered_partials) + fp.accessor_args(zero),
        [(core, [gathered_partials.buffer_address(), zero.buffer_address()])],
    )
    compute = fp.compute_kernel(LOWRANK_COMPUTE, one, [RANK_TILES], fp32_dest=True, unpack_to_dest_fp32=(0, 2))
    writer = fp.writer_kernel(
        LOWRANK_WRITER, one, [RANK_TILES] + fp.accessor_args(out), [(core, [out.buffer_address()])]
    )
    return fp.run_program([gathered_partials, zero, out], fp.program_descriptor([reader, compute, writer], cbs=cbs))


def gate(low_rank_row, normalized, up):
    """Stage 3b: sigmoid(low rank x up) times the normalized row, the four branches summed -> ``[1, 1, rows, 640]``
    (F2's GATE kernel with the mixer's K = 320 and its up matmul's 5-tile K blocks)."""

    rows = fp.rows_of(low_rank_row)
    gr._expect(low_rank_row, (1, 1, rows, RANK), BF16, "final mixer low-rank row")
    gr._expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "final mixer normalized row")
    gr._expect(up, (1, 1, RANK, FLAT_WIDTH), BF16, "final mixer up")
    mesh = low_rank_row.device()
    out = fp.allocate((1, 1, rows, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    n_tiles = gr._tiles_wide(up)
    work = fp.split_work(HIDDEN_TILES, mesh)
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, RANK_TILES, cores),
        fp.cb_descriptor(1, BF16, TILE_BF16, BRANCHES * RANK_TILES, cores),
        fp.cb_descriptor(2, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(3, BF16, TILE_BF16, 1, cores),
        fp.cb_descriptor(4, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(5, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(6, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(7, FP32, TILE_FP32, BRANCHES, cores),
        fp.cb_descriptor(16, BF16, TILE_BF16, 1, cores),
    ]
    reader = gr._reader(
        cores,
        [(low_rank_row, 0), (normalized, 2), (up, 1)],
        [(gr.CONST_ZERO, 3)],
        [
            (
                w.core,
                (
                    [
                        gr._stream(low_rank_row, 1, RANK_TILES, 0, 1, 0, RANK_TILES),
                        gr._stream(normalized, 1, BRANCHES, w.start, HIDDEN_TILES, 0, BRANCHES),
                        gr._stream(up, BRANCHES, RANK_TILES, w.start, n_tiles, HIDDEN_TILES, RANK_TILES),
                    ],
                    [0],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(
        gr.GATE,
        cores,
        [RANK_TILES, BRANCHES, 0, 1, 2, 3, 4, 5, 6, 16, UP_SPILL, 7],
        fp32_dest=True,
        unpack_to_dest_fp32=(2, 4, 5),
    )
    writer = gr._writer(cores, [(out, 16)], [(w.core, [(1, w.start, 1, 1)]) for w in work])
    return fp.run_program(
        [low_rank_row, normalized, up, out], fp.program_descriptor([reader, compute, writer], cbs=cbs)
    )


def final_mixer_fused(module, residual):
    """``Qwen38TTNNFinalMixer.__call__`` on the fused programs around the chain's two collectives."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    fm = _mixer_module()
    if tuple(int(v) for v in residual.shape) != fm.RESIDUAL_LOCAL_SHAPE or residual.dtype != BF16:
        raise ValueError(f"final mixer input must be TILE BF16 {fm.RESIDUAL_LOCAL_SHAPE}")
    module.mesh_contract.validate_tensor(residual, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    stats_local = gr.stats(residual)
    gr._topology(module, stats_local, 3)
    gathered_stats = ttnn.all_gather(stats_local, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(stats_local)
    module.mesh_contract.validate_tensor(gathered_stats, placement=TensorPlacement.REPLICATED)
    normalized = gr.normalize(residual, gathered_stats, module_norm_scale_rows(module))
    ttnn.deallocate(gathered_stats)
    gr._topology(module, normalized, 3)
    partial = down(normalized, module.weights.down)
    module.mesh_contract.mark_local_partial(
        partial, replicated_reference=module.weights.replicated_anchor, expected_shape=(1, 1, 1, RANK)
    )
    gathered_partials = ttnn.all_gather(partial, dim=0, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(partial)
    module.mesh_contract.validate_tensor(gathered_partials, placement=TensorPlacement.REPLICATED)
    low = low_rank(gathered_partials)
    low.update_tensor_topology(gathered_partials.tensor_topology())
    ttnn.deallocate(gathered_partials)
    block = gate(low, normalized, module.weights.up)
    ttnn.deallocate(low)
    ttnn.deallocate(normalized)
    gr._topology(module, block, 3)
    module.mesh_contract.validate_tensor(block, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    if tuple(int(v) for v in block.shape) != fm.OUTPUT_LOCAL_SHAPE or block.dtype != BF16:
        raise RuntimeError(f"fused final mixer produced {tuple(block.shape)} {block.dtype}")
    return block


def final_mixer_composed(module, residual):
    """The chain as written in ttnn/final_mixer.py (the class method, whatever the instance resolved to)."""

    return type(module).__call__(module, residual)


register(
    FusedKernel(
        name=NAME,
        replaces="Qwen38TTNNFinalMixer.__call__: rms_norm_pre/post_all_gather, gamma multiply, down linear, S2I, the fp32 "
        "all_reduce composite (7 programs), typecast, silu, up linear, S2I, sigmoid, gate multiply, fast_reduce_nc "
        "(21 programs once per step; the two collectives stay)",
        tolerance=BITWISE,
        fused=final_mixer_fused,
        composed=final_mixer_composed,
        gate=None,  # needs the TP4 collectives: component gate = the single-chip replica test (four slices, host gathers)
    )
)

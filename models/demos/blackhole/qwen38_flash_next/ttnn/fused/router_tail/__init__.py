# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``router_tail``: the MoE router tail as one program.  fp32 logits ``[1, 1, rows, 512]`` (TILE, DRAM) -> ROW_MAJOR
bf16 scores and uint16 indices ``[1, 1, rows, top_k]`` (DRAM); the chain ``softmax(numeric_stable) -> topk(k, largest,
sorted) -> sum -> div -> typecast(bf16) -> to_layout(ROW_MAJOR) x2 -> typecast(uint16)`` of ``Qwen38TTNNMoE._route``.

One core per 32-row tile.  The compute kernel issues each replaced op's instruction sequence on CBs of that op's data
format and unpack mode (``kernels/compute_router_tail.cpp``), so the values and the top-k tie order are the composed
chain's: tolerance class BITWISE.  ``rows`` 1..32 is one tile; 128 rows (the long chunk) is four tiles on four cores
(not proven on device yet; the model keeps the chunk on the composed chain).
"""

from __future__ import annotations

import os

import torch

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, GateSpec, register

NAME = "router_tail"
EXPERTS = 512
WIDTH_TILES = EXPERTS // fp.TILE
TOP_K = 10
STAGE_PAGES = 4  # 8 KB of bf16 tile pages for the writer's two 2 KB row stages plus alignment
KERNELS = {name: fp.kernel_source(NAME, f"{name}_router_tail.cpp") for name in ("reader", "compute", "writer")}

# Circular buffers: (name, index, dtype, pages).  The compute kernel unpacks cb_probs, cb_vals_t, cb_pad_reduce,
# cb_pad_div and cb_denom straight into the 32-bit dest (as topk, the accurate fp32 reduce and binary_ng unpack their
# fp32 operands); every other fp32 CB unpacks to the source registers (as softmax does).  Names are the kernels' named compile-time args.
CBS = (
    ("cb_in0", 0, ttnn.float32, WIDTH_TILES),
    ("cb_max_scaler", 1, ttnn.float32, 1),
    ("cb_sum_scaler", 2, ttnn.float32, 1),
    ("cb_norm_scaler", 3, ttnn.float32, 1),
    ("cb_max", 4, ttnn.float32, 1),
    ("cb_exps", 5, ttnn.float32, WIDTH_TILES),
    ("cb_recip", 6, ttnn.float32, 1),
    ("cb_probs", 7, ttnn.float32, WIDTH_TILES),
    ("cb_index", 8, ttnn.uint32, WIDTH_TILES),
    ("cb_vals_t", 9, ttnn.float32, 1),
    ("cb_idx_t", 10, ttnn.uint32, 1),
    ("cb_vals", 11, ttnn.float32, 1),  # [token, k] values; the reader zeroes the padding in place -> the sum's input
    ("cb_vals_ready", 12, ttnn.uint16, 1),  # token: the reader finished the zero fill
    ("cb_pad_div", 13, ttnn.float32, 1),  # [token, k] values for the division
    ("cb_sums", 14, ttnn.float32, 1),  # row sums; the reader broadcasts column 0 in place -> the division's rhs
    ("cb_sums_ready", 15, ttnn.uint16, 1),  # token: the reader finished the broadcast
    ("cb_scores", 16, ttnn.bfloat16, 1),
    ("cb_stage", 17, ttnn.bfloat16, STAGE_PAGES),  # writer staging: 2 x 32 rows x 64 B, 64-B aligned inside
)
TOKEN_CBS = ("cb_vals_ready", "cb_sums_ready")
TOKEN_PAGE_BYTES = 32
UNPACK_TO_DEST_FP32 = ("cb_probs", "cb_vals_t", "cb_vals", "cb_pad_div", "cb_sums")
CB_INDEX = {name: index for name, index, _dtype, _pages in CBS}
READER_ARGS = ("logits_addr", "index_addr", "tile_row", "rows_in_tile")
WRITER_ARGS = ("scores_addr", "indices_addr", "tile_row", "rows_in_tile")


def _compute_kernel_config(logits):
    return ttnn.init_device_compute_kernel_config(
        logits.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _rows_of(logits) -> int:
    shape = tuple(int(v) for v in logits.shape)
    rows = shape[-2] if len(shape) == 4 else 0
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != EXPERTS or not (1 <= rows <= fp.TILE or rows % fp.TILE == 0):
        raise ValueError(f"router tail logits must be [1, 1, rows (1..32 or n x 32), {EXPERTS}], got {list(shape)}")
    if logits.dtype != ttnn.float32 or logits.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"router tail logits must be fp32 TILE, got {logits.dtype} {logits.layout}")
    return rows


_INDEX_TEMPLATES: dict[int, object] = {}


def router_tail_prepare(mesh):
    """The constant index tiles the top-k sorts alongside the values, pre-transposed: tile ``w`` holds ``w*32+k`` in
    every column of row ``k`` (the transpose of the topk reader's ``w*32+c`` tile), as one uint32 TILE tensor
    ``[1, 1, 32, 512]`` in DRAM, allocated once per mesh (call before trace capture)."""

    key = id(mesh)
    if key not in _INDEX_TEMPLATES:
        k = torch.arange(fp.TILE, dtype=torch.int32).reshape(fp.TILE, 1)
        tile_base = (torch.arange(EXPERTS, dtype=torch.int32) // fp.TILE * fp.TILE).reshape(1, EXPERTS)
        template = (tile_base + k).reshape(1, 1, fp.TILE, EXPERTS)
        _INDEX_TEMPLATES[key] = ttnn.from_torch(
            template, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return _INDEX_TEMPLATES[key]


def router_tail_program(logits, index_template, scores, indices, *, rows: int, top_k: int) -> "ttnn.ProgramDescriptor":
    tile_rows = -(-rows // fp.TILE)
    cores = [ttnn.CoreCoord(0, y) for y in range(tile_rows)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[-1])])
    named = [(name, index) for name, index, _dtype, _pages in CBS] + [("Wt", WIDTH_TILES), ("top_k", top_k), ("stage_pages", STAGE_PAGES)]
    cbs = [
        fp.cb_descriptor(index, dtype, TOKEN_PAGE_BYTES if name in TOKEN_CBS else fp.TILE_BYTES[dtype], pages, grid)
        for name, index, dtype, pages in CBS
    ]

    def per_core(args_of):
        return [(core, args_of(tile_row, min(fp.TILE, rows - tile_row * fp.TILE))) for tile_row, core in enumerate(cores)]

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64  # one per circular buffer of the runtime (64)
    for name in UNPACK_TO_DEST_FP32:
        modes[CB_INDEX[name]] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = modes

    def kernel(source, compile_time_args, runtime_args, config):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[int(a) for a in compile_time_args],
            named_compile_time_args=named,
            runtime_args=runtime_args,
            config=config,
        )

    reader = kernel(
        KERNELS["reader"],
        fp.accessor_args(logits) + fp.accessor_args(index_template),
        per_core(lambda t, r: [logits.buffer_address(), index_template.buffer_address(), t, r]),
        ttnn.ReaderConfigDescriptor(),
    )
    writer = kernel(
        KERNELS["writer"],
        fp.accessor_args(scores) + fp.accessor_args(indices),
        per_core(lambda t, r: [scores.buffer_address(), indices.buffer_address(), t, r]),
        ttnn.WriterConfigDescriptor(),
    )
    compute = kernel(KERNELS["compute"], [], [], compute_config)
    compute.defines = _dev_defines()
    return fp.program_descriptor([reader, writer, compute], cbs=cbs)


def _dev_defines() -> list[tuple[str, str]]:
    """Timing knobs (dev only; both break the output): QWEN38_ROUTER_TAIL_TOPK_TILES=N sorts only the first N width
    tiles, QWEN38_ROUTER_TAIL_SOFTMAX_COPY_ONLY=1 skips the softmax math."""

    defines = []
    tiles = os.environ.get("QWEN38_ROUTER_TAIL_TOPK_TILES")
    if tiles:
        defines.append(("FRT_TOPK_TILES", str(int(tiles))))
    if os.environ.get("QWEN38_ROUTER_TAIL_SOFTMAX_COPY_ONLY") == "1":
        defines.append(("FRT_SOFTMAX_COPY_ONLY", "1"))
    return defines


def router_tail(logits, *, top_k: int = TOP_K, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """``(scores, indices)`` ROW_MAJOR bf16 / uint16 ``[1, 1, rows, top_k]``; ``compute_kernel_config`` is the chain's
    (HiFi4, fp32 dest) and is pinned inside the kernel."""

    rows = _rows_of(logits)
    if not 1 <= top_k <= 16:
        raise ValueError(f"router tail top_k must be 1..16 (one face of the sorted tile), got {top_k}")
    mesh = logits.device()
    scores = fp.allocate((1, 1, rows, top_k), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    indices = fp.allocate((1, 1, rows, top_k), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    index_template = router_tail_prepare(mesh)
    fp.run_program(
        [logits, index_template, scores, indices],
        router_tail_program(logits, index_template, scores, indices, rows=rows, top_k=top_k),
    )
    return scores, indices


def router_tail_composed(
    logits, *, top_k: int = TOP_K, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """The composed chain, op for op as ``Qwen38TTNNMoE._route`` issues it for one row tile."""

    _rows_of(logits)
    compute_config = _compute_kernel_config(logits) if compute_kernel_config is None else compute_kernel_config
    l1 = ttnn.L1_MEMORY_CONFIG
    probabilities = ttnn.softmax(
        logits, dim=-1, numeric_stable=True, memory_config=l1, compute_kernel_config=compute_config
    )
    scores, indices = ttnn.topk(probabilities, k=top_k, dim=-1, largest=True, sorted=True, memory_config=l1)
    ttnn.deallocate(probabilities)
    denominator = ttnn.sum(scores, dim=-1, keepdim=True, memory_config=l1, compute_kernel_config=compute_config)
    normalized_fp32 = ttnn.div(scores, denominator, memory_config=l1)
    ttnn.deallocate(scores)
    ttnn.deallocate(denominator)
    normalized = ttnn.typecast(normalized_fp32, ttnn.bfloat16, memory_config=l1)
    ttnn.deallocate(normalized_fp32)
    scores_rm = ttnn.to_layout(normalized, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    ttnn.deallocate(normalized)
    ttnn.deallocate(indices)
    if indices_rm.dtype != ttnn.uint16:
        converted = ttnn.typecast(indices_rm, ttnn.uint16, memory_config=memory_config)
        ttnn.deallocate(indices_rm)
        indices_rm = converted
    return scores_rm, indices_rm


def routing_table(result) -> torch.Tensor:
    """``[rows, 2 * top_k]`` fp32: the bf16 scores widened, then the indices; bitwise-comparable as fp32 bits."""

    scores, indices = result
    k = int(scores.shape[-1])
    return torch.cat(
        [ttnn.to_torch(scores).reshape(-1, k).float(), ttnn.to_torch(indices).reshape(-1, k).to(torch.int64).float()],
        dim=-1,
    )


def _gate_inputs(mesh, capture, positions, layer):
    index = [capture["positions"].index(p) for p in positions]
    logits = capture["router_logits"][index, layer].float().reshape(1, 1, len(positions), EXPERTS)
    return {
        "logits": ttnn.from_torch(
            logits, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    }


def _gate_reference(oracle, positions, layer):
    index = [oracle["positions"].index(p) for p in positions]
    scores = oracle["router_scores"][index, layer].to(torch.bfloat16).float()
    indices = oracle["router_indices"][index, layer].to(torch.int64).float()
    return torch.cat([scores, indices], dim=-1)


register(
    FusedKernel(
        name=NAME,
        replaces="softmax, topk, fill padding, sum, div, typecast bf16, to_layout x2, typecast uint16 (the 12-program router tail per layer)",
        tolerance=BITWISE,
        fused=router_tail,
        composed=router_tail_composed,
        gate=GateSpec(inputs=_gate_inputs, output=routing_table, reference=_gate_reference, layers=tuple(range(48))),
    )
)

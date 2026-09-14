# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused router tail without a device: its registry entry and switch, the CB / argument contract between the
Python side and the three kernels, the exactness pins in the compute kernel (the replaced ops' instruction sequences),
the writer's tile-face addressing against a torch model of the tile layout, and the model's resolve site."""

from __future__ import annotations

import inspect
import re

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt

SOURCES = {name: (fp.REPO_ROOT / path).read_text() for name, path in rt.KERNELS.items()}


def test_registered_bitwise_with_a_gate():
    entry = fused.kernel("router_tail")
    assert entry.tolerance == fused.BITWISE
    assert entry.fused is rt.router_tail and entry.composed is rt.router_tail_composed
    assert entry.gate is not None and entry.gate.layers == tuple(range(48)) and entry.gate.topk is None
    assert entry.default_on and fused.resolve("router_tail", {}) is rt.router_tail  # the foundation test pins the set
    assert fused.resolve("router_tail", {fused.OFF_ENV: "router_tail"}) is rt.router_tail_composed
    assert fused.resolve("router_tail", {fused.ENV: "router_tail", fused.OFF_ENV: "all"}) is rt.router_tail_composed
    assert inspect.signature(rt.router_tail).parameters.keys() == inspect.signature(rt.router_tail_composed).parameters.keys()


def test_cb_table_is_consistent():
    indices = [index for _name, index, _dtype, _pages in rt.CBS]
    assert len(set(indices)) == len(indices) and max(indices) < 32
    assert set(rt.UNPACK_TO_DEST_FP32) <= set(rt.CB_INDEX)
    for name in rt.UNPACK_TO_DEST_FP32:
        assert dict((n, d) for n, _i, d, _p in rt.CBS)[name] == ttnn.float32
    assert rt.WIDTH_TILES == 16 and rt.EXPERTS == 512 and rt.TOP_K == 10


@pytest.mark.parametrize("kernel", ["reader", "compute", "writer"])
def test_named_compile_time_args_exist_on_the_python_side(kernel):
    names = set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', SOURCES[kernel]))
    assert names, kernel
    assert names <= set(rt.CB_INDEX) | {"Wt", "top_k", "stage_pages"}, names - set(rt.CB_INDEX)


def test_runtime_arg_layout_matches_the_python_side():
    reader, writer = SOURCES["reader"], SOURCES["writer"]
    assert [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d)\)", reader)] == list(range(len(rt.READER_ARGS)))
    assert [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d)\)", writer)] == list(range(len(rt.WRITER_ARGS)))
    assert "TensorAccessorArgs<0, 0>()" in reader and "next_compile_time_args_offset()" in reader
    assert "TensorAccessorArgs<0, 0>()" in writer and "next_compile_time_args_offset()" in writer
    source = inspect.getsource(rt.router_tail_program)
    assert "[logits.buffer_address(), index_template.buffer_address(), t, r]" in source
    assert "[scores.buffer_address(), indices.buffer_address(), t, r]" in source


def test_compute_kernel_pins_the_replaced_ops_instruction_sequences():
    compute = SOURCES["compute"]
    # softmax.cpp numeric stable: MAX reduce, sub bcast cols + precise exp, SUM reduce with recip, mul bcast cols
    assert "PoolType::MAX,\n        ReduceDim::REDUCE_ROW" in compute
    assert "exp_tile_init<false>()" in compute and "exp_tile<false>(wt8)" in compute
    assert "sub_tiles_bcast_cols(cb_in0, cb_max" in compute and "mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recip" in compute
    assert "recip_tile_init();\n            recip_tile(0);" in compute
    # topk.cpp single core: unstable network, largest, end phase 5, values DST 0/1, indices DST 2/3
    assert "topk_local_sort<false>(0, 0 /* largest */, 5 /* end_phase */)" in compute
    assert "transpose_tile(cb_probs, w, slot)" in compute and "copy_tile(cb_index, w, slot + 2)" in compute
    # reduce.cpp: ttnn.sum on fp32 takes the accurate SFPU path (input to dest); binary_ng SFPU div; typecast fp32 -> bf16
    assert "ReduceFp32Mode::Accurate" in compute and "cb_vals" in rt.UNPACK_TO_DEST_FP32 and "cb_vals,\n        cb_norm_scaler,\n        cb_sums" in compute
    assert "div_binary_tile(0, 1, 0)" in compute
    assert "typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0)" in compute
    assert "stable_sort" not in compute.replace("stable_sort false", "")


def _tile_faces(matrix: torch.Tensor) -> torch.Tensor:
    """A 32x32 matrix in tile memory order: faces 0..3 (rows 0-15 / 16-31 x cols 0-15 / 16-31), row-major inside."""

    faces = [matrix[r : r + 16, c : c + 16].reshape(-1) for r in (0, 16) for c in (0, 16)]
    return torch.cat(faces)


def test_writer_face_addressing_matches_the_tile_layout():
    writer = SOURCES["writer"]
    scores_tile = torch.arange(32 * 32).reshape(32, 32)  # [token, k]
    index_tile = torch.arange(32 * 32).reshape(32, 32) * 7  # [k, token]
    sc, idx = _tile_faces(scores_tile), _tile_faces(index_tile)
    assert "sc[(face_row * 2) * 256 + in_face * 16 + k]" in writer
    assert "idx[face_row * 256 + k * 16 + in_face]" in writer
    for row in range(32):
        face_row, in_face = row >> 4, row & 15
        for k in range(rt.TOP_K):
            assert sc[(face_row * 2) * 256 + in_face * 16 + k] == scores_tile[row, k]
            assert idx[face_row * 256 + k * 16 + in_face] == index_tile[k, row]


def test_reader_zero_fill_and_column_broadcast_match_the_tile_layout():
    reader = SOURCES["reader"]
    assert "tile[256 + i] = 0u" in reader and "tile[768 + i] = 0u" in reader  # faces 1 and 3
    assert "const uint32_t first_zero = row < rows_in_tile ? top_k : 0u;" in reader
    assert "const uint32_t base = (row >> 4) * 2 * 256 + (row & 15) * 16;" in reader
    tile = torch.arange(1, 32 * 32 + 1).reshape(32, 32)
    flat = _tile_faces(tile)
    for face in range(4):
        row0, col0 = (face >> 1) * 16, (face & 1) * 16
        for i in range(256):
            row, col = row0 + (i >> 4), col0 + (i & 15)
            assert flat[face * 256 + i] == tile[row, col]
    for row in range(32):
        left_face = (row >> 4) * 2
        assert flat[left_face * 256 + (row & 15) * 16] == tile[row, 0]
    assert torch.equal(flat[256:512].reshape(16, 16), tile[:16, 16:]) and torch.equal(flat[768:].reshape(16, 16), tile[16:, 16:])


def test_index_template_is_the_transposed_topk_index_tile():
    k = torch.arange(32).reshape(32, 1)
    template = (torch.arange(512) // 32 * 32).reshape(1, 512) + k  # what router_tail_prepare uploads
    for w in range(16):
        generated = torch.arange(512)[w * 32 : (w + 1) * 32].repeat(32, 1)  # generate_index_tile: [r, c] = w*32 + c
        assert torch.equal(template[:, w * 32 : (w + 1) * 32], generated.T)


def test_model_resolves_once_and_uses_it_for_one_tile_rows():
    source = (fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/moe.py").read_text()
    assert source.count('fused.resolve("router_tail") if self.row_contract.row_tiles == 1 else None') == 1
    route = source[source.index("    def _route(") : source.index("    def _routed_partial(")]
    assert "self._route_tail(logits, top_k=TOP_K, compute_kernel_config=self.compute_config)" in route
    assert route.index("if self._route_tail is not None:") < route.index("probabilities = ttnn.softmax(")

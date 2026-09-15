# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused final mixer without a device: its registry entry, the low-rank kernels' argument contracts, the reduce
form (the accurate fp32 SFPU fold over the stacked device rows = the chain's `ttnn::sum` over a transposed-into-H
dim), the matmul
spills against the DRAM-sharded config rule, the reuse of F2's programs, and the hook pinned in ``final_mixer.py``."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import final_mixer as fm
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read as gr
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

HERE = Path(__file__).resolve().parents[1]
MIXER_SOURCE = (HERE / "ttnn" / "final_mixer.py").read_text()
COMPUTE = (fp.REPO_ROOT / fm.LOWRANK_COMPUTE).read_text()
READER = (fp.REPO_ROOT / fm.LOWRANK_READER).read_text()
WRITER = (fp.REPO_ROOT / fm.LOWRANK_WRITER).read_text()


def _in0_block_w(k: int, num_cores: int) -> int:
    """The rule of decode_matmul.dram_sharded_matmul_configs: the largest divisor <= 8 of the K tiles per core."""

    k_tiles_per_core = k // (32 * num_cores)
    return next(width for width in range(8, 0, -1) if k_tiles_per_core % width == 0)


def test_registered_bitwise_opt_in():
    entry = fused.kernel("final_mixer")
    assert entry.tolerance == fused.BITWISE and entry.gate is None
    assert entry.default_on is ("final_mixer" in fused.DEFAULT_ON)  # opt-in until its timing pin; the list decides
    assert entry.fused is fm.final_mixer_fused and entry.composed is fm.final_mixer_composed
    assert fused.resolve("final_mixer", {}) is fm.final_mixer_composed
    assert fused.resolve("final_mixer", {fused.ENV: "final_mixer"}) is fm.final_mixer_fused


def test_spills_follow_the_dram_sharded_matmul_rule():
    assert fm.DOWN_SPILL == _in0_block_w(2560, 5) == gr.DOWN_SPILL == 8
    assert fm.UP_SPILL == _in0_block_w(320, 2) == 5 and gr.UP_SPILL == _in0_block_w(384, 2) == 6
    assert fm.RANK_TILES == 10 and fm.RANK == 320


def test_low_rank_kernels_are_the_chains_reduce_form():
    assert "ReduceDim::REDUCE_COL," in COMPUTE and "ReduceFp32Mode::Accurate>(" in COMPUTE  # the chain's SFPU fp32 fold
    assert "compute_kernel_lib::ReduceInputBlockShape::of(1, T, 1)," in COMPUTE
    assert '#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"' in COMPUTE
    assert "reduce_tile<" not in COMPUTE  # never the FPU reduce: it truncates fp32 to tf32 on the way into SrcA/SrcB
    assert "typecast_tile<fp32, bf16>(0);" in COMPUTE and "silu_tile<false>(0);" in COMPUTE
    assert COMPUTE.index("ReduceFp32Mode::Accurate>(") < COMPUTE.index("typecast_tile<") < COMPUTE.index("silu_tile<")
    assert "ckernel::ReduceDim::REDUCE_COL>(1.0f);" in READER  # the chain's fp32 scaler tile
    assert "noc.async_read_barrier();  // the zero copies land before the rows" in READER
    assert ".page_id = d * T + t, .offset_bytes = 0}" in READER and ".offset_bytes = FACE_BYTES}" in READER
    for source, n_args, tensors in ((COMPUTE, 1, 0), (READER, 2, 2), (WRITER, 1, 1)):
        used = sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", source)})
        assert used == list(range(n_args)), used
        assert source.count("TensorAccessorArgs<") == tensors
    assert sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", READER)}) == [0, 1]
    assert sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", WRITER)}) == [0]


def test_python_side_matches_the_kernel_cbs():
    source = inspect.getsource(fm.low_rank)
    assert (
        "unpack_to_dest_fp32=(0, 2)" in source
    )  # the stacked tiles (the SFPU fold reads the dest) and the sum re-read
    assert "[RANK_TILES, TP_SIZE] + fp.accessor_args(gathered_partials) + fp.accessor_args(zero)" in source
    for index, dtype in ((0, "FP32"), (1, "FP32"), (2, "FP32"), (3, "BF16"), (16, "BF16")):
        assert f"fp.cb_descriptor({index}, {dtype}," in source, index
    assert "gr.DOWN" in inspect.getsource(fm.down) and "gr.GATE" in inspect.getsource(fm.gate)
    assert "[FLAT_TILES, blk, 0, 1, 16, DOWN_SPILL, 2]" in inspect.getsource(fm.down)
    assert "[RANK_TILES, BRANCHES, 0, 1, 2, 3, 4, 5, 6, 16, UP_SPILL, 7]" in inspect.getsource(fm.gate)
    fused_source = inspect.getsource(fm.final_mixer_fused)
    assert fused_source.count("ttnn.all_gather(") == 2 and "gr.stats(residual)" in fused_source
    assert "gr.normalize(residual, gathered_stats, module_norm_scale_rows(module))" in fused_source


def test_hook_is_pinned_in_the_mixer():
    assert (
        "    _fused_forward = None  # QWEN38_FUSED=final_mixer binds ttnn/fused/final_mixer per instance"
        in MIXER_SOURCE
    )
    assert 'if fused_kernels.enabled("final_mixer"):' in MIXER_SOURCE
    assert 'self._fused_forward = functools.partial(fused_kernels.kernel("final_mixer").fused, self)' in MIXER_SOURCE
    call = MIXER_SOURCE[MIXER_SOURCE.index("    def __call__(self, residual):") :]
    branch = "        if self._fused_forward is not None:\n            return self._fused_forward(residual)\n"
    assert branch in call and call.index(branch) < call.index("normalized_ws = self._normalize(residual)")


def test_manifest_lists_the_files():
    manifest = json.loads((HERE / "tools" / "release" / "manifest.json").read_text())["public"]
    for path in (
        "tests/test_fused_final_mixer_static.py",
        "ttnn/fused/final_mixer/__init__.py",
        "ttnn/fused/final_mixer/kernels/lowrank_reduce_compute.cpp",
        "ttnn/fused/final_mixer/kernels/lowrank_reduce_reader.cpp",
        "ttnn/fused/final_mixer/kernels/lowrank_reduce_writer.cpp",
    ):
        assert path in manifest, path

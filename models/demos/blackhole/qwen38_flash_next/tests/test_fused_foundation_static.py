# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused-kernel foundation without a device: the registry and its switch, the rows contract, the work split, the
kernel-source convention and the writer kernel's argument contract."""

import re
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import registry
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.untilize_rows import untilize_rows, untilize_rows_composed

NAME = "untilize_rows"


def test_registry_default_is_the_composed_chain():
    assert NAME in fused.kernels()
    assert fused.kernel(NAME).tolerance == fused.BITWISE
    assert fused.enabled_names({}) == frozenset()
    assert fused.resolve(NAME, {}) is untilize_rows_composed
    assert fused.enabled(NAME, {}) is False


@pytest.mark.parametrize("value", [NAME, f" {NAME} ,", f"{NAME},{NAME}", "all"])
def test_registry_switch_on(value):
    assert fused.enabled(NAME, {fused.ENV: value}) is True
    assert fused.resolve(NAME, {fused.ENV: value}) is untilize_rows


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="unregistered fused kernels \\['nope_kernel'\\]"):
        fused.enabled_names({fused.ENV: f"{NAME},nope_kernel"})
    with pytest.raises(KeyError, match="no fused kernel 'nope'"):
        fused.kernel("nope")


def test_registry_validates_entries():
    with pytest.raises(ValueError, match="registered twice"):
        registry.register(fused.kernel(NAME))
    with pytest.raises(ValueError, match="tolerance must be one of"):
        registry.FusedKernel("x_kernel", "y", "loose", untilize_rows, untilize_rows_composed)
    with pytest.raises(ValueError, match="name must match"):
        registry.FusedKernel("Router-Tail", "y", registry.BITWISE, untilize_rows, untilize_rows_composed)


def _tensor(shape, padded):
    return SimpleNamespace(shape=shape, padded_shape=padded)


@pytest.mark.parametrize("rows", [1, 5, 32])
def test_rows_contract_accepts_one_row_tile(rows):
    assert fp.rows_of(_tensor((1, 1, rows, 512), (1, 1, 32, 512))) == rows
    assert fp.tile_width_of(_tensor((1, 1, rows, 512), (1, 1, 32, 512))) == 512


@pytest.mark.parametrize(
    "shape, padded",
    [
        ((1, 1, 33, 512), (1, 1, 64, 512)),
        ((1, 1, 0, 512), (1, 1, 32, 512)),
        ((1, 4, 1, 640), (1, 4, 32, 640)),
        ((1, 32, 512), (1, 32, 512)),
    ],
)
def test_rows_contract_rejects_other_shapes(shape, padded):
    with pytest.raises(ValueError, match="one row tile"):
        fp.rows_of(_tensor(shape, padded))


def test_tile_width_rejects_partial_tiles():
    with pytest.raises(ValueError, match="whole tiles"):
        fp.tile_width_of(_tensor((1, 1, 1, 10), (1, 1, 32, 32)))


def test_split_work_covers_every_unit_once_in_linear_core_order():
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
    work = fp.split_work(16, mesh)
    assert [w.count for w in work] == [1] * 16
    assert [(w.core.x, w.core.y) for w in work[:11]] == [(0, y) for y in range(10)] + [(1, 0)]
    work = fp.split_work(250, mesh)
    assert len(work) == 110 and sum(w.count for w in work) == 250
    assert [w.start for w in work] == [sum(v.count for v in work[:i]) for i in range(len(work))]
    assert {w.count for w in work} == {2, 3}
    with pytest.raises(ValueError, match="nothing to split"):
        fp.split_work(0, mesh)


def test_kernel_source_convention():
    assert fp.kernel_source(NAME, "writer_rows.cpp") == f"{fp.KERNEL_ROOT}/{NAME}/kernels/writer_rows.cpp"
    assert (fp.REPO_ROOT / fp.KERNEL_ROOT / "program.py").is_file()
    with pytest.raises(FileNotFoundError):
        fp.kernel_source(NAME, "missing.cpp")


def test_writer_kernel_argument_contract_matches_the_python_side():
    source = (fp.REPO_ROOT / fp.kernel_source(NAME, "writer_rows.cpp")).read_text()
    assert "get_compile_time_arg_val(0)" in source and "elem_bytes" in source
    assert "get_compile_time_arg_val(1)" in source and "rows" in source
    assert "TensorAccessorArgs<2>()" in source
    assert re.search(
        r"get_arg_val<uint32_t>\(0\).*\n.*get_arg_val<uint32_t>\(1\).*\n.*get_arg_val<uint32_t>\(2\)", source
    )


def test_byte_tables_are_consistent():
    for dtype, elem in fp.ELEMENT_BYTES.items():
        assert fp.TILE_BYTES[dtype] == elem * fp.TILE * fp.TILE
    assert fp.TILE == 32 and fp.FACE == 16 and fp.ROWS_MAX == 32
    assert ttnn.TILE_SIZE == fp.TILE

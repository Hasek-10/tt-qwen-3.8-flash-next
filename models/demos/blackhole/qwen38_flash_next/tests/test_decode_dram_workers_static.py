# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The decode linears' DRAM readers per bank without a device: the default and its switch, the qualified table, the
bank layouts and the cache-name tag of the widened GDN input weight, the program configs (one tile row per call for
decode, the lanes and the MTP verify rows), the builder identity and the MTP build."""

import inspect
import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import CHECKPOINT_FILE_MANIFEST_SHA256, INDEX_SHA256
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256
from models.demos.blackhole.qwen38_flash_next.ttnn import builder as builder_module
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2 as mtp_v2_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    PINNED_CHECKPOINT_REVISION,
    PINNED_TENSOR_MANIFEST_SHA256,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import EXPECTED_LAYER_PATTERN

BANKS = 8
TILE = ttnn.TILE_SIZE
GDN_IN = (gdn_module.HIDDEN_SIZE, gdn_module.PROJECTION_WIDTH_PER_DEVICE)


def _mesh(banks: int = BANKS):
    return SimpleNamespace(dram_grid_size=lambda: ttnn.CoreCoord(banks, 1))


def test_two_readers_are_the_default_and_the_switch_admits_one_or_two() -> None:
    assert dm.WORKERS_ENV == "QWEN38_DRAM_WORKERS" and dm.DEFAULT_WORKERS_PER_DRAM_BANK == 2
    assert dm.default_decode_dram_workers({}) == 2
    assert dm.default_decode_dram_workers({dm.WORKERS_ENV: "1"}) == 1
    assert dm.default_decode_dram_workers({dm.WORKERS_ENV: " 2 "}) == 2
    for bad in ("0", "3", "two", "1.0", "1,2"):
        with pytest.raises(ValueError, match=dm.WORKERS_ENV):
            dm.default_decode_dram_workers({dm.WORKERS_ENV: bad})
    for bad in (True, 0, 3, "2", 2.0, None):
        with pytest.raises(ValueError):
            dm.validate_decode_dram_workers(bad)
    assert dm.validate_decode_dram_workers(1) == 1 and dm.validate_decode_dram_workers(2) == 2


class _Core:
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x, self.y = x, y


# A QuietBox 2 (2x p300c): three dies serve their banks from worker column 6, the fourth from column 5.
QB2_COLUMN_6 = ((0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4))
QB2_COLUMN_5 = ((0, 9), (0, 0), (0, 7), (0, 3), (5, 9), (5, 1), (5, 6), (5, 4))


def _qualify(per_die, requested, monkeypatch):
    mesh = SimpleNamespace(shape=(1, len(per_die)), dram_grid_size=lambda: ttnn.CoreCoord(BANKS, 1))
    calls = []

    def assignment(device, noc, coordinate):
        calls.append((noc, tuple(coordinate)))
        return [_Core(x, y) for x, y in per_die[tuple(coordinate)[1]]]

    monkeypatch.setattr(
        dm.ttnn.device, "get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate", assignment
    )
    return dm.qualify_decode_dram_workers(mesh, requested), calls


def test_two_readers_need_one_bank_to_worker_assignment_on_every_device(monkeypatch) -> None:
    (workers, reason), calls = _qualify([QB2_COLUMN_6] * 4, 2, monkeypatch)
    assert (workers, reason) == (2, None)
    assert [coordinate for _, coordinate in calls] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert {noc for noc, _ in calls} == {ttnn.NOC.RISCV_0_default}  # the matmul's in1 reader NOC
    (workers, reason), _ = _qualify([QB2_COLUMN_6, QB2_COLUMN_5, QB2_COLUMN_6, QB2_COLUMN_6], 2, monkeypatch)
    assert workers == 1 and "mesh coordinates [(0, 1)]" in reason and "differ from (0, 0)" in reason
    (workers, reason), _ = _qualify([QB2_COLUMN_5, QB2_COLUMN_6, QB2_COLUMN_6, QB2_COLUMN_6], 2, monkeypatch)
    assert workers == 1 and "[(0, 1), (0, 2), (0, 3)]" in reason
    (workers, reason), calls = _qualify([QB2_COLUMN_6, QB2_COLUMN_5, QB2_COLUMN_6, QB2_COLUMN_6], 1, monkeypatch)
    assert (workers, reason, calls) == (1, None, [])  # one reader asks the mesh nothing
    with pytest.raises(ValueError):
        dm.qualify_decode_dram_workers(SimpleNamespace(shape=(1, 4)), 3)


def test_the_builder_resolves_the_reader_count_against_the_live_mesh() -> None:
    source = inspect.getsource(builder_module.Qwen38TTNNBuilder.__init__)
    resolve = source.index("qualify_decode_dram_workers(")
    assert source.index("qualify_live_bf4_ring(mesh_device)") < resolve < source.index("live_identity = Qwen38LiveBuildIdentity(")
    assert "self.decode_dram_workers_fallback = decode_dram_workers_fallback" in source


def test_the_two_reader_table_names_the_screened_projections() -> None:
    assert dm.TWO_WORKER_PROJECTIONS == {
        (2560, 4160): 8,
        (1536, 2560): 16,
        (2560, 3072): 8,
        (2560, 8192): 40,
        (2560, 7040): 40,
    }
    assert GDN_IN == (2560, 4160) and (gdn_module.VALUE_WIDTH_PER_DEVICE, gdn_module.HIDDEN_SIZE) == (1536, 2560)
    assert (qsa_module.HIDDEN_SIZE, 2 * qsa_module.LOCAL_QUERY_WIDTH) == (2560, 3072)
    assert (qsa_module.LOCAL_QUERY_WIDTH, qsa_module.HIDDEN_SIZE) == (1536, 2560)
    assert set(embedding_module.LM_HEAD_CHUNK_COLUMNS) == {8192, 7040}
    # the K/V and index projections keep one reader: two are refused for their shapes and off the eight-bank grid
    for k, n in ((2560, qsa_module.HEAD_DIM), (2560, qsa_module.INDEX_HEAD_DIM)):
        with pytest.raises(ValueError, match="not qualified"):
            dm.bank_tiles(_mesh(), k, n, 2)
    with pytest.raises(ValueError, match="not qualified"):
        dm.bank_tiles(_mesh(7), *GDN_IN, 2)
    assert dm.bank_tiles(_mesh(7), *GDN_IN) == math.ceil(4160 / (TILE * 7))


def test_bank_layouts_pad_to_whole_tiles_per_reader_and_only_the_gdn_input_widens() -> None:
    mesh = _mesh()
    for (k, n), _cores in dm.TWO_WORKER_PROJECTIONS.items():
        one, two = dm.bank_tiles(mesh, k, n), dm.bank_tiles(mesh, k, n, 2)
        assert one == math.ceil(n / (TILE * BANKS)) and two % 2 == 0 and two >= one and two * BANKS * TILE >= n
        config = dm.dram_sharded_weight_memory_config(mesh, k, n, num_workers_per_dram_bank=2)
        assert tuple(config.shard_spec.shape) == (k, two * TILE)
    assert (dm.bank_tiles(mesh, *GDN_IN), dm.bank_tiles(mesh, *GDN_IN, 2)) == (17, 18)
    assert dm.weight_layout_tag(mesh, *GDN_IN, num_workers_per_dram_bank=2) == "_bank576"
    assert dm.weight_layout_tag(mesh, *GDN_IN, num_workers_per_dram_bank=1) == ""
    for k, n in ((1536, 2560), (2560, 3072), (2560, 8192), (2560, 7040)):
        assert dm.bank_tiles(mesh, k, n) == dm.bank_tiles(mesh, k, n, 2)
        assert dm.weight_layout_tag(mesh, k, n, num_workers_per_dram_bank=2) == ""
        assert dm.dram_sharded_weight_memory_config(
            mesh, k, n, num_workers_per_dram_bank=2
        ) == dm.dram_sharded_weight_memory_config(mesh, k, n)
    # the wider GDN input shard's cost per card: the GDN layers x 8 banks x 32 columns x 2560 rows x 2 bytes (bf16)
    gdn_layers = EXPECTED_LAYER_PATTERN.count("linear_attention")
    assert gdn_layers * BANKS * (576 - 544) * 2560 * 2 == 47_185_920


def test_program_configs_keep_one_tile_row_and_the_qualified_storage_grids() -> None:
    mesh = _mesh()
    for (k, n), cores in dm.TWO_WORKER_PROJECTIONS.items():
        act1, one = dm.dram_sharded_matmul_configs(mesh, k, n, num_cores=cores)
        act2, two = dm.dram_sharded_matmul_configs(mesh, k, n, num_cores=cores, num_workers_per_dram_bank=2)
        assert one.per_core_M == two.per_core_M == 1  # one tile row per call: decode, the lanes, the MTP verify rows
        assert one.in0_block_w == two.in0_block_w and one.fused_activation is None and two.fused_activation is None
        assert one.num_workers_per_dram_bank == 1 and two.num_workers_per_dram_bank == 2
        assert one.per_core_N == math.ceil(n / (TILE * cores)) and two.per_core_N == dm.bank_tiles(mesh, k, n, 2)
        assert act1 == act2  # the activation shard is the storage grid's, unchanged
    with pytest.raises(ValueError, match="storage cores"):
        dm.dram_sharded_matmul_configs(mesh, *GDN_IN, num_cores=16, num_workers_per_dram_bank=2)
    with pytest.raises(ValueError, match="not qualified"):
        dm.dram_sharded_matmul_configs(mesh, 2560, qsa_module.HEAD_DIM, num_cores=8, num_workers_per_dram_bank=2)


def test_the_modules_thread_the_reader_count_from_the_builder() -> None:
    gdn_init = inspect.getsource(gdn_module.Qwen38TTNNGDN.__init__)
    assert "workers = validate_decode_dram_workers(weights.decode_dram_workers_per_bank)" in gdn_init
    assert gdn_init.count("num_workers_per_dram_bank=workers") == 4  # two weight checks, two program configs
    loader = inspect.getsource(gdn_module.Qwen38TTNNGDNWeights.from_checkpoint)
    assert "layout_suffix = weight_layout_tag(" in loader
    assert 'f"qkvzab_tile_aligned_ab_dram_sharded{layout_suffix}.{dtype_tag}"' in loader
    assert "decode_dram_workers_per_bank=decode_dram_workers_per_bank," in loader
    qsa_init = inspect.getsource(qsa_module.Qwen38TTNNQSA.__init__)
    assert qsa_init.count("num_workers_per_dram_bank=workers") == 4
    assert "dram_sharded_matmul_configs(mesh_device, HIDDEN_SIZE, HEAD_DIM, num_cores=8)" in qsa_init
    head_init = inspect.getsource(embedding_module.Qwen38TTNNLMHead.__init__)
    assert head_init.count("num_workers_per_dram_bank=workers") == 2
    builder_source = inspect.getsource(builder_module.Qwen38TTNNBuilder)
    # the GDN weights, the QSA, the model I/O and the MTP layer's QSA all take the builder's count
    assert builder_source.count("decode_dram_workers_per_bank=self.decode_dram_workers_per_bank") == 4
    assert "decode_dram_workers_per_bank=self.decode_dram_workers_per_bank" in inspect.getsource(
        builder_module.Qwen38TTNNBuilder._build_mtp_components
    )
    assert "MTP is not qualified" not in builder_source
    parameters = inspect.signature(builder_module.Qwen38TTNNBuilder).parameters
    assert parameters["decode_dram_workers_per_bank"].default is None  # resolved by default_decode_dram_workers
    assert "default_decode_dram_workers()" in inspect.getsource(builder_module.Qwen38TTNNBuilder.__init__)


def test_the_build_identity_keeps_the_one_reader_key() -> None:
    provenance = builder_module.Qwen38BuildProvenance(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_index_sha256=INDEX_SHA256,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=PINNED_TENSOR_MANIFEST_SHA256,
        tt_metal_sha="1" * 40,
        ttnn_runtime_sha256="2" * 64,
    )
    common = dict(
        provenance=provenance,
        mesh_shape=(1, 4),
        physical_ids=(10, 11, 12, 13),
        collective_topology="Ring",
        dram_bank_ring_order=(6, 5, 4, 3, 2, 1, 0),
        ring_size=7,
    )
    legacy = builder_module.Qwen38LiveBuildIdentity(**common)
    one = builder_module.Qwen38LiveBuildIdentity(**common, decode_dram_workers_per_bank=1)
    two = builder_module.Qwen38LiveBuildIdentity(**common, decode_dram_workers_per_bank=2)
    # the one-reader key is the key of the identity before the field existed (the caches built under it stay valid)
    before_the_field = {k: v for k, v in asdict(legacy).items() if k != "decode_dram_workers_per_bank"}
    assert legacy.decode_dram_workers_per_bank == 1 and legacy.key == one.key == builder_module._payload_key(
        before_the_field
    )
    assert two.key != one.key and len(two.key) == 64 and two.key == builder_module._identity_key(two)
    with pytest.raises(ValueError):
        builder_module.Qwen38LiveBuildIdentity(**common, decode_dram_workers_per_bank=3)


def test_the_mtp_verify_rows_run_the_same_one_tile_row_linears() -> None:
    """The (k+1)-row verify body and the draft rows (rows <= 32, one tile) run the decode linears' program configs
    on the gathered shard: the GDN rows projection, the QSA rows linears and the LM head are the one-tile-row calls
    above, so two readers per bank need no separate MTP form (gated at the model level by the --mtp acceptance)."""

    assert gdn_module.CHUNK_SIZE == qsa_module.CHUNK_ROWS == TILE == 32
    project = inspect.getsource(gdn_module.Qwen38TTNNGDN._project_rows)
    assert project.count("ttnn.linear(") == 2 and project.count("program_config=self.in_proj_program_config") == 2
    rows = inspect.getsource(qsa_module.Qwen38TTNNQSA._linear_rows)
    assert rows.count("ttnn.linear(") == 2 and rows.count("program_config=program_config") == 2
    head = inspect.getsource(embedding_module.Qwen38TTNNLMHead.__call__)
    assert head.count("ttnn.linear(") == 1 and "program_config=program_config" in head
    verify = inspect.getsource(mtp_v2_module)
    assert "layer.attention.forward_rows(" in verify and "logits = lm_head(head_rows)" in verify

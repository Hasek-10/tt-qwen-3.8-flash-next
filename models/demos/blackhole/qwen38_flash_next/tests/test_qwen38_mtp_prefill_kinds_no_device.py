# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MTP drafting with the fast prefill, no device: one MTP chunk extension per chunk kind (32-row, 128-row, slab),
the driver routing every chunk to its kind's extension and writing rows-sized token rows one position ahead, the
extension's base plumbing and its rows contract, and the lifted exclusivity."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tests.test_ttnn_prefill_slab_no_device import SLAB, _FakeModel
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp as mtp_input_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2

PAD = driver_module.CHUNK_PAD_TOKEN_ID


class _Extension:
    """Records what the driver asks of one kind's extension."""

    def __init__(self, rows: int) -> None:
        self.rows, self.resets, self.writes, self.finishes = rows, 0, [], []

    def reset_chunk(self) -> None:
        self.resets += 1

    def write_tokens(self, model, tokens) -> None:
        self.writes.append([int(token) for token in tokens])

    def finish_chunk(self, model, *, prefilled: int) -> None:
        self.finishes.append(prefilled)


@pytest.fixture
def driver(monkeypatch):
    fake = SimpleNamespace(
        synchronize_device=lambda mesh: None,
        record_event=lambda mesh, cq_id: object(),
        event_synchronize=lambda event: None,
        _ttnn_execute_trace=lambda mesh, trace_id, cq_id, blocking: None,
    )
    monkeypatch.setattr(driver_module, "ttnn", fake)
    return fake


def _prefill(model, extensions, *, kinds=("short", "long", "slab")):
    states = {rows: SimpleNamespace(rows=rows) for rows in (32, 128, SLAB)}
    return driver_module.Qwen38ChunkPrefill(
        model,
        object(),
        object(),
        states[32],
        None,
        forced_step=lambda token, context: context,
        long_chunk_state=states[128] if "long" in kinds else None,
        slab_state=states[SLAB] if "slab" in kinds else None,
        mtp=extensions.get(32),
        long_mtp=extensions.get(128) if "long" in kinds else None,
        slab_mtp=extensions.get(SLAB) if "slab" in kinds else None,
    )


@pytest.mark.parametrize("start", (0, 5))
@pytest.mark.parametrize("count", (40, 128 + 40, 2048 + 128 + 40, 2 * 2048 + 3 * 128 + 7))
def test_driver_routes_every_chunk_to_its_kinds_extension_and_writes_the_tokens_one_ahead(driver, start, count):
    model = _FakeModel()
    extensions = {rows: _Extension(rows) for rows in (32, 128, SLAB)}
    tokens = list(range(1000, 1000 + count))
    result = _prefill(model, extensions).run(tokens, start_position=start, ple_context=None, following_token=7)
    aligned = driver_module.alignment_steps(start, count)
    remaining = tokens[aligned:]
    eager = [call for call in model.calls if call[0] == "eager"]
    assert eager and all(mtp is extensions[rows] for _, rows, _, mtp in eager), "each chunk ran with its kind's extension"
    # The MTP tokens: the remaining tokens shifted by one, then the following token, in chunk-sized pieces (padded)
    # in the order the kinds run: slabs, then 128-row chunks, then 32-row chunks.
    writes = [write for rows in (SLAB, 128, 32) for write in extensions[rows].writes]
    assert [len(write) for write in writes] == [rows for _, rows, _, _ in eager]
    written = [token for write in writes for token in write]
    following = remaining[1:] + [7]
    assert written[: len(following)] == following and all(token == PAD for token in written[len(following) :])
    # Every kind present was reset once at the prefill's start; the hand-off ran on the 32-row extension alone.
    assert [extensions[rows].resets for rows in (32, 128, SLAB)] == [1, 1, 1]
    assert extensions[32].finishes == [start + count] and not extensions[128].finishes and not extensions[SLAB].finishes
    assert result.position == start + count


def test_driver_with_only_the_32_row_kind_behaves_as_before(driver) -> None:
    model = _FakeModel()
    extensions = {32: _Extension(32)}
    tokens = list(range(1000, 1100))
    result = _prefill(model, extensions, kinds=("short",)).run(tokens, start_position=0, ple_context=None, following_token=3)
    eager = [call for call in model.calls if call[0] == "eager"]
    assert [rows for _, rows, _, _ in eager] == [32] * 4 and all(mtp is extensions[32] for _, _, _, mtp in eager)
    assert [len(write) for write in extensions[32].writes] == [32] * 4 and extensions[32].finishes == [100]
    assert result.position == 100


def test_driver_requires_the_extension_of_every_kind_it_runs_with_mtp_and_refuses_orphans(driver) -> None:
    states = {rows: SimpleNamespace(rows=rows) for rows in (32, 128, SLAB)}
    build = lambda **kwargs: driver_module.Qwen38ChunkPrefill(  # noqa: E731
        _FakeModel(), object(), object(), states[32], None, forced_step=lambda t, c: c, **kwargs
    )
    build(long_chunk_state=states[128], mtp=_Extension(32), long_mtp=_Extension(128))
    build(long_chunk_state=states[128], slab_state=states[SLAB], mtp=_Extension(32), long_mtp=_Extension(128), slab_mtp=_Extension(SLAB))
    for kwargs in (
        dict(long_chunk_state=states[128], mtp=_Extension(32)),  # the long chunks without their extension
        dict(long_chunk_state=states[128], long_mtp=_Extension(128)),  # an extension without the MTP chain
        dict(long_chunk_state=states[128], slab_state=states[SLAB], mtp=_Extension(32), long_mtp=_Extension(128)),
        dict(mtp=_Extension(32), slab_mtp=_Extension(SLAB)),  # a slab extension without a slab
    ):
        with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
            build(**kwargs)


def test_extension_allocation_shares_the_base_histories_and_sizes_the_token_rows(monkeypatch) -> None:
    monkeypatch.setattr(mtp_v2, "_validate_verify_state", lambda model_, verify_: None)
    calls: list[tuple] = []
    layer = SimpleNamespace(
        allocate_chunk_state=lambda constants, *, base=None, local_combine_output=None: (
            calls.append(("allocate", constants.rows, base, local_combine_output)) or SimpleNamespace(rows=constants.rows)
        ),
        release_chunk_state=lambda state: calls.append(("release", state.rows)),
    )
    alignment = SimpleNamespace(layer=layer, generic_state=object())
    verify = SimpleNamespace(alignment=alignment)
    model = SimpleNamespace(model_io=SimpleNamespace(embedding=SimpleNamespace(upload_token_rows=lambda rows: ("rows", rows))))
    chunk = SimpleNamespace(rows=32, rows_constants=SimpleNamespace(rows=32), local_combine_output=None)
    long_chunk = SimpleNamespace(rows=128, rows_constants=SimpleNamespace(rows=128), local_combine_output="combine-128")
    slab = SimpleNamespace(rows=SLAB, rows_constants=SimpleNamespace(rows=SLAB), local_combine_output="combine-slab")
    base = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, chunk)
    long = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, long_chunk, base=base)
    wide = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, slab, base=base)
    assert (base.rows, long.rows, wide.rows) == (32, 128, SLAB)
    assert (base.token_row, long.token_row, wide.token_row) == (("rows", 32), ("rows", 128), ("rows", SLAB))
    assert calls == [
        ("allocate", 32, None, None),
        ("allocate", 128, base.layer_chunk_state, "combine-128"),
        ("allocate", SLAB, base.layer_chunk_state, "combine-slab"),
    ]
    with pytest.raises(ValueError):  # allow-pytest.raises: a wider kind needs the base
        mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, long_chunk)
    with pytest.raises(ValueError):  # allow-pytest.raises: the 32-row kind takes none
        mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, chunk, base=base)
    with pytest.raises(ValueError):  # allow-pytest.raises: the base is the 32-row one
        mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, slab, base=long)
    with pytest.raises(ValueError):  # allow-pytest.raises: rows tokens per write
        long.write_tokens(model, [1] * 32)
    with pytest.raises(RuntimeError):  # allow-pytest.raises: the hand-off is the 32-row extension's
        long.finish_chunk(model, prefilled=128)
    assert mtp_v2.Qwen38TTNNMTPChunkExtension(alignment, object(), object()).rows == 32  # the field's default


def test_the_lifted_contracts_and_the_extension_lifecycle_are_pinned() -> None:
    assert "rows" in inspect.signature(mtp_input_module.Qwen38TTNNMTPInput.rows).parameters
    assert inspect.signature(mtp_input_module.Qwen38TTNNMTPInput.rows).parameters["rows"].default == 32
    forward = inspect.getsource(model_module.Qwen38TTNNTextModel.forward_prefill_chunk_generic)
    assert "is the 32-row chunk's option" not in forward and 'getattr(mtp, "rows", CHUNK_ROWS) != chunk_state.rows' in forward
    driver_init = inspect.getsource(driver_module.Qwen38ChunkPrefill.__init__)
    assert "no long chunks with MTP drafting" not in driver_init and "long_mtp" in driver_init and "slab_mtp" in driver_init
    session_open = inspect.getsource(session_module.Qwen38TracedChain.open)
    assert "long chunks and MTP drafting are alternatives" not in session_open
    assert session_open.count("base=chunk_extension") == 2
    assert "mtp=slab_extension" in session_open and "mtp=long_chunk_extension" in session_open
    assert "mtp=None if chain_mtp is None else chain_mtp.long_chunk_extension" in session_open
    assert "mtp=None if chain_mtp is None else chain_mtp.slab_extension" in session_open
    close = inspect.getsource(session_module.Qwen38TracedChain.close)
    assert "reversed(self.mtp.chunk_extensions())" in close and "for arm in sorted(self.mtp.arms.values()" in close
    main = inspect.getsource(server_module.main) if hasattr(server_module, "main") else inspect.getsource(server_module)
    assert "--prefill-slab and --mtp are alternatives" not in main
    extension = session_module.Qwen38ChainMTP.chunk_extensions
    assert callable(extension)
    admission = session_module.mtp_capacity_admission(32768, chunk_kinds=3)
    assert admission["chunk_kinds"] == 3 and (
        admission["required_free_bytes_per_bank"] - session_module.mtp_capacity_admission(32768)["required_free_bytes_per_bank"]
        == 2 * session_module.MTP_PREFILL_EXTENSION_BYTES_PER_BANK_UPPER_BOUND
    )
    with pytest.raises(ValueError):  # allow-pytest.raises: three kinds at most
        session_module.mtp_capacity_admission(32768, chunk_kinds=4)

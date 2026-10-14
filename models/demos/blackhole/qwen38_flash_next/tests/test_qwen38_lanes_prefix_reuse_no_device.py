# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prompt reuse on the lanes (``--lanes-prefix-reuse``, ``--lanes-conversations``) without a device: the plan (resume,
restore, capture on a chunk boundary, fresh), the admission on the real session over a fake chain whose caches are
positional and whose recurrent state is the stream consumed since the reset (a restored, captured or resumed
admission imports exactly the state a fresh one does, the MTP rows included, read from the chain on device and
bounded by the position), the records (a different prompt or an image prompt drops the snapshot; checkpoints replace
their conversation's own, then the least recently used), the counters, the server's flags and their startup pins on
the scheduler's fake device."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import math
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tests import test_qwen38_lane_scheduler_no_device as scheduler_tests
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_lanes_session as lanes_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler import Qwen38LaneTicket
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_prefill_driver import (
    Qwen38PrefillResult,
    Qwen38PrefillTiming,
    alignment_steps,
)

ARGMAX = "argmax"  # the MTP row of a step whose next token the device resolves (the prompt's last token)
SYSTEM = list(range(10_000, 10_300))  # 300 ids every agent shares: a system prompt and the tools
GENERATION = [7, 8, 9]  # the generation prompt's ids


def _prompt(task: int, *, system=SYSTEM, length: int = 40) -> list[int]:
    return [*system, *range(20_000 + 100 * task, 20_000 + 100 * task + length), *GENERATION]


class FakePrefillChain:
    """The traced chain's prefill primitives over a fake model.  The caches are positional (the id and the MTP row
    written at each position; a reset leaves them), the recurrent state is the stream consumed since the reset, the
    row's next token is a hash of the three (so a stale cache entry, recurrent state or MTP row shows), and the
    snapshot copies the recurrent state and the position only, as the device's does."""

    allocated_context = 32_768
    chunk_trace_id = 1  # the chunked mode is available

    def __init__(self) -> None:
        self.mtp = session_module.Qwen38ChainMTP(
            drafts=4, anchor="off", components=None, verify=None, draft=None, step_inputs=None, chunk_extension=None
        )
        self.kv: dict[int, int] = {}
        self.mtp_rows: dict[int, tuple[int, object]] = {}
        self.pos = 0
        self.history: tuple[int, ...] = ()
        self.row: int | None = None
        self.mtp_next: object = ARGMAX
        self.snapshot: tuple[int, tuple[int, ...]] | None = None
        self.calls: list[tuple] = []
        self.state = self  # the generic state ``model.set_generic_position`` takes

    def image(self) -> tuple:
        """What the eviction copies: the caches below the position and the recurrent state."""

        positions = range(self.pos)
        return tuple(self.kv[p] for p in positions), self.history, tuple(self.mtp_rows[p] for p in positions)

    def _consume(self, token: int, mtp_next: object) -> None:
        self.kv[self.pos] = token
        self.mtp_rows[self.pos] = (token, mtp_next)
        self.history += (token,)
        self.pos += 1

    def _next(self) -> int:
        return 1000 + int.from_bytes(hashlib.sha256(repr(self.image()).encode()).digest()[:4], "little") % 5000

    # -- the 1-row primitives (the session's forced steps)
    def reset_and_seed(self, token_id: int) -> None:
        self.calls.append(("reset",))
        self.pos, self.history, self.row = 0, (), token_id

    def write_token_row(self, token_id: int) -> None:
        self.row = token_id

    def write_mtp_next_token(self, next_token_id: int | None) -> None:
        self.mtp_next = ARGMAX if next_token_id is None else next_token_id
        self.mtp.step_written = True

    def execute_head(self, residue: int) -> None:
        if not self.mtp.step_written:
            self.mtp_next = ARGMAX

    def refresh_ple_row(self, token_id: int, context):
        return (token_id, 0 if context is None else context[0])

    def execute_tail(self, residue: int) -> None:
        self._consume(self.row, self.mtp_next)
        self.mtp.step_written = False
        self.row = self._next()

    def read_token_row(self) -> int:
        return self.row

    def position(self) -> int:
        return self.pos

    def loop_guard(self):
        return contextlib.nullcontext()

    def record_event(self):
        return None

    @staticmethod
    def event_synchronize(event) -> None:
        pass

    # -- the chunk driver
    def chunk_prefill(
        self,
        token_ids,
        *,
        start_position,
        ple_context,
        forced_step,
        following_token,
        should_stop,
        between_chunks=None,
        event_rows=None,
        **vision,
    ):
        assert start_position == self.pos
        self.calls.append(("chunks", start_position, len(token_ids), following_token, bool(vision)))
        forced = alignment_steps(start_position, len(token_ids))
        context = ple_context
        for token in token_ids[:forced]:
            context = forced_step(token, context)
        ahead = [*token_ids[1:], following_token]
        for index in range(forced, len(token_ids)):
            self._consume(token_ids[index], ahead[index])
            context = self.refresh_ple_row(token_ids[index], context)
        rows = len(token_ids) - forced
        if between_chunks is not None:
            between_chunks(rows, rows)
        timing = Qwen38PrefillTiming(
            alignment_steps=forced,
            chunks=math.ceil(rows / 32),
            tail_rows=rows % 32,
            chunk_replay_ms=(),
            chunk_host_ms=(),
            verify_ms=None,
            handoff_ms=0.0,
            wall_ms=0.0,
        )
        return Qwen38PrefillResult(position=self.pos, ple_context=context, timing=timing)

    # -- the prompt snapshot: the recurrent state and the position, not the caches
    def capture_prompt_snapshot(self, position: int) -> None:
        assert position == self.pos
        self.calls.append(("capture", position))
        self.snapshot = (self.pos, self.history)

    def restore_prompt_snapshot(self) -> None:
        self.pos, self.history = self.snapshot
        self.mtp.step_written = False
        self.calls.append(("restore", self.pos))


class FakePager:
    def __init__(self, chain: FakePrefillChain) -> None:
        self.chain = chain
        self.imports: list[tuple] = []


class FakeCheckpoints:
    """The host checkpoints over the fake chain: the backbone store keeps the cached ids below the saved rows and the
    recurrent state, the alignment store the MTP rows below them (``ttnn/lanes.Qwen38TTNNStateCheckpoints``' split:
    the packs whole, the KV bounded)."""

    def __init__(self, chain: FakePrefillChain, part: str) -> None:
        self.chain, self.part, self.slots, self.calls = chain, part, {}, []

    def save(self, slot: int, rows: int) -> float:
        chain = self.chain
        cache = chain.kv if self.part == "backbone" else chain.mtp_rows
        self.slots[slot] = ({p: cache[p] for p in range(rows)}, chain.history)
        self.calls.append(("save", slot, rows))
        return 1.0

    def restore(self, slot: int, rows: int) -> float:
        cached, history = self.slots[slot]
        cache = self.chain.kv if self.part == "backbone" else self.chain.mtp_rows
        for p in range(rows):
            cache[p] = cached[p]
        if self.part == "backbone":
            self.chain.history = history
        self.calls.append(("restore", slot, rows))
        return 1.0


def _set_position(state: FakePrefillChain, position: int) -> None:
    state.pos = position


@pytest.fixture(autouse=True)
def _fake_device(monkeypatch):
    def import_lane_state(
        model, verify, lane, *, position, ple_context, backbone_slot, backbone_pager, alignment_slot, kv_rows, **_
    ):
        # From the chain's state on device (no host slot), the KV bounded by the position.
        assert backbone_slot is None and alignment_slot is None and kv_rows == position
        backbone_pager.imports.append((lane, position, ple_context, backbone_pager.chain.image()))

    monkeypatch.setattr(lanes_module.mtp_lanes, "import_lane_state", import_lane_state)
    monkeypatch.setattr(lanes_module.ttnn, "synchronize_device", lambda mesh: None)


def _lanes(*, reuse: bool = True, min_tokens: int = lanes_module.REUSE_MIN_TOKENS, conversations: int = 0):
    chain = FakePrefillChain()
    session = session_module.Qwen38ChatSession(chain, template=None)
    lanes = lanes_module.Qwen38LanesSession(lanes=2, drafts=4, marker=lambda label: None, conversations=conversations)
    model = SimpleNamespace(layers=(), set_generic_position=_set_position)
    lanes.session, lanes.chain, lanes.model, lanes.mesh = session, chain, model, None
    lanes.lane_chain = object()
    lanes.backbone_pager = lanes.alignment_pager = FakePager(chain)
    if conversations:
        lanes.backbone_checkpoints = FakeCheckpoints(chain, "backbone")
        lanes.alignment_checkpoints = FakeCheckpoints(chain, "alignment")
    lanes.prefix_reuse, lanes.reuse_min_tokens = reuse, min_tokens
    return lanes, chain


def _ticket(prompt: list[int], **fields) -> Qwen38LaneTicket:
    return Qwen38LaneTicket(
        request_id="r", prompt_ids=list(prompt), max_tokens=8, stop_ids=(), think_budget=None, **fields
    )


# -- the plan -----------------------------------------------------------------------------------------------------------


def _plan(*args, **kwargs) -> tuple:
    planned = lanes_module.reuse_plan(*args, **kwargs)
    return planned.kind, planned.start if planned.kind != "captured" else planned.capture


def test_the_plan_restores_the_key_captures_on_a_chunk_boundary_below_the_shared_ids_or_starts_fresh() -> None:
    plan, shared = _plan, lanes_module.shared_prefix
    assert shared([1, 2, 3], [1, 2, 4]) == 2 and shared([1, 2], [1, 2, 3]) == 2 and shared([], [1]) == 0
    ids = list(range(1000, 1300))
    assert plan(ids, (), None, min_tokens=128) == ("fresh", 0)
    # the shared ids' last multiple of 32 strictly below them: the snapshot's key (one id more) is shared too
    assert plan(ids, ids, None, min_tokens=128) == ("captured", 288)
    assert plan(ids, ids[:200] + [0], None, min_tokens=128) == ("captured", 192)
    assert plan(ids, ids[:129] + [0], None, min_tokens=128) == ("captured", 128)
    assert plan(ids, ids[:128] + [0], None, min_tokens=128) == ("fresh", 0)  # 96 < 128
    assert plan(ids, ids[:128] + [0], None, min_tokens=96) == ("captured", 96)
    prefix = lanes_module.Qwen38LanePrefix(tuple(ids[:257]), (1, 2))
    assert prefix.position == 256
    assert plan(ids, (), prefix, min_tokens=128) == ("restored", 256)
    assert plan(ids[:257], (), prefix, min_tokens=128) == ("restored", 256)  # the key alone: its last id forced
    assert plan(ids[:256], ids, prefix, min_tokens=128) == ("captured", 224)  # shorter than the key
    # the ids below the snapshot but not the one after it: the MTP row at 255 read another token, no restore
    other = ids[:256] + [0] + ids[257:]
    assert plan(other, ids, prefix, min_tokens=128) == ("captured", 224)


def test_the_plan_resumes_a_checkpoint_that_reaches_past_the_snapshot_by_the_margin() -> None:
    ids = list(range(1000, 2000))
    prefix = lanes_module.Qwen38LanePrefix(tuple(ids[:257]), None)
    near = lanes_module.Qwen38LaneCheckpoint(tuple(ids[:449]), None, 0, 1)  # 448: 192 past the snapshot
    far = lanes_module.Qwen38LaneCheckpoint(tuple(ids[:705]), (3, 4), 1, 2)  # 704
    other = lanes_module.Qwen38LaneCheckpoint(tuple([7] * 900), None, 2, 3)
    planned = lanes_module.reuse_plan(ids, (), prefix, min_tokens=128, checkpoints=(near, far, other), margin=256)
    assert (planned.kind, planned.start, planned.checkpoint) == ("resumed", 704, far)
    assert _plan(ids, (), prefix, min_tokens=128, checkpoints=(near, other), margin=256) == ("restored", 256)
    assert _plan(ids, (), None, min_tokens=128, checkpoints=(near,), margin=256) == ("resumed", 448)
    assert _plan(ids[:704], (), prefix, min_tokens=128, checkpoints=(far,), margin=256) == ("restored", 256)
    # without a snapshot or a checkpoint the shared ids with the chain's prompt still capture
    assert _plan(ids, ids, None, min_tokens=128, checkpoints=(other,)) == ("captured", 992)


@pytest.mark.parametrize(
    "length,start,floor,expected",
    (
        (343, 0, 64, 320),  # the last id 342: 22 rows after 320 make a chunk
        (330, 0, 64, 288),  # 9 rows after 320 would be forced 1-row steps: one chunk earlier
        (321, 0, 64, 288),  # the last id on the boundary: one chunk earlier, the key short of the generation prompt
        (337, 0, 64, 320),  # 16 ids after 320: the least the rule leaves
        (343, 320, 64, None),  # not past the admission's start
        (343, 288, 64, 320),
        (343, 0, 400, None),  # below the floor
        (40, 0, 32, None),
    ),
)
def test_a_checkpoint_sits_on_a_chunk_boundary_that_leaves_the_tail_a_chunk(length, start, floor, expected) -> None:
    assert lanes_module.checkpoint_position(length, start, floor=floor) == expected


# -- the admission on the session ----------------------------------------------------------------------------------------


def test_reused_admissions_import_the_state_a_fresh_admission_does_and_the_records_follow_the_chain() -> None:
    lanes, chain = _lanes()
    reference, _ = _lanes(reuse=False)
    other_system = list(range(30_000, 30_300))
    near = SYSTEM[:288] + [99_999] + _prompt(9)[289:]  # the snapshot's ids but another token after them
    prompts = [
        (_prompt(1), "fresh", 0),
        (_prompt(2), "captured", 288),  # shares the 300 system ids with the first: the snapshot at 288
        (_prompt(3), "restored", 288),
        (_prompt(2), "restored", 288),
        (_prompt(4, system=other_system), "fresh", 0),  # another prompt: its prefill rewrote the caches below 288
        (_prompt(1), "fresh", 0),  # the snapshot is gone and nothing is shared with the last prompt
        (_prompt(3), "captured", 288),
        (near, "captured", 256),
        (_prompt(5), "restored", 256),
        (SYSTEM[:257] + [5, 6], "restored", 256),  # a short tail after the key: teacher-forced from the snapshot
    ]
    for index, (prompt, plan, split) in enumerate(prompts):
        chain.calls.clear()
        admitted = lanes.admit(0, _ticket(prompt))
        expected = reference.admit(0, _ticket(prompt))
        record = admitted.prefill
        assert (record["reuse"], record["prefix_reused"] or record["prefix_captured"]) == (plan, split), index
        # the lane's state: the caches, the recurrent state, the MTP rows, the n-gram context, the next token
        assert lanes.backbone_pager.imports[-1] == reference.backbone_pager.imports[-1], index
        assert (admitted.pending, admitted.position, admitted.ple_context) == (
            expected.pending,
            expected.position,
            expected.ple_context,
        ), index
        assert lanes.resident == tuple(prompt)
        if plan == "restored":
            assert chain.calls[0] == ("restore", split) and ("reset",) not in chain.calls
            assert lanes.prefix.position == split
        elif plan == "captured":
            assert chain.calls[:3] == [("reset",), ("chunks", 0, split, prompt[split], False), ("capture", split)]
            assert lanes.prefix.key == tuple(prompt[: split + 1])
        else:
            assert chain.calls[0] == ("reset",) and lanes.prefix is None
    assert lanes.reuse == {
        "fresh": 3,
        "captured": 3,
        "restored": 4,
        "resumed": 0,
        "restored_tokens": 288 * 2 + 256 * 2,
        "resumed_tokens": 0,
        "checkpoints_saved": 0,
    }
    # without the reuse every admission is fresh, the records track the chain and nothing is counted
    assert reference.prefix is None and reference.resident == tuple(prompts[-1][0])
    assert set(reference.reuse.values()) == {0}
    assert reference.summary()["prefix_reuse"] is None
    summary = lanes.summary()["prefix_reuse"]
    assert summary["min_tokens"] == 128 and summary["snapshot_position"] == 256 and summary["restored"] == 4
    lanes.forget()
    assert (lanes.resident, lanes.prefix) == ((), None) and lanes.reuse["restored"] == 0


def test_the_prefill_record_counts_the_tokens_the_admission_forced_and_chunked() -> None:
    lanes, _ = _lanes()
    fresh = lanes.admit(0, _ticket(_prompt(1))).prefill
    rows = len(_prompt(1)) - 1
    assert fresh == {
        "mode": "chunked",
        "tokens": rows + 1,
        "forced_tokens": 1,
        "chunks": math.ceil(rows / 32),
        "long_chunks": 0,
        "slabs": 0,
        "reuse": "fresh",
        "prefix_reused": 0,
        "prefix_captured": 0,
        "checkpoint": 0,
    }
    captured = lanes.admit(0, _ticket(_prompt(2))).prefill
    assert captured["prefix_captured"] == 288 and captured["forced_tokens"] == 1
    assert captured["chunks"] == 288 // 32 + math.ceil((rows - 288) / 32)  # both prefills from a chunk boundary
    restored = lanes.admit(0, _ticket(_prompt(3))).prefill
    assert restored["prefix_reused"] == 288 and restored["chunks"] == math.ceil((rows - 288) / 32)
    short = lanes.admit(0, _ticket(SYSTEM[:289] + [5, 6])).prefill  # the key and two ids: 3 ids from the snapshot on
    assert (short["mode"], short["prefix_reused"], short["forced_tokens"], short["chunks"]) == (
        "teacher_forced",
        288,
        3,
        0,
    )


def _turn(history: list[int], seed: int, length: int) -> list[int]:
    """The next turn of a conversation: its rendered history, the reply and tool result, the generation prompt."""

    return [*history[: -len(GENERATION)], *range(40_000 + 1000 * seed, 40_000 + 1000 * seed + length), *GENERATION]


def test_conversations_resume_their_checkpoints_and_import_the_state_a_fresh_admission_does() -> None:
    """Agents on two slots: each turn saves a checkpoint on its way through the prompt and the conversation's next turn
    resumes it (the reply and tool result prefilled, nothing before), past the shared snapshot by the margin; a third
    conversation takes the least recently used slot, after which the evicted conversation falls back to the snapshot.
    Every admission imports exactly the state a fresh prefill of its prompt gives."""

    lanes, chain = _lanes(conversations=2)
    lanes.checkpoint_min_tokens, lanes.resume_margin = 64, 64
    reference, _ = _lanes(reuse=False)
    a1, b1, c1 = (_prompt(task, length=200) for task in (1, 2, 3))
    a2, b2 = _turn(a1, 1, 200), _turn(b1, 2, 230)
    a3 = _turn(a2, 3, 150)
    c2 = _turn(c1, 4, 120)
    a4 = _turn(a3, 5, 90)
    turns = [
        # prompt, plan, the position it starts from, the checkpoint it saves
        (a1, "fresh", 0, 480),
        (b1, "captured", 0, 480),  # the system ids shared with a1: the snapshot at 288, a checkpoint past it
        (a2, "resumed", 480, 672),  # a1's checkpoint reaches 192 past the snapshot: resumed, a new one saved
        (b2, "resumed", 480, 704),
        (c1, "restored", 288, 480),  # a third conversation: the least recently used slot (a's) is replaced
        (a3, "restored", 288, 832),  # a's checkpoint is gone: the snapshot, and a new checkpoint (b's slot now)
        (c2, "resumed", 480, 576),  # 14 ids after 608 would be 1-row steps: one chunk earlier
        (a4, "resumed", 832, 896),
    ]
    for index, (prompt, plan, start, saved) in enumerate(turns):
        chain.calls.clear()
        admitted = lanes.admit(0, _ticket(prompt, checkpoint=True))
        expected = reference.admit(0, _ticket(prompt))
        record = admitted.prefill
        assert (record["reuse"], record["prefix_reused"], record["checkpoint"]) == (plan, start, saved), index
        assert lanes.backbone_pager.imports[-1] == reference.backbone_pager.imports[-1], index
        assert (admitted.pending, admitted.position, admitted.ple_context) == (
            expected.pending,
            expected.position,
            expected.ple_context,
        ), index
        assert ("checkpoint" in admitted.seconds) == bool(saved), index
        assert ("resume" in admitted.seconds) == (plan == "resumed"), index
        if plan == "resumed":
            assert ("reset",) not in chain.calls and chain.calls[0][:2] == ("chunks", start)
    assert lanes.reuse["resumed"] == 4 and lanes.reuse["checkpoints_saved"] == len(turns)
    assert lanes.reuse["resumed_tokens"] == 480 + 480 + 480 + 832
    assert sorted(record.slot for record in lanes.checkpoints) == [0, 1] and len(lanes.checkpoints) == 2
    assert {record.key[:-1] for record in lanes.checkpoints} == {tuple(c2[:576]), tuple(a4[:896])}
    # A one-shot request (no checkpoint asked for) saves nothing and leaves the records as they are.
    held = list(lanes.checkpoints)
    one_shot = lanes.admit(0, _ticket(_prompt(7, length=300)))
    assert one_shot.prefill["checkpoint"] == 0 and lanes.checkpoints == held
    summary = lanes.summary()["prefix_reuse"]
    assert summary["conversations"] == 2 and summary["checkpoint_positions"] == [576, 896]
    lanes.forget()
    assert lanes.checkpoints == [] and lanes.reuse["resumed"] == 0


def test_checkpoints_alone_resume_without_the_snapshot() -> None:
    """``--lanes-conversations`` without ``--lanes-prefix-reuse``: no snapshot is taken or restored, a conversation's
    next turn still resumes its own checkpoint."""

    lanes, chain = _lanes(reuse=False, conversations=1)
    lanes.checkpoint_min_tokens, lanes.resume_margin = 64, 64
    reference, _ = _lanes(reuse=False)
    a1 = _prompt(1, length=200)
    a2 = _turn(a1, 1, 200)
    for prompt, plan in ((a1, "fresh"), (a2, "resumed"), (_prompt(2, length=200), "fresh")):
        admitted = lanes.admit(0, _ticket(prompt, checkpoint=True))
        assert admitted.prefill["reuse"] == plan and admitted.prefill["prefix_captured"] == 0
        reference.admit(0, _ticket(prompt))
        assert lanes.backbone_pager.imports[-1] == reference.backbone_pager.imports[-1]
    assert lanes.prefix is None and ("capture", 288) not in chain.calls


def test_an_image_prompt_prefills_fresh_and_drops_the_records() -> None:
    lanes, chain = _lanes()
    vision = SimpleNamespace(positions=SimpleNamespace(shift_at=lambda position: 0), features=None)
    lanes.tower = lambda ids, positions, images: (vision, {"images": len(images)})
    lanes.admit(0, _ticket(_prompt(1)))
    lanes.admit(0, _ticket(_prompt(2)))
    assert lanes.prefix is not None
    chain.calls.clear()
    image = lanes.admit(0, _ticket(_prompt(3), images=["image"], vision_positions=object()))
    assert image.prefill["reuse"] == "fresh" and chain.calls[0] == ("reset",) and chain.calls[1][-1] is True
    assert (lanes.resident, lanes.prefix) == ((), None)
    assert lanes.admit(0, _ticket(_prompt(4))).prefill["reuse"] == "fresh"


def test_the_pin_threshold_reuses_a_short_prompt() -> None:
    lanes, _ = _lanes(min_tokens=32)
    prompt = list(range(500, 585))  # the json record's length
    plans = [lanes.admit(0, _ticket(prompt)).prefill for _ in range(3)]
    assert [(p["reuse"], p["prefix_captured"], p["prefix_reused"]) for p in plans] == [
        ("fresh", 0, 0),
        ("captured", 64, 0),
        ("restored", 0, 64),
    ]
    default, _ = _lanes()
    assert {default.admit(0, _ticket(prompt)).prefill["reuse"] for _ in range(3)} == {"fresh"}


# -- the server ---------------------------------------------------------------------------------------------------------


PIN_PLANS = ("fresh", "captured", "restored")


class ReuseDevice(scheduler_tests.FakeLanesDevice):
    """The scheduler's fake device with the reuse's records: each admission reports the next plan, and a corrupt
    restore (or resume) admits that lane on another sequence."""

    def __init__(self, lanes: int, *, corrupt_restore: bool = False, plans=PIN_PLANS) -> None:
        super().__init__(lanes)
        self.reuse_min_tokens = lanes_module.REUSE_MIN_TOKENS
        self.checkpoint_min_tokens = lanes_module.CHECKPOINT_MIN_TOKENS
        self.resume_margin = lanes_module.RESUME_MARGIN_TOKENS
        self.plans = list(plans)
        self.thresholds: list[tuple[int, int, int]] = []
        self.checkpoints_asked: list[bool] = []
        self.forgotten = 0
        self.corrupt_restore = corrupt_restore

    def forget(self) -> None:
        self.forgotten += 1

    def admit(self, lane, ticket, between=None):
        admitted = super().admit(lane, ticket, between)
        plan = self.plans.pop(0)
        self.thresholds.append((self.reuse_min_tokens, self.checkpoint_min_tokens, self.resume_margin))
        self.checkpoints_asked.append(ticket.checkpoint)
        if plan in ("restored", "resumed") and self.corrupt_restore:
            self.sequence[lane] = [*ticket.prompt_ids, 1]
            admitted = dataclasses.replace(admitted, pending=scheduler_tests.next_token(self.sequence[lane]))
        prefill = {
            "mode": "chunked",
            "reuse": plan,
            "prefix_reused": 64 * (plan in ("restored", "resumed")),
            "prefix_captured": 64 * (plan == "captured"),
            "checkpoint": 64 * (ticket.checkpoint and plan == "fresh"),
        }
        return dataclasses.replace(admitted, prefill=prefill)


def _record(prompt: list[int]) -> dict:
    stream, _ = scheduler_tests.reference_stream(prompt, max_tokens=97, stop_ids=(), think_budget=None)
    return {"prompt": "json", "prompt_token_ids": prompt, "generated_token_ids": stream}


def test_the_startup_pin_runs_fresh_captured_restored_and_refuses_a_restore_that_differs() -> None:
    prompt = list(range(500, 585))
    record = _record(prompt)
    single = {"prompts": [{"prompt": "json", "device_token_ids": record["generated_token_ids"][:96]}]}
    device = ReuseDevice(2)
    result = server.replay_prefix_reuse(device, record, lanes=2, drafts=4, single_stream=single, require_gate=True)
    assert result["pass"] and result["restore_equals_capture"] and result["gate_pass"] and result["plans_ran"]
    assert [row["plan"] for row in result["rows"].values()] == list(PIN_PLANS)
    assert all(row["equals_single_stream"] for row in result["rows"].values())
    assert [t[0] for t in device.thresholds] == [server.CHUNK_ROWS] * 3  # the 85 ids reach the reuse at one chunk
    assert device.checkpoints_asked == [False] * 3  # the snapshot's pin saves no conversation checkpoint
    assert device.reuse_min_tokens == lanes_module.REUSE_MIN_TOKENS and device.forgotten == 2
    corrupt = ReuseDevice(2, corrupt_restore=True)
    with pytest.raises(session_module.Qwen38ChatChainError, match="the restore differs from the capture"):
        server.replay_prefix_reuse(corrupt, record, lanes=2, drafts=4, single_stream=single, require_gate=True)
    assert corrupt.reuse_min_tokens == lanes_module.REUSE_MIN_TOKENS  # restored on the way out
    reported = server.replay_prefix_reuse(
        ReuseDevice(2, corrupt_restore=True), record, lanes=2, drafts=4, single_stream=single, require_gate=False
    )
    assert (
        not reported["pass"]
        and not reported["restore_equals_capture"]
        and reported["rows"]["captured"]["divergence_index"] is None
    )
    short = server.replay_prefix_reuse(
        ReuseDevice(2), _record(list(range(20))), lanes=2, drafts=4, single_stream=single, require_gate=True
    )
    assert short["skipped"] == "20-token prompt"


def test_the_flag_needs_the_lanes_and_turns_on_after_the_lanes_pin() -> None:
    assert server._parser().get_default("lanes_prefix_reuse") is False
    main = inspect.getsource(server.main)
    for refusal in (
        '"--lanes-prefix-reuse needs --lanes (the single stream keeps its own prompt-end snapshot)"',
        '"--lanes-prefix-reuse needs --prefill-mode chunked (the reuse splits the chunked prefill)"',
    ):
        assert refusal in main
    order = [
        'report["acceptance_lanes"] = replay_acceptance_lanes(',
        "lanes_session.prefix_reuse = args.lanes_prefix_reuse",
        "lanes_session.forget()",
        'report["prefix_reuse"] = replay_prefix_reuse(',
        '(evidence / "prefix-reuse.json").write_text(',
        '"prefix_reuse_pin": (',
        "server.serve_forever(poll_interval=0.5)",
    ]
    positions = [main.index(line) for line in order]
    assert positions == sorted(positions)
    # a lane request's record: the ids the admission did not prefill and the snapshot vocabulary
    handler = inspect.getsource(server.Qwen38ChatHandler._serve_lanes)
    for line in (
        'reused = int(ticket.prefill.get("prefix_reused") or 0)',
        '"prefix_reused": reused,',
        '"reset": not reused,',
        '"snapshot_schedule": "chunked" if ticket.prefill.get("reuse", "fresh") == "fresh" else "forced-tail",',
        '"prefill_tokens": len(prompt_ids) - reused,',
        "usage = _usage(len(prompt_ids), consumed, ticket.queue_wait, reused)",
    ):
        assert line in handler, line
    # ... and in OpenAI's usage shape, the prompt tokens served from a cache
    usage = server._usage(300, 20, 0.5, 256)
    assert usage["prompt_tokens_details"] == {"cached_tokens": 256} and usage["total_tokens"] == 320
    assert server._usage(300, 20, 0.5)["prompt_tokens_details"] == {"cached_tokens": 0}


def test_the_conversation_pin_saves_then_resumes_and_refuses_a_resume_that_differs() -> None:
    prompt = list(range(500, 585))
    record = _record(prompt)
    device = ReuseDevice(2, plans=("fresh", "resumed"))
    result = server.replay_conversation_reuse(device, record, lanes=2, drafts=4, require_gate=True)
    assert result["pass"] and result["resume_equals_save"] and result["gate_pass"] and result["plans_ran"]
    assert [row["plan"] for row in result["rows"].values()] == ["fresh", "resumed"]
    assert device.checkpoints_asked == [True, True]
    assert [t[1:] for t in device.thresholds] == [(server.CHUNK_ROWS, 0)] * 2  # the record reaches a checkpoint
    assert (device.checkpoint_min_tokens, device.resume_margin) == (
        lanes_module.CHECKPOINT_MIN_TOKENS,
        lanes_module.RESUME_MARGIN_TOKENS,
    ) and device.forgotten == 2
    corrupt = ReuseDevice(2, corrupt_restore=True, plans=("fresh", "resumed"))
    with pytest.raises(session_module.Qwen38ChatChainError, match="the resumed stream differs from the saving one"):
        server.replay_conversation_reuse(corrupt, record, lanes=2, drafts=4, require_gate=True)
    not_resumed = server.replay_conversation_reuse(
        ReuseDevice(2, plans=("fresh", "restored")), record, lanes=2, drafts=4, require_gate=False
    )
    assert not not_resumed["pass"] and not not_resumed["plans_ran"] and not_resumed["resume_equals_save"]
    short = server.replay_conversation_reuse(
        ReuseDevice(2), _record(list(range(30))), lanes=2, drafts=4, require_gate=True
    )
    assert short["skipped"] == "30-token prompt"


def test_the_conversations_flag_needs_the_lanes_and_its_pin_follows_the_snapshots() -> None:
    parser = server._parser()
    assert parser.get_default("lanes_conversations") == 0
    main = inspect.getsource(server.main)
    for refusal in (
        '"--lanes-conversations needs --lanes (the checkpoints are the lanes admission\'s)"',
        '"--lanes-conversations needs --prefill-mode chunked (a checkpoint splits the chunked prefill)"',
        'f"--lanes-conversations takes 0..{LANES_CONVERSATIONS_MAX}, got {args.lanes_conversations}"',
    ):
        assert refusal in main, refusal
    order = [
        "conversations=args.lanes_conversations,",
        'report["acceptance_lanes"] = replay_acceptance_lanes(',
        'report["prefix_reuse"] = replay_prefix_reuse(',
        'report["conversation_reuse"] = replay_conversation_reuse(',
        '(evidence / "conversation-reuse.json").write_text(',
        '"conversation_pin": (',
        "server.serve_forever(poll_interval=0.5)",
    ]
    positions = [main.index(line) for line in order]
    assert positions == sorted(positions)
    # the ticket asks for a checkpoint when the conversation is likely to come back: tools, or a past assistant turn
    handler = inspect.getsource(server.Qwen38ChatHandler._serve_lanes)
    assert 'conversation = bool(request["tools"]) or any(' in handler
    assert 'message.get("role") == "assistant" for message in request["messages"]' in handler
    assert "checkpoint=conversation," in handler
    assert '"prefill_reuse": ticket.prefill.get("reuse"),' in handler

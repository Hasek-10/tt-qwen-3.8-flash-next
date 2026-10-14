# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The lanes server's device side (``--lanes B``): the MTP lane chain beside the single-lane traced chain, and the
device protocol :class:`~models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler.Qwen38LaneScheduler`
drives from its one device thread.

The single-lane chain stays resident: it is the prefill engine (its chunk traces, the forced last step, the generic
and alignment generic states -- ``Qwen38ChatSession``'s own prefill section) and the referee of the startup pin.  The
lanes take the rest: ``B`` lanes verify ``R = k + 1`` rows each in one tile (``ttnn/mtp_lanes.py``), every request a
lane whose state was imported from the single-lane state after its prompt's prefill (the stage-2 admission path),
with the three lane traces (``verify_first``, ``commit``, ``draft``) replayed by the lane chain's pass loop.

Allocation order (the tracker's rule: every buffer a trace bakes an address around exists before any capture): the
pagers, their host slots and the lane verify / draft states are allocated in the traced chain's WARM HOOK (after its
warm pass, before its captures), the lane import programs compile there (the generic state as it stands, imported at
four consecutive positions into every lane: every GDN ring phase and QSA raw-history residue variant per lane index)
and the lane bodies run two eager passes (their programs); the chain then captures its own traces; the three lane
traces are captured after them, the allocation tracker verifies every trace, and the program cache must not have
grown.  From then on the program-cache miss guard stays closed: an admission's prefill, eviction and import are
replays and warmed eager programs, so a program the warm-up missed raises instead of compiling under live traces.
An admission is a sequence of device segments (the chunk groups of the prefill, one event per 128 prompt rows; the
forced last step; the eviction; the import), and between them the scheduler may run the decoding lanes' passes
(:meth:`Qwen38LanesSession.admit`'s ``between``): the lane traces and the chunk traces share the one command queue
in order, the lane states and the traced chain's state are disjoint buffers, and every eager move (the seed, the
hand-off, the eviction, the import) follows a device synchronize or a blocking readback, as it did before.

Greedy only (the lane body resolves by argmax; the sampled lanes form is a later wave).  The verify-rows fold
(``gdn_rows_scan``) serves the lanes through its lanes form (its prefix states are allocated with the lane verify state,
its dispatch is the registry's), so the referee chain and the lanes run the served default set, bitwise alike.

The admission's import reads the traced chain's state on device (``mtp_lanes.import_lane_state`` with no host slot:
the state packed and moved lane-ward in one segment, nothing crossing to the host), and its KV slabs only as far as
the prompt reaches (the import chunks below the position: a 2,400-token prompt moves one 4,096-row chunk per QSA layer
instead of the whole context).

Prompt reuse (``prefix_reuse``, the server's ``--lanes-prefix-reuse``): between admissions nothing touches the traced
chain's state, so its caches still hold the last admitted prompt.  An admission whose prompt shares at least
``reuse_min_tokens`` ids with that one prefills the shared part from position 0, takes the chain's prompt snapshot
there (the recurrent buffers; the caches are positional and stay) and prefills the rest; a later admission whose prompt
starts with the same ids restores the snapshot and prefills only what follows (:func:`reuse_plan`).  Agents that share
a system prompt and tools then pay for them once.  The split sits on a 32-row chunk boundary and the snapshot's key
includes the token after it (the MTP layer's last prefix row reads it), so a restore runs no 1-row step and repeats the
capturing admission's arithmetic exactly; against a fresh prefill of the same prompt it is the single stream's
``forced-tail`` class (the slab and chunk groupings differ: equal to rounding, ``docs/PREFILL.md``).

Conversation checkpoints (``conversations``, the server's ``--lanes-conversations N``): an admission whose ticket asks
for one (a request with tools or a past assistant turn: an agent loop or a chat in progress) also saves the chain's
whole state into one of ``N`` host checkpoints on its way through the prompt, at the last 32-row boundary that leaves
the prefill's tail a chunk (``ttnn/lanes.Qwen38TTNNStateCheckpoints``: the pack families and the KV rows below the
position).  The next turn of that conversation renders the same ids up to there and on, so its admission resumes:
the checkpoint written back, the position set, only the turn's new ids prefilled (the reply, the tool results, the
generation prompt).  A checkpoint must reach ``resume_margin`` ids past the shared-prefix snapshot to be preferred
over it (the host round trip against a few chunk replays); the least recently used one is replaced, and a
conversation's own older checkpoint first.  The restored state is the saved one bitwise, so a resumed admission
repeats the saving admission's arithmetic from the checkpoint on.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import torch

import ttnn
from ttnn.unsafe_allocation_tracker import UnsafeAllocationTracker

from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    CHUNK_PREFILL_MIN_ROWS,
    Qwen38PromptSnapshot,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS, LONG_CHUNK_ROWS
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler import (
    Qwen38LaneAdmitted,
    Qwen38LaneTicket,
    lane_geometry,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes, mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import (
    Qwen38LaneLayout,
    Qwen38TTNNLanePager,
    Qwen38TTNNStateCheckpoints,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import PLE_CHECKPOINT_LAYER

# The residue variants an import compiles: the GDN ring phase and the QSA raw-history slot rows are ``position % 4``.
IMPORT_RESIDUES = gdn_module.CONV_KERNEL_SIZE
# The eager passes before the captures: the bootstrap (host-written rows) and one split step (commit, PLE rows,
# verify, draft), as the stage-3 tool ran them; every program of the three bodies compiles here.
EAGER_PASSES = 2
WARM_TOKEN_ID = 1  # any exact vocabulary id for the warm passes' rows
# The admission prefill's event cadence in prompt rows: one event (and one ``between`` call, the scheduler's yield
# point) per 128-row chunk, or per four 32-row chunks, so the decoding lanes' passes can run between chunk groups.
ADMISSION_EVENT_ROWS = LONG_CHUNK_ROWS
# The admission's segments as ``between`` names them: an image prompt's images through the tower, the chunk groups
# of the prefill, a conversation checkpoint saved on the way, the forced last step and the row read; the import into
# the lane (the state packed and moved on device) ends the admission.
ADMISSION_SEGMENTS = ("tower", "chunks", "checkpoint", "prefilled")
# Prompt reuse: the fewest shared ids worth a snapshot (four 32-row chunks; a capture costs its admission one more
# hand-off), and how an admission met the chain (``reuse_plan``).
REUSE_MIN_TOKENS = 128
REUSE_PLANS = ("fresh", "captured", "restored", "resumed")
# Conversation checkpoints: the shortest prompt worth one, and how far past the shared-prefix snapshot a checkpoint
# must reach to be resumed instead (its host round trip, about 40 ms for a few-thousand-token conversation, against
# about 20 ms of chunk replays per 128 rows).
CHECKPOINT_MIN_TOKENS = 512
RESUME_MARGIN_TOKENS = 256


class Qwen38LanesSessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Qwen38LanePrefix:
    """The chain's prompt snapshot as the lanes keep it: the state after ``key[:-1]`` (``position``, a multiple of 32)
    with the MTP layer's row at ``position - 1`` taken over ``key[-1]``, and the n-gram context there.  It stays valid
    while the chain's caches below ``position`` hold these ids: a restoring admission writes from ``position`` on."""

    key: tuple[int, ...]
    ple_context: tuple[int, int] | None

    @property
    def position(self) -> int:
        return len(self.key) - 1


@dataclass(frozen=True)
class Qwen38LaneCheckpoint(Qwen38LanePrefix):
    """A conversation checkpoint: the chain's whole state after ``key[:-1]`` in host slot ``slot`` (the prefix's key
    rule: its last id is the one the MTP layer's last row read), ``used`` its least-recently-used stamp."""

    slot: int
    used: int


@dataclass(frozen=True)
class Qwen38LanePlan:
    """How an admission meets the chain (:func:`reuse_plan`): ``kind`` in REUSE_PLANS, ``start`` the position its
    prefill starts from (the snapshot's or the checkpoint's; 0 for a fresh or capturing admission), ``capture`` where a
    capturing admission takes the snapshot, ``checkpoint`` the one it resumes."""

    kind: str
    start: int = 0
    capture: int | None = None
    checkpoint: Qwen38LaneCheckpoint | None = None


def shared_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """The count of leading ids ``a`` and ``b`` share."""

    count = min(len(a), len(b))
    return next((index for index in range(count) if a[index] != b[index]), count)


def _extends(ids: Sequence[int], key: Sequence[int]) -> bool:
    return len(ids) >= len(key) and tuple(ids[: len(key)]) == tuple(key)


def reuse_plan(
    ids: Sequence[int],
    resident: Sequence[int],
    prefix: Qwen38LanePrefix | None,
    *,
    min_tokens: int,
    checkpoints: Sequence[Qwen38LaneCheckpoint] = (),
    margin: int = RESUME_MARGIN_TOKENS,
) -> Qwen38LanePlan:
    """How an admission of ``ids`` meets the chain: ``resumed`` from the longest checkpoint ``ids`` start with when it
    reaches ``margin`` ids past the snapshot (or past position 0 without one); else ``restored`` from the snapshot
    when ``ids`` start with its key (prefill ``ids[P:]``); else ``captured`` when ``ids`` share more than
    ``min_tokens`` ids with ``resident``, the prompt the chain's caches hold (prefill ``ids[:L]`` from position 0,
    snapshot there, prefill the rest; L is the last multiple of 32 below the shared ids, so the snapshot's key is
    shared too); else ``fresh``."""

    restorable = prefix.position if prefix is not None and _extends(ids, prefix.key) else None
    resumable = max((c for c in checkpoints if _extends(ids, c.key)), key=lambda c: c.position, default=None)
    if resumable is not None and resumable.position >= (restorable or 0) + margin:
        return Qwen38LanePlan("resumed", resumable.position, checkpoint=resumable)
    if restorable is not None:
        return Qwen38LanePlan("restored", restorable)
    common = shared_prefix(ids, resident)
    split = (common - 1) // CHUNK_ROWS * CHUNK_ROWS if common else 0
    return Qwen38LanePlan("captured", capture=split) if split >= min_tokens else Qwen38LanePlan("fresh")


def checkpoint_position(length: int, start: int, *, floor: int) -> int | None:
    """Where an admission of ``length`` ids saves its checkpoint: the last multiple of 32 that leaves at least
    CHUNK_PREFILL_MIN_ROWS ids before the last one, so the rest of the prompt's prefill is a chunk and the key stops
    short of the generation prompt (the next turn renders that spot as the past assistant turn: ``<think>\n`` there,
    ``<think>\n\n</think>`` or the turn's reasoning next time), past the admission's ``start``; None below ``floor``
    (or when no such position is past ``start``)."""

    last = length - 1
    position = last // CHUNK_ROWS * CHUNK_ROWS
    if last - position < CHUNK_PREFILL_MIN_ROWS:
        position -= CHUNK_ROWS
    return position if position >= max(floor, start + CHUNK_ROWS) else None


def _reuse_counts() -> dict[str, int]:
    """The reuse counters: admissions by plan, the prompt ids restored or resumed, the checkpoints saved."""

    return {**dict.fromkeys(REUSE_PLANS, 0), "restored_tokens": 0, "resumed_tokens": 0, "checkpoints_saved": 0}


def _device_rows(tensor) -> list[torch.Tensor]:
    """Every device's local of ``tensor`` on the host (the startup round trip's comparison)."""

    return [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]


def _dram_view(mesh) -> dict[str, int]:
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        "num_banks": int(view.num_banks),
        "free_bytes_per_bank": int(view.total_bytes_free_per_bank),
        "largest_contiguous_bytes_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
        "allocated_bytes_per_bank": int(view.total_bytes_allocated_per_bank),
    }


@dataclass
class Qwen38LanesSession:
    """The lane chain beside the traced chain; see the module docstring."""

    lanes: int
    drafts: int
    marker: Callable[[str], Any]
    admission: dict[str, Any] = field(default_factory=dict)  # lanes_capacity_admission's record (the server fills it)
    clock_ns: Callable[[], int] = time.perf_counter_ns
    # The resident tower's request path (the server's ``vision_prompt_for``: prompt ids, rotary positions, decoded
    # images -> the splice's prompt object and the request's record), set once the tower is resident; None on a
    # text-only process, where an image ticket never reaches the driver (the handler refuses it).
    tower: Callable[..., tuple[Any, dict[str, Any]]] | None = None
    # Prompt reuse (module docstring): off until the server turns it on after the lanes' startup pin.  ``resident`` is
    # the text prompt the chain's caches hold from position 0 (empty when unknown: none yet, or an image prompt),
    # ``prefix`` the snapshot taken inside it, ``reuse`` the admissions by plan and the prompt ids restored.
    prefix_reuse: bool = False
    reuse_min_tokens: int = REUSE_MIN_TOKENS
    resident: tuple[int, ...] = ()
    prefix: Qwen38LanePrefix | None = None
    reuse: dict[str, int] = field(default_factory=lambda: _reuse_counts())
    # Conversation checkpoints (module docstring): ``conversations`` host slots (0: none, nothing allocated), set
    # before ``prepare``; the stores are allocated there, the records name what each slot holds.
    conversations: int = 0
    checkpoint_min_tokens: int = CHECKPOINT_MIN_TOKENS
    resume_margin: int = RESUME_MARGIN_TOKENS
    checkpoints: list[Qwen38LaneCheckpoint] = field(default_factory=list)
    backbone_checkpoints: Any = None
    alignment_checkpoints: Any = None
    _stamps: Any = field(default_factory=itertools.count, repr=False)
    rows: int = field(init=False)
    chain: Any = None
    session: Any = None
    model: Any = None
    mesh: Any = None
    backbone_pager: Any = None
    alignment_pager: Any = None
    verify: Any = None
    draft: Any = None
    traces: Any = None
    output: Any = None
    lane_chain: Any = None
    eager_next_tokens: Any = None
    dram: dict[str, dict[str, int]] = field(default_factory=dict)
    capture_ms: dict[str, float] = field(default_factory=dict)
    warmup_ms: dict[str, float] = field(default_factory=dict)
    program_cache_entries: dict[str, int] = field(default_factory=dict)
    closed: bool = False

    def __post_init__(self) -> None:
        self.lanes, self.rows = lane_geometry(self.lanes, self.drafts)

    # -- the warm hook: allocation and compile before any capture ------------------------------

    def prepare(self, chain) -> None:
        """The traced chain's warm hook (after its warm pass, before its captures, misses allowed): the pagers, the
        conversation checkpoints' host slots and their round trip (``conversations``), the lane verify and draft
        states, the warm imports at every residue into every lane (from the chain's state on device, the form an
        admission runs), the eager lane passes.  Nothing of the chain's own state is changed."""

        if chain.mtp is None:
            raise Qwen38LanesSessionError("the lanes need an MTP chain (--mtp)")
        if chain.mtp.drafts != self.drafts:
            raise Qwen38LanesSessionError(f"the lanes draft {self.drafts}, the chain {chain.mtp.drafts}")
        if chain.misses_forbidden:
            raise Qwen38LanesSessionError("the lanes prepare before the chain's captures (misses allowed)")
        self.chain = chain
        self.mesh = mesh = chain.mesh
        self.model = model = chain.built_target.model
        contract = model.mesh_contract
        context = model.allocated_context
        self.marker("before-lanes-prepare")
        ttnn.synchronize_device(mesh)
        self.dram["before_prepare"] = _dram_view(mesh)
        started = self.clock_ns()
        backbone_layout = model.generic_lane_layout(chain.state)
        alignment_generic = chain.mtp.alignment.generic_state.attention
        alignment_layout = Qwen38LaneLayout(
            lanes=1,
            allocated_context=context,
            kv=(alignment_generic.packed_kv_cache,),
            compressed=(alignment_generic.compressed_index_cache,),
            staging=(alignment_generic.kv_staging,),
            ring=(alignment_generic.raw_key_ring,),
            recurrent=(),
            conv=(),
            ple=(),
        )
        for name, layout in (("backbone", backbone_layout), ("alignment", alignment_layout)):
            unallocated = [i for i, t in enumerate(layout.tensors()) if not t.is_allocated()]
            if unallocated:
                raise Qwen38LanesSessionError(f"{name} layout tensors {unallocated} are not allocated")
        self.backbone_pager = Qwen38TTNNLanePager(mesh, contract, backbone_layout)
        self.alignment_pager = Qwen38TTNNLanePager(mesh, contract, alignment_layout)
        ttnn.synchronize_device(mesh)
        self.dram["after_pagers"] = _dram_view(mesh)
        if self.conversations:
            checkpoints_started = self.clock_ns()
            self._prepare_checkpoints(context)
            self.warmup_ms["checkpoints"] = (self.clock_ns() - checkpoints_started) / 1e6
        self.verify = mtp_lanes.allocate_lane_verify_state(
            model,
            lanes=self.lanes,
            drafts=self.drafts,
            mtp_components=chain.mtp.components,
            positions=[0] * self.lanes,
        )
        self.draft = mtp_lanes.allocate_lane_draft_state(model, self.verify)
        ttnn.synchronize_device(mesh)
        self.dram["after_states"] = _dram_view(mesh)
        self.warmup_ms["allocate"] = (self.clock_ns() - started) / 1e6
        # -- the warm imports: the generic state as the warm pass left it, at four consecutive positions into every
        # lane, read on device as an admission reads it (every import chunk: an admission writes those below its
        # position)
        started = self.clock_ns()
        base = int(chain.state.position.read())
        gdn_layers = sum(1 for layer in model.layers if isinstance(layer.attention, gdn_module.Qwen38TTNNGDN))
        for shift in range(IMPORT_RESIDUES):
            position = base + shift
            for lane in range(self.lanes):
                mtp_lanes.import_lane_state(
                    model,
                    self.verify,
                    lane,
                    position=position,
                    ple_context=None,
                    gdn_phases=[position % gdn_module.CONV_KERNEL_SIZE] * gdn_layers,
                    backbone_slot=None,
                    backbone_pager=self.backbone_pager,
                    alignment_slot=None,
                    alignment_pager=self.alignment_pager,
                )
        ttnn.synchronize_device(mesh)
        self.warmup_ms["imports"] = (self.clock_ns() - started) / 1e6
        # -- the eager passes: the bootstrap on host-written rows, then one split step
        started = self.clock_ns()
        mtp_lanes.write_lane_accepted(model, self.verify, [-1] * self.lanes)
        mtp_lanes.write_lane_active(model, self.verify, [1] * self.lanes, commit=[1] * self.lanes)
        tokens = [[WARM_TOKEN_ID] + [mtp_lanes.BOOTSTRAP_DRAFT_TOKEN] * self.drafts for _ in range(self.lanes)]
        next_tokens = None
        for index in range(EAGER_PASSES):
            if index == 0:
                mtp_lanes.write_lane_verify_inputs(model, self.verify, tokens)
            else:
                mtp_lanes.forward_commit_lanes(model, self.verify)
                mtp_lanes.write_lane_ple_rows(model, self.verify, tokens)
            eager_output = mtp_lanes.forward_verify_lanes(model, self.verify, catch_up=False, land_accept=True)
            mtp_lanes.forward_draft_lanes(model, self.verify, self.draft, eager_output)
            ttnn.synchronize_device(mesh)
            readback, next_tokens = mtp_lanes.read_lane_pass_row(self.verify, self.draft)
            eager_output.release_tensors()
            mtp_lanes.commit_lane_verify_host(model, self.verify, [a + 1 for a in readback.accepted])
            tokens = [list(block) for block in next_tokens]
        device_positions = self.verify.positions.read()
        if device_positions != self.verify.positions.positions:
            raise Qwen38LanesSessionError(
                f"device lane positions {device_positions} vs the host mirror {self.verify.positions.positions} after the eager passes"
            )
        self.eager_next_tokens = next_tokens
        ttnn.synchronize_device(mesh)
        self.warmup_ms["eager_passes"] = (self.clock_ns() - started) / 1e6
        # The host-written and device-read buffers of the lane pass loop, acknowledged to the allocation tracker as the
        # chain acknowledges its own before the miss guard closes.
        verify = self.verify
        for tensor in (
            verify.token_row,
            verify.draft_lanes,
            verify.ple_rows.embedding_rows,
            verify.accepted_lanes,
            verify.accepted_row,
            verify.active_lanes,
            verify.active_row,
            verify.inactive_row,
            verify.commit_lanes,
            verify.positions.row,
            self.draft.pass_row,
        ):
            ttnn.mark_corruptible(tensor)
        for pager in (self.backbone_pager, self.alignment_pager):
            for tensor in (*pager.packs.values(), *pager.kv_stagings):
                ttnn.mark_corruptible(tensor)
        self.program_cache_entries["after_prepare"] = resident_decode.program_cache_count(mesh)
        self.dram["after_prepare"] = _dram_view(mesh)
        self.marker("after-lanes-prepare")

    def _prepare_checkpoints(self, context: int) -> None:
        """The conversation checkpoints' host slots and one round trip through slot 0 over the whole context (every
        KV chunk's slice and write compiles here, under the open miss guard), checked bitwise on the backbone's first
        KV cache and recurrent state and on the alignment layer's cache; slot 0's chunks are freed after."""

        chunk_rows = mtp_lanes.kv_scratch_rows(context)
        self.backbone_checkpoints = Qwen38TTNNStateCheckpoints(
            self.backbone_pager, self.conversations, chunk_rows=chunk_rows
        )
        self.alignment_checkpoints = Qwen38TTNNStateCheckpoints(
            self.alignment_pager, self.conversations, chunk_rows=chunk_rows
        )
        watched = {
            "backbone kv[0]": self.backbone_pager.layout.kv[0],
            "alignment kv[0]": self.alignment_pager.layout.kv[0],
        }
        if self.backbone_pager.layout.recurrent:
            watched["backbone recurrent[0]"] = self.backbone_pager.layout.recurrent[0]
        ttnn.synchronize_device(self.mesh)
        before = {label: _device_rows(tensor) for label, tensor in watched.items()}
        for store in (self.backbone_checkpoints, self.alignment_checkpoints):
            store.save(0, context)
            store.restore(0, context)
            store.release(0)
        ttnn.synchronize_device(self.mesh)
        for label, tensor in watched.items():
            for device, (expected, actual) in enumerate(zip(before[label], _device_rows(tensor), strict=True)):
                bits = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
                if not torch.equal(expected.view(bits), actual.view(bits)):
                    raise Qwen38LanesSessionError(
                        f"checkpoint round trip: {label} on device {device} differs after the restore: max abs "
                        f"{float((actual.float() - expected.float()).abs().max())}"
                    )
        # Every slot's first chunk now: a conversation's first save allocates nothing on the serving path.
        for store in (self.backbone_checkpoints, self.alignment_checkpoints):
            for slot in range(self.conversations):
                store.reserve(slot, 1)

    # -- the captures after the chain's --------------------------------------------------------

    def capture(self, chain, session) -> None:
        """After the traced chain opened (its traces captured, the miss guard closed): the three lane traces, the
        tracker over every trace, the lane chain with every lane parked."""

        if chain is not self.chain or self.verify is None:
            raise Qwen38LanesSessionError("capture follows prepare on the same chain")
        if not chain.misses_forbidden:
            raise Qwen38LanesSessionError("the lane captures run under the chain's closed miss guard")
        self.session = session
        mesh, model = self.mesh, self.model
        self.marker("before-lanes-captures")
        ttnn.synchronize_device(mesh)
        before = _dram_view(mesh)
        self.dram["before_captures"] = before
        guard = lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"lanes {label}")  # noqa: E731
        started = self.clock_ns()
        verify_id, output = mtp_lanes.capture_verify_lanes(
            model, self.verify, catch_up=False, land_accept=True, guard=guard
        )
        self.capture_ms["verify_first"] = (self.clock_ns() - started) / 1e6
        started = self.clock_ns()
        commit_id = mtp_lanes.capture_commit_lanes(model, self.verify, guard=guard)
        self.capture_ms["commit"] = (self.clock_ns() - started) / 1e6
        started = self.clock_ns()
        draft_id = mtp_lanes.capture_draft_lanes(model, self.verify, self.draft, output, guard=guard)
        self.capture_ms["draft"] = (self.clock_ns() - started) / 1e6
        self.traces = mtp_lanes.Qwen38TTNNMTPLaneTraces(verify_first=verify_id, commit=commit_id, draft=draft_id)
        self.output = output
        ttnn.mark_corruptible(output.readback)
        ttnn.synchronize_device(mesh)
        tracker = UnsafeAllocationTracker(mesh)
        for trace_id in (*chain.trace_ids(), verify_id, commit_id, draft_id):
            tracker.verify_before_replay(trace_id)
        after = _dram_view(mesh)
        self.dram["after_captures"] = after
        entries = resident_decode.program_cache_count(mesh)
        self.program_cache_entries["after_captures"] = entries
        if entries != chain.program_cache_entries:
            raise Qwen38LanesSessionError(
                f"program cache {entries} entries after the lane captures vs {chain.program_cache_entries} at the chain's"
            )
        self.lane_chain = mtp_lanes.Qwen38TTNNMTPLaneChain(
            model,
            self.verify,
            self.draft,
            self.traces,
            replay=chain._replay,
            enqueue=chain._enqueue,
            next_tokens=self.eager_next_tokens,
        )
        self.lane_chain.set_active([0] * self.lanes)  # every lane parked until an admission
        ttnn.synchronize_device(mesh)
        self.marker("after-lanes-captures")

    def growth_bytes_per_bank(self) -> dict[str, int]:
        """The MEASURED DRAM growth of the lanes per bank: the states (pagers + lane states, the hook's allocations)
        and the three traces."""

        states = (
            self.dram["after_prepare"]["allocated_bytes_per_bank"]
            - self.dram["before_prepare"]["allocated_bytes_per_bank"]
        )
        traces = (
            self.dram["after_captures"]["allocated_bytes_per_bank"]
            - self.dram["before_captures"]["allocated_bytes_per_bank"]
        )
        return {"states": states, "traces": traces}

    def summary(self) -> dict[str, Any]:
        return {
            "lanes": self.lanes,
            "drafts": self.drafts,
            "rows": self.rows,
            "moe_rows": None if self.verify is None else self.verify.moe_rows,
            "traces": 0 if self.traces is None else 3,
            "capture_ms": dict(self.capture_ms),
            "warmup_ms": dict(self.warmup_ms),
            "program_cache_entries": dict(self.program_cache_entries),
            "dram": {name: dict(view) for name, view in self.dram.items()},
            "dram_growth_bytes_per_bank": None if "after_captures" not in self.dram else self.growth_bytes_per_bank(),
            "admission": dict(self.admission),
            "prefix_reuse": (
                None
                if not (self.prefix_reuse or self.conversations)
                else {
                    "snapshot": self.prefix_reuse,
                    "min_tokens": self.reuse_min_tokens,
                    "snapshot_position": None if self.prefix is None else self.prefix.position,
                    "conversations": self.conversations,
                    "checkpoint_positions": sorted(record.position for record in self.checkpoints),
                    **self.reuse,
                }
            ),
        }

    def forget(self) -> None:
        """Start the reuse over (the startup pins): no records, so the next admission prefills from position 0 and
        resumes nothing, and no counts.  The chain's state and the checkpoints' host slots are unchanged."""

        self.resident, self.prefix, self.checkpoints = (), None, []
        self.reuse = _reuse_counts()

    # -- the device protocol (the driver thread) ------------------------------------------------

    def admit(
        self, lane: int, ticket: Qwen38LaneTicket, between: Callable[[str, int, int], float] | None = None
    ) -> Qwen38LaneAdmitted:
        """The request's prompt prefilled on the traced chain (the session's own prefill section: the reset, the
        chunked prefill of all but the last token when the prompt admits it, the last token forced, the model's next
        token read from the row), then imported into ``lane`` from the chain's state on device (nothing crosses to the
        host; the KV slabs below the position only).  An image prompt (``ticket.images``) runs its images through the
        resident tower first (the ``"tower"`` segment) and prefills through the chunk driver with its rotary positions
        and feature rows, as the single stream does; the lane takes the prompt's rotary shift at the import (the
        decode shift the single stream sets at its hand-off), 0 for text.  A text prompt meets the chain by
        :func:`reuse_plan` (``prefix_reuse``, ``conversations``): the snapshot restored or a checkpoint resumed instead
        of the reset, the snapshot taken and a checkpoint saved (``ticket.checkpoint``) on the way;
        ``prefill["reuse"]`` names the plan and ``prefix_reused`` the ids not prefilled.

        ``between`` is the scheduler's yield point between the admission's device segments: called after the tower
        (``"tower"``), after every 128 prompt rows of the chunked prefill (``"chunks"``, with the rows consumed and the
        rows of the prefill), after a checkpoint's save (``"checkpoint"``) and after the forced last step and its row
        read (``"prefilled"``), each time with the device idle and every chunk input for the next segment prepared but
        not uploaded; it returns the seconds it spent (the decoding lanes' passes), which the segment records below
        exclude.  None: the admission runs whole (the single-stream event cadence)."""

        session, chain, model = self.session, self.chain, self.model
        if session is None or self.lane_chain is None:
            raise Qwen38LanesSessionError("the lanes are not captured")
        ids = [int(token) for token in ticket.prompt_ids]
        if not ids:
            raise Qwen38LanesSessionError("an admission needs a prompt")
        spent = 0.0  # seconds ``between`` took inside the prefill segment

        def between_chunks(done: int, total: int) -> None:
            nonlocal spent
            spent += float(between("chunks", done, total))

        vision = vision_record = None
        tower_seconds = 0.0
        if ticket.images:
            # The tower on this (device) thread, before the prefill: the feature rows and the splice's prompt object
            # (validated against the prompt); a text-only process never sees an image ticket.
            if self.tower is None:
                raise Qwen38LanesSessionError("an image prompt reached the lanes without a resident tower")
            tower_started = self.clock_ns()
            vision, vision_record = self.tower(ids, ticket.vision_positions, ticket.images)
            tower_seconds = (self.clock_ns() - tower_started) / 1e9
            if between is not None:
                between("tower", len(ids), len(ids))
        started = self.clock_ns()
        chunked = []  # the chunk driver's results, in prefill order

        def prefill_chunks(token_ids: Sequence[int], following: int, chunk_vision) -> None:
            result = session._prefill_chunked(
                token_ids,
                following,
                None,
                chunk_vision,
                between_chunks=None if between is None else between_chunks,
                event_rows=None if between is None else ADMISSION_EVENT_ROWS,
            )
            if result.stopped is not None:
                raise Qwen38LanesSessionError(f"the chunked prefill stopped: {result.stopped!r}")
            chunked.append(result)

        # The plan (module docstring): an image prompt always prefills from position 0, and so does every prompt of a
        # chain without the chunk trace.  The records describe the chain's caches: they are dropped until this
        # admission's prefill has ended.
        text = vision is None and session.prefill_mode == "chunked"
        plan = Qwen38LanePlan("fresh")
        if text and (self.prefix_reuse or self.checkpoints):
            plan = reuse_plan(
                ids,
                self.resident,
                self.prefix if self.prefix_reuse else None,
                min_tokens=self.reuse_min_tokens if self.prefix_reuse else len(ids) + 1,
                checkpoints=self.checkpoints,
                margin=self.resume_margin,
            )
        prefix, captured = self.prefix, None
        self.resident, self.prefix = (), None
        seconds_resume = seconds_checkpoint = 0.0
        if plan.kind == "restored":
            session.snapshot = Qwen38PromptSnapshot(prefix.key[:-1], prefix.ple_context)
            session._restore_prompt_snapshot()  # the chain's state at the split: position, phases, n-gram context
        elif plan.kind == "resumed":
            seconds_resume = self._resume(plan.checkpoint)
        else:
            session.reset()
        # The stops on the way through the prompt (32-row boundaries, so every stretch between them is whole chunks):
        # the snapshot a capturing admission takes, the checkpoint a conversation's admission saves.
        save_at = None
        if text and self.conversations and ticket.checkpoint:
            # Past the snapshot this prompt will start with by a margin (else the snapshot serves its next turn).
            anchor = plan.capture if plan.kind == "captured" else None
            if anchor is None and prefix is not None and _extends(ids, prefix.key):
                anchor = prefix.position
            floor = max(self.checkpoint_min_tokens, -1 if anchor is None else anchor + self.resume_margin)
            save_at = checkpoint_position(len(ids), plan.start, floor=floor)
        at = plan.start
        for stop in sorted({stop for stop in (plan.capture, save_at) if stop is not None}):
            prefill_chunks(ids[at:stop], ids[stop], None)
            if stop == plan.capture:
                chain.capture_prompt_snapshot(stop)
                captured = Qwen38LanePrefix(tuple(ids[: stop + 1]), session.ple_context)
            if stop == save_at:
                seconds_checkpoint = self._save_checkpoint(ids, stop)
                if between is not None:
                    spent += float(between("checkpoint", stop, len(ids)))
            at = stop
        suffix = list(ids[at:])
        if vision is not None and (
            session.prefill_mode != "chunked" or session.chunk_prefill_rows(len(suffix) - 1) < CHUNK_PREFILL_MIN_ROWS
        ):
            raise Qwen38LanesSessionError("an image prompt needs the chunked prefill (its pads never take 1-row steps)")
        if session.prefill_mode == "chunked" and session.chunk_prefill_rows(len(suffix) - 1) >= CHUNK_PREFILL_MIN_ROWS:
            prefill_chunks(suffix[:-1], suffix[-1], vision)
            suffix = suffix[-1:]
        with chain.loop_guard():
            finish = session._prefill(suffix, None, None)
        if finish is not None:
            raise Qwen38LanesSessionError(f"the prefill ended early: {finish!r}")
        pending = int(chain.read_token_row())
        position = int(chain.position())
        if position != len(ids):
            raise Qwen38LanesSessionError(f"device position {position} after prefilling {len(ids)} tokens")
        if vision is None:
            self.resident = tuple(ids)
            if plan.kind == "captured":
                self.prefix = captured
            elif prefix is not None and _extends(ids, prefix.key):
                self.prefix = prefix  # the caches below it hold its ids (restored, or a checkpoint of them resumed)
        if self.prefix_reuse or self.conversations:
            self.reuse[plan.kind] += 1
            if plan.kind in ("restored", "resumed"):
                self.reuse[f"{plan.kind}_tokens"] += plan.start
        ple_context = session.ple_context
        prefilled = self.clock_ns()
        if between is not None:
            between("prefilled", len(ids), len(ids))
        import_started = self.clock_ns()
        gdn_layers = sum(1 for layer in model.layers if isinstance(layer.attention, gdn_module.Qwen38TTNNGDN))
        # The lane's rotary shift: the prompt's (the chunk driver's hand-off set the same on the traced chain; the
        # prompt ends in text after its last image, so the shift is within the position's block start), 0 for text.
        rope_shift = 0 if vision is None else min(vision.positions.shift_at(position), position & ~3)
        # From the chain's state on device, in this one segment: no trace replays between the pack and its use.
        mtp_lanes.import_lane_state(
            model,
            self.verify,
            lane,
            position=position,
            ple_context=ple_context,
            gdn_phases=[position % gdn_module.CONV_KERNEL_SIZE] * gdn_layers,
            backbone_slot=None,
            backbone_pager=self.backbone_pager,
            alignment_slot=None,
            alignment_pager=self.alignment_pager,
            rope_shift=rope_shift,
            kv_rows=position,
        )
        ttnn.synchronize_device(self.mesh)
        imported = self.clock_ns()
        timings = [result.timing for result in chunked]
        seconds = {"prefill": (prefilled - started) / 1e9 - spent - seconds_resume - seconds_checkpoint}
        if plan.kind == "resumed":
            seconds["resume"] = seconds_resume
        if save_at is not None:
            seconds["checkpoint"] = seconds_checkpoint
        seconds["import"] = (imported - import_started) / 1e9
        if vision is not None:
            seconds = {"tower": tower_seconds, **seconds}
        return Qwen38LaneAdmitted(
            pending=pending,
            position=position,
            ple_context=ple_context,
            seconds=seconds,
            vision=vision_record,
            prefill={
                "mode": "chunked" if chunked else "teacher_forced",
                "tokens": len(ids),
                "forced_tokens": len(suffix) + sum(timing.alignment_steps for timing in timings),
                "chunks": sum(timing.chunks for timing in timings),
                "long_chunks": sum(timing.long_chunks for timing in timings),
                "slabs": sum(timing.slabs for timing in timings),
                "reuse": plan.kind,
                "prefix_reused": plan.start,
                "prefix_captured": plan.capture or 0,
                "checkpoint": save_at or 0,
            },
        )

    def _resume(self, checkpoint: Qwen38LaneCheckpoint) -> float:
        """``checkpoint`` written back into the chain's state (the backbone and the alignment layer), the position, ring
        phases and n-gram context set, the session's record following it; the seconds it took."""

        started = self.clock_ns()
        position, session, chain = checkpoint.position, self.session, self.chain
        self.backbone_checkpoints.restore(checkpoint.slot, position)
        self.alignment_checkpoints.restore(checkpoint.slot, position)
        self.model.set_generic_position(chain.state, position)
        if chain.mtp is not None:
            chain.mtp.step_written = False
        ttnn.synchronize_device(self.mesh)
        actual = int(chain.position())
        if actual != position:
            raise Qwen38LanesSessionError(
                f"resumed checkpoint left the device position at {actual}, expected {position}"
            )
        session.committed = list(checkpoint.key[:-1])
        session.ple_context = checkpoint.ple_context
        session.committed_vision = ()
        session.snapshot = None
        session.last_finish = None
        session.row_unconsumed = False
        session.row_token = None
        self.checkpoints = [
            replace(record, used=next(self._stamps)) if record is checkpoint else record for record in self.checkpoints
        ]
        return (self.clock_ns() - started) / 1e9

    def _save_checkpoint(self, ids: Sequence[int], position: int) -> float:
        """The chain's state at ``position`` (the ids below it consumed) into a host slot: this conversation's own (a
        checkpoint ``ids`` extend), else a free one, else the least recently used; the seconds it took."""

        own = [record for record in self.checkpoints if _extends(ids, record.key)]
        held = {record.slot for record in self.checkpoints}
        if own:
            slot = max(own, key=lambda record: record.position).slot
        elif len(held) < self.conversations:
            slot = min(set(range(self.conversations)) - held)
        else:
            slot = min(self.checkpoints, key=lambda record: record.used).slot
        started = self.clock_ns()
        self.backbone_checkpoints.save(slot, position)
        self.alignment_checkpoints.save(slot, position)
        record = Qwen38LaneCheckpoint(tuple(ids[: position + 1]), self.session.ple_context, slot, next(self._stamps))
        self.checkpoints = [other for other in self.checkpoints if other.slot != slot] + [record]
        self.reuse["checkpoints_saved"] += 1
        return (self.clock_ns() - started) / 1e9

    def write_counts(self, counts: Sequence[int]) -> None:
        mtp_lanes.write_lane_accepted(self.model, self.verify, list(counts))

    def override_commit(self, lane: int, record, committed_rows: int) -> None:
        """Lane ``lane`` commits ``committed_rows`` of the pass ``record`` instead of the device's count (the forced
        ``</think>``): the position mirror and the device row rewound to the pass's start plus the rows kept, the
        checkpoint layer's n-gram context that of the rows kept.  The count itself goes with the boundary's
        ``write_counts``."""

        verify = self.verify
        if not 1 <= committed_rows <= verify.rows:
            raise ValueError(f"committed rows must be in [1, {verify.rows}], got {committed_rows!r}")
        mirror = list(verify.positions.positions)
        mirror[lane] = int(record.positions[lane]) + committed_rows
        verify.positions.write(mirror)
        ple_state = verify.layers[PLE_CHECKPOINT_LAYER].ple
        contexts = list(ple_state.token_contexts)
        contexts[lane] = verify.pass_contexts[lane][committed_rows]
        ple_state.token_contexts = tuple(contexts)
        ple_state.validate()

    def set_blocks(self, blocks: Sequence[Sequence[int] | None]) -> None:
        current = [list(block) for block in self.lane_chain.next_tokens]
        for lane, block in enumerate(blocks):
            if block is not None:
                current[lane] = [int(token) for token in block]
        self.lane_chain.set_next_tokens(current)

    def set_active(self, mask: Sequence[int]) -> None:
        flags = [int(flag) for flag in mask]
        if flags != list(self.lane_chain.active):
            self.lane_chain.set_active(flags)

    def step(self):
        return self.lane_chain.step()

    def room(self, position: int) -> bool:
        return mtp_v2.verify_pass_fits(int(position), self.model.allocated_context)

    # -- release ---------------------------------------------------------------------------------

    def close(self) -> None:
        """The lane traces, the output, the draft and verify states, the pagers; before the chain's own close."""

        if self.closed or self.mesh is None:
            return
        mesh = self.mesh
        ttnn.synchronize_device(mesh)
        if self.traces is not None:
            for trace_id in (self.traces.verify_first, self.traces.commit, self.traces.draft):
                ttnn.release_trace(mesh, trace_id)
            self.traces = None
        if self.output is not None and self.output.active:
            self.output.release_tensors()
        self.output = None
        if self.draft is not None:
            mtp_lanes.release_lane_draft_state(self.model, self.verify, self.draft)
            self.draft = None
        if self.verify is not None:
            mtp_lanes.release_lane_verify_state(self.model, self.verify)
            self.verify = None
        for pager in (self.backbone_pager, self.alignment_pager):
            if pager is not None:
                pager.release()
        self.backbone_pager = self.alignment_pager = None
        ttnn.synchronize_device(mesh)
        self.closed = True

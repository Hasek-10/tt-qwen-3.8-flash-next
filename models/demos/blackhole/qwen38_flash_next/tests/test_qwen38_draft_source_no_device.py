# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host drafting (``--draft-source hybrid|ngram``) and the per-request draft-count list without a device: the chain
extension's counters and summary, the session handing the committed ids to the pass loop's drafter, the open and the
server's switches and refusals.  The pass loop itself is pinned in ``test_mtp_v2_step6_draft_no_device.py`` (the
host-first form against the fixed-five stream) and the drafter in ``test_qwen38_ngram_drafter_no_device.py``."""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tests import test_mtp_sampled_session_no_device as sampled_session
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2, ngram_draft

K = sampled_session.K
BASE_SUMMARY_KEYS = {
    "k",
    "anchor",
    "sampled",
    "passes",
    "accepted_drafts",
    "tokens_per_pass",
    "accepted_histogram",
    "accepted_per_pass",
}


def _chain_mtp(**fields) -> session_module.Qwen38ChainMTP:
    return session_module.Qwen38ChainMTP(
        drafts=K, anchor="off", components=None, verify=None, draft=None, step_inputs=None, chunk_extension=None, **fields
    )


def _record(accepted: int, source: str) -> mtp_v2.Qwen38TTNNMTPPassRecord:
    return mtp_v2.Qwen38TTNNMTPPassRecord(0, 0, (1, 2, 3, 4, 5), accepted, (), (), 0, 0, False, {}, source=source)


# -- the chain extension ---------------------------------------------------------------------------------------------


def test_draft_sources_and_the_extension_records_host_passes_only_on_a_host_source() -> None:
    assert session_module.DRAFT_SOURCES == ("mtp", "hybrid", "ngram")
    device = _chain_mtp()
    assert device.draft_source == "mtp" and set(device.summary()) == BASE_SUMMARY_KEYS  # the default record unchanged
    assert set(device.counters()) == set(session_module.Qwen38ChainMTP.COUNTERS)
    host = _chain_mtp(draft_source="hybrid")
    host.record(_record(2, "host"))
    snapshot = host.counters()  # a request starting here
    host.record(_record(4, "host"))
    host.record(_record(0, "device"))
    assert host.host_drafted_passes == 2 and host.passes == 3
    since = host.summary(since=snapshot)
    assert set(since) == BASE_SUMMARY_KEYS | {"draft_source", "host_drafted_passes"}
    assert since["draft_source"] == "hybrid" and since["host_drafted_passes"] == 1 and since["passes"] == 2
    assert host.summary()["host_drafted_passes"] == 2
    with pytest.raises(ValueError, match="draft_source must be one of"):
        _chain_mtp(draft_source="lookahead")
    # a host default keeps the second queue: a host-drafted request's pass loop just takes no early reader
    assert _chain_mtp(draft_source="ngram", ple_early=True).ple_early


# -- the session: the committed ids reach the drafter, the stream stays the target's ------------------------------


class HostDraftChain(sampled_session.FakeMTPChain):
    """The scripted chain with a host source: ``mtp_enter`` takes the committed ids and seeds the real drafter, and a
    pass whose drafter proposes hands the next pass the proposal instead of the device drafts (the record says so)."""

    def __init__(self, source: str) -> None:
        super().__init__(sampled=False)
        self.mtp.draft_source = source
        self.histories: list[tuple[str, list[int]]] = []  # the host-drafted entries: (source, history)
        self.enter_sources: list[str | None] = []  # every entry's draft_source keyword (None: the plain call form)
        self.drafter: ngram_draft.Qwen38NgramDrafter | None = None

    def mtp_enter(self, first_token: int, ple_context, *, decide=None, history=(), draft_source=None):
        source = self.mtp.draft_source if draft_source is None else draft_source
        self.enter_sources.append(draft_source)
        self.drafter = None
        if source != "mtp":
            self.histories.append((source, list(history)))
            self.drafter = ngram_draft.Qwen38NgramDrafter(
                K, history=history, fill_token=session_module.MTP_BOOTSTRAP_DRAFT_TOKEN, decline=source == "hybrid"
            )
        return super().mtp_enter(first_token, ple_context, decide=decide)

    def _pass(self, tokens):
        record = super()._pass(tokens)
        if self.drafter is None:
            return record
        self.drafter.extend(list(tokens[: record.accepted + 1]))
        proposal = self.drafter.propose(record.next_token)
        if proposal is None:
            return record
        self.next_tokens = [record.next_token, *proposal]
        return replace(record, first_draft=proposal[0], source="host")


@pytest.mark.parametrize("source", ("ngram", "hybrid"))
def test_a_host_source_session_hands_the_committed_ids_to_the_drafter_and_keeps_the_greedy_stream(source) -> None:
    chain = HostDraftChain(source)
    session = session_module.Qwen38ChatSession(chain, template=None)
    completion = session.complete(sampled_session.PROMPT, 14, stop_ids=())
    assert completion.token_ids == sampled_session._greedy_stream(sampled_session.PROMPT, 14)
    # The drafter's history is the prompt the device consumed before the pass loop's first token.
    assert chain.histories == [(source, sampled_session.PROMPT)]
    assert completion.mtp["draft_source"] == source
    assert completion.mtp["host_drafted_passes"] == chain.mtp.host_drafted_passes
    if source == "ngram":  # the host alone: every pass but the last proposes (fill tokens when nothing recurs)
        assert chain.mtp.host_drafted_passes == chain.passes
    # A default chain keeps the call form every chain knows: no history keyword.
    default_chain = sampled_session.FakeMTPChain(sampled=False)
    default = session_module.Qwen38ChatSession(default_chain, template=None).complete(
        sampled_session.PROMPT, 14, stop_ids=()
    )
    assert default.token_ids == completion.token_ids and "draft_source" not in default.mtp


def test_a_request_picks_its_drafter_and_reports_it() -> None:
    """``draft_source`` per request: a device-default chain drafts one request on the host and the next on the
    device; a host-default chain serves a request that asks for the device; the stream is the target's every time,
    and the host fields appear exactly when a host source is involved."""

    expected = sampled_session._greedy_stream(sampled_session.PROMPT, 14)
    chain = HostDraftChain("mtp")
    session = session_module.Qwen38ChatSession(chain, template=None)
    host = session.complete(sampled_session.PROMPT, 14, stop_ids=(), draft_source="ngram")
    assert host.token_ids == expected and host.mtp["draft_source"] == "ngram"
    assert host.mtp["host_drafted_passes"] == host.mtp["passes"] > 0
    device = session.complete(sampled_session.PROMPT_B, 10, stop_ids=())
    assert set(device.mtp) == BASE_SUMMARY_KEYS  # the default record, unchanged
    assert chain.enter_sources == ["ngram", None]  # the device request kept the call form every chain knows
    assert [source for source, _ in chain.histories] == ["ngram"]
    host_default = HostDraftChain("hybrid")
    session = session_module.Qwen38ChatSession(host_default, template=None)
    chosen = session.complete(sampled_session.PROMPT, 14, stop_ids=(), draft_source="mtp")
    assert chosen.token_ids == expected and chosen.mtp["draft_source"] == "mtp"
    assert chosen.mtp["host_drafted_passes"] == 0 and host_default.histories == []
    with pytest.raises(session_module.Qwen38ChatRequestError, match="needs an MTP chain"):
        session.complete(sampled_session.PROMPT, 4, stop_ids=(), draft_source="lookahead")
    request = sampled_session.step.Qwen38SamplingRequest(sampled_session.THINKING)
    with pytest.raises(session_module.Qwen38ChatRequestError, match="greedy requests only"):
        session.complete(sampled_session.PROMPT, 4, stop_ids=(), draft_source="hybrid", sampling=request)


def test_mtp_enter_builds_the_drafter_for_greedy_requests_only_and_the_open_admits_the_source() -> None:
    enter = inspect.getsource(session_module.Qwen38TracedChain.mtp_enter)
    drafter_at = enter.index('if source != "mtp" and decide is None and before_verify_sampled is None:')
    assert drafter_at > enter.index("if mtp.ple_early and decide is None:")
    assert "source = mtp.draft_source if draft_source is None else draft_source" in enter
    assert enter.index("early_reader = None", drafter_at) < enter.index("ngram_draft.Qwen38NgramDrafter(", drafter_at)
    assert 'decline=source == "hybrid"' in enter
    assert '**({} if host_drafter is None else {"host_drafter": host_drafter}),' in enter
    open_source = inspect.getsource(session_module.Qwen38TracedChain.open)
    assert 'mtp_draft_source: str = "mtp"' in open_source
    assert "a host draft source needs mtp: it drafts the MTP pass loop" in open_source
    assert open_source.count("draft_source=mtp_draft_source,") == 2  # the default chain and every alternate
    construct = inspect.getsource(session_module.construct_chain)
    assert 'mtp_draft_source: str = "mtp"' in construct and "mtp_draft_source=mtp_draft_source," in construct
    chat = inspect.getsource(session_module.Qwen38ChatSession)
    assert "history=tuple(self.committed)" in chat and "draft_source=self.request_draft_source," in chat


# -- the server --------------------------------------------------------------------------------------------------------


def test_server_flag_needs_mtp_runs_one_queue_and_has_no_lanes_form() -> None:
    parser = server._parser()
    assert parser.get_default("draft_source") == "mtp"
    assert tuple(parser._option_string_actions["--draft-source"].choices) == session_module.DRAFT_SOURCES
    lanes_args = dict(
        lanes=4, mtp=4, sampling=False, sampling_discriminator=False, agreement_reference=None, prefill_mode="chunked"
    )
    assert server.lanes_switches({}, SimpleNamespace(**lanes_args, draft_source="mtp"))["lanes"] == 4
    with pytest.raises(SystemExit, match="no lanes form"):
        server.lanes_switches({}, SimpleNamespace(**lanes_args, draft_source="hybrid"))
    with pytest.raises(SystemExit, match="no lanes form"):  # any per-request value, the list form too
        server.lanes_switches({server.MTP_DRAFTS_PER_REQUEST_VARIABLE: "4,8"}, SimpleNamespace(**lanes_args))
    main = inspect.getsource(server.main)
    assert "runs the one-queue pass loop" not in main  # a host default no longer takes the second queue away
    order = [
        "lanes_switch = lanes_switches(os.environ, args)",
        "mtp_ple_early = ple_early_switch(os.environ, applicable=args.mtp is not None)",
        'if args.draft_source != "mtp" and args.mtp is None:',
        "draft_sources_admitted = DRAFT_SOURCES if args.mtp is not None and lanes_switch is None else ()",
        '"draft_source": args.draft_source if args.mtp is not None else None,',
        '"draft_sources_admitted": list(draft_sources_admitted),',
        "draft_sources_admitted=draft_sources_admitted,",
        "command_queues=2 if mtp_ple_early else 1",
        "mtp_draft_source=args.draft_source,",
    ]
    positions = [main.index(line) for line in order]
    assert positions == sorted(positions)


def _document(**fields):
    return {"model": server.MODEL_ID, "messages": [{"role": "user", "content": "hi"}], **fields}


def test_the_request_field_is_admitted_on_a_single_stream_mtp_server_and_refused_elsewhere() -> None:
    admitted = session_module.DRAFT_SOURCES
    assert "draft_source" in server.KNOWN_REQUEST_FIELDS
    assert server.parse_chat_request(_document(), draft_sources_admitted=admitted)["draft_source"] is None
    for source in admitted:
        assert server.parse_chat_request(_document(draft_source=source), draft_sources_admitted=admitted)[
            "draft_source"
        ] == source
    extra = server.parse_chat_request(_document(extra_body={"draft_source": "hybrid"}), draft_sources_admitted=admitted)
    assert extra["draft_source"] == "hybrid"
    with pytest.raises(server.Qwen38ChatRequestRejected, match="not admitted by this server"):
        server.parse_chat_request(_document(draft_source="hybrid"))  # no --mtp, or the lanes
    with pytest.raises(server.Qwen38ChatRequestRejected, match="must be one of"):
        server.parse_chat_request(_document(draft_source="lookahead"), draft_sources_admitted=admitted)
    with pytest.raises(server.Qwen38ChatRequestRejected, match="greedy requests only"):
        server.parse_chat_request(
            _document(draft_source="hybrid", temperature=0.7), draft_sources_admitted=admitted
        )
    served = inspect.getsource(server.Qwen38ChatHandler)
    assert "draft_sources_admitted=self.server.draft_sources_admitted," in served
    assert 'draft_source=request["draft_source"],' in served


def test_per_request_switch_takes_a_list_of_draft_counts_beside_the_measured_pair() -> None:
    switch = server.mtp_drafts_per_request_switch
    variable = server.MTP_DRAFTS_PER_REQUEST_VARIABLE
    assert switch({variable: "1"}, drafts=4) == (4, 5)  # the measured pair, unchanged
    assert switch({variable: "4,8,15"}, drafts=4) == (4, 8, 15)
    assert switch({variable: "15,4"}, drafts=4) == (4, 15)  # the default first
    for value, drafts in (("4,4", 4), ("4,16", 4), ("5,8", 4), ("4,2", 4), ("4,8", None)):
        with pytest.raises(SystemExit, match="needs distinct draft counts"):
            switch({variable: value}, drafts=drafts)
    with pytest.raises(SystemExit, match="must be 0 or 1"):
        switch({variable: "4,x"}, drafts=4)

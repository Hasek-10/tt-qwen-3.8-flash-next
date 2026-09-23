# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-request MTP draft length, no device: the server's ``--mtp K[,K...]`` arms, the request field and its policy,
the extension's arm selection and reporting, the admission with arms, and the verify state's shared alignment
generic state (source contract)."""

from __future__ import annotations

import argparse
import inspect
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2

MTP_DRAFTS = session_module.MTP_DRAFTS


def _arm(drafts: int, generic_state: object) -> session_module.Qwen38ChainMTPArm:
    alignment = SimpleNamespace(generic_state=generic_state, layer=object(), verify_state=object(), residual=None)
    verify = SimpleNamespace(drafts=drafts, rows=drafts + 1, alignment=alignment)
    traces = mtp_v2.Qwen38TTNNMTPTraces(verify_first=10 * drafts, draft=10 * drafts + 1, commit=10 * drafts + 2)
    return session_module.Qwen38ChainMTPArm(
        drafts=drafts, verify=verify, draft=SimpleNamespace(drafts=drafts), traces=traces, verify_output=object()
    )


def _extension(*counts: int, default: int | None = None, generic_state: object | None = None):
    generic_state = object() if generic_state is None else generic_state
    arms = {k: _arm(k, generic_state) for k in counts}
    return session_module.Qwen38ChainMTP(
        arms=arms,
        default_drafts=counts[0] if default is None else default,
        anchor="off",
        components=object(),
        step_inputs=SimpleNamespace(),
        chunk_extension=None,
    )


# --------------------------------------------------------------------------- --mtp K[,K...]


def test_parse_mtp_drafts_accepts_one_or_more_admitted_counts_in_order() -> None:
    assert server_module.parse_mtp_drafts("4") == (4,)
    assert server_module.parse_mtp_drafts("4,7") == (4, 7)
    assert server_module.parse_mtp_drafts(" 7 , 4 ") == (7, 4)  # the first is the default, whatever its size
    assert server_module.parse_mtp_drafts(",".join(map(str, MTP_DRAFTS))) == MTP_DRAFTS


@pytest.mark.parametrize("text", ("", "4,", "2", "16", "4,4", "4;7", "four", "4,7,4"))
def test_parse_mtp_drafts_rejects_what_the_arms_cannot_take(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):  # allow-pytest.raises: argparse contract
        server_module.parse_mtp_drafts(text)


def test_server_mtp_argument_is_the_parsed_arm_list() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtp", type=server_module.parse_mtp_drafts, default=None)
    assert parser.parse_args([]).mtp is None
    assert parser.parse_args(["--mtp", "4,7"]).mtp == (4, 7)


# --------------------------------------------------------------------------- the request field and its policy


def test_choose_speculative_drafts_explicit_then_tools_then_default() -> None:
    choose = server_module.choose_speculative_drafts
    assert choose({"speculative_drafts": 7, "tools": []}, (4, 7), 4) == 7
    assert choose({"speculative_drafts": 4, "tools": [{"type": "function"}]}, (4, 7), 4) == 4
    assert choose({"speculative_drafts": None, "tools": [{"type": "function"}]}, (4, 7), 4) == 7
    assert choose({"tools": [{"type": "function"}]}, (7, 4), 7) == 7
    assert choose({"tools": []}, (4, 7), 4) == 4
    assert choose({}, (4,), 4) == 4
    assert choose({"tools": [{"type": "function"}]}, (4,), 4) == 4  # one arm: tools change nothing


def test_parse_chat_request_reads_speculative_drafts_and_rejects_bad_values() -> None:
    document = {"messages": [{"role": "user", "content": "hi"}]}
    assert server_module.parse_chat_request(document, seed=1)["speculative_drafts"] is None
    assert server_module.parse_chat_request({**document, "speculative_drafts": 7}, seed=1)["speculative_drafts"] == 7
    assert "speculative_drafts" in server_module.KNOWN_REQUEST_FIELDS
    assert server_module.parse_chat_request({**document, "speculative_drafts": 4}, seed=1)["ignored"] == []
    for bad in (2, 16, "4", 4.0, True, [4]):
        with pytest.raises(server_module.Qwen38ChatRequestRejected) as info:  # allow-pytest.raises: exact param
            server_module.parse_chat_request({**document, "speculative_drafts": bad}, seed=1)
        assert info.value.param == "speculative_drafts", bad


# --------------------------------------------------------------------------- the extension's arms


def test_extension_defaults_to_the_first_arm_and_reads_the_active_arm_through_one_name() -> None:
    extension = _extension(4, 7)
    assert extension.drafts == extension.active_drafts == extension.default_drafts == 4
    assert extension.verify is extension.arms[4].verify and extension.draft is extension.arms[4].draft
    assert extension.traces is extension.arms[4].traces and extension.verify_output is extension.arms[4].verify_output
    assert extension.alignment is extension.arms[4].verify.alignment
    assert extension.select(7) == 7 and extension.drafts == 7 and extension.verify is extension.arms[7].verify
    assert extension.select(None) == 4 and extension.drafts == 4
    assert extension.admitted(7) and not extension.admitted(5) and not extension.admitted(True)
    assert extension.captured_trace_ids() == [40, 42, 41, 70, 72, 71]  # by k: verify_first, commit, draft


def test_extension_refuses_an_unopened_arm_and_a_switch_under_a_live_loop() -> None:
    extension = _extension(4, 7)
    with pytest.raises(session_module.Qwen38ChatRequestError):  # allow-pytest.raises: a 400, not a chain fault
        extension.select(5)
    extension.select(7)
    extension.chain = object()
    assert extension.select(7) == 7  # the live arm may be re-selected
    with pytest.raises(session_module.Qwen38ChatChainError):  # allow-pytest.raises: the loop owns the arm
        extension.select(4)
    extension.chain = None
    assert extension.select(4) == 4


def test_extension_counts_passes_per_arm_and_reports_the_active_one() -> None:
    extension = _extension(4, 7)
    extension.record(SimpleNamespace(accepted=3, source="device", sampled=False))
    extension.select(7)
    extension.record(SimpleNamespace(accepted=6, source="device", sampled=False))
    extension.record(SimpleNamespace(accepted=0, source="device", sampled=False))
    assert (extension.arms[4].passes, extension.arms[4].accepted_drafts) == (1, 3)
    assert (extension.arms[7].passes, extension.arms[7].accepted_drafts) == (2, 6)
    assert (extension.passes, extension.accepted_drafts) == (3, 9)
    summary = extension.summary()
    assert summary["k"] == 7 and summary["arms"] == [4, 7] and summary["default_k"] == 4
    assert summary["passes"] == 3 and summary["accepted_drafts"] == 9 and summary["tokens_per_pass"] == 4.0
    assert extension.summary(passes=2, accepted_drafts=6)["tokens_per_pass"] == 4.0
    assert extension.summary(passes=0, accepted_drafts=0)["tokens_per_pass"] is None
    # Host-drafted passes are counted per arm from the record's source.
    extension.record(SimpleNamespace(accepted=2, source="host", sampled=False))
    assert extension.arms[7].host_passes == 1 and extension.host_drafted_passes == 1
    assert summary["draft_source"] == "mtp" and extension.summary()["host_drafted_passes"] == 1
    assert extension.summary(passes=1, accepted_drafts=2, host_drafted_passes=1)["host_drafted_passes"] == 1


def test_extension_merges_per_arm_capture_and_trace_records() -> None:
    extension = _extension(4, 7)
    extension.arms[4].capture_ms["draft"] = 1.5
    extension.arms[7].capture_ms["draft"] = 2.5
    extension.arms[4].trace_dram_bytes_per_bank["mtp_traces"] = 100
    extension.arms[7].trace_dram_bytes_per_bank["mtp_traces"] = 200
    extension.context_trace_dram_bytes_per_bank = {"decode_traces": 10, "chunk_trace": 20}
    assert extension.capture_ms == {"draft@k4": 1.5, "draft@k7": 2.5}
    assert extension.trace_dram_bytes_per_bank == {
        "decode_traces": 10,
        "chunk_trace": 20,
        "mtp_traces@k4": 100,
        "mtp_traces@k7": 200,
    }


def test_extension_construction_is_fail_closed() -> None:
    generic = object()
    with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
        session_module.Qwen38ChainMTP(
            arms={}, default_drafts=4, anchor="off", components=None, step_inputs=None, chunk_extension=None
        )
    with pytest.raises(ValueError):  # allow-pytest.raises: keyed by another count
        session_module.Qwen38ChainMTP(
            arms={5: _arm(4, generic)}, default_drafts=5, anchor="off", components=None, step_inputs=None,
            chunk_extension=None,
        )
    with pytest.raises(ValueError):  # allow-pytest.raises: the default is not an arm
        _extension(4, 7, default=5)
    with pytest.raises(ValueError):  # allow-pytest.raises: an unknown draft source
        session_module.Qwen38ChainMTP(
            arms={4: _arm(4, generic)}, default_drafts=4, anchor="off", components=None, step_inputs=None,
            chunk_extension=None, draft_source="lookahead",
        )
    assert session_module.DRAFT_SOURCES == ("mtp", "hybrid", "ngram")
    with pytest.raises(ValueError):  # allow-pytest.raises: two MTP layer caches
        session_module.Qwen38ChainMTP(
            arms={4: _arm(4, object()), 7: _arm(7, object())}, default_drafts=4, anchor="off", components=None,
            step_inputs=None, chunk_extension=None,
        )


# --------------------------------------------------------------------------- the admission with arms


def test_admission_adds_the_arm_bound_per_further_arm() -> None:
    one = session_module.mtp_capacity_admission(32768)
    two = session_module.mtp_capacity_admission(32768, arms=2)
    assert one["arms"] == 1 and two["arms"] == 2
    assert (
        two["required_free_bytes_per_bank"] - one["required_free_bytes_per_bank"]
        == session_module.MTP_ARM_BYTES_PER_BANK_UPPER_BOUND
        == two["mtp_arm_bytes_per_bank_upper_bound"]
    )
    assert two["required_largest_contiguous_bytes_per_bank"] == one["required_largest_contiguous_bytes_per_bank"]
    for arms in (0, -1, True, 1.0):
        with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
            session_module.mtp_capacity_admission(32768, arms=arms)


# --------------------------------------------------------------------------- the shared alignment generic state


def test_verify_state_alignment_ownership_contract() -> None:
    assert inspect.signature(mtp_v2.allocate_verify_state).parameters["alignment_generic_state"].default is None
    fields = {field.name: field for field in mtp_v2.Qwen38TTNNVerifyAlignment.__dataclass_fields__.values()}
    assert fields["owns_generic_state"].default is True
    allocate = inspect.getsource(mtp_v2.allocate_verify_state)
    assert "owns_generic_state=alignment_generic_state is None" in allocate
    assert "generic_state = alignment_generic_state" in allocate
    release = inspect.getsource(mtp_v2.release_verify_state)
    assert "if alignment.owns_generic_state:" in release
    # The session's open shares the first arm's generic state with every further arm and captures per arm.
    source = inspect.getsource(session_module.Qwen38TracedChain.open)
    assert "alignment_generic_state=shared_generic_state" in source
    assert "for arm in (chain_mtp.arms[k] for k in mtp_arms):" in source
    assert "chain_mtp.select(arm.drafts)" in source and "chain_mtp.select(None)" in source

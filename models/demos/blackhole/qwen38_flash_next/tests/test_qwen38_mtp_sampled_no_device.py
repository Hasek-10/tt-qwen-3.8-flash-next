# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sampled requests through the MTP pass loop, no device: the pass loop's host sampler and its admission, the
extension's counters, the readback round trip, the warm-pass check, and the source contracts of the verify body,
the LM head's rows epilogue and the session's routing."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step5_verify_no_device import _replicated, make_verify_fake
from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step4_rows_no_device import FP32, ROW_MAJOR, FakeChunk
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as sampling_step
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import sampling as sampling_module

ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = embedding_module.VOCAB_SIZE


def _request(seed: int = 5, **fields) -> sampling_step.Qwen38SamplingRequest:
    parameters = sampling_step.parameters_from_request(
        {"temperature": 0.8, "top_k": 8, "top_p": 0.9, "seed": seed, **fields}, enable_thinking=False, seed=seed
    )
    return sampling_step.Qwen38SamplingRequest(parameters)


def _row(head: int, *, peaks: int = 12, tail: float = -100.0) -> tuple[sampling_module.Qwen38CandidateRow, torch.Tensor]:
    logits = torch.full((VOCAB_SIZE,), tail)
    for i in range(peaks):
        logits[head + i] = 12.0 - i
    bf16 = logits.to(torch.bfloat16)
    return sampling_module.Qwen38CandidateRow.emulate(bf16), bf16.float()


# --------------------------------------------------------------------------- the admission and the sampler


def test_pass_loop_admits_the_policies_the_candidate_rows_sample_exactly() -> None:
    assert sampling_step.mtp_pass_loop_admits(_request())
    assert not sampling_step.mtp_pass_loop_admits(_request(top_k=0))  # the nucleus may leave the rows
    assert not sampling_step.mtp_pass_loop_admits(_request(presence_penalty=-0.5))  # a penalty that raises logits
    assert not sampling_step.mtp_pass_loop_admits(_request(repetition_penalty=0.9))
    assert sampling_step.mtp_pass_loop_admits(_request(presence_penalty=1.5, repetition_penalty=1.1))
    with pytest.raises(TypeError):  # allow-pytest.raises: the request type
        sampling_step.mtp_pass_loop_admits(object())


def test_mtp_sampler_applies_the_policy_over_the_committed_prefix_and_the_pass_rows() -> None:
    """Row j's history is the session's committed tokens, then t_P, then the rows chosen so far; the draws are the
    request's generator in row order (the same tokens as sample_candidates with that history and generator)."""

    request = _request(seed=5)
    session = SimpleNamespace(committed=[1, 2, 3])
    sampler = sampling_step.Qwen38MTPSampler(session, request, prompt_tokens=3)
    row0, _ = _row(10)
    row1, _ = _row(30)
    reference = torch.Generator(device="cpu").manual_seed(5)
    expected0 = sampling_module.sample_candidates(
        row0, request.parameters, token_history=[1, 2, 3, 9], prompt_tokens=3, generator=reference
    )
    token0 = sampler.choose(0, row0, (9, 4, 5), (), full_logits=lambda: pytest.fail("no fallback"))
    assert token0 == expected0.token_id and sampler.samples[0].token_id == token0
    expected1 = sampling_module.sample_candidates(
        row1, request.parameters, token_history=[1, 2, 3, 9, token0], prompt_tokens=3, generator=reference
    )
    token1 = sampler.choose(1, row1, (9, 4, 5), (token0,), full_logits=lambda: pytest.fail("no fallback"))
    assert token1 == expected1.token_id and [s.token_id for s in sampler.samples] == [token0, token1]
    assert sampler.rows_sampled == 2 and request.clocks.fallbacks == 0
    assert torch.equal(request.generator.get_state(), reference.get_state())
    # Row 0 starts a new pass's samples; a row out of order is refused.
    sampler.choose(0, row0, (7, 1, 1), (), full_logits=lambda: pytest.fail("no fallback"))
    assert len(sampler.samples) == 1
    with pytest.raises(RuntimeError):  # allow-pytest.raises: rows are sampled in order
        sampler.choose(2, row1, (7, 1, 1), (token0,), full_logits=lambda: pytest.fail("no fallback"))
    with pytest.raises(TypeError):  # allow-pytest.raises: the request type
        sampling_step.Qwen38MTPSampler(session, object(), prompt_tokens=3)
    with pytest.raises(ValueError):  # allow-pytest.raises: prompt_tokens
        sampling_step.Qwen38MTPSampler(session, request, prompt_tokens=-1)


def test_mtp_sampler_falls_back_to_the_rows_full_logits_on_the_candidate_guard() -> None:
    request = _request(seed=9)
    session = SimpleNamespace(committed=[4, 4])
    sampler = sampling_step.Qwen38MTPSampler(session, request, prompt_tokens=2)
    flat = torch.zeros((VOCAB_SIZE,))
    flat[17] = 4.0
    row = sampling_module.Qwen38CandidateRow.emulate(flat.to(torch.bfloat16))
    calls = []

    def full_logits():
        calls.append(True)
        return flat.to(torch.bfloat16).float()

    reference = torch.Generator(device="cpu").manual_seed(9)
    expected = sampling_module.sample_full_vocabulary(
        flat.to(torch.bfloat16).float(), request.parameters, token_history=[4, 4, 8], prompt_tokens=2, generator=reference
    )
    token = sampler.choose(0, row, (8, 0, 0), (), full_logits=full_logits)
    assert token == expected.token_id and calls == [True] and request.clocks.fallbacks == 1
    assert torch.equal(request.generator.get_state(), reference.get_state())


# --------------------------------------------------------------------------- the extension's admission and counters


def _extension(*, sampling):
    alignment = SimpleNamespace(generic_state=object(), layer=object(), verify_state=object(), residual=None)
    verify = SimpleNamespace(drafts=4, rows=5, alignment=alignment, sampling=sampling)
    arm = session_module.Qwen38ChainMTPArm(drafts=4, verify=verify, draft=SimpleNamespace(drafts=4))
    return session_module.Qwen38ChainMTP(
        arms={4: arm}, default_drafts=4, anchor="off", components=object(), step_inputs=SimpleNamespace(), chunk_extension=None
    )


def test_extension_admits_sampled_requests_with_the_buffers_and_counts_sampled_passes() -> None:
    assert not _extension(sampling=None).sampling_admitted(_request())
    extension = _extension(sampling=object())
    assert extension.sampling_admitted(_request()) and not extension.sampling_admitted(_request(top_k=0))
    extension.record(SimpleNamespace(accepted=2, source="device", sampled=True))
    extension.record(SimpleNamespace(accepted=1, source="host", sampled=False))
    assert extension.sampled_passes == 1 and extension.arms[4].sampled_passes == 1
    summary = extension.summary()
    assert summary["sampled_passes"] == 1 and summary["passes"] == 2 and summary["host_drafted_passes"] == 1
    assert extension.summary(passes=1, accepted_drafts=2, host_drafted_passes=0, sampled_passes=1)["sampled_passes"] == 1


# --------------------------------------------------------------------------- the readback round trip and the warm check


@pytest.fixture
def fake(monkeypatch):
    fake_ttnn = make_verify_fake(FakeChunk())
    for module in (embedding_module, contracts_module, mtp_v2):
        monkeypatch.setattr(module, "ttnn", fake_ttnn)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    return fake_ttnn


def test_readback_row_written_by_the_host_reads_back_as_the_device_form(fake) -> None:
    output = mtp_v2.Qwen38TTNNVerifyOutput(_replicated(torch.zeros(1, 1, 1, mtp_v2.READBACK_WIDTH), FP32, ROW_MAJOR))
    readback = mtp_v2.Qwen38TTNNVerifyReadback(2, 77, 5, (11, 12, 77, 14, 15))
    mtp_v2.write_verify_readback(SimpleNamespace(mesh_device="mesh"), output, readback)
    assert mtp_v2.read_verify_output(output, rows=5) == readback
    lanes = output.readback.torch_shards()[0].reshape(-1)
    assert lanes[8:].tolist() == [-1.0] * (mtp_v2.READBACK_WIDTH - 8)
    output.release_tensors()
    with pytest.raises(RuntimeError):  # allow-pytest.raises: a released readback
        mtp_v2.write_verify_readback(SimpleNamespace(mesh_device="mesh"), output, readback)


def test_candidate_rows_and_alignment_lanes_parse_and_fail_closed(fake) -> None:
    rows = [_row(10 * (index + 1))[0] for index in range(4)]
    stacked = torch.stack([row.to_host_row().reshape(-1) for row in rows]).reshape(1, 1, 4, -1)
    sampling = SimpleNamespace(
        rows=4,
        candidate_rows=_replicated(stacked, FP32, ROW_MAJOR),
        alignment_lanes=_replicated(torch.tensor([[[[3.0, 4.0, 5.0, 6.0] + [-1.0] * 28]]]), FP32, ROW_MAJOR),
    )
    verify = SimpleNamespace(rows=4, drafts=3, sampling=sampling)
    parsed = mtp_v2.read_verify_candidates(verify)
    assert len(parsed) == 4 and all(torch.equal(a.ids, b.ids) and torch.equal(a.values, b.values) for a, b in zip(parsed, rows))
    assert mtp_v2.read_alignment_lanes(verify) == (3, 4, 5, 6)
    with pytest.raises(RuntimeError):  # allow-pytest.raises: no buffers
        mtp_v2.read_verify_candidates(SimpleNamespace(rows=4, sampling=None))
    bad = SimpleNamespace(rows=4, drafts=3, sampling=SimpleNamespace(rows=4, alignment_lanes=_replicated(torch.full((1, 1, 1, 32), -1.0), FP32, ROW_MAJOR)))
    with pytest.raises(RuntimeError):  # allow-pytest.raises: a real row must resolve an id
        mtp_v2.read_alignment_lanes(bad)


def test_warm_pass_check_ties_the_landed_rows_to_the_pass_resolve() -> None:
    rows = [_row(10 * (index + 1))[0] for index in range(3)]
    readback = mtp_v2.Qwen38TTNNVerifyReadback(1, 20, 41, (10, 20, 30))
    session_module._check_verify_candidates(readback, rows, (40, 41, 42), label="round")
    with pytest.raises(session_module.Qwen38ChatChainError):  # allow-pytest.raises: an argmax off its row's maximum
        session_module._check_verify_candidates(mtp_v2.Qwen38TTNNVerifyReadback(1, 20, 41, (11, 20, 30)), rows, (40, 41, 42), label="round")
    with pytest.raises(session_module.Qwen38ChatChainError):  # allow-pytest.raises: the alignment lane vs the first draft
        session_module._check_verify_candidates(readback, rows, (40, 99, 42), label="round")
    with pytest.raises(session_module.Qwen38ChatChainError):  # allow-pytest.raises: row counts
        session_module._check_verify_candidates(readback, rows[:2], (40, 41, 42), label="round")


# --------------------------------------------------------------------------- source contracts


def _calls(node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            names.append(ast.unparse(child.func))
    return names


def _functions(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(node.name, node)
    return found


def test_the_rows_epilogue_is_the_single_row_one_with_a_row_axis() -> None:
    functions = _functions(ROOT / "ttnn" / "embedding.py")
    single = [name for name in _calls(functions["sampling_candidates"]) if name.startswith("ttnn.")]
    rows = [name for name in _calls(functions["sampling_candidate_rows"]) if name.startswith("ttnn.")]
    assert rows == single, (single, rows)  # the same device ops in the same order; the copy lands in `into`
    source = ast.get_source_segment((ROOT / "ttnn" / "embedding.py").read_text(encoding="utf-8"), functions["sampling_candidate_rows"])
    assert "ttnn.copy(row, into)" in source and "constants.readback_row" not in source
    for forbidden in ("ttnn.from_torch", "ttnn.to_torch", "torch."):
        assert forbidden not in source, forbidden


def test_verify_body_lands_the_sampled_rows_before_the_position_update_and_the_state_owns_them() -> None:
    path = ROOT / "ttnn" / "mtp_v2.py"
    source = path.read_text(encoding="utf-8")
    functions = _functions(path)
    resolve = ast.get_source_segment(source, functions["_resolve_rows"])
    assert "lm_head.sampling_candidate_rows(logits, sampling.constants, into=sampling.candidate_rows)" in resolve
    assert resolve.index("sampling_candidate_rows(") < resolve.index("_deallocate(logits.tensor)")
    assert '_land(logits.tensor, sampling.logits_rows, label="verify logits rows")' in resolve
    body = ast.get_source_segment(source, functions["forward_verify"])
    assert "sampling=verify.sampling" in body and body.index("sampling=verify.sampling") < body.index("accept_rows(")
    alignment = ast.get_source_segment(source, functions["_forward_alignment"])
    assert alignment.index("_land(lanes, verify.sampling.alignment_lanes") < alignment.index("ttnn.gather(lanes, 3, accept.accepted_index")
    assert alignment.index("_land(residual, verify.sampling.residual_rows") < alignment.index("select_residual_row(")
    for name in ("_resolve_rows", "_forward_alignment", "forward_verify"):
        for called in _calls(functions[name]):
            assert not called.startswith(("ttnn.from_torch", "ttnn.to_torch", "ttnn.copy_host_to_device_tensor", "torch.")), (name, called)
    allocate = ast.get_source_segment(source, functions["allocate_verify_state"])
    for label in ("verify candidate rows", "verify alignment lanes", "verify alignment residual rows", "verify logits rows"):
        assert f'label="{label}"' in allocate, label
    assert "sampling=sampling," in allocate and "mtp_components is None" in allocate
    assert "verify.sampling.deallocate" in ast.get_source_segment(source, functions["release_verify_state"])
    assert "verify.sampling" in ast.get_source_segment(source, functions["_validate_verify_state"])
    accept = ast.get_source_segment(source, functions["apply_host_accept"])
    assert accept.index("write_verify_accepted(") < accept.index("state.position.reset(") < accept.index("write_verify_readback(") < accept.index("select_residual_row(")
    sample = ast.get_source_segment(source, functions["_sample_pass"])
    assert "if index < self.verify.drafts and drafts[index] == token:" in sample and "first_draft=alignment[accepted]" in sample
    assert "position=self.position + accepted + 1" in sample
    for name in ("apply_host_accept", "Qwen38TTNNVerifySampling", "read_alignment_lanes", "read_verify_candidates", "read_verify_logits_row", "write_verify_readback"):
        assert name in mtp_v2.__all__, name
    signature = inspect.signature(mtp_v2.Qwen38TTNNMTPChain.__init__)
    assert signature.parameters["sampler"].default is None and signature.parameters["state"].default is None
    fields = mtp_v2.Qwen38TTNNMTPPassRecord.__dataclass_fields__
    assert fields["chosen"].default == () and fields["sampled"].default is False


def test_session_routes_admitted_sampled_requests_through_the_pass_loop() -> None:
    generate = inspect.getsource(session_module.Qwen38ChatSession.complete)
    assert "(sampling is None or self.mtp.sampling_admitted(sampling))" in generate
    assert "self.sampling.begin_request(None if drafting else sampling)" in generate
    assert "sampled_passes=self.mtp.sampled_passes - mtp_before[3]" in generate
    assert "self._generate_mtp(max_tokens, stop_ids, think_budget, should_stop, sampling=sampling)" in generate
    loop = inspect.getsource(session_module.Qwen38ChatSession._generate_mtp)
    assert "pending = next_pending()" in loop and "self.sampling.read_candidate_row()" in loop
    assert "emitted = [int(token) for token in record.chosen]" in loop and "record.argmaxes[" not in loop
    assert loop.index("sampling.samples.append(sampler.samples[index])") < loop.index("yield token_id, None")
    assert loop.index("sampling.samples.append(pending_sample)") < loop.index("yield pending, None")
    assert "sampling_step.generate_sampled(" in loop and "history=self.committed, sampler=sampler)" in loop
    assert loop.count("sampling.samples.append(None)") == 2  # the two forced </think> yields
    enter = inspect.getsource(session_module.Qwen38TracedChain.mtp_enter)
    assert "state=self.state," in enter and "sampler=sampler," in enter
    opened = inspect.getsource(session_module.Qwen38TracedChain.open)
    assert "sampling_constants=None if sampling_extension is None else sampling_extension.constants," in opened
    assert opened.index("sampling_extension = (") < opened.index("chain_mtp = None")
    assert "sampling=sampling_extension," in opened and "_check_verify_candidates(" in opened
    for name in ("candidate_rows", "alignment_lanes", "residual_rows", "logits_rows"):
        assert f"verify.sampling.{name}," in opened, name

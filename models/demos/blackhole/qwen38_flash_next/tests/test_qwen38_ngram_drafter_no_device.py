# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The host's prompt-lookup drafter (``ttnn/ngram_draft.py``), no device: the lookup rule, its incremental index
against a from-scratch scan, padding and declining, and the constructor's contract."""

from __future__ import annotations

import random

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import ngram_draft
from models.demos.blackhole.qwen38_flash_next.ttnn.ngram_draft import Qwen38NgramDrafter


def _scan(history: list[int], next_token: int, *, drafts: int, min_n: int, max_n: int):
    """The rule from scratch: the longest n-gram ending in next_token that occurs earlier, its most recent
    occurrence that something follows, the tokens after it (next_token itself when it ran to the present)."""

    source = history + [next_token]
    for n in range(max_n, min_n - 1, -1):
        if len(source) < n + 1:
            continue
        gram = source[-n:]
        for end in range(len(source) - 1, n - 1, -1):  # gram ending at `end` (exclusive), strictly before the end
            if source[end - n : end] == gram:
                continuation = source[end : end + drafts]
                if continuation:
                    return n, tuple(continuation)
    return None


def test_lookup_takes_the_longest_recurring_gram_and_copies_what_followed_it_most_recently() -> None:
    drafter = Qwen38NgramDrafter(3, min_n=2, max_n=4, history=[1, 2, 3, 4, 9, 1, 2, 3, 5, 6, 7, 1, 2])
    # ... 1 2 3 [4 9 1] ... 1 2 3 [5 6 7] ... 1 2 | 3: the 3-gram (1, 2, 3) recurs; its most recent occurrence
    # was followed by 5 6 7.
    assert drafter.lookup(3) == (3, (5, 6, 7))
    assert drafter.propose(3) == (5, 6, 7)
    # (1, 2, 8) never occurred, (2, 8) neither: decline.
    assert drafter.lookup(8) is None and drafter.propose(8) is None
    # The 2-gram (2, 3) after a fresh 2: same answer through the shorter order when the 3-gram is new.
    drafter.extend([0, 2])
    assert drafter.lookup(3) == (2, (5, 6, 7))
    assert drafter.proposals == 1 and drafter.declines == 1  # lookup() counts nothing


def test_a_continuation_that_runs_to_the_present_includes_the_next_token_and_pads_with_the_fill_token() -> None:
    drafter = Qwen38NgramDrafter(4, min_n=2, max_n=3, history=[7, 8, 9, 7, 8], fill_token=0)
    # (7, 8, 9) recurs at the start; after it came 7 8, then the present 9: three real tokens, one fill.
    assert drafter.lookup(9) == (3, (7, 8, 9))
    assert drafter.propose(9) == (7, 8, 9, 0)
    # A gram whose only occurrence is the one ending at the present declines (nothing followed it yet).
    assert Qwen38NgramDrafter(2, min_n=2, max_n=2, history=[1, 2]).propose(3) is None
    # ... unless the same gram occurred earlier as well.
    assert Qwen38NgramDrafter(2, min_n=2, max_n=2, history=[2, 3, 4, 2]).propose(3) == (4, 2)


def test_host_only_mode_proposes_fill_tokens_instead_of_declining() -> None:
    drafter = Qwen38NgramDrafter(3, history=[1, 2, 3], fill_token=0, decline=False)
    assert drafter.propose(9) == (0, 0, 0) and drafter.proposals == 1 and drafter.declines == 0
    assert Qwen38NgramDrafter(3, history=[1, 2, 3, 1, 2], decline=False).propose(3) == (1, 2, 3)


@pytest.mark.parametrize("seed", range(12))
def test_the_incremental_index_agrees_with_a_from_scratch_scan_on_random_streams(seed: int) -> None:
    rng = random.Random(seed)
    vocab, drafts, min_n, max_n = rng.choice((3, 5, 12)), rng.choice((3, 4, 7)), rng.choice((1, 2, 3)), rng.choice((3, 4, 6))
    min_n = min(min_n, max_n)
    drafter = Qwen38NgramDrafter(drafts, min_n=min_n, max_n=max_n)
    history: list[int] = []
    for _ in range(300):
        token = rng.randrange(vocab)
        expected = _scan(history, token, drafts=drafts, min_n=min_n, max_n=max_n)
        assert drafter.lookup(token) == expected, (seed, len(history), token)
        proposal = drafter.propose(token)
        if expected is None:
            assert proposal is None
        else:
            _, continuation = expected
            assert proposal == (*continuation, *((0,) * (drafts - len(continuation))))
            assert len(proposal) == drafts
        committed = [token] + [rng.randrange(vocab) for _ in range(rng.randrange(drafts + 1))]
        drafter.extend(committed)
        history.extend(committed)
    assert drafter.history == tuple(history) and len(drafter) == len(history)


def test_constructor_and_inputs_fail_closed() -> None:
    for bad in dict(drafts=0), dict(drafts=True), dict(min_n=0), dict(max_n=9), dict(min_n=3, max_n=2), dict(fill_token=-1), dict(decline=1):
        kwargs = {"drafts": 4} | bad
        with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
            Qwen38NgramDrafter(**kwargs)
    with pytest.raises(ValueError):  # allow-pytest.raises: history ids are non-negative ints
        Qwen38NgramDrafter(4, history=[1, -1])
    with pytest.raises(ValueError):  # allow-pytest.raises: next_token likewise
        Qwen38NgramDrafter(4).propose(True)
    assert (ngram_draft.MIN_ORDER, ngram_draft.MAX_ORDER) == (1, 8)

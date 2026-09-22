# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prompt-lookup drafting for the MTP v2 pass loop: the k drafts of a pass from the request's own text, at no
device cost.

A pass verifies R = k + 1 rows whatever produced the k drafts and accepts the longest prefix that matches the
target's argmaxes, so the drafts change how much a pass commits and nothing else: the committed stream is the
target's, exact (``ttnn/mtp_v2.py``).  The MTP head drafts k tokens in k - 1 sequential device rows (about 4 ms
each on 4x p150).  This drafter proposes them from the host in microseconds when the last n tokens have occurred
before in the prompt or the generated text and copies what followed that occurrence (structured output, code,
quotation, summarisation that copies), and declines when they have not, so the hybrid loop falls back to the MTP
head for that pass.  Saxena's prompt lookup decoding, with the n-gram index kept incrementally.
"""

from __future__ import annotations

from collections.abc import Sequence

MIN_ORDER = 1
MAX_ORDER = 8


def _token(value, label: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative int, got {value!r}")
    return value


class Qwen38NgramDrafter:
    """``propose(next_token)``: the ``drafts`` tokens that followed the most recent earlier occurrence of the longest
    n-gram (``max_n`` down to ``min_n`` tokens) ending in ``next_token``, padded with ``fill_token`` to ``drafts``,
    or ``None`` when no n-gram of at least ``min_n`` tokens recurs (``decline``; with ``decline=False`` the fill
    tokens alone, for a host-only loop).  ``extend(tokens)`` appends the committed tokens: the history the n-grams
    are looked up in.  ``next_token`` is the token the target just predicted and the next pass's row 0; it is not
    in the history until that pass commits it."""

    def __init__(
        self,
        drafts: int,
        *,
        min_n: int = 2,
        max_n: int = 4,
        history: Sequence[int] = (),
        fill_token: int = 0,
        decline: bool = True,
    ) -> None:
        if isinstance(drafts, bool) or type(drafts) is not int or drafts < 1:
            raise ValueError(f"drafts must be a positive int, got {drafts!r}")
        for name, value in (("min_n", min_n), ("max_n", max_n)):
            if isinstance(value, bool) or type(value) is not int or not MIN_ORDER <= value <= MAX_ORDER:
                raise ValueError(f"{name} must be an int in [{MIN_ORDER}, {MAX_ORDER}], got {value!r}")
        if min_n > max_n:
            raise ValueError(f"min_n {min_n} exceeds max_n {max_n}")
        if type(decline) is not bool:
            raise ValueError(f"decline must be a bool, got {decline!r}")
        self.drafts, self.min_n, self.max_n = drafts, min_n, max_n
        self.fill_token = _token(fill_token, "fill_token")
        self.decline = decline
        self._history: list[int] = []
        # For every order n: the positions just past each occurrence of an n-gram, oldest first.
        self._index: dict[int, dict[tuple[int, ...], list[int]]] = {n: {} for n in range(min_n, max_n + 1)}
        self.proposals = 0  # passes this drafter proposed for
        self.declines = 0  # passes it declined
        self.extend(history)

    @property
    def history(self) -> tuple[int, ...]:
        return tuple(self._history)

    def __len__(self) -> int:
        return len(self._history)

    def extend(self, tokens: Sequence[int]) -> None:
        """Append committed tokens, indexing every n-gram they complete."""

        for value in tokens:
            self._history.append(_token(value, "history token"))
            end = len(self._history)
            for n in range(self.min_n, self.max_n + 1):
                if end >= n:
                    self._index[n].setdefault(tuple(self._history[end - n :]), []).append(end)

    def lookup(self, next_token: int) -> tuple[int, tuple[int, ...]] | None:
        """The longest recurring n-gram ending in ``next_token`` and the tokens that followed its most recent earlier
        occurrence (at most ``drafts``; ``next_token`` itself when the occurrence ran up to the present), or ``None``."""

        next_token = _token(next_token, "next_token")
        history = self._history
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(history) + 1 < n + 1:  # the n-gram and at least one earlier token for it to recur in
                continue
            gram = (*history[len(history) - (n - 1) :], next_token) if n > 1 else (next_token,)
            ends = self._index[n].get(gram)
            if not ends:
                continue
            end = ends[-1]  # the most recent occurrence; at len(history) it is followed by next_token itself
            continuation = history[end : end + self.drafts]
            if len(continuation) < self.drafts and end + len(continuation) == len(history):
                continuation = [*continuation, next_token]
            return n, tuple(continuation[: self.drafts])
        return None

    def propose(self, next_token: int) -> tuple[int, ...] | None:
        found = self.lookup(next_token)
        if found is None:
            if self.decline:
                self.declines += 1
                return None
            self.proposals += 1
            return (self.fill_token,) * self.drafts
        self.proposals += 1
        _, continuation = found
        return (*continuation, *((self.fill_token,) * (self.drafts - len(continuation))))


__all__ = ["MAX_ORDER", "MIN_ORDER", "Qwen38NgramDrafter"]

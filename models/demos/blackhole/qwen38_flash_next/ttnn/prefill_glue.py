# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The prefill slab's glue forms and the ``QWEN38_PREFILL_GLUE`` switch.

A glue form is another arrangement of existing ttnn ops for one term of the 2048-row slab body (``--prefill-slab``):
a collective's configuration, a per-slab hoist, a payload cut.  Every form is read only by a slab branch (under
``is_slab_rows``), so the decode path and the 32/128-row chunk bodies never see one.  The bitwise forms that measured
identical to the previous slab body run by default (``DEFAULT_ON``: the generic gather of the gated-residual partials
and the per-slab QSA block-mask hoist; 2026-09-25 on the 4-chip line, a 32k context: pins 12/12, the oracle column
identical at every scored position, 60.3 ms less per 2048-row slab; the gated-residual read's tile walks, row gamma
and gate activation, 2026-10-14, 65.2 ms less; its front as two fused programs, 2026-10-14, 14.3 ms less; its
write as one fused program, 2026-10-14, 6.9 ms less), and so does a tolerance form that met the prefill quality bar
(the gated-residual partials' reduce-scatter + all-gather, 2026-10-14, 14.9 ms less).  ``QWEN38_PREFILL_GLUE=today``
restores the previous slab body, and ``QWEN38_PREFILL_GLUE=<name>[,<name>...]`` runs exactly the named forms (the
measurement's arms: a list that wants a default form names it).  A name that is not registered raises, as does an
exclusive pair or ``today`` beside a name.  The model resolves the policy once at construction (``policy()``) and
keeps it through trace capture.

Classes.  BITWISE: the same per-element arithmetic and reduction order as the previous form (data movement, integer
work, or the same ops once instead of per layer); the slab's outputs are the previous form's bit for bit.  TOLERANCE:
a summation order or a rounding point moves (a reduce-scatter summing in hop order in place of a device-order reduce,
a normalization moved into a kernel), judged by the long-window KL / top-1 against the CPU oracle and, for a
discrete choice, by the selected block sets.  A TOLERANCE form may be in ``DEFAULT_ON`` only once it has met the
prefill quality bar (``docs/PREFILL.md``) and ``MET_QUALITY_BAR`` names it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

ENV = "QWEN38_PREFILL_GLUE"
# The spelling that runs no form: the slab body as it was before the forms.
TODAY = "today"
BITWISE, TOLERANCE = "bitwise", "tolerance"
# name -> (class, the slab branch that reads it)
FORMS: dict[str, tuple[str, str]] = {
    # The gated-residual read's partial gather (fp32 [rows, 384] on dim 0) as the generic all_gather (a default), or
    # as the async op with the MoE reduce-scatter's worker / sync settings; the gathered bytes and page order are the
    # same as the previous form's (the async op at one worker per link and one chunk per sync: ``today``).
    "gr_gather_generic": (BITWISE, "gr.read_rows"),
    "gr_gather_tuned": (BITWISE, "gr.read_rows"),
    # The QSA block selection's causal block masks derived once per slab (they depend on the position and the row
    # index only) and read by every QSA layer (a default).  Admitted once, with the slab's chunk constants, when the
    # masks fit HOISTED_MASK_BYTES_MAX and the free DRAM after the build holds them plus the slab's working set
    # (qsa.admit_hoisted_masks); otherwise the slab derives them per layer as before.
    "qsa_mask_hoist": (BITWISE, "qsa.slab_hoisted_mask_admission"),
    # The QSA block scores summed by a reduce-scatter over the block's rows, top-k per device, the block ids gathered
    # (in place of the all-broadcast + local sum of every device's scores).
    "qsa_scores_rs_ag": (TOLERANCE, "qsa._sparse_indices_slab"),
    # The gated-residual partials summed by a reduce-scatter + all-gather in place of the gather + device-order reduce
    # (a default: it met the quality bar).
    "gr_partial_rs_ag": (TOLERANCE, "gr.read_rows"),
    # The gated-residual read's branch-major <-> flat walks as tile moves: the RMS unit to the flat rows by one tile
    # permute (in place of four slices + a concat), and the gated flat rows summed over the branches through a
    # [row tiles, 4, 32, 640] view by the same fast_reduce_nc (in place of four slices + a concat).  Whole tiles move
    # unchanged and the reduce sums the same four tiles in the same order (a default).
    "gr_tile_walks": (BITWISE, "gr.read_rows"),
    # The gated-residual read's gamma multiply against the resident 32-row gamma broadcast over the row tiles, in
    # place of the 1-row gamma's row broadcast (whose reader fills each fp32 gamma tile's rows on the RISC); the same
    # tile operands reach the same compute (a default).
    "gr_gamma_rows": (BITWISE, "gr.read_rows"),
    # The gated-residual read's sigmoid as the gate multiply's input activation (packed to bf16 before the multiply,
    # as the separate sigmoid; the injection's form) (a default).
    "gr_gate_act": (BITWISE, "gr.read_rows"),
    # The gated-residual read's front as two fused programs around the stats gather (ttnn/fused/gr_read's slab stats
    # and normalize): every row block runs rms_norm_pre's, rms_norm_post's and the row gamma's LLK sequence on the
    # same tiles, and each normalized tile is written to its flat page (in place of rms_norm_pre, rms_norm_post, the
    # tile permute and the gamma multiply) (a default).
    "gr_slab_front": (BITWISE, "gr.read_rows"),
    # The gated-residual write as one fused program (ttnn/fused/gr_write's slab write): every output tile runs the
    # coefficient multiply's and the residual add's LLK sequence, the injection's columns patched into the broadcast
    # column by the reader (in place of the coefficient permute, the multiply and the add) (a default).
    "gr_slab_write": (BITWISE, "gr.write_rows"),
    # The GDN chunk kernel's flat q/k form: the raw conv q/k rows go to the kernel, which maps value heads to key
    # heads at read time and l2-normalizes in fp32 with the scale folded in.
    "gdn_qk_flat": (TOLERANCE, "gdn.allocate_rows_state"),
}
EXCLUSIVE: tuple[frozenset[str], ...] = (frozenset({"gr_gather_generic", "gr_gather_tuned"}),)
# The forms that run when QWEN38_PREFILL_GLUE is unset: bitwise the previous slab body on every readout taken
# (2026-09-25, the 4-chip line at 32k: acceptance pins 12/12, the corpus long windows' top-128 columns identical to the
# previous body's at all 160 positions), 1584.6 -> 1524.3 ms of kernel time per 2048-row slab (the gather -34.0, the
# hoist -26.5, additive), 1283 -> 1334 prompt tokens per second, TTFT at 32k 27.57 -> 26.64 s.  The three GR read
# forms (2026-10-14, the 4-chip line at 32k): greedy-32 and the 32 top-20 logprob rows identical to the previous body,
# GR read + write 166.8 -> 101.6 ms of kernel time per 2048-row slab, 3138 -> 3477 prompt tokens per second.  The GR
# read front (2026-10-14, the 4-chip line at 32k, against the five forms above): greedy-32 and the 32 top-20 logprob
# rows identical, GR read 79.8 -> 65.5 ms of kernel time per 2048-row slab, a 31,524-token prompt 8,687 -> 8,469 ms.
# The GR slab write (2026-10-14, the 4-chip line at 32k, against the other seven defaults): greedy-32 and the 32
# top-20 logprob rows identical, GR write 14.5 -> 7.6 ms of kernel time per 2048-row slab, a 31,524-token prompt
# 8,322 -> 8,206 ms.  gr_partial_rs_ag (tolerance, MET_QUALITY_BAR) supersedes the generic gather on the slab.
DEFAULT_ON: frozenset[str] = frozenset(
    {
        "gr_gather_generic",
        "qsa_mask_hoist",
        "gr_tile_walks",
        "gr_gamma_rows",
        "gr_gate_act",
        "gr_slab_front",
        "gr_slab_write",
        "gr_partial_rs_ag",
    }
)
# The tolerance forms that met the prefill quality bar, the only tolerance forms DEFAULT_ON may name (docs/PREFILL.md,
# "Glue forms": the reference corpus's long windows through the slab against the CPU oracle, and the measured result).
MET_QUALITY_BAR: frozenset[str] = frozenset({"gr_partial_rs_ag"})


def _check_default_on() -> None:
    """Every default form is registered and BITWISE, or TOLERANCE and in ``MET_QUALITY_BAR``; any other tolerance
    form never runs unnamed."""

    unknown = sorted(DEFAULT_ON - set(FORMS))
    if unknown:
        raise ValueError(f"DEFAULT_ON names unregistered prefill glue forms {unknown}; registered: {sorted(FORMS)}")
    unproven = sorted(name for name in DEFAULT_ON if FORMS[name][0] != BITWISE and name not in MET_QUALITY_BAR)
    if unproven:
        raise ValueError(
            f"DEFAULT_ON prefill glue forms must be bitwise or have met the prefill quality bar (MET_QUALITY_BAR); "
            f"{unproven} are neither"
        )
    for pair in EXCLUSIVE:
        if pair <= DEFAULT_ON:
            raise ValueError(f"DEFAULT_ON names the exclusive prefill glue forms {sorted(pair)}")


_check_default_on()


@dataclass(frozen=True)
class PrefillGluePolicy:
    """The forms that run, resolved once from the environment."""

    names: frozenset[str]

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] = os.environ) -> "PrefillGluePolicy":
        """Unset or empty: ``DEFAULT_ON``; ``today``: no form; a name list: exactly those forms."""

        raw = environ.get(ENV, "")
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if not tokens:
            return cls(DEFAULT_ON)
        if TODAY in tokens:
            if set(tokens) != {TODAY}:
                raise ValueError(
                    f"{ENV}={raw!r}: {TODAY!r} names the previous slab body and stands alone; to run some forms, "
                    f"list them (the defaults are {sorted(DEFAULT_ON)})"
                )
            return cls(frozenset())
        unknown = sorted(set(tokens) - set(FORMS))
        if unknown:
            raise ValueError(
                f"{ENV} names unregistered prefill glue forms {unknown}; registered: {sorted(FORMS)} or {TODAY!r}"
            )
        names = frozenset(tokens)
        for pair in EXCLUSIVE:
            if pair <= names:
                raise ValueError(f"{ENV} names the exclusive forms {sorted(pair)}: pick one")
        return cls(names)

    def enabled(self, name: str) -> bool:
        if name not in FORMS:
            raise KeyError(f"no prefill glue form {name!r}; registered: {sorted(FORMS)}")
        return name in self.names


@lru_cache(maxsize=1)
def policy() -> PrefillGluePolicy:
    """The process's policy, read from the environment on first use and kept."""

    return PrefillGluePolicy.from_environ()

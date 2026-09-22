# Performance work for the QuietBox 2 (branch `perf/flash-next-qb2`)

Everything here was prepared without Blackhole hardware: the changes are validated by the no-device suite and
their effect is to be measured on the QuietBox 2. Numbers quoted from the tree are the author's, 4x p150.

## 1. Where a decode token's 36.9 ms goes (the tree's own census, 2026-09-16)

| | measured | source |
|---|---|---|
| bandwidth floor: ~6.0 B active parameters (2.6 B experts bf4, ~3.4 B dense, the LM head), 5-8 GB per token | **3-4 ms** | HF config; 1,884 GB/s effective |
| kernel time, 3,370 programs per token | 34.5 ms of a 38.7 ms step | `docs/NUMERICS.md` |
| the composite GDN decode step: 49 programs, 213 us occupancy per layer x 36 GDN layers | ~7.7 ms, ~1,760 programs (52%) | `ttnn/fused/gdn_step` docstring |
| draft row of the MTP pass (MTP layer in its 32-row verify form + LM head + resolve) | ~4 ms per row | `--mtp 3` -> `--mtp 4`: 66.7 -> 70.7 ms per pass |
| verify body (R rows on the 32-row chunk operands, flat in R) | ~55-59 ms per pass | pass time less K x 4 ms |

The step is program-count-bound, not bandwidth-bound: the same 37 ms the dense 27B needs while moving four times
the bytes. Fusion is at diminishing returns (4,618 -> 3,370 programs bought 3.4 ms).

## 2. What this branch changes: MTP draft length 4 -> 7

The server admitted `--mtp 3|4`; the library admitted k = 5 but routed its 6-row verify MoE onto the 32-row
prefill form (`moe_rows_for`: "32 for k = 5 until rows 6 is admitted"). The verify pass runs on 32-row tiles
whatever R is, so verify cost is flat in k and only the k - 1 draft rows scale. The change:

| file | change |
|---|---|
| `ttnn/qsa.py` | `VERIFY_MAX_ROWS` 6 -> 8. A pass of R rows at P % 4 = r completes `(r + R) // 4` compressed blocks, at most `(R + 3) // 4` = 2 for every R <= 8, so `VERIFY_COMPLETED_BLOCKS` stays 2 and the pool select / block writes are unchanged; an import-time assert pins the invariant. `MAX_SPECULATIVE_STEPS` (the v1 fixed-five engine's reuse distance) is untouched: v2 never reads it. |
| `ttnn/moe.py` | rows 6, 7, 8 admitted (`TARGET_VERIFIER_ROW_COUNTS`), on the rows-5 code path: one `moe_compute` call of `rows` tokens, `output_height_shard_dim` 1. `ROWS6TO8_HARDWARE_PROVEN = False` until the first acceptance replay at k >= 5. |
| `ttnn/mtp_v2.py` | `SUPPORTED_DRAFTS = (3, 4, 5, 6, 7)`; `moe_rows_for` returns the exact 6/7/8. |
| `tools/qwen38_chat_session.py`, `tools/qwen38_chat_server.py`, `tools/run_qwen38_chat_server.sh` | `--mtp 3..7`. |
| tests | rows 7 and 8 / k 6 and 7 added to every verify, accept and draft parametrization; the pinned constants replaced by the invariants. |

No buffer shape changes: `SPARSE_INDEX_CAPACITY` stays 2,080, the verify and draft states are 32-row tiles.

### Expected effect (from the tree's pass costs; measure, do not trust)

Pass time T(k) ~= 54.7 + 4k ms. Tokens per pass = 1 + accepted drafts.

| prompt class | accepted per draft (from the pinned tables) | k = 4 | k = 5 | k = 7 |
|---|---|---|---|---|
| `json` (structured) | ~0.95 | 4.8 tok / 70.7 ms = **68 tok/s** | ~5.6 / 74.7 = ~75 | ~7.0 / 82.7 = ~85 |
| median acceptance prompt | ~0.68 | 2.76 / 70.7 = **39 tok/s** | ~2.9 / 74.7 = ~39 | ~3.1 / 82.7 = ~37 |

k = 5 is never worse than 4; k = 7 pays on structured output and costs 5-10% on chat. That asymmetry is why the
next step is per-request k (section 4), not a larger default.

## 3. Hardware validation order (QuietBox 2)

1. `--profile qb2 --acceptance` with no `--mtp`: pins the route; `json` 96/96.
2. `--mtp 4 --acceptance`: the pinned MTP divergence table (`tools/ci/baselines/A3-mtp4-32k-*`) must reproduce.
3. `--mtp 5`, `--mtp 6`, `--mtp 7`, each with `--acceptance`: the first silicon run of the 6/7/8-row MoE form and the
   7/8-row verify. Gate: `json` 96/96 and the eleven divergence indices unchanged from step 2 (the target's greedy
   stream does not depend on k; only pass timing does). Then flip `ROWS6TO8_HARDWARE_PROVEN`.
4. Timing at each k on the twelve acceptance prompts: pass time and tokens per pass. Compare with the table above.
5. The three opt-in fused kernels, one A/B: `QWEN38_FUSED=final_mixer,gdn_step,position_advance` in the launcher's
   environment (it `exec`s the server with the caller's environment; nothing to add). `gdn_step` replaces the
   49-program GDN chain with one program per layer and is off by default only because its rounding follows the
   torch oracle rather than the chain (tolerance class COMPONENT): the acceptance replay is the decision.

## 4. Next levers, in order, with the touch points

1. **Per-request draft length.** One `mtp` extension per admitted k allocated at open (`self.mtp` is a single
   extension today: verify + draft states and three traces, ~8 MB at 32k), `mtp_enter(first_token, ple_context,
   drafts=k)` selecting one, `--mtp 4,7`, and a request-level choice: `response_format` / `tools` present -> the
   larger k, else the smaller. Admission: the per-bank bound in `MTP_CHAIN_BYTES_PER_BANK_UPPER_BOUND` times the
   number of chains. Turns the k = 7 gain into a pure win.
2. **A cheaper draft row.** Each draft row runs the MTP layer in its 32-row verify form on one real row, the 1-row
   MoE, the final mixer, the full LM head (0.64 GB bf8) and the on-device resolve: ~4 ms. The draft's numerics do
   not affect losslessness (verify decides), so a bf4 LM head for drafting alone (90 MB per device; 3.8 GB per
   device is free at 32k) or a 1-row decode form of the MTP layer are both legal. Measure the row first with the
   `draft:*` observer stages.
3. **`gdn_step` on by default** if step 3.5 passes: ~1,760 programs and up to ~7 ms per token.
4. **Batched decode**: the largest lever (under 10% of roofline, B=8 is ~5-7x aggregate) and a redesign of the
   traced chain, sampler and server. Not before the numbers above exist.

Not levers: the n-gram lookup (`refresh_ple_row` computes the row on the host from the token it already read back and
does one host-to-device copy; no extra round trip) and GDN state precision (fp32 by construction).

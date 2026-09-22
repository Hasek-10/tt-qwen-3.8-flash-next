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

1. **Per-request draft length — on this branch.** `--mtp 4,7` opens one arm per draft count (the rows-dependent
   verify / draft states and three traces per arm; the MTP layer's caches and the TAIL step inputs shared through
   `allocate_verify_state(alignment_generic_state=...)`), the first the default; a request's `speculative_drafts`
   picks an arm, a request with tools takes the largest (`choose_speculative_drafts`), `/health.mtp` lists `arms`
   and `default_k`, the response's `qwen38.mtp.k` is the arm used. Admission adds `MTP_ARM_BYTES_PER_BANK_UPPER_BOUND`
   (12 MB per bank) per further arm and the open measures the growth against it; the warm pass runs every residue
   per arm. Unrun on silicon: the first `--mtp 4,7 --acceptance` start is its proof.
2. **A cheaper draft row — the host half is on this branch.** `--draft-source hybrid` runs the pass loop in a
   host-first form (`ttnn/mtp_v2.py`, `Qwen38TTNNMTPChain(host_drafter=...)`): after the verify row is read, a
   prompt-lookup drafter (`ttnn/ngram_draft.py`: the tokens that followed the most recent earlier occurrence of the
   last 2..4 tokens) proposes the next pass's k ids in microseconds and the k - 1 device draft rows (~4 ms each)
   are skipped; when nothing recurs, the device draft trace replays as before. Lossless by construction (the
   accept rule reads only the target's argmaxes). Expected: structured output, code and copy-heavy replies at
   close to verify-only pass time (~55 ms for up to k + 1 tokens); chat unchanged. `ngram` is the host-alone A/B
   arm. Still open on the device side: a bf4 LM head for the MTP draft (90 MB per device) or a 1-row decode form
   of the MTP layer for the fallback rows.
3. **`gdn_step` on by default** if step 3.5 passes: ~1,760 programs and up to ~7 ms per token.
4. **Batched decode**: the largest lever (under 10% of roofline, B=8 is ~5-7x aggregate) and a redesign of the
   traced chain, sampler and server. Not before the numbers above exist.

Not levers: the n-gram lookup (`refresh_ple_row` computes the row on the host from the token it already read back and
does one host-to-device copy; no extra round trip) and GDN state precision (fp32 by construction).

## 5. The step-back audit (2026-09-22): what was missed, ranked

| # | Lever | Evidence in the tree | Expected effect | Status |
|---|---|---|---|---|
| 1 | `--mtp` forced the slowest prefill | the driver ran the MTP chunk extension only for 32-row chunks, so MTP servers prefilled at 3.3 ms/token, not the slab's 0.87 | TTFT 3.8x in the mode that matters (a 4k prompt: 13.5 s -> 3.6 s) | **on this branch** (below) |
| 2 | verify rows to 16 (k <= 15) | with host drafting the matched drafts are free; the QSA chunk constants carry 8 `row_selects` and `derive_qsa_chunk_inputs` admits up to 8 completed blocks | structured output 11.2 tokens/pass at k = 15 vs 7.0 at k = 7 | next |
| 3 | slab prefill re-streams the experts 16x per layer | `_routed_partial_blocks`: one 128-token `moe_compute` call per 128-row block, each touching ~92% of the 512 experts: 48 x 16 x 1.3 GB per slab = 0.26 ms/token, the measured 0.25-0.31 | 512 tokens per call: the MoE stream / 3.7, prefill -20..25% | hardware A/B (`routed_tokens_per_call` admits (rows, 32, 128) only) |
| 4 | the verify commit reruns the GDN chunk kernel | `commit_rows` "reruns the kernel over the committed prefix": two runs x 36 layers per pass | part of the verify's 1.5x; #55548's per-token-state op makes the commit a slot copy | port the C++ op |
| 5 | dense weights are bf16 | every GDN / QSA / GR / MoE-dense upload is `bfloat16`: ~5.5 GB + a 1.27 GB LM head per token | floor 4.1 -> 2.6 ms; ~4% today, ~30% of the ceiling once fused | a precision decision |
| 6 | the host drafter transfers to the 27B branch | its K = 11 draft is ~18 ms of an 86 ms iteration | up to ~20% on copy-heavy output | later |
| 7 | defaults | the launcher runs the slowest prefill unless `--prefill-slab 2048`; `TT_METAL_TRACE_ALLOC_TRACKING=1` in production | free TTFT; the tracker's replay cost is unmeasured | A/B on the box |

Measure first: the verify pass's +18 ms composition, the collectives per decode token, the 55% of the slab prefill
that is not MoE / attention / GDN, the 256k decode slowdown.

### Item 1 on this branch: MTP drafting with the 128-row chunks and the slab

`Qwen38TTNNMTPChunkExtension` is one per chunk kind: the 32-row one owns the MTP layer's chunk histories, the
128-row and slab ones share them through `base` (the layer's own `allocate_chunk_state` contract) and take the
backbone's shared MoE combine buffer of their form; the MTP input mixer's `rows()` takes the row count; the prefill
driver routes every chunk to its kind's extension and writes rows-sized token rows one position ahead; the session
allocates, warms (every kind at its own residue) and captures the three chunk traces with their extensions and
releases them derived-first; the admission adds `MTP_PREFILL_EXTENSION_BYTES_PER_BANK_UPPER_BOUND` (8 MB) per
further kind. `--mtp` now combines with `--long-chunks` and `--prefill-slab`. Unrun on silicon: the first
`--mtp 4 --prefill-slab 2048 --acceptance` start is its proof (acceptance replays a 130-token prompt: the 128-row
kind runs there; a slab needs a prompt of 2,048+).

## 6. No-device suite, this branch vs its base (2026-09-22, pip `ttnn` wheel, no checkpoint)

| tree | tests | passed | failed | errors | skipped |
|---|---|---|---|---|---|
| base `545cb29d` | 1,641 | 1,458 | 38 | 16 | 86 |
| this branch | 1,706 | 1,523 | 38 | 16 | 86 |
| + per-request draft length | 1,725 | 1,542 | 38 | 16 | 86 |
| + host drafting (hybrid / ngram) | 1,754 | 1,614 | 38 | 16 | 86 |
| + MTP with the 128-row chunks and the slab | 1,766 | 1,626 | 38 | 16 | 86 |

The 65 added tests are the rows 7/8 and k 6/7 parametrizations. The failing and erroring set is identical on both
trees: the checkpoint-reading tests (`QWEN38_CHECKPOINT` unset) and `test_ttnn_bf4_static`, which pins the
checkout's patched `ttnn.load_tensor` that the pip wheel does not carry. Recipe: `PYTHONPATH=$PWD TT_METAL_HOME=$PWD
python -m pytest models/demos/blackhole/qwen38_flash_next/tests --confcutdir=models/demos/blackhole/qwen38_flash_next/tests`
with the checkout's `ttnn/ttnn/unsafe_allocation_tracker.py` and `trace_allocation_config.py` on the wheel's path.

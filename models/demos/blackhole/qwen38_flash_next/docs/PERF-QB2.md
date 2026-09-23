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

## 2. What this branch changes

Four changes. Each is validated by the no-device suite (section 7) and unrun on silicon; the start that proves each
is named in section 4.

### 2.1 MTP draft length 3..7 (`--mtp K`)

The server admitted `--mtp 3|4`; the library admitted k = 5 but routed its 6-row verify MoE onto the 32-row prefill
form (`moe_rows_for`: "32 for k = 5 until rows 6 is admitted"). The verify pass runs on 32-row tiles whatever R is,
so verify cost is flat in k and only the k - 1 draft rows scale.

| file | change |
|---|---|
| `ttnn/qsa.py` | `VERIFY_MAX_ROWS` 6 -> 8. A pass of R rows at P % 4 = r completes `(r + R) // 4` compressed blocks, at most `(R + 3) // 4` = 2 for every R <= 8, so `VERIFY_COMPLETED_BLOCKS` stays 2 and the pool select / block writes are unchanged; an import-time assert pins the invariant. `MAX_SPECULATIVE_STEPS` (the v1 fixed-five engine's reuse distance) is untouched: v2 never reads it. |
| `ttnn/moe.py` | rows 6, 7, 8 admitted (`TARGET_VERIFIER_ROW_COUNTS`), on the rows-5 code path: one `moe_compute` call of `rows` tokens, `output_height_shard_dim` 1. `ROWS6TO8_HARDWARE_PROVEN = False` until the first acceptance replay at k >= 5. |
| `ttnn/mtp_v2.py` | `SUPPORTED_DRAFTS = (3, 4, 5, 6, 7)`; `moe_rows_for` returns the exact 6/7/8. |
| `tools/qwen38_chat_session.py`, `tools/qwen38_chat_server.py`, `tools/run_qwen38_chat_server.sh` | `--mtp 3..7`. |
| tests | rows 7 and 8 / k 6 and 7 added to every verify, accept and draft parametrization; the pinned constants replaced by the invariants. |

No buffer shape changes: `SPARSE_INDEX_CAPACITY` stays 2,080, the verify and draft states are 32-row tiles.

### 2.2 Per-request draft length (`--mtp K[,K...]`, request field `speculative_drafts`)

`--mtp 4,7` opens one arm per draft count (`Qwen38ChainMTPArm`: the rows-dependent verify / draft states and three
traces; the MTP layer's caches and the TAIL step inputs shared through
`allocate_verify_state(alignment_generic_state=...)`), the first the default. A request's `speculative_drafts` picks
an arm, a request with tools takes the largest (`choose_speculative_drafts`: tool calls are structured output, where
more drafts are accepted per pass), `/health.mtp` lists `arms` and `default_k`, the response's `qwen38.mtp.k` is the
arm used. The admission adds `MTP_ARM_BYTES_PER_BANK_UPPER_BOUND` (12 MB per bank) per further arm and the open
measures the growth against it; the warm pass runs every residue per arm; `close()` releases every arm, the owner of
the shared generic state last. A sampled request takes the 1-row loop and ignores the field (section 5, lever 1).

### 2.3 Host drafting (`--draft-source hybrid|ngram`)

`ttnn/ngram_draft.py` (`Qwen38NgramDrafter`: prompt lookup, the tokens that followed the most recent earlier
occurrence of the last 2..4 tokens, the n-gram index kept incrementally) and the pass loop's host-first form
(`Qwen38TTNNMTPChain(host_drafter=...)`, `_finish_pass_host_first`): the verify row is read alone, the drafter is told
the pass's committed rows and asked for the next pass's k ids, a proposal is written into the token row and the draft
lanes (`write_verify_tokens`) and the k - 1 device draft rows (~4 ms each) are skipped; when nothing recurs the device
draft trace replays as before. Lossless by construction: the accept rule reads only the target's argmaxes, so a
proposal of any quality changes how much a pass commits and nothing else. The session seeds the drafter with the
request's committed ids (prompt and generation); the response's `qwen38.mtp.host_drafted_passes` counts the host
passes. `ngram` is the host-alone A/B arm (fill tokens when nothing recurs). Not combined with a commit queue or a
separate draft-history trace.

### 2.4 MTP drafting with the 128-row chunks and the slab

`--mtp` forced the slowest prefill: the driver ran the MTP chunk extension only for 32-row chunks, so an MTP server
prefilled at 3.3 ms per prompt token against the slab's 0.87 (a 4k prompt: 13.5 s against 3.6). Now
`Qwen38TTNNMTPChunkExtension` is one per chunk kind: the 32-row one owns the MTP layer's chunk histories, the 128-row
and slab ones share them through `base` (the layer's own `allocate_chunk_state` contract) and take the backbone's
shared MoE combine buffer of their form; the MTP input mixer's `rows()` takes the row count; the prefill driver routes
every chunk to its kind's extension (`mtp_by_kind`) and writes rows-sized token rows one position ahead; the session
allocates, warms (every kind at its own residue) and captures the three chunk traces with their extensions and
releases them derived-first; the admission adds `MTP_PREFILL_EXTENSION_BYTES_PER_BANK_UPPER_BOUND` (8 MB) per
further kind. `--mtp` combines with `--long-chunks` and `--prefill-slab`.

### 2.5 Sampled requests through the pass loop (`--mtp` with `--sampling`)

The pass loop drafted greedy requests only; the model card's defaults are temperature 0.7 / 1.0, so most chat
traffic ran the 1-row loop at 27 tok/s. Now every verify pass lands, beside its argmax lanes, the R candidate rows
of the target's logits (`Qwen38TTNNLMHead.sampling_candidate_rows`: the TAIL epilogue's per-shard top-32 with a
row axis), the MTP alignment's R argmaxes, the alignment residual rows and the logit rows
(`Qwen38TTNNVerifySampling`, allocated per arm with the chain's candidate-row constants). A sampled request the
loop admits (`sampling_step.mtp_pass_loop_admits`: `top_k` 1..32, no penalty that raises logits) takes the
host-first pass form with a sampler (`Qwen38TTNNMTPChain(sampler=..., state=...)`, `_sample_pass`): after the
verify the host reads the candidate rows and chooses row 0's token under the request's policy
(`sampling_step.Qwen38MTPSampler`: the same `sample_candidates` as the 1-row loop, the history the committed
tokens then `t_P` then the rows chosen so far, the request's generator), accepts draft 1 only where it equals that
choice, chooses row 1's, and so on to the first mismatch; the candidate guard's fallback samples the row's landed
logits (`read_verify_logits_row`, an eager gather). `apply_host_accept` then writes the accept where the device
wrote its own: the accept scalar (the next commit's selectors), `P` (the body advanced it by the device's count),
the readback row's fixed lanes (the draft body's `t'` and `d_1'`, the latter the alignment's argmax at row a) and
row a of the alignment residuals re-selected into `alignment.residual`; the draft trace then replays unchanged
(or a host drafter proposes). Exact by construction: every committed token is a draw from the target's
conditional given its committed prefix, whatever the drafts; rows past the first mismatch are never drawn for, so
one draw per committed token in row order, and a seed reproduces the 1-row sampled loop's draws up to the verify
pass's rounding (the no-device suite proves the streams equal, generator state included, for every acceptance
pattern at k 3/4/7, both launch forms, the fallback and a host drafter on top). The alignment ran on the
argmaxes, so a sampled pass's first draft and the MTP layer's rows past the sample assume the argmax at row a:
draft quality, never the stream. The session draws the first token from the last prompt TAIL's candidate row,
appends every sample before its token is yielded (the logprobs items), keeps the device sampler greedy for such
requests, and hands the 1-row sampled loop the rest where the greedy loop would hand its own. The buffers cost about 1.2 MB per device per arm (the logit rows are the
bulk: R x 62,080 bf16), inside the admission's per-arm bound; the verify trace gains the rows epilogue's few ops
and one 1 MB copy per pass, the greedy pass reads none of it.

## 3. Expected effect (from the tree's pass costs; measure, do not trust)

Pass time T(k) ~= 54.7 + 4k ms. Tokens per pass = 1 + accepted drafts.

| prompt class | accepted per draft (from the pinned tables) | k = 4 | k = 5 | k = 7 |
|---|---|---|---|---|
| `json` (structured) | ~0.95 | 4.8 tok / 70.7 ms = **68 tok/s** | ~5.6 / 74.7 = ~75 | ~7.0 / 82.7 = ~85 |
| median acceptance prompt | ~0.68 | 2.76 / 70.7 = **39 tok/s** | ~2.9 / 74.7 = ~39 | ~3.1 / 82.7 = ~37 |

k = 5 is never worse than 4; k = 7 pays on structured output and costs 5-10% on chat. That asymmetry is why the
draft length is per request (2.2), not a larger default. Host drafting (2.3): structured output, code and copy-heavy
replies at close to verify-only pass time (~55 ms for up to k + 1 tokens); chat unchanged. The prefill kinds (2.4):
TTFT of an MTP server at the slab's 0.87 ms per prompt token instead of 3.3. Sampled requests (2.5): the pass
loop's rates at the sampled acceptance — a draft is accepted with the probability the policy gives the drafted
token, so at temperature 0.7 on structured output most of the greedy 0.95 survives and on chat less of the 0.68;
measure per prompt class.

## 4. Hardware validation order (QuietBox 2)

1. `--profile qb2 --acceptance` with no `--mtp`: pins the route; `json` 96/96.
2. `--mtp 4 --acceptance`: the pinned MTP divergence table (`tools/ci/baselines/A3-mtp4-32k-*`) must reproduce.
3. `--mtp 5`, `--mtp 6`, `--mtp 7`, each with `--acceptance`: the first silicon run of the 6/7/8-row MoE form and the
   7/8-row verify. Gate: `json` 96/96 and the eleven divergence indices unchanged from step 2 (the target's greedy
   stream does not depend on k; only pass timing does). Then flip `ROWS6TO8_HARDWARE_PROVEN`.
4. `--mtp 4,7 --acceptance`: the arms. The records reproduce at each arm (`speculative_drafts` 4 and 7 on the same
   prompt commit the same stream), `/health.mtp.arms` is `[4, 7]`, the measured DRAM growth stays under the
   admission's bound (the open refuses otherwise).
5. `--mtp 4 --draft-source hybrid --acceptance`: the same committed stream as step 2 (the drafter cannot change it);
   `qwen38.mtp.host_drafted_passes` > 0 on `json` and the code prompts, ~0 on `chat`. Then `--draft-source ngram`
   for the drafter's own tokens per pass.
6. `--mtp 4 --prefill-slab 2048 --acceptance`: the acceptance prompts are ~130 tokens, so the 128-row kind runs
   there and the records must reproduce; a slab needs a prompt of 2,048+ tokens (one long request after): TTFT
   against `--mtp 4` alone.
7. Timing at each k on the twelve acceptance prompts: pass time and tokens per pass, against section 3.
   Then the sampled arm: `--mtp 4` (the launcher passes `--sampling`) with a `temperature 0.7` request carrying a
   `seed`, twice, and the same request on `--no-mtp`'s 1-row loop (or `speculative_drafts` absent on a plain
   server): the two seeded streams must agree until a near-tie of the verify pass's rounding, the pass loop's
   reply must report `qwen38.mtp.sampled_passes` > 0, and `qwen38.sampling.fallbacks` should stay at 0 on the
   twelve prompts (a fallback is a boundary tie; each costs an eager 1 MB gather).
8. A/Bs, one variable each: `QWEN38_FUSED=final_mixer,gdn_step,position_advance` in the launcher's environment (it
   `exec`s the server with the caller's environment; `gdn_step` replaces the 49-program GDN chain with one program per
   layer and is off by default only because its rounding follows the torch oracle rather than the chain, tolerance
   class COMPONENT: the acceptance replay is the decision); `--prefill-slab 4096` against 2048 (admitted, unmeasured;
   the DGX Spark port's 4096-token chunk was two thirds of its prefill gain); `TT_METAL_TRACE_ALLOC_TRACKING` unset
   (exported in production, replay cost unmeasured); `--draft-source hybrid` against `mtp` per prompt class.

## 5. Next levers, ranked (the 2026-09-22 audit, extended by the 2026-09-23 review of the other ports)

| # | lever | evidence | expected effect | status |
|---|---|---|---|---|
| 1 | **sampled requests through the pass loop** | the chain drafts greedy requests only (`drafting = ... sampling is None`); the model card's defaults are temperature 0.7 / 1.0, so most chat traffic runs the 1-row loop at 27 tok/s. Every other port serves sampled requests speculatively (MTPLX: "exact at any temperature") | the MTP rates (39-68 tok/s at k = 4) for temperature > 0, times the sampled acceptance | **on this branch** (2.5); the rejection-sampling form stays open |
| 2 | verify rows to 16 (k <= 15) | with host drafting the matched drafts are free; the QSA chunk constants carry 8 `row_selects` and `derive_qsa_chunk_inputs` admits up to 8 completed blocks; needs `VERIFY_COMPLETED_BLOCKS = (R + 3) // 4`, the pool-select stack for N rows, block-start RoPE rows, MoE rows to 16 | structured output 11.2 tokens per pass at k = 15 against 7.0 at k = 7 | no-device-testable |
| 3 | **index sharing for the draft rows** | SGLang's IndexShare and vLLM's "index sharing": the draft rows reuse the QSA block selection of the last accepted row instead of running the indexer per row (N + 1 extra columns for the positions drafted since); +3-4% on coding and JSON on the DGX Spark | a cheaper draft row (the k - 1 rows are ~4 ms each of which the MTP layer's indexer is an unmeasured part) | device; measure the draft row's program census first |
| 4 | **a 64k draft vocabulary and a bf4 draft LM head** | the DGX Spark recipe's `DRAFT_VOCAB=1` (the drafter's head over 65,536 ids); the draft row here re-reads the 1.27 GB bf16 LM head per row. Lossless: a draft outside the subset is a wrong draft, the verify head is the full one | 1.27 GB -> ~80 MB per draft row: ~0.6 ms of the ~4 ms, k - 1 times per pass | the sliced head and the host id remap build offline; silicon proves it |
| 5 | slab prefill re-streams the experts 16x per layer | `_routed_partial_blocks`: one 128-token `moe_compute` call per 128-row block, each touching ~92% of the 512 experts: 48 x 16 x 1.3 GB per slab = 0.26 ms per token, the measured 0.25-0.31 | 512 tokens per call: the MoE stream / 3.7, prefill -20..25% | hardware A/B (`routed_tokens_per_call` admits (rows, 32, 128) only) |
| 6 | the verify commit reruns the GDN chunk kernel | `commit_rows` "reruns the kernel over the committed prefix": two runs x 36 layers per pass | part of the verify's 1.5x; #55548's per-token-state op makes the commit a slot copy | port the C++ op |
| 7 | dense weights bf16 -> bfp8 | every GDN / QSA / GR / MoE-dense upload is `bfloat16`: ~5.5 GB + a 1.27 GB LM head per token. The reference deployment already runs these at 8 bits (Qwen's FP8 checkpoint; the DGX Spark ports' "FP8 heads and side layers"; MTPLX keeps the QSA projections at 8-bit under 4-bit everything else) | floor 4.1 -> 2.6 ms; ~4% today, ~30% of the ceiling once fused | a precision decision; keep the QSA projections at 8 bits |
| 8 | the PLE row lookup: warm, measure, overlap | the n-gram table is host-resident (page cache) and `refresh_ple_row` runs on the host per token, cost unmeasured. On the DGX Spark the table's placement moved prose decode 21.7 -> 37.2 tok/s (disk -> memory); SGLang's offloaded table costs -0.07% because the 16-row gather overlaps the first decoder block | a cold page cache costs NVMe reads per token; a host segment on the critical path costs its length every token | warm the table before serving (`vmtouch` / a read pass); time the segment; overlap the row copy with the trace launch |
| 9 | the host drafter transfers to the 27B branch | its K = 11 draft is ~18 ms of an 86 ms iteration | up to ~20% on copy-heavy output | later |
| 10 | defaults | the launcher runs the slowest prefill unless `--prefill-slab 2048`; `TT_METAL_TRACE_ALLOC_TRACKING=1` in production | free TTFT; the tracker's replay cost is unmeasured | A/B on the box (section 4, step 8) |
| 11 | batched decode | under 10% of roofline; every other port's aggregate is 6-9x its single stream by 32 streams; B = 8 is ~5-7x here | the largest aggregate lever; a redesign of the traced chain, sampler and server | not before the numbers above exist |

Measure first: the verify pass's +18 ms composition, the collectives per decode token, the 55% of the slab prefill
that is not MoE / attention / GDN, the 256k decode slowdown, the PLE host segment.

Not levers: the n-gram lookup's round trip (`refresh_ple_row` computes the row on the host from the token it already
read back and does one host-to-device copy) and GDN state precision (fp32 by construction).

### Lever 1, built (2.5): what stays open

The match rule accepts a draft with the probability the policy gives the drafted token; the rejection sampler
that MTPLX, vLLM and SGLang implement accepts with min(1, p(d) / q(d)) and draws the residual on a rejection, which
needs the draft's probabilities q from the k - 1 draft rows (a second readback row per pass) and a residual
distribution per rejected row. It is the follow-up if the match rule's sampled acceptance disappoints on the
box; the readback and the accept live in `_sample_pass` and would gain a q-row, nothing else moves. A cheaper
row-0 variant needs no device change: when draft 1 misses, the residual could be drawn on the host from the
candidate row alone, exactly, because the row carries the target's top-32 per shard.

## 6. What the other ports do (reviewed 2026-09-23)

Single-stream decode of Qwen3.8-Flash-Next, published by the people running it. The QuietBox 2 has 2,048 GB/s
against a DGX Spark's 273 and an M5 Max's ~546; the port's single stream sits in the Spark's band because it is
program-count-bound (section 1), not because the bytes are slow.

| platform | engine, weights | plain | with MTP | notes |
|---|---|---|---|---|
| this port (4x p150, QB2 parity 2026-09-07) | tt-metal, bf4 experts / bf16 dense | 27.1 | 39 median, 68 `json` (k = 4) | prefill 300 -> 1,150 tok/s with the slab |
| DGX Spark x1 (GB10, 128 GB) | vLLM, NVFP4, MTP4, n-gram table on NVMe | ~20 ("<20 before MTP", SGLang) | 32.5 median, 21.7 prose, 43.8 peak; SGLang 42.7 on code | 6 streams 194 aggregate; cold prefill ~1,650 tok/s |
| DGX Spark x2 (TP2, ConnectX) | vLLM, NVFP4, MTP3, table in memory | | 53.7 median, 37.2 prose, 63.7 peak | TTFT 180 ms; 4,096-token prefill chunks 2,784 tok/s |
| DGX Spark x4 (TP4 + EP) | vLLM, NVFP4 | | 40.5 median, 54.2 peak | the collectives cost more than the bandwidth buys |
| DGX Spark x1 | llama.cpp, UD-IQ1_S (table 10.4 GiB) | 34.5 | (MTP WIP upstream; unsloth branch 1.3-1.7x) | |
| 2x RTX PRO 6000 (TP2) | vLLM, FP8, table on host | 81.5 | 2.5x wall-clock on real prompts | acceptance 71% random / 55% real code + prose; 4 GPUs 26% slower than 2 (PCIe); 32 streams 739 |
| 2x Radeon PRO R9700 | vLLM ROCm, MTP | | ~75 (55-90) | |
| M5 Max (MLX) | MTPLX, 4-bit (QSA projections 8-bit), table 32 GB on SSD | ~56 | 125.8 (depth 3, OpenCode) | Leviathan-Chen rejection sampling: exact at any temperature |
| 4x B200 (TP4) | SGLang, NVFP4, MTP | | 540 | accept length 3.3 |

What transfers:

1. **Speculative decoding for sampled requests** (all of them) -> lever 1.
2. **Index sharing in the draft rows** (SGLang, vLLM) -> lever 3.
3. **A reduced draft vocabulary** (the DGX Spark recipe) -> lever 4, beside the bf4 draft head.
4. **Draft length per workload**: "MTP4 with index sharing helps coding and JSON by 3 to 4 percent and costs prose
   6 percent" (Spark), acceptance 55-71% by workload (RTX PRO 6000) -> section 2.2's arms and its tools -> largest
   policy are the right shape; the default stays 4.
5. **n-gram-only drafting is not a replacement**: llama.cpp's `ngram-mod` speculation went flat to -0.8% on long
   non-repetitive prompts -> `hybrid` is the mode, `ngram` the A/B arm; both lossless here by construction where
   theirs changed the output at temperature 0.
6. **8-bit dense is the reference deployment** -> lever 7's precision risk is bounded; keep the QSA projections at 8.
7. **Hyper-connection fusion pays in speculative decode**: SGLang's fused mix / combine kernels are 2x at M <= 16
   and 7.6% end to end -> the `final_mixer` A/B (section 4, step 8) is worth running first.
8. **The n-gram table's placement is a first-order term** (Spark: prose 21.7 -> 37.2 tok/s from NVMe to memory;
   SGLang: free once the gather overlaps the first block) -> lever 8.
9. **4,096-token prefill chunks** (two thirds of the Spark TP2 prefill gain) -> the `--prefill-slab 4096` A/B.
10. **MTP under load**: at 64 concurrent streams on 4x H100 the acceptance fell to ~36% and MTP made throughput
    8-36% worse -> when batching arrives (lever 11), draft length is per request and off above a concurrency
    threshold.

Not transferable: NVFP4 / FP8 kernels (bf4 / bfp8 are the Blackhole forms), CUDA graphs (the traces), the TP / EP
layouts (the resident build needs all four chips' DRAM), the engines' prefix-cache and top-k bugs.

Sources: NVIDIA developer forums (Qwen3.8-Flash-Next on 1, 2 and 4 DGX Sparks; single DGX Spark at ~43 tok/s in
coding; single node vLLM 24 tok/s), blazux/qwen3.8-Flash-DGX, kubesimplify (DGX Spark and RTX PRO 6000),
lmsys.org "Qwen3.8-Flash-Next: Day-0 Support in SGLang", recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next, ggml-org/llama.cpp
PR #27742, unsloth.ai/docs/models/qwen3.8-next, cat5edopeHA/qwen38-flash-next-ai1, youssofal/MTPLX.

## 7. No-device suite, this branch against its base (2026-09-22/23, pip `ttnn` wheel, no checkpoint)

| tree | tests | passed | failed | errors | skipped |
|---|---|---|---|---|---|
| base `545cb29d` | 1,641 | 1,458 | 38 | 16 | 86 |
| + draft length 3..7 | 1,706 | 1,523 | 38 | 16 | 86 |
| + per-request draft length | 1,725 | 1,542 | 38 | 16 | 86 |
| + host drafting (hybrid / ngram) | 1,754 | 1,614 | 38 | 16 | 86 |
| + MTP with the 128-row chunks and the slab | 1,766 | 1,626 | 38 | 16 | 86 |
| + the polish pass | 1,766 | 1,626 | 38 | 16 | 86 |
| + sampled requests through the pass loop | 1,783 | 1,643 | 38 | 16 | 86 |

The 17 tests added last are the sampled pass loop's (the sequential-stream equivalence for every acceptance
pattern, the fallback, the hybrid composition, the sampler, the readback round trip and the source contracts).
The failing and erroring set is identical on every row: the checkpoint-reading tests (`QWEN38_CHECKPOINT` unset) and
`test_ttnn_bf4_static`, which pins the checkout's patched `ttnn.load_tensor` that the pip wheel does not carry.
Recipe: `PYTHONPATH=$PWD TT_METAL_HOME=$PWD python -m pytest models/demos/blackhole/qwen38_flash_next/tests
--confcutdir=models/demos/blackhole/qwen38_flash_next/tests` with the checkout's
`ttnn/ttnn/unsafe_allocation_tracker.py` and `trace_allocation_config.py` on the wheel's path.

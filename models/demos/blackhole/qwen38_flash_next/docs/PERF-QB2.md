# Wider verify and host drafting (branch `perf/flash-next-qb2-v2`)

This branch adds three opt-in features to Sam's main and the launcher's `--lanes` passthrough
(`qb2-serve`). The served defaults are unchanged. Everything below the "measured" rows is a model to check
on silicon, not a result.

## What the branch adds

| Switch | What it does | Status |
|---|---|---|
| `--mtp 3..15` | The QSA verify path takes up to 16 rows per pass and writes every compressed block a row count can complete: two blocks up to 8 rows, three to 12, four to 16. k = 6..15 run the 32-row MoE form. Lanes stay at k <= 5. | k = 6..15 unrun on silicon |
| `--draft-source hybrid\|ngram` | A greedy pass's k drafts come from the host's prompt lookup (`ttnn/ngram_draft.py`) instead of the k - 1 sequential device draft rows. `hybrid` falls back to the device draft when nothing recurs. The committed stream is exact either way. | Unrun on silicon |
| `QWEN38_MTP_DRAFTS_PER_REQUEST=4,8,15` | Opens exactly those drafting chains (the measured `1` = the pair 4, 5 is unchanged). A request picks one with `extra_body.mtp_drafts`. | Unrun beyond the pair |

## The cost model

Measured:
- **A k = 4 pass** takes about 40-42 ms.
  - On this QuietBox 2 on 2026-10-13: 40.2 ms on a code answer, 229 tokens in 51 passes (4.49 tokens per pass, 111.6 tok/s).
  - Sam's QB2 on 2026-09-29 measured 104.9 tok/s on code and 68.7 on the 560-token chat prompt.
- **A device draft row** costs about 2.5-2.7 ms. The QB2 census measured 8.23 ms for k = 4's three rows.
- **Verify cost grows with the distinct experts the rows touch.** With 10 routed experts per row out of 512, R rows touch E(R) = 512 (1 - (1 - 10/512)^R):
  - E(5) = 48, E(8) = 75, E(12) = 108, E(16) = 138.
  - Sam's 4-lane pass (20 rows) cost 12.7 ms more than the single stream's 5 rows; this model predicts 13.6.

Estimated for one request:

| Form | Pass | When it pays |
|---|---|---|
| k = 4, device drafts (default) | ~41 ms | Always the baseline |
| k = 4, host drafts | ~33 ms on a host-drafted pass | Copy-heavy text (code edits, JSON, quoting); chat declines to the device draft |
| k = 7 / 11 / 15, host drafts | ~40 / 47 / 52 ms | Long verbatim spans: ~150 tok/s, up to ~300 on long copies |
| k = 15, device drafts | ~85 ms | Never: chat and reasoning would run near 35 tok/s |

**Wide k needs host drafts.** It is a per-request choice through the per-request chains. `--mtp 15` as the
server default would slow every chat request.

## Validation order on the QuietBox 2

Each step uses `--acceptance --require-json-96`. The json 96/96 replay is the correctness gate at every k.

1. **The default path is unchanged.**
   - Start `--mtp 4` from this branch.
   - The acceptance divergence indices must equal `tools/ci/baselines/A3-mtp4-32k-divergence_index.json` (up to 8 rows the verify writes the same two blocks).
2. **Price the wide verify.**
   - Start `--mtp 7`, `--mtp 11` and `--mtp 15` in turn.
   - Send the same code prompt to each and read the pass time: `timings.predicted_ms / (timings.draft_n / k)`.
   - This number decides whether wide k is worth keeping.
3. **Measure host drafts at k = 4.**
   - Run `--draft-source hybrid` against the default.
   - Use the acceptance prompts and one copy-heavy request, such as a 200-line file with "add type hints".
   - `qwen38.mtp.host_drafted_passes` says how often the lookup fired.
4. **Combine them**, if step 2 lands near the model.
   - Start with `QWEN38_MTP_DRAFTS_PER_REQUEST=4,15 --draft-source hybrid`.
   - Send copy-heavy requests with `extra_body.mtp_drafts: 15`.

## What this branch does not do

- **The lanes** (`--lanes`, the many-agents mode) stay device-drafted at k <= 5. Host drafting has no lanes form.
- **Host drafting runs on one command queue.** It turns off the early PLE-rows read (`QWEN38_MTP_PLE_EARLY`). On a pass the host declines, that costs about a millisecond.
- **Per-step widening is the next lever.** It would switch k inside a request only when the lookup finds a long match. Today a request keeps one k.
- **Agent workloads are prefill-bound.** In a 6-turn agent task about three quarters of the chip time re-reads prompts. Prefix reuse across turns on the lanes is the larger lever for Quarterdeck than anything here.

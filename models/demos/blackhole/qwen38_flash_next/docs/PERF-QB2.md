# Wider verify and host drafting (branch `perf/flash-next-qb2-v2`)

This branch adds three opt-in features to Sam's main and the launcher's `--lanes` passthrough
(`qb2-serve`). The served defaults are unchanged. The numbers below were measured on this QuietBox 2
on 2026-10-13 unless marked as a model.

## What the branch adds

| Switch | What it does |
|---|---|
| `--mtp 3..15` | The QSA verify path takes up to 16 rows per pass and writes every compressed block a row count can complete: two blocks up to 8 rows, three to 12, four to 16. k = 6..15 run the 32-row MoE form. Lanes stay at k <= 5. |
| `extra_body.draft_source` (`--draft-source` sets the default) | `mtp` (the device's MTP head), `hybrid` (the host's prompt lookup, `ttnn/ngram_draft.py`, when the request's text recurs, the MTP head otherwise) or `ngram` (the host alone). Chosen per greedy request. A host-drafted request's pass loop reads its verify row on the main queue; the device-drafted requests keep the early PLE-rows read. The committed stream is exact either way. |
| `QWEN38_MTP_DRAFTS_PER_REQUEST=4,8,15` | Opens exactly those drafting chains (the measured `1` = the pair 4, 5 is unchanged). A request picks one with `extra_body.mtp_drafts`. |

## Measured

Every server start ran `--acceptance --require-json-96`, so each configuration below passed the json 96/96 replay
before serving, the k = 15 verify included.

| Server and request | Prompt | tok/s | Tokens per pass | ms per pass | Host-drafted passes |
|---|---|---|---|---|---|
| `--mtp 4` (device drafts) | merge two sorted lists (229 tokens) | 111.6 | 4.49 | 40.2 | - |
| `--mtp 4`, `draft_source: hybrid` | the same | 93.7 | 3.52 | 37.6 | 27 / 65 |
| `--mtp 15`, `draft_source: hybrid` | the same | 89.2 | 6.21 | 68.7 | 9 / 38 |
| `--mtp 15`, `draft_source: hybrid` | rename a word in a 116-line file (400 tokens) | 276.2 | 14.14 | 49.9 | 28 / 29 |

What they say:
- **The wide verify works and costs about 10 ms.** A 16-row pass is 49.9 ms against 40 ms at 5 rows. The
  distinct-expert model below predicted 52.
- **On copied text, host drafts at k = 15 are 2.5x the default.** The lookup proposed 28 of 29 passes and each
  pass committed 14 tokens.
- **On ordinary text, the device's drafts win.** Hybrid at k = 4 lost 16%: the lookup fires on 2- and 3-token
  repeats that are everywhere in code, its guesses committed about 2 tokens per pass against the MTP head's 4.5,
  and skipping three device draft rows saves only about 8 ms. At k = 15 every declined pass pays 14 device draft
  rows, so the average pass rose to 68.7 ms.
- **So both choices belong on the request:** `mtp_drafts: 15, draft_source: hybrid` for edits and rewrites, the
  defaults for everything else, on one server (`QWEN38_MTP_DRAFTS_PER_REQUEST=4,15`).

The lanes on `qb2-serve` (`--mtp 4`, the same code prompt sent concurrently):

| Lanes | Requests at once | Total tok/s | Per request |
|---|---|---|---|
| 4 | 1 | 79 | 79 (a 56.6 ms pass: every pass computes all four lanes) |
| 4 | 4 | 195 | 56-85 |
| 6 | 6 | 226 | 43-75 |

## The cost model

- **A device draft row** costs about 2.5-2.7 ms (the QB2 census: 8.23 ms for k = 4's three rows).
- **Verify cost grows with the distinct experts the rows touch.** With 10 routed experts per row out of 512, R rows
  touch E(R) = 512 (1 - (1 - 10/512)^R): E(5) = 48, E(8) = 75, E(12) = 108, E(16) = 138. Sam's 4-lane pass (20
  rows) cost 12.7 ms more than the single stream's 5 rows; the model predicts 13.6.

## Next levers

- **Per-pass widening.** Switch to the wide chain only when the lookup finds a long match, so no client has to
  choose k or the drafter. These numbers say it is worth building.
- **Host drafting on the lanes.** The lanes (the many-agents mode) stay device-drafted at k <= 5. Agents that edit
  code would gain the most from a lanes form.
- **Prefix reuse on the lanes.** Agent workloads are prefill-bound: in a 6-turn agent task about three quarters of
  the chip time re-reads prompts. For Quarterdeck this is the largest lever of all.

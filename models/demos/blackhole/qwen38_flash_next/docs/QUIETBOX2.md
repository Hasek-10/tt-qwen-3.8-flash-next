# Qwen3.8-Flash-Next for agents on one QuietBox 2

This branch serves Qwen3.8-Flash-Next to six agents at once from one Tenstorrent QuietBox 2 (two p300c cards, four
Blackhole dies). With all six lanes decoding, real agent requests ran at a median 35.9 tokens/s each (68,071
requests, 2026-10-20 to 10-29). It has served a 20-role agent company since 2026-10-19.

It is [sjettTT/tt-qwen-3.8-flash-next](https://github.com/sjettTT/tt-qwen-3.8-flash-next) (Apache-2.0) at
`3c8d37d305` (2026-10-06), plus three changes for multi-agent serving and upstream's prefill work of 2026-10-10
to 10-15. Everything in the model's [README](../README.md) still applies: the build, the checkpoint, the caches and
the server's own documents ([SERVER.md](SERVER.md), [PREFILL.md](PREFILL.md), [NUMERICS.md](NUMERICS.md)).

## Branches

| Branch | Built on | Adds | Run on the QuietBox 2 |
|---|---|---|---|
| `perf/lanes-upstream-oct15` (serve this one) | `perf/lanes-resume` | upstream's 14 commits of Oct 10-15 | 2026-10-31: startup acceptance 96/96 |
| `perf/lanes-resume` | `perf/lanes-prefix-reuse` | conversation checkpoints | served 2026-10-19 to 10-31 |
| `perf/lanes-prefix-reuse` | `qb2-serve` | shared-prefix reuse on the lanes | 2026-10-13 |
| `qb2-serve` | `3c8d37d305` | the launcher passes `--lanes` to the server | 2026-10-13 |
| `perf/flash-next-qb2-v2` | `qb2-serve` | verify up to 16 rows (`--mtp` to 15), host drafting per request; its `docs/PERF-QB2.md` | 2026-10-13, single stream |
| `perf/qb2-v2-upstream-oct15` | `perf/flash-next-qb2-v2` | the same 14 upstream commits | not yet |
| `perf/flash-next-qb2` | upstream of 2026-09-22 | the first version of the wider verify and host drafting; `perf/flash-next-qb2-v2` supersedes it | - |

## What changed from the fork point

### Three commits for multi-agent serving

| Commit | What it does |
|---|---|
| `94f71eec69` | The launcher passes `--lanes B` and `--lanes-stall-budget` to the server, which has served lanes since 2026-09-28 but could only be started by hand with them. `--lanes` implies greedy decode; B outside 2..8, lanes without `--mtp`, and a stall budget without lanes are refused before the server starts. |
| `cd9f47955d` | `--lanes-prefix-reuse`: an admission whose prompt shares more than 128 ids with the last admitted one prefills the shared ids once, keeps a snapshot at the last 32-row boundary below them, and later admissions restore it. Agents that share a system prompt and tools pay for it once. Pinned at every start: the json record fresh, captured and restored must match token for token and 96/96 against the CPU. |
| `eaff88b62f` | `--lanes-conversations N` (1..64): a request with tools or a past assistant turn saves its prompt's state into one of N host checkpoints, so the conversation's next turn resumes it and prefills only the new ids (least recently used replaced). The admission's import now stays on the device. `usage.prompt_tokens_details.cached_tokens` reports the ids a request did not prefill. Pinned at every start like the prefix reuse. |

### Fourteen upstream commits, 2026-10-10 to 10-15

Cherry-picked with `-x`, authors kept. They make `--prefill-slab` prefill faster and open the fabric with 8192 B
packets. The conflicts were imports and parameter lists only: the server keeps `CHUNK_ROWS` for the prefix reuse,
and the session keeps `ngram_draft` and `mtp_draft_source` beside upstream's `tail_slab_rows`.

| Here | Upstream | Change |
|---|---|---|
| `9d1b7068df` | `c8455d81f4` | one program-config rule for every prefill-slab linear |
| `b4b04987e0` | `c55c5df2d9` | GR slab read as tile walks, row gamma and gate activation |
| `610d751272` | `762ab3863e` | tail slab forms for the prefill remainder after the full slabs |
| `502d8949f7` | `def8b63d4f` | slab forms of 512 rows and fewer; tail slab forms are powers of two |
| `d92197f0e5` | `c9ebfe5e83` | the tail slab forms documented in PREFILL.md |
| `2491e72fa0` | `70e16f9718` | `--prefill-slab` takes powers of two only |
| `0200e21ff2` | `bb212392d8` | the slab policy import pin reads the imports |
| `91e9659ffb` | `301be0a905` | every slab entry point refuses non-power-of-two rows alike |
| `ecc9ef9076` | `5e8af6377b` | the model's fabric opens with 8192 B packets |
| `c13c7ec02b` | `e3aca67e1d` | `gr_partial_rs_ag` on by default |
| `4a5f655cfb` | `5a1e99d421` | GR slab read front as two fused programs, on by default |
| `15c9674b33` | `2e778e9fbc` | GR slab write as one fused program (off) |
| `3dad607270` | `4b00553cf3` | `gr_partial_rs_ag`'s measured prompt time on the 8192 B packets |
| `987c366eb6` | `79472f411f` | GR slab write as one fused program, on by default |

Only model files and JIT kernels changed, so a tt-metal built for the fork point serves this branch without a
rebuild. Left out: upstream's experimental expert-rows path in `moe_compute` (`fecc347ce5`, `b314b3dedb`), which
changes C++ ops and needs a rebuild. Upstream main also carried a rare decode race on that path, fixed then only on
upstream's `q4p/mcr` branch (`c95dacd58e`); none of it is in this branch.

## Benchmarks

One QuietBox 2, `--mtp 4`, greedy, every server started with `--acceptance --require-json-96`.

**Lanes, the same code prompt sent at once** (2026-10-13, `qb2-serve`):

| Lanes | Requests at once | Total tok/s | Per request |
|---|---|---|---|
| 4 | 1 | 79 | 79 (every pass computes all four lanes) |
| 4 | 4 | 195 | 56 to 85 |
| 6 | 6 | 226 | 43 to 75 |

The single stream measured 111.6 tok/s on the same code prompt (2026-10-13). The `perf/flash-next-qb2-v2`
branch reached 276.2 tok/s on a 400-token rename edit with host drafting at `--mtp 15`; its `docs/PERF-QB2.md` has
the rest.

**Real agent traffic** (`perf/lanes-resume`, 64k, `--lanes 6`, from the server's request log, 2026-10-20 to 10-29):

| Lanes decoding at the request's midpoint | Requests | Decode tok/s per request (median) |
|---|---|---|
| 6 | 15,546 | 35.9 |
| 4 to 5 | 24,353 | 33.3 |
| 1 to 3 | 28,172 | 43.7 |

A long first prompt (20k to 45k tokens) freezes the other lanes while it is admitted: the time a request spent frozen
had a median of 2.2 s and a 90th percentile of 18.5 s (2026-10-30 to 10-31).

## Run it

1. **Build and download** as the model's [README](../README.md) says in sections 1 to 3 (clone, submodules,
   `build_metal.sh`, `create_venv.sh`, the 360 GB checkpoint), cloning this repository and checking out
   `perf/lanes-upstream-oct15`.
2. **Start the server:**

   ```bash
   models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh \
       --profile qb2 --host 127.0.0.1 --port 8000 --allocated-context 65536 \
       --mtp 4 --lanes 6 --prefill-slab 2048 --lanes-prefix-reuse --lanes-conversations 20 \
       --checkpoint /data/Qwen3.8-Flash-Next --cache-root /data/qwen38-cache \
       --acceptance --require-json-96
   ```

   | Flag | Why |
   |---|---|
   | `--allocated-context 65536` | a 64k window per lane; six lanes fit to about 82k each, 128k only at three lanes or fewer |
   | `--mtp 4 --lanes 6` | six greedy requests decode together, four tokens drafted per pass |
   | `--prefill-slab 2048` | long prompts prefill in 2,048-row slabs |
   | `--lanes-prefix-reuse` | the shared system prompt and tools are read once |
   | `--lanes-conversations 20` | one checkpoint per agent: 0.2 GB of host memory each plus 0.2 GB per 4k tokens held, up to 68 GB at 64k |
   | `--acceptance --require-json-96` | the server replays the reference prompts and refuses to serve unless they match the CPU 96/96, the reuse pins included |

3. **Wait for ready.** The log prints `"event": "ready"` after about 5 minutes. The first start at a new window
   builds that window's caches under `--cache-root` (about 10 GB, once). Then `curl -s 127.0.0.1:8000/health` and
   `curl -s 127.0.0.1:8000/v1/models`; the second reports the window clients should use.
4. **Run it as a service** so a crash restarts it from a clean device:

   ```ini
   [Unit]
   Description=Qwen3.8-Flash-Next chat server (6 lanes, 64k)
   After=network-online.target
   StartLimitIntervalSec=0

   [Service]
   User=ttuser
   WorkingDirectory=/data/flash-next
   ExecStartPre=-/home/ttuser/.tenstorrent-venv/bin/tt-smi -r
   ExecStart=/data/flash-next/models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile qb2 --host 127.0.0.1 --port 8000 --allocated-context 65536 --mtp 4 --lanes 6 --prefill-slab 2048 --lanes-prefix-reuse --lanes-conversations 20 --checkpoint /data/Qwen3.8-Flash-Next --cache-root /data/qwen38-cache --acceptance --require-json-96
   Restart=on-failure
   RestartSec=30
   TimeoutStopSec=180
   LimitMEMLOCK=infinity

   [Install]
   WantedBy=multi-user.target
   ```

   The server drains the request in flight on SIGTERM, then closes the mesh. tt-metal pins host memory, so the
   service needs `LimitMEMLOCK=infinity`. The server has no authentication: keep it on loopback or a private bridge.
5. **Measure it under agent traffic:** `python3 models/demos/blackhole/qwen38_flash_next/tools/bench_agents.py
   --url http://127.0.0.1:8000 --levels 6 --turns 6` runs six agents for six turns each, every turn the whole
   conversation with a tool defined. Pause any other client first: its six agents take all six lanes.

## Known issues on the QuietBox 2

- **A start right after `tt-smi -r` can fail at the mesh open** with `MMIO per-op timeout: 4B load took 220641 us`:
  a chip that answered at driver start stopped answering two seconds later. The start 30 s later worked
  (2026-10-30). `Restart=on-failure` with `RestartSec=30` covers it.
- **Each die links at PCIe Gen4 x4**, against the x8 it supports, most likely the board's 16 CPU lanes split four
  ways rather than a fault. `sudo lspci -d 1e52: -vv | grep -E 'LnkCap|LnkSta'` shows both.
- **Harmless warnings:** tt-metal does not know the board (`Unknown motherboard 'B850M-C' ... falling back to
  bus_id as tray_id`), and the launcher still prints that the qb2 profile is untested; upstream verified it on
  2026-09-19 and this branch has served on it since 2026-10-13.
- **The lanes are greedy only.** A request with sampling fields is refused.
- **Long first prompts stall the other lanes** while they are admitted (above). Agents whose conversations stay
  checkpointed avoid most of it: keep `--lanes-conversations` at least the number of agents.

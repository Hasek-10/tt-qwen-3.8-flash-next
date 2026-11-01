#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Sawyer Pence
# SPDX-License-Identifier: Apache-2.0
"""Measure what the chat server sustains under agent-shaped traffic.

Runs agent-shaped requests (a long stable system prompt, a varying task, a few
hundred output tokens) straight at the server at increasing concurrency, and
suggests how many agents to run at once:

  start = the highest concurrency where every agent still gets >= floor tokens/s
          and median latency stays under the ceiling
  max   = the highest concurrency that still raises total throughput

Requests carry no sampling fields: the lanes (``--lanes``) serve greedy requests
only. The first level the server refuses (HTTP 429 or 503: its queue is full) ends
the sweep. Needs only httpx (``pip install httpx``).

  python3 bench_agents.py --url http://127.0.0.1:8000 --floor 15

``--turns T`` runs agent tasks instead: every agent sends T turns, each the whole
conversation so far (its replies sent back as agents send them: content and tool
calls, the reasoning dropped) plus a tool result, with a tool defined. That is the
traffic ``--lanes-prefix-reuse`` and ``--lanes-conversations`` serve; the report adds
the share of prompt tokens the server did not prefill (its ``qwen38.prefix_reused``).

  python3 bench_agents.py --url http://127.0.0.1:8000 --levels 6 --turns 6
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

SYSTEM = ("You are an agent in a small company. Follow the charter. " * 180).strip()  # ~2k tokens, cache-friendly
TASKS = [
    "Summarize the tradeoffs of three website hosting options for a small electrician business.",
    "Draft a five-step plan to evaluate a grant opportunity against our capabilities.",
    "List risks in sending a cold outreach email and how to mitigate each one.",
    "Explain how to backtest an opening-range breakout strategy without lookahead bias.",
]


REFUSED = (429, 503)  # the server's queue is full at this concurrency
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a fact in the company knowledge base.",
            "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
        },
    }
]
TOOL_RESULT = (
    "Result: the knowledge base holds three relevant notes on pricing, hosting and support "
    "terms; each lists its source and date. "
) * 12  # ~300 tokens, like a short tool reply


async def one(client: httpx.AsyncClient, url: str, model: str, i: int, max_tokens: int) -> tuple[float, int]:
    t0 = time.monotonic()
    # No sampling fields, like the agents' own requests. Some servers batch greedy requests
    # only (Flash-Next's --lanes refuses a sampled one).
    r = await client.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": TASKS[i % len(TASKS)] + f" (variant {i})"},
            ],
        },
    )
    r.raise_for_status()
    return time.monotonic() - t0, int(r.json()["usage"]["completion_tokens"])


async def task(
    client: httpx.AsyncClient, url: str, model: str, i: int, turns: int, max_tokens: int
) -> list[tuple[float, int, int, int]]:
    """One agent's task: (seconds, completion tokens, prompt tokens, prompt tokens the server reused) per turn."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASKS[i % len(TASKS)] + f" (variant {i})"},
    ]
    out = []
    for _ in range(turns):
        t0 = time.monotonic()
        r = await client.post(
            f"{url}/v1/chat/completions",
            json={"model": model, "max_tokens": max_tokens, "messages": messages, "tools": TOOLS},
        )
        r.raise_for_status()
        body = r.json()
        reply = {
            k: v
            for k, v in body["choices"][0]["message"].items()
            if v is not None and k not in ("reasoning_content", "reasoning")
        }
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        messages += [{"role": "tool", "tool_call_id": c["id"], "content": TOOL_RESULT} for c in calls] or [
            {"role": "user", "content": TOOL_RESULT + "Use a tool, or call finish."}
        ]
        usage = body["usage"]
        out.append(
            (
                time.monotonic() - t0,
                int(usage["completion_tokens"]),
                int(usage["prompt_tokens"]),
                int((body.get("qwen38") or {}).get("prefix_reused") or 0),
            )
        )
    return out


async def tasks(
    url: str, model: str, n: int, turns: int, max_tokens: int, transport: httpx.AsyncBaseTransport | None = None
) -> dict | None:
    """``n`` agents at once, ``turns`` turns each, or None when the server refuses requests at this level."""
    async with httpx.AsyncClient(timeout=900, transport=transport) as c:
        t0 = time.monotonic()
        out = await asyncio.gather(
            *(task(c, url, model, i, turns, max_tokens) for i in range(n)), return_exceptions=True
        )
        elapsed = time.monotonic() - t0
    errors = [e for e in out if isinstance(e, BaseException)]
    if any(isinstance(e, httpx.HTTPStatusError) and e.response.status_code in REFUSED for e in errors):
        return None
    if errors:
        raise errors[0]
    rows = [row for agent in out for row in agent]
    prompt = sum(r[2] for r in rows)
    return {
        "concurrency": n,
        "turns": turns,
        "total_tps": sum(r[1] for r in rows) / elapsed,
        "median_turn_s": statistics.median(r[0] for r in rows),
        "task_s": elapsed,
        "reused_share": sum(r[3] for r in rows) / prompt if prompt else 0.0,
    }


async def preflight(client: httpx.AsyncClient, url: str, model: str) -> str:
    """Fail fast with a plain answer instead of a traceback, and return the model id
    to benchmark (servers often register the model under a different name)."""
    try:
        r = await client.get(f"{url}/v1/models", timeout=10)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        raise SystemExit(
            f"No model server answering at {url} ({type(e).__name__}).\n"
            f"Check that it is running and reachable from this machine:  curl {url}/v1/models"
        ) from None
    if model in ids:
        return model
    if len(ids) == 1:
        print(f"The server serves {ids[0]!r}, not {model!r}; benchmarking {ids[0]!r}.")
        return ids[0]
    raise SystemExit(f"{model!r} is not served at {url}. Served: {', '.join(ids) or 'nothing'}. Pass --model.")


async def level(
    url: str, model: str, n: int, rounds: int, max_tokens: int, transport: httpx.AsyncBaseTransport | None = None
) -> dict | None:
    """One concurrency level, or None when the server refuses requests at it."""
    async with httpx.AsyncClient(timeout=900, transport=transport) as c:
        results = []
        t0 = time.monotonic()
        for r in range(rounds):
            out = await asyncio.gather(
                *(one(c, url, model, r * n + i, max_tokens) for i in range(n)), return_exceptions=True
            )
            errors = [e for e in out if isinstance(e, BaseException)]
            if any(isinstance(e, httpx.HTTPStatusError) and e.response.status_code in REFUSED for e in errors):
                return None
            if errors:
                raise errors[0]
            results += out
        elapsed = time.monotonic() - t0
    per_agent = [tok / sec for sec, tok in results if sec > 0]
    wall = [sec for sec, _ in results]
    return {
        "concurrency": n,
        "per_agent_tps": statistics.median(per_agent),
        "p10_agent_tps": sorted(per_agent)[max(0, len(per_agent) // 10 - 1)],
        "median_latency_s": statistics.median(wall),
        "total_tps": sum(tok for _, tok in results) / elapsed,
    }


def recommend(rows: list[dict], floor_tps: float, max_latency_s: float) -> dict:
    ok = [r for r in rows if r["p10_agent_tps"] >= floor_tps and r["median_latency_s"] <= max_latency_s]
    start = max((r["concurrency"] for r in ok), default=1)
    best_total, top = 0.0, rows[0]["concurrency"]
    for r in sorted(rows, key=lambda r: r["concurrency"]):
        if r["total_tps"] > best_total * 1.05:  # still gaining >5% aggregate throughput
            best_total, top = r["total_tps"], r["concurrency"]
    return {"start": start, "min": max(1, min(2, start)), "max": max(start, top), "floor_tps": floor_tps}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--floor", type=float, default=15.0)
    ap.add_argument("--max-latency", type=float, default=60.0)
    ap.add_argument("--turns", type=int, default=1, help="turns per agent task (1: single requests)")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    async with httpx.AsyncClient() as c:
        model = await preflight(c, url, a.model)
    if a.turns > 1:
        for n in (int(x) for x in a.levels.split(",")):
            row = await tasks(url, model, n, a.turns, a.max_tokens)
            if row is None:
                print(f"agents {n:>2}: refused, the server's queue is full; the sweep stops here")
                break
            print(
                f"agents {n:>2} x {a.turns} turns: {row['total_tps']:7.1f} tok/s total, median turn "
                f"{row['median_turn_s']:.1f}s, tasks done in {row['task_s']:.1f}s, "
                f"{100 * row['reused_share']:.0f}% of prompt tokens reused"
            )
        return
    rows = []
    for n in (int(x) for x in a.levels.split(",")):
        row = await level(url, model, n, a.rounds, a.max_tokens)
        if row is None:
            print(f"concurrency {n:>2}: refused, the server's queue is full; the sweep stops here")
            break
        rows.append(row)
        print(
            f"concurrency {n:>2}: {row['per_agent_tps']:6.1f} tok/s per agent (p10 {row['p10_agent_tps']:.1f}), "
            f"{row['total_tps']:7.1f} tok/s total, median latency {row['median_latency_s']:.1f}s"
        )
    if not rows:
        raise SystemExit("The server refused even one request at a time. Is another client using it?")
    rec = recommend(rows, a.floor, a.max_latency)
    print(
        f"\n{model}: run {rec['start']} agents at once (every one at {rec['floor_tps']:g} tok/s or more); "
        f"up to {rec['max']} still raises the total"
    )


if __name__ == "__main__":
    asyncio.run(main())

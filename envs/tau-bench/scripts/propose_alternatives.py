"""Propose K=5 alternative tool calls per expert observation in D_expert.

One LLM call per observation. The proposer is given the full conversation as text
plus a 'forbidden tool' line (the tool the expert actually used — None for respond
obs). System prompt carries the WIKI + all 16 tool schemas verbatim and is identical
across all calls (DeepSeek prompt caching will eat most of its cost after warm-up).

Output: alternatives.jsonl, one line per (task_id, trial, obs_idx). The IWM env
probe is a separate script that reads this file.

Run:
    PYTHONNOUSERSITE=1 DEEPSEEK_API_KEY=... \\
        conda run -n tau-bench-ee --no-capture-output python \\
        envs/tau-bench/scripts/propose_alternatives.py \\
        [--max-trajs N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tau_bench.envs.retail.tools import ALL_TOOLS
from tau_bench.envs.retail.wiki import WIKI

D_EXPERT_DEFAULT = "envs/tau-bench/data/rollout/D_expert.jsonl"
OUT_DEFAULT = "envs/tau-bench/data/rollout/alternatives.jsonl"

# Tools excluded from the proposer menu because they carry no IWM signal:
#   - `think`: no-op, invoke() returns ''  (a Anthropic-style reasoning scratchpad)
#   - `transfer_to_human_agents`: terminate tool; invoke output is fixed/empty
# Expert use of these is rare in D_expert (think=1, transfer=10 across 5,249 obs),
# so the rare "expert used a tool not in our proposer menu" case is acceptable.
EXCLUDED_TOOLS = {"think", "transfer_to_human_agents"}
MENU_TOOLS = [t for t in ALL_TOOLS
              if t.get_info()["function"]["name"] not in EXCLUDED_TOOLS]


def build_system_prompt() -> str:
    lines = [
        "You are evaluating ALTERNATIVE TOOL CALLS for a retail customer-service agent.",
        "",
        "The agent's policy document is shown below verbatim. Read it once, then use it",
        "as context for every request.",
        "",
        "=== POLICY DOCUMENT (the agent's WIKI) ===",
        WIKI,
        "",
        "=== AVAILABLE TOOLS ===",
    ]
    for tool in MENU_TOOLS:
        info = tool.get_info()["function"]
        name = info["name"]
        desc = info["description"]
        params = info.get("parameters", {}).get("properties", {})
        required = set(info.get("parameters", {}).get("required", []))
        sig = ", ".join(
            f"{p}{'' if p in required else '?'}: {pi.get('type','any')}"
            for p, pi in params.items()
        )
        lines.append(f"- {name}({sig}) — {desc}")
    lines += [
        "",
        "=== YOUR TASK ===",
        "Given the conversation so far and a possibly-forbidden tool, propose K=5",
        "alternative tool calls the agent might plausibly make as its next action.",
        "These will be executed against the environment to generate world-model",
        "training data, so favor reasonable-looking attempts (including some with",
        "intentionally varied arguments) over completely random nonsense.",
        "",
        "=== OUTPUT FORMAT ===",
        'Return a single JSON object: {"candidates": [{"name": "...", "arguments": {...}}, ...]}',
        "",
        "Rules:",
        "- Exactly 5 candidates.",
        "- Each candidate must use one of the tools listed above.",
        "- Candidates must be pairwise distinct (not byte-identical name+arguments).",
        "- If a 'Forbidden tool' is named in the user message, NEVER use it.",
        "- 'arguments' must be a JSON object (not a JSON-encoded string).",
        "- No prose, no markdown, no explanation — only the JSON object.",
    ]
    return "\n".join(lines)


def render_obs(messages: list[dict]) -> str:
    """Render the conversation history as plain text. The system message (WIKI) is
    skipped because it's already in our own system prompt."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        elif role == "user":
            out.append(f"[user] {m.get('content','')}")
        elif role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                for t in tcs:
                    fn = t["function"]
                    out.append(f"[assistant tool_call] {fn['name']}({fn['arguments']})")
            else:
                out.append(f"[assistant] {m.get('content','')}")
        elif role == "tool":
            name = m.get("name") or "?"
            out.append(f"[tool: {name}] {m.get('content','')}")
    return "\n".join(out)


def expert_tool_name(assistant_turn: dict) -> str | None:
    if assistant_turn.get("tool_calls"):
        return assistant_turn["tool_calls"][0]["function"]["name"]
    return None


def build_user_prompt(obs_messages: list[dict], expert_turn: dict) -> str:
    forbidden = expert_tool_name(expert_turn)
    if forbidden:
        fline = (
            f"!!! FORBIDDEN TOOL: `{forbidden}` !!!\n"
            f"The expert agent's actual next action used `{forbidden}`. You MUST NOT propose "
            f"`{forbidden}` under ANY arguments (no different IDs, no different casing, nothing). "
            f"Pick 5 candidates from the OTHER tools in the menu — explore different tool FAMILIES."
        )
    else:
        fline = (
            "FORBIDDEN TOOL: (none — the expert's actual next action was a text response, not a tool call). "
            "Pick 5 candidates that explore different tool families."
        )
    return (
        "=== CONVERSATION SO FAR ===\n"
        + render_obs(obs_messages)
        + "\n\n"
        + fline
        + "\n\nPropose 5 alternative tool calls now as a JSON object."
    )


def parse_alts(raw: str, valid_tool_names: set[str], forbidden: str | None) -> tuple[list[dict], bool]:
    """Parse + validate + dedup. Returns (alts, parse_ok)."""
    try:
        obj = json.loads(raw)
    except Exception:
        return [], False
    cands = obj.get("candidates") if isinstance(obj, dict) else None
    if not isinstance(cands, list):
        return [], False
    seen, final = set(), []
    for c in cands:
        if not isinstance(c, dict):
            continue
        name, args = c.get("name"), c.get("arguments")
        if not isinstance(name, str) or not isinstance(args, dict):
            continue
        if name == forbidden:
            continue
        if name not in valid_tool_names:
            continue  # LLM proposed a tool outside our menu (e.g. excluded `think`)
        sig = (name, json.dumps(args, sort_keys=True))
        if sig in seen:
            continue
        seen.add(sig)
        final.append({"name": name, "arguments": args})
    return final, True


def refill_to_k(alts: list[dict], all_tool_names: list[str], forbidden: str | None,
                k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    available = [n for n in all_tool_names if n != forbidden]
    out = [{**a, "source": "llm"} for a in alts[:k]]
    used_names = {a["name"] for a in out}
    while len(out) < k:
        unused = [n for n in available if n not in used_names]
        pool = unused if unused else available
        choice = rng.choice(pool)
        out.append({"name": choice, "arguments": {}, "source": "fallback_random"})
        used_names.add(choice)
    return out


def iter_jobs(d_expert: str, max_trajs: int | None):
    with open(d_expert) as f:
        for ti, line in enumerate(f):
            if max_trajs is not None and ti >= max_trajs:
                return
            r = json.loads(line)
            for i, m in enumerate(r["traj"]):
                if m["role"] == "assistant":
                    yield (ti, r["task_id"], r["trial"], i, r["traj"][:i], m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-expert", default=D_EXPERT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-trajs", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys_prompt = build_system_prompt()
    all_tool_names = [t.get_info()["function"]["name"] for t in MENU_TOOLS]
    print(f"system prompt: {len(sys_prompt):,} chars (~{len(sys_prompt)//4:,} tokens)")
    print(f"tool menu    : {len(all_tool_names)} tools")

    if args.dry_run:
        for i, (_, tid, tr, oi, obs, exp) in enumerate(iter_jobs(args.d_expert, max_trajs=1)):
            if i >= 2:
                break
            up = build_user_prompt(obs, exp)
            print()
            print(f"---- DRY [tid={tid} tr={tr} oi={oi}] expert={expert_tool_name(exp) or '(respond)'} ----")
            print(f"user prompt: {len(up):,} chars (~{len(up)//4:,} tokens)")
            print("USER PROMPT (first 1500 chars):")
            print(up[:1500])
            if len(up) > 1500:
                print(f"...(+{len(up)-1500:,} chars)")
        return

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_jobs(args.d_expert, args.max_trajs))
    print(f"jobs: {len(jobs)}")

    write_lock, stats_lock = Lock(), Lock()
    fh = open(args.out, "w")
    stats = {"calls": 0, "parse_ok": 0, "refills": 0, "in": 0, "out": 0, "cached": 0, "errors": 0}

    def work(job):
        _, tid, tr, oi, obs, exp = job
        forbidden = expert_tool_name(exp)
        try:
            res = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": build_user_prompt(obs, exp)},
                ],
                temperature=args.temp,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as e:
            return {"task_id": tid, "trial": tr, "obs_idx": oi, "error": str(e)}
        raw = res.choices[0].message.content or ""
        u = res.usage
        cached = (u.prompt_tokens_details.cached_tokens
                  if u.prompt_tokens_details else 0) or 0
        alts, ok = parse_alts(raw, set(all_tool_names), forbidden)
        refills_needed = max(0, args.k - len(alts))
        final = refill_to_k(alts, all_tool_names, forbidden, args.k, seed=hash((tid, tr, oi)) & 0xffffffff)
        return {
            "task_id": tid, "trial": tr, "obs_idx": oi,
            "expert_kind": "tool" if forbidden else "respond",
            "forbidden_tool": forbidden,
            "llm_raw": raw, "parse_ok": ok, "refills_needed": refills_needed,
            "alts": final,
            "usage": {"in": u.prompt_tokens, "out": u.completion_tokens, "cached": cached},
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            rec = fut.result()
            with write_lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            with stats_lock:
                stats["calls"] += 1
                if "error" in rec:
                    stats["errors"] += 1
                else:
                    if rec["parse_ok"]:
                        stats["parse_ok"] += 1
                    stats["refills"] += rec["refills_needed"]
                    stats["in"] += rec["usage"]["in"]
                    stats["out"] += rec["usage"]["out"]
                    stats["cached"] += rec["usage"]["cached"]
            if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-6)
                eta = (len(jobs) - i - 1) / max(rate, 1e-6)
                print(f"  [{i+1:5d}/{len(jobs)}] {rate:5.1f} req/s "
                      f"eta {eta/60:5.1f} min | {stats}")
    fh.close()

    print()
    print(f"DONE in {(time.time()-t0)/60:.2f} min")
    print(f"  calls           : {stats['calls']}")
    print(f"  parse OK        : {stats['parse_ok']}/{stats['calls']-stats['errors']} "
          f"({stats['parse_ok']/max(stats['calls']-stats['errors'],1)*100:.1f}%)")
    print(f"  refills total   : {stats['refills']} (avg {stats['refills']/max(stats['calls']-stats['errors'],1):.2f}/obs)")
    print(f"  errors          : {stats['errors']}")
    print(f"  tokens in/cached/out: {stats['in']:,} / {stats['cached']:,} / {stats['out']:,}")
    print(f"  output          : {args.out}")


if __name__ == "__main__":
    main()

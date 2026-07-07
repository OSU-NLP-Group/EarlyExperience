"""Generate self-reflection CoTs over the IWM rollout.

For each obs in iwm_rollout.jsonl:
  1. Greedy-pick 3 of 5 alts to maximize (outcome_class, tool_family) coverage.
  2. Call DeepSeek (default v4-pro for reasoning quality) with a system prompt that
     carries the WIKI + leakage-prevention guidelines (banned vocab, no numbered
     labels, convergence anchor, soft length cap via prompt only — no max_tokens).
  3. Write one JSONL line per obs with the raw reflection + annotated quality flags.
     NO records are dropped at this stage; cleanup / re-run happens downstream.

Run:
    PYTHONNOUSERSITE=1 DEEPSEEK_API_KEY=... \\
        conda run -n tau-bench-ee --no-capture-output python \\
        envs/tau-bench/scripts/propose_reflections.py \\
        [--max-trajs N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tau_bench.envs.retail.wiki import WIKI

IWM_DEFAULT = "envs/tau-bench/data/rollout/iwm_rollout.jsonl"
OUT_DEFAULT = "envs/tau-bench/data/rollout/sr_rollout.jsonl"

# ---------------------------------------------------------------------------- #
# Greedy alt selection
# ---------------------------------------------------------------------------- #

TOOL_FAMILY = {
    "find_user_id_by_email": "lookup", "find_user_id_by_name_zip": "lookup",
    "get_order_details": "lookup", "get_user_details": "lookup",
    "get_product_details": "lookup", "list_all_product_types": "lookup",
    "cancel_pending_order": "mutate_order",
    "modify_pending_order_address": "mutate_order",
    "modify_pending_order_items": "mutate_order",
    "modify_pending_order_payment": "mutate_order",
    "exchange_delivered_order_items": "mutate_order",
    "return_delivered_order_items": "mutate_order",
    "modify_user_address": "mutate_user",
    "calculate": "compute",
}


def outcome_class(nxt: str) -> str:
    if not nxt:
        return "empty"
    if nxt.startswith("Unknown action"):
        return "unknown"
    if nxt.startswith("Error:"):
        nl = nxt.lower()
        if "not found" in nl:
            return "not_found"
        if "missing" in nl and "required positional" in nl:
            return "arg_error"
        return "business_rule"
    return "ok"


def tool_family(name: str) -> str:
    return TOOL_FAMILY.get(name, "unknown_fam")


def greedy_pick(alts: list[dict]) -> tuple[list[dict], dict]:
    """Pick 3 alts maximizing (outcome_class, tool_family) coverage.
    Returns (picked, diagnostic_dict)."""
    tagged = [
        {**a, "outcome_class": outcome_class(a["next_obs"]),
              "tool_family": tool_family(a["action"]["name"])}
        for a in alts
    ]
    chosen: list[dict] = []
    seen_classes, seen_families = set(), set()
    pool = list(tagged)

    # Priority 1: ensure at least one 'ok' candidate if any exists (anchor for contrast)
    ok_cands = [t for t in pool if t["outcome_class"] == "ok"]
    if ok_cands:
        c = ok_cands[0]
        chosen.append(c)
        seen_classes.add(c["outcome_class"])
        seen_families.add(c["tool_family"])
        pool.remove(c)

    # Rounds 2-3: greedy max-coverage
    while len(chosen) < 3 and pool:
        def cov_score(t):
            return (t["outcome_class"] not in seen_classes) + (t["tool_family"] not in seen_families)
        pool.sort(key=cov_score, reverse=True)
        c = pool[0]
        chosen.append(c)
        seen_classes.add(c["outcome_class"])
        seen_families.add(c["tool_family"])
        pool.remove(c)

    fallback = False
    while len(chosen) < 3 and tagged:  # only triggers if alts<3 originally — shouldn't happen
        fallback = True
        chosen.append(tagged.pop(0))

    return chosen[:3], {
        "distinct_classes": len({c["outcome_class"] for c in chosen}),
        "distinct_families": len({c["tool_family"] for c in chosen}),
        "fallback_random_used": fallback,
    }


# ---------------------------------------------------------------------------- #
# Rendering
# ---------------------------------------------------------------------------- #

def render_obs(messages: list[dict]) -> str:
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
            out.append(f"[tool: {m.get('name','?')}] {m.get('content','')}")
    return "\n".join(out)


def render_action(action: dict, kind: str) -> str:
    """Render an action (tool call or respond) as a single descriptive line."""
    if kind == "tool":
        return f'{action["name"]}({json.dumps(action.get("arguments", {}))})'
    # respond
    content = action.get("arguments", {}).get("content", "")
    return f'said to the customer: "{content}"'


def truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + f"…[truncated +{len(s)-n} chars]"


# ---------------------------------------------------------------------------- #
# Prompts
# ---------------------------------------------------------------------------- #

def build_system_prompt() -> str:
    return f"""You are reflecting on a single customer-service decision inside a retail support conversation. Your output is the agent's own internal monologue at the moment that decision was made.

The agent's policy document is shown verbatim below — treat it as fixed background context.

=== POLICY DOCUMENT (the retail agent's WIKI) ===
{WIKI}

=== YOUR TASK ===
You will be shown:
  - the conversation so far,
  - the action the agent went with at this point and the observation it produced,
  - a small set of OTHER actions that could have been considered, each with what that action would have produced if it had been taken instead.

Write one continuous first-person internal monologue (no headings, no markdown bullets) that:
  1. Frames the customer's goal and the constraints at this moment.
  2. Walks through the other actions you might have considered, explaining for each why it would have been less helpful — anchor every comparison in the OBSERVATION that action would have produced and what the policy says.
  3. Justifies the action you went with, grounded in what it actually produced and what the policy required.
  4. Surfaces any subtle clues, constraints, or consequences from the situation.

=== HARD OUTPUT CONSTRAINTS — READ CAREFULLY ===

* Length: target 200-400 words. Soft maximum 500. Write ONE complete monologue; do not pad, do not start a second monologue, do not duplicate your own paragraph.

* Voice: first-person ("I", "we"). This will be used to train an agent's inference-time reasoning, so it must read as that agent's own thinking — not as an evaluator or a teacher describing what someone else did.

* BANNED VOCABULARY in your output. These words leak supervision labels and corrupt the training signal. Do NOT use any of them, in any case:
    expert, expert action, selected action, chosen action, the chosen, the selected,
    correct action, right action, optimal action, best option, best alternative,
    best move, best choice, gold, ground truth, the action taken here, the action we took
  Refer to the action that was actually taken with natural phrasing such as: "I went with X because…", "Calling Y was the move that fit because…", "I decided to do Z since…".

* DO NOT reference the other actions by numbered/lettered labels. Forbidden: "Action 1", "Option 2", "Alternative 3", "Choice A", "the first alternative", "a_i^1". Instead use natural inline phrasing such as: "I could have tried X, but…", "Another option I considered was Y…", "Reaching for Z would have…".

* The monologue MUST converge on the action that was actually taken. Do NOT end by recommending a different action; do NOT hedge that an alternative would be "equally good".

* Stay strictly inside the information provided (conversation, observation outputs, policy). Do not invent IDs, prices, or facts.

* No meta-commentary about being an AI, no apologies, no "let me explain", no opening fluff like "Okay, so…"."""


def build_user_prompt(obs_messages, expert_kind, expert_action, expert_next_obs, picked_alts):
    parts = ["=== CONVERSATION SO FAR ===", render_obs(obs_messages), ""]
    parts.append("=== THE ACTION I WENT WITH AT THIS POINT ===")
    parts.append(render_action(expert_action, expert_kind))
    parts.append("")
    parts.append("=== WHAT THAT ACTION PRODUCED ===")
    parts.append(truncate(expert_next_obs, 600))
    parts.append("")
    parts.append("=== OTHER ACTIONS I COULD HAVE TAKEN AT THIS SAME POINT (each shown with what it WOULD have produced if I had taken it instead) ===")
    for alt in picked_alts:
        alt_str = f'{alt["action"]["name"]}({json.dumps(alt["action"].get("arguments", {}))})'
        nxt = truncate(alt["next_obs"], 400)
        parts.append(f"- {alt_str} would have produced: {nxt}")
    parts.append("")
    parts.append("Now write the single internal monologue described in the system prompt.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------- #
# Flag detection (no filtering — just annotation)
# ---------------------------------------------------------------------------- #

BANNED_VOCAB = [
    "expert", "expert action", "selected action", "chosen action", "the chosen",
    "the selected", "correct action", "right action", "optimal action",
    "best option", "best alternative", "best move", "best choice",
    "gold", "ground truth", "the action taken here", "the action we took",
]
NUMBERED_LABEL_RE = re.compile(
    r"\b(action|option|alternative|choice)\s+[1-9a-eA-E]\b|"
    r"\ba_i\s*\^?\s*[1-9]\b|"
    r"\b(first|second|third|fourth|fifth)\s+(alternative|option|choice|action)\b",
    re.IGNORECASE,
)


def detect_flags(text: str) -> dict:
    flags: dict = {}
    lower = text.lower()

    # is_doubled: first 100 chars appear again somewhere past char 100
    head = text[:100].strip()
    if head and len(text) > 250:
        idx = text.find(head, 100)
        if idx > -1:
            flags["is_doubled"] = True
            # Safe to dedup if the two halves are similar length (within 50 chars)
            half_len = idx  # length of first half
            second_half_len = len(text) - idx
            flags["doubled_safe_to_dedup"] = abs(half_len - second_half_len) < 50
        else:
            flags["is_doubled"] = False
            flags["doubled_safe_to_dedup"] = False
    else:
        flags["is_doubled"] = False
        flags["doubled_safe_to_dedup"] = False

    # banned vocab
    hits = [v for v in BANNED_VOCAB if v in lower]
    flags["has_banned_vocab"] = bool(hits)
    flags["banned_vocab_hits"] = hits

    # numbered labels
    m = NUMBERED_LABEL_RE.search(text)
    flags["has_numbered_label"] = bool(m)
    flags["numbered_label_hit"] = m.group(0) if m else None

    # paragraph breaks
    flags["has_paragraph_breaks"] = bool(re.search(r"\n\s*\n", text))

    # word count
    wc = len(text.split())
    flags["word_count"] = wc
    flags["exceeded_soft_cap"] = wc > 500
    return flags


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #

def iter_jobs(iwm_path: str, max_trajs: int | None):
    """Yield (obs_record, traj_obs_messages_so_far_for_render)."""
    # We need to rebuild conversation history per obs from D_expert (the original
    # messages list). iwm_rollout has only the action + next_obs, not the rendered
    # obs context. Load D_expert and index it.
    d_expert_path = "envs/tau-bench/data/rollout/D_expert.jsonl"
    by_key = {}
    with open(d_expert_path) as f:
        for line in f:
            r = json.loads(line)
            by_key[(r["task_id"], r["trial"])] = r["traj"]

    seen_trajs = set()
    with open(iwm_path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["task_id"], r["trial"])
            traj = by_key.get(key)
            if traj is None:
                continue
            if max_trajs is not None:
                seen_trajs.add(key)
                if len(seen_trajs) > max_trajs:
                    return
            obs_msgs = traj[:r["obs_idx"]]
            yield r, obs_msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iwm", default=IWM_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--frequency-penalty", type=float, default=0.3)
    ap.add_argument("--max-trajs", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true",
                    help="show 2 sample prompts (no API call)")
    args = ap.parse_args()

    sys_prompt = build_system_prompt()
    print(f"system prompt: {len(sys_prompt):,} chars (~{len(sys_prompt)//4:,} tokens)")

    if args.dry_run:
        for i, (r, obs_msgs) in enumerate(iter_jobs(args.iwm, max_trajs=1)):
            if i >= 2:
                break
            picked, diag = greedy_pick(r["alts"])
            up = build_user_prompt(obs_msgs, r["expert_kind"],
                                   r["expert"]["action"], r["expert"]["next_obs"], picked)
            print()
            print(f"---- DRY [tid={r['task_id']} tr={r['trial']} oi={r['obs_idx']} kind={r['expert_kind']}] ----")
            print(f"user prompt: {len(up):,} chars (~{len(up)//4:,} tokens)")
            print(f"picked tags: {[(p['outcome_class'], p['tool_family']) for p in picked]}  diag={diag}")
            print("USER PROMPT (first 1800 chars):")
            print(up[:1800])
            if len(up) > 1800:
                print(f"...(+{len(up)-1800:,} chars)")
        return

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_jobs(args.iwm, max_trajs=args.max_trajs))
    print(f"jobs: {len(jobs)}")

    write_lock, stats_lock = Lock(), Lock()
    fh = open(args.out, "w")
    stats = {"calls": 0, "errors": 0, "in": 0, "out": 0, "cached": 0,
             "is_doubled": 0, "has_banned_vocab": 0, "has_numbered_label": 0,
             "exceeded_soft_cap": 0, "wc_total": 0}

    def work(job):
        r, obs_msgs = job
        picked, diag = greedy_pick(r["alts"])
        up = build_user_prompt(obs_msgs, r["expert_kind"],
                               r["expert"]["action"], r["expert"]["next_obs"], picked)
        try:
            res = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": up}],
                temperature=args.temperature,
                frequency_penalty=args.frequency_penalty,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as e:
            return {"task_id": r["task_id"], "trial": r["trial"], "obs_idx": r["obs_idx"],
                    "error": str(e), "attempt": 1, "model": args.model}
        raw = res.choices[0].message.content or ""
        u = res.usage
        cached = (u.prompt_tokens_details.cached_tokens
                  if u.prompt_tokens_details else 0) or 0
        flags = detect_flags(raw)
        rec = {
            "task_id": r["task_id"], "trial": r["trial"], "obs_idx": r["obs_idx"],
            "expert_kind": r["expert_kind"],
            "expert_action": r["expert"]["action"],
            "expert_next_obs": r["expert"]["next_obs"],
            "picked_alts": [
                {"action": p["action"], "next_obs": p["next_obs"],
                 "source": p.get("source", "llm"),
                 "outcome_class": p["outcome_class"], "tool_family": p["tool_family"]}
                for p in picked
            ],
            "alt_selection_diagnostic": diag,
            "reflection_raw": raw,
            "flags": flags,
            "model": args.model, "attempt": 1,
            "usage": {"in": u.prompt_tokens, "out": u.completion_tokens, "cached": cached},
        }
        return rec

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
                    stats["in"] += rec["usage"]["in"]
                    stats["out"] += rec["usage"]["out"]
                    stats["cached"] += rec["usage"]["cached"]
                    f = rec["flags"]
                    for k in ("is_doubled", "has_banned_vocab", "has_numbered_label", "exceeded_soft_cap"):
                        if f.get(k):
                            stats[k] += 1
                    stats["wc_total"] += f.get("word_count", 0)
            if (i + 1) % 25 == 0 or (i + 1) == len(jobs):
                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-6)
                eta = (len(jobs) - i - 1) / max(rate, 1e-6)
                print(f"  [{i+1:5d}/{len(jobs)}] {rate:5.2f} req/s eta {eta/60:5.1f} min | {stats}")
    fh.close()
    print()
    print(f"DONE in {(time.time()-t0)/60:.2f} min")
    ok = stats["calls"] - stats["errors"]
    print(f"  calls / errors          : {stats['calls']} / {stats['errors']}")
    print(f"  avg word count          : {stats['wc_total']/max(ok,1):.0f}")
    print(f"  flags (count / pct of ok):")
    for k in ("is_doubled", "has_banned_vocab", "has_numbered_label", "exceeded_soft_cap"):
        print(f"    {k:22s} {stats[k]:5d}  ({stats[k]/max(ok,1)*100:5.2f}%)")
    print(f"  tokens in/cached/out    : {stats['in']:,} / {stats['cached']:,} / {stats['out']:,}")
    print(f"  output                  : {args.out}")


if __name__ == "__main__":
    main()

"""
Phase 3 SR (self-reflection) smoke for BFCL v4.

For each expert fcall state in the IWM-summarized rollout, randomly subsample
K=3 valid alt actions (with their summarized outcomes), build a reflection
prompt, and call DeepSeek to produce the agent's internal monologue.

Guardrails inherited from SciW pitfalls (pitfalls.md §4 + §5):
  - banned vocab: "expert" / "selected" / "chosen" / "correct" / "best" / ...
  - no numbered alt labels ("Action 1", "Alternative")
  - single paragraph, convergence anchor (must end committing to expert action)
  - frequency_penalty=0.3 to combat long-form paragraph-duplication mode collapse

Run:
    # dry-run
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/smoke_sr.py --dry-run \
      --in envs/bfcl_v4/data/rollout/iwm_smoke_n50_summarized.jsonl \
      --out envs/bfcl_v4/data/rollout/sr_smoke_n50.jsonl

    # live
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/smoke_sr.py \
      --in  envs/bfcl_v4/data/rollout/iwm_smoke_n50_summarized.jsonl \
      --out envs/bfcl_v4/data/rollout/sr_smoke_n50.jsonl \
      --workers 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "envs/bfcl_v4/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
FUNC_DOC = DATA / "multi_turn_func_doc"
OUT_DIR = REPO / "envs/bfcl_v4/data/rollout"

CLASS_TO_FILE_STEM = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SR_SYSTEM = (
    "You roleplay an agent who has just decided on an action in a multi-turn "
    "function-calling environment. Write your INTERNAL THOUGHT MONOLOGUE explaining "
    "the reasoning that led to your action.\n\n"
    "You will be shown:\n"
    "  - The conversation history so far\n"
    "  - The current environment state\n"
    "  - The available tools\n"
    "  - The action you took at this step (what you actually called)\n"
    "  - A few other tool calls you also considered, with what their outcomes would "
    "have been if you had run them\n\n"
    "Your output is the monologue itself — a single paragraph, written in first person "
    "at decision time, that:\n"
    "  - reasons through the situation (what is needed, what the current state offers, "
    "what constraints apply)\n"
    "  - naturally considers possibilities you weighed (in fluid natural language)\n"
    "  - arrives at and commits to the action you took\n"
    "  - ends by stating that action\n\n"
    "STRICT FORMAT CONSTRAINTS (output will be discarded if you violate any):\n"
    "  - Single paragraph. No headings, no bullets, no numbered lists.\n"
    "  - Do NOT use these words: 'expert', 'selected', 'chosen', 'correct', 'right', "
    "'best', 'optimal', 'preferred', 'ideal', 'official', 'recommended'. These reveal "
    "training labels that should never appear in the agent's own thinking.\n"
    "  - Do NOT refer to alternatives by labels like 'Action 1', 'Option B', "
    "'Alternative #2', 'the first/second option', 'option (a)'. Use only natural inline "
    "phrasing: 'I could have tried X', 'Another approach would have been to Y', 'What if "
    "I instead Z'd?'.\n"
    "  - Do NOT explicitly mention that you were shown alternatives, or that you are "
    "comparing against any reference. Frame all reasoning as your own internal deliberation.\n"
    "  - Stay grounded in the actual observed outcomes — when discussing a possibility's "
    "effect, refer only to what is shown in its outcome note. Do not invent unrelated "
    "consequences.\n"
    "  - CONVERGENCE: the monologue MUST end by committing to and stating the action you "
    "took. Do NOT trail off into 'I'm not sure', do NOT recommend a different action.\n\n"
    "Length: aim for 200-300 words. Hard upper bound around 500 words. Avoid filler.\n\n"
    "Output ONLY the monologue text. No prefix, no quotes around the action, no metadata."
)


def load_tool_schemas(involved_classes: list[str]) -> list[dict]:
    schemas = []
    for cls in involved_classes:
        stem = CLASS_TO_FILE_STEM.get(cls)
        if not stem:
            continue
        with (FUNC_DOC / f"{stem}.json").open() as f:
            for line in f:
                line = line.strip()
                if line:
                    schemas.append(json.loads(line))
    return schemas


def render_history_with_summaries(prior_steps: list[dict], current_step: dict) -> str:
    """Flatten prior expert steps into a chat-like text block using summarized outcomes."""
    lines = []
    for r in prior_steps:
        if r.get("user_msg_for_turn"):
            lines.append(f"\n[USER turn {r['turn_idx']}]: {r['user_msg_for_turn']}")
        for c, s in zip(r["expert_emit_decoded"], r.get("expert_summaries") or []):
            lines.append(f"  [YOU did]: {c}")
            lines.append(f"  [OBSERVED]: {s}")
    if current_step.get("user_msg_for_turn"):
        lines.append(f"\n[USER current turn {current_step['turn_idx']}]: "
                     f"{current_step['user_msg_for_turn']}")
    return "\n".join(lines).strip()


def _tool_desc(s: dict) -> str:
    """Extract the per-tool description part, dropping the boilerplate prefix.
    Most BFCL descriptions look like:
        'This tool belongs to the X, which provides Y. Tool description: REAL DESC.'
    Pull only REAL DESC."""
    d = s.get("description", "")
    marker = "Tool description:"
    if marker in d:
        d = d.split(marker, 1)[1].strip()
    return d[:200]  # cap length per tool to control prompt size


def build_sr_user(history: str, state_snap: dict, tool_schemas: list[dict],
                   expert_calls: list[str], expert_summaries: list,
                   alts_sampled: list[dict]) -> str:
    # tool schemas: name + per-tool description (post-boilerplate)
    schema_block = json.dumps(
        [{"name": s["name"], "description": _tool_desc(s)} for s in tool_schemas],
        indent=None,
    )
    state_block = json.dumps(state_snap, default=str, indent=2)
    if len(state_block) > 2500:
        state_block = state_block[:2500] + "\n…[state truncated]…"
    if len(history) > 4000:
        history = "…[history truncated]…\n" + history[-4000:]

    expert_lines = []
    for c, s in zip(expert_calls, expert_summaries or [None]*len(expert_calls)):
        expert_lines.append(f"  Call:    {c}")
        expert_lines.append(f"  Outcome: {s or '(no summary)'}")

    alt_lines = []
    for a in alts_sampled:
        alt_lines.append(f"  Call:    {a['call']}")
        alt_lines.append(f"  Outcome would have been: {a['summary']}")

    expert_call_disp = " then ".join(expert_calls) if len(expert_calls) > 1 else expert_calls[0]

    return "\n\n".join([
        f"## Conversation history so far\n{history}",
        f"## State at this decision point\n```json\n{state_block}\n```",
        f"## Available tools (name + short description)\n{schema_block}",
        f"## The action you took at this step\n" + "\n".join(expert_lines),
        f"## Other tool calls you also considered\n" + "\n".join(alt_lines),
        (f"## Your task\nWrite your internal monologue explaining your reasoning at this "
         f"decision point, per the system rules. End by committing to (and stating) the "
         f"action you took:\n  {expert_call_disp}"),
    ])


def call_sr(client: OpenAI, system: str, user: str, model: str,
            temperature: float = 0.7, freq_penalty: float = 0.3,
            max_retries: int = 1) -> dict:
    last = None
    for attempt in range(max_retries + 1):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
                frequency_penalty=freq_penalty,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return {
                "content": (r.choices[0].message.content or "").strip(),
                "usage": {
                    "prompt_tokens": r.usage.prompt_tokens,
                    "completion_tokens": r.usage.completion_tokens,
                },
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < max_retries:
                time.sleep(0.5)
                continue
            return {"content": None, "usage": None, "error": f"{type(e).__name__}: {e}"}
    return {"content": None, "usage": None, "error": f"unknown: {last}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k-alts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = [json.loads(l) for l in args.inp.open() if l.strip()]
    by_case = defaultdict(list)
    for r in records:
        by_case[r["case_id"]].append(r)
    for cid in by_case:
        by_case[cid].sort(key=lambda r: r["global_emit_idx"])

    jobs = []
    skipped_no_alts = 0
    for r in records:
        case_recs = by_case[r["case_id"]]
        prior = [pr for pr in case_recs if pr["global_emit_idx"] < r["global_emit_idx"]]
        # valid alts: parsed AND have summary AND no Python exec exception
        valid_alts = [
            a for a in r.get("alts", [])
            if a.get("valid") and a.get("summary") and not a.get("exec_error")
        ]
        if len(valid_alts) == 0:
            skipped_no_alts += 1
            continue
        rng = random.Random(args.seed + r["global_emit_idx"])
        sampled = (rng.sample(valid_alts, args.k_alts)
                   if len(valid_alts) >= args.k_alts else valid_alts)
        jobs.append({
            "case_id": r["case_id"], "turn_idx": r["turn_idx"], "step_idx": r["step_idx"],
            "global_emit_idx": r["global_emit_idx"],
            "involved_classes": r["involved_classes"],
            "prior_steps": prior,
            "state_snap": r["s_i_state"],
            "expert_emit_decoded": r["expert_emit_decoded"],
            "expert_summaries": r.get("expert_summaries") or [],
            "user_msg_for_turn": r.get("user_msg_for_turn"),
            "alts_sampled": [{"call": a["call"], "summary": a["summary"]} for a in sampled],
        })

    print(f"\n{'='*60}\nPRE-CALL GATE (TEAM_GUIDE §2)\n{'='*60}")
    print(f"  task: BFCL SR smoke — self-reflection monologue per expert state")
    print(f"  input: {args.inp.relative_to(REPO)} ({len(records)} state records)")
    print(f"  jobs : {len(jobs)} reflections   (skipped {skipped_no_alts} states with no valid alt)")
    print(f"  K_alts per state: {args.k_alts}")
    print(f"  model: {args.model} (thinking disabled, T=0.7, freq_penalty=0.3)")
    print(f"  workers: {args.workers}")
    avg_in = 5500; avg_out = 500
    print(f"  est tokens: ~{len(jobs)*avg_in/1000:.0f}k input + ~{len(jobs)*avg_out/1000:.0f}k output "
          f"≈ ~{len(jobs)*(avg_in+avg_out)/1000:.0f}k total")
    if "pro" in args.model:
        print(f"  est cost  : ~$1-3 at pro pricing")
    else:
        print(f"  est cost  : ~$0.30-0.80 at flash pricing")
    print(f"  dry-run: {args.dry_run}\n{'='*60}\n")

    if args.dry_run:
        if jobs:
            j = jobs[0]
            tool_schemas = load_tool_schemas(j["involved_classes"])
            history = render_history_with_summaries(j["prior_steps"], j)
            user_content = build_sr_user(history, j["state_snap"], tool_schemas,
                                          j["expert_emit_decoded"], j["expert_summaries"],
                                          j["alts_sampled"])
            print(f"--- sample SR prompt (job[0]: {j['case_id']} t{j['turn_idx']} s{j['step_idx']}) ---")
            print(f"[SYSTEM] {len(SR_SYSTEM)} chars")
            print(f"[USER]   {len(user_content)} chars:\n")
            print(user_content[:3500] + ("\n…[truncated]…" if len(user_content) > 3500 else ""))
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set"); return 2
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # pre-load tool schemas per case (cache)
    case_schemas: dict[str, list[dict]] = {}
    for j in jobs:
        cid = j["case_id"]
        if cid not in case_schemas:
            case_schemas[cid] = load_tool_schemas(j["involved_classes"])

    results: dict[int, dict] = {}
    tok_in = tok_out = 0; n_err = 0
    t0 = time.time()

    def work(idx):
        j = jobs[idx]
        history = render_history_with_summaries(j["prior_steps"], j)
        user_content = build_sr_user(history, j["state_snap"], case_schemas[j["case_id"]],
                                       j["expert_emit_decoded"], j["expert_summaries"],
                                       j["alts_sampled"])
        return idx, call_sr(client, SR_SYSTEM, user_content, args.model)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, i): i for i in range(len(jobs))}
        done = 0
        for fut in as_completed(futures):
            idx, r = fut.result()
            results[idx] = r
            done += 1
            if r["error"]:
                n_err += 1
            else:
                tok_in += r["usage"]["prompt_tokens"]
                tok_out += r["usage"]["completion_tokens"]
            if done % 25 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)} done, errors so far: {n_err}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fout:
        for idx in sorted(results.keys()):
            j = jobs[idx]; r = results[idx]
            out_rec = {
                "case_id": j["case_id"],
                "turn_idx": j["turn_idx"],
                "step_idx": j["step_idx"],
                "global_emit_idx": j["global_emit_idx"],
                "involved_classes": j["involved_classes"],
                "expert_emit_decoded": j["expert_emit_decoded"],
                "expert_summaries": j["expert_summaries"],
                "alts_sampled": j["alts_sampled"],
                "reflection": r["content"],
                "usage": r["usage"],
                "error": r["error"],
            }
            fout.write(json.dumps(out_rec, default=str) + "\n")

    elapsed = time.time() - t0
    print(f"\n{'='*60}\nSR SMOKE COMPLETE\n{'='*60}")
    print(f"  output: {args.out.relative_to(REPO)}")
    print(f"  jobs  : {len(jobs)}, errors: {n_err}")
    print(f"  tokens: input {tok_in:,}  output {tok_out:,}  total {tok_in+tok_out:,}")
    print(f"  wall  : {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

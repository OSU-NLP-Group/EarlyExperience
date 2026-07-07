"""
IWM rollout summarizer — converts every (action, raw tool response) pair
in an IWM rollout file into a 1-sentence natural-language summary, matching
paper §B.3 Training Example style.

  Successful action ↦ specific verb summary ("Moved final_report.pdf to temp")
  Read action       ↦ "Found that ..."
  Error / irrelevant↦ EXACTLY "Cannot help fulfill the user's task."  (paper template)

Concurrent (ThreadPoolExecutor); writes incremental output (crash-safe).

Run:
    # smoke summarize (N=44 rollout)
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/summarize_iwm_rollout.py \
      --in  envs/bfcl_v4/data/rollout/iwm_smoke_n8.jsonl \
      --out envs/bfcl_v4/data/rollout/iwm_smoke_n8_summarized.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI

REPO = Path(__file__).resolve().parents[3]

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SUMMARIZER_SYSTEM = (
    "You translate raw tool execution outcomes into concise English. Given a function "
    "call and its raw return value from a simulated environment, output ONE sentence "
    "that states FACTUALLY what the tool did or what data it returned.\n\n"
    "Rules:\n"
    "  - Successful action (file moved, directory created, value set, message sent, "
    "data written, etc.) → state what changed, past tense, naming the affected object(s).\n"
    "    e.g. cd(folder='document') → {\"current_working_directory\":\"document\"}\n"
    "         output: \"Working directory changed to 'document'.\"\n"
    "    e.g. mv(source='a',destination='b/') → {\"result\":\"'a' moved to 'b/a'\"}\n"
    "         output: \"Moved 'a' into 'b/'.\"\n"
    "  - Read-only call (ls, pwd, get_*, list_*, view_*, search_*) → state what data was returned.\n"
    "    e.g. ls() → {\"current_directory_content\":[\"a\",\"b\"]}\n"
    "         output: \"Current directory contains: a, b.\"\n"
    "    e.g. get_tweet(tweet_id=0) → {\"id\":0,\"content\":\"Excited!\",\"username\":\"x\"}\n"
    "         output: \"Retrieved tweet id=0 by user 'x' with content 'Excited!'.\"\n"
    "  - Error response (any returned value with an error field, exception text, or "
    "rejection message) → restate the error factually.\n"
    "    e.g. {\"error\":\"grep: X: No such file or directory\"}\n"
    "         output: \"Error: file 'X' not found in current directory.\"\n"
    "    e.g. {\"error\":\"User not authenticated. Please authenticate first.\"}\n"
    "         output: \"Error: user not authenticated.\"\n"
    "  - Null/empty return → describe the call as completing with no return value.\n\n"
    "STRICT CONSTRAINTS:\n"
    "  - DO NOT mention any user, any task, any goal, or whether the call was useful.\n"
    "  - DO NOT compare to expected outcomes, expert actions, or other alternatives.\n"
    "  - DO NOT add reasoning, judgment, or commentary about correctness.\n"
    "  - Preserve any specific identifiers / values present in the raw response.\n"
    "Output ONLY the one factual sentence — no quotes, no prefix, no explanation."
)


def build_summarizer_user(user_task: str, prior_actions_summary: str,
                           call_str: str, raw_response: str) -> str:
    """Build the user content for one summarization call.

    NOTE: user_task and prior_actions_summary are accepted for API stability but
    intentionally NOT injected into the prompt. The summarizer must produce a
    purely descriptive summary that does not depend on the user's broader task,
    so that downstream SR (and only SR) carries the task-relevance reasoning.
    """
    return (
        f"## Tool call\n`{call_str}`\n\n"
        f"## Raw tool response\n```\n{raw_response}\n```\n\n"
        f"## Output\nOne factual sentence describing what the tool did or returned, "
        f"per the system rules. No mention of user, task, goal, or correctness."
    )


def call_summarizer(client: OpenAI, user_task: str, prior_summary: str,
                    call_str: str, raw_response: str,
                    temperature: float = 0.3, max_retries: int = 1) -> dict:
    """Make one summarization call. Returns {summary, usage, error}."""
    user_content = build_summarizer_user(user_task, prior_summary, call_str, raw_response)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SUMMARIZER_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                temperature=temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
            summary = (r.choices[0].message.content or "").strip()
            # strip surrounding quotes if model added them
            if summary.startswith('"') and summary.endswith('"'):
                summary = summary[1:-1].strip()
            return {
                "summary": summary,
                "usage": {
                    "prompt_tokens": r.usage.prompt_tokens,
                    "completion_tokens": r.usage.completion_tokens,
                },
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                time.sleep(0.5 + attempt * 0.5)
                continue
            return {"summary": None, "usage": None, "error": f"{type(e).__name__}: {e}"}
    return {"summary": None, "usage": None, "error": f"unknown failure: {last_err}"}


def _stringify(raw):
    if raw is None: return ""
    if isinstance(raw, str): return raw
    try: return json.dumps(raw, default=str)
    except Exception: return str(raw)


def collect_calls_to_summarize(records: list[dict]) -> list[dict]:
    """Walk all rollout records, emit summarization jobs.

    Each job = {kind, record_idx, sub_key, user_task, prior_summary,
                call_str, raw_response}
    where kind ∈ {expert, alt}; sub_key locates the slot for write-back.
    """
    jobs = []
    # build (case, turn) → user_msg map by scanning records in order
    turn_user_msg = {}
    for rec in records:
        if rec.get("is_first_step_of_turn", False) is False:
            pass  # not first step
        if rec.get("user_msg_for_turn"):
            turn_user_msg[(rec["case_id"], rec["turn_idx"])] = rec["user_msg_for_turn"]

    # collect: 1 job per expert call + 1 job per valid alt
    for i, rec in enumerate(records):
        cid = rec["case_id"]; tidx = rec["turn_idx"]
        user_task = turn_user_msg.get((cid, tidx), "(unknown task)")
        # prior_summary for this record: we don't have summaries of prior steps yet
        # (they're being computed); pass empty for now. Could be filled in a 2-pass if needed.
        prior = ""
        # expert calls in this record
        for j, (call, raw) in enumerate(
            zip(rec.get("expert_emit_decoded", []),
                rec.get("expert_tool_responses_recorded") or [])
        ):
            jobs.append({
                "kind": "expert",
                "record_idx": i, "sub_key": ("expert", j),
                "user_task": user_task, "prior_summary": prior,
                "call_str": call, "raw_response": _stringify(raw),
            })
        # alt calls
        for j, alt in enumerate(rec.get("alts", [])):
            if not alt.get("valid"): continue
            raw = alt.get("exec_result")
            if raw is None and alt.get("exec_error"):
                # for Python-level exec errors, summary should be the canonical "cannot help"
                # but we still pass it through summarizer (could deterministic-template instead);
                # keep through LLM for consistency.
                raw = f"(exec error: {alt['exec_error']})"
            jobs.append({
                "kind": "alt",
                "record_idx": i, "sub_key": ("alt", j),
                "user_task": user_task, "prior_summary": prior,
                "call_str": alt.get("call"), "raw_response": _stringify(raw),
            })
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10,
                        help="concurrent threads")
    parser.add_argument("--dry-run", action="store_true",
                        help="print plan + sample prompt, no API call")
    args = parser.parse_args()

    records = [json.loads(l) for l in args.inp.open() if l.strip()]
    jobs = collect_calls_to_summarize(records)

    # ----- pre-call gate -----
    n_expert = sum(1 for j in jobs if j["kind"] == "expert")
    n_alt = sum(1 for j in jobs if j["kind"] == "alt")
    print(f"\n{'='*60}\nPRE-CALL GATE (TEAM_GUIDE §2)\n{'='*60}")
    print(f"  task: IWM rollout summarizer (raw responses → 1-sentence summaries)")
    print(f"  input file: {args.inp.relative_to(REPO)} ({len(records)} state records)")
    print(f"  jobs: {len(jobs)} total ({n_expert} expert + {n_alt} alt)")
    print(f"  model: {DEEPSEEK_MODEL} (thinking disabled, temperature 0.3)")
    print(f"  concurrency: {args.workers} threads")
    # rough token estimate
    avg_in = 600; avg_out = 35
    print(f"  est tokens: ~{len(jobs)*avg_in/1000:.0f}k input + ~{len(jobs)*avg_out/1000:.0f}k output ≈ ~{len(jobs)*(avg_in+avg_out)/1000:.0f}k total")
    print(f"  est cost  : ~$0.05-0.20 at flash pricing")
    print(f"  dry-run   : {args.dry_run}")
    print(f"{'='*60}\n")

    if args.dry_run:
        if jobs:
            j = jobs[0]
            content = build_summarizer_user(j["user_task"], j["prior_summary"],
                                              j["call_str"], j["raw_response"])
            print("--- sample summarization prompt (job[0]) ---")
            print(f"[SYSTEM] {len(SUMMARIZER_SYSTEM)} chars: {SUMMARIZER_SYSTEM[:200]}…")
            print(f"[USER]   {len(content)} chars:")
            print(content[:1500] + ("\n…[truncated]…" if len(content) > 1500 else ""))
        return 0

    # ----- LIVE -----
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set"); return 2
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # parallel run
    summaries: dict[int, dict] = {}
    write_lock = Lock()
    tok_in = 0; tok_out = 0; n_err = 0
    t0 = time.time()

    def work(idx):
        j = jobs[idx]
        r = call_summarizer(client, j["user_task"], j["prior_summary"],
                            j["call_str"], j["raw_response"])
        return idx, r

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, i): i for i in range(len(jobs))}
        done_n = 0
        for fut in as_completed(futures):
            idx, r = fut.result()
            summaries[idx] = r
            done_n += 1
            if r["error"]:
                n_err += 1
            else:
                tok_in += r["usage"]["prompt_tokens"]
                tok_out += r["usage"]["completion_tokens"]
            if done_n % 50 == 0 or done_n == len(jobs):
                print(f"  ... {done_n}/{len(jobs)} done, errors so far: {n_err}")

    # write back into records
    for idx, r in summaries.items():
        j = jobs[idx]; rec = records[j["record_idx"]]
        kind, sub_idx = j["sub_key"]
        if kind == "expert":
            rec.setdefault("expert_summaries", [None]*len(rec.get("expert_emit_decoded", [])))
            rec["expert_summaries"][sub_idx] = r["summary"]
        else:  # alt
            rec["alts"][sub_idx]["summary"] = r["summary"]
            if r["error"]:
                rec["alts"][sub_idx]["summary_error"] = r["error"]

    # output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fout:
        for rec in records:
            fout.write(json.dumps(rec, default=str) + "\n")

    elapsed = time.time() - t0
    print(f"\n{'='*60}\nSUMMARIZER COMPLETE\n{'='*60}")
    print(f"  output  : {args.out.relative_to(REPO)}")
    print(f"  jobs    : {len(jobs)}, errors: {n_err}")
    print(f"  tokens  : input {tok_in:,}  output {tok_out:,}  total {tok_in+tok_out:,}")
    print(f"  wall    : {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

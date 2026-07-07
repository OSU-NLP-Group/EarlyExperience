"""
Repair IWM rollout records that had JSON-parse failures on the proposer batch.

For each state with ANY invalid alt:
  1. Re-instantiate the case's sim env, replay expert fcall steps to the affected step
  2. Re-roll the failed alt slots via DeepSeek (same K, same name pool, retry up to N times
     with progressively higher temperature)
  3. Execute new alts on deepcopy, capture (result, post_state, exec_error)
  4. Summarize the new alts inline (DeepSeek flash)
  5. Patch the original record's `alts` list — invalid slots replaced with new valid ones
  6. Write patched records to a v2 summarized file

Read source : data/rollout/iwm_full_summarized.jsonl  (KEEPS valid alts/summaries as-is)
Output       : data/rollout/iwm_full_summarized_v2.jsonl   (drop-in replacement)

Run:
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/repair_iwm_alts.py
"""

from __future__ import annotations

import copy
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_iwm import (  # noqa: E402
    PROPOSER_SYSTEM, build_proposer_user_content, call_deepseek,
    eval_call, instantiate_case, load_tool_schemas, parse_proposer_response,
    render_history, snapshot_instances, _extract_fn_name,
)
from summarize_iwm_rollout import (  # noqa: E402
    SUMMARIZER_SYSTEM, build_summarizer_user, call_summarizer, _stringify,
)

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "envs/bfcl_v4/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
PARSED = REPO / "envs/bfcl_v4/data/parsed/opus_expert_steps.jsonl"
IN = REPO / "envs/bfcl_v4/data/rollout/iwm_full_summarized.jsonl"
OUT = REPO / "envs/bfcl_v4/data/rollout/iwm_full_summarized_v2.jsonl"

K_ALTS = 10
SEED_ALTS = 42
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# retry temperature ladder for proposer
PROPOSER_RETRY_TEMPS = [1.0, 0.7, 0.3]


def load_initial_configs(case_ids: set[str]) -> dict[str, dict]:
    out = {}
    with (DATA / "BFCL_v4_multi_turn_base.json").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            if o["id"] in case_ids:
                out[o["id"]] = o["initial_config"]
    return out


def load_parsed_steps_by_case(case_ids: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    with PARSED.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r["case_id"] in case_ids:
                out[r["case_id"]].append(r)
    for cid in out:
        out[cid].sort(key=lambda r: r["global_emit_idx"])
    return out


def repair_case(case_id: str, summarized_records: list[dict],
                 parsed_steps: list[dict], initial_config: dict,
                 client: OpenAI) -> tuple[list[dict], dict]:
    """Walk the case sequentially; for affected states, re-roll the invalid alt slots."""
    stats = Counter()
    out_records = []

    summ_by_geix = {r["global_emit_idx"]: r for r in summarized_records}
    involved = parsed_steps[0]["involved_classes"] if parsed_steps else summarized_records[0]["involved_classes"]
    schemas = load_tool_schemas(involved)
    all_names = [s["name"] for s in schemas]

    instances = instantiate_case(involved, initial_config)

    for i, step in enumerate(parsed_steps):
        prior_parsed = parsed_steps[:i]
        if step["step_type"] != "function_call":
            continue
        geix = step["global_emit_idx"]
        original_rec = summ_by_geix.get(geix)
        if original_rec is None:
            # state not in summarized file (shouldn't happen, but safe-skip)
            for c in step["expert_emit_decoded"]:
                eval_call(c, instances)
            continue

        # detect failure
        alts = original_rec.get("alts", [])
        invalid_idxs = [j for j, a in enumerate(alts) if not a.get("valid")]
        if not invalid_idxs:
            # nothing to fix; advance expert and move on
            out_records.append(original_rec)
            for c in step["expert_emit_decoded"]:
                eval_call(c, instances)
            continue

        # repair: prepare prompt and re-roll
        s_i_state = snapshot_instances(instances)
        expert_names = [_extract_fn_name(c) for c in step["expert_emit_decoded"]]
        pool = [n for n in all_names if n not in expert_names]
        # We re-roll the FULL K=10 alt list; we won't honor the original alt_names
        # set (since their batch failed wholesale, a different name draw is fine
        # and may even be healthier for diversity).
        rng_local = random.Random(SEED_ALTS + geix + 100)  # +100 to differ from original seed
        alt_names_new = rng_local.sample(pool, min(K_ALTS, len(pool)))

        # retry loop with temperature escalation
        parsed_new = None
        for attempt, temp in enumerate(PROPOSER_RETRY_TEMPS):
            user_content = build_proposer_user_content(
                render_history(prior_parsed, step), s_i_state,
                step["expert_emit_decoded"], alt_names_new, schemas,
            )
            try:
                resp = call_deepseek(client, PROPOSER_SYSTEM, user_content, temperature=temp)
            except Exception as e:  # noqa: BLE001
                stats[f"llm_error_attempt_{attempt}"] += 1
                continue
            parsed = parse_proposer_response(resp["content"], alt_names_new)
            n_valid = sum(1 for p in parsed if p["valid"])
            if n_valid >= K_ALTS - 1:  # accept if at most 1 invalid
                parsed_new = parsed
                stats[f"repair_ok_attempt_{attempt}"] += 1
                stats["repair_token_in"] += resp["usage"]["prompt_tokens"]
                stats["repair_token_out"] += resp["usage"]["completion_tokens"]
                break
            stats[f"repair_partial_attempt_{attempt}"] += 1

        if parsed_new is None:
            # all retries failed; keep original record (still has invalid alts)
            stats["repair_failed"] += 1
            out_records.append(original_rec)
            for c in step["expert_emit_decoded"]:
                eval_call(c, instances)
            continue

        # execute new alts on deepcopy + summarize
        new_alts_with_results = []
        user_task = step.get("user_msg_for_turn") or ""
        for p in parsed_new:
            if not p["valid"] or not p["call"]:
                new_alts_with_results.append({**p, "exec_result": None,
                                                "exec_error": p["error"],
                                                "post_state": None,
                                                "summary": None})
                continue
            clone = copy.deepcopy(instances)
            result, err = eval_call(p["call"], clone)
            raw_for_summary = result if err is None else f"(exec error: {err})"
            summ = call_summarizer(client, user_task, "", p["call"], _stringify(raw_for_summary))
            new_alts_with_results.append({**p,
                                            "exec_result": result,
                                            "exec_error": err,
                                            "post_state": snapshot_instances(clone),
                                            "summary": summ["summary"]})
            if summ["error"]:
                stats["summary_error"] += 1
            else:
                stats["summary_token_in"] += summ["usage"]["prompt_tokens"]
                stats["summary_token_out"] += summ["usage"]["completion_tokens"]

        # patch the record: replace all alts with new ones
        patched = dict(original_rec)
        patched["alts"] = new_alts_with_results
        patched["alt_names"] = alt_names_new
        patched["proposer_raw"] = resp["content"]
        patched["repaired"] = True
        out_records.append(patched)
        stats["states_repaired"] += 1

        # advance expert
        for c in step["expert_emit_decoded"]:
            eval_call(c, instances)

    return out_records, stats


def main() -> int:
    # load existing summarized records
    by_case: dict[str, list[dict]] = defaultdict(list)
    with IN.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            by_case[r["case_id"]].append(r)
    for cid in by_case:
        by_case[cid].sort(key=lambda r: r["global_emit_idx"])

    # find cases with any failure
    affected_cases = set()
    n_states_with_failures = 0
    for cid, recs in by_case.items():
        for r in recs:
            if any(not a.get("valid") for a in r.get("alts", [])):
                affected_cases.add(cid)
                n_states_with_failures += 1
                break
    n_states_failed_total = sum(
        1 for cid, recs in by_case.items() for r in recs
        if any(not a.get("valid") for a in r.get("alts", []))
    )

    # pre-call gate
    print(f"\n{'='*60}\nPRE-CALL GATE (TEAM_GUIDE §2)\n{'='*60}")
    print(f"  task: repair IWM alts (states with JSON-parse failures)")
    print(f"  affected states: {n_states_failed_total} across {len(affected_cases)} cases")
    print(f"  budget per state: up to 3 LLM retries (temperature escalation) + 10 summarizer calls")
    print(f"  worst-case calls: {n_states_failed_total * 3 + n_states_failed_total * 10} = "
          f"~{n_states_failed_total*13} LLM calls")
    print(f"  expected calls : ~{n_states_failed_total} proposer + ~{n_states_failed_total*10} summarizer "
          f"= ~{n_states_failed_total*11}")
    print(f"  est tokens     : ~150-200k total")
    print(f"  est cost       : <$0.10 flash")
    print(f"  model: {DEEPSEEK_MODEL}, thinking disabled\n{'='*60}\n")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set"); return 2
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    parsed_by_case = load_parsed_steps_by_case(affected_cases)
    configs = load_initial_configs(affected_cases)

    # repair each affected case concurrently (across cases)
    all_repaired: dict[str, list[dict]] = {}
    overall_stats = Counter()
    t0 = time.time()

    def work(cid):
        return cid, repair_case(cid, by_case[cid], parsed_by_case[cid],
                                  configs[cid], client)

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(work, cid): cid for cid in affected_cases}
        done = 0
        for fut in as_completed(futures):
            cid, (recs, stats) = fut.result()
            all_repaired[cid] = recs
            overall_stats.update(stats)
            done += 1
            print(f"  ... {done}/{len(affected_cases)} cases done  ({cid}: "
                  f"{stats.get('states_repaired',0)} states repaired, "
                  f"{stats.get('repair_failed',0)} still-failed)")

    # merge into final output: affected cases use repaired; others copy as-is
    final: list[dict] = []
    for cid in sorted(by_case.keys(), key=lambda s: int(s.rsplit("_",1)[-1])):
        if cid in all_repaired:
            final.extend(all_repaired[cid])
        else:
            final.extend(by_case[cid])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in final:
            f.write(json.dumps(r, default=str) + "\n")

    # report
    print(f"\n{'='*60}\nREPAIR COMPLETE\n{'='*60}")
    print(f"  output: {OUT.relative_to(REPO)}")
    print(f"  states repaired       : {overall_stats['states_repaired']}")
    print(f"  states still failed   : {overall_stats['repair_failed']}")
    print(f"  proposer tokens       : in {overall_stats['repair_token_in']:,}  out {overall_stats['repair_token_out']:,}")
    print(f"  summarizer tokens     : in {overall_stats['summary_token_in']:,}  out {overall_stats['summary_token_out']:,}")
    print(f"  wall: {time.time()-t0:.1f}s")

    # quick validity check on output
    total_alts = 0; total_valid = 0
    for r in final:
        for a in r.get("alts", []):
            total_alts += 1
            if a.get("valid"): total_valid += 1
    print(f"  final iwm_full_summarized_v2: {len(final)} states, "
          f"{total_valid}/{total_alts} alts valid ({total_valid/total_alts*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

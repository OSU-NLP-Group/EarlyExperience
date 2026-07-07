"""
Walk Opus FC's inference_log for each expert_id case and emit one JSONL record
per emit step.

This is a FORMAT-AGNOSTIC intermediate file — it just captures, per Opus emit
step:
  * the case_id / turn / step indices
  * the user message text at the start of this turn (if it's a turn-start)
  * the decoded function-call strings Opus emitted in this step (the "action")
  * the tool execution results that BFCL fed back for those calls
  * the recorded post-turn sim-state snapshot (when present in the log)

The eventual SFT user-content rendering decision (FC native vs flattened text)
operates DOWNSTREAM of this file — this file just captures Opus's behavior
losslessly so we can re-render however we want.

Also emits per-step category labels so downstream consumers can decide whether
to keep / skip:
  * type="function_call"   — assistant emit contained ≥1 decoded call (the
    normal case; trains action-prediction)
  * type="empty_emit"      — assistant emit decoded to []; signals turn end
  * type="text_only"       — assistant emit contained no function call and
    no empty list (chat reply to user)

Run:
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/parse_inference_log.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = (
    REPO_ROOT
    / "envs/bfcl_v4/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
)
RAW_DIR = REPO_ROOT / "envs/bfcl_v4/data/raw"
SPLIT_DIR = REPO_ROOT / "envs/bfcl_v4/data/split"
PARSED_DIR = REPO_ROOT / "envs/bfcl_v4/data/parsed"


def load_expert_ids() -> set[str]:
    return set(json.loads((SPLIT_DIR / "expert_ids.json").read_text()))


def load_opus_results(want: set[str]) -> dict[str, dict]:
    found = {}
    with (RAW_DIR / "opus_base_result.json").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["id"] in want:
                found[o["id"]] = o
    return found


def load_questions(want: set[str]) -> dict[str, dict]:
    found = {}
    with (DATA_DIR / "BFCL_v4_multi_turn_base.json").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["id"] in want:
                found[o["id"]] = o
    return found


def _extract_user_msg(begin_of_turn_query) -> str:
    """begin_of_turn_query is a list of {role, content}. Pull text content."""
    msgs = []
    if not isinstance(begin_of_turn_query, list):
        return ""
    for m in begin_of_turn_query:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            msgs.append(c)
        elif isinstance(c, list):
            # FC mode: content is sometimes [{"type":"text","text":"..."}]
            for part in c:
                if isinstance(part, dict) and "text" in part:
                    msgs.append(part["text"])
    return "\n".join(msgs)


def _extract_step_calls_and_tools(step_entries: list) -> tuple[list[str], list, str]:
    """Returns (decoded_calls, tool_responses, raw_assistant_emit).

    decoded_calls: list of Python-syntax call strings (from handler_log.model_response_decoded)
    tool_responses: list of recorded tool content strings (from role:tool entries)
    raw_assistant_emit: the assistant content as recorded (for later FC-format rendering)
    """
    decoded_calls = []
    tool_responses = []
    raw_assistant = None
    decode_error = False
    for e in step_entries:
        if not isinstance(e, dict):
            continue
        role = e.get("role")
        if role == "assistant":
            raw_assistant = e.get("content")
        elif role == "handler_log":
            if "model_response_decoded" in e:
                decoded_calls = list(e["model_response_decoded"])
            elif e.get("content", "").startswith("Error"):
                decode_error = True
        elif role == "tool":
            tool_responses.append(e.get("content"))
    return decoded_calls, tool_responses, raw_assistant


def classify_step(decoded_calls: list[str], raw_assistant) -> str:
    if len(decoded_calls) > 0:
        return "function_call"
    # decoded_calls is empty — either it's a clean empty emit (turn-end) or
    # the assistant said text without calling functions.
    # Opus FC content shape: list[{tool_name: args_str}] for tool calls,
    # OR list[str] for plain-text replies, OR list[{type:text,text:...}] in some variants.
    if isinstance(raw_assistant, list):
        for part in raw_assistant:
            if isinstance(part, str) and part.strip():
                return "text_only"
            if isinstance(part, dict):
                if isinstance(part.get("text"), str) and part["text"].strip():
                    return "text_only"
                if part.get("type") == "text" and isinstance(part.get("text"), str) and part["text"].strip():
                    return "text_only"
    elif isinstance(raw_assistant, str) and raw_assistant.strip():
        return "text_only"
    return "empty_emit"


def parse_case(case_id: str, opus: dict, question: dict) -> list[dict]:
    """Walk one case's inference_log, emit one record per emit step."""
    records = []
    inference_log = opus["inference_log"]

    # The first list item is the initial state_info entries.
    # Remaining items: for each user turn, a dict { begin_of_turn_query, step_0, ..., (state_info at end?) }
    # Plus possibly raw state_info lists at the top level between turns.
    turn_idx = -1
    global_emit_idx = 0
    for item in inference_log:
        if isinstance(item, list):
            # top-level state_info list (initial or between turns) — not an emit step
            continue
        if not isinstance(item, dict):
            continue
        turn_idx += 1
        user_msg = _extract_user_msg(item.get("begin_of_turn_query", []))
        step_keys = sorted(
            (k for k in item.keys() if k.startswith("step_")),
            key=lambda k: int(k.split("_")[1]),
        )
        for sk in step_keys:
            step_idx = int(sk.split("_")[1])
            step_entries = item[sk]
            calls, tools, raw_assistant = _extract_step_calls_and_tools(step_entries)
            step_type = classify_step(calls, raw_assistant)
            rec = {
                "case_id": case_id,
                "involved_classes": question["involved_classes"],
                "turn_idx": turn_idx,
                "step_idx": step_idx,
                "global_emit_idx": global_emit_idx,
                "is_first_step_of_turn": step_idx == 0,
                "user_msg_for_turn": user_msg if step_idx == 0 else None,
                "step_type": step_type,
                "expert_emit_decoded": calls,                  # list[str], Python-syntax
                "tool_responses_recorded": tools,              # list[str] (raw JSON strings)
                "expert_emit_raw": raw_assistant,              # FC structured content (untouched)
            }
            records.append(rec)
            global_emit_idx += 1
    return records


def main() -> None:
    expert_ids = load_expert_ids()
    print(f"loading Opus traces for {len(expert_ids)} expert_ids ...")
    opus = load_opus_results(expert_ids)
    questions = load_questions(expert_ids)
    missing = expert_ids - set(opus.keys())
    if missing:
        print(f"WARNING: {len(missing)} expert_ids not found in Opus result file")

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARSED_DIR / "opus_expert_steps.jsonl"

    case_count = 0
    total_records = 0
    type_counter = Counter()
    per_case_total = Counter()      # case → all emit steps
    per_case_funcalls = Counter()   # case → function_call type only
    user_turns_per_case = Counter()

    with out_path.open("w") as fout:
        for cid in sorted(expert_ids, key=lambda s: int(s.rsplit("_", 1)[-1])):
            if cid not in opus or cid not in questions:
                continue
            recs = parse_case(cid, opus[cid], questions[cid])
            case_count += 1
            for r in recs:
                fout.write(json.dumps(r) + "\n")
                total_records += 1
                type_counter[r["step_type"]] += 1
                per_case_total[cid] += 1
                if r["step_type"] == "function_call":
                    per_case_funcalls[cid] += 1
                if r["is_first_step_of_turn"]:
                    user_turns_per_case[cid] += 1

    print(f"\n=== output ===")
    print(f"file              : {out_path.relative_to(REPO_ROOT)}")
    print(f"cases processed   : {case_count}")
    print(f"total emit-step records: {total_records}")
    print(f"  by step_type    : {dict(type_counter)}")
    print()
    print(f"=== per-case stats ===")
    if case_count:
        fc_vals = list(per_case_funcalls.values())
        tot_vals = list(per_case_total.values())
        ut_vals = list(user_turns_per_case.values())
        print(f"  user turns / case   : mean={sum(ut_vals)/len(ut_vals):.2f}  min={min(ut_vals)}  max={max(ut_vals)}")
        print(f"  all emit-step / case: mean={sum(tot_vals)/len(tot_vals):.2f}  min={min(tot_vals)}  max={max(tot_vals)}")
        print(f"  func_call / case    : mean={sum(fc_vals)/len(fc_vals):.2f}  min={min(fc_vals)}  max={max(fc_vals)}")

    # also count: total function calls (not steps) — for direct comparison vs GT call count
    total_individual_calls = sum(
        sum(len(r["expert_emit_decoded"]) for r in parse_case(c, opus[c], questions[c]))
        for c in opus if c in questions
    )
    # re-do via the written file would be more efficient but we already have records in memory at parse time
    # (recomputed lazily here — small enough)
    print(f"\n=== paper-comparable counts ===")
    print(f"  trajectories (cases) : {case_count}                   [paper: 125]")
    print(f"  individual GT-style calls (sum across function_call emits): {total_individual_calls}")
    print(f"    [paper IWM examples derived from D_expert: 1,264]")
    print(f"  emit-step count (our per-emit-step granularity, function_call only): "
          f"{type_counter['function_call']}")


if __name__ == "__main__":
    main()

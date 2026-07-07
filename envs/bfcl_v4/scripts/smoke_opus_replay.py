"""
Phase 0 — Opus 4.5 FC trajectory replay smoke.

For 1-3 valid==true Opus FC cases on BFCL_v4_multi_turn_base, walk the case's
inference_log, replay each emit step's decoded calls against our pinned env, and
verify per-step tool outputs match the recorded ones. Per-turn final state is
compared to recorded `state_info` entries as a stronger check.

If outputs match → our pin (`aed2de1`) is compatible with the 2025-12-16
leaderboard run that produced these trajectories. If they don't match → either
the pin is wrong or there's a sim-API non-determinism that needs handling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = (
    REPO_ROOT
    / "envs/bfcl_v4/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
)
OPUS_RESULT = REPO_ROOT / "envs/bfcl_v4/data/raw/opus_base_result.json"
OPUS_SCORE = REPO_ROOT / "envs/bfcl_v4/data/raw/opus_base_score.json"


def load_failing_ids() -> set[str]:
    """Parse score file. Line 0 = summary. Lines 1+ = per-failed-case records."""
    failing = set()
    with OPUS_SCORE.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:
                continue
            o = json.loads(line)
            if not o.get("valid", True):
                failing.add(o["id"])
    return failing


def load_passing_opus_cases(case_ids: Iterable[str]) -> dict:
    """Load Opus entries by id."""
    want = set(case_ids)
    found = {}
    with OPUS_RESULT.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["id"] in want:
                found[o["id"]] = o
                if len(found) == len(want):
                    break
    return found


def load_questions(case_ids: Iterable[str]) -> dict:
    """Load question entries (initial_config, involved_classes) by id."""
    want = set(case_ids)
    found = {}
    with (DATA_DIR / "BFCL_v4_multi_turn_base.json").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["id"] in want:
                found[o["id"]] = o
                if len(found) == len(want):
                    break
    return found


def extract_step_calls(step_entries: list) -> list[str]:
    """From a step's role-entries, find the handler_log and return decoded calls."""
    for e in step_entries:
        if isinstance(e, dict) and e.get("role") == "handler_log":
            decoded = e.get("model_response_decoded")
            if decoded:
                return list(decoded)
    return []


def extract_step_tool_outputs(step_entries: list) -> list[str]:
    """From a step's role-entries, collect all `role:tool` content strings."""
    outs = []
    for e in step_entries:
        if isinstance(e, dict) and e.get("role") == "tool":
            outs.append(e.get("content"))
    return outs


def extract_turn_state_info(turn_dict: dict) -> dict[str, dict]:
    """Some logs include `state_info` entries at the tail of the turn dict.
    Returns {class_name: state_dict} for the recorded post-turn state, if present."""
    # state_info entries may appear as values for a 'state_info' key, OR inline in steps
    # Realistically they're emitted between turns at top-level inference_log,
    # not inside the turn dict. So this helper returns {} if none found.
    out = {}
    for v in turn_dict.values():
        if isinstance(v, list):
            for e in v:
                if isinstance(e, dict) and e.get("role") == "state_info":
                    out[e["class_name"]] = e["content"]
    return out


def snapshot_instances(instances: dict) -> dict[str, dict]:
    """Public-attr snapshot per class."""
    out = {}
    for cls_name, inst in instances.items():
        out[cls_name] = {
            k: v for k, v in vars(inst).items() if not k.startswith("_")
        }
    return out


def _normalize_state_str(s) -> str:
    """For comparison, render the value as JSON string (default=str for non-JSONable)."""
    return json.dumps(s, default=str, sort_keys=True)


def replay_case(case_id: str, opus: dict, question: dict) -> dict:
    """Replay one Opus case. Returns a stats dict."""
    initial_config = question["initial_config"]
    involved_classes = question["involved_classes"]
    inference_log = opus["inference_log"]

    # inference_log[0] = initial state_info entries
    # inference_log[1..] = per-user-turn dicts with begin_of_turn_query + step_K + maybe state_info
    initial_state_recorded = {}
    if inference_log and isinstance(inference_log[0], list):
        for e in inference_log[0]:
            if isinstance(e, dict) and e.get("role") == "state_info":
                initial_state_recorded[e["class_name"]] = e["content"]

    print(f"\n=== case {case_id} ===")
    print(f"  involved_classes: {involved_classes}")
    print(f"  # turn dicts: {sum(1 for x in inference_log[1:] if isinstance(x, dict))}")

    stats = {
        "case_id": case_id,
        "tool_output_step_count": 0,
        "tool_output_match_count": 0,
        "tool_output_mismatch_examples": [],
        "post_turn_state_recorded": 0,
        "post_turn_state_match": 0,
        "post_turn_state_mismatch_classes": [],
    }

    # walk per-turn dicts
    turn_idx = -1
    for item in inference_log[1:]:
        if not isinstance(item, dict):
            # tail-level state_info list — skip; turn-level state comparison happens inside
            continue
        turn_idx += 1

        # collect step keys in order
        step_keys = sorted(
            (k for k in item.keys() if k.startswith("step_")),
            key=lambda k: int(k.split("_")[1]),
        )

        for sk in step_keys:
            step_entries = item[sk]
            calls = extract_step_calls(step_entries)
            recorded_tools = extract_step_tool_outputs(step_entries)
            if not calls:
                # empty emit (turn end signal) — no tools to compare
                continue
            # execute the calls (BFCL maintains instance state via globals cache,
            # so subsequent execute_multi_turn_func_call calls in the same case
            # naturally continue from prior state)
            try:
                results, instances = execute_multi_turn_func_call(
                    func_call_list=calls,
                    initial_config=initial_config,
                    involved_classes=involved_classes,
                    model_name=f"smoke_opus_replay_{case_id}",
                    test_entry_id=case_id,
                    long_context=False,
                    is_evaL_run=False,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  turn {turn_idx} {sk}: EXEC FAILED  ({type(e).__name__}: {e})")
                continue

            for cidx, (rec, our) in enumerate(zip(recorded_tools, results)):
                stats["tool_output_step_count"] += 1
                # both are JSON strings; compare by normalized JSON
                try:
                    rec_norm = _normalize_state_str(json.loads(rec)) if isinstance(rec, str) else _normalize_state_str(rec)
                    our_norm = _normalize_state_str(json.loads(our)) if isinstance(our, str) else _normalize_state_str(our)
                    match = rec_norm == our_norm
                except (json.JSONDecodeError, TypeError):
                    match = rec == our
                if match:
                    stats["tool_output_match_count"] += 1
                elif len(stats["tool_output_mismatch_examples"]) < 3:
                    stats["tool_output_mismatch_examples"].append(
                        {
                            "turn": turn_idx, "step": sk, "call_idx": cidx,
                            "call": calls[cidx] if cidx < len(calls) else None,
                            "recorded": (rec[:200] + "…") if isinstance(rec, str) and len(rec) > 200 else rec,
                            "ours": (our[:200] + "…") if isinstance(our, str) and len(our) > 200 else our,
                        }
                    )

        # if this turn dict carries state_info at the end, compare snapshot
        recorded_post_state = extract_turn_state_info(item)
        if recorded_post_state and instances is not None:
            our_snap = snapshot_instances(instances)
            for cls_name, recorded_state in recorded_post_state.items():
                stats["post_turn_state_recorded"] += 1
                ours = our_snap.get(cls_name, {})
                if _normalize_state_str(ours) == _normalize_state_str(recorded_state):
                    stats["post_turn_state_match"] += 1
                else:
                    stats["post_turn_state_mismatch_classes"].append(
                        {"turn": turn_idx, "class": cls_name}
                    )

    # report
    print(
        f"  tool outputs: {stats['tool_output_match_count']}/{stats['tool_output_step_count']} match"
    )
    if stats["tool_output_mismatch_examples"]:
        for m in stats["tool_output_mismatch_examples"]:
            print(f"    MISMATCH @ turn {m['turn']} {m['step']} call_idx {m['call_idx']}")
            print(f"      call    : {m['call']}")
            print(f"      recorded: {m['recorded']}")
            print(f"      ours    : {m['ours']}")
    if stats["post_turn_state_recorded"] > 0:
        print(
            f"  post-turn state: {stats['post_turn_state_match']}/{stats['post_turn_state_recorded']} match"
        )
        if stats["post_turn_state_mismatch_classes"]:
            print(f"    mismatched: {stats['post_turn_state_mismatch_classes'][:5]}")

    return stats


def main() -> int:
    failing = load_failing_ids()
    print(f"loaded score file: {len(failing)} failing case_ids (expected 38)")

    # pick 3 cases: one simple single-class, two multi-class
    candidates = [
        "multi_turn_base_0",   # TwitterAPI + GorillaFileSystem, 4 turns, 10 GT calls
        "multi_turn_base_3",   # random multi-class candidate
        "multi_turn_base_50",  # mid-range candidate
    ]
    # filter to passing only
    passing = [c for c in candidates if c not in failing]
    print(f"chosen cases (passing only): {passing}")

    opus_cases = load_passing_opus_cases(passing)
    questions = load_questions(passing)

    overall_ok = True
    for cid in passing:
        if cid not in opus_cases:
            print(f"  case {cid}: not in opus result file (skip)")
            continue
        if cid not in questions:
            print(f"  case {cid}: not in question dataset (skip)")
            continue
        stats = replay_case(cid, opus_cases[cid], questions[cid])
        if stats["tool_output_step_count"] > 0:
            ratio = stats["tool_output_match_count"] / stats["tool_output_step_count"]
            if ratio < 0.95:
                overall_ok = False
        if stats["post_turn_state_recorded"] > 0:
            if stats["post_turn_state_match"] < stats["post_turn_state_recorded"]:
                overall_ok = False

    print(f"\n{'='*60}\nOverall: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

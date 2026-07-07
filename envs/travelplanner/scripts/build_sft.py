"""Build the three final SFT files for TravelPlanner (Stage D).

Produces (under data/sft/):
  expert_sft.jsonl     — 1,370 records, (state, expert_action)
  iwm_sft.jsonl        — 52,754 records, (state + action, next_state)
  reflection_sft.jsonl — 1,148 records, (state, reflection + expert_action)

State representation = paper author's gym `_format_state_sft` (byte-identical
to what the trained model sees at inference). IWM next-state narrative =
paper §B.7-style structural budget delta + a templated field-effect remark.

Run with the travelplanner-paper conda env:
  PYTHONNOUSERSITE=1 TRAVELPLANNER_DB=<global_db> \\
    /home/ulss/miniconda3/envs/travelplanner-paper/bin/python build_sft.py
"""
from __future__ import annotations
import ast
import copy
import functools
import json
import os
import re
import time
from pathlib import Path

# Singleton-cache APIs (same pattern as rollout_iwm_paper.py)
from travelplanner.tools.flights.apis import Flights
from travelplanner.tools.accommodations.apis import Accommodations
from travelplanner.tools.restaurants.apis import Restaurants
from travelplanner.tools.attractions.apis import Attractions
from travelplanner.tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
_API_CACHE: dict = {}
def _patch_singleton(cls):
    orig = cls.__init__
    @functools.wraps(orig)
    def cached_init(self, *args, **kwargs):
        key = (cls.__name__,) + tuple(args) + tuple(sorted(kwargs.items()))
        if key in _API_CACHE:
            for attr, val in _API_CACHE[key].__dict__.items():
                setattr(self, attr, val)
        else:
            orig(self, *args, **kwargs)
            _API_CACHE[key] = self
    cls.__init__ = cached_init
for _cls in (Flights, Accommodations, Restaurants, Attractions, GoogleDistanceMatrix):
    _patch_singleton(_cls)

from travelplanner.envs.travel_planner_sole_planning_env import (
    TravelPlannerEnv, FIELD_ORDER, COMPLETE_PLAN)
from datasets import load_dataset
from rollout_iwm_paper import (fast_step, find_action_idx, build_expert_action,
                               TRUNCATE_AFTER_DAY)

ENV_ROOT = Path(__file__).resolve().parents[1]
IWM_PATH = ENV_ROOT / "data" / "rollout" / "iwm_rollout_paper.jsonl"
SR_PATH = ENV_ROOT / "data" / "rollout" / "sr_rollout.jsonl"
SR_RERUN_PATH = ENV_ROOT / "data" / "rollout" / "sr_rerun.jsonl"
OUT_DIR = ENV_ROOT / "data" / "sft"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYS_AGENT = (
    "You are a travel planning agent. Given the current planning state, "
    "decide the next action that best advances the plan within the budget "
    "and constraints. Output only the action as a JSON object."
)
SYS_WORLD = (
    "You are a travel planning environment model. Given the current planning "
    "state and an action, predict the resulting state after the action is "
    "applied. Focus on budget changes and which field gets filled."
)


# ---------------------------- action / next-state renderers (paper-style)
def _extract_flight_no(val: str) -> str:
    m = re.search(r"Flight Number:\s*(\S+?)(?:,|$)", val)
    return m.group(1) if m else ""


def _extract_name(val: str) -> str:
    if "," in val:
        return val.rsplit(",", 1)[0].strip()
    return val


def render_action_compact(action: dict) -> str:
    at = action["action_type"]
    val = action.get("value", "")
    cost = action.get("cost", 0)
    if at.startswith("SKIP_") or at == "COMPLETE_PLAN":
        return at
    if at == "SET_TRANSPORTATION":
        fn = _extract_flight_no(val)
        if fn:
            return f"SET_TRANSPORTATION ({fn}, ${int(cost)})"
        mode = "Self-driving" if "self-driving" in val.lower() else "Taxi"
        return f"SET_TRANSPORTATION ({mode}, ${int(cost)})"
    if at == "SET_MEAL":
        return f"SET_MEAL ({_extract_name(val)}, ${int(cost)})"
    if at == "SET_ACCOMMODATION":
        return f"SET_ACCOMMODATION ({_extract_name(val)}, ${int(cost)})"
    if at == "ADD_ATTRACTION":
        return f"ADD_ATTRACTION ({_extract_name(val)})"
    return f"{at} ({val[:30]}, ${int(cost)})"


def render_next_state(action: dict, day: int, field: str, budget: float,
                      spent_after: float, total_days: int) -> str:
    at = action["action_type"]
    pct = round(100.0 * spent_after / budget) if budget else 0
    remaining = budget - spent_after
    budget_line = (f"After this action, you've spent ${int(spent_after)}, "
                   f"leaving ${int(remaining)} from your ${int(budget)} budget ({pct}% used).")
    if at == "COMPLETE_PLAN":
        remark = "The plan is now complete and ready for submission."
    elif at == "SET_TRANSPORTATION":
        if day == 1:
            remark = "You're traveling to your destination."
        elif day == total_days:
            remark = "Return travel is locked in."
        else:
            remark = "An inter-city transition is set."
    elif at == "SKIP_TRANSPORTATION":
        remark = f"No transportation is needed for day {day}."
    elif at == "SET_MEAL":
        remark = f"Day {day} {field} is planned."
    elif at == "SKIP_MEAL":
        remark = f"Day {day} {field} is skipped."
    elif at == "ADD_ATTRACTION":
        remark = f"An attraction is added to day {day}."
    elif at == "SKIP_ATTRACTION":
        remark = f"Day {day} attraction is skipped."
    elif at == "SET_ACCOMMODATION":
        remark = f"Accommodation for night {day} is booked."
    elif at == "SKIP_ACCOMMODATION":
        if day == total_days:
            remark = "No overnight stay needed on the final day."
        else:
            remark = f"Day {day} accommodation is skipped."
    else:
        remark = ""
    return f"{budget_line} {remark}".strip()


# ---------------------------- SR lookup (rerun preferred + context-aware filter)
_ADJECTIVAL_CORRECT = re.compile(
    r'\b(in|to|from|of|the|on|stay in|head to|visit|gives|reach|return to|back to|already in|stays in|leaves us in)\s+(the\s+)?(correct|right)\s+(city|location|route|leg|direction|order|sequence|side|place|destination|country|region|state|area)\b',
    re.I)
_REAL_LABEL_LEAK = re.compile(
    # Meta-style references to a pre-existing decision: "X was picked", "the
    # decision says ...", "I picked X", "this option was selected", etc.
    r'\b(was|is|were|seems\s+to\s+be|appears\s+to\s+be|seems|appears|already|previously|pre-?selected)\s+'
    r'(picked|selected|chosen|the\s+correct|the\s+optimal|the\s+right\s+choice|the\s+right\s+action)\b'
    r'|\b(the\s+decision\s+says|the\s+system\s+accepted|i\s+picked|i\s+selected|i\s+chose|pre[- ]?determined)\b'
    r'|\bthis\s+(action|option|choice)\s+is\s+(picked|selected|chosen|correct|optimal|the\s+best)\b',
    re.I)


def is_real_label_leak(reflection: str) -> bool:
    """True if reflection contains the LLM commenting on the input slot as a
    past decision (real label-leak). Adjectival 'correct city' / 'right
    direction' usages are NOT considered leaks."""
    if not reflection:
        return False
    return bool(_REAL_LABEL_LEAK.search(reflection))


def build_sr_lookup() -> dict:
    """Map (traj_idx, step_idx) → reflection_text (None for unusable).
    Strategy: prefer attempt-2 rerun when cleaner; drop only records that have
    a real label leak (LLM meta-commenting on input slot), keep records whose
    only flag is adjectival 'correct city' / 'right direction' etc."""
    sr_rows = [json.loads(l) for l in SR_PATH.read_text().splitlines() if l.strip()]
    rerun_rows = []
    if SR_RERUN_PATH.exists():
        rerun_rows = [json.loads(l) for l in SR_RERUN_PATH.read_text().splitlines() if l.strip()]
    rerun_by_key = {(r["traj_idx"], r["step_idx"]): r for r in rerun_rows}

    def severity(rec):
        # Real label leaks dominate; raw flag count is the tiebreaker
        if not rec or not rec.get("reflection"):
            return 99
        s = 0
        if is_real_label_leak(rec["reflection"]):
            s += 10
        f = rec.get("flags") or {}
        if f.get("has_numbered_labels"):
            s += 5
        if f.get("is_doubled"):
            s += 5
        # context-blind banned_vocab as a small tiebreaker
        s += len(f.get("banned_vocab") or []) * 0.1
        return s

    out = {}
    dropped_real_leak = 0
    kept_with_adj_only = 0
    for r in sr_rows:
        key = (r["traj_idx"], r["step_idx"])
        rerun = rerun_by_key.get(key)
        chosen = r
        if rerun and rerun.get("reflection") and severity(rerun) < severity(r):
            chosen = rerun
        text = chosen.get("reflection")
        if not text:
            continue
        # Drop only on real label leak / numbered / doubled
        f = chosen.get("flags") or {}
        if is_real_label_leak(text) or f.get("has_numbered_labels") or f.get("is_doubled"):
            dropped_real_leak += 1
            continue
        # If raw banned_vocab list non-empty but no real leak → keep (adjectival)
        if f.get("banned_vocab"):
            kept_with_adj_only += 1
        out[key] = text
    print(f"  [sr-filter] kept {len(out)}; dropped real-leaks: {dropped_real_leak}; "
          f"kept-despite-adjectival-flag: {kept_with_adj_only}")
    return out


# ---------------------------- main
def main():
    t0 = time.time()
    ds = load_dataset("osunlp/TravelPlanner", "train", split="train")
    iwm_rows = [json.loads(l) for l in IWM_PATH.read_text().splitlines() if l.strip()]
    print(f"[load] {len(iwm_rows)} IWM state records  ({time.time()-t0:.1f}s)")

    sr_lookup = build_sr_lookup()
    print(f"[load] {len(sr_lookup)} usable SR reflections (1 dropped: still-flagged after rerun)")

    # Group IWM rows by traj, sorted by step_idx
    from collections import defaultdict
    by_traj = defaultdict(list)
    for r in iwm_rows:
        by_traj[r["traj_idx"]].append(r)
    for k in by_traj:
        by_traj[k].sort(key=lambda r: r["step_idx"])

    expert_records = []
    iwm_records = []
    refl_records = []
    refl_drops_no_alt = 0  # states with SR scope but no rec in sr_lookup

    for traj_idx in sorted(by_traj):
        rec = ds[int(traj_idx)]
        env = TravelPlannerEnv(rec)
        env.reset()
        total_days = int(rec["days"])

        for state in by_traj[traj_idx]:
            state_text = env._format_state_sft()
            expert = next(t["action"] for t in state["transitions"] if t["is_expert"])

            # === expert_sft ===
            expert_records.append({
                "messages": [
                    {"role": "system", "content": SYS_AGENT},
                    {"role": "user", "content": state_text},
                    {"role": "assistant", "content": json.dumps(expert, ensure_ascii=False)},
                ]
            })

            # === iwm_sft (one record per transition) ===
            for t in state["transitions"]:
                act = t["action"]
                iwm_records.append({
                    "messages": [
                        {"role": "system", "content": SYS_WORLD},
                        {"role": "user", "content":
                            state_text + f"\nAction: {render_action_compact(act)}"},
                        {"role": "assistant", "content": render_next_state(
                            act, state["day"], state["field"], state["budget"],
                            t["spent_after"], total_days)},
                    ]
                })

            # === reflection_sft (only states in SR scope, with usable refl) ===
            # As of v6 attraction states are included (they're the key signal
            # for the wrong-city / Within-Current-City check at inference).
            key = (state["traj_idx"], state["step_idx"])
            if key in sr_lookup:
                refl_records.append({
                    "messages": [
                        {"role": "system", "content": SYS_AGENT},
                        {"role": "user", "content": state_text},
                        {"role": "assistant", "content":
                            sr_lookup[key] + "\n\n" + json.dumps(expert, ensure_ascii=False)},
                    ]
                })

            # Advance env with the GYM's expert action (correct cost)
            exp_idx = find_action_idx(env, expert["action_type"], expert["day"],
                                      expert["field"], expert["value"])
            if exp_idx is None:
                # Should not happen — we just replayed this path during IWM rollout
                raise RuntimeError(f"expert not found at traj{traj_idx} step{state['step_idx']}")
            fast_step(env, json.loads(env.valid_actions[exp_idx]))

    # Write files
    out_expert = OUT_DIR / "expert_sft.jsonl"
    out_iwm = OUT_DIR / "iwm_sft.jsonl"
    out_refl = OUT_DIR / "reflection_sft.jsonl"
    out_expert.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in expert_records) + "\n")
    out_iwm.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in iwm_records) + "\n")
    out_refl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in refl_records) + "\n")

    print()
    print(f"[done] expert_sft.jsonl     : {len(expert_records):>6,} records "
          f"({out_expert.stat().st_size / 1e6:.1f} MB)")
    print(f"[done] iwm_sft.jsonl        : {len(iwm_records):>6,} records "
          f"({out_iwm.stat().st_size / 1e6:.1f} MB)")
    print(f"[done] reflection_sft.jsonl : {len(refl_records):>6,} records "
          f"({out_refl.stat().st_size / 1e6:.1f} MB)")
    print(f"[wall] {time.time()-t0:.1f}s")
    print(f"[out]  {OUT_DIR}")


if __name__ == "__main__":
    main()

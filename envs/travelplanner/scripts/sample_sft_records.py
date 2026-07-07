"""Generate 3 (+1) sample SFT records for hand-inspection before batch build.

Uses paper gym's _format_state_sft (so the state rendering at training is
byte-identical to what the trained model sees at inference).
"""
from __future__ import annotations
import ast
import copy
import json
import os
from pathlib import Path

# Singleton-cache APIs as in rollout_iwm_paper.py
import functools
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

from travelplanner.envs.travel_planner_sole_planning_env import TravelPlannerEnv, FIELD_ORDER, COMPLETE_PLAN, SKIP_TRANSPORTATION, ADD_ATTRACTION
from datasets import load_dataset

ENV_ROOT = Path(__file__).resolve().parents[1]
IWM_PATH = ENV_ROOT / "data" / "rollout" / "iwm_rollout_paper.jsonl"
SR_PATH = ENV_ROOT / "data" / "rollout" / "sr_rollout.jsonl"
SR_RERUN_PATH = ENV_ROOT / "data" / "rollout" / "sr_rerun.jsonl"

# Two distinct system prompts
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


def fast_step(env, action):
    """Same as rollout_iwm_paper.fast_step."""
    if action["action_type"] == COMPLETE_PLAN:
        env.state.is_done = True
        return
    env._apply_action(action)
    env.state.spent += action["cost"]
    env._advance_state(action)


def render_state_via_gym(rec_iwm, ds_record):
    """Use paper gym's _format_state_sft to get the canonical state string at
    the moment of this IWM record's state_before. We do this by replaying the
    expert path up to step_idx-1, then capturing the obs."""
    env = TravelPlannerEnv(ds_record)
    env.reset()
    # Replay expert path up to this state
    iwm_rows = [json.loads(l) for l in IWM_PATH.read_text().splitlines() if l.strip()]
    traj_states = [r for r in iwm_rows if r["traj_idx"] == rec_iwm["traj_idx"]]
    traj_states.sort(key=lambda r: r["step_idx"])
    for r in traj_states:
        if r["step_idx"] == rec_iwm["step_idx"]:
            break
        # Step with this state's expert action
        expert = next(t["action"] for t in r["transitions"] if t["is_expert"])
        # Find idx in valid_actions
        for i, aj in enumerate(env.valid_actions):
            a = json.loads(aj)
            if (a["action_type"] == expert["action_type"]
                    and a["day"] == expert["day"]
                    and a["field"] == expert["field"]
                    and a["value"] == expert["value"]):
                fast_step(env, a)
                break
        else:
            # softer match: by content
            from rollout_iwm_paper import _value_key
            target_key = _value_key(expert["action_type"], expert["value"])
            for i, aj in enumerate(env.valid_actions):
                a = json.loads(aj)
                if (a["action_type"] == expert["action_type"]
                        and a["day"] == expert["day"]
                        and a["field"] == expert["field"]
                        and _value_key(a["action_type"], a["value"]) == target_key):
                    fast_step(env, a)
                    break
    return env._format_state_sft(), env


def _extract_flight_no(val: str) -> str:
    import re
    m = re.search(r"Flight Number:\s*(\S+?)(?:,|$)", val)
    return m.group(1) if m else ""


def _extract_name(val: str) -> str:
    # "Name, City(State)" → "Name"
    if "," in val:
        return val.rsplit(",", 1)[0].strip()
    return val


def render_action_compact(action):
    """Paper §B.7-style compact action representation: SET_X (key, $cost) or SKIP_X."""
    at = action["action_type"]
    val = action.get("value", "")
    cost = action.get("cost", 0)
    if at.startswith("SKIP_") or at == "COMPLETE_PLAN":
        return at
    if at == "SET_TRANSPORTATION":
        fn = _extract_flight_no(val)
        if fn:
            return f"SET_TRANSPORTATION ({fn}, ${int(cost)})"
        # self-drive / taxi
        mode = "Self-driving" if "self-driving" in val.lower() else "Taxi"
        return f"SET_TRANSPORTATION ({mode}, ${int(cost)})"
    if at == "SET_MEAL":
        return f"SET_MEAL ({_extract_name(val)}, ${int(cost)})"
    if at == "SET_ACCOMMODATION":
        return f"SET_ACCOMMODATION ({_extract_name(val)}, ${int(cost)})"
    if at == "ADD_ATTRACTION":
        return f"ADD_ATTRACTION ({_extract_name(val)})"
    return f"{at} ({val[:30]}, ${int(cost)})"


def render_next_state(action, day, field, budget, spent_after, total_days):
    """IWM next-state: paper §B.7 structural part + a brief templated remark
    about which field got affected (matches paper example's 'Good start!
    You're traveling to your destination.' style without LLM cost)."""
    at = action["action_type"]
    pct = round(100.0 * spent_after / budget) if budget else 0
    remaining = budget - spent_after
    budget_line = (f"After this action, you've spent ${int(spent_after)}, "
                   f"leaving ${int(remaining)} from your ${int(budget)} budget ({pct}% used).")

    # Templated remark
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


# ------------------------------------------------------------------ build samples
def main():
    ds = load_dataset("osunlp/TravelPlanner", "train", split="train")
    iwm_rows = [json.loads(l) for l in IWM_PATH.read_text().splitlines() if l.strip()]
    sr_rows = [json.loads(l) for l in SR_PATH.read_text().splitlines() if l.strip()]
    sr_rerun = {(r["traj_idx"], r["step_idx"]): r for r in
                (json.loads(l) for l in SR_RERUN_PATH.read_text().splitlines() if l.strip())}
    sr_by_key = {(r["traj_idx"], r["step_idx"]): r for r in sr_rows}

    # Use traj 0 d1 transportation — paper §B.7 example
    target = next(r for r in iwm_rows if r["traj_idx"] == 0 and r["day"] == 1 and r["field"] == "transportation")
    state_text, env = render_state_via_gym(target, ds[0])

    print("=" * 78)
    print("SAMPLE STATE (rendered by paper gym _format_state_sft)")
    print("=" * 78)
    print(state_text)
    print()

    # --- 1) expert_sft sample ---
    expert = next(t["action"] for t in target["transitions"] if t["is_expert"])
    expert_record = {
        "messages": [
            {"role": "system", "content": SYS_AGENT},
            {"role": "user", "content": state_text},
            {"role": "assistant", "content": json.dumps(expert, ensure_ascii=False)},
        ]
    }
    print("=" * 78)
    print("1) expert_sft.jsonl SAMPLE")
    print("=" * 78)
    print(json.dumps(expert_record, indent=2, ensure_ascii=False))
    print()

    # --- 2) reflection_sft sample ---
    sr_key = (target["traj_idx"], target["step_idx"])
    sr_rec = sr_rerun.get(sr_key) or sr_by_key.get(sr_key)
    if sr_rec and sr_rec.get("reflection"):
        reflection = sr_rec["reflection"]
        refl_record = {
            "messages": [
                {"role": "system", "content": SYS_AGENT},
                {"role": "user", "content": state_text},
                {"role": "assistant", "content": reflection + "\n\n" + json.dumps(expert, ensure_ascii=False)},
            ]
        }
        print("=" * 78)
        print("2) reflection_sft.jsonl SAMPLE")
        print("=" * 78)
        print(json.dumps(refl_record, indent=2, ensure_ascii=False))
        print()
    else:
        print("[no SR record found for this state]")

    # --- 3) iwm_sft sample (expert SET) ---
    expert_spent_after = next(t["spent_after"] for t in target["transitions"] if t["is_expert"])
    iwm_record_expert = {
        "messages": [
            {"role": "system", "content": SYS_WORLD},
            {"role": "user", "content": (
                state_text + f"\nAction: {render_action_compact(expert)}"
            )},
            {"role": "assistant", "content": render_next_state(
                expert, target["day"], target["field"], target["budget"],
                expert_spent_after, int(ds[0]["days"]))},
        ]
    }
    print("=" * 78)
    print("3) iwm_sft.jsonl SAMPLE — EXPERT transition (SET_TRANSPORTATION F3573659 $474)")
    print("=" * 78)
    print(json.dumps(iwm_record_expert, indent=2, ensure_ascii=False))
    print()

    # --- 4) iwm_sft sample (an alt: SKIP_TRANSPORTATION) ---
    alt = next((t for t in target["transitions"] if not t["is_expert"]), None)
    if alt:
        iwm_record_alt = {
            "messages": [
                {"role": "system", "content": SYS_WORLD},
                {"role": "user", "content": (
                    state_text + f"\nAction: {render_action_compact(alt['action'])}"
                )},
                {"role": "assistant", "content": render_next_state(
                    alt["action"], target["day"], target["field"], target["budget"],
                    alt["spent_after"], int(ds[0]["days"]))},
            ]
        }
        print("=" * 78)
        print("4) iwm_sft.jsonl SAMPLE — ALT transition (SKIP_TRANSPORTATION)")
        print("=" * 78)
        print(json.dumps(iwm_record_alt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

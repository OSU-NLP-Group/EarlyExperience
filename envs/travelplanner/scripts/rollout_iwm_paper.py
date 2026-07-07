"""IWM rollout using the paper author's gym (Stage B').

Drives travelplanner.envs.travel_planner_sole_planning_env.TravelPlannerEnv as
the ground-truth action enumerator (transport bounded by ref_info-derived
city_list; attraction = ADD with max_attractions_per_day=1; COMPLETE_PLAN
terminal; over-budget alts excluded). Bypasses the slow
_evaluate_partial_plan via fast_step (only the 3 state-mutating calls in
env.step), which is provably bit-equivalent on env.state (smoke asserts).

Replaces our refinfo (A, 29,767) and anywhere (C, 74,942) variants.

Run with the travelplanner-paper conda env:
  PYTHONNOUSERSITE=1 TRAVELPLANNER_DB=<global_db> \\
    /home/ulss/miniconda3/envs/travelplanner-paper/bin/python rollout_iwm_paper.py
Options:
  --smoke         only the first 2 trajectories + equivalence assertions
  --limit N       only first N trajectories
"""
from __future__ import annotations
import argparse
import ast
import copy
import functools
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

# Singleton-cache the pandas-backed tool APIs so each per-env TravelPlannerEnv
# init doesn't re-read the 305 MB flights CSV.
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
    TravelPlannerEnv, FIELD_ORDER, MEAL_FIELDS,
    SET_TRANSPORTATION, SKIP_TRANSPORTATION, SET_MEAL, SKIP_MEAL,
    ADD_ATTRACTION, SKIP_ATTRACTION, SET_ACCOMMODATION, SKIP_ACCOMMODATION,
    COMPLETE_PLAN,
)
from datasets import load_dataset

ENV_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ENV_ROOT / "data" / "rollout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# traj 41 (Seattle -> Texas, 7d, 4p, hard): gold annotation has cross-city
# attractions on d4 (San Angelo) and d5 (Houston) while current_city stays at
# San Antonio. Truncate this trajectory after day 3.
TRUNCATE_AFTER_DAY: dict = {41: 3}


# ---------- robust expert-action matcher
_PAREN = re.compile(r"\([\w\s]+\)$")
def _norm_city(s): return _PAREN.sub("", s.strip()).strip()


def _value_key(action_type, value):
    if action_type == SET_TRANSPORTATION:
        m = re.search(r"Flight Number:\s*(\S+?)(?:,|$)", value)
        if m:
            return ("FLIGHT", m.group(1))
        mode = ("selfdrive" if "self-driving" in value.lower()
                else "taxi" if "taxi" in value.lower() else "?")
        m = re.search(r"from\s+(.+?)\s+to\s+(.+?)(?:,|$)", value)
        if m:
            return (mode, _norm_city(m.group(1)), _norm_city(m.group(2)))
        return (mode, value)
    if "," in value:
        name, city = value.rsplit(",", 1)
        return (name.strip(), _norm_city(city))
    return (value,)


def find_action_idx(env, action_type, day, field, value):
    target = (action_type, day, field, _value_key(action_type, value))
    for i, aj in enumerate(env.valid_actions):
        a = json.loads(aj)
        k = (a["action_type"], a["day"], a["field"], _value_key(a["action_type"], a["value"]))
        if k == target:
            return i
    return None


def build_expert_action(field, day, day_dict):
    gold = day_dict.get(field, "-")
    if gold in (None, "-", ""):
        atype = {
            "transportation": SKIP_TRANSPORTATION, "breakfast": SKIP_MEAL,
            "lunch": SKIP_MEAL, "dinner": SKIP_MEAL,
            "attraction": SKIP_ATTRACTION, "accommodation": SKIP_ACCOMMODATION,
        }[field]
        return {"action_type": atype, "day": day, "field": field, "value": "-", "cost": 0}
    if field == "transportation":
        return {"action_type": SET_TRANSPORTATION, "day": day, "field": field, "value": gold, "cost": 0}
    if field in MEAL_FIELDS:
        return {"action_type": SET_MEAL, "day": day, "field": field, "value": gold, "cost": 0}
    if field == "attraction":
        first = gold.split(";")[0].strip()
        return {"action_type": ADD_ATTRACTION, "day": day, "field": field, "value": first, "cost": 0}
    if field == "accommodation":
        return {"action_type": SET_ACCOMMODATION, "day": day, "field": field, "value": gold, "cost": 0}
    raise ValueError(field)


# ---------- fast_step (state-equivalent to env.step, no reward eval)
def fast_step(env, action):
    if action["action_type"] == COMPLETE_PLAN:
        env.state.is_done = True
        return
    env._apply_action(action)
    env.state.spent += action["cost"]
    env._advance_state(action)


def snapshot(env):
    return copy.deepcopy(env.state), env.attraction_count


def restore(env, snap):
    env.state = copy.deepcopy(snap[0])
    env.attraction_count = snap[1]


def state_snap_for_record(env):
    return {
        "spent": round(env.state.spent, 2),
        "current_day": env.state.current_day,
        "current_field": env.state.current_field,
        "field_idx": env.state.field_idx,
        "attraction_count": env.attraction_count,
        "is_done": env.state.is_done,
        "current_plan": copy.deepcopy(env.state.current_plan),
    }


# ---------- equivalence assertion (smoke only)
def assert_equivalent_on_traj(rec, n_steps=18):
    env_slow = TravelPlannerEnv(rec); env_slow.reset()
    env_fast = TravelPlannerEnv(rec); env_fast.reset()
    plan = [d for d in ast.literal_eval(rec["annotated_plan"])[1] if d]
    count = 0
    for d_idx in range(int(rec["days"])):
        day_dict = plan[d_idx] if d_idx < len(plan) else {}
        for field in FIELD_ORDER:
            expert = build_expert_action(field, d_idx + 1, day_dict)
            idx = find_action_idx(env_slow, expert["action_type"], expert["day"],
                                  expert["field"], expert["value"])
            if idx is None:
                return
            env_slow.step(idx)
            fast_step(env_fast, json.loads(env_fast.valid_actions[idx]))
            for attr in ("spent", "current_day", "current_field", "field_idx", "is_done"):
                v_slow = getattr(env_slow.state, attr)
                v_fast = getattr(env_fast.state, attr)
                assert v_slow == v_fast, f"step {count} {attr}: slow={v_slow!r} fast={v_fast!r}"
            assert env_slow.state.current_plan == env_fast.state.current_plan, \
                f"step {count} current_plan differs"
            assert env_slow.attraction_count == env_fast.attraction_count, \
                f"step {count} attraction_count differs"
            count += 1
            if count >= n_steps:
                return


# ---------- per-trajectory rollout
def rollout_one(rec, traj_idx):
    env = TravelPlannerEnv(rec)
    env.reset()
    plan = [d for d in ast.literal_eval(rec["annotated_plan"])[1] if d]
    total_days = int(rec["days"])
    max_day = TRUNCATE_AFTER_DAY.get(traj_idx, total_days)
    records = []
    meta = {"truncated": traj_idx in TRUNCATE_AFTER_DAY,
            "truncate_after_day": TRUNCATE_AFTER_DAY.get(traj_idx)}

    for d_idx in range(max_day):
        day_dict = plan[d_idx] if d_idx < len(plan) else {}
        for field in FIELD_ORDER:
            assert env.state.current_day == d_idx + 1
            assert env.state.current_field == field
            expert = build_expert_action(field, d_idx + 1, day_dict)
            exp_idx = find_action_idx(env, expert["action_type"], expert["day"],
                                      expert["field"], expert["value"])
            if exp_idx is None:
                return records, f"expert not found at d{d_idx+1} {field}: {expert['value'][:60]}", meta
            state_before = state_snap_for_record(env)
            backup = snapshot(env)
            transitions = []
            for i, aj in enumerate(env.valid_actions):
                act = json.loads(aj)
                if not env._is_action_valid(act):
                    continue
                fast_step(env, act)
                transitions.append({
                    "is_expert": (i == exp_idx),
                    "action": act,
                    "spent_after": round(env.state.spent, 2),
                })
                restore(env, backup)
            records.append({
                "traj_idx": traj_idx, "step_idx": len(records),
                "day": d_idx + 1, "field": field,
                "state_before": state_before, "budget": env.initial_budget,
                "n_transitions": len(transitions), "transitions": transitions,
            })
            # Advance with the GYM's version of the expert action (has correct
            # cost). Our build_expert_action returns cost=0 because we only
            # use it to construct an action_key for find_action_idx; the real
            # cost is in valid_actions[exp_idx].
            fast_step(env, json.loads(env.valid_actions[exp_idx]))

    if traj_idx not in TRUNCATE_AFTER_DAY:
        assert env.state.current_day > total_days
        complete_idx = None
        for i, aj in enumerate(env.valid_actions):
            if json.loads(aj)["action_type"] == COMPLETE_PLAN:
                complete_idx = i
                break
        state_before = state_snap_for_record(env)
        backup = snapshot(env)
        transitions = []
        for i, aj in enumerate(env.valid_actions):
            act = json.loads(aj)
            if not env._is_action_valid(act):
                continue
            fast_step(env, act)
            transitions.append({
                "is_expert": (i == complete_idx),
                "action": act, "spent_after": round(env.state.spent, 2),
            })
            restore(env, backup)
        records.append({
            "traj_idx": traj_idx, "step_idx": len(records),
            "day": 0, "field": "complete",
            "state_before": state_before, "budget": env.initial_budget,
            "n_transitions": len(transitions), "transitions": transitions,
        })

    return records, None, meta


# ---------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    ds = load_dataset("osunlp/TravelPlanner", "train", split="train")
    print(f"[load] {len(ds)} training records  ({time.time()-t0:.1f}s)")

    if args.smoke:
        print("[equivalence] env.step vs fast_step on traj 0 (18 steps)...")
        assert_equivalent_on_traj(ds[0], n_steps=18)
        print("[equivalence] OK on traj 0")
        print("[equivalence] same on traj 22 (multi-city, 5d, 7p)...")
        assert_equivalent_on_traj(ds[22], n_steps=18)
        print("[equivalence] OK on traj 22")

    if args.smoke:
        n = 2
    elif args.limit:
        n = min(args.limit, len(ds))
    else:
        n = len(ds)

    smoke_tag = "_smoke" if args.smoke else ("_limit" + str(args.limit) if args.limit else "")
    out_path = Path(args.out) if args.out else OUT_DIR / f"iwm_rollout_paper{smoke_tag}.jsonl"
    summary_path = OUT_DIR / f"iwm_rollout_paper{smoke_tag}_summary.json"

    all_records = []
    per_traj_summary = []
    field_totals = defaultdict(int)
    field_states = defaultdict(int)
    over_budget_alts = Counter()
    expert_misses = 0

    for i in range(n):
        t1 = time.time()
        records, err, meta = rollout_one(ds[i], i)
        all_records.extend(records)
        for r in records:
            field_states[r["field"]] += 1
            field_totals[r["field"]] += r["n_transitions"]
            for t in r["transitions"]:
                if not t["is_expert"] and t["spent_after"] > r["budget"]:
                    over_budget_alts["over"] += 1
                else:
                    over_budget_alts["within"] += 1
        per_traj_summary.append({
            "idx": i, "ok": err is None,
            "n_states": len(records),
            "n_transitions": sum(r["n_transitions"] for r in records),
            "truncated": meta.get("truncated", False),
            "truncate_after_day": meta.get("truncate_after_day"),
            "wall_ms": int((time.time() - t1) * 1000),
            **({"error": err} if err else {}),
        })
        if err:
            expert_misses += 1
            print(f"  [{i+1:>2}/{n}] traj {i:>2}: ERR {err}")
        elif (i + 1) % 10 == 0:
            cum = sum(t["n_transitions"] for t in per_traj_summary)
            print(f"  [{i+1:>2}/{n}] cum transitions = {cum:,}  ({time.time()-t0:.1f}s)")

    out_path.write_text("\n".join(json.dumps(r, default=str) for r in all_records) + "\n")
    total_states = sum(t.get("n_states", 0) for t in per_traj_summary)
    total_trans = sum(t.get("n_transitions", 0) for t in per_traj_summary)

    summary = {
        "n_trajectories": n,
        "total_states": total_states,
        "total_transitions": total_trans,
        "expert_match_failures": expert_misses,
        "alt_budget": dict(over_budget_alts),
        "by_field": {f: {"states": field_states[f], "transitions": field_totals[f],
                         "avg_per_state": round(field_totals[f] / max(field_states[f], 1), 1)}
                     for f in list(FIELD_ORDER) + ["complete"]},
        "per_traj": per_traj_summary,
        "wall_seconds": round(time.time() - t0, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"[done] {n} traj  ->  {total_states} states, {total_trans:,} transitions")
    print(f"[wall] {time.time()-t0:.1f}s")
    print("[by field]")
    for f in list(FIELD_ORDER) + ["complete"]:
        print(f"  {f:<16} states={field_states[f]:>4}  trans={field_totals[f]:>8,}  "
              f"avg/state={field_totals[f] / max(field_states[f], 1):>6.1f}")
    print(f"[out]  {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[sum]  {summary_path}")


if __name__ == "__main__":
    main()

"""
Replay the 45 TravelPlanner training-set expert plans through a pure-function
state machine, emitting (s_i, a_i, s_{i+1}) triples per (day, field) step.

No gym dependency. No LLM. No global database CSV — all costs come from the
per-query ref_info (the exact same data the agent sees at inference time).

Inputs:
  HuggingFace dataset osunlp/TravelPlanner (split=train, 45 records) — provides
  the structured query metadata (org, dest, days, people_number, budget,
  local_constraint, level, date) plus the annotated_plan in canonical form.

  TravelPlanner/database/train_ref_info.jsonl (submodule) — provides per-query
  reference_information as clean JSON dicts (HF ships it as a stringified repr,
  so we prefer the submodule's JSON).

Writes:
  data/replay/replay_full.jsonl     one line per (traj, step)
  data/replay/replay_summary.json   aggregate stats + per-traj outcomes
"""
from __future__ import annotations
import ast
import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset

ENV_ROOT = Path(__file__).resolve().parents[1]
SUBMOD = ENV_ROOT / "TravelPlanner"
DATA_DIR = ENV_ROOT / "data"

PLAN_FIELDS = ("transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation")


def hf_record_to_query(rec: dict) -> dict:
    """Project a HuggingFace osunlp/TravelPlanner record to the structured query dict
    we need for replay. local_constraint and date are stored as stringified Python
    literals upstream — eval them to native types."""
    return {
        "budget": int(rec["budget"]),
        "days": int(rec["days"]),
        "people_number": int(rec["people_number"]),
        "org": rec["org"],
        "dest": rec["dest"],
        "visiting_city_number": int(rec["visiting_city_number"]),
        "date": ast.literal_eval(rec["date"]) if isinstance(rec["date"], str) else rec["date"],
        "local_constraint": (ast.literal_eval(rec["local_constraint"])
                             if isinstance(rec["local_constraint"], str)
                             else rec["local_constraint"]),
        "level": rec["level"],
        "raw_query": rec["query"],
    }


def hf_record_to_plan(rec: dict) -> list[dict]:
    """annotated_plan upstream is a stringified [{meta_dict}, [day1, day2, ..., {}, {}]].
    Return just the days list (and trim trailing empty dicts)."""
    parsed = ast.literal_eval(rec["annotated_plan"])
    days_list = parsed[1]  # parsed[0] is metadata redundancy
    return [d for d in days_list if d]


# ----------------------------------------------------- cost lookups (ref_info)
_FN_RE = re.compile(r"Flight Number:\s*(\S+?)(?:,|$)")
_DRIVE_COST_RE = re.compile(r"cost:\s*\$?([\d.]+)", re.I)


def _ref_lookup_list(ref_info: dict, key_substring: str) -> list[tuple[str, list]]:
    """Find ref_info keys containing the substring whose value is a list."""
    return [(k, v) for k, v in ref_info.items()
            if key_substring in k and isinstance(v, list)]


def find_flight_cost(value: str, ref_info: dict, people: int) -> float:
    m = _FN_RE.search(value)
    if not m:
        raise ValueError(f"cannot parse Flight Number from: {value!r}")
    fn = m.group(1)
    for _, entries in _ref_lookup_list(ref_info, "Flight"):
        for e in entries:
            if e.get("Flight Number") == fn:
                return float(e["Price"]) * people
    raise KeyError(f"flight {fn} not in ref_info")


def find_driving_cost(value: str, ref_info: dict, people: int) -> float:
    """value example: 'Self-driving, from X to Y, duration: ..., distance: ..., cost: 71'."""
    is_self = "self-driving" in value.lower()
    is_taxi = "taxi" in value.lower()
    if not (is_self or is_taxi):
        raise ValueError(f"unrecognized ground transport: {value!r}")
    needle = "Self-driving" if is_self else "Taxi"
    occupancy = 5 if is_self else 4

    for k, v in ref_info.items():
        if needle not in k:
            continue
        if isinstance(v, str) and "no valid information" not in v.lower():
            mc = _DRIVE_COST_RE.search(v)
            if mc:
                return float(mc.group(1)) * math.ceil(people / occupancy)
    raise KeyError(f"{needle} cost for value={value!r} not in ref_info")


def find_meal_cost(value: str, ref_info: dict, people: int) -> float:
    name, city = _split_name_city(value)
    for _, entries in _ref_lookup_list(ref_info, "Restaurants"):
        for e in entries:
            if str(e.get("Name", "")).strip() == name and str(e.get("City", "")).strip() == city:
                return float(e["Average Cost"]) * people
    raise KeyError(f"restaurant {name!r} @ {city!r} not in ref_info")


def find_accom_cost(value: str, ref_info: dict, people: int) -> float:
    name, city = _split_name_city(value)
    for _, entries in _ref_lookup_list(ref_info, "Accommodations"):
        for e in entries:
            if str(e.get("NAME", "")).strip() == name and str(e.get("city", "")).strip() == city:
                return float(e["price"]) * math.ceil(people / int(e["maximum occupancy"]))
    raise KeyError(f"accommodation {name!r} @ {city!r} not in ref_info")


_PAREN_SUFFIX_RE = re.compile(r"\([\w\s]+\)$")


def _split_name_city(value: str) -> tuple[str, str]:
    """value like 'Coco Bambu, Rockford' → ('Coco Bambu', 'Rockford').
    Also handles 'Apna Punjabi Zayka, Boise(Idaho)' → ('Apna Punjabi Zayka', 'Boise')
    — the trailing (State) suffix is what TravelPlanner uses to disambiguate
    same-named cities (matches submodule utils/func.get_valid_name_city)."""
    if "," not in value:
        return value.strip(), ""
    name, city = value.rsplit(",", 1)
    city = _PAREN_SUFFIX_RE.sub("", city.strip()).strip()
    return name.strip(), city


# ----------------------------------------------------- pure-function transition
@dataclass
class TPState:
    query: dict
    spent: float = 0.0
    plan: list[dict] = field(default_factory=list)   # one dict per day
    cursor: int = 0                                   # 0..(days*6)
    done: bool = False

    @classmethod
    def init(cls, query: dict) -> "TPState":
        return cls(query=query,
                   plan=[{f: None for f in PLAN_FIELDS} for _ in range(query["days"])])

    def current_pointer(self) -> tuple[int | None, str | None]:
        if self.done:
            return (None, None)
        d, f = divmod(self.cursor, len(PLAN_FIELDS))
        return d + 1, PLAN_FIELDS[f]


def step(s: TPState, action: dict) -> TPState:
    """Pure transition: deepcopy s, apply action, advance cursor."""
    s2 = copy.deepcopy(s)
    if s2.done:
        raise RuntimeError("step on done state")
    day = action["day"]
    fld = action["field"]
    if action["action_type"].startswith("SET_"):
        s2.plan[day - 1][fld] = action["value"]
        s2.spent += float(action["cost"])
    elif action["action_type"].startswith("SKIP_"):
        s2.plan[day - 1][fld] = "-"
    else:
        raise ValueError(f"unknown action_type: {action['action_type']}")
    s2.cursor += 1
    if s2.cursor >= s.query["days"] * len(PLAN_FIELDS):
        s2.done = True
    return s2


def state_snapshot(s: TPState) -> dict:
    """JSON-serializable snapshot of TPState (no query — that's constant per traj)."""
    return {"spent": s.spent, "cursor": s.cursor, "done": s.done,
            "plan": s.plan}


# ----------------------------------------------------- expert decomposer
def derive_action(d_idx: int, field_name: str, value: str | None,
                  ref_info: dict, people: int) -> dict:
    """Map (day, field, gold-value) → SET_/SKIP_ action with cost."""
    if value in (None, "-"):
        return {"action_type": f"SKIP_{field_name.upper()}",
                "day": d_idx + 1, "field": field_name,
                "value": "-", "cost": 0.0}

    if field_name == "transportation":
        if "flight number" in value.lower():
            cost = find_flight_cost(value, ref_info, people)
        else:  # self-driving / taxi
            cost = find_driving_cost(value, ref_info, people)
    elif field_name in ("breakfast", "lunch", "dinner"):
        cost = find_meal_cost(value, ref_info, people)
    elif field_name == "attraction":
        cost = 0.0   # attractions are free in TravelPlanner
    elif field_name == "accommodation":
        cost = find_accom_cost(value, ref_info, people)
    else:
        raise ValueError(f"unknown field: {field_name}")

    return {"action_type": f"SET_{field_name.upper()}",
            "day": d_idx + 1, "field": field_name,
            "value": value, "cost": cost}


def replay_plan(plan_list: list[dict], ref_info: dict, query: dict) -> tuple[TPState, list[dict]]:
    """Walk the gold plan in state-machine order, returning final state and per-step records."""
    s = TPState.init(query)
    steps = []
    days_in_plan = min(query["days"], len([d for d in plan_list if d]))
    for d_idx in range(query["days"]):
        day_plan = plan_list[d_idx] if d_idx < days_in_plan else {}
        for field_name in PLAN_FIELDS:
            value = day_plan.get(field_name, "-")
            action = derive_action(d_idx, field_name, value, ref_info, query["people_number"])
            s_next = step(s, action)
            steps.append({"step_idx": s.cursor,
                          "day": d_idx + 1, "field": field_name,
                          "state_before": state_snapshot(s),
                          "action": action,
                          "state_after": state_snapshot(s_next)})
            s = s_next
    return s, steps


# ----------------------------------------------------- driver
def main():
    ref_path = SUBMOD / "database" / "train_ref_info.jsonl"
    out_dir = DATA_DIR / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_full = out_dir / "replay_full.jsonl"
    out_summary = out_dir / "replay_summary.json"

    ds = load_dataset("osunlp/TravelPlanner", "train", split="train")
    refs = [json.loads(line) for line in ref_path.read_text().splitlines() if line.strip()]
    assert len(ds) == len(refs) == 45, f"expected 45 records, got HF={len(ds)} refs={len(refs)}"

    per_traj_summary = []
    per_step_dump = []

    for i in range(len(ds)):
        rec = ds[i]
        ref = refs[i]
        try:
            query = hf_record_to_query(rec)
            plan_list = hf_record_to_plan(rec)
        except Exception as e:
            per_traj_summary.append({"idx": i, "ok": False, "phase": "hf_parse",
                                     "error": f"{type(e).__name__}: {e}"})
            continue
        try:
            final_state, steps = replay_plan(plan_list, ref, query)
        except Exception as e:
            per_traj_summary.append({"idx": i, "ok": False, "phase": "replay",
                                     "error": f"{type(e).__name__}: {e}",
                                     "query": query})
            continue

        n_set = sum(1 for s in steps if s["action"]["action_type"].startswith("SET_"))
        n_skip = sum(1 for s in steps if s["action"]["action_type"].startswith("SKIP_"))
        per_traj_summary.append({
            "idx": i, "ok": True,
            "query": query,
            "num_steps": len(steps),
            "num_set": n_set, "num_skip": n_skip,
            "final_spent": round(final_state.spent, 2),
            "budget": query["budget"],
            "within_budget": final_state.spent <= query["budget"],
            "over_by": round(max(0.0, final_state.spent - query["budget"]), 2),
        })
        for s in steps:
            per_step_dump.append({"traj_idx": i, **s})

    out_full.write_text("\n".join(json.dumps(r, default=str) for r in per_step_dump) + "\n")

    ok_count = sum(1 for r in per_traj_summary if r["ok"])
    within = sum(1 for r in per_traj_summary if r.get("within_budget"))
    summary = {
        "n_total": len(per_traj_summary),
        "n_replay_ok": ok_count,
        "n_within_budget": within,
        "n_over_budget": ok_count - within,
        "trajectories": per_traj_summary,
    }
    out_summary.write_text(json.dumps(summary, indent=2, default=str))

    print(f"[replay] {ok_count}/{len(per_traj_summary)} trajectories replayed clean")
    print(f"[budget] {within}/{ok_count} within budget; {ok_count - within} over")
    print(f"[steps]  {sum(r.get('num_steps', 0) for r in per_traj_summary)} total step-records → {out_full}")
    print(f"[out]    summary → {out_summary}")
    # surface a few sample errors / over-budget for the operator
    for r in per_traj_summary:
        if not r["ok"]:
            print(f"  FAIL  idx={r['idx']}  phase={r['phase']}  err={r['error'][:120]}")
        elif not r["within_budget"]:
            print(f"  OVER  idx={r['idx']}  spent={r['final_spent']} > budget={r['budget']} (over_by={r['over_by']})")


if __name__ == "__main__":
    main()

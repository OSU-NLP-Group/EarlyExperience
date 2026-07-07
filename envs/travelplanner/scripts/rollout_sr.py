"""SR (Self-Reflection) generation for TravelPlanner — Stage C.

Walks the IWM rollout (data/rollout/iwm_rollout_paper.jsonl), categorizes each
state into one of five SR prompt types, and calls DeepSeek V4 Pro to produce
the reflection CoT.

Categories (locked 2026-06-01, see NOTES.md):
  A — contrastive (K=6): regular state with ≥1 alt available (1,035 states)
  B — SKIP_ACCOMMODATION on the last day (44, env structurally forces SKIP)
  C — SKIP_MEAL with budget exhausted (24, only remaining option is SKIP)
  D — SKIP_TRANSPORTATION on a stay-day (2, no inter-city travel today)
  E — COMPLETE_PLAN (44 states; ONE template generated, reused with
       per-traj numeric substitution)
  skip — attraction (221) and structural-only forced states without
       reasoning content (none in this design after the cuts above)

Usage:
  PYTHONNOUSERSITE=1 DEEPSEEK_API_KEY=... \
    /home/ulss/miniconda3/envs/travelplanner-ee/bin/python rollout_sr.py --smoke
  (smoke: 10 states, serial, verbose; full: 1,106 calls, concurrent)
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

ENV_ROOT = Path(__file__).resolve().parents[1]
IWM_PATH = ENV_ROOT / "data" / "rollout" / "iwm_rollout_paper.jsonl"
GLOBAL_DB = ENV_ROOT / "global_db"
OUT_DIR = ENV_ROOT / "data" / "rollout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEAL_FIELDS = ("breakfast", "lunch", "dinner")
FIELD_ORDER = ("transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation")
BANNED_VOCAB = (
    "expert", "selected", "chosen", "picked", "correct", "optimal",
    "right action", "right choice",
)
# Note: "best" and "ideal" intentionally excluded — both are common adverbs in
# English ("works best", "not ideal") that generate false positives without
# indicating actual expert-label leakage. The prompt's guidelines still tell
# the LLM not to use them as action labels; we just don't grep for them.
NUMBERED_LABELS = re.compile(r"\b(Action|Alternative|Option)\s+\d+\b", re.I)
_PAREN = re.compile(r"\([\w\s]+\)$")
def norm_city(s): return _PAREN.sub("", s.strip()).strip()


# =========================================================================
# Categorization
# =========================================================================
def categorize(record: dict, query_meta: dict) -> tuple[str | None, dict]:
    """Return (type, info) where type ∈ {A, B, C, D, E, None=skip}."""
    field = record["field"]
    expert = next(t["action"] for t in record["transitions"] if t["is_expert"])
    n_alts = sum(1 for t in record["transitions"] if not t["is_expert"])

    if expert["action_type"] == "COMPLETE_PLAN":
        return "E", {"expert": expert}

    if n_alts == 0:
        # Forced choice — classify by why
        if expert["action_type"] == "SKIP_ACCOMMODATION" and record["day"] == query_meta["days"]:
            return "B", {"expert": expert}
        if expert["action_type"] == "SKIP_MEAL":
            return "C", {"expert": expert}
        if expert["action_type"] == "SKIP_TRANSPORTATION":
            return "D", {"expert": expert}
        return None, {"reason": "forced choice, no reasoning content"}

    return "A", {"expert": expert}


# =========================================================================
# Contrastive alt selection for type A
# =========================================================================
def load_cuisine_lookup(db_root: Path) -> dict:
    df = pd.read_csv(db_root / "restaurants" / "clean_restaurant_2022.csv").dropna()
    return {(str(r["Name"]).strip(), str(r["City"]).strip()): str(r["Cuisines"])
            for _, r in df.iterrows()}


def load_accom_lookup(db_root: Path) -> dict:
    """Return {(NAME_normalized, city_normalized): {min_nights, room_type,
    max_occupancy, rating, price}}. Used to add min_nights/room_type to alt
    rendering (paper gym's _format_accommodation_options exposes these)."""
    df = pd.read_csv(db_root / "accommodations" / "clean_accommodations_2022.csv").dropna()
    out = {}
    for _, r in df.iterrows():
        key = (str(r["NAME"]).strip(), str(r["city"]).strip())
        out[key] = {
            "min_nights": int(r["minimum nights"]) if not pd.isna(r["minimum nights"]) else None,
            "room_type": str(r["room type"]).strip(),
            "max_occupancy": int(r["maximum occupancy"]) if not pd.isna(r["maximum occupancy"]) else None,
            "rating": float(r["review rate number"]) if not pd.isna(r["review rate number"]) else None,
            "price": float(r["price"]) if not pd.isna(r["price"]) else None,
            "house_rules": str(r["house_rules"]).strip() if not pd.isna(r["house_rules"]) else "",
        }
    return out


def _classify_mode(v):
    if not v or v in ("-", "PENDING", None): return None
    s = v.lower()
    if "flight" in s: return "Flight"
    if "self-driving" in s: return "Self-driving"
    if "taxi" in s: return "Taxi"
    return None


def _used_modes_before(plan, day):
    out = set()
    for d in range(1, day):
        m = _classify_mode(plan.get(f"day_{d}", {}).get("transportation", ""))
        if m: out.add(m)
    return out


def _restaurants_used(plan, days):
    out = []
    for d in range(1, days + 1):
        day = plan.get(f"day_{d}", {})
        for meal in MEAL_FIELDS:
            v = day.get(meal, "")
            if v and v not in ("-", "PENDING", None):
                out.append(v)
    return out


def _attractions_used(plan, days):
    out = []
    for d in range(1, days + 1):
        v = plan.get(f"day_{d}", {}).get("attraction", "")
        if v and v not in ("-", "PENDING", None):
            for venue in str(v).split(";"):
                venue = venue.strip()
                if venue:
                    out.append(venue)
    return out


def _parse_transport_destination(value: str) -> str | None:
    if not value or value in ("-", "PENDING"):
        return None
    m = re.search(r"\bto\s+([A-Za-z][^,]*?)(?:,|$)", value)
    if m:
        return norm_city(m.group(1))
    return None


def _allowed_cities_today(plan, day, query_meta, field=None):
    """Return the set of cities an alt at this (day, field) could be in.

    - For accommodation: only the END city (where the agent SLEEPS tonight,
      which equals today's transport destination if any, else today's
      starting city). FROM-city accommodation alts on travel days are
      treated as wrong-city, matching the gym's semantics.
    - For meals / attraction / transportation: both FROM and TO cities are
      acceptable (the gym enumerates options in both cities on travel days).
    """
    today = plan.get(f"day_{day}", {})
    if day == 1:
        start = norm_city(query_meta.get("org", ""))
    else:
        prev = plan.get(f"day_{day - 1}", {})
        start = None
        for f in ("accommodation", "dinner", "lunch", "breakfast", "attraction"):
            v = prev.get(f, "")
            if v and v != "-" and v != "PENDING" and "," in v:
                start = norm_city(v.rsplit(",", 1)[1]); break
        if start is None:
            start = norm_city(query_meta.get("org", ""))
    trans_val = today.get("transportation", "")
    end = _parse_transport_destination(trans_val) if trans_val else None
    if field == "accommodation":
        sleep_city = end or start
        return {sleep_city} if sleep_city else set()
    allowed = {start}
    if end:
        allowed.add(end)
    return {c for c in allowed if c}


def _tonight_sleep_city(plan, day, query_meta):
    """City where the agent sleeps tonight (alias for the accommodation case
    of _allowed_cities_today). Used to give the SR prompt an explicit hint
    so the model doesn't have to derive it from a late-evening flight."""
    s = _allowed_cities_today(plan, day, query_meta, field="accommodation")
    return next(iter(s)) if s else None


def _visited_destination_cities(plan, day, query_meta):
    """Return ordered list of (first_day_seen, city) for destination cities
    the agent has actually been in so far (excluding origin). Derived from
    every filled meal/attraction/accommodation in the plan-so-far."""
    origin = norm_city(query_meta.get("org", ""))
    seen: dict[str, int] = {}
    for d in range(1, day):  # only days prior to (or up to but not including) today
        day_plan = plan.get(f"day_{d}", {})
        for f in ("accommodation", "dinner", "lunch", "breakfast", "attraction"):
            v = day_plan.get(f, "")
            if v and v not in ("-", "PENDING", None) and "," in v:
                c = norm_city(v.rsplit(",", 1)[1])
                if c and c != origin and c not in seen:
                    seen[c] = d
    return [(d, c) for c, d in sorted(seen.items(), key=lambda kv: kv[1])]


def _is_travel_day(plan, day):
    cc = plan.get(f"day_{day}", {}).get("current_city", "") or ""
    return "from" in cc and " to " in cc


def tag_alt(record: dict, alt_action: dict, query_meta: dict,
            cuisine_lookup: dict, accom_lookup: dict) -> list[str]:
    """Return WRONG-tags for this alt. Empty list means the alt is just a
    valid alternative (no grounded reason to reject it) — caller will skip
    such alts to avoid fabricated-preference SR text."""
    tags = []
    field = record["field"]
    day = record["day"]
    plan = record["state_before"]["current_plan"]
    days = query_meta["days"]
    constraints = query_meta.get("constraints") or {}
    at = alt_action["action_type"]
    val = alt_action.get("value", "")

    # 1. SKIP on a mandatory stay-day/non-final-day field
    if at.startswith("SKIP_"):
        if field in MEAL_FIELDS and not _is_travel_day(plan, day):
            tags.append("mandatory_field_skip")
        elif field == "attraction" and not _is_travel_day(plan, day):
            tags.append("mandatory_field_skip")
        elif field == "accommodation" and day < days:
            tags.append("mandatory_field_skip")
        elif field == "transportation" and _is_travel_day(plan, day):
            tags.append("mandatory_field_skip")

    # 2. repeat checks
    if field in MEAL_FIELDS and not at.startswith("SKIP_") and val in _restaurants_used(plan, days):
        tags.append("repeats_restaurant")
    if field == "attraction" and not at.startswith("SKIP_") and val in _attractions_used(plan, days):
        tags.append("repeats_attraction")

    # 3. cuisine_violate
    if field in MEAL_FIELDS and constraints.get("cuisine") and not at.startswith("SKIP_"):
        if "," in val:
            name, city = val.rsplit(",", 1)
            cui = cuisine_lookup.get((name.strip(), norm_city(city)), "")
            if cui and not any(c in cui for c in constraints["cuisine"]):
                tags.append("cuisine_violate")

    # 4. transport_violate (query preference)
    if field == "transportation" and constraints.get("transportation") and not at.startswith("SKIP_"):
        v = val.lower()
        pref = str(constraints["transportation"]).lower()
        if "no self-driving" in pref and "self-driving" in v:
            tags.append("transport_violate")
        if "no flight" in pref and "flight" in v:
            tags.append("transport_violate")

    # 5. house_rule_violate
    if field == "accommodation" and constraints.get("house rule") and not at.startswith("SKIP_"):
        if "," in val:
            name, city = val.rsplit(",", 1)
            info = accom_lookup.get((name.strip(), norm_city(city)), {})
            rules = info.get("house_rules", "")
            qhr = str(constraints["house rule"]).lower()
            ban_map = {"smoking": "No smoking", "parties": "No parties",
                       "children under 10": "No children under 10",
                       "visitors": "No visitors", "pets": "No pets"}
            if ban_map.get(qhr) and ban_map[qhr] in rules:
                tags.append("house_rule_violate")

    # 6. room_type_violate
    if field == "accommodation" and constraints.get("room type") and not at.startswith("SKIP_"):
        if "," in val:
            name, city = val.rsplit(",", 1)
            info = accom_lookup.get((name.strip(), norm_city(city)), {})
            rt = info.get("room_type", "")
            qrt = str(constraints["room type"]).lower()
            if ((qrt == "private room" and rt != "Private room")
                or (qrt == "shared room" and rt != "Shared room")
                or (qrt == "entire room" and rt != "Entire home/apt")
                or (qrt == "not shared room" and rt == "Shared room")):
                tags.append("room_type_violate")

    # 7. breaks_min_nights — alt accommodation at a NEW place that can't get
    # enough consecutive nights before the trip's last day (no accom required).
    if field == "accommodation" and not at.startswith("SKIP_"):
        prev_accom = plan.get(f"day_{day - 1}", {}).get("accommodation", "")
        if val != prev_accom:
            remaining_for_accom = max(1, (days - 1) - day + 1)
            if "," in val:
                name, city = val.rsplit(",", 1)
                info = accom_lookup.get((name.strip(), norm_city(city)), {})
                mn = info.get("min_nights")
                if mn and mn > remaining_for_accom:
                    tags.append("breaks_min_nights")

    # 8. breaks_mode_chain
    if field == "transportation" and not at.startswith("SKIP_"):
        prev = _used_modes_before(plan, day)
        m = _classify_mode(val)
        if m:
            if "Self-driving" in prev and m != "Self-driving":
                tags.append("breaks_mode_chain")
            if m == "Self-driving" and ("Flight" in prev or "Taxi" in prev):
                tags.append("breaks_mode_chain")

    # 9. wrong_city — alt for meal/attraction/accommodation is in a city the
    # agent cannot be in for THIS field. Accommodation is restricted to the
    # END city (where the agent sleeps tonight); meals/attractions can be
    # in either the FROM or TO city on travel days.
    if field in MEAL_FIELDS + ("attraction", "accommodation") and not at.startswith("SKIP_"):
        if "," in val:
            _, city = val.rsplit(",", 1)
            alt_city = norm_city(city)
            allowed = _allowed_cities_today(plan, day, query_meta, field=field)
            if allowed and alt_city not in allowed:
                tags.append("wrong_city")

    return tags  # empty → not a "wrong" alt; caller should drop


_TAG_PRIORITY = {
    # Severity used for tie-breaking when slots compete. wrong_city sits with
    # repeats — both are basic "this option just doesn't fit at all" failures.
    "mandatory_field_skip": 6, "breaks_min_nights": 6, "breaks_mode_chain": 6,
    "wrong_city": 5, "repeats_restaurant": 5, "repeats_attraction": 5,
    "house_rule_violate": 4, "room_type_violate": 4, "transport_violate": 4,
    "cuisine_violate": 3,
}


def select_alts(record: dict, query_meta: dict, cuisine_lookup: dict,
                accom_lookup: dict, K: int = 8) -> list[dict]:
    """Stratified selection of up to K truly-wrong alts.
    Round 1: at least one alt per distinct wrong-tag class actually present
    (highest-priority tag first), so every available failure mode gets at
    least one example in the prompt. Round 2: fill remaining slots by
    priority. Returns [] if no wrong alt exists — caller skips the state."""
    alts = [t for t in record["transitions"] if not t["is_expert"]]
    wrong = []
    for t in alts:
        tags = tag_alt(record, t["action"], query_meta, cuisine_lookup, accom_lookup)
        if tags:
            wrong.append({"transition": t, "tags": tags,
                          "prio": max(_TAG_PRIORITY[tag] for tag in tags)})

    by_tag: dict[str, list[dict]] = {}
    for w in wrong:
        for tag in w["tags"]:
            by_tag.setdefault(tag, []).append(w)

    selected: list[dict] = []
    seen_ids: set[int] = set()
    # Round 1: one alt per tag class, highest priority first
    for tag in sorted(by_tag.keys(), key=lambda t: -_TAG_PRIORITY[t]):
        for w in by_tag[tag]:
            if id(w) not in seen_ids:
                selected.append(w); seen_ids.add(id(w))
                break
        if len(selected) >= K:
            break
    # Round 2: fill remaining slots by priority
    for w in sorted(wrong, key=lambda x: -x["prio"]):
        if len(selected) >= K:
            break
        if id(w) not in seen_ids:
            selected.append(w); seen_ids.add(id(w))
    # Stable display order: by priority again so highest severity shows first
    selected.sort(key=lambda x: -x["prio"])
    return selected[:K]


# =========================================================================
# State + action rendering
# =========================================================================
_FLIGHT_NO_RE = re.compile(r"Flight Number:\s*(\S+?)(?:,|$)")


def _classify_transport_mode(value: str) -> str | None:
    if not value or value in ("-", "PENDING"):
        return None
    if "Flight" in value:
        return "Flight"
    if "self-driving" in value.lower():
        return "Self-driving"
    if "taxi" in value.lower():
        return "Taxi"
    return None


def render_plan(current_plan: dict, total_days: int,
                cuisine_lookup: dict | None = None,
                accom_lookup: dict | None = None) -> str:
    """Render the plan-so-far WITHOUT truncating values. Annotate accommodation
    with its min_nights (critical: paper evaluator fails plans whose consecutive
    nights at a place < min_nights). Annotate transportation with mode label."""
    lines = []
    for d in range(1, total_days + 1):
        day = current_plan.get(f"day_{d}", {})
        cc = day.get("current_city", "?")
        lines.append(f"Day {d} (current_city: {cc}):")
        for f in FIELD_ORDER:
            v = day.get(f, "PENDING")
            extra = ""
            if v not in (None, "-", "PENDING") and isinstance(v, str):
                if f == "transportation":
                    mode = _classify_transport_mode(v)
                    if mode:
                        extra = f"  [mode: {mode}]"
                elif f == "accommodation" and accom_lookup and "," in v:
                    name, city = v.rsplit(",", 1)
                    info = accom_lookup.get((name.strip(), norm_city(city)), {})
                    if info.get("min_nights") is not None:
                        extra = f"  [min_nights={info['min_nights']}]"
            lines.append(f"  {f}: {v}{extra}")
    return "\n".join(lines)


def render_constraints(constraints: dict) -> str:
    parts = [f"{k}: {v!r}" for k, v in constraints.items() if v is not None]
    return ", ".join(parts) if parts else "None specified (only budget applies)"


def render_action_short(action: dict,
                        cuisine_lookup: dict | None = None,
                        accom_lookup: dict | None = None) -> str:
    """Compact action description with metadata vital to the 4 hard constraints:
    meal → cuisine; accommodation → min_nights + room_type + max_occupancy;
    transportation → mode (Flight/Self-driving/Taxi) tag."""
    at = action["action_type"]
    cost = action.get("cost", 0)
    val = action.get("value", "")
    if at.startswith("SKIP_"):
        return f"SKIP {at.replace('SKIP_', '').lower()} (no cost)"
    if at == "COMPLETE_PLAN":
        return "COMPLETE the plan"
    if at == "SET_TRANSPORTATION":
        mode = _classify_transport_mode(val) or "?"
        return f"{val} (cost ${cost}, mode: {mode})"
    if at == "SET_MEAL":
        cuisine = ""
        if cuisine_lookup and "," in val:
            name, city = val.rsplit(",", 1)
            cuisine = cuisine_lookup.get((name.strip(), norm_city(city)), "")
        if cuisine:
            return f"{val} (${cost}, cuisine: {cuisine})"
        return f"{val} (${cost})"
    if at == "SET_ACCOMMODATION":
        parts = [f"${cost}/night total"]
        if accom_lookup and "," in val:
            name, city = val.rsplit(",", 1)
            info = accom_lookup.get((name.strip(), norm_city(city)), {})
            if info.get("min_nights") is not None:
                parts.append(f"min {info['min_nights']} nights")
            if info.get("room_type"):
                parts.append(info["room_type"])
            if info.get("max_occupancy") is not None:
                parts.append(f"max {info['max_occupancy']} guests/room")
            if info.get("house_rules"):
                parts.append(f'house_rules: "{info["house_rules"]}"')
        return f"{val} ({', '.join(parts)})"
    if at == "ADD_ATTRACTION":
        return f"Visit {val} (free)"
    return f"{val} (${cost})"


def render_alt_line(picked: dict, budget: float,
                    cuisine_lookup: dict | None = None,
                    accom_lookup: dict | None = None) -> str:
    """v5: NO annotation suffix. The alt was pre-selected by `select_alts`
    because it has at least one real rule violation — but the LLM must
    derive the rejection reason from the raw plan-so-far + alt metadata,
    not by quoting an annotation that won't exist at inference time."""
    t = picked["transition"]
    a = t["action"]
    short = render_action_short(a, cuisine_lookup, accom_lookup)
    spent_after = t["spent_after"]
    rem = budget - spent_after
    return (f"- {short} → would leave spent ${spent_after:.0f}/${int(budget)} "
            f"({rem:.0f} remaining)")


def build_history_digest(state: dict, query_meta: dict,
                         accom_lookup: dict | None = None) -> str:
    """Produce a compact summary of what's already locked in, so the LLM
    doesn't have to reconstruct it from plan-so-far. Surfaces exactly the
    facts needed to reason about the 4 ASR-killing constraints."""
    plan = state["current_plan"]
    days = query_meta["days"]
    origin = query_meta.get("org", "?")

    # Cities visited (compact sequence, dedup consecutive)
    city_seq = [origin]
    for d in range(1, days + 1):
        day = plan.get(f"day_{d}", {})
        cc = day.get("current_city", "")
        if cc in ("", "?", None):
            continue
        m = re.match(r"from\s+(.+?)\s+to\s+(.+)", cc)
        if m:
            a = norm_city(m.group(1)); b = norm_city(m.group(2))
            if city_seq and city_seq[-1] != a:
                city_seq.append(a)
            if city_seq[-1] != b:
                city_seq.append(b)
        else:
            c = norm_city(cc)
            if city_seq and city_seq[-1] != c:
                city_seq.append(c)

    # Transportation modes used
    transport_log = []
    for d in range(1, days + 1):
        day = plan.get(f"day_{d}", {})
        v = day.get("transportation", "")
        mode = _classify_transport_mode(v) if v else None
        if mode:
            transport_log.append((d, mode))
    modes_used = sorted(set(m for _, m in transport_log))
    transport_lock = ""
    if "Self-driving" in modes_used:
        transport_lock = " — LOCKED: Self-driving used; all subsequent transport must also be Self-driving (no Flight/Taxi)."

    # Restaurants used
    restaurants = []
    for d in range(1, days + 1):
        day = plan.get(f"day_{d}", {})
        for meal in ("breakfast", "lunch", "dinner"):
            v = day.get(meal, "")
            if v and v not in ("-", "PENDING", None):
                restaurants.append(v)

    # Attractions visited (split each day's attraction by ';' since one day may
    # hold multiple venues; the evaluator checks for repeats per-venue)
    attractions = []
    for d in range(1, days + 1):
        day = plan.get(f"day_{d}", {})
        v = day.get("attraction", "")
        if v and v not in ("-", "PENDING", None):
            for venue in str(v).split(";"):
                venue = venue.strip()
                if venue:
                    attractions.append(venue)

    # Visiting-city target
    target_vcn = query_meta.get("visiting_city_number")
    distinct_dest_cities = set(city_seq) - {origin}

    # Accommodation blocks (consecutive same name)
    accom_blocks = []
    last_name = None
    for d in range(1, days + 1):
        day = plan.get(f"day_{d}", {})
        v = day.get("accommodation", "")
        if v and v not in ("-", "PENDING", None):
            if v == last_name and accom_blocks:
                accom_blocks[-1]["days"].append(d)
            else:
                accom_blocks.append({"name": v, "days": [d]})
                last_name = v
        else:
            last_name = None

    lines = []
    lines.append(f"Cities visited so far: {' → '.join(city_seq)}")
    lines.append(f"  → Trip must close back to {origin} by the final day (day {days}).")
    if target_vcn is not None:
        n_so_far = len(distinct_dest_cities)
        status = "✓" if n_so_far == target_vcn else f"need {target_vcn - n_so_far} more" if n_so_far < target_vcn else f"already {n_so_far - target_vcn} OVER target"
        lines.append(f"  → visiting_city_number target = {target_vcn}; distinct destination cities so far = {n_so_far} ({status}).")
    if transport_log:
        log = ", ".join(f"day {d} {m}" for d, m in transport_log)
        lines.append(f"Transportation used so far: {log}{transport_lock}")
    else:
        lines.append("Transportation used so far: (none yet)")
    if restaurants:
        lines.append("Restaurants used so far (CANNOT repeat any of these):")
        for r in restaurants:
            lines.append(f"  - {r}")
    else:
        lines.append("Restaurants used so far: (none yet)")
    if attractions:
        lines.append("Attractions visited so far (CANNOT repeat any of these):")
        for a in attractions:
            lines.append(f"  - {a}")
    else:
        lines.append("Attractions visited so far: (none yet)")
    if accom_blocks:
        lines.append("Accommodation bookings so far:")
        for blk in accom_blocks:
            name = blk["name"]
            ds = blk["days"]
            consec = len(ds)
            day_range = f"day {ds[0]}" if len(ds) == 1 else f"days {ds[0]}-{ds[-1]}"
            min_n_str = "?"
            ok = ""
            if accom_lookup and "," in name:
                n, c = name.rsplit(",", 1)
                info = accom_lookup.get((n.strip(), norm_city(c)), {})
                if info.get("min_nights") is not None:
                    min_n = info["min_nights"]
                    min_n_str = str(min_n)
                    if consec >= min_n:
                        ok = " ✓ min_nights satisfied"
                    else:
                        ok = f" ⚠ NEEDS {min_n - consec} MORE consecutive night(s)"
            lines.append(f"  - {name} ({day_range}, {consec} consecutive night(s), min_nights={min_n_str}){ok}")
    else:
        lines.append("Accommodation bookings so far: (none yet)")
    return "\n".join(lines)


HARD_CONSTRAINTS_BLOCK = """These are the rules a plan must obey. You know them as an experienced planner — use them naturally, not by quoting their names:

- The trip must end back where it started. Day 1 leaves the origin city; the final day's transportation must bring you back.
- The trip should visit exactly the number of destination cities the traveler asked for — don't skip planned cities and don't add unplanned ones.
- A restaurant can be used at most once across the entire trip. If you've eaten somewhere already (any day, any meal), pick somewhere new.
- An attraction venue can be visited at most once across the entire trip.
- Transportation modes don't mix freely. If you drove yourself on any day, you have a car — every other day's transportation must also be self-driving (you can't suddenly take a flight and leave the car behind). Flights and taxis can mix freely with each other, but neither can mix with self-driving.
- Every accommodation has a minimum-nights requirement. The number of CONSECUTIVE nights you actually stay at a place must reach that minimum. If you book somewhere with a 3-night minimum but you only stay 2 nights, the booking is invalid.
- max_occupancy is PER ROOM, not per booking. If the group is bigger, multiple rooms are automatically booked and the price shown already includes them. Never reject a place just because the per-room capacity is smaller than the group.
- If the traveler asked for a specific house rule (e.g. they want to be able to smoke / bring pets / host parties / travel with children under 10 / have visitors), the chosen accommodation's house rules must not explicitly forbid that activity. Other unrelated rules are fine.
- If the traveler specified a room type (private room / shared room / entire room / not shared room), the chosen accommodation's room type must match.
- If the traveler specified a transportation preference (e.g. "no self-driving" or "no flight"), avoid that mode entirely.
- If the traveler specified preferred cuisines, each restaurant should serve at least one of them.
- Travel days require transportation. Stay days require breakfast, lunch, dinner, AND an attraction (skipping any of them fails the plan, even to save money). Accommodation is required on every day except the final day. If the budget is too tight to satisfy these requirements, the plan was already overspent earlier — the fix isn't to skip a mandatory field here.
- Every meal, attraction, and accommodation must be in the city you actually are in that day. On a travel day this can be either the FROM city (early in the day) or the TO city (after the journey), depending on the field. Some options shown may be in a different city entirely — those are out of scope and must be rejected explicitly."""


# =========================================================================
# Prompt builders
# =========================================================================
GUIDELINES = """- You are deciding in real time, weighing the options now. Do NOT write phrases like "X was selected", "X was chosen", "this option was picked", "the action taken is X", or "the decision made was X". These treat the decision as already made and externally labeled, which is wrong — YOU are arriving at the decision through analysis right now.
- Instead, reason from the constraints/budget to a conclusion: "I'll go with X because...", "X works here because...", "Let me take X — it...". The reflection must ARRIVE AT the action through analysis, not announce it with a label.
- Do NOT use these labeling words to describe the action you decide on: expert, selected, chosen, picked, correct, optimal, best, right action, right choice. The reasoning arrives at the action; it does not name it.
- Write as the agent's own first-person reasoning (e.g., "I'll go with...", "My budget is tight, so..."), not as a third-party analysis.
- Do not refer to the alternatives by numbers (e.g., "Action 1", "Alternative 2"). Refer to them by their content ("the $97 dinner at Shree Balaji", "skipping breakfast").
- Use natural, step-by-step reasoning.

=== NO PROMPT-STRUCTURE LEAKAGE (critical) ===
At inference time the agent sees a PLAIN state — no "Restaurants used so far" section header, no "INVALID because:" annotations, no "HARD CONSTRAINTS" block, no numbered rule list. The agent must reason from scratch on every step. So your reflection must read like the agent's own natural inner voice working from the raw plan-so-far, NOT like someone reading off pre-marked annotations.

FORBIDDEN phrasings (these reference scaffolding that vanishes at inference and will mistrain the model):
- "the annotation says..." / "marked invalid" / "the system shows..." / "the prompt tells me..."
- "from the 'Restaurants used so far' list" / "as listed in the digest" / quoting any section name verbatim
- "the is_not_absent rule" / "breaks_mode_chain" / "violates the no-repeat constraint" / any internal-sounding rule name
- "the hard constraint says..." / "constraint #5" / "per rule N"
- "exactly as the annotation states" / "the annotation confirms"

INSTEAD reason directly from the raw plan-so-far. Examples:
  bad:  "Coco Bambu is in the Restaurants used so far list, so it's a repeat violation."
  good: "I already had dinner at Coco Bambu on day 1, so picking it again would put the same restaurant in the plan twice — that's not allowed."

  bad:  "The taxi options are all marked invalid because they break the mode-chain constraint."
  good: "I self-drove on day 1, so I have a car with me. Taking a taxi now would leave the car stranded — I have to keep self-driving the rest of the trip."

  bad:  "Skipping breakfast fails the is_not_absent rule on a stay day."
  good: "Day 2 is a full day in Rockford, not a travel day — every meal slot needs to be filled, so skipping breakfast isn't an option."

=== NO FABRICATED PREFERENCES ===
The agent's choice is NOT necessarily the cheapest, most expensive, highest-rated, or most "popular" option. It is one valid choice among potentially several. Do NOT invent reasons that aren't in the data:
- no "X has better reviews / food quality / ambience" (no such data)
- no "the cheaper option saves money for later" unless you can name a specific upcoming required cost the agent wouldn't otherwise afford
- no "staying at the same accommodation is more convenient" except as a direct consequence of the min-nights chain
- no "this restaurant feels appropriate for breakfast / lunch / dinner" (no such data)
If multiple options here would all be valid, say so honestly: "Several options here would work; X is one fine choice."

Output the monologue text only — no headings, no disclaimers, no meta-commentary. Follow the word target given in the TASK section; pick the action and explain the relevant rule(s) it depends on, no more."""

SYS_PROMPT = "You are reasoning through a travel-planning decision as the agent yourself."


def build_prompt_A(record, expert, picked_alts, query_meta, constraints,
                   cuisine_lookup, accom_lookup):
    state = record["state_before"]
    budget = record["budget"]
    spent_b = state["spent"]
    remaining_b = budget - spent_b
    expert_short = render_action_short(expert, cuisine_lookup, accom_lookup)
    expert_after = next(t["spent_after"] for t in record["transitions"] if t["is_expert"])
    alts_block = "\n".join(render_alt_line(p, budget, cuisine_lookup, accom_lookup)
                           for p in picked_alts)
    digest = build_history_digest(state, query_meta, accom_lookup)
    # Accommodation hint: tonight's sleep city — disambiguates evening-travel
    # days where the agent's intuition can say "still in FROM city" but the
    # gym rule is accom = where you sleep = destination of today's transport.
    accom_hint = ""
    if record["field"] == "accommodation":
        sleep_city = _tonight_sleep_city(state["current_plan"], record["day"], query_meta)
        if sleep_city:
            today_trans = state["current_plan"].get(f"day_{record['day']}", {}).get("transportation", "")
            note = ""
            if today_trans and today_trans != "-":
                src = re.search(r"\bfrom\s+([A-Za-z][^,]*?)(?:\s+to\b|,)", today_trans)
                if src:
                    note = f" (the agent arrives there via today's transport from {norm_city(src.group(1))})"
            accom_hint = f"\n\n=== TONIGHT'S SLEEP CITY ===\n{sleep_city}{note}. Every accommodation option must be in {sleep_city} — options in any other city are out of scope, even on a travel day where part of the day was spent elsewhere."

    # Transportation gets an explicit city-sequence tally so reasoning can
    # engage with high-level "which destination next" instead of only mode.
    city_seq_hint = ""
    if record["field"] == "transportation":
        visited = _visited_destination_cities(state["current_plan"], record["day"], query_meta)
        target_count = query_meta.get("visiting_city_number", 0)
        origin = norm_city(query_meta.get("org", ""))
        days = query_meta["days"]
        visited_str = ", ".join(f"{c} (entered d{d})" for d, c in visited) if visited else "none yet"
        still_need = max(0, target_count - len(visited))
        is_final = (record["day"] == days)
        return_note = (" Today is the final day — transportation must return to origin."
                       if is_final else f" Final day (d{days}) must return to {origin}.")
        city_seq_hint = (
            f"\n\n=== CITY SEQUENCE ===\n"
            f"Origin: {origin}. Destination cities required by query: {target_count}.\n"
            f"Visited so far: {visited_str}.\n"
            f"Still need {still_need} more destination city/cities before the day-{days} return."
            f"{return_note}"
        )

    # Cuisine requirement marker — surfaces the query's cuisine preference
    # so the reasoning is forced to address it when picking a meal.
    cuisine_hint = ""
    if record["field"] in MEAL_FIELDS:
        cui_pref = (constraints or {}).get("cuisine") if constraints else None
        if cui_pref:
            cuisine_hint = (
                f"\n\n=== CUISINE PREFERENCE ===\n"
                f"The traveler asked for: {', '.join(cui_pref)}. "
                f"The chosen restaurant should serve at least one of these. "
                f"If no available option does, the reflection must explicitly acknowledge the forced mismatch."
            )
    return f"""=== CURRENT PLANNING STATE ===
Query: {query_meta['query']}
Total Days: {query_meta['days']}, Initial Budget: ${int(budget)}
Spent so far: ${spent_b:.0f}, Remaining: ${remaining_b:.0f}

Plan so far:
{render_plan(state['current_plan'], query_meta['days'], cuisine_lookup, accom_lookup)}

=== HISTORY DIGEST ===
{digest}

=== QUERY CONSTRAINTS ===
{render_constraints(constraints)}

=== HARD CONSTRAINTS (evaluator will FAIL the plan if any violated) ===
{HARD_CONSTRAINTS_BLOCK}{accom_hint}{city_seq_hint}{cuisine_hint}

=== NEXT DECISION ===
Plan day {record['day']} {record['field']}

=== MY DECISION FOR THIS STEP ===
{expert_short}
Result: spent becomes ${expert_after:.0f}/${int(budget)} (${budget - expert_after:.0f} remaining).

=== OPTIONS CONSIDERED ===
{alts_block}

=== TASK ===
Write a CONCISE inner monologue (target 100–160 words) in the agent's own first-person voice. Identify the factors that actually drive this decision, reason through them, then arrive at the action and state the budget delta (spent before → spent after, remaining). Be specific (real names, real numbers). Do not restate the state; do not speculate about future days; do not hedge ("comfortable / workable").

Always-required checks (when applicable to this action type):
  - CUISINE-MATCH (REQUIRED for meal picks when the query specifies cuisine preference shown above — one sentence stating whether the chosen restaurant serves at least one preferred cuisine, OR explicitly acknowledging a forced mismatch when no available option satisfies the preference);
  - CITY-SEQUENCE engagement (REQUIRED for transportation picks — refer to the visited-cities list and the still-need count shown above, and explain how this destination fits the trip's required city count and return-to-origin constraint).

Pick from the situational checks the 1-2 that ACTUALLY decide this step:
  - WRONG-CITY (when an alt is in a city the agent cannot be in);
  - REPEAT (when an alt is already in the plan — meals/attractions only);
  - MIN-NIGHTS / consecutive-stay chain (accommodation picks only);
  - MODE-CHAIN (transportation — prior self-drive vs flight chain);
  - MANDATORY-FIELD (when "skip" is being weighed for a slot that must be filled).

Ignore checks that have no bearing here.

=== GUIDELINES ===
{GUIDELINES}
"""


def build_prompt_B(record, expert, query_meta, constraints,
                   cuisine_lookup, accom_lookup):
    state = record["state_before"]
    budget = record["budget"]
    spent_b = state["spent"]
    digest = build_history_digest(state, query_meta, accom_lookup)
    return f"""=== CURRENT PLANNING STATE ===
Query: {query_meta['query']}
Total Days: {query_meta['days']}, Initial Budget: ${int(budget)}
Spent so far: ${spent_b:.0f}, Remaining: ${budget - spent_b:.0f}

Plan so far:
{render_plan(state['current_plan'], query_meta['days'], cuisine_lookup, accom_lookup)}

=== HISTORY DIGEST ===
{digest}

=== QUERY CONSTRAINTS ===
{render_constraints(constraints)}

=== HARD CONSTRAINTS (evaluator will FAIL the plan if any violated) ===
{HARD_CONSTRAINTS_BLOCK}

=== NEXT DECISION ===
Plan day {record['day']} accommodation (the FINAL day of the trip)

=== MY DECISION FOR THIS STEP ===
SKIP accommodation for the last day.

=== ENVIRONMENT CONTEXT ===
The return transportation for day {record['day']} has already been planned. The trip ends today — the agent travels back to the origin city this evening. Booking accommodation on the final day would be wasteful (the agent does not sleep at the destination tonight).

=== TASK ===
Write a CONCISE inner monologue (50–90 words) in the agent's own voice, explaining in one or two sentences why no accommodation is needed tonight (return-trip already in place, sleeping at home tonight), then close. No restatement, no hedging.

=== GUIDELINES ===
{GUIDELINES}
"""


def build_prompt_C(record, expert, query_meta, constraints, cheapest_cost,
                   cuisine_lookup, accom_lookup):
    state = record["state_before"]
    budget = record["budget"]
    spent_b = state["spent"]
    remaining_b = budget - spent_b
    meal_name = record["field"]
    digest = build_history_digest(state, query_meta, accom_lookup)
    return f"""=== CURRENT PLANNING STATE ===
Query: {query_meta['query']}
Total Days: {query_meta['days']}, Initial Budget: ${int(budget)}
Spent so far: ${spent_b:.0f}, Remaining: ${remaining_b:.0f}

Plan so far:
{render_plan(state['current_plan'], query_meta['days'], cuisine_lookup, accom_lookup)}

=== HISTORY DIGEST ===
{digest}

=== QUERY CONSTRAINTS ===
{render_constraints(constraints)}

=== HARD CONSTRAINTS (evaluator will FAIL the plan if any violated) ===
{HARD_CONSTRAINTS_BLOCK}

=== NEXT DECISION ===
Plan day {record['day']} {meal_name}

=== MY DECISION FOR THIS STEP ===
SKIP {meal_name}.

=== BUDGET SITUATION ===
Remaining budget: ${remaining_b:.0f}
The cheapest available {meal_name} option at this city would cost ${cheapest_cost:.0f} (would put spent at ${spent_b + cheapest_cost:.0f}, exceeding the ${int(budget)} budget by ${spent_b + cheapest_cost - budget:.0f}).
Every {meal_name} option at this state exceeds the remaining budget — the agent has no choice but to skip.

=== TASK ===
Write a CONCISE inner monologue (50–90 words) in the agent's own voice showing the budget arithmetic that forces skipping (remaining $X, cheapest option $Y, would overshoot by $Z). One sentence noting this is a forced exception, not a general license to skip meals. No hedging, no restatement.

=== GUIDELINES ===
{GUIDELINES}
"""


def build_prompt_D(record, expert, query_meta, constraints,
                   cuisine_lookup, accom_lookup):
    state = record["state_before"]
    budget = record["budget"]
    spent_b = state["spent"]
    day_dict = state["current_plan"].get(f"day_{record['day']}", {})
    cc = day_dict.get("current_city", "the current city")
    digest = build_history_digest(state, query_meta, accom_lookup)
    return f"""=== CURRENT PLANNING STATE ===
Query: {query_meta['query']}
Total Days: {query_meta['days']}, Initial Budget: ${int(budget)}
Spent so far: ${spent_b:.0f}, Remaining: ${budget - spent_b:.0f}

Plan so far:
{render_plan(state['current_plan'], query_meta['days'], cuisine_lookup, accom_lookup)}

=== HISTORY DIGEST ===
{digest}

=== QUERY CONSTRAINTS ===
{render_constraints(constraints)}

=== HARD CONSTRAINTS (evaluator will FAIL the plan if any violated) ===
{HARD_CONSTRAINTS_BLOCK}

=== NEXT DECISION ===
Plan day {record['day']} transportation

=== MY DECISION FOR THIS STEP ===
SKIP transportation.

=== TRIP CONTEXT ===
The agent is staying in {cc} today. No city change is scheduled for day {record['day']} — looking at the planned route, the next inter-city travel happens on a later day. No flights, drives, or taxis exist on this date in the trip's planned route.

=== TASK ===
Write a CONCISE inner monologue (50–80 words) in the agent's own voice noting that this is a stay day in {cc} (no city change), so no transportation is needed. Briefly reference the accommodation chain or min-nights commitment if it's the reason for staying. No restatement, no hedging.

=== GUIDELINES ===
{GUIDELINES}
"""


def build_prompt_E(record, expert, query_meta, constraints,
                   cuisine_lookup, accom_lookup):
    state = record["state_before"]
    budget = record["budget"]
    spent_b = state["spent"]
    digest = build_history_digest(state, query_meta, accom_lookup)
    return f"""=== CURRENT PLANNING STATE ===
Query: {query_meta['query']}
Total Days: {query_meta['days']}, Initial Budget: ${int(budget)}
Final spent: ${spent_b:.0f}, Remaining: ${budget - spent_b:.0f}

Plan so far (every day filled or intentionally skipped):
{render_plan(state['current_plan'], query_meta['days'], cuisine_lookup, accom_lookup)}

=== HISTORY DIGEST ===
{digest}

=== QUERY CONSTRAINTS ===
{render_constraints(constraints)}

=== HARD CONSTRAINTS (evaluator will FAIL the plan if any violated) ===
{HARD_CONSTRAINTS_BLOCK}

=== NEXT DECISION ===
Submit the completed plan (the only remaining action).

=== MY DECISION FOR THIS STEP ===
COMPLETE the plan.

=== TASK ===
Write a CONCISE inner monologue (100–150 words) in the agent's own voice, doing a quick final review of the plan. Cover only what matters: that the route closes back to origin, no restaurant repeats, transportation mode is consistent, accommodations met their minimum nights, and final spend is under budget. One sentence per check, ending with the commit. No restatement of the full plan day by day, no hedging.

=== GUIDELINES ===
{GUIDELINES}
"""


# =========================================================================
# Post-hoc quality flags (METHOD/SKILL pitfalls)
# =========================================================================
def quality_flags(reflection: str) -> dict:
    text = reflection.lower()
    banned_hits = [v for v in BANNED_VOCAB if re.search(rf"\b{re.escape(v)}\b", text)]
    numbered = NUMBERED_LABELS.search(reflection)
    half = len(reflection) // 2
    doubled = (
        half > 50
        and reflection[:100] and len(reflection) > 200
        and reflection[:80].strip() in reflection[half:]
    )
    word_count = len(reflection.split())
    return {
        "banned_vocab": banned_hits,
        "has_numbered_labels": bool(numbered),
        "numbered_label_match": numbered.group(0) if numbered else None,
        "is_doubled": doubled,
        "word_count": word_count,
    }


# =========================================================================
# LLM call
# =========================================================================
def call_llm(client: OpenAI, prompt: str, model: str = "deepseek-v4-pro",
             temperature: float = 0.7, frequency_penalty: float = 0.3) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        extra_body={"thinking": {"type": "disabled"}},
    )
    msg = resp.choices[0].message.content
    usage = {"input_tokens": resp.usage.prompt_tokens,
             "output_tokens": resp.usage.completion_tokens}
    return msg, usage


# =========================================================================
# Cheapest-cost lookup for type C (budget-exhausted SKIP_MEAL)
# =========================================================================
def cheapest_meal_cost(record, query_meta, db_root: Path) -> float:
    """Find the cheapest meal cost the agent would face at this state.
    Use the city of the day's accommodation or current_city."""
    state = record["state_before"]
    day = record["day"]
    day_dict = state["current_plan"].get(f"day_{day}", {})
    accom = day_dict.get("accommodation", "")
    cc = day_dict.get("current_city", "")
    city = None
    if accom and accom not in ("-", "PENDING") and "," in accom:
        _, c = accom.rsplit(",", 1)
        city = norm_city(c)
    elif cc:
        m = re.match(r"from\s+(.+?)\s+to\s+(.+)", cc)
        if m:
            city = norm_city(m.group(2))
        else:
            city = norm_city(cc)
    if not city:
        return 999.0
    df = pd.read_csv(db_root / "restaurants" / "clean_restaurant_2022.csv").dropna()
    df = df[df["City"] == city]
    if df.empty:
        return 999.0
    return float(df["Average Cost"].min()) * query_meta["people_number"]


# =========================================================================
# Driver
# =========================================================================
def load_queries() -> dict:
    ds = load_dataset("osunlp/TravelPlanner", "train", split="train")
    out = {}
    for i in range(len(ds)):
        rec = ds[i]
        lc = ast.literal_eval(rec["local_constraint"]) if isinstance(rec["local_constraint"], str) else rec["local_constraint"]
        out[i] = {
            "query": rec["query"], "days": int(rec["days"]),
            "people_number": int(rec["people_number"]),
            "budget": int(rec["budget"]),
            "constraints": lc, "level": rec["level"],
            "org": rec["org"], "dest": rec["dest"],
            "visiting_city_number": int(rec["visiting_city_number"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run ~10 representative states, serial + verbose")
    ap.add_argument("--smoke-n", type=int, default=0,
                    help="larger smoke: N states (uses concurrency when N>10)")
    ap.add_argument("--max-workers", type=int, default=32)
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY not set")

    rows = [json.loads(l) for l in IWM_PATH.read_text().splitlines() if l.strip()]
    queries = load_queries()
    cuisines = load_cuisine_lookup(GLOBAL_DB)
    accoms = load_accom_lookup(GLOBAL_DB)
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com")

    # Categorize all states
    plan = []  # list of dicts with: {record, type, info, query_meta}
    for r in rows:
        qmeta = queries[r["traj_idx"]]
        ctype, info = categorize(r, qmeta)
        if ctype is None:
            continue
        plan.append({"record": r, "type": ctype, "info": info, "query_meta": qmeta})

    type_counts = Counter(p["type"] for p in plan)
    print(f"[plan] {len(plan)} states will get SR reflections")
    for t in "ABCDE":
        print(f"  type {t}: {type_counts[t]}")

    # EXPANDED smoke: deterministic stratified sample of N states
    if args.smoke_n > 0:
        import random
        rng = random.Random(42)
        a_pool = [p for p in plan if p["type"] == "A"]
        # Stratify by (level, field, days) so we get diverse coverage
        strata = defaultdict(list)
        for p in a_pool:
            q = p["query_meta"]; r = p["record"]
            strata[(q["level"], r["field"], q["days"])].append(p)
        # Round-robin pick from strata until target_a
        target_a = args.smoke_n - 10  # reserve 10 for B/C/D/E
        a_picks = []
        keys = list(strata.keys())
        rng.shuffle(keys)
        idx = 0
        while len(a_picks) < target_a:
            k = keys[idx % len(keys)]
            if strata[k]:
                a_picks.append(strata[k].pop(rng.randrange(len(strata[k]))))
            else:
                # remove exhausted stratum
                keys.pop(idx % len(keys))
                if not keys: break
                idx = idx % len(keys); continue
            idx += 1
        # B / C / D / E samples
        bcde = []
        for t, n in (("B", 4), ("C", 3), ("D", 2), ("E", 1)):
            cands = [p for p in plan if p["type"] == t]
            rng.shuffle(cands)
            bcde.extend(cands[:n])
        plan = a_picks + bcde
        print(f"[smoke-n] picked {len(plan)} states "
              f"(A={len(a_picks)}, B={sum(1 for p in plan if p['type']=='B')}, "
              f"C={sum(1 for p in plan if p['type']=='C')}, "
              f"D={sum(1 for p in plan if p['type']=='D')}, "
              f"E={sum(1 for p in plan if p['type']=='E')})")

    # SMOKE: hand-pick 10 representative states
    elif args.smoke:
        smoke_pick = []
        # Type A: pick 6 representative states across single-city / multi-city / fields
        a_candidates = [p for p in plan if p["type"] == "A"]
        smoke_pick.extend([
            p for p in a_candidates if p["record"]["traj_idx"] == 0 and p["record"]["day"] == 1 and p["record"]["field"] == "dinner"][:1])
        smoke_pick.extend([
            p for p in a_candidates if p["record"]["traj_idx"] == 0 and p["record"]["day"] == 3 and p["record"]["field"] == "transportation"][:1])
        smoke_pick.extend([
            p for p in a_candidates if p["record"]["traj_idx"] == 22 and p["record"]["field"] == "accommodation"][:1])
        smoke_pick.extend([
            p for p in a_candidates if p["record"]["traj_idx"] == 30 and p["record"]["field"] == "transportation"][:1])
        smoke_pick.extend([
            p for p in a_candidates if p["record"]["traj_idx"] == 17 and p["record"]["field"] in MEAL_FIELDS][:1])
        smoke_pick.extend([
            p for p in a_candidates if p["record"]["traj_idx"] == 41 and p["record"]["field"] in ("breakfast", "lunch", "dinner")][:1])
        # B / C / D / E one each
        for t in "BCDE":
            cands = [p for p in plan if p["type"] == t]
            if cands:
                smoke_pick.append(cands[0])
        # Cap at 10
        smoke_pick = smoke_pick[:10]
        plan = smoke_pick
        print(f"[smoke] picked {len(plan)} states for hand-inspection")

    # Build prompts
    records_with_prompt = []
    dropped_no_wrong = 0
    for p in plan:
        r = p["record"]; qm = p["query_meta"]
        if p["type"] == "A":
            picked_alts = select_alts(r, qm, cuisines, accoms, K=6)
            if not picked_alts:
                # No truly-wrong alts at this state — skip SR (would force the
                # LLM to fabricate preferences to differentiate equally-valid
                # options). This is v4 design (see _audit_wrong_alts.py).
                dropped_no_wrong += 1
                continue
            prompt = build_prompt_A(r, p["info"]["expert"], picked_alts, qm,
                                    qm["constraints"], cuisines, accoms)
            p["picked_alts"] = picked_alts
        elif p["type"] == "B":
            prompt = build_prompt_B(r, p["info"]["expert"], qm, qm["constraints"],
                                    cuisines, accoms)
        elif p["type"] == "C":
            cheapest = cheapest_meal_cost(r, qm, GLOBAL_DB)
            prompt = build_prompt_C(r, p["info"]["expert"], qm, qm["constraints"],
                                    cheapest, cuisines, accoms)
            p["cheapest"] = cheapest
        elif p["type"] == "D":
            prompt = build_prompt_D(r, p["info"]["expert"], qm, qm["constraints"],
                                    cuisines, accoms)
        elif p["type"] == "E":
            prompt = build_prompt_E(r, p["info"]["expert"], qm, qm["constraints"],
                                    cuisines, accoms)
        p["prompt"] = prompt
        records_with_prompt.append(p)

    if dropped_no_wrong:
        print(f"[v4-skip] {dropped_no_wrong} type-A states dropped (no truly-wrong alt available)")

    # LLM calls
    use_concurrent = (not args.smoke) or args.smoke_n > 10
    print(f"\n[llm] {'concurrent (workers=' + str(args.max_workers) + ')' if use_concurrent else 'serial'}...")
    t0 = time.time()

    def call_one(p):
        try:
            reflection, usage = call_llm(client, p["prompt"])
            return p, reflection, usage, None
        except Exception as e:
            return p, None, None, f"{type(e).__name__}: {e}"

    results = []
    if not use_concurrent:
        for p in records_with_prompt:
            results.append(call_one(p))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            for f in as_completed([ex.submit(call_one, p) for p in records_with_prompt]):
                results.append(f.result())

    wall = time.time() - t0

    # Render output
    out_records = []
    total_in = total_out = 0
    for p, reflection, usage, err in results:
        r = p["record"]
        rec_out = {
            "traj_idx": r["traj_idx"], "step_idx": r["step_idx"],
            "day": r["day"], "field": r["field"], "type": p["type"],
            "prompt": p["prompt"],
            "reflection": reflection, "error": err,
            "flags": quality_flags(reflection) if reflection else None,
            "usage": usage,
            "alts_picked": [
                {"action": pa["transition"]["action"], "tags": pa["tags"]}
                for pa in p.get("picked_alts", [])
            ] if p["type"] == "A" else None,
        }
        out_records.append(rec_out)
        if usage:
            total_in += usage["input_tokens"]
            total_out += usage["output_tokens"]

    suffix = "_smoke" if args.smoke else (f"_smoke{args.smoke_n}" if args.smoke_n else "")
    out_path = OUT_DIR / f"sr_rollout{suffix}.jsonl"
    out_path.write_text("\n".join(json.dumps(r, default=str) for r in out_records) + "\n")

    # Summary
    cost = (total_in * 0.27 + total_out * 1.10) / 1_000_000
    print(f"\n[done] {len(out_records)} reflections")
    print(f"[wall] {wall:.1f}s")
    print(f"[tokens] input={total_in:,}, output={total_out:,}, est cost=${cost:.4f}")
    n_err = sum(1 for r in out_records if r["error"])
    if n_err:
        print(f"[errors] {n_err}")
    # Flag aggregates
    n_banned = sum(1 for r in out_records if r.get("flags") and r["flags"]["banned_vocab"])
    n_numbered = sum(1 for r in out_records if r.get("flags") and r["flags"]["has_numbered_labels"])
    n_doubled = sum(1 for r in out_records if r.get("flags") and r["flags"]["is_doubled"])
    print(f"[flags] banned_vocab={n_banned}, numbered_labels={n_numbered}, doubled={n_doubled}")
    print(f"[out] {out_path}")

    if args.smoke:
        print("\n" + "=" * 78)
        print("SMOKE OUTPUTS — hand-inspect each reflection below")
        print("=" * 78)
        for i, rec in enumerate(out_records):
            print(f"\n--- [{i+1}/{len(out_records)}] type={rec['type']} traj={rec['traj_idx']} d{rec['day']} {rec['field']} ---")
            if rec.get("error"):
                print(f"ERROR: {rec['error']}")
                continue
            print(f"flags: {rec['flags']}")
            print(f"reflection ({rec['flags']['word_count']} words):")
            print(rec["reflection"])
            print()
    elif args.smoke_n > 0:
        # Compact summary table + dump flagged + sample for full read
        print("\n" + "=" * 90)
        print(f"SMOKE-N SUMMARY TABLE ({len(out_records)} reflections)")
        print("=" * 90)
        print(f"{'#':>3} {'type':<5} {'traj':>4} {'d':>2} {'field':<14} {'words':>5} {'level':<7} flags")
        for i, rec in enumerate(out_records):
            f = rec.get("flags") or {}
            flag_str = ""
            if f.get("banned_vocab"):
                flag_str += f"banned={f['banned_vocab']} "
            if f.get("has_numbered_labels"):
                flag_str += f"NUMBERED "
            if f.get("is_doubled"):
                flag_str += "DOUBLED "
            if rec.get("error"):
                flag_str += "ERR"
            # find level
            qm = queries.get(rec["traj_idx"], {})
            print(f"{i+1:>3} {rec['type']:<5} {rec['traj_idx']:>4} {rec['day']:>2} {rec['field']:<14} {f.get('word_count','?'):>5} {qm.get('level',''):<7} {flag_str}")
        # Print all FLAGGED in full
        flagged = [r for r in out_records if r.get("flags") and (r["flags"]["banned_vocab"] or r["flags"]["has_numbered_labels"] or r["flags"]["is_doubled"])]
        if flagged:
            print("\n" + "=" * 90)
            print(f"FLAGGED REFLECTIONS ({len(flagged)}) — full text")
            print("=" * 90)
            for rec in flagged:
                print(f"\n--- type={rec['type']} traj={rec['traj_idx']} d{rec['day']} {rec['field']} | flags={rec['flags']} ---")
                print(rec["reflection"])
        # Print 5 random non-flagged for variety
        import random
        rng2 = random.Random(7)
        clean = [r for r in out_records if r not in flagged and r.get("reflection")]
        rng2.shuffle(clean)
        print("\n" + "=" * 90)
        print(f"5 RANDOM CLEAN REFLECTIONS (for variety check)")
        print("=" * 90)
        for rec in clean[:5]:
            print(f"\n--- type={rec['type']} traj={rec['traj_idx']} d{rec['day']} {rec['field']} | {rec['flags']['word_count']} words ---")
            print(rec["reflection"])
        print(f"\n(full data in {out_path})")


if __name__ == "__main__":
    main()

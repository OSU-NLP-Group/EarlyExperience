"""Audit: per SR-scope state, count how many "truly wrong" alts are available.

A truly-wrong alt is one that can be REJECTED via a grounded rule, not via a
fabricated preference. This lets us decide which states are worth SR.

Computes per-alt the following wrong-tags from current data:
  repeats_restaurant       alt restaurant already used in plan-so-far
  repeats_attraction       alt attraction venue already visited
  cuisine_violate          meal alt cuisine doesn't satisfy query.cuisine
  transport_violate        transport alt violates query.transportation
  house_rule_violate       accom alt house_rules ban query.house_rule
  room_type_violate        accom alt room_type doesn't match query.room_type
  breaks_min_nights        accom alt at new place with remaining_days < min_nights
  breaks_mode_chain        transport alt conflicts with Self-driving lock-in
  mandatory_field_skip     SKIP alt on a stay-day mandatory field (breakfast/
                           lunch/dinner/attraction on stay days; accommodation
                           pre-final-day)

Reports: states with ≥1 wrong alt vs states with 0 (would be SR-skipped).
"""
from __future__ import annotations
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ENV_ROOT = Path(__file__).resolve().parents[1]
IWM_PATH = ENV_ROOT / "data" / "rollout" / "iwm_rollout_paper.jsonl"
GLOBAL_DB = ENV_ROOT / "global_db"

MEAL_FIELDS = ("breakfast", "lunch", "dinner")
_PAREN = re.compile(r"\([\w\s]+\)$")
def norm_city(s): return _PAREN.sub("", s.strip()).strip()


def load_cuisine_lookup(db_root: Path) -> dict:
    df = pd.read_csv(db_root / "restaurants" / "clean_restaurant_2022.csv").dropna()
    return {(str(r["Name"]).strip(), str(r["City"]).strip()): str(r["Cuisines"])
            for _, r in df.iterrows()}


def load_accom_lookup(db_root: Path) -> dict:
    df = pd.read_csv(db_root / "accommodations" / "clean_accommodations_2022.csv").dropna()
    out = {}
    for _, r in df.iterrows():
        key = (str(r["NAME"]).strip(), str(r["city"]).strip())
        out[key] = {
            "min_nights": int(r["minimum nights"]) if not pd.isna(r["minimum nights"]) else None,
            "room_type": str(r["room type"]).strip(),
            "house_rules": str(r["house_rules"]).strip() if not pd.isna(r["house_rules"]) else "",
        }
    return out


def classify_mode(value):
    if not value or value in ("-", "PENDING"):
        return None
    v = value.lower()
    if "flight" in v:
        return "Flight"
    if "self-driving" in v:
        return "Self-driving"
    if "taxi" in v:
        return "Taxi"
    return None


def used_modes_so_far(plan, day, days):
    """Set of transport modes used in days strictly before `day`."""
    modes = set()
    for d in range(1, day):
        t = plan.get(f"day_{d}", {}).get("transportation", "")
        m = classify_mode(t)
        if m:
            modes.add(m)
    return modes


def restaurants_used(plan, days):
    out = []
    for d in range(1, days + 1):
        day = plan.get(f"day_{d}", {})
        for meal in MEAL_FIELDS:
            v = day.get(meal, "")
            if v and v not in ("-", "PENDING", None):
                out.append(v)
    return out


def attractions_visited(plan, days):
    out = []
    for d in range(1, days + 1):
        day = plan.get(f"day_{d}", {})
        v = day.get("attraction", "")
        if v and v not in ("-", "PENDING", None):
            for venue in str(v).split(";"):
                venue = venue.strip()
                if venue:
                    out.append(venue)
    return out


def is_travel_day(plan, day):
    cc = plan.get(f"day_{day}", {}).get("current_city", "") or ""
    return "from" in cc and " to " in cc


def tag_alt_wrong(record, alt, query_meta, cuisine_lookup, accom_lookup) -> list[str]:
    """Return wrong-tags for this alt; empty list means alt is just 'valid but
    different'."""
    tags = []
    field = record["field"]
    day = record["day"]
    state = record["state_before"]
    plan = state["current_plan"]
    days = query_meta["days"]
    constraints = query_meta.get("constraints") or {}
    at = alt["action_type"]
    val = alt.get("value", "")

    # 1. mandatory_field_skip: SKIP on a stay-day mandatory field
    if at.startswith("SKIP_"):
        if field in MEAL_FIELDS and not is_travel_day(plan, day):
            tags.append("mandatory_field_skip")
        elif field == "attraction" and not is_travel_day(plan, day):
            tags.append("mandatory_field_skip")
        elif field == "accommodation" and day < days:
            tags.append("mandatory_field_skip")
        elif field == "transportation" and is_travel_day(plan, day):
            tags.append("mandatory_field_skip")

    # 2. repeats_restaurant
    if field in MEAL_FIELDS and not at.startswith("SKIP_"):
        used = restaurants_used(plan, days)
        if val in used:
            tags.append("repeats_restaurant")

    # 3. repeats_attraction
    if field == "attraction" and not at.startswith("SKIP_"):
        visited = attractions_visited(plan, days)
        if val in visited:
            tags.append("repeats_attraction")

    # 4. cuisine_violate (already computed in rollout_sr; replicate)
    if field in MEAL_FIELDS and constraints.get("cuisine") and not at.startswith("SKIP_"):
        if "," in val:
            name, city = val.rsplit(",", 1)
            cui = cuisine_lookup.get((name.strip(), norm_city(city)), "")
            if cui and not any(c in cui for c in constraints["cuisine"]):
                tags.append("cuisine_violate")

    # 5. transport_violate
    if field == "transportation" and constraints.get("transportation") and not at.startswith("SKIP_"):
        v = val.lower()
        pref = str(constraints["transportation"]).lower()
        if "no self-driving" in pref and "self-driving" in v:
            tags.append("transport_violate")
        if "no flight" in pref and "flight" in v:
            tags.append("transport_violate")

    # 6. house_rule_violate
    if field == "accommodation" and constraints.get("house rule") and not at.startswith("SKIP_"):
        if "," in val:
            name, city = val.rsplit(",", 1)
            info = accom_lookup.get((name.strip(), norm_city(city)), {})
            rules = info.get("house_rules", "")
            qhr = str(constraints["house rule"]).lower()
            ban_map = {
                "smoking": "No smoking", "parties": "No parties",
                "children under 10": "No children under 10",
                "visitors": "No visitors", "pets": "No pets",
            }
            ban_str = ban_map.get(qhr)
            if ban_str and ban_str in rules:
                tags.append("house_rule_violate")

    # 7. room_type_violate
    if field == "accommodation" and constraints.get("room type") and not at.startswith("SKIP_"):
        if "," in val:
            name, city = val.rsplit(",", 1)
            info = accom_lookup.get((name.strip(), norm_city(city)), {})
            rt = info.get("room_type", "")
            qrt = str(constraints["room type"]).lower()
            if qrt == "private room" and rt != "Private room":
                tags.append("room_type_violate")
            elif qrt == "shared room" and rt != "Shared room":
                tags.append("room_type_violate")
            elif qrt == "entire room" and rt != "Entire home/apt":
                tags.append("room_type_violate")
            elif qrt == "not shared room" and rt == "Shared room":
                tags.append("room_type_violate")

    # 8. breaks_min_nights — accom alt at NEW place with remaining_days < min_nights
    if field == "accommodation" and not at.startswith("SKIP_"):
        # If we pick this alt, we'd start a new accommodation run from `day`
        # ending at most at `days-1` (since last day no accom). So we have
        # remaining_days = days - day available for consecutive nights.
        # Exception: if the previous day's accom is the SAME as this alt, we extend.
        prev_accom = plan.get(f"day_{day - 1}", {}).get("accommodation", "")
        if val != prev_accom:
            remaining_for_accom = (days - 1) - day + 1   # days [day, day+1, ..., days-1]
            if remaining_for_accom < 1:
                remaining_for_accom = 1
            if "," in val:
                name, city = val.rsplit(",", 1)
                info = accom_lookup.get((name.strip(), norm_city(city)), {})
                mn = info.get("min_nights")
                if mn and mn > remaining_for_accom:
                    tags.append("breaks_min_nights")

    # 9. breaks_mode_chain
    if field == "transportation" and not at.startswith("SKIP_"):
        prev_modes = used_modes_so_far(plan, day, days)
        m = classify_mode(val)
        if m:
            # Self-driving lock-in: if previously used Self-driving, alt must be Self-driving
            if "Self-driving" in prev_modes and m != "Self-driving":
                tags.append("breaks_mode_chain")
            # And if previously used Flight/Taxi, alt can't introduce Self-driving
            if m == "Self-driving" and ("Flight" in prev_modes or "Taxi" in prev_modes):
                tags.append("breaks_mode_chain")
    return tags


def categorize_sr_scope(record, query_meta):
    """Return 'A', 'B', 'C', 'D', 'E', or None (skipped from SR)."""
    field = record["field"]
    if field == "attraction":
        return None
    expert = next(t["action"] for t in record["transitions"] if t["is_expert"])
    n_alts = sum(1 for t in record["transitions"] if not t["is_expert"])
    if expert["action_type"] == "COMPLETE_PLAN":
        return "E"
    if n_alts == 0:
        if expert["action_type"] == "SKIP_ACCOMMODATION" and record["day"] == query_meta["days"]:
            return "B"
        if expert["action_type"] == "SKIP_MEAL":
            return "C"
        if expert["action_type"] == "SKIP_TRANSPORTATION":
            return "D"
        return None
    return "A"


def main():
    iwm = [json.loads(l) for l in IWM_PATH.read_text().splitlines() if l.strip()]
    ds = load_dataset("osunlp/TravelPlanner", "train", split="train")
    queries = {}
    for i in range(len(ds)):
        rec = ds[i]
        lc = ast.literal_eval(rec["local_constraint"]) if isinstance(rec["local_constraint"], str) else rec["local_constraint"]
        queries[i] = {
            "days": int(rec["days"]),
            "people_number": int(rec["people_number"]),
            "constraints": lc,
            "org": rec["org"], "dest": rec["dest"],
            "visiting_city_number": int(rec["visiting_city_number"]),
        }
    cuisine_lookup = load_cuisine_lookup(GLOBAL_DB)
    accom_lookup = load_accom_lookup(GLOBAL_DB)

    n_total = 0
    n_skip_attraction = 0
    n_skip_forced_no_content = 0
    type_counts = Counter()
    states_with_wrong_alt = Counter()  # field -> count
    states_total_by_field = Counter()
    no_wrong_states = []
    wrong_alt_tag_counts = Counter()

    for r in iwm:
        n_total += 1
        qm = queries[r["traj_idx"]]
        ctype = categorize_sr_scope(r, qm)
        if ctype is None:
            if r["field"] == "attraction":
                n_skip_attraction += 1
            else:
                n_skip_forced_no_content += 1
            continue
        type_counts[ctype] += 1
        states_total_by_field[r["field"]] += 1

        # For B/C/D/E (forced single-action), there are no alts. Wrong-alt audit
        # is only meaningful for type A.
        if ctype != "A":
            continue

        wrong_count = 0
        for t in r["transitions"]:
            if t["is_expert"]:
                continue
            tags = tag_alt_wrong(r, t["action"], qm, cuisine_lookup, accom_lookup)
            if tags:
                wrong_count += 1
                for tag in tags:
                    wrong_alt_tag_counts[tag] += 1
        if wrong_count > 0:
            states_with_wrong_alt[r["field"]] += 1
        else:
            no_wrong_states.append((r["traj_idx"], r["day"], r["field"]))

    print(f"=== SR scope audit ===")
    print(f"total IWM states:                          {n_total}")
    print(f"skipped (attraction):                      {n_skip_attraction}")
    print(f"skipped (forced no-content):               {n_skip_forced_no_content}")
    print(f"SR scope (A+B+C+D+E):                      {sum(type_counts.values())}")
    print(f"  type A (with alts):                      {type_counts['A']}")
    print(f"  type B (SKIP_ACCOMMODATION last day):    {type_counts['B']}")
    print(f"  type C (SKIP_MEAL budget-exhausted):     {type_counts['C']}")
    print(f"  type D (SKIP_TRANSPORTATION stay-day):   {type_counts['D']}")
    print(f"  type E (COMPLETE_PLAN):                  {type_counts['E']}")
    print()

    print(f"=== Type-A wrong-alt availability ===")
    print(f"{'field':<16}{'A states':>10}{'with ≥1 wrong alt':>22}{'no wrong alt':>16}")
    n_A = type_counts["A"]
    total_with = 0
    total_no = 0
    for f in ("transportation", "breakfast", "lunch", "dinner", "accommodation"):
        n = states_total_by_field[f]
        w = states_with_wrong_alt[f]
        nw = n - w
        total_with += w; total_no += nw
        print(f"  {f:<14}{n:>10}{w:>22}{nw:>16}")
    print(f"  {'TOTAL':<14}{n_A:>10}{total_with:>22}{total_no:>16}")
    print()
    print(f"=== Wrong-alt tag distribution (across all wrong alts in type A) ===")
    for tag, n in wrong_alt_tag_counts.most_common():
        print(f"  {tag:<30}{n:>6}")
    print()
    print(f"=== Sample of 'no wrong alt' states (first 10) ===")
    for s in no_wrong_states[:10]:
        print(f"  traj{s[0]:>2} d{s[1]} {s[2]}")
    print()
    final_sr = total_with + type_counts["B"] + type_counts["C"] + type_counts["D"] + type_counts["E"]
    print(f"=== Projected SR scope under new design (drop type-A with 0 wrong alts) ===")
    print(f"  type A kept (≥1 wrong alt available): {total_with}")
    print(f"  type B + C + D + E:                   {type_counts['B'] + type_counts['C'] + type_counts['D'] + type_counts['E']}")
    print(f"  TOTAL SR states under new design:     {final_sr}  (vs current {sum(type_counts.values())})")
    print(f"  dropped (no grounded contrast):        {total_no}")


if __name__ == "__main__":
    main()

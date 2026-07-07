"""Admissible-action enumerator for TravelPlanner (Stage B).

At a given (day, field) state, enumerate ALL valid actions from the per-query
ref_info (Decision A: ref_info, not global DB — matches sole-planning inference
distribution). Each action is a fully-formed dict with computed cost, schema-
identical to replay_expert.derive_action:

    {"action_type", "day", "field", "value", "cost"}

City handling:
- ref_info entries store plain city names ("Boise"); gold plan values use a
  state suffix ("Boise(Idaho)"). We render alt values with the suffix-bearing
  city taken from the gold plan's own values for that day, so alts and the
  expert action share the same city rendering and dedup cleanly.
- "today city" (where meals/attractions/accommodation live for the day) is
  the city the gold plan's own non-empty values reference; falls back to the
  non-origin city of a "from X to Y" current_city.

Attraction is atomic (Decision: each alt is a SINGLE venue; the expert's
multi-venue string is recorded separately by replay). Attraction cost is
always 0 in TravelPlanner.
"""
from __future__ import annotations
import math
import re

PLAN_FIELDS = ("transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation")
MEAL_FIELDS = ("breakfast", "lunch", "dinner")

_PAREN_SUFFIX = re.compile(r"\([\w\s]+\)$")
_DRIVE_COST = re.compile(r"cost:\s*\$?([\d.]+)", re.I)


def norm_city(s: str) -> str:
    return _PAREN_SUFFIX.sub("", s.strip()).strip()


def _leg(day_dict: dict) -> tuple[str | None, str | None]:
    """(src, dst) from current_city 'from A to B'; (None, None) if not a travel day."""
    cc = day_dict.get("current_city", "") or ""
    m = re.match(r"from\s+(.+?)\s+to\s+(.+?)$", cc)
    if m:
        return norm_city(m.group(1)), norm_city(m.group(2))
    return None, None


def _value_city(val: str) -> str | None:
    """City (with suffix as rendered) from a 'Name, City' value; first venue if multi."""
    if not val or val == "-":
        return None
    first = val.split(";")[0]
    if "," in first:
        return first.rsplit(",", 1)[1].strip()
    return None


def _day_base_city(day_dict: dict, query_org: str) -> str | None:
    """Fallback city for a SKIPPED field: prefer the day's own SET values, then
    the non-origin city of a 'from X to Y' current_city."""
    for fld in ("accommodation", "dinner", "lunch", "breakfast", "attraction"):
        c = _value_city(day_dict.get(fld, "-"))
        if c:
            return c
    cc = day_dict.get("current_city", "") or ""
    m = re.match(r"from\s+(.+?)\s+to\s+(.+?)$", cc)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        return b if norm_city(a) == norm_city(query_org) else a
    if cc and cc != "-":
        return cc.strip()
    return None


def _field_city(field: str, day_dict: dict, query_org: str) -> str | None:
    """The city whose candidates are admissible for THIS (day, field). When the
    expert SET this field, use that value's own city (guarantees the expert
    action is in the admissible set, and matches the per-field city flips on
    travel days — breakfast in the FROM city, dinner in the TO city). When the
    expert SKIPPED it, fall back to the day's base city."""
    own = _value_city(day_dict.get(field, "-"))
    if own:
        return own
    return _day_base_city(day_dict, query_org)


def _ref_city_entries(ref_info: dict, prefix: str, today_norm: str) -> list[dict]:
    """ref_info list entries under a key like 'Restaurants in <city>' matching the city."""
    out = []
    for k, v in ref_info.items():
        if k.startswith(prefix) and isinstance(v, list):
            # key city is the text after the prefix; compare normalized
            key_city = norm_city(k[len(prefix):].strip())
            if key_city == today_norm:
                out.extend(v)
    return out


def enumerate_admissible(day_idx: int, field: str, day_dict: dict,
                         ref_info: dict, query: dict) -> list[dict]:
    """All valid actions at this (day, field). Includes SKIP. Includes the
    action that coincides with the expert (caller dedups if desired)."""
    people = int(query["people_number"])
    org = query.get("org", "")
    day_num = day_idx + 1
    actions: list[dict] = []

    if field == "transportation":
        src, dst = _leg(day_dict)
        if src and dst and src != dst:
            # Direction-aware route prefix. ref_info keys use plain city names;
            # substring matching would wrongly catch the reverse-direction key
            # (both contain both city names), so anchor on "from {src} to {dst}".
            flight_prefix = f"Flight from {src} to {dst} on"
            selfd_prefix = f"Self-driving from {src} to {dst}"
            taxi_prefix = f"Taxi from {src} to {dst}"
            for k, v in ref_info.items():
                if k.startswith(flight_prefix) and isinstance(v, list):
                    for fl in v:
                        value = (f"Flight Number: {fl['Flight Number']}, "
                                 f"from {fl['OriginCityName']} to {fl['DestCityName']}, "
                                 f"Departure Time: {fl['DepTime']}, Arrival Time: {fl['ArrTime']}")
                        actions.append({"action_type": "SET_TRANSPORTATION", "day": day_num,
                                        "field": "transportation", "value": value,
                                        "cost": float(fl["Price"]) * people})
                elif k.startswith(selfd_prefix) and isinstance(v, str) and "no valid information" not in v.lower():
                    mc = _DRIVE_COST.search(v)
                    if mc:
                        # canonical full value, capitalized to match gold rendering
                        value = "Self-driving, " + v.split(",", 1)[1].strip() if "," in v else v
                        actions.append({"action_type": "SET_TRANSPORTATION", "day": day_num,
                                        "field": "transportation", "value": value,
                                        "cost": float(mc.group(1)) * math.ceil(people / 5)})
                elif k.startswith(taxi_prefix) and isinstance(v, str) and "no valid information" not in v.lower():
                    mc = _DRIVE_COST.search(v)
                    if mc:
                        value = "Taxi, " + v.split(",", 1)[1].strip() if "," in v else v
                        actions.append({"action_type": "SET_TRANSPORTATION", "day": day_num,
                                        "field": "transportation", "value": value,
                                        "cost": float(mc.group(1)) * math.ceil(people / 4)})
        actions.append(_skip("transportation", day_num))
        return actions

    city_disp = _field_city(field, day_dict, org)
    city_norm = norm_city(city_disp) if city_disp else None

    if field in MEAL_FIELDS:
        if city_norm:
            for e in _ref_city_entries(ref_info, "Restaurants in", city_norm):
                actions.append({"action_type": f"SET_{field.upper()}", "day": day_num,
                                "field": field, "value": f"{e['Name']}, {city_disp}",
                                "cost": float(e["Average Cost"]) * people})
        actions.append(_skip(field, day_num))
        return actions

    if field == "attraction":
        if city_norm:
            for e in _ref_city_entries(ref_info, "Attractions in", city_norm):
                actions.append({"action_type": "SET_ATTRACTION", "day": day_num,
                                "field": "attraction", "value": f"{e['Name']}, {city_disp}",
                                "cost": 0.0})
        actions.append(_skip("attraction", day_num))
        return actions

    if field == "accommodation":
        if city_norm:
            for e in _ref_city_entries(ref_info, "Accommodations in", city_norm):
                occ = int(e["maximum occupancy"]) if e.get("maximum occupancy") else 1
                actions.append({"action_type": "SET_ACCOMMODATION", "day": day_num,
                                "field": "accommodation", "value": f"{e['NAME']}, {city_disp}",
                                "cost": float(e["price"]) * math.ceil(people / max(occ, 1))})
        actions.append(_skip("accommodation", day_num))
        return actions

    raise ValueError(f"unknown field {field}")


def _skip(field: str, day_num: int) -> dict:
    return {"action_type": f"SKIP_{field.upper()}", "day": day_num,
            "field": field, "value": "-", "cost": 0.0}


_FN_RE = re.compile(r"Flight Number:\s*(\S+?)(?:,|$)")
_FROMTO_RE = re.compile(r"from\s+(.+?)\s+to\s+(.+?)(?:,|$)")


def action_key(a: dict) -> tuple:
    """Format-robust identity for dedup between the expert action (gold raw
    string) and an enumerated alt (canonical rendering). Keys on the underlying
    entity, not the exact value string:
      - flight        → flight number
      - self-drive/taxi → (mode, norm src, norm dst)
      - meal/attr/accom → (name, norm city)
      - skip          → (type, day)
    """
    at = a["action_type"]
    day = a["day"]
    if at.startswith("SKIP_"):
        return (at, day)
    val = a["value"]
    if at == "SET_TRANSPORTATION":
        m = _FN_RE.search(val)
        if m:
            return ("FLIGHT", day, m.group(1))
        mode = "selfdrive" if "self-driving" in val.lower() else "taxi" if "taxi" in val.lower() else "?"
        mft = _FROMTO_RE.search(val)
        if mft:
            return (mode, day, norm_city(mft.group(1)), norm_city(mft.group(2)))
        return (mode, day, val)
    # meal / attraction / accommodation: name + normalized city
    if "," in val:
        name, city = val.rsplit(",", 1)
        return (at, day, name.strip(), norm_city(city))
    return (at, day, val)


# ---------------------------------------------------------------- unit check
if __name__ == "__main__":
    import ast
    import json
    from pathlib import Path
    from datasets import load_dataset

    ENV_ROOT = Path(__file__).resolve().parents[1]
    SUB = ENV_ROOT / "TravelPlanner"
    ds = load_dataset("osunlp/TravelPlanner", "train", split="train")
    refs = [json.loads(l) for l in (SUB / "database/train_ref_info.jsonl").read_text().splitlines() if l.strip()]

    # Correctness check across ALL 45 trajectories: for every SET expert action
    # in a non-attraction field, the expert's action must appear in the
    # admissible set (else the gym can't reproduce the expert / dedup breaks).
    miss = 0
    checked = 0
    for idx in range(len(ds)):
        rec = ds[idx]
        plan = [d for d in ast.literal_eval(rec["annotated_plan"])[1] if d]
        query = {"people_number": int(rec["people_number"]), "org": rec["org"], "days": int(rec["days"])}
        for d_idx in range(query["days"]):
            day = plan[d_idx] if d_idx < len(plan) else {}
            for field in PLAN_FIELDS:
                gold = day.get(field, "-")
                if gold in ("-", None, "") or field == "attraction":
                    continue  # attraction expert is multi-venue, not a single adm entry
                adm = enumerate_admissible(d_idx, field, day, refs[idx], query)
                adm_keys = {action_key(a) for a in adm}
                gold_action = {"action_type": f"SET_{field.upper()}", "day": d_idx + 1,
                               "field": field, "value": gold}
                checked += 1
                if action_key(gold_action) not in adm_keys:
                    miss += 1
                    if miss <= 15:
                        print(f"  MISS traj{idx} day{d_idx+1} {field}: gold={gold[:55]!r}")
    print(f"\n[gold-in-admissible] {checked - miss}/{checked} SET expert actions found in admissible "
          f"({miss} misses)")

    for idx in (0, 22):  # detailed dump: single-city + multi-city
        rec = ds[idx]
        plan = [d for d in ast.literal_eval(rec["annotated_plan"])[1] if d]
        query = {"people_number": int(rec["people_number"]), "org": rec["org"], "days": int(rec["days"])}
        print(f"\n{'='*60}\ntraj {idx}: {rec['org']} -> {rec['dest']}, {rec['days']}d, "
              f"{rec['people_number']}p, {rec['level']}\n{'='*60}")
        for d_idx in range(query["days"]):
            day = plan[d_idx] if d_idx < len(plan) else {}
            for field in PLAN_FIELDS:
                adm = enumerate_admissible(d_idx, field, day, refs[idx], query)
                gold = day.get(field, "-")
                gold_disp = (gold[:42] + "...") if gold and len(gold) > 42 else gold
                print(f"  day{d_idx+1:>1} {field:<14} |adm|={len(adm):>3}  gold={gold_disp}")

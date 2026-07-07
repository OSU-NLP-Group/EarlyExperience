"""Audit alt-coverage of v5 SR data.

For each state actually fed into SR generation (after categorize()), compute:
  - pre_select_tags: which wrong-tags exist in *any* alt for this state
  - shown_tags     : which wrong-tags are in the K=6 alts after select_alts()
  - missing_*      : pre_select tag present but not shown (truncated by K)
  - never_*        : tag class for which NO alt exists at all (e.g. wrong_city
                     because IWM filters by current city)

Also explicitly probes whether any meal/attraction alt is from a different
city than the expert chose (i.e. would the model ever see a wrong-city
example, even if untagged?).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT / "scripts"))

from rollout_sr import (categorize, tag_alt, select_alts, MEAL_FIELDS,
                        load_cuisine_lookup, load_accom_lookup, load_queries,
                        norm_city, _restaurants_used, _attractions_used,
                        GLOBAL_DB)

IWM_PATH = ENV_ROOT / "data" / "rollout" / "iwm_rollout_paper.jsonl"

def _alt_city(val):
    if not val or val == "-":
        return None
    if "," in val:
        _, city = val.rsplit(",", 1)
        return norm_city(city)
    return None

def _current_city_of_state(rec):
    """Best-effort: read from prior plan day or expert action."""
    plan = rec["state_before"]["current_plan"]
    day = rec["day"]
    # Look at previous day's accommodation city (where we slept = today's start)
    prev = plan.get(f"day_{day-1}", {})
    for f in ("dinner", "lunch", "breakfast", "accommodation", "attraction"):
        v = prev.get(f, "")
        if v and v != "-" and "," in v:
            return norm_city(v.rsplit(",",1)[1])
    # fallback: today's earlier filled field
    today = plan.get(f"day_{day}", {})
    for f in ("breakfast", "lunch", "dinner", "attraction", "accommodation"):
        v = today.get(f, "")
        if v and v != "-" and "," in v:
            return norm_city(v.rsplit(",",1)[1])
    return None

def main():
    rows = [json.loads(l) for l in IWM_PATH.read_text().splitlines() if l.strip()]
    queries = load_queries()
    cuisines = load_cuisine_lookup(GLOBAL_DB)
    accoms = load_accom_lookup(GLOBAL_DB)

    plan = []
    for r in rows:
        qmeta = queries[r["traj_idx"]]
        ctype, _ = categorize(r, qmeta)
        if ctype is None:
            continue
        plan.append((r, ctype, qmeta))

    type_total = Counter(t for _,t,_ in plan)
    print(f"[plan] {len(plan)} SR states (matches rollout_sr.py)")
    for t in "ABCDE":
        print(f"  type {t}: {type_total[t]}")
    print()

    # Per-state audit
    per_field_pre = defaultdict(Counter)    # field -> tag-class -> n states w/ that tag in pre-select
    per_field_shown = defaultdict(Counter)  # field -> tag-class -> n states w/ that tag actually shown
    per_field_n = Counter()                 # how many states of each field
    truncation_loss = Counter()             # tag-class -> n states where pre had it but shown lost it

    # Specific commonsense probes:
    wrong_city_alts_seen = Counter()        # field -> n states having ≥1 alt from a city different than expert's
    wrong_city_meals_states = 0
    wrong_city_attr_states = 0
    meal_total = 0
    attr_total = 0

    tag_classes = ["mandatory_field_skip","repeats_restaurant","repeats_attraction",
                   "wrong_city","cuisine_violate","transport_violate","house_rule_violate",
                   "room_type_violate","breaks_min_nights","breaks_mode_chain"]

    for r, ctype, qmeta in plan:
        field = r["field"]
        per_field_n[field] += 1
        alts = [t for t in r["transitions"] if not t["is_expert"]]
        expert = next(t for t in r["transitions"] if t["is_expert"])
        exp_city = _alt_city(expert["action"].get("value",""))

        # Pre-select: tag every alt
        pre_tag_set = set()
        for t in alts:
            tags = tag_alt(r, t["action"], qmeta, cuisines, accoms)
            for tag in tags:
                pre_tag_set.add(tag)
        # Shown: re-run select_alts (matches rollout)
        sel = select_alts(r, qmeta, cuisines, accoms, K=6)
        shown_tag_set = set()
        for s in sel:
            for tag in s["tags"]:
                shown_tag_set.add(tag)

        for tag in tag_classes:
            if tag in pre_tag_set:
                per_field_pre[field][tag] += 1
            if tag in shown_tag_set:
                per_field_shown[field][tag] += 1
            if tag in pre_tag_set and tag not in shown_tag_set:
                truncation_loss[tag] += 1

        # Cross-city probe
        if field in MEAL_FIELDS:
            meal_total += 1
            for t in alts:
                ac = _alt_city(t["action"].get("value",""))
                if ac and exp_city and ac != exp_city:
                    wrong_city_meals_states += 1
                    wrong_city_alts_seen[field] += 1
                    break
        elif field == "attraction":
            attr_total += 1
            for t in alts:
                ac = _alt_city(t["action"].get("value",""))
                if ac and exp_city and ac != exp_city:
                    wrong_city_attr_states += 1
                    wrong_city_alts_seen[field] += 1
                    break

    # Print
    print("="*88)
    print("PRE-SELECT vs SHOWN coverage  (n = states where ≥1 alt of that tag exists)")
    print("="*88)
    # Order fields meaningfully
    field_order = ["breakfast","lunch","dinner","attraction","accommodation","transportation"]
    fields_present = [f for f in field_order if per_field_n[f] > 0]
    for f in fields_present:
        n = per_field_n[f]
        print(f"\n--- field={f}  (n_states={n}) ---")
        print(f"  {'tag':25} | {'pre':>5} ({'%':>4})  -> {'shown':>5} ({'%':>4})  | gap")
        relevant = []
        if f in MEAL_FIELDS:
            relevant = ["mandatory_field_skip","repeats_restaurant","wrong_city","cuisine_violate"]
        elif f == "attraction":
            relevant = ["mandatory_field_skip","repeats_attraction","wrong_city"]
        elif f == "accommodation":
            relevant = ["mandatory_field_skip","breaks_min_nights","wrong_city","house_rule_violate","room_type_violate"]
        elif f == "transportation":
            relevant = ["mandatory_field_skip","breaks_mode_chain","transport_violate"]
        for tag in relevant:
            p = per_field_pre[f][tag]
            s = per_field_shown[f][tag]
            pp, sp = (100*p/n, 100*s/n) if n else (0,0)
            gap = p - s
            mark = " *** GAP" if gap > 0 else ""
            print(f"  {tag:25} | {p:5d} ({pp:4.1f})  -> {s:5d} ({sp:4.1f})  | {gap:3d}{mark}")

    print()
    print("="*88)
    print("CROSS-CITY ALT PROBE  (untagged class — does any alt come from a different city than expert?)")
    print("="*88)
    print(f"  meal states with ≥1 cross-city alt:       {wrong_city_meals_states}/{meal_total}  ({100*wrong_city_meals_states/max(1,meal_total):.1f}%)")
    print(f"  attraction states with ≥1 cross-city alt: {wrong_city_attr_states}/{attr_total}  ({100*wrong_city_attr_states/max(1,attr_total):.1f}%)")
    print(f"  by field: {dict(wrong_city_alts_seen)}")

    print()
    print("="*88)
    print("TRUNCATION LOSS (K=6 cap dropped this tag from prompt)")
    print("="*88)
    for tag, n in sorted(truncation_loss.items(), key=lambda x:-x[1]):
        if n: print(f"  {tag:25} : {n}")

if __name__ == "__main__":
    main()

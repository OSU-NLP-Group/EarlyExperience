"""IWM rollout for TravelPlanner (Stage B).

Walks each expert trajectory; at every (day, field) expert state s_i, enumerates
ALL valid actions, steps the deterministic state machine on each, and records
(s_i, action, s_{i+1}) triples. The expert's own action is recorded with
is_expert=True; alts that coincide with the expert (by action_key) are deduped.

Two dataset variants (kept separate for downstream A/B training comparison):
  --mode refinfo  (default, "A") transportation drawn only from ref_info
                  (gold-route flights + self-drive + taxi). ~29,767 transitions.
                  Matches sole-planning inference distribution.
  --mode anywhere ("C") transportation enumerates EVERY mode to EVERY reachable
                  destination from the global DB (paper's "exhaustive" ~70-78k).
                  Needs --db-root pointing at the global DB.
Non-transportation fields (meals/attraction/accommodation) are identical in both
variants (ref_info).

No LLM. Pure deterministic local rollout. Output grouped one line per state.

Usage:
  python rollout_iwm.py                              # A, all 45 traj
  python rollout_iwm.py --limit 2                    # A smoke
  python rollout_iwm.py --mode anywhere --db-root ../global_db          # C
  python rollout_iwm.py --mode anywhere --db-root ../global_db --limit 2  # C smoke
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from replay_expert import (TPState, step, state_snapshot, derive_action,
                           hf_record_to_query, hf_record_to_plan)
from admissible import enumerate_admissible, action_key, PLAN_FIELDS, _leg, norm_city

ENV_ROOT = Path(__file__).resolve().parents[1]
SUB = ENV_ROOT / "TravelPlanner"
DATA_DIR = ENV_ROOT / "data"


def rollout_one(traj_idx: int, rec: dict, ref_info: dict,
                mode: str = "refinfo", gdb: dict | None = None) -> tuple[list[dict], dict]:
    """Return (state-grouped IWM records, query) for one trajectory."""
    query = hf_record_to_query(rec)
    plan = hf_record_to_plan(rec)
    s = TPState.init(query)
    records = []

    for d_idx in range(query["days"]):
        day_dict = plan[d_idx] if d_idx < len(plan) else {}
        for field in PLAN_FIELDS:
            gold_val = day_dict.get(field, "-")
            expert_action = derive_action(d_idx, field, gold_val, ref_info, query["people_number"])
            expert_key = action_key(expert_action)

            # expert transition
            s_next_expert = step(s, expert_action)
            transitions = [{"is_expert": True, "action": expert_action,
                            "spent_after": round(s_next_expert.spent, 2)}]

            # candidate alternatives
            if mode == "anywhere" and field == "transportation":
                from transport_anywhere import enumerate_transportation_anywhere
                cands = enumerate_transportation_anywhere(d_idx, day_dict, query, gdb)
            else:
                cands = enumerate_admissible(d_idx, field, day_dict, ref_info, query)

            for alt in cands:
                if action_key(alt) == expert_key:
                    continue
                s_next_alt = step(s, alt)
                transitions.append({"is_expert": False, "action": alt,
                                    "spent_after": round(s_next_alt.spent, 2)})

            records.append({
                "traj_idx": traj_idx,
                "step_idx": s.cursor,
                "day": d_idx + 1,
                "field": field,
                "state_before": state_snapshot(s),     # {spent, cursor, done, plan}
                "budget": query["budget"],
                "n_transitions": len(transitions),
                "transitions": transitions,
            })
            # advance the real trajectory with the expert action
            s = s_next_expert

    return records, query


def collect_needed_routes(ds, n: int) -> tuple[set, set]:
    """Pre-scan trajectories for the (origin, date) flight pairs and origin
    cities needed, so the global-DB loader can filter cheaply."""
    needed_pairs, needed_origins = set(), set()
    for i in range(n):
        query = hf_record_to_query(ds[i])
        plan = hf_record_to_plan(ds[i])
        dates = query["date"]
        for d_idx in range(query["days"]):
            day_dict = plan[d_idx] if d_idx < len(plan) else {}
            src, dst = _leg(day_dict)
            if src and dst and src != dst:
                date = dates[d_idx] if d_idx < len(dates) else None
                needed_pairs.add((src, date))
                needed_origins.add(src)
    return needed_pairs, needed_origins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N trajectories (0 = all)")
    ap.add_argument("--mode", choices=["refinfo", "anywhere"], default="refinfo")
    ap.add_argument("--db-root", type=str, default=None, help="global DB root (required for --mode anywhere)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ds = load_dataset("osunlp/TravelPlanner", "train", split="train")
    refs = [json.loads(l) for l in (SUB / "database/train_ref_info.jsonl").read_text().splitlines() if l.strip()]
    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))

    gdb = None
    if args.mode == "anywhere":
        if not args.db_root:
            ap.error("--mode anywhere requires --db-root")
        from transport_anywhere import load_global_db
        needed_pairs, needed_origins = collect_needed_routes(ds, n)
        print(f"[gdb] loading global DB for {len(needed_origins)} origins / "
              f"{len(needed_pairs)} (origin,date) pairs ...")
        gdb = load_global_db(Path(args.db_root), needed_pairs, needed_origins)
        print(f"[gdb] flights index: {sum(len(v) for v in gdb['flights'].values()):,} flights; "
              f"distance index: {sum(len(v) for v in gdb['distance'].values()):,} reachable pairs")

    out_dir = DATA_DIR / "rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "refinfo" if args.mode == "refinfo" else "anywhere"
    smoke = "_smoke" if args.limit else ""
    out_path = Path(args.out) if args.out else (out_dir / f"iwm_rollout_{suffix}{smoke}.jsonl")

    all_records = []
    field_trans = Counter()
    field_states = Counter()
    n_expert = n_alt = 0
    skip_alt_outcomes = Counter()  # over/under budget on alt transitions

    for i in range(n):
        recs, query = rollout_one(i, ds[i], refs[i], mode=args.mode, gdb=gdb)
        all_records.extend(recs)
        for r in recs:
            field_states[r["field"]] += 1
            for t in r["transitions"]:
                field_trans[r["field"]] += 1
                if t["is_expert"]:
                    n_expert += 1
                else:
                    n_alt += 1
                    over = t["spent_after"] > r["budget"]
                    skip_alt_outcomes["over_budget" if over else "within_budget"] += 1

    out_path.write_text("\n".join(json.dumps(r, default=str) for r in all_records) + "\n")

    total = n_expert + n_alt
    print(f"[iwm] {n} trajectories → {len(all_records)} states, {total:,} transitions")
    print(f"      expert={n_expert:,}  alt={n_alt:,}")
    print(f"      alt budget outcome: within={skip_alt_outcomes['within_budget']:,} "
          f"over={skip_alt_outcomes['over_budget']:,}")
    print(f"[by field] transitions:")
    for f in PLAN_FIELDS:
        print(f"   {f:<16} states={field_states[f]:>4}  transitions={field_trans[f]:>7,}  "
              f"avg/state={field_trans[f]/max(field_states[f],1):>5.1f}")
    print(f"[out] {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

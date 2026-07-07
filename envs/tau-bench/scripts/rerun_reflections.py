"""Second-pass re-run for any sr_rollout.jsonl record that tripped any quality flag.

Reads sr_rollout.jsonl, finds every record with ANY of {is_doubled, has_banned_vocab,
has_numbered_label, exceeded_soft_cap}, re-issues the same prompt at the same
temperature (relying on sampling variation to produce a different output), and
writes attempt=2 records to sr_rerun.jsonl. Original sr_rollout.jsonl is untouched.

The SFT build script later picks between attempt 1 and attempt 2 per
(task_id, trial, obs_idx) — preferring whichever has fewer/no flags.

Run:
    PYTHONNOUSERSITE=1 DEEPSEEK_API_KEY=... \\
        conda run -n tau-bench-ee --no-capture-output python \\
        envs/tau-bench/scripts/rerun_reflections.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from propose_reflections import build_system_prompt, build_user_prompt, detect_flags

D_EXPERT_DEFAULT = "envs/tau-bench/data/rollout/D_expert.jsonl"
SR_IN_DEFAULT = "envs/tau-bench/data/rollout/sr_rollout.jsonl"
SR_OUT_DEFAULT = "envs/tau-bench/data/rollout/sr_rerun.jsonl"

FLAG_KEYS = ("is_doubled", "has_banned_vocab", "has_numbered_label", "exceeded_soft_cap")


def is_flagged(flags: dict) -> bool:
    return any(flags.get(k) for k in FLAG_KEYS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr-in", default=SR_IN_DEFAULT)
    ap.add_argument("--d-expert", default=D_EXPERT_DEFAULT)
    ap.add_argument("--out", default=SR_OUT_DEFAULT)
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--frequency-penalty", type=float, default=0.3)
    ap.add_argument("--max-workers", type=int, default=128)
    ap.add_argument("--count-only", action="store_true",
                    help="just print how many would be re-run, do not call API")
    args = ap.parse_args()

    sr_recs = [json.loads(l) for l in open(args.sr_in)]
    flagged = [r for r in sr_recs if "error" not in r and is_flagged(r["flags"])]
    print(f"sr_rollout: {len(sr_recs)} total, {len(flagged)} flagged for rerun")
    # break down by flag
    by_flag = {k: sum(1 for r in flagged if r["flags"].get(k)) for k in FLAG_KEYS}
    print(f"  by flag (overlap allowed): {by_flag}")

    if args.count_only:
        return

    # Index D_expert for traj reconstruction
    d_traj = {}
    with open(args.d_expert) as f:
        for line in f:
            r = json.loads(line)
            d_traj[(r["task_id"], r["trial"])] = r["traj"]

    sys_prompt = build_system_prompt()
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "w")
    write_lock, stats_lock = Lock(), Lock()
    stats = {"calls": 0, "errors": 0, "in": 0, "out": 0, "cached": 0,
             "now_clean": 0, "still_flagged": 0, "wc_total": 0,
             **{k: 0 for k in FLAG_KEYS}}

    def work(orig):
        traj = d_traj.get((orig["task_id"], orig["trial"]))
        if traj is None:
            return {"task_id": orig["task_id"], "trial": orig["trial"], "obs_idx": orig["obs_idx"],
                    "error": "traj not found in D_expert", "attempt": 2, "model": args.model}
        obs_msgs = traj[:orig["obs_idx"]]
        up = build_user_prompt(obs_msgs, orig["expert_kind"],
                               orig["expert_action"], orig["expert_next_obs"],
                               orig["picked_alts"])
        try:
            res = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": up}],
                temperature=args.temperature,
                frequency_penalty=args.frequency_penalty,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as e:
            return {"task_id": orig["task_id"], "trial": orig["trial"], "obs_idx": orig["obs_idx"],
                    "error": str(e), "attempt": 2, "model": args.model}
        raw = res.choices[0].message.content or ""
        u = res.usage
        cached = (u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0) or 0
        new_flags = detect_flags(raw)
        return {
            "task_id": orig["task_id"], "trial": orig["trial"], "obs_idx": orig["obs_idx"],
            "expert_kind": orig["expert_kind"],
            "expert_action": orig["expert_action"], "expert_next_obs": orig["expert_next_obs"],
            "picked_alts": orig["picked_alts"],
            "alt_selection_diagnostic": orig["alt_selection_diagnostic"],
            "reflection_raw": raw, "flags": new_flags,
            "model": args.model, "attempt": 2,
            "prev_flags": orig["flags"],
            "usage": {"in": u.prompt_tokens, "out": u.completion_tokens, "cached": cached},
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(work, r) for r in flagged]
        for i, fut in enumerate(as_completed(futures)):
            rec = fut.result()
            with write_lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            with stats_lock:
                stats["calls"] += 1
                if "error" in rec:
                    stats["errors"] += 1
                else:
                    stats["in"] += rec["usage"]["in"]
                    stats["out"] += rec["usage"]["out"]
                    stats["cached"] += rec["usage"]["cached"]
                    f = rec["flags"]
                    for k in FLAG_KEYS:
                        if f.get(k):
                            stats[k] += 1
                    stats["wc_total"] += f.get("word_count", 0)
                    if is_flagged(f):
                        stats["still_flagged"] += 1
                    else:
                        stats["now_clean"] += 1
            if (i + 1) % 25 == 0 or (i + 1) == len(flagged):
                dt = time.time() - t0
                print(f"  [{i+1:4d}/{len(flagged)}] {(i+1)/max(dt,1e-6):5.2f} req/s "
                      f"| clean={stats['now_clean']} still_flagged={stats['still_flagged']} err={stats['errors']}")
    fh.close()
    ok = stats["calls"] - stats["errors"]
    print()
    print(f"DONE in {(time.time()-t0)/60:.2f} min")
    print(f"  calls / errors  : {stats['calls']} / {stats['errors']}")
    print(f"  now clean       : {stats['now_clean']}  ({stats['now_clean']/max(ok,1)*100:.1f}% of rerun)")
    print(f"  still flagged   : {stats['still_flagged']}")
    print(f"  flag breakdown (attempt 2):")
    for k in FLAG_KEYS:
        print(f"    {k:22s} {stats[k]}")
    print(f"  tokens in/cached/out: {stats['in']:,} / {stats['cached']:,} / {stats['out']:,}")
    print(f"  output          : {args.out}")


if __name__ == "__main__":
    main()

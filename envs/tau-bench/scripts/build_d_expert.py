"""Build expert_rollouts_raw.jsonl + D_expert.jsonl from tau-bench's raw run.py output.

Source: a single big JSON written by tau-bench's run.py over 500 train tasks x 4 trials
(2000 rollouts). Each rollout is the result of running the tool-calling agent
(deepseek-v4-flash) against the retail env with deepseek-v4-flash as user simulator,
temperature=1.0.

Output:
  1. expert_rollouts_raw.jsonl - one rollout per line, full traj preserved (audit trail
     for everything: replay, IWM probing, SR reflection generation, expert SFT render).
  2. D_expert.jsonl            - paper-faithful filter (paper SS B.4):
        per task, keep at most one trial with reward=1, random pick if multiple;
        drop the task entirely if 0 successes. Random pick is seeded for reproducibility.

The submodule is not touched. This is a pure-Python post-process.

Run:
    PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output python \\
        envs/tau-bench/scripts/build_d_expert.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

SRC_DEFAULT = (
    "envs/tau-bench/data/rollout/"
    "tool-calling-deepseek-v4-flash-1.0_range_0-500_user-deepseek-v4-flash-llm_*.json"
)
RAW_OUT_DEFAULT = "envs/tau-bench/data/rollout/expert_rollouts_raw.jsonl"
DEXP_OUT_DEFAULT = "envs/tau-bench/data/rollout/D_expert.jsonl"


def count_assistant_turns(traj: list[dict]) -> tuple[int, int]:
    """Return (tool_call_turns, respond_turns) for an assistant trajectory."""
    tool = sum(1 for m in traj if m["role"] == "assistant" and m.get("tool_calls"))
    resp = sum(1 for m in traj if m["role"] == "assistant" and not m.get("tool_calls"))
    return tool, resp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT, help="glob for raw run.py JSON")
    ap.add_argument("--raw-out", default=RAW_OUT_DEFAULT)
    ap.add_argument("--dexp-out", default=DEXP_OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=10, help="random.seed for tiebreak pick (paper-faithful: 10)")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.src))
    if not matches:
        raise SystemExit(f"no source matches {args.src!r}")
    src = matches[-1]
    print(f"src   : {src}")

    data = json.load(open(src))
    print(f"raw rollouts: {len(data)}")

    # 1) raw stream-write
    Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.raw_out, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    print(f"raw -> {args.raw_out}  ({Path(args.raw_out).stat().st_size/1e6:.1f} MB)")

    # 2) D_expert filter
    by_task: dict[int, list[dict]] = collections.defaultdict(list)
    for r in data:
        by_task[r["task_id"]].append(r)

    rng = random.Random(args.seed)
    d_expert = []
    kept_per_task = collections.Counter()
    dropped_tasks: list[int] = []
    for tid in sorted(by_task):
        successes = [r for r in by_task[tid] if r["reward"] > 0.99]
        if not successes:
            dropped_tasks.append(tid)
            continue
        chosen = rng.choice(successes)
        d_expert.append(chosen)
        kept_per_task[tid] = len(successes)

    with open(args.dexp_out, "w") as f:
        for r in d_expert:
            f.write(json.dumps(r) + "\n")
    print(f"D_expert -> {args.dexp_out}  ({Path(args.dexp_out).stat().st_size/1e6:.1f} MB)")

    # 3) summary
    tool_total, resp_total = 0, 0
    msg_counts = []
    for r in d_expert:
        t, p = count_assistant_turns(r["traj"])
        tool_total += t
        resp_total += p
        msg_counts.append(len(r["traj"]))

    print()
    print("=" * 72)
    print("D_expert summary")
    print("=" * 72)
    print(f"  tasks total              : 500")
    print(f"  tasks dropped (0 hits)   : {len(dropped_tasks)}")
    print(f"  tasks kept in D_expert   : {len(d_expert)}  ({len(d_expert)/500*100:.1f}%)")
    print(f"  paper baseline (B.4)     : 452/495 = 91.3%")
    print()
    print(f"  (obs, action) pairs total: {tool_total + resp_total}")
    print(f"    tool-call (obs)        : {tool_total}")
    print(f"    respond (obs)          : {resp_total}")
    print(f"  paper baseline (B.4)     : 5,239 pairs")
    print()
    if d_expert:
        print(f"  messages/traj min/median/max: {min(msg_counts)} / "
              f"{sorted(msg_counts)[len(msg_counts)//2]} / {max(msg_counts)}")
    print()
    print(f"  successes-per-kept-task distribution: {dict(collections.Counter(kept_per_task.values()))}")
    print(f"  (e.g. {{4: N}} means N tasks had all 4 trials succeed and we picked 1 of 4)")
    print()
    if dropped_tasks:
        print(f"  first 20 dropped task_ids: {dropped_tasks[:20]}{' ...' if len(dropped_tasks) > 20 else ''}")


if __name__ == "__main__":
    main()

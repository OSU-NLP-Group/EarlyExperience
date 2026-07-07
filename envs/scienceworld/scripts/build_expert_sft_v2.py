"""Build expert_sft.v2.jsonl — multi-turn, eval-aligned, truncated at completion.

Changes vs v1:
  - Instruction moved from a `system` role to user-0; the "OK..." ack restored as
    assistant-0 (matches eval: no system message in the messages list — the Qwen
    template injects the default system; conversation_start[0]/[1] are user/assistant).
  - Instruction text = the eval client's conversation_start[0] (includes `examine OBJ`),
    so user-0 matches what the policy sees at eval time exactly.
  - Task line: NO "Task Description:" prefix (= getTaskDescription form, what the eval
    server returns). Derived by stripping the prefix from info.taskDesc.
  - Trajectory TRUNCATED at the goal-completing action: trailing post-completion turns
    (pre-action score already == 100 — padding `wait`/`look`) are dropped, so the
    trajectory ends when the task is solved (teaches "stop when done").
  - Ends on an assistant turn (the trailing env observation is dropped).

Loss: all assistant turns (full imitation).

Run:
    python envs/scienceworld/scripts/build_expert_sft_v2.py [--limit N] [--out PATH]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

REPLAY = "envs/scienceworld/data/replay/replay_full.jsonl"
ROLLOUT = "envs/scienceworld/data/rollout/iwm_rollout.jsonl"
CONV = "envs/scienceworld/data/conversation_start_eval.json"
OUT = "envs/scienceworld/data/sft/expert_sft.v2.jsonl"
TD_PREFIX = "Task Description:\n"


def truncate_at_completion(steps):
    """Return n = number of leading turns to keep: drop turns whose pre-action
    score is already == 100 (post-completion padding)."""
    n = len(steps)
    for i in range(1, len(steps)):
        if (steps[i - 1].get("score") or 0) >= 100:
            n = i
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    conv = json.load(open(CONV))
    INSTR, ACK = conv["instruction"], conv["ack"]

    initial_obs = {}
    for line in open(ROLLOUT):
        r = json.loads(line)
        if r["kind"] == "initial":
            initial_obs[r["item_id"]] = r["initial_obs"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_traj = 0
    n_dropped_turns = 0
    with open(out, "w") as fo:
        for line in open(REPLAY):
            rec = json.loads(line)
            if not (rec.get("final_done") and rec.get("final_score") == 100):
                continue
            iid = rec["item_id"]
            if iid not in initial_obs:
                continue
            steps = rec["replay_steps"]
            if not steps:
                continue
            keep = truncate_at_completion(steps)
            n_dropped_turns += len(steps) - keep
            thoughts = rec["agenttraj_thoughts"]
            actions = rec["agenttraj_actions"]
            observations = [s["observation"] for s in steps]
            td = steps[0]["info"].get("taskDesc", "")
            if td.startswith(TD_PREFIX):
                td = td[len(TD_PREFIX):]   # no-prefix task line (eval form)

            msgs = [
                {"role": "user", "content": INSTR},
                {"role": "assistant", "content": ACK},
                {"role": "user", "content": f"{td.rstrip()}\n{initial_obs[iid].rstrip()}"},
            ]
            for i in range(keep):
                th = thoughts[i].strip() if i < len(thoughts) and thoughts[i] else ""
                act = actions[i].strip() if i < len(actions) else ""
                asst = f"Thought:\n{th}\n\nAction:\n{act}" if th else f"Action:\n{act}"
                msgs.append({"role": "assistant", "content": asst})
                if i < keep - 1:   # drop trailing obs -> end on assistant
                    msgs.append({"role": "user", "content": observations[i].rstrip()})
            fo.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n_traj += 1
            if args.limit and n_traj >= args.limit:
                break
    print(f"trajectories written: {n_traj}  post-completion turns dropped: {n_dropped_turns}")
    print(f"output: {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

"""Build iwm_sft.v2.jsonl — world-model format (eval-isolated).

Changes vs v1:
  - system  : WORLD-MODEL role (not the agent prompt). Keeps the action list
              (from the eval-aligned conversation_start) for action grammar.
  - assistant: "Observation:\\n<next_state>"  (delimited, not a bare obs).
  - user history: Action / Observation pairs only, NO Thought lines.
              Observations in history are labelled "Observation:".
  - Single "Task Description:\\n" prefix (from info.taskDesc as-is; the v1
    double-prefix bug is gone — we no longer prepend on top of it).
  - NOT filtered: post-completion and "already" no-op triples are kept — they
    are valid world-model dynamics ("wait -> nothing changes", "open already-open
    door -> already open").

Single-turn (system, user, assistant), loss on the assistant (next-state).

Run:
    python envs/scienceworld/scripts/build_iwm_sft_v2.py [--limit N] [--out PATH]
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

REPLAY = "envs/scienceworld/data/replay/replay_full.jsonl"
ROLLOUT = "envs/scienceworld/data/rollout/iwm_rollout.jsonl"
CONV = "envs/scienceworld/data/conversation_start_eval.json"
OUT = "envs/scienceworld/data/sft/iwm_sft.v2.jsonl"

TD_PREFIX = "Task Description:\n"


def wm_system(instruction: str) -> str:
    """World-model system prompt: WM role + the action list from the eval instruction."""
    al = re.search(r"(\[\{.*\}\])", instruction, re.S).group(1)
    return ("You are a world model for ScienceWorld. Given the interaction history and a "
            "proposed action, predict the single resulting observation the environment "
            "returns. Here are the actions the agent may take: " + al)


def history_block(task_desc, initial_obs, actions, observations, up_to):
    """Action/Observation history (NO thoughts) up to step `up_to`."""
    pieces = [task_desc.rstrip(), initial_obs.rstrip()]
    for j in range(up_to):
        pieces.append("")
        pieces.append("Action:")
        pieces.append(actions[j].strip() if j < len(actions) else "")
        pieces.append("Observation:")
        pieces.append(observations[j].rstrip() if j < len(observations) else "")
    return "\n".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    conv = json.load(open(CONV))
    SYSTEM = wm_system(conv["instruction"])

    initial_obs = {}
    alts = defaultdict(list)
    for line in open(ROLLOUT):
        r = json.loads(line)
        if r["kind"] == "initial":
            initial_obs[r["item_id"]] = r["initial_obs"]
        elif r["kind"] == "alt":
            alts[(r["item_id"], r["step"])].append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_exp = n_alt = n_traj = 0
    with open(out, "w") as fo:
        for line in open(REPLAY):
            rec = json.loads(line)
            iid = rec["item_id"]
            if iid not in initial_obs:
                continue
            steps = rec["replay_steps"]
            if not steps:
                continue
            io = initial_obs[iid]
            actions = rec["agenttraj_actions"]
            observations = [s["observation"] for s in steps]
            td = steps[0]["info"].get("taskDesc", "")
            if not td.startswith(TD_PREFIX):   # ensure single prefix
                td = TD_PREFIX + td
            n_traj += 1
            for i in range(len(steps)):
                hist = history_block(td, io, actions, observations, i)
                # expert triple
                u = f"{hist}\n\nAction:\n{actions[i].strip()}"
                fo.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": f"Observation:\n{observations[i].rstrip()}"},
                ]}, ensure_ascii=False) + "\n")
                n_exp += 1
                # alternative triples
                for a in alts.get((iid, i), []):
                    ua = f"{hist}\n\nAction:\n{a['action'].strip()}"
                    fo.write(json.dumps({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": ua},
                        {"role": "assistant", "content": f"Observation:\n{a['next_state'].rstrip()}"},
                    ]}, ensure_ascii=False) + "\n")
                    n_alt += 1
            if args.limit and n_traj >= args.limit:
                break
    print(f"trajectories: {n_traj}  expert: {n_exp}  alt: {n_alt}  total: {n_exp+n_alt}")
    print(f"output: {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

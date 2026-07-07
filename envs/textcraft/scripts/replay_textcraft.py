"""
Replay AgentTraj-L textcraft expert trajectories against the running
agentenv-textcraft HTTP server, and record per-trajectory outcomes.

Mapping note: AgentTraj-L's `item_id "textcraft_N"` does NOT correspond to
the env's `data_idx=N` (verified empirically). We instead map each recorded
trajectory to a `data_idx` whose env-reset produces the same goal item.

Caveat: the env's commands list (recipes + distractors) differs from the
recorded one in distractor positions only — the goal-path recipes are
preserved. Expert craft actions should still execute as long as the expert
never relies on a distractor recipe (they shouldn't, by construction).

Usage:
    # 1. Start server in another terminal:
    #    conda run -n agentenv-textcraft textcraft --host 127.0.0.1 --port 36011
    # 2. Run replay:
    #    conda run -n agentenv-textcraft python envs/textcraft/scripts/replay_textcraft.py
"""
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = os.environ.get("TEXTCRAFT_BASE", "http://127.0.0.1:36011")
DATA = "/mnt/data/xiangchao/verl-agent-ee-final/envs/textcraft/data/raw/textcraft_train.json"
OUT = "/mnt/data/xiangchao/verl-agent-ee-final/envs/textcraft/data/replay/replay_full.jsonl"
SUMMARY = "/mnt/data/xiangchao/verl-agent-ee-final/envs/textcraft/data/replay/replay_summary.json"

ACTION_RE = re.compile(r"Action:\s*\n?(.+?)(?:\n\n|\Z)", re.DOTALL)


def extract_task_obs(rec):
    for msg in rec["conversations"]:
        if msg["from"] == "human" and "Crafting commands:" in msg["value"]:
            return msg["value"].split("Instruction:\n", 1)[-1].strip()
    return None


def extract_expert_actions(rec):
    """Pull out the expert's action strings (one per gpt turn after handshake)."""
    actions = []
    for msg in rec["conversations"]:
        if msg["from"] == "gpt" and msg.get("loss"):
            m = ACTION_RE.search(msg["value"])
            if m:
                action_text = m.group(1).strip()
                # take only the first line of the action (matches TextCraftEnvClient.step)
                action_text = action_text.split("\n")[0].strip()
                # match server-side cleanup in agentenv/envs/textcraft.py
                action_text = re.sub(r"[^A-Za-z0-9, ]+", "", action_text)
                action_text = " ".join(action_text.split()).strip()
                actions.append(action_text)
    return actions


def build_goal_to_data_idx():
    """Build goal_name → smallest data_idx that produces it."""
    sys.path.insert(
        0,
        "/mnt/data/xiangchao/verl-agent-ee-final/envs/textcraft/agentgym/agentenv-textcraft",
    )
    from agentenv_textcraft.crafting_tree import CraftingTree

    tree = CraftingTree(
        minecraft_dir="/mnt/data/xiangchao/verl-agent-ee-final/envs/textcraft/agentgym/agentenv-textcraft/agentenv_textcraft"
    )
    item_depth_list = sorted(tree.item_recipes_min_depth(1), key=lambda x: x[1])
    period = len(item_depth_list)
    # data_idx i → goal item id (stripped of "minecraft:" prefix and "_" → " ")
    goal_to_idx = {}
    for i, (item_id, _) in enumerate(item_depth_list):
        name = item_id.replace("minecraft:", "").replace("_", " ")
        goal_to_idx.setdefault(name, i)
    return goal_to_idx, period


def replay_one(rec, goal_to_idx):
    """Replay one trajectory; returns a dict with per-step results."""
    task_obs = extract_task_obs(rec)
    goal = task_obs.split("Goal: craft ")[-1].rstrip(".")
    actions = extract_expert_actions(rec)
    out = {
        "item_id": rec["item_id"],
        "goal": goal,
        "n_actions": len(actions),
        "data_idx": None,
        "final_done": False,
        "final_reward": 0,
        "steps": [],
        "failure_reason": None,
    }
    if goal not in goal_to_idx:
        out["failure_reason"] = "goal_not_in_env_universe"
        return out
    data_idx = goal_to_idx[goal]
    out["data_idx"] = data_idx

    # create dedicated env_id for this trajectory
    try:
        cr = requests.post(f"{BASE}/create", json={}, timeout=30).json()
        env_id = cr["id"]
        rs = requests.post(
            f"{BASE}/reset", json={"id": env_id, "data_idx": data_idx}, timeout=30
        ).json()
    except Exception as e:
        out["failure_reason"] = f"create/reset_error: {e}"
        return out

    last = {"observation": rs["observation"], "reward": 0, "done": False}
    for i, action in enumerate(actions):
        try:
            r = requests.post(
                f"{BASE}/step", json={"id": env_id, "action": action}, timeout=30
            ).json()
        except Exception as e:
            out["steps"].append({"action": action, "error": str(e)})
            out["failure_reason"] = f"step_error_at_{i}"
            break
        out["steps"].append(
            {
                "action": action,
                "observation": r.get("observation"),
                "reward": r.get("reward", 0),
                "done": r.get("done", False),
            }
        )
        last = r
        if r.get("done"):
            break
    out["final_done"] = bool(last.get("done"))
    out["final_reward"] = int(last.get("reward", 0))
    if not out["final_done"]:
        out["failure_reason"] = out["failure_reason"] or "expert_actions_exhausted_without_done"

    # cleanup
    try:
        requests.post(f"{BASE}/close", json={"id": env_id}, timeout=10)
    except Exception:
        pass
    return out


def main():
    print(f"Loading trajectories from {DATA} ...")
    with open(DATA) as f:
        data = json.load(f)
    print(f"  {len(data)} trajectories")

    print("Building goal → data_idx map ...")
    goal_to_idx, period = build_goal_to_data_idx()
    print(f"  env has {period} unique goals; covers all 374 traj goals: "
          f"{all(extract_task_obs(r).split('Goal: craft ')[-1].rstrip('.') in goal_to_idx for r in data)}")

    # parallel via 8 threads (server handles concurrent IDs fine)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(replay_one, rec, goal_to_idx): rec for rec in data}
        for j, f in enumerate(as_completed(futures)):
            results.append(f.result())
            if (j + 1) % 50 == 0:
                print(f"  replayed {j+1}/{len(data)} ({time.time()-t0:.1f}s)")
    print(f"replay done in {time.time()-t0:.1f}s")

    # stream out
    results.sort(key=lambda x: int(x["item_id"].split("_")[1]))
    with open(OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(results)} rows to {OUT}")

    # summary
    pass_done = sum(1 for r in results if r["final_done"] and r["final_reward"] == 1)
    fail_no_done = sum(1 for r in results if not r["final_done"])
    fail_done_no_reward = sum(1 for r in results if r["final_done"] and r["final_reward"] == 0)
    reasons = Counter(r["failure_reason"] for r in results if not (r["final_done"] and r["final_reward"] == 1))
    summary = {
        "total": len(results),
        "pass_done_reward1": pass_done,
        "fail_no_done": fail_no_done,
        "fail_done_no_reward": fail_done_no_reward,
        "fail_reason_breakdown": dict(reasons),
        "pass_rate": pass_done / len(results),
    }
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Replay every AgentTraj-L SciWorld trajectory through scienceworld 1.2.3
and persist the full per-step env response so the trajectory-quality filter
can be applied later from raw data, without re-running the env.

Inputs
------
- envs/scienceworld/data/raw/sciworld_train.json (AgentTraj-L SciWorld split)

Outputs
-------
- envs/scienceworld/data/replay/replay_full.jsonl
    One line per trajectory. Each line includes:
      * AgentTraj-L's recorded actions / thoughts / observations
      * For each step we successfully execute: action, env observation,
        reward, score, done, full info dict
      * replay_status (completed / no_known_action / step_exception /
        load_failed / index_out_of_range / worker_crash)
      * final_score, final_done, n_steps_executed
- envs/scienceworld/data/replay/replay_summary.jsonl
    One line per trajectory with just the summary fields. For quick stats.

Run (from workspace root)
-------------------------
    conda run -n agentenv-sciworld --no-capture-output python \\
        envs/scienceworld/scripts/replay_agenttraj.py [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# AgentGym's SciWorldEnv excludes these 7 task names from its games list, so we
# match that to keep item_id <-> games[N] mapping consistent.
EXCLUDED_TASKS = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}

ACTION_RE = re.compile(r"Action:\s*\n?(.+?)(?:\n\n|\Z)", re.DOTALL)
THOUGHT_RE = re.compile(r"Thought:\s*\n?(.+?)\n\nAction:", re.DOTALL)


def parse_trajectory(traj):
    """Extract parallel lists of actions, thoughts, and recorded observations
    from one AgentTraj-L ShareGPT-format trajectory."""
    actions, thoughts, rec_obs = [], [], []
    for i, m in enumerate(traj["conversations"]):
        if m.get("from") == "gpt" and "Action:" in m.get("value", ""):
            am = ACTION_RE.search(m["value"])
            actions.append(am.group(1).strip() if am else "")
            tm = THOUGHT_RE.search(m["value"])
            thoughts.append(tm.group(1).strip() if tm else "")
        # Turn 2 is the initial task description + observation; turn 4, 6, ...
        # are env responses after each agent action. We collect from turn 4 on.
        if i >= 4 and m.get("from") == "human":
            rec_obs.append(m["value"])
    return actions, thoughts, rec_obs


def build_games_list():
    """Re-enumerate AgentGym's (task_key, task_value, variation_idx) list in the
    same order it constructs internally, so item_id `sciworld_N` resolves to
    games[N]."""
    from scienceworld import ScienceWorldEnv

    e = ScienceWorldEnv()
    games = []
    for key, value in e.tasks.items():
        if key not in EXCLUDED_TASKS:
            for v in range(e.getMaxVariations(value)):
                games.append((key, value, v))
    # `close` was added in scienceworld 1.2.x; on 1.1.x just drop the reference.
    if hasattr(e, "close"):
        e.close()
    return games


# --- per-worker state ------------------------------------------------------
# ProcessPoolExecutor spawns separate Python processes; each one gets its own
# ScienceWorldEnv (own JVM) initialized once in _worker_init.

_WORKER_ENV = None
_WORKER_GAMES = None


def _worker_init():
    global _WORKER_ENV, _WORKER_GAMES
    from scienceworld import ScienceWorldEnv

    _WORKER_ENV = ScienceWorldEnv()
    _WORKER_GAMES = build_games_list()


def _safe_info(info):
    """Coerce env.step's info dict into something json-serializable. Most fields
    are primitives; we stringify anything weird rather than dropping it."""
    if not isinstance(info, dict):
        return {"_raw": str(info)}
    out = {}
    for k, v in info.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def replay_one(traj):
    """Replay one trajectory through the worker's env. Returns a dict
    capturing everything needed to do filter / quality analysis later."""
    item_id = traj["item_id"]
    n = int(item_id.split("_")[-1])

    if n >= len(_WORKER_GAMES):
        return {
            "item_id": item_id,
            "data_idx": n,
            "replay_status": "index_out_of_range",
            "replay_error": f"item_id {n} >= games count {len(_WORKER_GAMES)}",
        }

    task_key, task_value, var = _WORKER_GAMES[n]
    actions, thoughts, rec_obs = parse_trajectory(traj)

    try:
        _WORKER_ENV.load(task_value, var)
    except Exception as exc:  # noqa: BLE001 — record whatever the env throws
        return {
            "item_id": item_id,
            "data_idx": n,
            "task_key": task_key,
            "task_value": task_value,
            "variation_idx": var,
            "n_actions": len(actions),
            "agenttraj_actions": actions,
            "agenttraj_thoughts": thoughts,
            "agenttraj_observations": rec_obs,
            "replay_status": "load_failed",
            "replay_error": repr(exc),
        }

    steps = []
    replay_status = "completed"
    replay_error = None
    final_score = 0
    final_done = False

    for i, action in enumerate(actions):
        try:
            ob, reward, done, info = _WORKER_ENV.step(action)
        except Exception as exc:  # noqa: BLE001
            replay_status = "step_exception"
            replay_error = f"step {i}: {exc!r}"
            break

        score = info.get("score", 0) if isinstance(info, dict) else 0
        steps.append(
            {
                "step": i,
                "action": action,
                "observation": ob,
                "reward": reward,
                "score": score,
                "done": bool(done),
                "info": _safe_info(info),
            }
        )
        final_score = score
        final_done = bool(done)
        if isinstance(ob, str) and "No known action" in ob:
            replay_status = "no_known_action"
            replay_error = f"step {i}: action {action!r} produced No-known-action"
            break

    return {
        "item_id": item_id,
        "data_idx": n,
        "task_key": task_key,
        "task_value": task_value,
        "variation_idx": var,
        "n_actions": len(actions),
        "agenttraj_actions": actions,
        "agenttraj_thoughts": thoughts,
        "agenttraj_observations": rec_obs,
        "replay_status": replay_status,
        "replay_error": replay_error,
        "replay_steps": steps,
        "final_score": final_score,
        "final_done": final_done,
        "n_steps_executed": len(steps),
    }


SUMMARY_KEYS = (
    "item_id",
    "data_idx",
    "task_key",
    "variation_idx",
    "n_actions",
    "replay_status",
    "final_score",
    "final_done",
    "n_steps_executed",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        default="envs/scienceworld/data/normalized/sciworld_train.json",
        help=(
            "AgentTraj-L SciWorld split JSON. Default is the workspace's "
            "normalized version (produced by normalize_agenttraj.py), which "
            "fixes 'green house'/'greenhouse', stale terminal-1 grammar, and "
            "AgentTraj-L typos so trajectories execute in unmodified "
            "scienceworld. Pass envs/scienceworld/data/raw/sciworld_train.json "
            "to replay the un-normalized original for comparison."
        ),
    )
    ap.add_argument(
        "--output-dir",
        default="envs/scienceworld/data/replay",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Replay only the first N trajectories (smoke test).",
    )
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    full_out = out_dir / "replay_full.jsonl"
    summary_out = out_dir / "replay_summary.jsonl"

    print(f"input        : {input_path}", flush=True)
    print(f"output (full): {full_out}", flush=True)
    print(f"output (summary): {summary_out}", flush=True)
    print(f"workers      : {args.workers}", flush=True)

    with open(input_path) as f:
        data = json.load(f)
    if args.limit is not None:
        data = data[: args.limit]
    total = len(data)
    print(f"trajectories : {total}", flush=True)
    if total == 0:
        return

    t_start = time.time()
    done_count = 0
    summary_rows = []

    with open(full_out, "w") as fout, ProcessPoolExecutor(
        max_workers=args.workers, initializer=_worker_init
    ) as pool:
        futures = {pool.submit(replay_one, traj): traj["item_id"] for traj in data}
        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "item_id": iid,
                    "replay_status": "worker_crash",
                    "replay_error": repr(exc),
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            summary_rows.append({k: rec.get(k) for k in SUMMARY_KEYS})
            done_count += 1
            if done_count % 50 == 0 or done_count == total:
                elapsed = time.time() - t_start
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                print(
                    f"  [{done_count:>5}/{total}] elapsed={elapsed:6.1f}s "
                    f"rate={rate:5.1f}/s eta={eta:6.1f}s",
                    flush=True,
                )

    with open(summary_out, "w") as f:
        for r in summary_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.1f}s. {done_count} records written.", flush=True)
    print(f"  full    : {full_out}  ({full_out.stat().st_size:,} bytes)", flush=True)
    print(f"  summary : {summary_out}  ({summary_out.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()

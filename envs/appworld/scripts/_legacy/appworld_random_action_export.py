"""
AppWorld Random Action Export
==============================

Along expert trajectories, sample random actions at each step and collect
the resulting observations. Mirrors the WebShop random_action_export.py
pattern adapted for AppWorld's free-form code action space.

Each step's flow:
  base_env advances along gold trajectory -> at current state, sample N
  random code actions -> for each, create fresh env, replay prefix,
  execute random action, record (states_before, states_after, gold_action,
  random_action, ...)

Usage:
  python -m agent_system.environments.appworld_random_action_export 0 100 --port 8000
  python -m agent_system.environments.appworld_random_action_export              # all tasks
"""

import os
import sys
import json
import random
import time
import uuid
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# ---------------------------------------------------------------------------
# From replay module
# ---------------------------------------------------------------------------
from agent_system.environments.appworld_expert_replay import (
    AppWorldExpertReplay,
    ReplayEnvManager,
    SingleWorkerEnv,
    StepInfo,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_RANDOM_ACTIONS = 5
CHECKPOINT_INTERVAL = 50


# ---------------------------------------------------------------------------
# Generic branch action pool
# ---------------------------------------------------------------------------

# Pre-defined pool of alternative actions for AppWorld.
# Categories:
#   - EXPLORATION: API discovery calls (always valid, informative output)
#   - SUPERVISOR: common supervisor interactions
#   - WRONG_COMPLETION: premature or incorrect task completion
#   - NOOP: syntactically valid but useless code
#   - SYNTAX_ERROR: malformed code

GENERIC_BRANCH_ACTIONS = [
    # ---- Exploration ----
    "print(apis.api_docs.show_app_descriptions())",
    "print(apis.api_docs.show_api_descriptions(app_name='supervisor'))",
    "print(apis.api_docs.show_api_descriptions(app_name='spotify'))",
    "print(apis.api_docs.show_api_descriptions(app_name='venmo'))",
    "print(apis.api_docs.show_api_descriptions(app_name='phone'))",
    "print(apis.api_docs.show_api_descriptions(app_name='file_system'))",
    "print(apis.api_docs.show_api_descriptions(app_name='amazon'))",
    "print(apis.api_docs.show_api_descriptions(app_name='gmail'))",
    "print(apis.api_docs.show_api_descriptions(app_name='todoist'))",
    "print(apis.api_docs.show_api_descriptions(app_name='simple_note'))",
    # ---- Supervisor ----
    "print(apis.supervisor.show_account_passwords())",
    "print(apis.supervisor.show_profile())",
    "print(apis.supervisor.show_addresses())",
    "print(apis.supervisor.show_payment_cards())",
    # ---- Wrong completions ----
    "apis.supervisor.complete_task(status='fail')",
    "apis.supervisor.complete_task(answer='unknown')",
    "apis.supervisor.complete_task(answer='0')",
    "apis.supervisor.complete_task(answer=None)",
    # ---- Noop / trivial ----
    "print('hello world')",
    "x = 1\nprint(x)",
    "pass",
    "import datetime\nprint(datetime.datetime.now())",
    "print(type(apis))",
    # ---- Syntax errors ----
    "print(apis.",
    "for i in range(",
    "result = apis.supervisor.show_profile(\nprint(result",
]


def sample_branch_actions(gold_action: str, n_random: int = 5) -> List[str]:
    """Sample n_random actions from GENERIC_BRANCH_ACTIONS, excluding the gold."""
    pool = [a for a in GENERIC_BRANCH_ACTIONS if a.strip() != gold_action.strip()]
    if len(pool) >= n_random:
        return random.sample(pool, n_random)
    return list(pool)


# ---------------------------------------------------------------------------
# Core: probe a single action in a fresh environment
# ---------------------------------------------------------------------------

def probe_action(
    port: int,
    task_id: str,
    prefix_actions: List[str],
    action_to_probe: str,
    history_length: int = 2,
    max_interactions: int = 50,
) -> Dict[str, Any]:
    """
    Fresh-env probe (Strategy A):
    1. Create new AppWorld instance on given port
    2. Reset to task_id
    3. Replay prefix_actions (expert steps 0..step_idx-1)
    4. Execute action_to_probe
    5. Return observation, reward, done, error
    """
    name = f"probe_{uuid.uuid4().hex[:8]}"
    env = SingleWorkerEnv(
        port=port,
        experiment_name=name,
        max_interactions=max_interactions,
    )
    manager = ReplayEnvManager(env, history_length=history_length)
    try:
        text_obs, raw_obs, info = manager.reset(task_id)

        # Replay prefix
        for act in prefix_actions:
            try:
                manager.step(act)
            except Exception as e:
                return {
                    "obs_text": None,
                    "obs_raw": None,
                    "reward": None,
                    "done": None,
                    "info": None,
                    "error": f"prefix_replay_failed at {act[:50]!r}: {e}",
                }

        # Execute probe action
        text_obs, raw_obs, reward, done, info = manager.step(action_to_probe)
        return {
            "obs_text": text_obs,
            "obs_raw": raw_obs,
            "reward": reward,
            "done": done,
            "info": {k: v for k, v in info.items()},
            "error": None,
        }
    except Exception as e:
        return {
            "obs_text": None,
            "obs_raw": None,
            "reward": None,
            "done": None,
            "info": None,
            "error": str(e),
        }
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def run_export(
    task_start: int = 0,
    task_end: int = 999999,
    n_random: int = N_RANDOM_ACTIONS,
    port: int = 8000,
    dataset_name: str = "train",
    seed: int = 42,
    output_dir: str = "appworld_obs_from_random",
    log_dir: str = "logs",
    history_length: int = 2,
    max_interactions: int = 50,
    cache_dir: Optional[str] = None,
):
    random.seed(seed)

    # ---- 1. Create replay manager (loads GT from disk, no server needed) ----
    replay = AppWorldExpertReplay(
        dataset_name=dataset_name,
        port=port,
        history_length=history_length,
        max_interactions=max_interactions,
        cache_dir=cache_dir,
    )

    # ---- 2. Filter tasks by index range ----
    all_tasks = replay.matched_tasks
    pairs = [
        (tid, blocks, meta)
        for i, (tid, blocks, meta) in enumerate(all_tasks)
        if task_start <= i < task_end
    ]
    print(f"Range [{task_start}, {task_end}): {len(pairs)} tasks to process")

    if not pairs:
        print("WARNING: No tasks in range, exiting")
        return

    # ---- 3. Output paths ----
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(
        output_dir,
        f"random_{n_random}act_{timestamp}_{task_start}-{task_end}.json",
    )
    task_log = os.path.join(
        log_dir,
        f"tasks_{timestamp}_{task_start}-{task_end}.txt",
    )

    # ---- 4. Main loop ----
    data: List[Dict[str, Any]] = []
    task_counter = 0
    rollout_t0 = time.time()

    for task_id, code_blocks, metadata in tqdm(pairs, desc="Tasks", ncols=100):
        total_steps = len(code_blocks)

        # Replay expert trajectory step by step
        for step in replay.replay_trajectory(task_id, code_blocks):
            current_obs = step.observation_text
            current_obs_raw = step.raw_observation
            gold_act = step.gold_action
            step_idx = step.step_idx

            # Sample branch actions
            branch_actions = sample_branch_actions(gold_act, n_random)

            for branch_act in branch_actions:
                probe_result = probe_action(
                    port=port,
                    task_id=task_id,
                    prefix_actions=code_blocks[:step_idx],
                    action_to_probe=branch_act,
                    history_length=history_length,
                    max_interactions=max_interactions,
                )

                record = {
                    # ---- Identifiers ----
                    "task_id": task_id,
                    "step": step_idx,
                    "total_steps": total_steps,
                    "is_first_step": step.is_first_step,
                    "is_last_step": step.is_last_step,

                    # ---- Task metadata ----
                    "task": step.task_description,
                    "supervisor": step.supervisor,
                    "difficulty": metadata.get("difficulty"),
                    "required_apps": metadata.get("required_apps"),

                    # ---- Actions ----
                    "gold_action": gold_act,
                    "all_gold_actions": code_blocks,
                    "extracted_action": branch_act,
                    "on_traj": (branch_act.strip() == gold_act.strip()),

                    # ---- Current state (before probe) ----
                    "states_before": current_obs,
                    "states_before_raw": current_obs_raw,

                    # ---- After probe action ----
                    "states_after": probe_result["obs_text"],
                    "states_after_raw": probe_result["obs_raw"],

                    # ---- Probe feedback ----
                    "probe_reward": probe_result["reward"],
                    "probe_done": probe_result["done"],
                    "probe_info": probe_result.get("info"),

                    # ---- Error ----
                    "error": probe_result["error"],
                }
                data.append(record)

            # Log task on first step
            if step_idx == 0:
                with open(task_log, "a", encoding="utf-8") as fp:
                    fp.write(f"{task_id}\t{step.task_description[:100]}\n")

        # Checkpoint
        task_counter += 1
        if task_counter % CHECKPOINT_INTERVAL == 0 and data:
            ckpt = f"{out_json}.ckpt_{task_counter}"
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            tqdm.write(
                f"Checkpoint @ task#{task_counter} "
                f"({len(data):,} records) -> {ckpt}"
            )

    # ---- 5. Save final output ----
    rollout_t1 = time.time()
    duration = rollout_t1 - rollout_t0

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    summary = [
        f"Start: {datetime.fromtimestamp(rollout_t0).strftime('%Y-%m-%d %H:%M:%S')}",
        f"End:   {datetime.fromtimestamp(rollout_t1).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration: {duration:.1f}s ({duration/3600:.2f}h)",
        f"Range: [{task_start}, {task_end})",
        f"Tasks: {len(pairs)}",
        f"Records: {len(data):,}",
        f"Errors: {sum(1 for r in data if r.get('error')):,}",
        f"Output: {out_json}",
    ]
    time_log = os.path.join(log_dir, f"time_{timestamp}_{task_start}-{task_end}.log")
    with open(time_log, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    for line in summary:
        print(line)
    print(f"Saved {len(data):,} records -> {out_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AppWorld: sample random actions along expert trajectory and export observations"
    )
    parser.add_argument("start", type=int, nargs="?", default=0,
                        help="Start task index (inclusive)")
    parser.add_argument("end", type=int, nargs="?", default=999999,
                        help="End task index (exclusive)")
    parser.add_argument("--n-random", type=int, default=N_RANDOM_ACTIONS,
                        help="Number of random actions per step")
    parser.add_argument("--port", type=int, default=8000,
                        help="AppWorld server port")
    parser.add_argument("--dataset", default="train",
                        help="AppWorld dataset split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output-dir", default="appworld_obs_from_random")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-interactions", type=int, default=50)
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to cache parsed ground truth")
    args = parser.parse_args()

    run_export(
        task_start=args.start,
        task_end=args.end,
        n_random=args.n_random,
        port=args.port,
        dataset_name=args.dataset,
        seed=args.seed,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        history_length=args.history_length,
        max_interactions=args.max_interactions,
        cache_dir=args.cache_dir,
    )

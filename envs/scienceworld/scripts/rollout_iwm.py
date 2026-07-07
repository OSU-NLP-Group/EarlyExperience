"""Probe K=3 random non-expert alternative actions at every expert state in
D_expert, capturing (s_i, alt, env's response) triples for IWM.

For each trajectory that survives the §3 filter (replay_status=='completed'
AND final_done AND final_score==100):

  Walk the env forward step-by-step. At each expert state s_i:
    1. Query env.getValidActionObjectCombinations() to get the admissible
       action list at s_i.
    2. Sample K=3 actions uniformly without replacement from
       admissible \\ {expert action at this step}.
    3. For each alternative:
         a. env.step(alt) -> capture env's response (s_i^j).
         b. env.load(task, var) + replay expert actions 0..i-1 to restore
            env to s_i (scienceworld has no native state save/restore so
            we replay from scratch).
    4. env.step(expert_action_i) to advance to s_{i+1} for the next step.

The expert triples (s_i, a_i, s_{i+1}) are already captured in
replay_full.jsonl and don't need to be re-probed; only alternative triples
are written here.

We also write one "initial" record per trajectory with the env's clean
initial observation (from env.look()), so the SFT builder can render the
full trajectory history starting from s_0 without re-querying the env.

Determinism: alternative sampling uses a deterministic seed per
(item_id, step) so reruns produce identical alternatives.

Output: envs/scienceworld/data/rollout/iwm_rollout.jsonl
  - One "initial" record per trajectory:
      {"item_id", "kind": "initial", "initial_obs": "..."}
  - K alt records per trajectory step:
      {"item_id", "kind": "alt", "step", "alt_idx", "action", "next_state"}

Run (from workspace root):
    conda run -n agentenv-sciworld --no-capture-output python \\
        envs/scienceworld/scripts/rollout_iwm.py [--workers N]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import md5
from pathlib import Path

K = 3

REPLAY_DEFAULT = "envs/scienceworld/data/replay/replay_full.jsonl"
OUTPUT_DEFAULT = "envs/scienceworld/data/rollout/iwm_rollout.jsonl"


# Worker globals — one ScienceWorldEnv (JVM) per process, initialized once.
_WORKER_ENV = None


def _worker_init():
    global _WORKER_ENV
    from scienceworld import ScienceWorldEnv

    _WORKER_ENV = ScienceWorldEnv()


def _seed_for(item_id: str, step: int) -> int:
    """Deterministic seed per (item_id, step) so the same alternatives are
    sampled across reruns."""
    return int(md5(f"{item_id}/{step}".encode()).hexdigest()[:16], 16)


def _sample_alternatives(admissible: list[str], expert_action: str, seed: int) -> list[str]:
    """Sample K distinct non-expert actions uniformly without replacement
    from admissible \\ {expert_action}. Returns up to K (fewer if the pool
    is too small — shouldn't happen in ScienceWorld where admissible is
    typically a few hundred)."""
    pool = [a for a in admissible if a != expert_action]
    rng = random.Random(seed)
    if len(pool) <= K:
        return list(pool)
    return rng.sample(pool, K)


def probe_trajectory(record: dict) -> list[dict]:
    """Walk one filtered trajectory and emit rollout records."""
    env = _WORKER_ENV
    item_id = record["item_id"]
    task_value = record["task_value"]
    var = record["variation_idx"]
    expert_actions = record["agenttraj_actions"]
    n_steps = len(record["replay_steps"])

    out: list[dict] = []

    # 1. Initial state via env.look() (does not advance env state).
    env.load(task_value, var)
    initial_obs = env.look()
    out.append({"item_id": item_id, "kind": "initial", "initial_obs": initial_obs})

    # 2. Walk forward. At each step i, env is at s_i (post-step-i-1 = pre-step-i).
    for i in range(n_steps):
        expert_action = expert_actions[i]

        # Query admissible at current state (s_i).
        admissible = env.getValidActionObjectCombinations()
        alts = _sample_alternatives(admissible, expert_action, _seed_for(item_id, i))

        # Probe each alternative. After each, env state has advanced to s_i^j;
        # restore env to s_i by reload+replay for the next probe.
        for alt_idx, alt in enumerate(alts):
            ob_after_alt, _, _, _ = env.step(alt)
            out.append(
                {
                    "item_id": item_id,
                    "kind": "alt",
                    "step": i,
                    "alt_idx": alt_idx,
                    "action": alt,
                    "next_state": ob_after_alt,
                }
            )
            # Restore env to s_i: reload + replay expert actions 0..i-1.
            env.load(task_value, var)
            for k in range(i):
                env.step(expert_actions[k])

        # Advance to s_{i+1} via expert step (env is currently at s_i).
        env.step(expert_action)

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", default=REPLAY_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N surviving trajectories (smoke).",
    )
    args = ap.parse_args()

    replay_path = Path(args.replay).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"replay : {replay_path}")
    print(f"output : {out_path}")
    print(f"workers: {args.workers}")
    print(f"K      : {K} alternatives per expert state")

    # Load filtered trajectories: stream replay_full.jsonl, keep only filter-passing.
    print("scanning replay_full for done && score==100 trajectories...")
    keep = []
    for line in open(replay_path):
        r = json.loads(line)
        if r.get("final_done") and r.get("final_score") == 100:
            keep.append(r)
    print(f"  kept {len(keep)} trajectories")

    if args.limit is not None:
        keep = keep[: args.limit]
        print(f"  limit applied: keeping {len(keep)}")

    t_start = time.time()
    n_done = 0
    n_records = 0

    with open(out_path, "w") as fout, ProcessPoolExecutor(
        max_workers=args.workers, initializer=_worker_init
    ) as pool:
        futures = {pool.submit(probe_trajectory, rec): rec["item_id"] for rec in keep}
        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                lines = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  worker failed on {iid}: {exc!r}", flush=True)
                continue
            for line in lines:
                fout.write(json.dumps(line, ensure_ascii=False) + "\n")
                n_records += 1
            fout.flush()
            n_done += 1
            if n_done % 50 == 0 or n_done == len(keep):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(keep) - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done:>5}/{len(keep)}] elapsed={elapsed:7.1f}s "
                    f"rate={rate:5.2f}/s eta={eta:7.1f}s  rec={n_records}",
                    flush=True,
                )

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.1f}s.")
    print(f"  trajectories: {n_done}")
    print(f"  records     : {n_records}")
    print(f"  output      : {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

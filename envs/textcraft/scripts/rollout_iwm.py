"""IWM rollout: probe K=5 random non-expert alternative actions at every
expert state in D_expert, capturing (s_i, alt, env_response_to_alt) triples.

Sampling strategy (NOTES.md "IWM alt sampling", user-confirmed 2026-05-25):
  uniform random without replacement from `admissible(s_i) \ {expert_i}`,
  where admissible is computed client-side by `admissible.AdmissibleEnumerator`.
  No LLM. K=5 alts per state, accept K_actual < 5 when admissible is short.

State restoration: TextCraft has no native save/restore (env_wrapper exposes
only /create, /reset(data_idx), /step). To probe alt j at state s_i we:
  1. step alt -> capture next_state observation
  2. /reset(data_idx) + replay expert[0..i-1] -> restore to s_i
This costs O(i) replays per alt. Per traj: ~N(K+2) reset+replay ops, ~N
expert steps. Total ≈ 374 traj × ~8 steps × 7 ops × ~4 steps each ≈ 80k
env steps. With HTTP concurrency this finishes in well under an hour.

Determinism: alt sampling seed = md5(f"{item_id}/{step}")[:16], so reruns
produce identical alt sets. Same pattern as sciworld.

Output: envs/textcraft/data/rollout/iwm_rollout.jsonl
  - One "initial" record per trajectory:
      {"item_id", "kind": "initial", "data_idx", "initial_obs", "commands_list"}
  - K_actual alt records per expert state:
      {"item_id", "kind": "alt", "step", "alt_idx", "action", "next_state",
       "reward", "done"}

Expert IWM triples (s_i, expert_i, s_{i+1}) are NOT re-probed — they are
already captured in replay_full.jsonl's `steps[i].observation` and are
joined in by `build_iwm_sft.py` later.

Run (server must be up on $TEXTCRAFT_BASE, default port 36011):
    conda run -n agentenv-textcraft --no-capture-output textcraft \\
        --host 127.0.0.1 --port 36011 &
    conda run -n agentenv-textcraft --no-capture-output python \\
        envs/textcraft/scripts/rollout_iwm.py [--workers N] [--limit M]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from pathlib import Path

import requests

# Local imports (rely on the script being launched from workspace root).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from admissible import AdmissibleEnumerator

K = 5

BASE = os.environ.get("TEXTCRAFT_BASE", "http://127.0.0.1:36011")
REPLAY_DEFAULT = "envs/textcraft/data/replay/replay_full.jsonl"
OUTPUT_DEFAULT = "envs/textcraft/data/rollout/iwm_rollout.jsonl"

# "Inventory: [item name] (count) [item name] (count) " -> dict
_INV_RE = re.compile(r"\[([^\]]+)\]\s*\((\d+)\)")


def _seed_for(item_id: str, step: int) -> int:
    return int(md5(f"{item_id}/{step}".encode()).hexdigest()[:16], 16)


def _parse_inventory(obs: str) -> dict:
    """Parse env's 'Inventory: ...' observation into {item_name: count}."""
    if not obs or not obs.startswith("Inventory:"):
        return {}
    return {m.group(1).strip(): int(m.group(2)) for m in _INV_RE.finditer(obs)}


def _sample_alternatives(admissible, expert_action, seed):
    """Uniform without replacement from `admissible \\ {expert_action}`,
    returning up to K (or fewer if the pool is short)."""
    pool = [a for a in admissible if a != expert_action]
    rng = random.Random(seed)
    if len(pool) <= K:
        rng.shuffle(pool)  # stable but seeded ordering
        return list(pool)
    return rng.sample(pool, K)


def _post(path, payload, timeout=30):
    r = requests.post(f"{BASE}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def probe_trajectory(record, enum: AdmissibleEnumerator):
    """Walk one trajectory; emit one initial record + up to K records per step.

    Inputs from replay_full row:
      item_id, data_idx, steps[i].action
    """
    item_id = record["item_id"]
    data_idx = record["data_idx"]
    expert_actions = [s["action"] for s in record["steps"]]
    n_steps = len(expert_actions)

    out = []

    # 1. Allocate env, reset, capture initial obs + commands_list
    cr = _post("/create", {})
    env_id = cr["id"]
    rs = _post("/reset", {"id": env_id, "data_idx": data_idx})
    initial_obs = rs["observation"]
    commands_list = [l for l in initial_obs.split("\n") if l.startswith("craft ")]
    out.append(
        {
            "item_id": item_id,
            "kind": "initial",
            "data_idx": data_idx,
            "initial_obs": initial_obs,
            "commands_list": commands_list,
        }
    )

    # 2. Walk forward. At iteration i, env is at s_i (after expert[0..i-1]).
    for i, expert_action in enumerate(expert_actions):
        # Query inventory (does not modify env state).
        inv_resp = _post("/step", {"id": env_id, "action": "inventory"})
        inventory = _parse_inventory(inv_resp["observation"])

        # Compute admissible + sample alts (excluding expert).
        admissible = enum.enumerate(commands_list, inventory)
        alts = _sample_alternatives(admissible, expert_action, _seed_for(item_id, i))

        # Probe each alt. After /step alt, env state has advanced; reset+replay
        # to restore to s_i before next probe (or before the post-loop advance).
        for alt_idx, alt in enumerate(alts):
            r = _post("/step", {"id": env_id, "action": alt})
            out.append(
                {
                    "item_id": item_id,
                    "kind": "alt",
                    "step": i,
                    "alt_idx": alt_idx,
                    "action": alt,
                    "next_state": r.get("observation"),
                    "reward": r.get("reward", 0),
                    "done": r.get("done", False),
                }
            )
            # Restore env to s_i. Even after the LAST alt we still need to
            # restore so the post-loop expert step starts from s_i, not s_i^last.
            _post("/reset", {"id": env_id, "data_idx": data_idx})
            for k in range(i):
                _post("/step", {"id": env_id, "action": expert_actions[k]})

        # If the loop produced 0 alts (admissible \ {expert} was empty), env
        # is still at s_i from the inventory query (which doesn't move state).
        # Advance to s_{i+1} via expert step.
        _post("/step", {"id": env_id, "action": expert_action})

    # 3. Cleanup
    _post("/close", {"id": env_id})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", default=REPLAY_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N trajectories (smoke).",
    )
    args = ap.parse_args()

    replay_path = Path(args.replay).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"server  : {BASE}")
    print(f"replay  : {replay_path}")
    print(f"output  : {out_path}")
    print(f"workers : {args.workers}")
    print(f"K       : {K} alternatives per expert state")

    # Server reachability check.
    try:
        requests.get(BASE + "/", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"FATAL: textcraft server unreachable at {BASE}: {e!r}")

    print("loading replay_full.jsonl (full 374, no filter)...")
    rows = [json.loads(l) for l in open(replay_path)]
    print(f"  loaded {len(rows)} trajectories")

    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"  limit applied: keeping first {len(rows)}")

    # Shared enumerator (thread-safe: read-only after load).
    enum = AdmissibleEnumerator()
    print(f"  crafting_tree loaded ({len(enum.tree.itemid_recipes)} craftable items)")

    t0 = time.time()
    write_lock = threading.Lock()
    n_traj = 0
    n_alt = 0
    n_initial = 0

    with open(out_path, "w") as fout, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {pool.submit(probe_trajectory, rec, enum): rec["item_id"] for rec in rows}
        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                lines = fut.result()
            except Exception as exc:
                print(f"  worker failed on {iid}: {exc!r}", flush=True)
                continue
            with write_lock:
                for line in lines:
                    fout.write(json.dumps(line, ensure_ascii=False) + "\n")
                    if line["kind"] == "alt":
                        n_alt += 1
                    elif line["kind"] == "initial":
                        n_initial += 1
                fout.flush()
            n_traj += 1
            if n_traj % 20 == 0 or n_traj == len(rows):
                el = time.time() - t0
                rate = n_traj / el if el > 0 else 0
                eta = (len(rows) - n_traj) / rate if rate > 0 else 0
                print(
                    f"  [{n_traj:>4}/{len(rows)}] elapsed={el:6.1f}s "
                    f"rate={rate:4.2f}traj/s eta={eta:6.1f}s alt={n_alt}",
                    flush=True,
                )

    el = time.time() - t0
    print(f"\nDONE in {el:.1f}s.")
    print(f"  trajectories probed: {n_traj}")
    print(f"  initial records   : {n_initial}")
    print(f"  alt records       : {n_alt}")
    print(f"  output            : {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

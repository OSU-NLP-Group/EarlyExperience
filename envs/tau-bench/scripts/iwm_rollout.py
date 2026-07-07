"""Probe each alternative tool call against the env at the right state.

Pure Python, no LLM. For every (task_id, trial, obs_idx) in D_expert:
  1. Reconstruct env.data at state s_i by replaying recorded expert tool calls
     0..i-1 on a fresh load_data().
  2. Snapshot the data dict.
  3. For each alternative (from alternatives.jsonl): deepcopy snapshot, invoke the
     tool, capture the observation string. Errors and unknown-tool responses are
     kept verbatim (paper §5 says invalid-action signals are valid IWM training data).
  4. Record the expert's (action, next_obs) too — next_obs is read straight from
     the recorded traj since the env is pinned and deterministic.

Output: iwm_rollout.jsonl, one line per obs.

Run:
    PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output python \\
        envs/tau-bench/scripts/iwm_rollout.py [--max-trajs N]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from tau_bench.envs.retail.data import load_data
from tau_bench.envs.retail.tools import ALL_TOOLS

TOOLS_MAP = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}

D_EXPERT_DEFAULT = "envs/tau-bench/data/rollout/D_expert.jsonl"
ALTS_DEFAULT = "envs/tau-bench/data/rollout/alternatives.jsonl"
OUT_DEFAULT = "envs/tau-bench/data/rollout/iwm_rollout.jsonl"


def invoke(name: str, args: dict, data: dict) -> str:
    """Mirror tau_bench Env.step's tool branch."""
    if name not in TOOLS_MAP:
        return f"Unknown action {name}"
    try:
        return TOOLS_MAP[name].invoke(data=data, **args)
    except Exception as e:
        return f"Error: {e}"


def expert_action_from_turn(m: dict) -> tuple[str, dict, bool]:
    """Return (name, args, is_tool). For respond turns, returns ('respond', {'content': ...}, False)."""
    if m.get("tool_calls"):
        tc = m["tool_calls"][0]["function"]
        args = tc["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        return tc["name"], args, True
    return "respond", {"content": m.get("content", "")}, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-expert", default=D_EXPERT_DEFAULT)
    ap.add_argument("--alts", default=ALTS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--max-trajs", type=int, default=None)
    args = ap.parse_args()

    # Index alts by (task_id, trial, obs_idx)
    alt_map = {}
    with open(args.alts) as f:
        for line in f:
            r = json.loads(line)
            alt_map[(r["task_id"], r["trial"], r["obs_idx"])] = r.get("alts", [])
    print(f"alternatives indexed: {len(alt_map)} obs")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "w")

    stats = {
        "trajs": 0, "obs_total": 0,
        "expert_tool": 0, "expert_respond": 0,
        "obs_w_alts": 0, "obs_wo_alts": 0,
        "alts_probed": 0, "alts_unknown": 0, "alts_error": 0,
        "alts_source_llm": 0, "alts_source_fallback_random": 0,
    }
    t0 = time.time()

    with open(args.d_expert) as f:
        for ti, line in enumerate(f):
            if args.max_trajs is not None and ti >= args.max_trajs:
                break
            r = json.loads(line)
            traj = r["traj"]
            task_id, trial = r["task_id"], r["trial"]
            data = load_data()  # fresh DB at start of every traj

            for i, m in enumerate(traj):
                if m["role"] != "assistant":
                    continue
                stats["obs_total"] += 1
                name, eargs, is_tool = expert_action_from_turn(m)
                stats["expert_tool" if is_tool else "expert_respond"] += 1

                # expert next_obs is the literal next message's content
                exp_next = traj[i+1].get("content", "") if i+1 < len(traj) else ""

                # snapshot, probe alts
                snapshot = copy.deepcopy(data)
                alts = alt_map.get((task_id, trial, i), [])
                if alts:
                    stats["obs_w_alts"] += 1
                else:
                    stats["obs_wo_alts"] += 1
                alt_results = []
                for a in alts:
                    probe = copy.deepcopy(snapshot)
                    nxt = invoke(a["name"], a["arguments"], probe)
                    if nxt.startswith("Unknown action"):
                        stats["alts_unknown"] += 1
                    elif nxt.startswith("Error:"):
                        stats["alts_error"] += 1
                    src = a.get("source", "llm")
                    stats[f"alts_source_{src}"] = stats.get(f"alts_source_{src}", 0) + 1
                    alt_results.append({
                        "action": {"name": a["name"], "arguments": a["arguments"]},
                        "next_obs": nxt,
                        "source": src,
                    })
                    stats["alts_probed"] += 1

                # Advance state: execute expert tool on the snapshot (respond doesn't mutate)
                data = snapshot
                if is_tool:
                    invoke(name, eargs, data)

                out.write(json.dumps({
                    "task_id": task_id, "trial": trial, "obs_idx": i,
                    "expert_kind": "tool" if is_tool else "respond",
                    "expert": {"action": {"name": name, "arguments": eargs},
                               "next_obs": exp_next},
                    "alts": alt_results,
                }) + "\n")

            stats["trajs"] += 1
            if stats["trajs"] % 50 == 0:
                print(f"  [{stats['trajs']} traj, {stats['obs_total']} obs] "
                      f"{time.time()-t0:.1f}s elapsed")

    out.close()
    dt = time.time() - t0
    print()
    print(f"DONE in {dt:.2f}s ({dt/60:.2f} min)")
    for k, v in stats.items():
        print(f"  {k:30s} {v}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()

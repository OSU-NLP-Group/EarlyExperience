"""Build reflection_sft.jsonl from sr_rollout.jsonl + sr_rerun.jsonl.

Merge attempt 1 + attempt 2 per (task_id, trial, obs_idx), preferring whichever
has FEWER quality flags. One record per obs = 5,249 records.

Native OpenAI Chat Completions format:
  - Top-level `messages` (multi-turn) + `tools` (16 retail tool schemas).
  - LiteLLM-specific metadata fields on each historical message are stripped.
  - Target last assistant turn:
      * tool turn:    {content: <reflection_CoT>, tool_calls: [<expert_tool>]}
        — reflection in OpenAI `content`, action in structured `tool_calls`.
        qwen2_tool template's if/elif renders only tool_calls and skips content,
        so content here trains the reflection text on the loss applied to the
        target assistant tokens.
      * respond turn: {content: <reflection_CoT> + "\\n\\n" + <respond_text>}
        — no `tool_calls` field; customer-facing reply appended after reflection.

Run:
    PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output python \\
        envs/tau-bench/scripts/build_reflection_sft.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from tau_bench.envs.retail.tools import ALL_TOOLS

D_EXPERT_DEFAULT = "envs/tau-bench/data/rollout/D_expert.jsonl"
SR_IN_DEFAULT = "envs/tau-bench/data/rollout/sr_rollout.jsonl"
SR_RERUN_DEFAULT = "envs/tau-bench/data/rollout/sr_rerun.jsonl"
OUT_DEFAULT = "envs/tau-bench/data/sft/reflection_sft.jsonl"

FLAG_KEYS = ("is_doubled", "has_banned_vocab", "has_numbered_label", "exceeded_soft_cap")
TOOL_SCHEMAS = [t.get_info() for t in ALL_TOOLS]


def clean_tool_call(tc: dict) -> dict:
    return {
        "id": tc["id"],
        "type": tc.get("type", "function"),
        "function": {
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
        },
    }


def clean_message(m: dict) -> dict:
    """Strip LiteLLM-specific metadata; keep only canonical OpenAI Chat
    Completions fields."""
    role = m.get("role")
    if role in ("system", "user"):
        return {"role": role, "content": m.get("content", "")}
    if role == "assistant":
        out = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            out["tool_calls"] = [clean_tool_call(tc) for tc in m["tool_calls"]]
        return out
    if role == "tool":
        out = {"role": "tool", "content": m.get("content") or ""}
        if "tool_call_id" in m:
            out["tool_call_id"] = m["tool_call_id"]
        if "name" in m:
            out["name"] = m["name"]
        return out
    return dict(m)


def count_flags(flags: dict) -> int:
    return sum(1 for k in FLAG_KEYS if flags.get(k))


def pick_attempt(rec1: dict, rec2: dict | None) -> tuple[dict, int]:
    """Return (chosen_rec, attempt_used).
    Prefer attempt 2 if its flag count is strictly LOWER; otherwise stay with 1.
    """
    if rec2 is None or "error" in rec2:
        return rec1, 1
    f1, f2 = count_flags(rec1["flags"]), count_flags(rec2["flags"])
    return (rec2, 2) if f2 < f1 else (rec1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-expert", default=D_EXPERT_DEFAULT)
    ap.add_argument("--sr-in", default=SR_IN_DEFAULT)
    ap.add_argument("--sr-rerun", default=SR_RERUN_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    d_traj: dict[tuple[int, int], list] = {}
    with open(args.d_expert) as f:
        for line in f:
            r = json.loads(line)
            d_traj[(r["task_id"], r["trial"])] = r["traj"]

    # Index sr_rollout (attempt 1) and sr_rerun (attempt 2) by (task,trial,obs)
    sr1 = {}
    with open(args.sr_in) as f:
        for line in f:
            r = json.loads(line)
            sr1[(r["task_id"], r["trial"], r["obs_idx"])] = r
    sr2 = {}
    if Path(args.sr_rerun).exists():
        with open(args.sr_rerun) as f:
            for line in f:
                r = json.loads(line)
                sr2[(r["task_id"], r["trial"], r["obs_idx"])] = r

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_total, n_tool, n_resp = 0, 0, 0
    attempt_used = collections.Counter()
    flag_count_after = collections.Counter()

    with open(args.out, "w") as fout:
        for key, rec1 in sr1.items():
            if "error" in rec1:
                continue
            chosen, att = pick_attempt(rec1, sr2.get(key))
            attempt_used[att] += 1
            flag_count_after[count_flags(chosen["flags"])] += 1

            tid, tr, oi = key
            traj = d_traj.get((tid, tr))
            if traj is None:
                continue
            history = [clean_message(m) for m in traj[:oi]]
            assistant_turn = traj[oi]
            kind = chosen["expert_kind"]
            reflection = chosen["reflection_raw"]

            if kind == "tool":
                target = {
                    "role": "assistant",
                    "content": reflection,
                    "tool_calls": [clean_tool_call(tc) for tc in assistant_turn["tool_calls"]],
                }
                n_tool += 1
            else:
                respond_text = assistant_turn.get("content", "") or ""
                target = {
                    "role": "assistant",
                    "content": f"{reflection}\n\n{respond_text}",
                }
                n_resp += 1

            messages = history + [target]
            out_rec = {
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "metadata": {
                    "task_id": tid, "trial": tr, "obs_idx": oi,
                    "kind": kind,
                    "attempt_used": att,
                    "flag_count": count_flags(chosen["flags"]),
                    "flags": chosen["flags"],
                    "alt_selection_diagnostic": chosen.get("alt_selection_diagnostic"),
                },
            }
            fout.write(json.dumps(out_rec) + "\n")
            n_total += 1

    sz = Path(args.out).stat().st_size
    print(f"reflection_sft: {n_total} records ({n_tool} tool + {n_resp} respond)")
    print(f"  attempt used:  {dict(attempt_used)}")
    print(f"  flag count per record after merge: {dict(sorted(flag_count_after.items()))}")
    print(f"  tools schemas embedded per record: {len(TOOL_SCHEMAS)}")
    print(f"  output: {args.out}  ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

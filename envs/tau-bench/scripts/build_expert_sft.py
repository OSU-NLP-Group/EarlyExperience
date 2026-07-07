"""Build expert_sft.jsonl from D_expert.jsonl.

One record per (obs, expert_action) pair across the 458 expert trajectories.
Native OpenAI Chat Completions format:
  - Top-level `messages` (multi-turn) + `tools` (16 retail tool schemas, identical
    across records — qwen2_tool / hermes2_tool / llama3_tool-style chat templates
    render this into the prompt in their own canonical location, matching exactly
    what tau-bench's eval loop will see at inference time).
  - LiteLLM-specific metadata fields on each message (`function_call`,
    `reasoning_content`, `provider_specific_fields`) are stripped — they're not
    OpenAI-canonical, and the chat templates we care about don't read them.
  - Assistant tool-only turns keep `content=""` and `tool_calls=[...]` —
    qwen2_tool template's if/elif logic reads tool_calls and skips content,
    so this is the cleanest paper-faithful shape.

One target drop: task 316 / trial 1 / obs 10 was a DeepSeek API anomaly during
expert collection that returned an empty assistant with NO content and NO
tool_calls. Using that as a training target would teach the model to emit
nothing in that situation. Dropped.

Run:
    PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output python \\
        envs/tau-bench/scripts/build_expert_sft.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau_bench.envs.retail.tools import ALL_TOOLS

D_EXPERT_DEFAULT = "envs/tau-bench/data/rollout/D_expert.jsonl"
OUT_DEFAULT = "envs/tau-bench/data/sft/expert_sft.jsonl"

# Tool schemas in OpenAI Chat Completions `tools` format. All 16 retail tools
# (including `think` and `transfer_to_human_agents`) because the env exposed
# all 16 during expert rollout — model needs to know they exist.
TOOL_SCHEMAS = [t.get_info() for t in ALL_TOOLS]


def clean_tool_call(tc: dict) -> dict:
    """Strip provider-specific fields, keep canonical OpenAI shape."""
    return {
        "id": tc["id"],
        "type": tc.get("type", "function"),
        "function": {
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
        },
    }


def clean_message(m: dict) -> dict:
    """Strip LiteLLM-specific metadata from a recorded message; keep only the
    canonical OpenAI Chat Completions fields."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-expert", default=D_EXPERT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_pairs, tool_n, resp_n, n_dropped = 0, 0, 0, 0
    msgs_total = 0
    with open(args.d_expert) as fin, open(args.out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            traj = r["traj"]
            for i, m in enumerate(traj):
                if m["role"] != "assistant":
                    continue
                # Drop malformed target: empty respond with no tool_calls.
                # 1 such case in D_expert (task 316 trial 1 pos 10 — DeepSeek
                # API anomaly during expert rollout).
                if not m.get("content") and not m.get("tool_calls"):
                    n_dropped += 1
                    continue
                msgs = [clean_message(x) for x in traj[: i + 1]]
                rec = {
                    "messages": msgs,
                    "tools": TOOL_SCHEMAS,
                    "metadata": {
                        "task_id": r["task_id"],
                        "trial": r["trial"],
                        "obs_idx": i,
                        "kind": "tool" if m.get("tool_calls") else "respond",
                    },
                }
                fout.write(json.dumps(rec) + "\n")
                n_pairs += 1
                msgs_total += len(msgs)
                if m.get("tool_calls"):
                    tool_n += 1
                else:
                    resp_n += 1

    sz = Path(args.out).stat().st_size
    print(f"expert_sft: {n_pairs} pairs ({tool_n} tool + {resp_n} respond)")
    print(f"  dropped (malformed empty target): {n_dropped}")
    print(f"  avg messages/record: {msgs_total/n_pairs:.1f}")
    print(f"  tools schemas embedded per record: {len(TOOL_SCHEMAS)}")
    print(f"  output: {args.out}  ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

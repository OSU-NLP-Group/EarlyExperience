"""Build expert_sft.jsonl from AgentTraj-L's textcraft_train.json.

Per NOTES.md "Approved decisions" (user-confirmed 2026-05-25):
  - D_expert = all 374 trajectories (no replay-based filter).
  - Expert assistant content keeps AgentTraj-L's `Thought:\n...\n\nAction:\n...`
    text (CoT preserved).
  - One SFT record per trajectory; multi-turn chat structure (matches
    sciworld's expert_sft pattern; iwm_sft / reflection_sft will be
    single-turn).

Output schema (one line per trajectory):
    {"messages": [
        {"role": "system",    "content": "<AgentGym REACT system prompt>"},
        {"role": "user",      "content": "Instruction:\\n<task + commands + goal>"},
        {"role": "assistant", "content": "Thought:\\n<thought>\\n\\nAction:\\n<action>"},
        {"role": "user",      "content": "Instruction:\\n<env response>"},
        ...
    ]}

The AgentTraj-L "OK, I'll follow your instructions..." ack turn (turn 1,
gpt loss=False) is dropped — it carries no training signal and the
canonical chat format starts straight from (system, user, assistant).

We use AgentTraj-L's data verbatim (no env replay). Rationale:
  - 371/374 traj are env-consistent (env's current commands list differs
    only in distractors, which experts don't reference).
  - For the 3 gold-nugget-drift traj (kept per user decision), AgentTraj-L's
    recorded env responses are internally consistent with the expert's
    Thoughts (back when gold_nugget WAS a base item), while live replay
    would produce env responses inconsistent with those Thoughts. Verbatim
    preserves Thought-vs-response coherence, which is the more important
    property for training.

Run (from workspace root):
    conda run -n agentenv-textcraft --no-capture-output python \\
        envs/textcraft/scripts/build_expert_sft.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAW_DEFAULT = "envs/textcraft/data/raw/textcraft_train.json"
OUT_DEFAULT = "envs/textcraft/data/sft/expert_sft.jsonl"


def build_one(rec: dict) -> dict:
    """Convert one AgentTraj-L record to chat-messages format."""
    conv = rec["conversations"]
    # Sanity: expect canonical 4-turn handshake at the start
    assert conv[0]["from"] == "human" and conv[0].get("loss") is None, (
        f"unexpected turn 0 in {rec['item_id']}"
    )
    assert conv[1]["from"] == "gpt" and conv[1].get("loss") is False, (
        f"unexpected turn 1 in {rec['item_id']}"
    )
    assert conv[2]["from"] == "human" and conv[2].get("loss") is None, (
        f"unexpected turn 2 in {rec['item_id']}"
    )

    messages = [
        {"role": "system", "content": conv[0]["value"]},
        {"role": "user", "content": conv[2]["value"]},
    ]
    for msg in conv[3:]:
        role = "assistant" if msg["from"] == "gpt" else "user"
        messages.append({"role": role, "content": msg["value"]})
    return {"messages": messages}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--output", default=OUT_DEFAULT)
    args = ap.parse_args()

    raw_path = Path(args.raw).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"raw    : {raw_path}")
    print(f"output : {out_path}")

    with open(raw_path) as f:
        data = json.load(f)
    print(f"loaded {len(data)} trajectories")

    # Sanity stats before writing
    turn_counts = [len(r["conversations"]) for r in data]
    print(
        f"  turn counts: min={min(turn_counts)} median={sorted(turn_counts)[len(turn_counts)//2]} max={max(turn_counts)}"
    )

    n_written = 0
    n_asst = 0
    n_user = 0
    with open(out_path, "w") as out:
        for rec in data:
            sft = build_one(rec)
            for m in sft["messages"]:
                if m["role"] == "assistant":
                    n_asst += 1
                elif m["role"] == "user":
                    n_user += 1
            out.write(json.dumps(sft, ensure_ascii=False) + "\n")
            n_written += 1

    bytes_out = out_path.stat().st_size
    print(f"\nDONE.")
    print(f"  records written: {n_written}")
    print(f"  assistant turns: {n_asst} (these get loss in SFT training)")
    print(f"  user turns:      {n_user}")
    print(f"  output size:     {bytes_out:,} bytes")


if __name__ == "__main__":
    main()

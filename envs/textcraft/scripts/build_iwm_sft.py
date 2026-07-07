"""Build iwm_sft.jsonl from raw textcraft_train.json + replay_full.jsonl + iwm_rollout.jsonl.

For each trajectory in D_expert (all 374, no filter — user-decided 2026-05-25),
emit per expert state s_i:
  - 1 expert IWM record: (s_i, expert_action_i, env_response_to_expert)
    where env_response comes from replay_full.steps[i].observation (current
    env's actual response, not AgentTraj-L's historical recording)
  - K_actual alt IWM records: (s_i, alt_j, env_response_to_alt)
    where alt_j comes from iwm_rollout.jsonl

Output schema (single-turn chat-messages, one line per IWM record):
  system    : AgentGym REACT system prompt (from raw[0].conversations[0].value)
  user      : <task obs with "Instruction:\\n" prefix>\\n\\n
              <prior expert Thought:\\n...\\n\\nAction:\\n... verbatim>\\n\\n
              Instruction:\\n<env response>\\n\\n   # repeated for each prior step
              Action:\\n<action being probed>
  assistant : <env's response after this action>  # the IWM target

Notes on "what goes in the prompt vs. target":
  - History uses raw Thought + raw Action from AgentTraj-L verbatim (same as
    expert_sft's assistant content) — this is what the model emits at
    inference. Env responses in history are REPLAYED (current env's actual
    behavior, matching what inference will see), not AgentTraj-L's
    recordings.
  - For the action being probed: expert step uses replay_full's cleaned
    action (the actual string env stepped on); alt uses iwm_rollout's alt
    string (already cleaned by admissible.py).
  - Target = single string the env returned (matches paper §3: raw or
    summarized next state; TextCraft observations are 1-line strings, no
    summarizer needed).

Why single-turn: the IWM target (next-state) would be a `user` turn in a
multi-turn agent chat, but SFT trainers compute loss only on `assistant`
turns. Single-turn (system, user, assistant) puts the target as
`assistant` so the trainer learns it without ambiguity. Matches sciworld
iwm_sft's pattern.

Run (from workspace root):
    conda run -n agentenv-textcraft --no-capture-output python \\
        envs/textcraft/scripts/build_iwm_sft.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RAW_DEFAULT = "envs/textcraft/data/raw/textcraft_train.json"
REPLAY_DEFAULT = "envs/textcraft/data/replay/replay_full.jsonl"
ROLLOUT_DEFAULT = "envs/textcraft/data/rollout/iwm_rollout.jsonl"
OUTPUT_DEFAULT = "envs/textcraft/data/sft/iwm_sft.jsonl"


def render_history(raw_conversations: list, replayed_observations: list, up_to_step: int) -> str:
    """Render the trajectory's conversation up to (not including) `up_to_step`.

    raw_conversations: textcraft_train.json record's `conversations` list.
        Index 0 = system; 1 = "OK..." ack; 2 = task obs (user, with
        "Instruction:\\n" prefix); 3, 5, ... = gpt action turns (Thought +
        Action verbatim); 4, 6, ... = env response turns.
    replayed_observations: list of env's CURRENT response strings, one per
        step (from replay_full.steps[i].observation).
    up_to_step: 0 means the prompt is just the task obs; i means include
        the conversation through env's response to step i-1.
    """
    pieces = [raw_conversations[2]["value"]]  # task obs (already has "Instruction:\n" prefix)
    for i in range(up_to_step):
        gpt_turn_idx = 2 * i + 3
        if gpt_turn_idx >= len(raw_conversations):
            break
        # Raw "Thought:\n... \n\nAction:\n..." verbatim — same as expert_sft.
        pieces.append(raw_conversations[gpt_turn_idx]["value"])
        # Env response: REPLAYED (current env), wrapped with "Instruction:\n".
        if i < len(replayed_observations):
            pieces.append(f"Instruction:\n{replayed_observations[i]}")
    return "\n\n".join(pieces)


def build_user_content(history_text: str, action_to_probe: str) -> str:
    return f"{history_text}\n\nAction:\n{action_to_probe.strip()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--replay", default=REPLAY_DEFAULT)
    ap.add_argument("--rollout", default=ROLLOUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    args = ap.parse_args()

    raw_path = Path(args.raw).resolve()
    replay_path = Path(args.replay).resolve()
    rollout_path = Path(args.rollout).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"raw     : {raw_path}")
    print(f"replay  : {replay_path}")
    print(f"rollout : {rollout_path}")
    print(f"output  : {out_path}")

    # Load raw textcraft_train.json
    print("loading raw textcraft_train.json...")
    with open(raw_path) as f:
        raw = json.load(f)
    raw_by_iid = {r["item_id"]: r for r in raw}
    system_prompt = raw[0]["conversations"][0]["value"]
    print(f"  {len(raw)} trajectories; system prompt {len(system_prompt)} chars")

    # Load replay_full.jsonl
    print("loading replay_full.jsonl...")
    replay_by_iid = {}
    for line in open(replay_path):
        r = json.loads(line)
        replay_by_iid[r["item_id"]] = r
    print(f"  {len(replay_by_iid)} trajectories")

    # Load iwm_rollout.jsonl
    print("loading iwm_rollout.jsonl...")
    rollout_alts: dict[tuple[str, int], list[dict]] = defaultdict(list)
    rollout_initial: dict[str, dict] = {}
    for line in open(rollout_path):
        r = json.loads(line)
        if r["kind"] == "initial":
            rollout_initial[r["item_id"]] = r
        elif r["kind"] == "alt":
            rollout_alts[(r["item_id"], r["step"])].append(r)
    print(f"  {len(rollout_initial)} initial records, "
          f"{sum(len(v) for v in rollout_alts.values())} alt records")

    # Write iwm_sft.jsonl
    print(f"writing {out_path}...")
    n_expert = 0
    n_alt = 0
    n_traj = 0
    with open(out_path, "w") as fout:
        for iid in sorted(replay_by_iid.keys(), key=lambda x: int(x.split("_")[1])):
            replay = replay_by_iid[iid]
            if iid not in raw_by_iid:
                print(f"  WARN: {iid} not in raw (skipping)")
                continue
            if iid not in rollout_initial:
                print(f"  WARN: {iid} not in rollout (skipping)")
                continue
            conv = raw_by_iid[iid]["conversations"]
            replayed_obs = [s["observation"] for s in replay["steps"]]
            actions_clean = [s["action"] for s in replay["steps"]]
            n_steps = len(replay["steps"])

            for i in range(n_steps):
                history = render_history(conv, replayed_obs, up_to_step=i)

                # Expert IWM record at this step
                expert_user = build_user_content(history, actions_clean[i])
                expert_target = replayed_obs[i]
                fout.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": expert_user},
                                {"role": "assistant", "content": expert_target},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_expert += 1

                # Alt IWM records at this step
                for alt in rollout_alts.get((iid, i), []):
                    alt_user = build_user_content(history, alt["action"])
                    fout.write(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": alt_user},
                                    {"role": "assistant", "content": alt["next_state"]},
                                ]
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_alt += 1
            n_traj += 1

    bytes_out = out_path.stat().st_size
    print(f"\nDONE.")
    print(f"  trajectories : {n_traj}")
    print(f"  expert IWM   : {n_expert}")
    print(f"  alt IWM      : {n_alt}")
    print(f"  total IWM    : {n_expert + n_alt}")
    print(f"  output       : {out_path}  ({bytes_out:,} bytes)")


if __name__ == "__main__":
    main()

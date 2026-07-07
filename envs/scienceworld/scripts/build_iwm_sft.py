"""Build iwm_sft.jsonl from replay_full.jsonl + iwm_rollout.jsonl.

For each surviving trajectory in D_expert, emit K+1 IWM SFT records per
expert state s_i:
  - 1 record for the expert (s_i, a_i, s_{i+1}) — pulled from replay_full
  - K=3 records for the alternatives (s_i, alt_j, s_i^j) — pulled from rollout

Output schema (one line per IWM record, single-turn chat-messages):
  system    : AgentGym REACT system prompt (same prompt expert_sft uses)
  user      : <task_desc>\\n<initial_obs>\\n<prior expert Thought/Action +
              env-obs history rendered inline>\\n\\nAction:\\n<action>
  assistant : <env's response after the action>

The user content is rendered as plain text so a single-turn SFT trainer
treats it as one prompt, with loss applied only to the assistant content
(the next-state prediction). This avoids the multi-turn loss-masking
ambiguity that would arise from putting the next-state as a `user` turn.

Run (from workspace root):
    conda run -n agentenv-sciworld --no-capture-output python \\
        envs/scienceworld/scripts/build_iwm_sft.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPLAY_DEFAULT = "envs/scienceworld/data/replay/replay_full.jsonl"
ROLLOUT_DEFAULT = "envs/scienceworld/data/rollout/iwm_rollout.jsonl"
RAW_AGENTTRAJ_DEFAULT = "envs/scienceworld/data/raw/sciworld_train.json"
OUTPUT_DEFAULT = "envs/scienceworld/data/sft/iwm_sft.jsonl"


def render_history(task_desc: str, initial_obs: str,
                   thoughts: list[str], actions: list[str], observations: list[str],
                   up_to_step: int) -> str:
    """Render the trajectory's conversation up to (not including) `up_to_step`.

    The rendering matches the natural AgentGym REACT conversation flow as
    plain text:

        {task_desc}
        {initial_obs}

        Thought:
        {thought_0}

        Action:
        {action_0}

        {observation_after_action_0}

        Thought:
        {thought_1}

        Action:
        {action_1}

        {observation_after_action_1}

        ...

        {observation_at_state_s_i}  ← the state we're about to act from

    If a step's thought is empty (rare — would only happen if AgentTraj-L
    didn't have a recoverable Thought:), the "Thought:" block is omitted
    for that step.
    """
    pieces = [task_desc.rstrip(), initial_obs.rstrip()]
    for i in range(up_to_step):
        thought = thoughts[i].strip() if i < len(thoughts) and thoughts[i] else ""
        action = actions[i].strip() if i < len(actions) else ""
        if thought:
            pieces.append("")
            pieces.append("Thought:")
            pieces.append(thought)
        pieces.append("")
        pieces.append("Action:")
        pieces.append(action)
        pieces.append("")
        if i < len(observations):
            pieces.append(observations[i].rstrip())
    return "\n".join(pieces)


def build_user_content(history_text: str, action_to_probe: str) -> str:
    return f"{history_text}\n\nAction:\n{action_to_probe.strip()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", default=REPLAY_DEFAULT)
    ap.add_argument("--rollout", default=ROLLOUT_DEFAULT)
    ap.add_argument("--raw-agenttraj", default=RAW_AGENTTRAJ_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    args = ap.parse_args()

    replay_path = Path(args.replay).resolve()
    rollout_path = Path(args.rollout).resolve()
    raw_path = Path(args.raw_agenttraj).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Canonical system prompt from any AgentTraj-L turn 0 (same as expert_sft).
    print(f"system prompt: extracting from {raw_path}...")
    with open(raw_path) as f:
        raw = json.load(f)
    system_prompt = raw[0]["conversations"][0]["value"]
    del raw
    print(f"  length: {len(system_prompt)} chars")

    # Read rollout: initial_obs per trajectory + alt records keyed by (iid, step).
    print(f"loading rollout from {rollout_path}...")
    initial_obs_by_iid: dict[str, str] = {}
    alts_by_iid_step: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for line in open(rollout_path):
        r = json.loads(line)
        if r["kind"] == "initial":
            initial_obs_by_iid[r["item_id"]] = r["initial_obs"]
        elif r["kind"] == "alt":
            alts_by_iid_step[(r["item_id"], r["step"])].append(r)
    print(f"  trajectories with initial_obs: {len(initial_obs_by_iid)}")
    print(f"  total alt records             : {sum(len(v) for v in alts_by_iid_step.values())}")

    # Render. Stream replay_full and emit SFT lines.
    print(f"writing {out_path}...")
    n_traj_seen = 0
    n_traj_written = 0
    n_expert_records = 0
    n_alt_records = 0

    with open(out_path, "w") as fout:
        for line in open(replay_path):
            rec = json.loads(line)
            n_traj_seen += 1
            iid = rec["item_id"]
            if iid not in initial_obs_by_iid:
                continue  # not in the filtered D_expert

            initial_obs = initial_obs_by_iid[iid]
            thoughts = rec["agenttraj_thoughts"]
            actions = rec["agenttraj_actions"]
            steps = rec["replay_steps"]
            observations = [s["observation"] for s in steps]
            if not steps:
                continue
            task_desc = steps[0]["info"].get("taskDesc", "")
            # Sanity: task_desc should not be empty.
            if not task_desc:
                # Fall back to whatever happens to be in info.taskDesc anywhere.
                for s in steps:
                    if s.get("info", {}).get("taskDesc"):
                        task_desc = s["info"]["taskDesc"]
                        break

            n_traj_written += 1
            n_steps = len(steps)
            for i in range(n_steps):
                history_text = render_history(
                    task_desc, initial_obs, thoughts, actions, observations, up_to_step=i
                )

                # Expert record at this step
                expert_user = build_user_content(history_text, actions[i])
                expert_next = observations[i]  # s_{i+1}
                fout.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": expert_user},
                                {"role": "assistant", "content": expert_next},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_expert_records += 1

                # Alternative records at this step
                for alt in alts_by_iid_step.get((iid, i), []):
                    alt_user = build_user_content(history_text, alt["action"])
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
                    n_alt_records += 1

    print(f"\nDONE.")
    print(f"  trajectories seen (in replay)  : {n_traj_seen}")
    print(f"  trajectories written (filtered): {n_traj_written}")
    print(f"  expert IWM records             : {n_expert_records}")
    print(f"  alternative IWM records        : {n_alt_records}")
    print(f"  total IWM records              : {n_expert_records + n_alt_records}")
    print(f"  output                         : {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

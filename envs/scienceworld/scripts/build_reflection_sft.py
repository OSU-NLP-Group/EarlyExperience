"""Build reflection_sft.jsonl from sr_rollout.jsonl + replay_full.jsonl.

For each kept SR record this writes one single-turn chat-messages line:
  system    : AgentGym REACT system prompt (same as expert_sft / iwm_sft)
  user      : task_desc + initial_obs + (prior expert Thought/Action +
              env-obs history rendered inline) + current state s_i
  assistant : "Thought:\\n<reflection>\\n\\nAction:\\n<expert_action>"

Filtering applied (per the §3-style decisions logged in NOTES.md):
  - skip k_final < 3 (incomplete pipeline output)
  - skip records where the proposer error or reflection error was set
  - dedup "doubled" reflections: when the LLM mode-collapsed and wrote the
    same monologue twice in one response, keep just the first copy
  - skip records flagged as doubled but where the two copies aren't a
    clean byte-identical pair (unsafe to auto-dedup)
  - skip records containing any banned-vocabulary word
    (expert / selected action / chosen action / correct choice / right
    action / best option / optimal action / best alternative — these
    leak the supervision label into what should be the agent's own
    internal monologue)
  - skip records containing numbered alternative labels
    ("Action 1" / "Action 2" / "Alternative 1" / "a_i^1")
  - replace any internal "\\n\\n" with " " (paragraph-break normalization)

Run:
    conda run -n agentenv-sciworld --no-capture-output python \\
        envs/scienceworld/scripts/build_reflection_sft.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_REPLAY = "envs/scienceworld/data/replay/replay_full.jsonl"
DEFAULT_ROLLOUT_IWM = "envs/scienceworld/data/rollout/iwm_rollout.jsonl"
DEFAULT_SR_ROLLOUT = "envs/scienceworld/data/rollout/sr_rollout.jsonl"
DEFAULT_RAW_AGENTTRAJ = "envs/scienceworld/data/raw/sciworld_train.json"
DEFAULT_OUTPUT = "envs/scienceworld/data/sft/reflection_sft.jsonl"

BANNED_WORDS = (
    "expert",
    "selected action",
    "chosen action",
    "correct choice",
    "right action",
    "best option",
    "optimal action",
    "best alternative",
)
NUMBERED_LABEL_RE = re.compile(
    r"\b(Action [1-9]|Alternative [1-9]|a_i\^[1-9])\b"
)


def dedupe_doubled(text: str) -> tuple[str, str]:
    """Return (deduped_text, status). status is one of:
      'unchanged'      : not duplicated
      'deduped'        : was duplicated, first copy preserved
      'unsafe_dup'     : was duplicated but the two copies aren't a clean
                         byte-identical pair (length diff > 50). Skip this
                         record rather than risk a wrong dedup.
    """
    if len(text) < 200:
        return text, "unchanged"
    fp = text[:100]
    idx = text.find(fp, 100)
    if idx == -1:
        return text, "unchanged"
    first = text[:idx].rstrip()
    second = text[idx:].rstrip()
    if abs(len(first) - len(second)) > 50:
        return text, "unsafe_dup"
    return first, "deduped"


def strip_paragraph_breaks(s: str) -> str:
    """Replace any run of consecutive newlines with a single space, then
    collapse remaining single newlines and runs of whitespace to one space.
    The SR prompt asked for one continuous paragraph but ~27% of records
    still contain "\\n\\n"; normalizing here keeps the training target
    structurally uniform without losing content."""
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def render_history(
    task_desc: str,
    initial_obs: str,
    thoughts: list[str],
    actions: list[str],
    observations: list[str],
    up_to_step: int,
) -> str:
    """Same renderer as build_iwm_sft.py: lay out the trajectory's prior
    Thought/Action/Obs turns as plain text up to (and not including) step
    `up_to_step`. The final piece is the env observation that the agent is
    about to act from (state s_i)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", default=DEFAULT_REPLAY)
    ap.add_argument("--iwm-rollout", default=DEFAULT_ROLLOUT_IWM,
                    help="Used only to pick up env.look() initial_obs per trajectory.")
    ap.add_argument("--sr-rollout", default=DEFAULT_SR_ROLLOUT)
    ap.add_argument("--raw-agenttraj", default=DEFAULT_RAW_AGENTTRAJ,
                    help="Used only to extract the canonical AgentGym REACT system prompt.")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    replay_path = Path(args.replay).resolve()
    iwm_path = Path(args.iwm_rollout).resolve()
    sr_path = Path(args.sr_rollout).resolve()
    raw_path = Path(args.raw_agenttraj).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"replay      : {replay_path}")
    print(f"iwm_rollout : {iwm_path}")
    print(f"sr_rollout  : {sr_path}")
    print(f"output      : {out_path}")

    # System prompt
    with open(raw_path) as f:
        raw0 = json.load(f)[0]
    system_prompt = raw0["conversations"][0]["value"]
    print(f"system prompt length: {len(system_prompt)} chars")

    # Build a {item_id: trajectory_record} index from replay_full.
    print("loading replay_full into memory (filter pass)...")
    traj_by_iid: dict[str, dict] = {}
    for line in open(replay_path):
        rec = json.loads(line)
        if rec.get("final_done") and rec.get("final_score") == 100:
            traj_by_iid[rec["item_id"]] = rec
    print(f"  surviving trajectories: {len(traj_by_iid)}")

    # Initial-obs per trajectory from iwm_rollout
    print("loading initial_obs from iwm_rollout...")
    initial_obs_by_iid: dict[str, str] = {}
    for line in open(iwm_path):
        r = json.loads(line)
        if r.get("kind") == "initial":
            initial_obs_by_iid[r["item_id"]] = r["initial_obs"]
    print(f"  initial_obs records: {len(initial_obs_by_iid)}")

    # Stream sr_rollout and write the filtered SFT lines.
    print("processing sr_rollout...")
    n_in = 0
    skip_kfinal = 0
    skip_error = 0
    skip_unsafe_dup = 0
    skip_banned = 0
    skip_numbered = 0
    skip_no_trajectory = 0
    deduped_count = 0
    paragraph_stripped = 0
    n_out = 0

    with open(out_path, "w") as fout:
        for line in open(sr_path):
            r = json.loads(line)
            n_in += 1

            # 1. Skip if pipeline failed or returned <3 alts.
            if r.get("k_final", 0) < 3:
                skip_kfinal += 1
                continue
            if r.get("reflection_error") or r.get("proposer_error"):
                skip_error += 1
                continue
            cot = r.get("reflection_cot") or ""
            if not cot:
                skip_error += 1
                continue

            # 2. Dedup doubled reflections.
            new_cot, dup_status = dedupe_doubled(cot)
            if dup_status == "unsafe_dup":
                skip_unsafe_dup += 1
                continue
            if dup_status == "deduped":
                deduped_count += 1
                cot = new_cot

            # 3. Banned-word filter (after dedup, in case the duplicate
            #    half contained a leak we'd already discarded).
            cot_lower = cot.lower()
            if any(w in cot_lower for w in BANNED_WORDS):
                skip_banned += 1
                continue
            if NUMBERED_LABEL_RE.search(cot):
                skip_numbered += 1
                continue

            # 4. Paragraph-break normalization.
            if "\n\n" in cot:
                paragraph_stripped += 1
            cot = strip_paragraph_breaks(cot)

            # 5. Build the user history.
            iid = r["item_id"]
            step_i = r["step"]
            traj = traj_by_iid.get(iid)
            if traj is None:
                skip_no_trajectory += 1
                continue
            initial_obs = initial_obs_by_iid.get(iid, "")
            thoughts = traj["agenttraj_thoughts"]
            actions = traj["agenttraj_actions"]
            replay_steps = traj["replay_steps"]
            observations = [s["observation"] for s in replay_steps]
            task_desc = replay_steps[0]["info"].get("taskDesc", "")

            user_content = render_history(
                task_desc, initial_obs, thoughts, actions, observations, up_to_step=step_i
            )

            # 6. Assemble the assistant content: reflection (in Thought:) + Action.
            expert_action = r["expert_action"].strip()
            assistant_content = f"Thought:\n{cot}\n\nAction:\n{expert_action}"

            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_out += 1

    print()
    print(f"=== build summary ===")
    print(f"  raw rollout records read       : {n_in}")
    print(f"  skipped (k_final < 3)          : {skip_kfinal}")
    print(f"  skipped (proposer/reflection err or empty cot): {skip_error}")
    print(f"  skipped (unsafe duplicate)     : {skip_unsafe_dup}")
    print(f"  skipped (banned-word leak)     : {skip_banned}")
    print(f"  skipped (numbered-label leak)  : {skip_numbered}")
    print(f"  skipped (trajectory not in D_expert): {skip_no_trajectory}")
    print(f"  dedup-recovered (kept 1st copy): {deduped_count}")
    print(f"  paragraph-stripped              : {paragraph_stripped}")
    print(f"  final reflection_sft records   : {n_out}")
    print(f"  output file size               : {out_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

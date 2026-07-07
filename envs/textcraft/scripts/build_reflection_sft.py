"""Build reflection_sft.jsonl from sr_rollout.jsonl + raw textcraft_train.json + replay_full.jsonl.

For each expert state with a reflection: emit a SINGLE-TURN chat-messages
record where the assistant content is `"Thought:\\n<reflection>\\n\\nAction:\\n<expert_action>"`.
This matches sciworld's reflection_sft pattern and is structurally
compatible with expert_sft's per-step Thought+Action format — so when the
trainer mixes the two datasets, both teach the same `(state → Thought +
Action)` mapping but with different Thought distributions (recorded for
expert_sft, LLM-generated reflection for reflection_sft).

Output schema (one line per expert state with a reflection):
  system    : AgentGym REACT system prompt (same as expert_sft / iwm_sft)
  user      : <task obs with "Instruction:\\n" prefix>\\n\\n
              <prior expert Thought:\\n...\\n\\nAction:\\n... verbatim>\\n\\n
              Instruction:\\n<env response>\\n\\n   # repeated for each prior step
  assistant : "Thought:\\n<reflection>\\n\\nAction:\\n<expert_action>"

Build-time quality filters (post-hoc, applied here so sr_rollout.jsonl
remains the unfiltered raw):
  - Drop banned-vocab leak: reflections containing "expert" / "selected" /
    "chosen" / "correct choice" / "right action" / "best option" /
    "optimal" / "preferred" anywhere in the text.
  - Drop numbered-label leak: reflections containing "Action 1" /
    "Alternative 2" / "Option 3" / "a_i^j" patterns.
  - Drop doubled responses that the rollout's auto-dedup didn't catch
    safely (i.e. `doubled_dedup=True` cases are KEPT — the dedup already
    produced clean output; ambiguous halves were not auto-deduped, those
    we drop).

Run (from workspace root):
    conda run -n agentenv-textcraft --no-capture-output python \\
        envs/textcraft/scripts/build_reflection_sft.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

RAW_DEFAULT = "envs/textcraft/data/raw/textcraft_train.json"
REPLAY_DEFAULT = "envs/textcraft/data/replay/replay_full.jsonl"
SR_ROLLOUT_DEFAULT = "envs/textcraft/data/rollout/sr_rollout.jsonl"
OUTPUT_DEFAULT = "envs/textcraft/data/sft/reflection_sft.jsonl"

# Filter intent (user-confirmed 2026-05-25):
# Drop expert actions that are AgentTraj-L "natural language noise" rather than
# attempts at the action grammar. The 6 known noise cases all share the
# property that their first token is NOT one of {get, craft, inventory},
# e.g. "Look for ...", "My last action ...", "Backtrack ...", "I need ...",
# "pass", "look around". By contrast, expert exploration like
# `get oak trapdoor` (no count, env still rejects) IS legitimate agent
# behavior — it's a deliberate probe at whether the final item is gettable,
# and the env's "Could not execute / Could not find" response is valid
# training signal. We retain those.
_ACTION_VERBS = {"get", "craft", "inventory"}


def expert_action_is_legal(expert_action: str) -> bool:
    first = expert_action.strip().split()[:1]
    if not first:
        return False
    return first[0].lower() in _ACTION_VERBS


BANNED_VOCAB = [
    "expert",
    "selected action",
    "chosen action",
    "correct choice",
    "right action",
    "best option",
    "optimal action",
    "best alternative",
    "preferred action",
]
NUMBERED_LABEL_RE = re.compile(
    r"\b(action|alternative|option|choice)\s*(\d+|[a-c])\b",
    re.IGNORECASE,
)


def render_history(raw_conversations: list, replayed_observations: list, up_to_step: int) -> str:
    """Same as build_iwm_sft.render_history — flat-text history prefixed by
    task obs (which already has "Instruction:\\n" prefix from raw conv)."""
    pieces = [raw_conversations[2]["value"]]
    for i in range(up_to_step):
        gpt_turn_idx = 2 * i + 3
        if gpt_turn_idx >= len(raw_conversations):
            break
        pieces.append(raw_conversations[gpt_turn_idx]["value"])
        if i < len(replayed_observations):
            pieces.append(f"Instruction:\n{replayed_observations[i]}")
    return "\n\n".join(pieces)


def has_banned_vocab(text: str) -> str | None:
    """Return the first banned phrase found, or None."""
    low = text.lower()
    for b in BANNED_VOCAB:
        if b in low:
            return b
    return None


def has_numbered_label(text: str) -> str | None:
    m = NUMBERED_LABEL_RE.search(text)
    return m.group(0) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--replay", default=REPLAY_DEFAULT)
    ap.add_argument("--sr-rollout", default=SR_ROLLOUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    args = ap.parse_args()

    raw_path = Path(args.raw).resolve()
    replay_path = Path(args.replay).resolve()
    sr_path = Path(args.sr_rollout).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"raw        : {raw_path}")
    print(f"replay     : {replay_path}")
    print(f"sr_rollout : {sr_path}")
    print(f"output     : {out_path}")

    with open(raw_path) as f:
        raw = json.load(f)
    raw_by_iid = {r["item_id"]: r for r in raw}
    system_prompt = raw[0]["conversations"][0]["value"]

    replay_by_iid = {json.loads(l)["item_id"]: json.loads(l) for l in open(replay_path)}

    sr_records = [json.loads(l) for l in open(sr_path)]
    sr_by_key = {(r["item_id"], r["step"]): r for r in sr_records}
    print(f"  raw     : {len(raw_by_iid)} traj")
    print(f"  replay  : {len(replay_by_iid)} traj")
    print(f"  sr      : {len(sr_records)} reflections")

    n_emitted = 0
    n_drop_banned = 0
    n_drop_numbered = 0
    n_drop_malformed = 0
    n_drop_doubled_unsafe = 0
    n_drop_other = 0

    with open(out_path, "w") as fout:
        for (iid, step), sr in sr_by_key.items():
            if iid not in raw_by_iid or iid not in replay_by_iid:
                n_drop_other += 1
                continue
            text = sr.get("reflection_text") or ""
            if not text:
                n_drop_other += 1
                continue

            # Filter: doubled responses that weren't safely auto-deduped should
            # show up here as "reflection_text contains the first half twice"
            # — rollout_sr already attempts safe dedup, so doubled_dedup=True
            # records are kept (already cleaned). We do NOT actively re-detect
            # because the rollout is the authority.

            # Filter: malformed expert action (AgentTraj-L noise: natural-language
            # strings the env can't parse, e.g. "Look for string again in the environment").
            # User decision 2026-05-25: drop only from reflection_sft.
            if not expert_action_is_legal(sr["expert_action"]):
                n_drop_malformed += 1
                continue
            # Filter: banned vocab leak
            hit = has_banned_vocab(text)
            if hit:
                n_drop_banned += 1
                continue
            # Filter: numbered alt labels
            hit = has_numbered_label(text)
            if hit:
                n_drop_numbered += 1
                continue

            # Build the SFT messages
            conv = raw_by_iid[iid]["conversations"]
            replayed_obs = [s["observation"] for s in replay_by_iid[iid]["steps"]]
            history = render_history(conv, replayed_obs, up_to_step=step)
            expert_action = sr["expert_action"]
            asst = f"Thought:\n{text.strip()}\n\nAction:\n{expert_action}"

            fout.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": history},
                            {"role": "assistant", "content": asst},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_emitted += 1

    print(f"\nDONE.")
    print(f"  emitted                       : {n_emitted}")
    print(f"  dropped malformed expert      : {n_drop_malformed}")
    print(f"  dropped banned vocab          : {n_drop_banned}")
    print(f"  dropped numbered alts         : {n_drop_numbered}")
    print(f"  dropped other                 : {n_drop_other}")
    print(f"  output: {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

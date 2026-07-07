"""
Build expert_sft_text.jsonl + expert_sft_fc.jsonl.

For each emit step (fcall AND text_only) in each case in expert_ids:
  - text version: assistant content = "call_str" or "[call_a, call_b]" or natural-language text
  - fc version  : assistant content=null + tool_calls=[...]  (for fcall step)
                  assistant content=text                       (for text_only step)

Run:
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/build_expert_sft.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sft_common import (  # noqa: E402
    REPO, build_expert_system_prompt, calls_to_fc_tool_calls, calls_to_text_block,
    load_expert_ids, load_parsed_steps_by_case, load_tool_schemas,
    merge_consecutive_user_messages,
    render_history_messages_fc, render_history_messages_text, write_jsonl,
)

OUT_DIR = REPO / "envs/bfcl_v4/data/sft"


def main() -> int:
    expert_ids = load_expert_ids()
    grouped = load_parsed_steps_by_case()

    text_records = []
    fc_records = []
    n_fcall = n_text = 0

    for cid in sorted(expert_ids, key=lambda s: int(s.rsplit("_", 1)[-1])):
        steps = grouped.get(cid)
        if not steps:
            continue
        involved = steps[0]["involved_classes"]
        schemas = load_tool_schemas(involved)
        system_content = build_expert_system_prompt(schemas)

        for i, step in enumerate(steps):
            prior = steps[:i]
            current_user = step.get("user_msg_for_turn")

            # build base messages (system + history + current user)
            text_msgs = [{"role": "system", "content": system_content}] + \
                        render_history_messages_text(prior, current_user)
            fc_msgs = [{"role": "system", "content": system_content}] + \
                      render_history_messages_fc(prior, current_user)

            base = {
                "case_id": cid,
                "turn_idx": step["turn_idx"],
                "step_idx": step["step_idx"],
                "global_emit_idx": step["global_emit_idx"],
                "involved_classes": involved,
                "step_type": step["step_type"],
            }

            if step["step_type"] == "function_call":
                n_fcall += 1
                calls = step["expert_emit_decoded"]
                # text variant
                tr = dict(base, messages=merge_consecutive_user_messages(text_msgs + [
                    {"role": "assistant", "content": calls_to_text_block(calls)}
                ]))
                text_records.append(tr)
                # fc variant
                fr = dict(base, messages=merge_consecutive_user_messages(fc_msgs + [
                    {"role": "assistant", "content": None,
                     "tool_calls": calls_to_fc_tool_calls(calls, step["turn_idx"], step["step_idx"])}
                ]))
                fc_records.append(fr)

            elif step["step_type"] == "text_only":
                n_text += 1
                t = step["expert_emit_raw"]
                if isinstance(t, list) and t and isinstance(t[0], str):
                    t = t[0]
                t = str(t)
                # both formats: assistant content is the text
                tr = dict(base, messages=merge_consecutive_user_messages(text_msgs + [
                    {"role": "assistant", "content": t}
                ]))
                fr = dict(base, messages=merge_consecutive_user_messages(fc_msgs + [
                    {"role": "assistant", "content": t}
                ]))
                text_records.append(tr)
                fc_records.append(fr)
            else:
                # skip empty_emit (shouldn't occur in Opus FC, but safe)
                continue

    write_jsonl(OUT_DIR / "expert_sft_text.jsonl", text_records)
    write_jsonl(OUT_DIR / "expert_sft_fc.jsonl", fc_records)

    print(f"expert_sft built:")
    print(f"  text records: {len(text_records)}  → {OUT_DIR/'expert_sft_text.jsonl'}")
    print(f"  fc   records: {len(fc_records)}  → {OUT_DIR/'expert_sft_fc.jsonl'}")
    print(f"  breakdown   : fcall={n_fcall}, text_only={n_text}, total={n_fcall+n_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Build iwm_sft_text.jsonl + iwm_sft_fc.jsonl.

For each function_call state in iwm_full_summarized.jsonl, emit:
  - 1 expert IWM record per expert call (with its summary as target)
  - 1 alt IWM record per valid alt    (with its summary as target)

The IWM training target is the SUMMARY (next-state in natural language).
Per METHOD.md §3: "L_IWM = -Σ log p(s_i^j | s_i, a_i^j)".

User content = system + tools + history-up-to-s_i + the probe action.
Assistant content = the summary.

Run:
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/build_iwm_sft.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sft_common import (  # noqa: E402
    REPO, build_iwm_system_prompt, calls_to_fc_tool_calls, calls_to_text_block,
    load_tool_schemas, merge_consecutive_user_messages,
    render_history_messages_fc, render_history_messages_text, write_jsonl,
)

IN = REPO / "envs/bfcl_v4/data/rollout/iwm_full_summarized.jsonl"
OUT_DIR = REPO / "envs/bfcl_v4/data/sft"


def main() -> int:
    # group fcall records by case_id, ordered by global_emit_idx, for history rendering
    by_case: dict[str, list[dict]] = defaultdict(list)
    with IN.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_case[r["case_id"]].append(r)
    for cid in by_case:
        by_case[cid].sort(key=lambda r: r["global_emit_idx"])

    text_records = []
    fc_records = []
    n_expert = n_alt = n_alt_skipped = 0

    for cid in sorted(by_case.keys(), key=lambda s: int(s.rsplit("_", 1)[-1])):
        case_recs = by_case[cid]
        involved = case_recs[0]["involved_classes"]
        schemas = load_tool_schemas(involved)
        system_content = build_iwm_system_prompt(schemas)

        for r in case_recs:
            # prior fcall records — note: text_only steps NOT in this jsonl,
            # so history may skip text_only contextual messages. acceptable: text_only
            # doesn't affect environment dynamics, so IWM context isn't lost.
            prior = [pr for pr in case_recs if pr["global_emit_idx"] < r["global_emit_idx"]]
            current_user = r.get("user_msg_for_turn")

            # NOTE: do NOT pass current_user to render — we'll build the final user
            # message ourselves by combining current_user + probe action into ONE
            # message. This avoids consecutive user→user messages.
            base_text_msgs = [{"role": "system", "content": system_content}] + \
                              render_history_messages_text(prior, None)
            base_fc_msgs = [{"role": "system", "content": system_content}] + \
                            render_history_messages_fc(prior, None)

            base = {
                "case_id": cid,
                "turn_idx": r["turn_idx"],
                "step_idx": r["step_idx"],
                "global_emit_idx": r["global_emit_idx"],
                "involved_classes": involved,
            }

            def _combined_user(probe_call_str: str) -> str:
                # Combine the turn's user message (if turn-start) with the probe
                # into ONE user message. For mid-turn steps, just the probe.
                parts = []
                if current_user:
                    parts.append(current_user)
                parts.append(f"[probe action] {probe_call_str}")
                return "\n\n".join(parts)

            # ---- expert IWM samples: one per expert call ----
            # IWM target is the next-state summary regardless of format.
            # Both text and FC variants end with: user (current_turn + probe), assistant (summary).
            # Only history rendering differs between formats.
            expert_calls = r["expert_emit_decoded"]
            expert_sums = r.get("expert_summaries") or [None] * len(expert_calls)
            for j, (c, s) in enumerate(zip(expert_calls, expert_sums)):
                if not s:
                    continue
                user_msg = {"role": "user", "content": _combined_user(c)}
                target_msg = {"role": "assistant", "content": s}
                text_records.append(dict(base, kind="expert", probe_call=c,
                                          messages=merge_consecutive_user_messages(base_text_msgs + [user_msg, target_msg])))
                fc_records.append(dict(base, kind="expert", probe_call=c,
                                        messages=merge_consecutive_user_messages(base_fc_msgs + [user_msg, target_msg])))
                n_expert += 1

            # ---- alt IWM samples ----
            for alt in r.get("alts", []):
                if not alt.get("valid") or not alt.get("call") or not alt.get("summary"):
                    n_alt_skipped += 1
                    continue
                c = alt["call"]
                s = alt["summary"]
                user_msg = {"role": "user", "content": _combined_user(c)}
                target_msg = {"role": "assistant", "content": s}
                text_records.append(dict(base, kind="alt", probe_call=c,
                                          messages=merge_consecutive_user_messages(base_text_msgs + [user_msg, target_msg])))
                fc_records.append(dict(base, kind="alt", probe_call=c,
                                        messages=merge_consecutive_user_messages(base_fc_msgs + [user_msg, target_msg])))
                n_alt += 1

    write_jsonl(OUT_DIR / "iwm_sft_text.jsonl", text_records)
    write_jsonl(OUT_DIR / "iwm_sft_fc.jsonl", fc_records)

    print(f"iwm_sft built:")
    print(f"  text records: {len(text_records)}  → {OUT_DIR/'iwm_sft_text.jsonl'}")
    print(f"  fc   records: {len(fc_records)}  → {OUT_DIR/'iwm_sft_fc.jsonl'}")
    print(f"  breakdown   : expert={n_expert}, alt={n_alt}, alt_skipped={n_alt_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

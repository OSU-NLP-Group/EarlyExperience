"""
Build reflection_sft_text.jsonl + reflection_sft_fc.jsonl.

For each reflection in sr_full.jsonl:
  - reconstruct s_i (system + tools + history up to that step) — same as iwm_sft user-side
  - assistant content =
      text variant: "Thought:\\n<reflection>\\n\\nAction:\\n<calls>"
      fc   variant: "Thought:\\n<reflection>" + tool_calls (structured)

`quality_flags` dict attached per record — count-only (no drops).

Run:
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/build_reflection_sft.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sft_common import (  # noqa: E402
    REPO, build_reflection_system_prompt, calls_to_fc_tool_calls, calls_to_text_block,
    load_tool_schemas, merge_consecutive_user_messages,
    render_history_messages_fc, render_history_messages_text, write_jsonl,
)

IN_SR = REPO / "envs/bfcl_v4/data/rollout/sr_full.jsonl"
# IWM-summarized provides the per-step history (with expert summaries used as observations)
IN_IWM = REPO / "envs/bfcl_v4/data/rollout/iwm_full_summarized.jsonl"
OUT_DIR = REPO / "envs/bfcl_v4/data/sft"

BANNED_VOCAB = ['expert', 'selected', 'chosen', 'correct', 'best', 'optimal',
                'preferred', 'ideal', 'official', 'recommended']
META_PATTERNS = [r'\balternatives?\b', r'\bgiven the options\b', r'\bI was shown\b']


def compute_quality_flags(reflection: str) -> dict:
    """All checks are COUNT-ONLY — records are NEVER dropped on these flags."""
    if not reflection:
        return {"banned_vocab_hits": [], "is_mode_collapsed": True,
                "word_count": 0, "has_meta_reference": False}
    low = reflection.lower()
    banned_hits = sorted({w for w in BANNED_VOCAB if re.search(r'\b' + w + r'\b', low)})
    wc = len(reflection.split())
    # mode-collapse: word count > 500 OR short n-gram density very high
    is_collapse = wc > 500
    if not is_collapse and len(reflection) > 160:
        # heuristic: if first 80 chars appear again later, flag
        head = reflection[:80].strip().lower()
        if head and head in reflection[80:].strip().lower():
            is_collapse = True
    has_meta = any(re.search(p, reflection, re.IGNORECASE) for p in META_PATTERNS)
    return {
        "banned_vocab_hits": banned_hits,
        "is_mode_collapsed": is_collapse,
        "word_count": wc,
        "has_meta_reference": has_meta,
    }


def main() -> int:
    # need iwm records for history rendering (they have the expert_summaries to fold into history)
    iwm_by_case: dict[str, list[dict]] = defaultdict(list)
    with IN_IWM.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            iwm_by_case[r["case_id"]].append(r)
    for cid in iwm_by_case:
        iwm_by_case[cid].sort(key=lambda r: r["global_emit_idx"])

    text_records = []
    fc_records = []
    flag_counts = {"banned_vocab_any": 0, "mode_collapse": 0,
                   "meta_reference": 0, "all_clean": 0}

    with IN_SR.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sr = json.loads(line)
            if not sr.get("reflection"):
                continue

            cid = sr["case_id"]
            geix = sr["global_emit_idx"]
            case_iwm = iwm_by_case.get(cid, [])

            # match this SR record to its IWM record (same global_emit_idx)
            this_iwm = next((r for r in case_iwm if r["global_emit_idx"] == geix), None)
            if this_iwm is None:
                continue
            prior = [pr for pr in case_iwm if pr["global_emit_idx"] < geix]
            involved = this_iwm["involved_classes"]
            schemas = load_tool_schemas(involved)
            system_content = build_reflection_system_prompt(schemas)
            current_user = this_iwm.get("user_msg_for_turn")

            text_msgs_base = [{"role": "system", "content": system_content}] + \
                              render_history_messages_text(prior, current_user)
            fc_msgs_base = [{"role": "system", "content": system_content}] + \
                            render_history_messages_fc(prior, current_user)

            expert_calls = sr["expert_emit_decoded"]
            refl = sr["reflection"].strip()

            # text variant: "Thought:\n<reflection>\n\nAction:\n<call_str>"
            text_assistant = f"Thought:\n{refl}\n\nAction:\n{calls_to_text_block(expert_calls)}"
            text_msgs = text_msgs_base + [{"role": "assistant", "content": text_assistant}]

            # fc variant: content=Thought:\n<reflection>, tool_calls=[...]
            fc_tool_calls = calls_to_fc_tool_calls(expert_calls, sr["turn_idx"], sr["step_idx"])
            fc_msgs = fc_msgs_base + [{
                "role": "assistant",
                "content": f"Thought:\n{refl}",
                "tool_calls": fc_tool_calls,
            }]

            quality = compute_quality_flags(refl)
            if quality["banned_vocab_hits"]:
                flag_counts["banned_vocab_any"] += 1
            if quality["is_mode_collapsed"]:
                flag_counts["mode_collapse"] += 1
            if quality["has_meta_reference"]:
                flag_counts["meta_reference"] += 1
            if (not quality["banned_vocab_hits"]
                    and not quality["is_mode_collapsed"]
                    and not quality["has_meta_reference"]):
                flag_counts["all_clean"] += 1

            base = {
                "case_id": cid,
                "turn_idx": sr["turn_idx"],
                "step_idx": sr["step_idx"],
                "global_emit_idx": geix,
                "involved_classes": involved,
                "quality_flags": quality,
            }
            text_records.append(dict(base, messages=merge_consecutive_user_messages(text_msgs)))
            fc_records.append(dict(base, messages=merge_consecutive_user_messages(fc_msgs)))

    write_jsonl(OUT_DIR / "reflection_sft_text.jsonl", text_records)
    write_jsonl(OUT_DIR / "reflection_sft_fc.jsonl", fc_records)

    n = len(text_records)
    print(f"reflection_sft built (count-only filters, no drops):")
    print(f"  text records: {n}  → {OUT_DIR/'reflection_sft_text.jsonl'}")
    print(f"  fc   records: {n}  → {OUT_DIR/'reflection_sft_fc.jsonl'}")
    print(f"  quality flag distribution (overlaps possible):")
    print(f"    banned_vocab_any        : {flag_counts['banned_vocab_any']}  ({flag_counts['banned_vocab_any']/n*100:.1f}%)")
    print(f"    is_mode_collapsed=True  : {flag_counts['mode_collapse']}  ({flag_counts['mode_collapse']/n*100:.1f}%)")
    print(f"    has_meta_reference=True : {flag_counts['meta_reference']}  ({flag_counts['meta_reference']/n*100:.1f}%)")
    print(f"    fully clean (no flags)  : {flag_counts['all_clean']}  ({flag_counts['all_clean']/n*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Attempt-2 rerun on flagged SR reflections.

Reads sr_rollout.jsonl, picks records whose flags fired (banned_vocab,
numbered_labels, or doubled), re-issues each prompt with the SAME settings
(same model, temperature, frequency_penalty) — just relying on sampling
variation to land on a cleaner output. Saves to sr_rerun.jsonl with the
attempt 1 flags preserved as prev_flags for downstream comparison.

Pattern from tau-bench / sciworld: ~50% of flagged records come back
clean on attempt 2; the residue is typically a small fraction of false
positives or hard cases that the SFT-build step can drop or re-context-filter.

Usage:
  DEEPSEEK_API_KEY=... \\
    /home/ulss/miniconda3/envs/travelplanner-ee/bin/python rerun_sr_flagged.py
"""
from __future__ import annotations
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ENV_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ENV_ROOT / "data" / "rollout" / "sr_rollout.jsonl"
OUT_PATH = ENV_ROOT / "data" / "rollout" / "sr_rerun.jsonl"

# Same constants as rollout_sr.py (kept here to avoid re-importing the whole module)
BANNED_VOCAB = (
    "expert", "selected", "chosen", "picked", "correct", "optimal",
    "right action", "right choice",
)
NUMBERED_LABELS = re.compile(r"\b(Action|Alternative|Option)\s+\d+\b", re.I)
SYS_PROMPT = "You are reasoning through a travel-planning decision as the agent yourself."


def quality_flags(reflection: str) -> dict:
    text = reflection.lower()
    banned_hits = [v for v in BANNED_VOCAB if re.search(rf"\b{re.escape(v)}\b", text)]
    numbered = NUMBERED_LABELS.search(reflection)
    half = len(reflection) // 2
    doubled = (
        half > 50
        and reflection[:100] and len(reflection) > 200
        and reflection[:80].strip() in reflection[half:]
    )
    return {
        "banned_vocab": banned_hits,
        "has_numbered_labels": bool(numbered),
        "numbered_label_match": numbered.group(0) if numbered else None,
        "is_doubled": doubled,
        "word_count": len(reflection.split()),
    }


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com")

    rows = [json.loads(l) for l in IN_PATH.read_text().splitlines() if l.strip()]
    flagged = [r for r in rows
               if r.get("flags")
               and (r["flags"]["banned_vocab"]
                    or r["flags"]["has_numbered_labels"]
                    or r["flags"]["is_doubled"])]
    print(f"[input] {len(rows)} attempt-1 records, {len(flagged)} flagged")

    def one(r):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": r["prompt"]},
                ],
                temperature=0.7,
                frequency_penalty=0.3,
                extra_body={"thinking": {"type": "disabled"}},
            )
            text = resp.choices[0].message.content
            usage = {"input_tokens": resp.usage.prompt_tokens,
                     "output_tokens": resp.usage.completion_tokens}
            return r, text, usage, None
        except Exception as e:
            return r, None, None, f"{type(e).__name__}: {e}"

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for f in as_completed([ex.submit(one, r) for r in flagged]):
            results.append(f.result())
    wall = time.time() - t0

    out = []
    total_in = total_out = 0
    cleaner = 0
    same_or_worse = 0
    for r, text, usage, err in results:
        flags2 = quality_flags(text) if text else None
        rec = {
            "traj_idx": r["traj_idx"], "step_idx": r["step_idx"],
            "day": r["day"], "field": r["field"], "type": r["type"],
            "attempt": 2,
            "reflection": text, "error": err,
            "flags": flags2,
            "prev_flags": r["flags"],
            "usage": usage,
        }
        out.append(rec)
        if usage:
            total_in += usage["input_tokens"]; total_out += usage["output_tokens"]
        if flags2 and not (flags2["banned_vocab"] or flags2["has_numbered_labels"] or flags2["is_doubled"]):
            cleaner += 1
        else:
            same_or_worse += 1

    OUT_PATH.write_text("\n".join(json.dumps(r, default=str) for r in out) + "\n")
    cost = (total_in * 0.27 + total_out * 1.10) / 1_000_000
    print(f"[done] {len(out)} reruns")
    print(f"[wall] {wall:.1f}s")
    print(f"[tokens] in={total_in:,}, out={total_out:,}, cost=${cost:.4f}")
    print(f"[result] {cleaner} now clean / {same_or_worse} still flagged")
    print(f"[out] {OUT_PATH}")


if __name__ == "__main__":
    main()

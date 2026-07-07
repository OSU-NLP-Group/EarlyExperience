"""Second-pass compression of SR reflections.

Reads sr_rollout.jsonl + sr_rerun.jsonl (full v9), sends each reflection
to DeepSeek with a compress-prompt that PRESERVES the decisive checks
(wrong-city, repeat, min-nights, mode-chain, city-sequence, cuisine-match)
while stripping hedging, restated state, and multi-alt enumeration.

Output: sr_rollout_compressed.jsonl — drop-in replacement for sr_rollout.jsonl
with shorter `reflection` text. The structure matches sr_rollout.jsonl
exactly so build_sft.py picks it up via the same SR_PATH symlink swap.

Usage:
  DEEPSEEK_API_KEY=... \\
    /home/ulss/miniconda3/envs/travelplanner-ee/bin/python compress_sr.py
"""
from __future__ import annotations
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ENV_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_PATH = ENV_ROOT / "data" / "rollout" / "sr_rollout.jsonl"
RERUN_PATH = ENV_ROOT / "data" / "rollout" / "sr_rerun.jsonl"
OUT_PATH = ENV_ROOT / "data" / "rollout" / "sr_rollout_compressed.jsonl"

SYS_PROMPT = "You compress travel-planning reasoning."

COMPRESS_TEMPLATE = """Below is an inner monologue produced by a travel-planning agent reasoning about a single step (one decision). Compress it to a SHORTER version (target 50–90 words) while preserving the key information.

PRESERVE (verbatim or near-verbatim):
1. The rule check that justifies the action — usually one of: wrong-city rejection, repeat-restaurant/attraction rejection, min-nights / consecutive-stay logic, mode-chain (self-driving lock), city-sequence (which destination cities visited / still needed), forced mandatory-field SKIP.
2. CUISINE MATCH or MISMATCH if the original mentions it (the query's cuisine preference and whether the chosen restaurant satisfies it).
3. Budget delta — the dollar numbers: spent before → spent after, remaining.
4. The final commitment ("I'll go with X", "I'll skip Y").

DROP:
- Restating which day / city the agent is in if it's already clear from the decision.
- Hedging adjectives ("comfortable", "workable", "manageable", "still well within").
- Enumerating every rejected option (keep at most 2-3 most-relevant rejections).
- Speculation about future days ("I'll still need lunch later, and that should be...").
- Multi-sentence elaboration when one sentence suffices.

DO NOT introduce phrases not present in the original. DO NOT add meta-references like "the decision recorded", "the annotation", "the system shows" — those are leakage. DO NOT add the word "compressed" or any meta-commentary. Output ONLY the compressed prose.

ORIGINAL:
{reflection}

COMPRESSED:"""


def call_llm(client: OpenAI, reflection: str) -> tuple[str, dict]:
    prompt = COMPRESS_TEMPLATE.format(reflection=reflection)
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        extra_body={"thinking": {"type": "disabled"}},
    )
    msg = resp.choices[0].message.content.strip()
    usage = {"input_tokens": resp.usage.prompt_tokens,
             "output_tokens": resp.usage.completion_tokens}
    return msg, usage


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY not set")

    # Load v9 base rollout and the rerun (which overrides flagged records)
    base = [json.loads(l) for l in ROLLOUT_PATH.read_text().splitlines() if l.strip()]
    rerun_map = {}
    if RERUN_PATH.exists():
        for l in RERUN_PATH.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            rerun_map[(r["traj_idx"], r["step_idx"])] = r
    # Build effective input set: use rerun where available
    effective = []
    for r in base:
        key = (r["traj_idx"], r["step_idx"])
        if key in rerun_map:
            # Use the rerun's reflection but keep original metadata wrappers
            merged = dict(r); merged["reflection"] = rerun_map[key]["reflection"]
            merged["prev_flags"] = r.get("flags")
            merged["flags"] = rerun_map[key].get("flags")
            effective.append(merged)
        else:
            effective.append(r)

    print(f"[input] {len(effective)} effective records (base={len(base)}, rerun={len(rerun_map)})")

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    def worker(idx_rec):
        idx, rec = idx_rec
        if not rec.get("reflection"):
            return idx, rec, None
        try:
            compressed, usage = call_llm(client, rec["reflection"])
        except Exception as e:
            return idx, rec, {"error": str(e)}
        out = dict(rec)
        out["reflection_original"] = rec["reflection"]
        out["reflection"] = compressed
        out["compress_usage"] = usage
        return idx, out, None

    t0 = time.time()
    results = [None] * len(effective)
    in_tokens = out_tokens = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(worker, (i, r)) for i, r in enumerate(effective)]
        for fut in as_completed(futures):
            idx, rec, err = fut.result()
            if err:
                errors += 1
                results[idx] = effective[idx]  # keep original on error
                continue
            results[idx] = rec
            if rec.get("compress_usage"):
                in_tokens += rec["compress_usage"]["input_tokens"]
                out_tokens += rec["compress_usage"]["output_tokens"]

    wall = time.time() - t0
    # DeepSeek-V4-Pro pricing (approx): $0.27/M input, $1.10/M output
    cost = in_tokens * 0.27e-6 + out_tokens * 1.10e-6
    print(f"[done] {len(results)} compressed in {wall:.1f}s")
    print(f"[tokens] in={in_tokens:,}, out={out_tokens:,}, cost=${cost:.4f}")
    print(f"[errors] {errors}")

    with OUT_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[out] {OUT_PATH}")


if __name__ == "__main__":
    main()

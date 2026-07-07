"""Smoke for the response summarizer.

For ~80 stratified-sampled probes across buckets (data_json_dict/list, http_4xx,
py_nameerror, py_other), call DeepSeek V4 Flash with a bfcl_v4-style summarizer
prompt adapted for AppWorld. Eyeball quality + measure tokens to project full-scale cost.

Cost target: <$0.10. Wall: ~30-60s @ 16 workers.
"""
import os, sys, json, random, time, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

DEEPSEEK_MODEL = "deepseek-v4-flash"
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

PROBE_PATH = Path("envs/appworld/data/rollout/probe_full.jsonl")
OUT_PATH = Path("envs/appworld/data/_recon/smoke_summary.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_RAW_CHARS = 3000   # truncate raw response before sending to LLM
MAX_CONCURRENT = 16

# Stratified-sample sizes per bucket (total ~80)
SAMPLE_SIZES = {
    "data_json_dict": 25,
    "data_json_list": 20,
    "http_401":       12,
    "http_422":       12,
    "py_nameerror":    8,
    "py_other":        3,
}


SUMMARIZER_SYSTEM = """You translate raw AppWorld tool execution outcomes into concise English. Given a Python API call and its raw return value from a simulated environment, output ONE sentence that states FACTUALLY what the tool did or what data it returned.

Rules:
  - Successful action (state change — song liked, item added, account created, password reset, payment made, etc.) → state what changed, past tense, naming the affected object(s).
    e.g. call: apis.spotify.like_song(song_id=5, access_token=...) → {"liked_at":"2023-05-18"}
         output: "Liked Spotify song with id 5."
    e.g. call: apis.spotify.add_payment_card(...) → {"card_id":7}
         output: "Added a new Spotify payment card with id 7."
  - Read-only call (show_*, search_*, list_*, get_*) → state what data was returned, including key identifiers/values.
    e.g. call: apis.supervisor.show_profile() → {"first_name":"Joyce","email":"joyce-weav@gmail.com",...}
         output: "Retrieved supervisor profile for Joyce Weaver, email joyce-weav@gmail.com."
    e.g. call: apis.spotify.show_genres() → ["EDM","R&B","jazz",...]
         output: "Retrieved Spotify genres list: EDM, R&B, jazz, hip-hop, rock, pop, classical, reggae, country."
  - Error response (HTTP 4xx wrapped in Python Exception, or Python-level error like NameError/TypeError) → restate the error factually, preserving status code and key message.
    e.g. Exception: Response status code is 401: {"message":"Not logged in to spotify"}
         output: "The Spotify API returned HTTP 401: not logged in."
    e.g. NameError: name 'access_token' is not defined
         output: "Python NameError: 'access_token' is not defined in current scope."
  - Null/empty return → state the call completed without returning data.

STRICT CONSTRAINTS:
  - DO NOT mention any user, any task, any goal, or whether the call was useful.
  - DO NOT compare to expected outcomes, expert actions, or other alternatives.
  - DO NOT add reasoning, judgment, or commentary about correctness.
  - DO NOT analyze WHY the error happened or suggest fixes.
  - Preserve specific identifiers/values present in the raw response.

Output ONLY the one factual sentence — no quotes, no prefix, no markdown, no explanation."""


def build_user(call_str, raw_response):
    raw_truncated = raw_response[:MAX_RAW_CHARS]
    suffix = "\n…[truncated]…" if len(raw_response) > MAX_RAW_CHARS else ""
    return (
        f"## Tool call\n`{call_str}`\n\n"
        f"## Raw tool response\n```\n{raw_truncated}{suffix}\n```\n\n"
        f"## Output\nOne factual sentence per the system rules."
    )


def call_summarizer(call_str, raw_response):
    user = build_user(call_str, raw_response)
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            extra_body={"thinking": {"type": "disabled"}},
        )
        summary = (r.choices[0].message.content or "").strip()
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1].strip()
        return {
            "summary": summary,
            "usage": {
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
            },
            "wall": round(time.time() - t0, 2),
            "error": None,
        }
    except Exception as e:
        return {"summary": None, "usage": None, "wall": round(time.time()-t0,2), "error": str(e)[:200]}


def main():
    rng = random.Random(42)
    print(f"Loading probes from {PROBE_PATH}…")
    all_probes = [json.loads(l) for l in PROBE_PATH.open()]
    print(f"  total: {len(all_probes):,}")

    by_bucket = collections.defaultdict(list)
    for r in all_probes:
        by_bucket[r["bucket"]].append(r)
    print("Bucket distribution:")
    for b, lst in sorted(by_bucket.items(), key=lambda x: -len(x[1])):
        print(f"  {b:<22s} {len(lst):>6,}")

    # Stratified sample
    sample = []
    for bucket, n in SAMPLE_SIZES.items():
        avail = by_bucket.get(bucket, [])
        if not avail:
            print(f"  WARN: no {bucket} probes available")
            continue
        sample.extend(rng.sample(avail, min(n, len(avail))))
    rng.shuffle(sample)
    print(f"\nSampled {len(sample)} probes across buckets")

    # Run
    print(f"\nRunning summarizer @ {MAX_CONCURRENT} workers…")
    results = [None]*len(sample)

    def work(i):
        r = sample[i]
        out = call_summarizer(r["call"], r["response"] or "")
        return i, {
            "probe": {
                "task_id": r["task_id"], "step": r["step"], "app": r["app"], "endpoint": r["endpoint"],
                "call": r["call"], "bucket": r["bucket"],
                "response_head": (r["response"] or "")[:300],
                "response_len": r["response_len"],
            },
            **out,
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = [ex.submit(work, i) for i in range(len(sample))]
        for f in as_completed(futures):
            i, r = f.result()
            results[i] = r

    wall = time.time() - t0
    tok_in = sum(r["usage"]["prompt_tokens"] for r in results if r["usage"])
    tok_out = sum(r["usage"]["completion_tokens"] for r in results if r["usage"])
    cost = tok_in*0.10/1e6 + tok_out*0.40/1e6
    n_err = sum(1 for r in results if r["error"])

    print(f"\nDone in {wall:.1f}s. tokens in={tok_in:,} out={tok_out:,} cost=${cost:.4f}  errors={n_err}")

    # Per-bucket stats
    print(f"\n=== Per-bucket avg token + sample summary ===")
    for bucket in SAMPLE_SIZES:
        bsam = [r for r in results if r["probe"]["bucket"] == bucket]
        if not bsam: continue
        avg_in = sum(r["usage"]["prompt_tokens"] for r in bsam if r["usage"])/len(bsam)
        avg_out = sum(r["usage"]["completion_tokens"] for r in bsam if r["usage"])/len(bsam)
        print(f"\n  --- {bucket} (n={len(bsam)})  avg in={avg_in:.0f}  out={avg_out:.0f} ---")
        for r in bsam[:3]:
            p = r["probe"]
            call_brief = p["call"][:80]
            resp_brief = (p["response_head"] or "")[:120].replace("\n", " ")
            print(f"    call: {call_brief}")
            print(f"    raw:  {resp_brief}")
            print(f"    sum:  {r['summary']}")
            print()

    # Project full cost
    full_n = 84863
    proj_cost = cost * full_n / len(sample)
    proj_in = tok_in * full_n / len(sample)
    proj_out = tok_out * full_n / len(sample)
    proj_wall_min = (full_n / len(sample)) * (wall/60) * (MAX_CONCURRENT / 50)  # if we use 50 workers
    print(f"\n=== Full-scale projection (84,863 records, 50 workers) ===")
    print(f"  tokens: ~{proj_in/1e6:.1f}M in + ~{proj_out/1e6:.1f}M out = ~{(proj_in+proj_out)/1e6:.1f}M total")
    print(f"  cost (Flash): ~${proj_cost:.2f}")
    print(f"  wall: ~{proj_wall_min:.0f} min")

    json.dump({
        "n_sample": len(sample),
        "wall_seconds": round(wall, 1),
        "totals": {"input_tokens": tok_in, "output_tokens": tok_out, "cost_usd": round(cost, 4)},
        "projection_84k": {
            "cost_usd": round(proj_cost, 2),
            "input_tokens": int(proj_in),
            "output_tokens": int(proj_out),
        },
        "results": results,
    }, OUT_PATH.open("w"), indent=2, default=str)
    print(f"\nFull output → {OUT_PATH}")


if __name__ == "__main__":
    main()

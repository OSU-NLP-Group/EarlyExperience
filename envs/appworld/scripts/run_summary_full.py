"""Full summarizer pass — LLM-summarize every probe response.

Reads envs/appworld/data/rollout/probe_full.jsonl (84,863 records). For each,
sends (call, raw response) to DeepSeek V4 Flash with bfcl_v4-style summarizer
prompt; writes original record + `summary` field to probe_full_summarized.jsonl.

Streaming output, resumable on (task_id, step, endpoint) key.
"""
import os, sys, json, time, threading, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

DEEPSEEK_MODEL = "deepseek-v4-flash"
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

IN_PATH = Path("envs/appworld/data/rollout/probe_full.jsonl")
OUT_PATH = Path("envs/appworld/data/rollout/probe_full_summarized.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENT = 50
MAX_RAW_CHARS = 3000
N_RETRIES = 3
RETRY_BACKOFF = (1.0, 3.0, 8.0)


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


def call_with_retry(call_str, raw_response):
    user = build_user(call_str, raw_response or "")
    last_err = None
    for attempt in range(N_RETRIES):
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
                "error": None,
            }
        except Exception as e:
            last_err = str(e)[:200]
            if attempt < N_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
    return {"summary": None, "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "error": last_err}


def load_done_keys():
    if not OUT_PATH.exists(): return set()
    done = set()
    with OUT_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["step"], r["endpoint"]))
            except Exception:
                continue
    return done


def main():
    done = load_done_keys()
    print(f"Already summarized: {len(done):,} records (will skip)")

    print(f"Loading probes from {IN_PATH}…")
    all_probes = [json.loads(l) for l in IN_PATH.open()]
    print(f"  total in probe_full.jsonl: {len(all_probes):,}")

    jobs = [r for r in all_probes if (r["task_id"], r["step"], r["endpoint"]) not in done]
    print(f"  jobs to summarize: {len(jobs):,}")
    if not jobs:
        print("Nothing to do."); return

    out_f = OUT_PATH.open("a", buffering=1)
    write_lock = threading.Lock()
    n_done = 0
    n_err = 0
    total_in = 0; total_out = 0
    t_start = time.time()
    bucket_counts = collections.Counter()

    def work(probe):
        r = call_with_retry(probe["call"], probe["response"] or "")
        merged = dict(probe)
        merged["summary"] = r["summary"]
        merged["summary_error"] = r["error"]
        merged["summary_usage"] = r["usage"]
        return merged

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = {ex.submit(work, j): j for j in jobs}
        for f in as_completed(futures):
            try:
                m = f.result()
            except Exception as e:
                print(f"  WORKER CRASH: {e}", flush=True); continue
            with write_lock:
                out_f.write(json.dumps(m, default=str) + "\n")
                n_done += 1
                if m.get("summary_error"): n_err += 1
                total_in += m["summary_usage"]["prompt_tokens"]
                total_out += m["summary_usage"]["completion_tokens"]
                bucket_counts[m["bucket"]] += 1
                if n_done % 1000 == 0 or n_done == len(jobs):
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed
                    eta = (len(jobs) - n_done) / rate if rate > 0 else 0
                    cost = total_in*0.10/1e6 + total_out*0.40/1e6
                    print(
                        f"  [{n_done:>6}/{len(jobs)}] {rate:.1f}/s err={n_err} "
                        f"tok in={total_in/1e6:.2f}M out={total_out/1e6:.2f}M "
                        f"cost=${cost:.2f} eta={eta/60:.1f}min",
                        flush=True
                    )

    out_f.close()
    wall = time.time() - t_start
    final_cost = total_in*0.10/1e6 + total_out*0.40/1e6
    print(f"\n=== DONE: {n_done} summaries in {wall/60:.1f}min  rate={n_done/wall:.1f}/s ===")
    print(f"  tokens: in={total_in:,}  out={total_out:,}  cost=${final_cost:.2f}")
    print(f"  errors: {n_err}")
    print(f"  output: {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Full SR (self-reflection) generation over all 931 expert states — v5, K=0.

Reuses the reviewed v5 reflection prompt + builder from smoke_sr.py (imported,
not copied, so there is no prompt drift). For each expert state:
  - history = all prior expert steps with their outcome summaries
  - NO alternatives (K=0). AppWorld's action space is too large for random alts
    to be genuine competitors; alt-comparison SR taught hallucination + password
    guessing (see NOTES.md). The reflection is a pure "why is this action the
    sensible move here" CoT, grounded in task+history+real outcome (author-only).
  - DeepSeek V4 Pro writes the monologue ONLY. The <code> block is attached
    deterministically at SFT-build time — never rely on the generator to reprint.

Streaming JSONL output, resumable on (task_id, step). Records leak-scan + doubling
flags for the downstream build-time filter (we do NOT drop here).

Model: deepseek-v4-pro, temp 0.7, thinking disabled. Cost ~$0.5, ~6 min @ 20 workers.
"""
import os, sys, json, re, time, random, threading, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)
APPWORLD_ROOT = os.environ["APPWORLD_ROOT"]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_sr import (  # reuse reviewed v5 prompt + builder verbatim
    REFLECTION_SYSTEM, build_reflection_user, check_leakage, PRO, client,
)

EXPERT_PATH = Path("envs/appworld/data/rollout/expert_outcomes.jsonl")
OUT_PATH = Path("envs/appworld/data/rollout/reflection_full.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_CONCURRENT = 20
N_RETRIES = 3
RETRY_BACKOFF = (1.0, 3.0, 8.0)


def reflect_with_retry(task, history, expert_code, expert_outcome):
    user = build_reflection_user(task, history, expert_code, expert_outcome)
    last_err = None
    for attempt in range(N_RETRIES):
        try:
            r = client.chat.completions.create(
                model=PRO,
                messages=[{"role": "system", "content": REFLECTION_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.7,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return (r.choices[0].message.content or ""), {
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
            }, None
        except Exception as e:
            last_err = str(e)[:200]
            if attempt < N_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
    return None, {"prompt_tokens": 0, "completion_tokens": 0}, last_err


def load_done():
    if not OUT_PATH.exists():
        return set()
    done = set()
    with OUT_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["step"]))
            except Exception:
                continue
    return done


def main():
    # index expert states + alts
    expert_by_state = {}
    max_step = collections.defaultdict(int)
    for line in EXPERT_PATH.open():
        r = json.loads(line)
        expert_by_state[(r["task_id"], r["step"])] = r
        max_step[r["task_id"]] = max(max_step[r["task_id"]], r["step"])

    task_instr = {}
    def instr(tid):
        if tid not in task_instr:
            task_instr[tid] = json.load(
                (Path(APPWORLD_ROOT)/"data"/"tasks"/tid/"specs.json").open()
            )["instruction"]
        return task_instr[tid]

    done = load_done()
    print(f"Resumable: {len(done)} states already done")

    # build jobs — K=0, no alternatives
    jobs = []
    for (tid, si) in sorted(expert_by_state.keys()):
        if (tid, si) in done:
            continue
        history = [{"code": expert_by_state[(tid, j)]["expert_code"],
                    "outcome": expert_by_state[(tid, j)]["outcome_summary"]}
                   for j in range(si)]
        jobs.append({
            "task_id": tid, "step": si,
            "task": instr(tid),
            "history": history,
            "expert_code": expert_by_state[(tid, si)]["expert_code"],
            "expert_outcome": expert_by_state[(tid, si)]["outcome_summary"],
        })
    print(f"Jobs: {len(jobs)}")
    if not jobs:
        print("Nothing to do."); return

    out_f = OUT_PATH.open("a", buffering=1)
    lock = threading.Lock()
    n_done = 0; n_err = 0; n_leak = 0; n_doubled = 0; n_stray = 0
    tin = 0; tout = 0
    t0 = time.time()

    def work(job):
        cot, usage, err = reflect_with_retry(
            job["task"], job["history"], job["expert_code"], job["expert_outcome"])
        mono = None; leak = None; doubled = None; stray = None
        if cot is not None:
            mono = cot.strip()
            leak = check_leakage(mono)
            doubled = len(mono) > 80 and mono.count(mono[:60]) >= 2
            stray = "<code>" in cot
        return {
            "task_id": job["task_id"], "step": job["step"],
            "expert_code": job["expert_code"],
            "expert_outcome": job["expert_outcome"],
            "reflection_raw": mono,       # monologue only; <code> attached at build time
            "leak": leak,
            "doubled": doubled,
            "stray_code": stray,
            "word_count": (len(mono.split()) if mono else 0),
            "usage": usage,
            "error": err,
        }

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            with lock:
                out_f.write(json.dumps(r, default=str) + "\n")
                n_done += 1
                if r["error"]: n_err += 1
                if r["leak"]: n_leak += 1
                if r["doubled"]: n_doubled += 1
                if r["stray_code"]: n_stray += 1
                tin += r["usage"]["prompt_tokens"]; tout += r["usage"]["completion_tokens"]
                if n_done % 50 == 0 or n_done == len(jobs):
                    el = time.time()-t0
                    rate = n_done/el
                    eta = (len(jobs)-n_done)/rate if rate>0 else 0
                    cost = tin*0.27/1e6 + tout*1.10/1e6
                    print(f"  [{n_done:>4}/{len(jobs)}] {rate:.1f}/s err={n_err} "
                          f"leak={n_leak} doubled={n_doubled} stray={n_stray} "
                          f"cost=${cost:.2f} eta={eta/60:.1f}min", flush=True)

    out_f.close()
    wall = time.time()-t0
    cost = tin*0.27/1e6 + tout*1.10/1e6
    print(f"\n=== DONE: {n_done} in {wall/60:.1f}min ===")
    print(f"  errors: {n_err}   leak-flagged: {n_leak}   doubled: {n_doubled}   stray-code: {n_stray}")
    print(f"  tokens in={tin:,} out={tout:,}  cost=${cost:.2f}")
    print(f"  output: {OUT_PATH}")
    print(f"  (NOTE: leak/doubled flags are for the downstream filter pass; nothing dropped here.)")


if __name__ == "__main__":
    main()

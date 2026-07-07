"""Probe smoke — run all proposer alts of one task through fresh env probes.

Target task: 82e2fac_1 (10 steps × 102 endpoints = 1,020 probes).

Validates:
  - server handles N-way concurrent fresh-env probes without hang/crash
  - LLM-filled-args produce richer signal than naive no-args baseline
  - response length distribution / bucket mix is sane

Output: envs/appworld/data/_recon/probe_smoke_82e2fac_1.json
"""
import os, sys, json, ast, time, uuid, re, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)
from appworld import AppWorld
from appworld.ground_truth import GroundTruth

TASK_ID = "82e2fac_1"
URL = "http://0.0.0.0:7050"
MAX_CONCURRENT = 1
OUT_PATH = Path("envs/appworld/data/_recon/probe_smoke_82e2fac_1.json")


def parse_blocks(src):
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solution"), None)
    if not fn: return []
    lines = src.splitlines()
    indent = len(lines[fn.body[0].lineno - 1]) - len(lines[fn.body[0].lineno - 1].lstrip())
    blocks = []
    for stmt in fn.body:
        chunk = "\n".join(
            l[indent:] if len(l) >= indent else l.lstrip()
            for l in lines[stmt.lineno - 1 : stmt.end_lineno]
        ).strip()
        if chunk: blocks.append(chunk)
    return blocks


def classify(text):
    if text is None: return "probe_crash"
    if not text: return "empty"
    s = text.lower()
    if "execution successful" in s and len(s) < 50:
        return "noop_success"
    m = re.search(r"response status code is (\d+)", s)
    if m:
        return f"http_{m.group(1)}"
    if "typeerror" in s:    return "py_typeerror"
    if "nameerror" in s:    return "py_nameerror"
    if "syntaxerror" in s:  return "py_syntaxerror"
    if "traceback" in s or "exception" in s:
        return "py_other"
    if s.strip().startswith("["):  return "data_json_list"
    if s.strip().startswith("{"):  return "data_json_dict"
    return "data_text"


def probe_one(task_id, prior_code, call_str):
    name = f"probe_{uuid.uuid4().hex[:8]}"
    env = None
    try:
        env = AppWorld(task_id=task_id, experiment_name=name, remote_environment_url=URL)
        for code in prior_code:
            env.execute(code)
        resp = env.execute(call_str)
        return resp, None
    except Exception as e:
        return None, str(e)[:200]
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Load proposer records for target task
    recs = []
    with open("envs/appworld/data/rollout/proposer_full.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["task_id"] == TASK_ID:
                recs.append(r)
    recs.sort(key=lambda r: r["step_idx"])
    print(f"Loaded {len(recs)} state records for {TASK_ID}")

    gt = GroundTruth.load(TASK_ID, mode="full")
    blocks = parse_blocks(gt.compiled_solution_code)
    print(f"Expert blocks: {len(blocks)}")

    jobs = []
    for r in recs:
        prior = blocks[: r["step_idx"]]
        for c in r["calls"]:
            if not c.get("call"):
                continue
            jobs.append({
                "step": r["step_idx"],
                "endpoint": (c.get("asked_endpoint") or "?"),
                "app": (c.get("asked_app") or "?"),
                "call": c["call"],
                "prior": prior,
            })
    # Quick concurrency-validation sub-sample: pick 20 probes from each of 10 steps
    import random
    rnd = random.Random(42)
    by_step = {}
    for j in jobs:
        by_step.setdefault(j["step"], []).append(j)
    sub = []
    for s in sorted(by_step):
        sub.extend(rnd.sample(by_step[s], min(20, len(by_step[s]))))
    jobs = sub
    print(f"Total probes (subsampled 20/step): {len(jobs)}  (max_concurrent={MAX_CONCURRENT})")

    results = []
    t_start = time.time()
    n_done = 0

    def work(job):
        resp, err = probe_one(TASK_ID, job["prior"], job["call"])
        return {
            "step": job["step"],
            "app": job["app"],
            "endpoint": job["endpoint"],
            "call": job["call"],
            "response_full": resp,
            "response_len": len(resp) if resp else 0,
            "error": err,
            "bucket": classify(resp),
        }

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            n_done += 1
            if n_done % 100 == 0 or n_done == len(jobs):
                rate = n_done / (time.time() - t_start)
                eta = (len(jobs) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done:>4}/{len(jobs)}]  {rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

    wall = time.time() - t_start
    print(f"\nAll {n_done} probes done in {wall:.1f}s ({wall/60:.1f}min)")

    # Bucket distribution
    print(f"\n{'='*60}\nBucket distribution (all {n_done} probes)\n{'='*60}")
    buckets = collections.Counter(r["bucket"] for r in results)
    for b, c in buckets.most_common():
        print(f"  {b:<22s} {c:>4d}  ({100*c/n_done:.1f}%)")

    # By-step
    print(f"\n{'='*60}\nBy expert step (trajectory phase)\n{'='*60}")
    by_step = {}
    for r in results:
        by_step.setdefault(r["step"], collections.Counter())[r["bucket"]] += 1
    for s in sorted(by_step):
        bs = by_step[s]
        total = sum(bs.values())
        good = sum(c for b, c in bs.items() if not b.startswith("py_") and b not in ("probe_crash", "empty"))
        print(f"  step {s:>2}: n={total:>3}  env-meaningful={good}/{total} ({100*good/total:.0f}%)  "
              f"top3: {dict(bs.most_common(3))}")

    # Response length stats
    lens = sorted(r["response_len"] for r in results if r["response_len"] > 0)
    if lens:
        def pct(p): return lens[int(len(lens)*p)]
        print(f"\n{'='*60}\nResponse length (non-zero, chars)\n{'='*60}")
        print(f"  n={len(lens)}  min={lens[0]}  p25={pct(0.25)}  median={pct(0.5)}  "
              f"p75={pct(0.75)}  p90={pct(0.9)}  max={lens[-1]}")

    # Compare against naive baseline at step 5 (where naive was sampled)
    print(f"\n{'='*60}\nComparison to naive baseline at step 5\n{'='*60}")
    print("Naive (no-args, 5 random alts) at step 5 (recon_naive_alts):")
    print("  expected types: ~89% errors (mostly 401/422), ~11% data")
    step5 = [r for r in results if r["step"] == 5]
    if step5:
        b5 = collections.Counter(r["bucket"] for r in step5)
        for b, c in b5.most_common():
            print(f"  LLM-filled step 5: {b:<22s} {c:>3d}  ({100*c/len(step5):.1f}%)")

    # Errors
    n_err = sum(1 for r in results if r["error"])
    if n_err:
        print(f"\n=== probe crashes: {n_err} ===")
        for r in results[:5]:
            if r["error"]:
                print(f"  step={r['step']} ep={r['endpoint']} err={r['error']}")

    # Save full
    json.dump({
        "task_id": TASK_ID,
        "n_probes": n_done,
        "wall_seconds": round(wall, 1),
        "buckets": dict(buckets),
        "results": results,
    }, OUT_PATH.open("w"), indent=2, default=str)
    print(f"\nFull output → {OUT_PATH}")


if __name__ == "__main__":
    main()

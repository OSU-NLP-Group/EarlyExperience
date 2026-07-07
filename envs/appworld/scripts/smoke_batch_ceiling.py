"""Step 2: gated LLM smoke — DeepSeek-v4-flash batched-fill quality ceiling.

Design B (coverage): for one representative AppWorld state, shuffle the full
endpoint pool and partition into chunks of size B. Send each chunk as one
batched JSON call. Repeat with 3 different shuffles per batch size to see
variance. Across reps, ALL endpoints are covered (each rep covers the pool
fully). This mirrors the full-pipeline usage rather than just sampling.

Metrics per call:
  - completeness: returned count vs asked
  - well-formed:  each entry parses as `apis.X.Y(args)`
  - correct_ep:   each entry uses the asked app.endpoint (no hallucination/swap)
  - uses_scope:   any arg references a known scope var (heuristic)
  - covered_req:  % of REQUIRED params filled

Cost gate already approved by user.
"""
import os, sys, json, ast, random, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)

from openai import OpenAI
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

DEEPSEEK_MODEL = "deepseek-v4-flash"
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

DOCS_DIR = Path(os.environ["APPWORLD_ROOT"]) / "data" / "api_docs" / "standard"
APP_DOCS = {f.stem: json.load(f.open()) for f in DOCS_DIR.glob("*.json")}
OUT_DIR = Path("envs/appworld/data/_recon")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_endpoint_schema(app, ep, schema):
    parts = [f"## {app}.{ep}"]
    if isinstance(schema, dict):
        if schema.get("description"):
            parts.append(f"  description: {schema['description'][:200]}")
        params = schema.get("parameters", [])
        if isinstance(params, list) and params:
            parts.append("  parameters:")
            for p in params:
                pname = p.get("name", "?")
                ptype = p.get("type", "?")
                preq = p.get("required", p.get("is_required", False))
                pdesc = (p.get("description") or "")[:80]
                marker = "REQUIRED" if preq else "optional"
                parts.append(f"    - {pname} ({ptype}, {marker}): {pdesc}")
        elif isinstance(params, list):
            parts.append("  parameters: (none)")
    return "\n".join(parts)


SYSTEM_PROMPT = """You are filling argument values for a batch of Python API calls in the AppWorld environment.

For each endpoint listed under "Endpoints to fill", output one entry in a JSON array. Each entry must be:
  {"endpoint": "<app>.<ep>", "call": "apis.<app>.<ep>(<args>)"}

Rules:
- Use variables already in REPL scope (listed below) for arg values where possible.
  For example if `access_token` is in scope, pass `access_token=access_token`.
- For string literals, use plausible values that fit the API's parameter type.
- For integer ids, prefer scope variables; else use 1 as a placeholder.
- Return EXACTLY one entry per requested endpoint, in the same order, no extras.
- Output a single JSON object: {"calls": [<entry>, ...]}. No prose, no markdown."""


def build_user_prompt(state_info, batch):
    lines = [
        "# Task instruction",
        state_info["instruction"],
        "",
        "# Expert's prior steps already executed in this REPL (most recent at bottom):",
    ]
    for i, code in enumerate(state_info["prior_code"]):
        lines.append(f"## step {i}")
        lines.append(code)
        lines.append("")
    lines.extend([
        "# Live variables now in REPL scope (name : type):",
        json.dumps(state_info["scope_user_vars"], indent=2),
        "",
        f"# Endpoints to fill ({len(batch)} total — return exactly {len(batch)} JSON entries):",
        "",
    ])
    for app, ep, schema in batch:
        lines.append(render_endpoint_schema(app, ep, schema))
        lines.append("")
    return "\n".join(lines)


def parse_call_string(s):
    """Parse 'apis.X.Y(args)' or 'print(apis.X.Y(args))'. Return (app, ep, n_kwargs) or None."""
    try:
        tree = ast.parse(s.strip(), mode="exec")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute):
                inner = f.value
                if isinstance(inner.value, ast.Name) and inner.value.id == "apis":
                    return inner.attr, f.attr, len(node.keywords), node
    return None


def score_response(asked, response_json, scope_var_names):
    """Compute per-batch metrics from parsed JSON response."""
    asked_set = [(a, e) for a, e, s in asked]
    asked_keys = [f"{a}.{e}" for a, e in asked_set]
    asked_dict = {f"{a}.{e}": s for a, e, s in asked}

    calls = response_json.get("calls", []) if isinstance(response_json, dict) else []
    n_returned = len(calls)

    n_well_formed = 0
    n_correct_ep = 0
    n_used_scope = 0
    n_covered_req = 0
    n_req_total = 0
    detail = []

    for idx, entry in enumerate(calls):
        if not isinstance(entry, dict): continue
        call_str = entry.get("call", "")
        ep_claimed = entry.get("endpoint", "")

        parsed = parse_call_string(call_str)
        if parsed is None:
            detail.append({"idx": idx, "asked": asked_keys[idx] if idx < len(asked_keys) else None,
                           "claimed": ep_claimed, "call": call_str, "issue": "parse_fail"})
            continue
        n_well_formed += 1
        app, ep, n_kw, ast_call = parsed
        actual_key = f"{app}.{ep}"
        # Correct endpoint: matches the asked at this index
        if idx < len(asked_keys) and actual_key == asked_keys[idx]:
            n_correct_ep += 1
        # Uses scope?
        kw_src = []
        for kw in ast_call.keywords:
            try:
                v_src = ast.unparse(kw.value)
            except Exception:
                v_src = "?"
            kw_src.append((kw.arg, v_src))
        for arg_name, arg_src in kw_src:
            if arg_src in scope_var_names or arg_src.split("[")[0] in scope_var_names:
                n_used_scope += 1
                break
        # Required-param coverage
        schema = asked_dict.get(actual_key) or (asked_dict.get(asked_keys[idx]) if idx < len(asked_keys) else None)
        if schema and isinstance(schema.get("parameters"), list):
            req_params = [p["name"] for p in schema["parameters"]
                          if p.get("required") or p.get("is_required")]
            n_req_total += len(req_params)
            filled = {a for a, _ in kw_src}
            n_covered_req += len(filled & set(req_params))

    return {
        "asked_n": len(asked),
        "returned_n": n_returned,
        "completeness": n_returned / len(asked) if asked else 0,
        "well_formed": n_well_formed,
        "correct_endpoint": n_correct_ep,
        "uses_scope": n_used_scope,
        "covered_req": n_covered_req,
        "req_total": n_req_total,
        "detail_issues": [d for d in detail if d.get("issue")],
    }


def call_deepseek(system, user, temperature=1.0):
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def parse_blocks(src):
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solution"), None)
    lines = src.splitlines()
    indent = len(lines[fn.body[0].lineno - 1]) - len(lines[fn.body[0].lineno - 1].lstrip())
    blocks = []
    for stmt in fn.body:
        chunk = "\n".join(
            l[indent:] if len(l) >= indent else l.lstrip()
            for l in lines[stmt.lineno-1:stmt.end_lineno]
        ).strip()
        if chunk: blocks.append(chunk)
    return blocks


def main():
    from appworld.ground_truth import GroundTruth

    task_id = "82e2fac_1"
    step_idx = 5

    gt = GroundTruth.load(task_id, mode="full")
    blocks = parse_blocks(gt.compiled_solution_code)
    specs = json.load((Path(os.environ["APPWORLD_ROOT"]) / "data" / "tasks" / task_id / "specs.json").open())
    report = json.load(open("envs/appworld/data/_recon/naive_alts_report.json"))
    t0 = next(t for t in report["tasks"] if t["task_id"] == task_id)
    state = t0["steps"][step_idx]

    BLACKLIST = {
        "In","Out","get_ipython","exit","quit","open","builtins","calendar",
        "datetime","date","time","timedelta","json","math","random","re","string","sys",
        "os","Path","pathlib","collections","statistics","itertools","Counter","defaultdict",
        "Iterator","deepcopy","reduce","pendulum",
        "MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY",
        "ApiCollection","Date","DateTime","Time","Requester",
        "print","input",
        "apis","requester","environment",
    }
    scope_user_vars = {k: v for k, v in state["scope"].items()
                       if k not in BLACKLIST and not k.startswith("_")}
    scope_var_names = set(scope_user_vars.keys())

    state_info = {
        "instruction": specs["instruction"],
        "prior_code": blocks[:step_idx],
        "scope_user_vars": scope_user_vars,
    }

    # Build pool excluding expert (None at this step, but kept for generality)
    pool = []
    for app in list(t0["required_apps"]) + ["supervisor", "api_docs"]:
        if app in APP_DOCS:
            for ep, schema in APP_DOCS[app].items():
                pool.append((app, ep, schema))
    expert_app, expert_ep = state["expert_call"]["app"], state["expert_call"]["endpoint"]
    candidates = [(a, e, s) for a, e, s in pool if not (a == expert_app and e == expert_ep)]
    pool_size = len(candidates)
    print(f"=== State: task={task_id}, step={step_idx} ===")
    print(f"  pool_size: {pool_size}, scope_user_vars: {sorted(scope_var_names)}")
    print()

    BATCH_SIZES = [10, 30, 50, 80, 100]
    N_REPS = 3
    SEEDS = [42, 43, 44]
    MAX_CONCURRENT = 8

    # Build job list: each job = one DeepSeek call on one chunk
    jobs = []
    for B in BATCH_SIZES:
        for rep, seed in enumerate(SEEDS[:N_REPS]):
            rnd = random.Random(seed)
            shuffled = candidates[:]
            rnd.shuffle(shuffled)
            for chunk_idx in range(0, len(shuffled), B):
                chunk = shuffled[chunk_idx : chunk_idx + B]
                jobs.append({
                    "batch_size": B,
                    "rep": rep,
                    "seed": seed,
                    "chunk_idx": chunk_idx // B,
                    "chunk_actual_size": len(chunk),
                    "chunk": chunk,
                })
    print(f"Total jobs: {len(jobs)}  (parallel={MAX_CONCURRENT})")
    print(f"  by batch_size: " + ", ".join(
        f"B={B}:{sum(1 for j in jobs if j['batch_size']==B)}" for B in BATCH_SIZES))
    print()

    all_results = [None] * len(jobs)
    total_in_tok, total_out_tok = 0, 0
    n_done = 0
    t_start = time.time()

    def run_job(i, job):
        user = build_user_prompt(state_info, job["chunk"])
        t0_call = time.time()
        try:
            resp = call_deepseek(SYSTEM_PROMPT, user, temperature=1.0)
            content = resp["content"]
            usage = resp["usage"]
            err = None
        except Exception as e:
            content = None
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            err = str(e)
        dt = time.time() - t0_call

        metrics = {}
        parsed_resp = None
        if content:
            try:
                parsed_resp = json.loads(content)
                metrics = score_response(job["chunk"], parsed_resp, scope_var_names)
            except json.JSONDecodeError as e:
                metrics = {"json_decode_error": str(e)}

        return i, {
            "batch_size": job["batch_size"],
            "rep": job["rep"],
            "seed": job["seed"],
            "chunk_idx": job["chunk_idx"],
            "chunk_actual_size": job["chunk_actual_size"],
            "wall_seconds": round(dt, 1),
            "usage": usage,
            "error": err,
            "asked_endpoints": [f"{a}.{e}" for a, e, _ in job["chunk"]],
            "raw_response": content,
            "parsed": parsed_resp,
            "metrics": metrics,
        }

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = {ex.submit(run_job, i, j): i for i, j in enumerate(jobs)}
        for f in as_completed(futures):
            i, result = f.result()
            all_results[i] = result
            total_in_tok += result["usage"]["prompt_tokens"]
            total_out_tok += result["usage"]["completion_tokens"]
            n_done += 1
            m = result.get("metrics") or {}
            tag = (f"B={result['batch_size']:>3} rep={result['rep']} "
                   f"chunk={result['chunk_idx']}({result['chunk_actual_size']})")
            if "returned_n" in m:
                cov_pct = (100 * m["covered_req"] / m["req_total"]) if m["req_total"] else 0
                summary = (f"compl={m['returned_n']}/{m['asked_n']} "
                           f"wf={m['well_formed']} ce={m['correct_endpoint']} "
                           f"req_cov={cov_pct:.0f}%")
            elif result["error"]:
                summary = f"ERROR: {result['error'][:80]}"
            else:
                summary = f"json_decode_fail: {m}"
            print(f"  [{n_done:>2}/{len(jobs)}] {tag:<32s} {result['wall_seconds']:.1f}s  {summary}", flush=True)

    out_path = OUT_DIR / "smoke_batch_ceiling.json"
    with out_path.open("w") as f:
        json.dump({
            "state": {"task_id": task_id, "step_idx": step_idx,
                      "pool_size": pool_size, "scope_vars": list(scope_var_names)},
            "config": {"model": DEEPSEEK_MODEL, "batch_sizes": BATCH_SIZES, "n_reps": N_REPS},
            "totals": {"input_tokens": total_in_tok, "output_tokens": total_out_tok},
            "results": all_results,
        }, f, indent=2, default=str)

    total_wall = time.time() - t_start
    print(f"\nAll {n_done} jobs done in {total_wall:.1f}s wall")

    print(f"\n{'='*78}\nSummary table — aggregates by nominal batch_size\n{'='*78}")
    print(f"{'batch':>6} {'chunks':>7} {'compl%':>8} {'wf%':>6} {'corr_ep%':>9} {'scope%':>8} {'req_cov%':>9}")
    by_batch = {}
    for r in all_results:
        by_batch.setdefault(r["batch_size"], []).append(r)
    for B in sorted(by_batch):
        rs = by_batch[B]
        ms = [r["metrics"] for r in rs if r["metrics"] and "returned_n" in r["metrics"]]
        if not ms: continue
        compl = sum(m["returned_n"]/m["asked_n"] for m in ms) / len(ms) * 100
        wf = sum(m["well_formed"]/max(1, m["returned_n"]) for m in ms) / len(ms) * 100
        ce = sum(m["correct_endpoint"]/max(1, m["returned_n"]) for m in ms) / len(ms) * 100
        sc = sum(m["uses_scope"]/max(1, m["well_formed"]) for m in ms) / len(ms) * 100
        rc = sum((m["covered_req"]/max(1, m["req_total"]))*100 for m in ms) / len(ms)
        print(f"{B:>6} {len(rs):>7} {compl:>7.1f}% {wf:>5.0f}% {ce:>8.0f}% {sc:>7.0f}% {rc:>8.0f}%")

    print(f"\nTotal tokens: in={total_in_tok:,} out={total_out_tok:,}")
    print(f"Estimated Flash cost: ${total_in_tok*0.10/1e6 + total_out_tok*0.40/1e6:.4f}")
    print(f"Full output → {out_path}")


if __name__ == "__main__":
    main()

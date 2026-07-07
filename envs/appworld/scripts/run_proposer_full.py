"""Full-scale LLM proposer pass over all 90 train tasks.

For every (task, step) in train:
  - pool = endpoints of required_apps + supervisor + api_docs (~100)
  - shuffle pool with deterministic seed
  - partition into batch_size=10 chunks → ~11 LLM calls per state
  - each call: DeepSeek V4 Flash, JSON output, fills args for 10 endpoints
  - stitch chunks back to one full-pool record per state

Output: envs/appworld/data/rollout/proposer_full.jsonl  (one line per state)
Resumable: skips any (task_id, step_idx) already present in output.
Streaming: each completed state appends one line immediately.

NO env-side probing in this phase. Scope is reconstructed from prior expert
code via static AST analysis. (LLM was 100% well-formed under same prompt
shape in the smoke; static scope is a close approximation of env-live scope.)
"""
import os, sys, json, ast, random, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)

from openai import OpenAI
from appworld import load_task_ids
from appworld.ground_truth import GroundTruth

DEEPSEEK_MODEL = "deepseek-v4-flash"
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

DOCS_DIR = Path(os.environ["APPWORLD_ROOT"]) / "data" / "api_docs" / "standard"
APP_DOCS = {f.stem: json.load(f.open()) for f in DOCS_DIR.glob("*.json")}

OUT_PATH = Path("envs/appworld/data/rollout/proposer_full.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 10
SEED = 42
MAX_CONCURRENT = 20
N_RETRIES = 3
RETRY_BACKOFF = (1.0, 3.0, 8.0)


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
        "# Live variables now in REPL scope (name : type-guess from static analysis):",
        json.dumps(state_info["scope_user_vars"], indent=2),
        "",
        f"# Endpoints to fill ({len(batch)} total — return exactly {len(batch)} JSON entries):",
        "",
    ])
    for app, ep, schema in batch:
        lines.append(render_endpoint_schema(app, ep, schema))
        lines.append("")
    return "\n".join(lines)


def parse_blocks(src):
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solution"), None)
    if fn is None:
        return []
    lines = src.splitlines()
    indent = len(lines[fn.body[0].lineno - 1]) - len(lines[fn.body[0].lineno - 1].lstrip())
    blocks = []
    for stmt in fn.body:
        chunk = "\n".join(
            l[indent:] if len(l) >= indent else l.lstrip()
            for l in lines[stmt.lineno - 1 : stmt.end_lineno]
        ).strip()
        if chunk:
            blocks.append(chunk)
    return blocks


def _guess_type(node):
    if isinstance(node, ast.Constant):
        return type(node.value).__name__
    if isinstance(node, ast.List):     return "list"
    if isinstance(node, ast.Dict):     return "dict"
    if isinstance(node, ast.Set):      return "set"
    if isinstance(node, ast.Tuple):    return "tuple"
    if isinstance(node, ast.ListComp): return "list"
    if isinstance(node, ast.DictComp): return "dict"
    if isinstance(node, ast.SetComp):  return "set"
    if isinstance(node, ast.Subscript):return "?"
    if isinstance(node, ast.Call):     return "?"
    if isinstance(node, ast.Attribute):return "?"
    return "?"


def static_scope(prior_code_blocks):
    """Walk all prior code blocks, collect user-defined var names + crude types."""
    scope = {}
    for code in prior_code_blocks:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                t = _guess_type(node.value)
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        scope[tgt.id] = t
                    elif isinstance(tgt, (ast.Tuple, ast.List)):
                        for elt in tgt.elts:
                            if isinstance(elt, ast.Name):
                                scope[elt.id] = "?"
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                scope.setdefault(node.target.id, "?")
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                scope[node.target.id] = "?"
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                scope[node.target.id] = "iter_var"
            elif isinstance(node, ast.For) and isinstance(node.target, (ast.Tuple, ast.List)):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        scope[elt.id] = "?"
    return scope


def build_pool(required_apps):
    pool = []
    for app in list(required_apps) + ["supervisor", "api_docs"]:
        if app in APP_DOCS:
            for ep, schema in APP_DOCS[app].items():
                pool.append((app, ep, schema))
    return pool


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
        },
    }


def call_with_retry(system, user):
    last_err = None
    for attempt in range(N_RETRIES):
        try:
            return call_deepseek(system, user), None
        except Exception as e:
            last_err = str(e)
            if attempt < N_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
    return None, last_err


def run_one_state(task_id, step_idx, prior_code, instruction, pool):
    scope = static_scope(prior_code)
    state_info = {
        "instruction": instruction,
        "prior_code": prior_code,
        "scope_user_vars": scope,
    }

    rnd = random.Random(SEED)
    shuffled = pool[:]
    rnd.shuffle(shuffled)

    all_calls = []
    total_in, total_out = 0, 0
    errors = []

    for chunk_start in range(0, len(shuffled), BATCH_SIZE):
        chunk = shuffled[chunk_start : chunk_start + BATCH_SIZE]
        user = build_user_prompt(state_info, chunk)
        resp, err = call_with_retry(SYSTEM_PROMPT, user)
        if err is not None:
            errors.append({"chunk_start": chunk_start, "kind": "api_fail", "msg": err})
            for app, ep, _ in chunk:
                all_calls.append({
                    "endpoint": f"{app}.{ep}", "call": None,
                    "asked_app": app, "asked_endpoint": ep,
                    "_chunk_error": err[:200],
                })
            continue

        total_in += resp["usage"]["prompt_tokens"]
        total_out += resp["usage"]["completion_tokens"]

        try:
            parsed = json.loads(resp["content"])
            if isinstance(parsed, dict):
                calls = parsed.get("calls", [])
            elif isinstance(parsed, list):
                calls = parsed
            else:
                errors.append({"chunk_start": chunk_start, "kind": "unexpected_json_type",
                               "msg": type(parsed).__name__})
                calls = []
        except json.JSONDecodeError as e:
            errors.append({"chunk_start": chunk_start, "kind": "json_decode", "msg": str(e)})
            calls = []
        except Exception as e:
            errors.append({"chunk_start": chunk_start, "kind": "parse_other", "msg": str(e)[:200]})
            calls = []

        for i, (app, ep, _) in enumerate(chunk):
            if i < len(calls) and isinstance(calls[i], dict):
                entry = calls[i]
                entry["asked_app"] = app
                entry["asked_endpoint"] = ep
                all_calls.append(entry)
            else:
                all_calls.append({
                    "endpoint": f"{app}.{ep}", "call": None,
                    "asked_app": app, "asked_endpoint": ep,
                    "_chunk_error": "missing_in_response",
                })

    return {
        "task_id": task_id,
        "step_idx": step_idx,
        "n_pool": len(pool),
        "n_returned": sum(1 for c in all_calls if c.get("call")),
        "scope": scope,
        "calls": all_calls,
        "usage": {"prompt_tokens": total_in, "completion_tokens": total_out,
                  "total_tokens": total_in + total_out},
        "errors": errors,
    }


def load_done_keys():
    if not OUT_PATH.exists():
        return set()
    done = set()
    with OUT_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["step_idx"]))
            except Exception:
                continue
    return done


def main():
    done = load_done_keys()
    print(f"Resumability: {len(done)} (task,step) records already in {OUT_PATH}")

    train_ids = load_task_ids("train")
    jobs = []
    for tid in train_ids:
        gt = GroundTruth.load(tid, mode="full")
        if not gt.compiled_solution_code:
            continue
        blocks = parse_blocks(gt.compiled_solution_code)
        specs = json.load((Path(os.environ["APPWORLD_ROOT"]) / "data" / "tasks" / tid / "specs.json").open())
        pool = build_pool(gt.required_apps)
        for i in range(len(blocks)):
            if (tid, i) in done:
                continue
            jobs.append({
                "task_id": tid,
                "step_idx": i,
                "prior_code": blocks[:i],
                "instruction": specs["instruction"],
                "pool": pool,
            })

    total_chunks = sum((len(j["pool"]) + BATCH_SIZE - 1) // BATCH_SIZE for j in jobs)
    print(f"Jobs: {len(jobs)} states  →  ~{total_chunks} LLM calls (batch={BATCH_SIZE})")
    if not jobs:
        print("Nothing to do.")
        return

    out_f = OUT_PATH.open("a", buffering=1)
    write_lock = threading.Lock()

    n_done = 0
    t_start = time.time()
    total_in, total_out = 0, 0

    def work(job):
        return run_one_state(
            job["task_id"], job["step_idx"],
            job["prior_code"], job["instruction"], job["pool"],
        )

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = {ex.submit(work, j): j for j in jobs}
        for f in as_completed(futures):
            try:
                r = f.result()
            except Exception as e:
                print(f"  WORKER CRASH: {e}", flush=True)
                continue

            with write_lock:
                out_f.write(json.dumps(r, default=str) + "\n")
                out_f.flush()

            total_in += r["usage"]["prompt_tokens"]
            total_out += r["usage"]["completion_tokens"]
            n_done += 1
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta_s = (len(jobs) - n_done) / rate if rate > 0 else 0
            cost = total_in * 0.10 / 1e6 + total_out * 0.40 / 1e6
            err_tag = f" ERR={len(r['errors'])}" if r["errors"] else ""
            print(
                f"  [{n_done:>4}/{len(jobs)}] {r['task_id']}:{r['step_idx']:>2} "
                f"n={r['n_returned']}/{r['n_pool']}{err_tag} "
                f"toks_state={r['usage']['total_tokens']:>5}  "
                f"rate={rate*60:.1f}/min  eta={eta_s/60:.0f}min  cost=${cost:.2f}",
                flush=True,
            )

    out_f.close()
    print(f"\nDONE. n_states_done={n_done}  total in={total_in:,} out={total_out:,}  "
          f"cost=${total_in*0.10/1e6 + total_out*0.40/1e6:.2f}  "
          f"wall={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()

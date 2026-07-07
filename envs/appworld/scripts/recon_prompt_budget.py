"""Step 1: zero-cost prompt-budget recon.

For a representative AppWorld state (task 82e2fac_1, step 5 — mid-trajectory,
scope already has login + access_token), build the actual batched-fill prompt
at several batch_sizes. Report:
  - input/output token count per call
  - total tokens needed to cover the FULL pool at each batch_size
  - prompt preview so we can see what the LLM will actually receive

NO LLM calls. Pure local prompt construction + tiktoken counting.
"""
import os, json, ast, random
from pathlib import Path

os.environ.setdefault(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
def n_toks(s): return len(enc.encode(s))

DOCS_DIR = Path(os.environ["APPWORLD_ROOT"]) / "data" / "api_docs" / "standard"
APP_DOCS = {f.stem: json.load(f.open()) for f in DOCS_DIR.glob("*.json")}


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

For each endpoint listed under "Endpoints to fill", output ONE LINE of executable Python in the format:
  print(apis.<app>.<endpoint>(<args>))

Rules:
- Use the variables already in REPL scope (listed below) for arg values where possible.
  For example if you see `access_token` in scope, pass `access_token=access_token`.
- For string literals (queries, names), use plausible values that fit the API's parameter type.
- For integer ids, prefer scope variables; if none fit, use 1 as a placeholder.
- Output exactly one line per endpoint, in the same order as listed. No prose, no markdown, no blank lines."""


def build_user_prompt(state_info, batch):
    lines = [
        f"# Task instruction",
        state_info["instruction"],
        "",
        f"# Expert's prior steps already executed in this REPL (most recent at bottom):",
    ]
    for i, code in enumerate(state_info["prior_code"]):
        lines.append(f"## step {i}")
        lines.append(code)
        lines.append("")
    lines.extend([
        f"# Live variables now in REPL scope (name : type):",
        json.dumps(state_info["scope_user_vars"], indent=2),
        "",
        f"# Endpoints to fill ({len(batch)} total — output exactly {len(batch)} lines, one per endpoint):",
        "",
    ])
    for app, ep, schema in batch:
        lines.append(render_endpoint_schema(app, ep, schema))
        lines.append("")
    return "\n".join(lines)


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

    specs_path = Path(os.environ["APPWORLD_ROOT"]) / "data" / "tasks" / task_id / "specs.json"
    specs = json.load(specs_path.open())

    # Pull scope from previous recon's saved report (don't re-probe env)
    report = json.load(open("envs/appworld/data/_recon/naive_alts_report.json"))
    t0 = next(t for t in report["tasks"] if t["task_id"] == task_id)
    state = t0["steps"][step_idx]
    raw_scope = state["scope"]

    # Filter to user-defined vars (drop IPython/builtins/imports)
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
    scope_user_vars = {
        k: v for k, v in raw_scope.items()
        if k not in BLACKLIST and not k.startswith("_")
    }
    print(f"=== State: task={task_id}, step={step_idx} ===")
    print(f"  required_apps: {t0['required_apps']}")
    print(f"  pool size:     {t0['pool_size']}")
    print(f"  prior steps:   {step_idx}")
    print(f"  scope user vars ({len(scope_user_vars)}):")
    for k, v in scope_user_vars.items():
        print(f"    {k:<30s} : {v}")
    print()

    state_info = {
        "instruction": specs["instruction"],
        "prior_code": blocks[:step_idx],
        "scope_user_vars": scope_user_vars,
    }

    # Build pool (exclude expert)
    pool = []
    for app in list(t0["required_apps"]) + ["supervisor", "api_docs"]:
        if app in APP_DOCS:
            for ep, schema in APP_DOCS[app].items():
                pool.append((app, ep, schema))
    expert_app, expert_ep = state["expert_call"]["app"], state["expert_call"]["endpoint"]
    candidates = [(a, e, s) for a, e, s in pool if not (a == expert_app and e == expert_ep)]
    print(f"  candidates (excl expert {expert_app}.{expert_ep}): {len(candidates)}")
    print()

    sys_tok = n_toks(SYSTEM_PROMPT)
    print(f"=== System prompt: {sys_tok} tokens ===")

    # Static context length (everything but the endpoint schemas)
    static_ctx_user = build_user_prompt(state_info, [])
    static_ctx_tok = n_toks(static_ctx_user)
    print(f"=== Static context (instruction+prior+scope, no endpoints): {static_ctx_tok} tokens ===\n")

    # Per-batch profiles
    pool_size = len(candidates)
    print(f"{'batch':>6} {'calls':>6} {'in/call':>10} {'out/call':>10} {'total_in':>12} {'total_out':>12} {'≈ Flash $':>10}")
    print("-" * 75)
    rows = []
    for B in [5, 10, 20, 50, 100]:
        random.seed(42)
        # Sample B endpoints (deterministic for measurement)
        batch = candidates[:B]
        user = build_user_prompt(state_info, batch)
        per_in = sys_tok + n_toks(user)
        per_out = B * 30  # rough: each filled call ~30 tokens
        n_calls = (pool_size + B - 1) // B
        total_in = per_in * n_calls
        total_out = per_out * n_calls
        # Flash pricing assumption: $0.10/M in, $0.40/M out
        flash_cost = total_in * 0.10 / 1e6 + total_out * 0.40 / 1e6
        print(f"{B:>6} {n_calls:>6} {per_in:>10,} {per_out:>10,} {total_in:>12,} {total_out:>12,}  ${flash_cost:>7.4f}")
        rows.append((B, n_calls, per_in, per_out, total_in, total_out, flash_cost))

    # Project to full 931-state dataset, picking a "middle" batch_size for cost projection
    print()
    print(f"=== Projected cost for full 931-state dataset ===")
    print(f"(Using this state's prompt size as representative; actual will vary.)")
    print(f"{'batch':>6} {'≈ Flash $':>10} {'≈ Pro $':>10} {'note':<40}")
    for B, n_calls, per_in, per_out, total_in, total_out, _ in rows:
        full_in = total_in * 931
        full_out = total_out * 931
        flash = full_in * 0.10/1e6 + full_out * 0.40/1e6
        pro = full_in * 0.27/1e6 + full_out * 1.10/1e6
        note = "single call/state" if B >= pool_size else f"{n_calls} calls/state"
        print(f"{B:>6}  ${flash:>7.2f}  ${pro:>7.2f}  {note}")

    # Save the batch=10 prompt as a sample for human inspection
    sample_path = Path("envs/appworld/data/_recon/prompt_sample_batch10.txt")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_b10 = build_user_prompt(state_info, candidates[:10])
    with sample_path.open("w") as f:
        f.write("=== SYSTEM ===\n" + SYSTEM_PROMPT + "\n\n=== USER ===\n" + sample_b10)
    print(f"\nSample prompt (batch=10) saved → {sample_path}")
    print(f"\n--- USER prompt preview (batch=10, tail 1000 chars) ---")
    print(sample_b10[-1000:])


if __name__ == "__main__":
    main()

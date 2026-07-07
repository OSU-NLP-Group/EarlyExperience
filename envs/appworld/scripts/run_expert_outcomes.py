"""Capture + summarize the expert action outcome at every step of every train task.

Two phases in one script:
  Phase 1  — env replay (30 parallel appworld servers). For each train task,
             assign one server, run the expert trajectory sequentially in a
             single AppWorld session. At each step, execute the expert block
             AND capture what it produced (see `capture_outcome_stmt` below).
             Output: raw outcome per (task_id, step).
  Phase 2  — Flash summarizer (same prompt/model as run_summary_full.py).
             Wraps every raw outcome into a one-sentence NL summary.
             Output: envs/appworld/data/rollout/expert_outcomes.jsonl

Capture convention (matches how alt outcomes were captured, so IWM/SR see
symmetric information between expert action and alts):

  block is Assign  `x = apis.X.Y(...)`  → execute block (binds x, mutates env)
                                          then execute `print(x)` to capture return
  block is bare Expr `apis.X.Y(...)`    → execute `print((expr))` once (single call)
  block is control flow (For/If/While)  → execute block as-is;
                                          outcome = "Execution successful."
  block is anything else                → execute; outcome = "Execution successful."

Cost:
  Phase 1: $0. ~90 trajectories * ~20 execs ≈ 1800 HTTP calls, ~2 min at 30 servers.
  Phase 2: 931 Flash calls (~700 in + ~80 out per call) ≈ ~730k tokens ≈ ~$0.10, ~1 min at 30 workers.
"""
import os, sys, json, ast, time, uuid, subprocess, socket, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)
from appworld import AppWorld, load_task_ids
from appworld.ground_truth import GroundTruth
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
FLASH = "deepseek-v4-flash"

APPWORLD_ROOT = os.environ["APPWORLD_ROOT"]
APPWORLD_BIN = "/home/ulss/miniconda3/envs/appworld/bin/appworld"
SERVER_LOG_DIR = Path("/tmp/appworld_servers"); SERVER_LOG_DIR.mkdir(exist_ok=True)

N_SERVERS = 6              # was 30 — machine downsized to 12 cores / 23GB RAM
PORT_START = 7500
SUMMARIZE_WORKERS = 20     # Flash summary I/O bound, safe to keep parallel high
MAX_RAW_CHARS = 3000

OUT_PATH = Path("envs/appworld/data/rollout/expert_outcomes.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------- ast helpers ----------
def parse_blocks(src):
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solution"), None)
    if not fn: return []
    lines = src.splitlines()
    indent = len(lines[fn.body[0].lineno-1]) - len(lines[fn.body[0].lineno-1].lstrip())
    out = []
    for stmt in fn.body:
        chunk = "\n".join(l[indent:] if len(l) >= indent else l.lstrip()
                          for l in lines[stmt.lineno-1:stmt.end_lineno]).strip()
        if chunk: out.append(chunk)
    return out


def capture_outcome_stmt(block):
    """Return (execute_stmt, follow_up_stmt_or_None).
    If follow_up_stmt is None, outcome = env.execute(execute_stmt).
    If follow_up_stmt is a string, first run execute_stmt for side effects then
    return env.execute(follow_up_stmt) as outcome.
    If both are None: control flow / non-capturable — outcome hard-coded 'Execution successful.'
    """
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return (block, None)  # broken code: just run it, take whatever env says
    if len(tree.body) != 1:
        return (block, None)  # multi-stmt block; run as-is
    node = tree.body[0]
    # Assignment: x = ...  → run it, then print(x)
    if isinstance(node, ast.Assign):
        # single-target simple Name
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return (block, f"print({node.targets[0].id})")
        # tuple unpack / subscript / attr — skip capture
        return (block, None)
    # Bare expression: print-wrap once
    if isinstance(node, ast.Expr):
        # If already a top-level print(...), don't double-wrap
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) \
                and node.value.func.id == "print":
            return (block, None)
        return (f"print(({block}))", None)
    # Control flow / anything else: run as-is
    return (block, None)


# ---------- server lifecycle ----------
def start_server(port):
    log = open(SERVER_LOG_DIR / f"port_{port}.log", "w")
    return subprocess.Popen(
        [APPWORLD_BIN, "serve", "environment", "--port", str(port),
         "--root", APPWORLD_ROOT, "--no-show-usage"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def wait_ready(port, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def start_all_servers(n, port_start):
    print(f"Spawning {n} servers on ports {port_start}..{port_start+n-1}")
    procs = {p: start_server(p) for p in range(port_start, port_start + n)}
    ready = []
    for p in sorted(procs):
        if wait_ready(p, timeout=120):
            ready.append(p)
        else:
            print(f"  WARN port {p} not ready")
    print(f"  {len(ready)}/{n} ready")
    return procs, ready


def teardown(procs):
    for p in procs.values():
        try: p.terminate()
        except Exception: pass
    time.sleep(2)
    for p in procs.values():
        try: p.kill()
        except Exception: pass


# ---------- phase 1: capture raw outcomes ----------
def replay_trajectory(task_id, blocks, port):
    """Run the expert trajectory in a single fresh env on the given port.
    Return list of raw outcome strings, one per block."""
    url = f"http://0.0.0.0:{port}"
    env = None
    outcomes = []
    try:
        env = AppWorld(task_id=task_id,
                       experiment_name=f"exp_{uuid.uuid4().hex[:8]}",
                       remote_environment_url=url)
        for block in blocks:
            exec_stmt, follow = capture_outcome_stmt(block)
            try:
                first_resp = env.execute(exec_stmt)
                if follow is not None:
                    outcome = env.execute(follow)
                else:
                    outcome = first_resp
            except Exception as e:
                outcome = f"<env_call_failed: {str(e)[:200]}>"
            outcomes.append(outcome)
    except Exception as e:
        return [f"<traj_setup_failed: {str(e)[:200]}>" for _ in blocks]
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
    return outcomes


# ---------- phase 2: summarize (reused from run_summary_full) ----------
SUMMARIZER_SYSTEM = """You translate raw AppWorld tool execution outcomes into concise English. Given a Python API call and its raw return value from a simulated environment, output ONE sentence that states FACTUALLY what the tool did or what data it returned.

Rules:
  - Successful action (state change — song liked, item added, account created, password reset, payment made, etc.) → state what changed, past tense, naming the affected object(s).
  - Read-only call (show_*, search_*, list_*, get_*) → state what data was returned, including key identifiers/values.
  - Error response (HTTP 4xx wrapped in Python Exception, or Python-level error like NameError/TypeError) → restate the error factually, preserving status code and key message.
  - Null/empty return → state the call completed without returning data.

STRICT CONSTRAINTS:
  - DO NOT mention any user, any task, any goal, or whether the call was useful.
  - DO NOT compare to expected outcomes, expert actions, or other alternatives.
  - DO NOT add reasoning, judgment, or commentary about correctness.
  - DO NOT analyze WHY the error happened or suggest fixes.
  - Preserve specific identifiers/values present in the raw response.

Output ONLY the one factual sentence — no quotes, no prefix, no markdown, no explanation."""


def build_summ_user(call_str, raw):
    raw = raw or ""
    tail = "\n…[truncated]…" if len(raw) > MAX_RAW_CHARS else ""
    return (
        f"## Tool call\n`{call_str}`\n\n"
        f"## Raw tool response\n```\n{raw[:MAX_RAW_CHARS]}{tail}\n```\n\n"
        f"## Output\nOne factual sentence per the system rules."
    )


def summarize(call_str, raw):
    user = build_summ_user(call_str, raw)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=FLASH,
                messages=[{"role": "system", "content": SUMMARIZER_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.3,
                extra_body={"thinking": {"type": "disabled"}},
            )
            s = (r.choices[0].message.content or "").strip()
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1].strip()
            return s, {"prompt_tokens": r.usage.prompt_tokens,
                       "completion_tokens": r.usage.completion_tokens}, None
        except Exception as e:
            if attempt < 2:
                time.sleep(1 + attempt * 2)
    return None, {"prompt_tokens": 0, "completion_tokens": 0}, str(e)[:200]


# ---------- main ----------
def main():
    train_ids = load_task_ids("train")
    print(f"Train tasks: {len(train_ids)}")

    # Prepare per-task expert blocks
    tasks = []
    for tid in train_ids:
        gt = GroundTruth.load(tid, mode="full")
        blocks = parse_blocks(gt.compiled_solution_code or "")
        if blocks:
            tasks.append({"task_id": tid, "blocks": blocks})
    print(f"Tasks with expert code: {len(tasks)}  total steps: {sum(len(t['blocks']) for t in tasks)}")

    # ----- phase 1 -----
    procs, ready = start_all_servers(N_SERVERS, PORT_START)
    if not ready:
        teardown(procs); return

    try:
        # RAM snapshot
        try:
            m = subprocess.check_output(["free", "-h"]).decode().splitlines()[1]
            print(f"RAM after servers up:\n  {m}")
        except Exception: pass

        import queue as _q
        port_q = _q.Queue()
        for p in ready: port_q.put(p)

        t0 = time.time()
        raw_outcomes = {}  # task_id -> list[str]
        n_done = 0
        lock = threading.Lock()

        def replay_one(t):
            port = port_q.get()
            try:
                outs = replay_trajectory(t["task_id"], t["blocks"], port)
            finally:
                port_q.put(port)
            return t["task_id"], outs

        with ThreadPoolExecutor(max_workers=len(ready)) as ex:
            futures = [ex.submit(replay_one, t) for t in tasks]
            for f in as_completed(futures):
                tid, outs = f.result()
                with lock:
                    raw_outcomes[tid] = outs
                    n_done += 1
                    if n_done % 10 == 0 or n_done == len(tasks):
                        el = time.time() - t0
                        print(f"  replay [{n_done}/{len(tasks)}] elapsed={el:.1f}s", flush=True)

        phase1_wall = time.time() - t0
        print(f"\nPhase 1 done in {phase1_wall:.1f}s")
    finally:
        teardown(procs)
        print("Servers down.")

    # Flatten to (task_id, step, block, outcome_raw) records
    records = []
    for t in tasks:
        tid = t["task_id"]
        outs = raw_outcomes.get(tid, [])
        for i, b in enumerate(t["blocks"]):
            raw = outs[i] if i < len(outs) else "<missing_outcome>"
            records.append({
                "task_id": tid,
                "step": i,
                "expert_code": b,
                "outcome_raw": raw,
            })
    print(f"Records to summarize: {len(records)}")

    # ----- phase 2 -----
    t0 = time.time()
    out_f = OUT_PATH.open("w", buffering=1)
    write_lock = threading.Lock()
    n_done = 0
    n_err = 0
    tok_in = 0
    tok_out = 0

    def do_summary(rec):
        summary, usage, err = summarize(rec["expert_code"], rec["outcome_raw"])
        merged = dict(rec)
        merged["outcome_summary"] = summary
        merged["summary_error"] = err
        merged["summary_usage"] = usage
        return merged

    with ThreadPoolExecutor(max_workers=SUMMARIZE_WORKERS) as ex:
        futures = [ex.submit(do_summary, r) for r in records]
        for f in as_completed(futures):
            m = f.result()
            with write_lock:
                out_f.write(json.dumps(m, default=str) + "\n")
                n_done += 1
                if m.get("summary_error"): n_err += 1
                tok_in += m["summary_usage"]["prompt_tokens"]
                tok_out += m["summary_usage"]["completion_tokens"]
                if n_done % 100 == 0 or n_done == len(records):
                    el = time.time() - t0
                    rate = n_done / el if el > 0 else 0
                    cost = tok_in*0.10/1e6 + tok_out*0.40/1e6
                    print(f"  summary [{n_done}/{len(records)}] {rate:.1f}/s err={n_err} cost=${cost:.3f}", flush=True)

    out_f.close()
    phase2_wall = time.time() - t0
    cost = tok_in*0.10/1e6 + tok_out*0.40/1e6
    print(f"\nPhase 2 done in {phase2_wall:.1f}s")
    print(f"DONE. records={n_done} err={n_err} tokens_in={tok_in:,} out={tok_out:,} cost=${cost:.3f}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()

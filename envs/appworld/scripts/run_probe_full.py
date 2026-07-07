"""Full env-probe pass — execute every proposer-filled call in fresh probe envs.

For each (task_id, step_idx, endpoint, filled_call) in proposer_full.jsonl,
spin a fresh AppWorld session on one of N parallel servers, replay the expert
prefix, execute the alt call, capture the response. One row per probe to
envs/appworld/data/rollout/probe_full.jsonl.

Architecture:
  - N parallel appworld servers on ports [PORT_START, PORT_START+N)
  - Each server: c=1 (one probe at a time — server has global `world` state, no isolation)
  - Port queue distributes work; N threads = N concurrent probes
  - Streaming JSONL output, line-buffered, resumable on restart

Cost: $0 (no LLM). Wall: ~25 min @ 200 servers (1-task smoke showed 0.285 probe/s/server).
"""
import os, sys, json, ast, time, uuid, re, collections
import queue, threading, subprocess, socket
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)
from appworld import AppWorld
from appworld.ground_truth import GroundTruth

N_SERVERS = int(os.environ.get("N_SERVERS", "200"))
PORT_START = int(os.environ.get("PORT_START", "7200"))
APPWORLD_ROOT = os.environ["APPWORLD_ROOT"]
APPWORLD_BIN = "/home/ulss/miniconda3/envs/appworld/bin/appworld"
SERVER_LOG_DIR = Path("/tmp/appworld_servers")
SERVER_LOG_DIR.mkdir(exist_ok=True)
OUT_PATH = Path("envs/appworld/data/rollout/probe_full.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
IN_PATH = Path("envs/appworld/data/rollout/proposer_full.jsonl")


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


def wrap_for_print(call_str):
    """Wrap a bare apis.X.Y(...) expression in print() so env.execute returns the value.
    LLM proposer outputs are 99.64% bare expressions, 0.36% syntax errors (kept as-is)."""
    s = call_str.strip()
    try:
        ast.parse(s)
        return f"print(({s}))"
    except SyntaxError:
        return s


def probe_one(task_id, prior_code, call_str, port):
    name = f"probe_{uuid.uuid4().hex[:8]}"
    url = f"http://0.0.0.0:{port}"
    env = None
    try:
        env = AppWorld(task_id=task_id, experiment_name=name, remote_environment_url=url)
        for code in prior_code:
            env.execute(code)
        resp = env.execute(wrap_for_print(call_str))
        return resp, None
    except Exception as e:
        return None, str(e)[:200]
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass


def start_server(port):
    log = open(SERVER_LOG_DIR / f"port_{port}.log", "w")
    cmd = [APPWORLD_BIN, "serve", "environment",
           "--port", str(port), "--root", APPWORLD_ROOT, "--no-show-usage"]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def wait_port_ready(port, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    return False


def start_all_servers(n, port_start):
    print(f"Spawning {n} servers on ports {port_start}..{port_start+n-1}")
    procs = {}
    t0 = time.time()
    for i in range(n):
        port = port_start + i
        procs[port] = start_server(port)
    print(f"  spawn done in {time.time()-t0:.1f}s, waiting for ready...")
    ready = []
    t0 = time.time()
    for port in sorted(procs):
        if wait_port_ready(port, timeout=120):
            ready.append(port)
        else:
            print(f"  WARN port {port} not ready in 120s")
    print(f"  {len(ready)}/{n} servers ready in {time.time()-t0:.1f}s")
    return procs, ready


def teardown(procs):
    print(f"Tearing down {len(procs)} servers...")
    for p in procs.values():
        try: p.terminate()
        except Exception: pass
    time.sleep(3)
    n_alive = sum(1 for p in procs.values() if p.poll() is None)
    for p in procs.values():
        try: p.kill()
        except Exception: pass
    print(f"  done ({n_alive} needed SIGKILL)")


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


# Cache expert blocks per task_id to avoid re-parsing
_BLOCKS_CACHE = {}
def get_blocks(task_id):
    if task_id not in _BLOCKS_CACHE:
        gt = GroundTruth.load(task_id, mode="full")
        _BLOCKS_CACHE[task_id] = parse_blocks(gt.compiled_solution_code or "")
    return _BLOCKS_CACHE[task_id]


def main():
    print(f"Loading proposer records from {IN_PATH}...")
    recs = [json.loads(l) for l in IN_PATH.open()]
    print(f"  {len(recs)} state records")

    done = load_done_keys()
    print(f"Already done probes: {len(done)} (resumable, will skip)")

    # Build jobs list
    jobs = []
    for r in recs:
        task_id = r["task_id"]
        step_idx = r["step_idx"]
        blocks = get_blocks(task_id)
        prior = blocks[:step_idx]
        for c in r["calls"]:
            if not c.get("call"): continue
            ep_key = c.get("asked_endpoint") or c.get("endpoint") or "?"
            if (task_id, step_idx, ep_key) in done:
                continue
            jobs.append({
                "task_id": task_id,
                "step": step_idx,
                "endpoint": ep_key,
                "app": c.get("asked_app") or "?",
                "call": c["call"],
                "prior": prior,
            })
    print(f"Jobs to run: {len(jobs)}")
    if not jobs:
        print("Nothing to do."); return

    procs, ready_ports = start_all_servers(N_SERVERS, PORT_START)
    if len(ready_ports) < N_SERVERS:
        print(f"WARN: only {len(ready_ports)} ready (wanted {N_SERVERS})")

    try:
        # Memory snapshot
        try:
            out = subprocess.check_output(["free", "-h"]).decode().splitlines()[:2]
            print(f"\nRAM after {len(ready_ports)} servers up:\n  {out[0]}\n  {out[1]}\n")
        except Exception: pass

        port_q = queue.Queue()
        for p in ready_ports: port_q.put(p)

        out_f = OUT_PATH.open("a", buffering=1)
        write_lock = threading.Lock()
        n_done = 0
        t_start = time.time()
        bucket_counts = collections.Counter()

        def work(job):
            port = port_q.get()
            try:
                resp, err = probe_one(job["task_id"], job["prior"], job["call"], port)
            finally:
                port_q.put(port)
            return {
                "task_id": job["task_id"],
                "step": job["step"],
                "app": job["app"],
                "endpoint": job["endpoint"],
                "call": job["call"],
                "response": resp,
                "response_len": len(resp) if resp else 0,
                "error": err,
                "bucket": classify(resp),
            }

        with ThreadPoolExecutor(max_workers=len(ready_ports)) as ex:
            futures = {ex.submit(work, j): j for j in jobs}
            for f in as_completed(futures):
                try:
                    r = f.result()
                except Exception as e:
                    print(f"  WORKER CRASH: {e}", flush=True); continue
                with write_lock:
                    out_f.write(json.dumps(r, default=str) + "\n")
                    n_done += 1
                    bucket_counts[r["bucket"]] += 1
                    if n_done % 1000 == 0 or n_done == len(jobs):
                        elapsed = time.time() - t_start
                        rate = n_done / elapsed
                        eta_s = (len(jobs) - n_done) / rate if rate > 0 else 0
                        n_crash = bucket_counts.get("probe_crash", 0)
                        print(
                            f"  [{n_done:>6}/{len(jobs)}] {rate:.1f}/s "
                            f"crash={n_crash}({100*n_crash/n_done:.1f}%) "
                            f"eta={eta_s/60:.1f}min", flush=True
                        )

        wall = time.time() - t_start
        out_f.close()

        print(f"\n=== DONE: {n_done} probes in {wall:.1f}s ({wall/60:.1f}min)  rate={n_done/wall:.1f}/s ===")
        print(f"\n=== Bucket distribution ===")
        for b, c in bucket_counts.most_common():
            print(f"  {b:<22s} {c:>6d}  ({100*c/n_done:.1f}%)")
        n_crash = bucket_counts.get("probe_crash", 0)
        n_py = sum(c for b, c in bucket_counts.items() if b.startswith("py_"))
        print(f"\n  clean rate:        {100*(n_done - n_crash)/n_done:.2f}%")
        print(f"  env-meaningful:    {100*(n_done - n_crash - n_py)/n_done:.2f}%")
        print(f"  output: {OUT_PATH}")

    finally:
        teardown(procs)


if __name__ == "__main__":
    main()

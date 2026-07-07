"""Probe smoke with N parallel appworld servers (one worker per port).

Architecture:
  - Start N servers on ports [PORT_START, PORT_START+N)
  - Port queue: workers pull a port, run one probe, push port back
  - This guarantees per-server c=1 (no server-side global-world race),
    while achieving N-way real parallelism across processes
  - Teardown all servers at end

Validates that 100 servers can drive ~84k-probe-scale env_probe phase
in single-digit minutes without losing data to "closed task world" crashes.
"""
import os, sys, json, ast, time, uuid, re, collections
import queue, threading, subprocess, socket, signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)
from appworld import AppWorld
from appworld.ground_truth import GroundTruth

# ----- config -----
N_SERVERS = int(os.environ.get("N_SERVERS", "100"))
PORT_START = int(os.environ.get("PORT_START", "7100"))
TASK_ID = "82e2fac_1"
APPWORLD_ROOT = os.environ["APPWORLD_ROOT"]
APPWORLD_BIN = "/home/ulss/miniconda3/envs/appworld/bin/appworld"
SERVER_LOG_DIR = Path("/tmp/appworld_servers")
SERVER_LOG_DIR.mkdir(exist_ok=True)
OUT_PATH = Path("envs/appworld/data/_recon/probe_smoke_82e2fac_1_100s.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ----- helpers reused from earlier smoke -----
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


def probe_one(task_id, prior_code, call_str, port):
    name = f"probe_{uuid.uuid4().hex[:8]}"
    url = f"http://0.0.0.0:{port}"
    env = None
    try:
        env = AppWorld(task_id=task_id, experiment_name=name, remote_environment_url=url)
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


# ----- server lifecycle -----
def start_server(port):
    log = open(SERVER_LOG_DIR / f"port_{port}.log", "w")
    cmd = [APPWORLD_BIN, "serve", "environment",
           "--port", str(port), "--root", APPWORLD_ROOT, "--no-show-usage"]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def wait_port_ready(port, timeout=90):
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
    # Probe sequentially (each takes <1s once up; total ~30-60s)
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
    time.sleep(2)
    n_alive = 0
    for p in procs.values():
        if p.poll() is None:
            n_alive += 1
            try: p.kill()
            except Exception: pass
    print(f"  {n_alive} needed SIGKILL")


def main():
    procs, ready_ports = start_all_servers(N_SERVERS, PORT_START)
    if len(ready_ports) < N_SERVERS:
        print(f"WARN: only {len(ready_ports)} servers ready (wanted {N_SERVERS})")

    # Memory check
    try:
        out = subprocess.check_output(["free", "-h"]).decode()
        print(f"\nRAM after {len(ready_ports)} servers up:")
        for line in out.splitlines()[:2]:
            print(f"  {line}")
    except Exception:
        pass

    try:
        # ---- load jobs ----
        recs = []
        with open("envs/appworld/data/rollout/proposer_full.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["task_id"] == TASK_ID:
                    recs.append(r)
        recs.sort(key=lambda r: r["step_idx"])
        gt = GroundTruth.load(TASK_ID, mode="full")
        blocks = parse_blocks(gt.compiled_solution_code)

        jobs = []
        for r in recs:
            prior = blocks[: r["step_idx"]]
            for c in r["calls"]:
                if not c.get("call"): continue
                jobs.append({
                    "step": r["step_idx"],
                    "endpoint": (c.get("asked_endpoint") or "?"),
                    "app": (c.get("asked_app") or "?"),
                    "call": c["call"],
                    "prior": prior,
                })
        print(f"\nTotal probes: {len(jobs)}  workers={len(ready_ports)}")

        # ---- port queue: each server handles 1 at a time ----
        port_q = queue.Queue()
        for p in ready_ports:
            port_q.put(p)

        results = []
        t_start = time.time()
        n_done = 0
        lock = threading.Lock()

        def work(job):
            port = port_q.get()
            try:
                resp, err = probe_one(TASK_ID, job["prior"], job["call"], port)
            finally:
                port_q.put(port)
            return {
                "step": job["step"],
                "app": job["app"],
                "endpoint": job["endpoint"],
                "call": job["call"],
                "response_full": resp,
                "response_len": len(resp) if resp else 0,
                "error": err,
                "bucket": classify(resp),
                "port": port,
            }

        with ThreadPoolExecutor(max_workers=len(ready_ports)) as ex:
            futures = [ex.submit(work, j) for j in jobs]
            for f in as_completed(futures):
                r = f.result()
                with lock:
                    results.append(r)
                    n_done += 1
                if n_done % 100 == 0 or n_done == len(jobs):
                    rate = n_done / (time.time() - t_start)
                    eta = (len(jobs) - n_done) / rate if rate > 0 else 0
                    print(f"  [{n_done:>4}/{len(jobs)}]  {rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

        wall = time.time() - t_start
        print(f"\nAll {n_done} probes done in {wall:.1f}s ({wall/60:.2f}min)  rate={n_done/wall:.1f}/s")

        # ---- stats ----
        buckets = collections.Counter(r["bucket"] for r in results)
        print(f"\n=== bucket distribution ===")
        for b, c in buckets.most_common():
            print(f"  {b:<22s} {c:>4d}  ({100*c/n_done:.1f}%)")

        n_crash = buckets.get("probe_crash", 0)
        n_py = sum(c for b, c in buckets.items() if b.startswith("py_"))
        env_meaningful = n_done - n_crash - n_py
        print(f"\nclean rate (non-crash):     {(n_done - n_crash)/n_done*100:.1f}%")
        print(f"env-meaningful (no py err): {env_meaningful/n_done*100:.1f}%")

        # by step
        print(f"\n=== by step ===")
        by_step = {}
        for r in results:
            by_step.setdefault(r["step"], collections.Counter())[r["bucket"]] += 1
        for s in sorted(by_step):
            bs = by_step[s]
            tot = sum(bs.values())
            good = sum(c for b, c in bs.items() if not b.startswith("py_") and b not in ("probe_crash", "empty"))
            print(f"  step {s:>2}: n={tot:>3}  env-meaningful={good}/{tot} ({100*good/tot:.0f}%)")

        # save
        json.dump({
            "task_id": TASK_ID,
            "n_servers": len(ready_ports),
            "n_probes": n_done,
            "wall_seconds": round(wall, 1),
            "throughput_per_sec": round(n_done / wall, 2),
            "buckets": dict(buckets),
            "results": results,
        }, OUT_PATH.open("w"), indent=2, default=str)
        print(f"\nOutput → {OUT_PATH}")

    finally:
        teardown(procs)


if __name__ == "__main__":
    main()

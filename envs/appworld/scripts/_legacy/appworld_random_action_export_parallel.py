"""
Parallel AppWorld Random Action Export
=======================================

Splits tasks across N worker subprocesses, each with its own dedicated
AppWorld server port. After all workers finish, merges shard outputs.

Mirrors the WebShop random_action_export_parallel.py pattern adapted
for AppWorld's port-based architecture.

Usage:
  python -m agent_system.environments.appworld_random_action_export_parallel          # default 10 workers
  python -m agent_system.environments.appworld_random_action_export_parallel -w 16    # 16 workers
"""

import os
import sys
import json
import time
import math
import signal
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from appworld import load_task_ids

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PYTHON = sys.executable
SCRIPT_MODULE = "agent_system.environments.appworld_random_action_export"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_available_ports(port_file: str = "appworld_ports.ports") -> List[int]:
    """Load available port list from file."""
    if not os.path.exists(port_file):
        raise FileNotFoundError(
            f"Port file {port_file} does not exist. "
            "Please run the service startup script first."
        )
    ports = []
    with open(port_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                ports.append(int(line))
    if not ports:
        raise ValueError(f"No valid ports found in {port_file}.")
    return ports


def get_task_ranges(n_workers: int, total_tasks: int) -> List[Tuple[int, int]]:
    """Split task index space evenly across n_workers."""
    chunk = math.ceil(total_tasks / n_workers)
    ranges = []
    for i in range(n_workers):
        start = i * chunk
        end = min((i + 1) * chunk, total_tasks)
        if start < total_tasks:
            ranges.append((start, end))
    return ranges


def main():
    parser = argparse.ArgumentParser(description="Parallel AppWorld random action export")
    parser.add_argument("-w", "--workers", type=int, default=10,
                        help="Number of parallel worker processes")
    parser.add_argument("--n-random", type=int, default=5,
                        help="Random actions per step")
    parser.add_argument("--dataset", default="train",
                        help="AppWorld dataset split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port-file", default="appworld_ports.ports",
                        help="File containing available ports")
    parser.add_argument("-o", "--output-dir",
                        default=os.path.join(REPO_ROOT, "appworld_obs_from_random"))
    parser.add_argument("--log-dir",
                        default=os.path.join(REPO_ROOT, "logs"))
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-interactions", type=int, default=50)
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to cache parsed ground truth")
    args = parser.parse_args()

    n_workers = args.workers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Load ports ----
    ports = load_available_ports(args.port_file)
    if len(ports) < n_workers:
        raise ValueError(
            f"Need {n_workers} ports, only {len(ports)} available in {args.port_file}. "
            f"Either reduce --workers or start more AppWorld servers."
        )

    # ---- Load task IDs to determine total count ----
    task_ids = load_task_ids(args.dataset)
    total_tasks = len(task_ids)

    # ---- Split task ranges ----
    ranges = get_task_ranges(n_workers, total_tasks)
    # Adjust n_workers to actual number of ranges (in case total < n_workers)
    n_workers = len(ranges)

    shard_dir = os.path.join(args.output_dir, f"shards_{timestamp}")
    shard_log_dir = os.path.join(args.log_dir, f"shards_{timestamp}")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(shard_log_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"Parallel AppWorld Random Action Export")
    print(f"  Workers:    {n_workers}")
    print(f"  n_random:   {args.n_random}")
    print(f"  dataset:    {args.dataset}")
    print(f"  total tasks: {total_tasks}")
    print(f"  output:     {shard_dir}")
    print(f"  timestamp:  {timestamp}")
    print(f"  Task range assignment:")
    for i, (s, e) in enumerate(ranges):
        print(f"    worker {i:2d}: [{s}, {e})  port={ports[i]}")
    print(f"{'='*70}\n")

    t0 = time.time()

    # ---- Launch worker subprocesses ----
    procs: List[subprocess.Popen] = []
    log_files = []
    for i, (start, end) in enumerate(ranges):
        worker_out = os.path.join(shard_dir, f"shard_{i:02d}")
        worker_log = os.path.join(shard_log_dir, f"shard_{i:02d}")
        os.makedirs(worker_out, exist_ok=True)
        os.makedirs(worker_log, exist_ok=True)

        cmd = [
            PYTHON, "-m", SCRIPT_MODULE,
            str(start), str(end),
            "--port", str(ports[i]),
            "--dataset", args.dataset,
            "--n-random", str(args.n_random),
            "--seed", str(args.seed),
            "-o", worker_out,
            "--log-dir", worker_log,
            "--history-length", str(args.history_length),
            "--max-interactions", str(args.max_interactions),
        ]
        if args.cache_dir:
            cmd.extend(["--cache-dir", args.cache_dir])

        log_path = os.path.join(shard_log_dir, f"worker_{i:02d}.log")
        log_f = open(log_path, "w")
        log_files.append(log_f)

        p = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
        )
        procs.append(p)
        print(f"Launched worker {i:2d} (PID {p.pid}): [{start}, {end}) port={ports[i]}")

    print(f"\nAll {n_workers} workers launched, waiting for completion...\n")

    # ---- Wait with progress monitoring ----
    try:
        while True:
            all_done = True
            for p in procs:
                if p.poll() is None:
                    all_done = False
                    break
            if all_done:
                break

            # Count completed tasks from log files
            total_done_tasks = 0
            for i in range(n_workers):
                worker_log = os.path.join(shard_log_dir, f"shard_{i:02d}")
                for f_path in Path(worker_log).glob("tasks_*.txt"):
                    try:
                        total_done_tasks += sum(1 for _ in open(f_path))
                    except Exception:
                        pass

            elapsed = time.time() - t0
            running = sum(1 for p in procs if p.poll() is None)
            finished = n_workers - running
            rate = total_done_tasks / elapsed * 60 if elapsed > 0 else 0
            eta_min = (total_tasks - total_done_tasks) / rate if rate > 0 else float("inf")
            print(
                f"  [{elapsed/60:5.1f}min] "
                f"tasks: {total_done_tasks}/{total_tasks} | "
                f"workers: {running} running, {finished} done | "
                f"rate: {rate:.1f} tasks/min | "
                f"ETA: {eta_min:.0f} min"
            )
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nInterrupt received, terminating all workers...")
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs:
            p.wait()
        print("All workers terminated")
        return

    for f in log_files:
        f.close()

    # ---- Check exit codes ----
    failed = []
    for i, p in enumerate(procs):
        if p.returncode != 0:
            failed.append((i, p.returncode))
    if failed:
        print(f"\nWARNING: {len(failed)} worker(s) failed:")
        for i, rc in failed:
            log_path = os.path.join(shard_log_dir, f"worker_{i:02d}.log")
            print(f"  worker {i}: exit code {rc}  (log: {log_path})")

    # ---- Merge shard outputs ----
    print(f"\nMerging shard outputs...")
    merged = []
    for i in range(n_workers):
        worker_out = os.path.join(shard_dir, f"shard_{i:02d}")
        for f_path in sorted(Path(worker_out).glob("random_*.json")):
            if ".ckpt_" in f_path.name:
                continue
            try:
                shard_data = json.loads(f_path.read_text(encoding="utf-8"))
                merged.extend(shard_data)
                print(f"  shard {i:02d}: {len(shard_data):,} records from {f_path.name}")
            except Exception as e:
                print(f"  WARNING: shard {i:02d}: read failed — {e}")

    merged_path = os.path.join(
        args.output_dir,
        f"random_{args.n_random}act_ALL_{timestamp}.json",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    t1 = time.time()
    duration = t1 - t0

    # Count unique tasks and errors
    unique_tasks = len(set(r["task_id"] for r in merged)) if merged else 0
    error_count = sum(1 for r in merged if r.get("error"))

    print(f"\n{'='*70}")
    print(f"Done!")
    print(f"  Total records: {len(merged):,}")
    print(f"  Unique tasks:  {unique_tasks:,}")
    print(f"  Errors:        {error_count:,}")
    print(f"  Duration:      {duration:.0f}s ({duration/60:.1f}min)")
    print(f"  Merged output: {merged_path}")
    print(f"  Shard dir:     {shard_dir}")
    print(f"{'='*70}")

    # Save timing log
    time_log = os.path.join(
        args.log_dir,
        f"time_parallel_{timestamp}.log",
    )
    with open(time_log, "w") as f:
        f.write(f"workers: {n_workers}\n")
        f.write(f"n_random: {args.n_random}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"duration: {duration:.1f}s\n")
        f.write(f"total_records: {len(merged)}\n")
        f.write(f"total_tasks: {unique_tasks}\n")
        f.write(f"errors: {error_count}\n")
        f.write(f"output: {merged_path}\n")


if __name__ == "__main__":
    main()

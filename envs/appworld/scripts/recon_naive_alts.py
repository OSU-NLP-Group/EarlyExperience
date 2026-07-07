"""Recon: zero-LLM naive-alt characterization for AppWorld.

For 3 train tasks of varying complexity, walk the expert trajectory step
by step. At each expert state s_i:
  - Compute the candidate pool (endpoints across required_apps + supervisor + api_docs)
  - Sample K random alt endpoint names (excluding the expert's)
  - Capture scope variables at s_i (via a probe env)
  - For each alt, execute a naive no-args call in a fresh probe and classify the response

The goal: see what the WORST-CASE error distribution looks like — if we
just sampled endpoint names and called them with no args, how much of the
resulting IWM data would be "no, you need args" noise vs informative
state-grounded responses. This gives us the floor; the real LLM proposer
should do strictly better (and we'll measure by how much in a later smoke).

Output: stdout summary + per-step detail at envs/appworld/data/_recon/naive_alts_report.json
This script makes NO LLM calls. No pre-call gate needed.
"""
import os, sys, json, ast, random, uuid, re, collections, time
from pathlib import Path

os.environ.setdefault(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)

from appworld import AppWorld, load_task_ids
from appworld.ground_truth import GroundTruth

PORT = int(os.environ.get("APPWORLD_PORT", "7050"))
URL = f"http://0.0.0.0:{PORT}"
DOCS_DIR = Path(os.environ["APPWORLD_ROOT"]) / "data" / "api_docs" / "standard"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "_recon"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "naive_alts_report.json"

K_ALTS = 5
ALWAYS_AVAILABLE = ("supervisor", "api_docs")

APP_DOCS = {f.stem: json.load(f.open()) for f in DOCS_DIR.glob("*.json")}


def build_pool(required_apps):
    pool = {}
    for app in list(required_apps) + list(ALWAYS_AVAILABLE):
        if app not in APP_DOCS:
            continue
        for ep in APP_DOCS[app].keys():
            pool[f"{app}.{ep}"] = (app, ep)
    return pool


def parse_blocks(src):
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solution"),
        None,
    )
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


def extract_call(code):
    """Statically extract (app, endpoint, args_repr) of the first apis.X.Y(...) call."""
    try:
        tree = ast.parse(code, mode="exec")
    except Exception:
        return None, None, None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute):
            inner = f.value
            if isinstance(inner.value, ast.Name) and inner.value.id == "apis":
                args_kw = []
                for kw in node.keywords:
                    try:
                        args_kw.append(f"{kw.arg}={ast.unparse(kw.value)[:40]}")
                    except Exception:
                        pass
                return inner.attr, f.attr, ", ".join(args_kw)
    return None, None, None


def make_probe(task_id):
    return AppWorld(
        task_id=task_id,
        experiment_name=f"recon_{uuid.uuid4().hex[:8]}",
        remote_environment_url=URL,
    )


def replay_prefix(env, blocks):
    for b in blocks:
        env.execute(b)


def classify_response(text):
    if not text:
        return "empty"
    s = text.lower()
    if "execution successful" in s and len(s) < 50:
        return "noop_success"
    if re.search(r"\btraceback\b", s) or re.search(r"\berror:?\b", s) or "exception" in s:
        if "typeerror" in s or ("missing" in s and "argument" in s):
            return "py_typeerror"
        if "nameerror" in s:
            return "py_nameerror"
        if "syntaxerror" in s:
            return "py_syntaxerror"
        return "py_other_error"
    if re.search(r'"message"\s*:\s*"', s) or "validation error" in s:
        return "app_validation_error"
    if "401" in s or "unauthorized" in s or "not logged in" in s:
        return "app_unauthorized"
    if "404" in s or "not found" in s:
        return "app_not_found"
    if "403" in s or "forbidden" in s:
        return "app_forbidden"
    if s.strip().startswith(("[", "{")):
        return "data_json"
    return "data_text"


def main():
    t0 = time.time()
    train_ids = load_task_ids("train")

    # Pick 3 tasks of varying complexity
    target_tasks = ["82e2fac_1"]  # 1-app, 10 steps (verified)
    pick2, pick3 = None, None
    for tid in train_ids:
        gt = GroundTruth.load(tid, mode="full")
        n = len(gt.required_apps)
        if n == 2 and pick2 is None:
            pick2 = tid
        elif n == 3 and pick3 is None:
            pick3 = tid
        if pick2 and pick3:
            break
    if pick2:
        target_tasks.append(pick2)
    if pick3:
        target_tasks.append(pick3)

    print(f"Recon targets: {target_tasks}")
    random.seed(42)
    all_reports = []

    for task_id in target_tasks:
        print(f"\n{'='*78}\nTask {task_id}\n{'='*78}")
        gt = GroundTruth.load(task_id, mode="full")
        blocks = parse_blocks(gt.compiled_solution_code)
        pool = build_pool(gt.required_apps)
        pool_keys = list(pool.keys())

        # Read instruction from specs
        specs_path = Path(os.environ["APPWORLD_ROOT"]) / "data" / "tasks" / task_id / "specs.json"
        specs = json.load(specs_path.open())

        task_report = {
            "task_id": task_id,
            "required_apps": list(gt.required_apps),
            "instruction": specs.get("instruction"),
            "n_steps": len(blocks),
            "pool_size": len(pool),
            "pool_apps": sorted({a for a, _ in pool.values()}),
            "steps": [],
        }

        print(f"required_apps={gt.required_apps}, pool_size={len(pool)}, n_steps={len(blocks)}")
        print(f"instruction: {specs.get('instruction', '')[:200]}")

        for i, expert_code in enumerate(blocks):
            expert_app, expert_ep, expert_args = extract_call(expert_code)
            expert_key = f"{expert_app}.{expert_ep}" if expert_app else None

            choices = [k for k in pool_keys if k != expert_key]
            alt_keys = random.sample(choices, min(K_ALTS, len(choices)))

            # Scope inspection probe
            scope_probe = make_probe(task_id)
            try:
                replay_prefix(scope_probe, blocks[:i])
                scope_text = scope_probe.execute(
                    "import json as _j; "
                    "print(_j.dumps({k: type(v).__name__ for k, v in dict(locals()).items() "
                    "if not k.startswith('_')}))"
                )
            except Exception as e:
                scope_text = f"<scope inspect failed: {e}>"
            finally:
                try:
                    scope_probe.close()
                except Exception:
                    pass

            try:
                scope_dict = json.loads(scope_text.strip().split("\n")[-1])
            except Exception:
                scope_dict = {"_raw_first_200": scope_text[:200]}

            alt_results = []
            for alt_key in alt_keys:
                alt_app, alt_ep = pool[alt_key]
                naive_call = f"print(apis.{alt_app}.{alt_ep}())"
                probe = make_probe(task_id)
                try:
                    replay_prefix(probe, blocks[:i])
                    resp = probe.execute(naive_call)
                except Exception as e:
                    resp = f"<probe crashed: {e}>"
                finally:
                    try:
                        probe.close()
                    except Exception:
                        pass
                alt_results.append(
                    {
                        "alt": alt_key,
                        "naive_call": naive_call,
                        "bucket": classify_response(resp),
                        "resp_head": (resp or "")[:200],
                        "resp_len": len(resp) if resp else 0,
                    }
                )

            print(f"\nStep {i}: expert={expert_key}({expert_args[:80] if expert_args else ''})")
            scope_preview = {k: v for k, v in list(scope_dict.items())[:8]
                             if not k.startswith("_")}
            print(f"  scope ({len(scope_dict)} vars): {scope_preview}")
            for ar in alt_results:
                head = ar["resp_head"].replace("\n", " ")[:90]
                print(f"  alt {ar['alt']:<32s} → {ar['bucket']:<22s} | {head!r}")

            task_report["steps"].append(
                {
                    "step": i,
                    "expert_call": {"app": expert_app, "endpoint": expert_ep, "args_repr": expert_args},
                    "scope": scope_dict,
                    "alts": alt_results,
                }
            )
        all_reports.append(task_report)

    # Aggregate
    print(f"\n\n{'='*78}\nAggregate bucket distribution across all probes\n{'='*78}")
    bucket_counts = collections.Counter()
    bucket_by_step_phase = {"early": collections.Counter(), "mid": collections.Counter(), "late": collections.Counter()}
    for tr in all_reports:
        N = tr["n_steps"]
        for st in tr["steps"]:
            phase = "early" if st["step"] < N / 3 else ("mid" if st["step"] < 2 * N / 3 else "late")
            for ar in st["alts"]:
                bucket_counts[ar["bucket"]] += 1
                bucket_by_step_phase[phase][ar["bucket"]] += 1
    total = sum(bucket_counts.values())
    print(f"Total probes: {total}")
    for b, c in bucket_counts.most_common():
        print(f"  {b:<22s} {c:>4d}  ({100 * c / total:.1f}%)")

    print(f"\nBy step phase (early/mid/late thirds of trajectory):")
    for phase in ["early", "mid", "late"]:
        ph_total = sum(bucket_by_step_phase[phase].values())
        print(f"  --- {phase} (n={ph_total}) ---")
        for b, c in bucket_by_step_phase[phase].most_common():
            print(f"      {b:<22s} {c:>4d}  ({100 * c / ph_total:.1f}%)")

    with OUT_PATH.open("w") as f:
        json.dump(
            {
                "summary": {
                    "n_tasks": len(all_reports),
                    "K_alts": K_ALTS,
                    "bucket_distribution": dict(bucket_counts),
                    "bucket_by_phase": {k: dict(v) for k, v in bucket_by_step_phase.items()},
                    "wall_seconds": round(time.time() - t0, 1),
                },
                "tasks": all_reports,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWall: {time.time() - t0:.1f}s")
    print(f"Full report → {OUT_PATH}")


if __name__ == "__main__":
    main()

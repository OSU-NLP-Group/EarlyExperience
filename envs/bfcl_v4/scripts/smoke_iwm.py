"""
Phase 2 IWM smoke for BFCL v4.

Per expert state (function_call type only):
  1) sample K=10 alt function names from `involved_classes` method pool \ {expert_name}
  2) ONE DeepSeek call → fill plausible args for all 10 names (single batched call)
  3) for each parsed alt: deepcopy sim instances, eval the call, snapshot post-state
  4) write per-alt records to data/rollout/iwm_smoke.jsonl for hand-inspection

Pre-call gate (TEAM_GUIDE §2): use `--dry-run` to print first prompt without
hitting the API; remove flag only after explicit user go-ahead.

Run:
    # dry-run, no API call
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/smoke_iwm.py --dry-run --n-cases 3

    # live (will send ~N_cases × avg_fcall_steps DeepSeek calls)
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/smoke_iwm.py --n-cases 3
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    CLASS_FILE_PATH_MAPPING,
    STATELESS_CLASSES,
)

REPO = Path(__file__).resolve().parents[3]
PARSED = REPO / "envs/bfcl_v4/data/parsed/opus_expert_steps.jsonl"
DATA = REPO / "envs/bfcl_v4/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
FUNC_DOC = DATA / "multi_turn_func_doc"
SPLIT_DIR = REPO / "envs/bfcl_v4/data/split"
OUT_DIR = REPO / "envs/bfcl_v4/data/rollout"

# class-name (used in involved_classes) → file stem under multi_turn_func_doc/
CLASS_TO_FILE_STEM = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# safety blacklist mirrored from BFCL's execute_multi_turn_func_call
UNSAFE_BUILTINS = {"kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"}

DEEPSEEK_MODEL = "deepseek-v4-flash"        # smoke uses Flash for cost; Pro for full per TEAM_GUIDE §1.2
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
SEED_ALTS = 42                              # fixed seed for alt-name sampling
K_ALTS = 10                                  # alts per expert state (paper §B.3)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_tool_schemas(involved_classes: list[str]) -> list[dict]:
    """Return list of all method schemas across involved classes (JSONL files)."""
    schemas = []
    for cls in involved_classes:
        stem = CLASS_TO_FILE_STEM.get(cls)
        if not stem:
            continue
        with (FUNC_DOC / f"{stem}.json").open() as f:
            for line in f:
                line = line.strip()
                if line:
                    schemas.append(json.loads(line))
    return schemas


def load_expert_steps_grouped() -> dict[str, list[dict]]:
    """Read opus_expert_steps.jsonl, return {case_id: [records sorted by global_emit_idx]}."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    with PARSED.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            grouped[r["case_id"]].append(r)
    for cid in grouped:
        grouped[cid].sort(key=lambda r: r["global_emit_idx"])
    return grouped


def load_initial_configs(case_ids: set[str]) -> dict[str, dict]:
    """Return {case_id: initial_config}."""
    out = {}
    with (DATA / "BFCL_v4_multi_turn_base.json").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["id"] in case_ids:
                out[o["id"]] = o["initial_config"]
    return out


# ---------------------------------------------------------------------------
# Sim instance helpers
# ---------------------------------------------------------------------------


def instantiate_case(involved_classes: list[str], initial_config: dict) -> dict:
    instances = {}
    for cls_name in involved_classes:
        mod = importlib.import_module(CLASS_FILE_PATH_MAPPING[cls_name])
        Cls = getattr(mod, cls_name)
        inst = Cls()
        if cls_name not in STATELESS_CLASSES:
            cfg = initial_config.get(cls_name, {})
            inst._load_scenario(copy.deepcopy(cfg), long_context=False)
        instances[cls_name] = inst
    return instances


def _snapshot_gfs(inst) -> dict:
    """GFS-specific clean serializer. The default vars() dump produces
    a Directory.__repr__ blob that hides cwd and obscures depth — the LLM
    can read filenames but loses path context. This walks the tree into a
    JSON dict and surfaces the cwd path explicitly. Does NOT modify any
    env behavior — only what we show to the proposer LLM."""
    Directory = type(inst.root)

    def walk(d):
        out = {}
        for name, child in d.contents.items():
            if isinstance(child, Directory):
                out[name + "/"] = walk(child)
            else:
                content = getattr(child, "content", "") or ""
                preview = content[:200] + ("…" if len(content) > 200 else "")
                out[name] = {"_file": True, "size": len(content), "preview": preview}
        return out

    cwd = getattr(inst, "_current_dir", inst.root)
    parts = []
    p = cwd
    while p is not None:
        parts.append(p.name)
        p = getattr(p, "parent", None)
    cwd_path = "/" + "/".join(reversed(parts))

    return {
        "cwd": cwd_path,
        "long_context": getattr(inst, "long_context", False),
        "tree": {inst.root.name + "/": walk(inst.root)},
    }


def snapshot_instances(instances: dict) -> dict:
    out = {}
    for cls, inst in instances.items():
        if cls == "GorillaFileSystem":
            out[cls] = _snapshot_gfs(inst)
        else:
            out[cls] = {k: v for k, v in vars(inst).items() if not k.startswith("_")}
    return out


def _build_method_namespace(instances: dict) -> dict:
    """All instance methods exposed as bare names, for eval()."""
    ns = {}
    for inst in instances.values():
        for name in dir(inst):
            if name.startswith("_"):
                continue
            attr = getattr(inst, name)
            if callable(attr):
                ns[name] = attr
    return ns


def eval_call(call_str: str, instances: dict) -> tuple[str | None, str | None]:
    """Eval a Python-syntax call string against the instances' methods.
    Returns (result_text_or_None, error_text_or_None)."""
    # safety: extract function name and refuse blacklist
    m = re.match(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s*\(", call_str)
    if not m:
        return None, f"ParseError: cannot find function name in {call_str!r}"
    fn = m.group(1)
    if fn in UNSAFE_BUILTINS:
        return None, f"SafetyError: function {fn!r} is blacklisted"
    ns = _build_method_namespace(instances)
    if fn not in ns:
        return None, f"NameError: function {fn!r} not in involved classes' methods"
    try:
        result = eval(call_str, {"__builtins__": {}}, ns)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    if isinstance(result, dict):
        try:
            return json.dumps(result, default=str), None
        except (TypeError, ValueError):
            return repr(result), None
    if isinstance(result, str):
        return result, None
    return repr(result), None


# ---------------------------------------------------------------------------
# History rendering
# ---------------------------------------------------------------------------


def render_history(prior_steps: list[dict], current_step: dict, char_cap: int = 5000) -> str:
    lines = []
    for r in prior_steps:
        if r.get("user_msg_for_turn"):
            lines.append(f"\n[USER turn {r['turn_idx']}]: {r['user_msg_for_turn']}")
        if r["step_type"] == "function_call":
            for c, tr in zip(r["expert_emit_decoded"], r["tool_responses_recorded"] or []):
                tr_s = tr if isinstance(tr, str) else str(tr)
                tr_short = (tr_s[:200] + "…") if len(tr_s) > 200 else tr_s
                lines.append(f"  [ACT step {r['step_idx']}]: {c}")
                lines.append(f"  [OBS]: {tr_short}")
        elif r["step_type"] == "text_only":
            text = r["expert_emit_raw"]
            if isinstance(text, list) and text and isinstance(text[0], str):
                text = text[0]
            text = text if isinstance(text, str) else str(text)
            text = (text[:200] + "…") if len(text) > 200 else text
            lines.append(f"  [TEXT step {r['step_idx']}]: {text}")
    if current_step.get("user_msg_for_turn"):
        lines.append(f"\n[USER current turn {current_step['turn_idx']}]: {current_step['user_msg_for_turn']}")
    out = "\n".join(lines).strip()
    if len(out) > char_cap:
        out = "…[history truncated]…\n" + out[-char_cap:]
    return out


# ---------------------------------------------------------------------------
# Proposer prompt + DeepSeek call
# ---------------------------------------------------------------------------


PROPOSER_SYSTEM = (
    "You roleplay a function-calling agent in a multi-turn tool-use environment. "
    "We will give you the conversation history, the current simulated environment "
    "state, the tool schemas, and the action the agent actually took at this step. "
    "Your job is to propose ALTERNATIVE function call strings — one for each of K "
    "given function names. Each call must:\n"
    "  - use Python-syntax: \"name(arg1='val', arg2=42)\";\n"
    "  - use keyword arguments matching the schema's `parameters` field;\n"
    "  - fill in plausible argument values consistent with the visible history/state;\n"
    "  - NOT repeat the expert's exact action.\n"
    "These alternatives will be executed as counterfactuals to study env dynamics. "
    "Output STRICT JSON: an array of objects {\"name\": ..., \"call\": ...}, one per "
    "requested function name, in the same order. Output ONLY the JSON array, no prose."
)


def _extract_fn_name(call_str: str) -> str | None:
    m = re.match(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s*\(", call_str)
    return m.group(1) if m else None


def build_proposer_user_content(
    history_text: str,
    state_snapshot: dict,
    expert_emit_decoded: list[str],
    alt_names: list[str],
    tool_schemas: list[dict],
) -> str:
    target_set = set(alt_names) | {n for n in (_extract_fn_name(c) for c in expert_emit_decoded) if n}
    relevant = [s for s in tool_schemas if s["name"] in target_set]
    schema_block = json.dumps(relevant, indent=2)
    state_text = json.dumps(state_snapshot, default=str, indent=2)
    if len(state_text) > 3000:
        state_text = state_text[:3000] + "\n…[state truncated]…"
    return (
        f"## Tool schemas (relevant subset)\n```json\n{schema_block}\n```\n\n"
        f"## Conversation history so far\n{history_text}\n\n"
        f"## Current simulated environment state\n```json\n{state_text}\n```\n\n"
        f"## The action the agent took at this step (DO NOT REPEAT)\n"
        f"{json.dumps(expert_emit_decoded)}\n\n"
        f"## Task\nProduce a JSON array of {len(alt_names)} objects, one per name "
        f"in this order: {alt_names}\n"
        f"Each object: {{\"name\": <function name>, \"call\": <python-syntax call string>}}"
    )


def call_deepseek(client: OpenAI, system: str, user: str, temperature: float = 1.0) -> dict:
    """Make one DeepSeek API call, return {'content': str, 'usage': {...}}."""
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},  # TEAM_GUIDE §1.2 hard rule
    )
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def parse_proposer_response(raw_text: str, requested_names: list[str]) -> list[dict]:
    """Parse DeepSeek JSON output. Returns list of {name, call, validation} dicts
    in the same order as requested_names. Missing names get a stub with valid=False."""
    parsed_by_name: dict[str, dict] = {}
    try:
        # DeepSeek may wrap in {"calls": [...]} or just return array
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            # try common key names
            for k in ("calls", "alternatives", "results", "array", "items"):
                if k in obj and isinstance(obj[k], list):
                    obj = obj[k]
                    break
            else:
                # if dict has exactly one list-valued key, take it
                list_vals = [v for v in obj.values() if isinstance(v, list)]
                if len(list_vals) == 1:
                    obj = list_vals[0]
        if not isinstance(obj, list):
            return [{"name": n, "call": None, "valid": False,
                     "error": "response was not a JSON array"}
                    for n in requested_names]
        for entry in obj:
            if not isinstance(entry, dict):
                continue
            n = entry.get("name")
            c = entry.get("call")
            if not isinstance(n, str) or not isinstance(c, str):
                continue
            parsed_by_name[n] = {"name": n, "call": c, "valid": True, "error": None}
    except json.JSONDecodeError as e:
        return [{"name": n, "call": None, "valid": False,
                 "error": f"JSONDecodeError: {e}"}
                for n in requested_names]
    out = []
    for n in requested_names:
        if n in parsed_by_name:
            entry = parsed_by_name[n]
            # also validate that the call string's function name matches
            fn = _extract_fn_name(entry["call"] or "")
            if fn != n:
                entry["valid"] = False
                entry["error"] = f"call's function name {fn!r} != requested {n!r}"
            out.append(entry)
        else:
            out.append({"name": n, "call": None, "valid": False,
                        "error": "name missing from response"})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_case_worker(cid: str, info: dict, args, client: OpenAI) -> tuple:
    """Run IWM rollout for ONE case. Returns (cid, records, stats, validity, tokens).

    Per-case sequential (state advances via expert actions); cases run concurrently
    across workers in the main pool.
    """
    instances = instantiate_case(info["involved"], info["initial_config"])
    steps = info["steps"]
    records = []
    stats = Counter()
    validity = Counter()
    tokens = {"prompt": 0, "completion": 0}

    for i, step in enumerate(steps):
        prior = steps[:i]
        if step["step_type"] != "function_call":
            continue
        s_i_state = snapshot_instances(instances)
        expert_names = [_extract_fn_name(c) for c in step["expert_emit_decoded"]]
        pool = [n for n in info["candidate_names"] if n not in expert_names]
        rng_local = random.Random(args.seed + step["global_emit_idx"])
        alt_names = rng_local.sample(pool, min(args.k_alts, len(pool)))

        user_content = build_proposer_user_content(
            render_history(prior, step), s_i_state,
            step["expert_emit_decoded"], alt_names, info["schemas"],
        )
        try:
            resp = call_deepseek(client, PROPOSER_SYSTEM, user_content)
        except Exception as e:  # noqa: BLE001
            stats["llm_error"] += 1
            continue
        tokens["prompt"] += resp["usage"]["prompt_tokens"]
        tokens["completion"] += resp["usage"]["completion_tokens"]
        parsed = parse_proposer_response(resp["content"], alt_names)
        for p in parsed:
            validity["valid" if p["valid"] else "invalid"] += 1

        alts_emitted = []
        for p in parsed:
            if not p["valid"] or not p["call"]:
                alts_emitted.append({**p, "exec_result": None,
                                      "exec_error": p["error"],
                                      "post_state": None})
                continue
            clone = copy.deepcopy(instances)
            result, err = eval_call(p["call"], clone)
            alts_emitted.append({**p,
                                  "exec_result": result,
                                  "exec_error": err,
                                  "post_state": snapshot_instances(clone)})
            if err is None:
                stats["alt_exec_success"] += 1
            else:
                stats["alt_exec_error"] += 1

        records.append({
            "case_id": cid,
            "turn_idx": step["turn_idx"],
            "step_idx": step["step_idx"],
            "global_emit_idx": step["global_emit_idx"],
            "involved_classes": info["involved"],
            "user_msg_for_turn": step.get("user_msg_for_turn"),
            "s_i_state": s_i_state,
            "expert_emit_decoded": step["expert_emit_decoded"],
            "expert_tool_responses_recorded": step["tool_responses_recorded"],
            "alt_names": alt_names,
            "proposer_raw": resp["content"],
            "proposer_usage": resp["usage"],
            "alts": alts_emitted,
        })
        stats["records_written"] += 1

        # advance via expert action(s) on real instances
        for c in step["expert_emit_decoded"]:
            eval_call(c, instances)

    return cid, records, stats, validity, tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cases", type=int, default=3,
                        help="number of expert cases to smoke-rollout on")
    parser.add_argument("--k-alts", type=int, default=K_ALTS,
                        help="number of alternative actions per expert fcall state")
    parser.add_argument("--seed", type=int, default=SEED_ALTS,
                        help="random seed for alt-name sampling")
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent case workers (default 1 = sync)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print first prompt only; do NOT send any API call")
    parser.add_argument("--out", type=Path,
                        default=OUT_DIR / "iwm_smoke.jsonl",
                        help="output path for per-alt records")
    args = parser.parse_args()

    # ----- select smoke cases (deterministic) -----
    expert_ids = json.loads((SPLIT_DIR / "expert_ids.json").read_text())
    rng = random.Random(args.seed)
    smoke_cases = sorted(rng.sample(expert_ids, args.n_cases),
                         key=lambda s: int(s.rsplit("_", 1)[-1]))

    # ----- load data -----
    grouped = load_expert_steps_grouped()
    configs = load_initial_configs(set(smoke_cases))

    # ----- precompute per-case info -----
    case_pool: dict[str, dict] = {}
    for cid in smoke_cases:
        steps = grouped.get(cid, [])
        if not steps:
            print(f"warn: no parsed steps for {cid}")
            continue
        involved = steps[0]["involved_classes"]
        schemas = load_tool_schemas(involved)
        all_names = [s["name"] for s in schemas]
        case_pool[cid] = {
            "steps": steps,
            "involved": involved,
            "schemas": schemas,
            "candidate_names": all_names,
            "initial_config": configs[cid],
        }

    # ----- pre-call gate disclosure -----
    n_fcall_states = sum(
        sum(1 for s in info["steps"] if s["step_type"] == "function_call")
        for info in case_pool.values()
    )
    n_llm_calls = n_fcall_states
    print(f"\n{'='*60}\n"
          f"PRE-CALL GATE (TEAM_GUIDE §2)\n"
          f"{'='*60}\n"
          f"  task: BFCL IWM smoke — alt-action arg-fill, single batched call per state\n"
          f"  smoke cases ({args.n_cases}): {smoke_cases}\n"
          f"  fcall states across smoke set: {n_fcall_states}\n"
          f"  K_alts per state: {args.k_alts}\n"
          f"  → DeepSeek calls to send: {n_llm_calls}\n"
          f"  → alt actions to execute (on deepcopies): {n_llm_calls * args.k_alts}\n"
          f"  model: {DEEPSEEK_MODEL}, thinking disabled (TEAM_GUIDE §1.2)\n"
          f"  dry-run: {args.dry_run}\n"
          f"{'='*60}\n")

    if args.dry_run:
        # build and print ONE example proposer prompt without API call
        for cid, info in case_pool.items():
            steps = info["steps"]
            for i, step in enumerate(steps):
                if step["step_type"] != "function_call":
                    continue
                prior = steps[:i]
                state_snap = (
                    snapshot_instances(instantiate_case(info["involved"], info["initial_config"]))
                    if i == 0 else
                    "(state would be reconstructed by replaying expert actions up to s_i)"
                )
                expert_names = [_extract_fn_name(c) for c in step["expert_emit_decoded"]]
                pool = [n for n in info["candidate_names"] if n not in expert_names]
                rng_local = random.Random(args.seed)
                alt_names = rng_local.sample(pool, min(args.k_alts, len(pool)))
                user_content = build_proposer_user_content(
                    render_history(prior, step),
                    state_snap if isinstance(state_snap, dict) else {"_note": state_snap},
                    step["expert_emit_decoded"],
                    alt_names,
                    info["schemas"],
                )
                print(f"--- DRY-RUN sample prompt: case={cid} turn={step['turn_idx']} step={step['step_idx']} ---")
                print(f"[SYSTEM] ({len(PROPOSER_SYSTEM)} chars): {PROPOSER_SYSTEM[:200]}…")
                print(f"[USER]   ({len(user_content)} chars):")
                print(user_content[:2500] + ("\n…[truncated]…" if len(user_content) > 2500 else ""))
                return 0  # only print one
        return 0

    # ----- LIVE: rollout -----
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set"); return 2
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_stats = Counter()
    arg_validity = Counter()
    total_tokens = {"prompt": 0, "completion": 0}
    t0 = time.time()
    all_records: dict[str, list] = {}

    workers = max(1, args.workers)
    if workers == 1:
        for cid, info in case_pool.items():
            cid_r, recs, stats, validity, tokens = run_case_worker(cid, info, args, client)
            all_records[cid_r] = recs
            overall_stats.update(stats); arg_validity.update(validity)
            total_tokens["prompt"] += tokens["prompt"]; total_tokens["completion"] += tokens["completion"]
            print(f"  case {cid_r}: {stats['records_written']} records, "
                  f"{stats['alt_exec_success']+stats['alt_exec_error']} alt execs")
    else:
        print(f"  using {workers} concurrent case workers (one OpenAI client shared)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_case_worker, cid, info, args, client): cid
                       for cid, info in case_pool.items()}
            done = 0
            for fut in as_completed(futures):
                try:
                    cid_r, recs, stats, validity, tokens = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  case {futures[fut]}: WORKER FAILED ({type(e).__name__}: {e})")
                    continue
                all_records[cid_r] = recs
                overall_stats.update(stats); arg_validity.update(validity)
                total_tokens["prompt"] += tokens["prompt"]; total_tokens["completion"] += tokens["completion"]
                done += 1
                if done % 10 == 0 or done == len(case_pool):
                    print(f"  ... {done}/{len(case_pool)} cases done")

    # write all records, sorted by case_id for diffability
    with args.out.open("w") as fout:
        for cid in sorted(all_records.keys(), key=lambda s: int(s.rsplit("_", 1)[-1])):
            for rec in all_records[cid]:
                fout.write(json.dumps(rec, default=str) + "\n")

    elapsed = time.time() - t0
    print(f"\n{'='*60}\nSMOKE COMPLETE\n{'='*60}")
    print(f"  output: {args.out.relative_to(REPO)}")
    print(f"  records written: {overall_stats['records_written']}")
    print(f"  alt exec success: {overall_stats['alt_exec_success']}")
    print(f"  alt exec error  : {overall_stats['alt_exec_error']}")
    valid_tot = arg_validity['valid'] + arg_validity['invalid']
    if valid_tot:
        print(f"  args validity   : {dict(arg_validity)}  "
              f"(valid rate: {arg_validity['valid']/valid_tot:.1%})")
    print(f"  tokens: prompt={total_tokens['prompt']:,}  completion={total_tokens['completion']:,}  "
          f"total={sum(total_tokens.values()):,}")
    print(f"  wall: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""SR (Self-Reflection) smoke run for ScienceWorld.

Goals
-----
- Validate the SR pipeline end-to-end before committing to the full run.
- Measure how often the LLM proposer's actions survive the
  canonicalize / dedup / drop=expert / drop-not-in-admissible filter,
  i.e. whether oversample=K+2=5 is enough or we need to dial it up.
- Eyeball a few reflection CoTs to confirm the output is usable as
  training signal.

What this script does, per the 10 sampled expert states:
  Stage 1 (LLM proposer): ask DeepSeek V4 Pro (thinking disabled) for
                          5 alternative actions at temperature 1.0.
  Filter             :    canonicalize, dedup, drop=expert,
                          drop-not-in-admissible. Take up to K=3.
                          If <3 remain, accept variable K (no random refill).
  Stage 2 (env)      :    for each surviving alternative, env.load +
                          replay expert actions 0..i-1, then env.step(alt)
                          to capture s_i^j.
  Stage 3 (LLM refl) :    DeepSeek with the METHOD.md §4.3 reflection
                          prompt template, filled with
                          (situation, expert action, expected outcome,
                           alternative actions and resulting states).

The §2 pre-call gate must be approved BEFORE this script is run — every
LLM batch goes through it. The script reads DEEPSEEK_API_KEY from env.

Sampling is deterministic (seed=42): the same 10 (item_id, step) pairs
on every rerun.

Output
------
- envs/scienceworld/data/rollout/sr_smoke.jsonl
    one line per sampled state with all stages' raw outputs:
      {
        "item_id", "step",
        "expert_action", "expert_next_state",
        "proposer_raw_response",
        "proposer_parsed":[...5 candidates],
        "filter_trace":[ {action, status: kept/dedup/=expert/not-admissible} ],
        "filtered_alternatives":[
            {"action", "next_state"}, ...
        ],
        "reflection_prompt", "reflection_cot",
        "k_final"
      }
- Stdout: aggregate stats + 2 example reflections.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

DEFAULT_REPLAY = "envs/scienceworld/data/replay/replay_full.jsonl"
DEFAULT_ROLLOUT = "envs/scienceworld/data/rollout/iwm_rollout.jsonl"
DEFAULT_OUTPUT = "envs/scienceworld/data/rollout/sr_smoke.jsonl"
DEFAULT_N_SAMPLES = 50
OVERSAMPLE_N = 7
TARGET_K = 3

DEFAULT_LLM_CONCURRENCY = 30   # concurrent DeepSeek requests
DEFAULT_ENV_WORKERS = 16       # process pool for env probing (each worker = one JVM)

DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

PROPOSER_SYSTEM_PROMPT = """\
You are playing as the agent in a ScienceWorld text-based simulation. Your job in this turn is NOT to choose the best action, but to enumerate plausible alternative actions that the agent COULD take at the current state — alternatives other than the expert's chosen action. These will be used to train a separate model on reasoning about why the expert action is preferable.

Available action templates (use objects that exist in the current state):
{action_menu}

Output format:
- Exactly {n} lines.
- Each line is one action string, formatted exactly as the agent would emit it (lowercase, no quotes, no period, no numbering, e.g. "open door to kitchen", "go to hallway", "pick up thermometer", "wait1").
- The {n} actions must be distinct from each other.
- None of the {n} actions may be identical to the expert's chosen action.
- No commentary, no explanation, just {n} action lines.
"""

PROPOSER_USER_TEMPLATE = """\
Task:
{task_description}

Conversation history so far (the agent's prior reasoning, actions, and the environment's responses):
{history_block}

Current state (the environment's most recent observation):
{current_state}

Expert's chosen action at this point: {expert_action}

Propose {n} distinct alternative actions the agent could take instead.
"""

# Reflection prompt — paper §B.6 / METHOD.md §4.3 in spirit, but:
# (a) the word "expert"/"selected" is removed from input labels (otherwise
#     DeepSeek echoes those labels into the CoT — see method_recap.md
#     "The reflection CoT must read as the model's own internal thinking").
# (b) the action being justified is presented as a fact ("The action the
#     agent takes: X"), so the model is anchored to converge on X. Without
#     this anchor ~20% of CoTs drift to a different action — particularly
#     for wait/look-around steps whose goal-relevance is non-local.
# (c) hard length cap (~500 words / ~700 tokens) per workspace policy
#     (SKILL.md SR length section).
REFLECTION_PROMPT_TEMPLATE = """\
You will be presented with a situation in which a text-based agent is acting. The agent ends up taking one specific action at this step; your task is to write the agent's internal monologue that would naturally arrive at that decision.

- Situation (s_i):
{situation}
- The action the agent takes here: {expert_action}
  - Resulting next state: {expert_next_state}
- Other actions the agent considered (but did not take):
{alternatives_block}

Your monologue should:
1. Analyze the situation and the goal.
2. Briefly consider each of the other actions in turn, working out why the agent does not take it.
3. Arrive at the action above being the right next step, grounded in its resulting state.
4. Highlight any relevant clues, constraints, or consequences from the situation.

Guidelines:
- Stay strictly within the provided information.
- Avoid meta-commentary about being an AI.
- Use natural, step-by-step reasoning.
- Focus on logical decision-making.

CRITICAL — this monologue becomes TRAINING DATA. At inference, the agent has NO external label distinguishing "the action being taken" from "other actions considered" — it must reason its way to that action from scratch. Therefore:
- Do NOT use the words "expert", "selected", "chosen", "correct choice", "right action", "best option", "optimal action", "best alternative", or any phrasing that announces a privileged label before reasoning about it.
- Do NOT refer to the other options by their numbered labels ("Action 1", "Action 2", "Alternative 1", "a_i^1", etc.). At inference the agent does not see a pre-enumerated list — it considers options in its own head. Phrase them inline as natural thoughts: "I could try X", "Another option I have is to Y", "I also notice that I could Z", "I'm not sure if doing W would help".
- Write as if YOU are the agent freshly deliberating in the moment. The voice is genuine deliberation, with hedging where appropriate ("I'm not sure...", "It seems...", "I think...", "It's possible that..."), NOT a confident essay justifying a pre-known answer.
- Write as ONE continuous paragraph. No paragraph breaks, no bullet points, no numbered lists, no section headings.
- Length: aim for 200–400 words. Hard cap ~500 words; do not exceed this even if the situation seems complex.

ALSO CRITICAL — the monologue must CONVERGE on the action the agent takes (the one labeled above). Even when that action seems passive at first glance (e.g. waiting, looking around) or seems suboptimal in isolation (e.g. one step of a longer circuit-building plan), find the reasoning that justifies it given the task's overall goal. Do NOT let the monologue end by recommending a different action than the one the agent takes.

Output: Directly write the monologue, no extra headings, disclaimers, or external notes.
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def extract_action_menu_from_system_prompt(system_prompt: str) -> str:
    """Pull the action-menu chunk out of the AgentGym REACT system prompt."""
    # The system prompt looks like "...take: [{...},{...},...]\nYour response..."
    m = re.search(r"take:\s*(\[.+?\])\s*\n", system_prompt, flags=re.DOTALL)
    return m.group(1) if m else ""


def render_history_block(thoughts: list[str], actions: list[str], observations: list[str]) -> str:
    """Render the trajectory history up to step i as a chat-flow text block."""
    if not actions:
        return "(none — this is the very first step.)"
    pieces = []
    for i in range(len(actions)):
        if i < len(thoughts) and thoughts[i].strip():
            pieces.append(f"Thought:\n{thoughts[i].strip()}")
        pieces.append(f"Action:\n{actions[i].strip()}")
        if i < len(observations):
            pieces.append(f"Observation:\n{observations[i].strip()}")
        pieces.append("")
    return "\n".join(pieces).rstrip()


def canonicalize_action(s: str) -> str:
    """Minimal canonicalization: strip outer whitespace and outer quotes,
    strip trailing periods. Do NOT lowercase — admissible action strings
    are already lowercase, and lowercasing the proposed string when it is
    already lowercase is a no-op."""
    s = s.strip()
    # strip surrounding quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    # strip trailing period
    s = s.rstrip(".")
    return s.strip()


def parse_proposer_response(text: str, n: int) -> list[str]:
    """Pull out the N action lines from the proposer's raw response."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip leading enumeration like "1.", "1)", "- "
        line = re.sub(r"^[\-\d]+[.)]\s*", "", line).strip()
        line = canonicalize_action(line)
        if line:
            lines.append(line)
    # We expected exactly n lines but accept more (truncate) or fewer
    return lines[: max(n, len(lines))]


def filter_candidates(
    candidates: list[str], expert_action: str, admissible: list[str], target_k: int
) -> tuple[list[str], list[dict]]:
    """Hybrid SR filter:
        1. canonicalize + dedup + drop=expert
        2. split into valid (in admissible) / invalid (not in admissible),
           preserving LLM's original ordering inside each bucket
        3. take up to K from valid, then top up with invalid fillers if short
    Rationale: prioritize valid-vs-valid comparison (the rich SR signal:
    'why is the expert valid action better than these other valid actions')
    while still ensuring K=target_k per state. Invalid fillers (whose env
    response is 'No known action matches that input.') serve as a smaller
    'recovery from invalid-action attempts' signal — useful because the
    policy will produce invalid actions at inference, and SR teaches it to
    recognize them as inferior to the expert's choice.
    Returns (kept, trace) where each trace entry has status in
    {duplicate, equals_expert, kept_valid, kept_invalid_filler,
     extra_valid, extra_invalid}."""
    expert_canon = canonicalize_action(expert_action)
    admissible_set = {canonicalize_action(a) for a in admissible}

    # Pass 1: canonicalize + dedup + drop=expert; partition into valid/invalid.
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    trace: list[dict] = []
    for c in candidates:
        rec = {"action": c}
        if not c:
            rec["status"] = "empty"
        elif c in seen:
            rec["status"] = "duplicate"
        elif c == expert_canon:
            rec["status"] = "equals_expert"
        else:
            seen.add(c)
            if c in admissible_set:
                rec["status"] = "valid"
                valid.append(c)
            else:
                rec["status"] = "invalid"
                invalid.append(c)
        trace.append(rec)

    # Pass 2: take valid first up to K, then top up with invalid fillers.
    kept_valid = valid[:target_k]
    kept_invalid = invalid[: max(0, target_k - len(kept_valid))]
    kept = kept_valid + kept_invalid
    extra_valid = valid[len(kept_valid):]
    extra_invalid = invalid[len(kept_invalid):]

    # Annotate trace with the per-bucket fate.
    kept_set = set(kept)
    for rec in trace:
        if rec["status"] == "valid":
            rec["status"] = "kept_valid" if rec["action"] in kept_set else "extra_valid"
        elif rec["status"] == "invalid":
            rec["status"] = "kept_invalid_filler" if rec["action"] in kept_set else "extra_invalid"

    return kept, trace


# -----------------------------------------------------------------------------
# Worker globals — each worker holds its own scienceworld JVM
# -----------------------------------------------------------------------------

_WORKER_ENV = None


def _worker_init():
    global _WORKER_ENV
    from scienceworld import ScienceWorldEnv

    _WORKER_ENV = ScienceWorldEnv()


def probe_alt_next_state(task_value: str, var: int, expert_actions: list[str],
                         step_i: int, alt_action: str) -> str:
    """Replay env from scratch up to s_i then step(alt) to capture s_i^j."""
    env = _WORKER_ENV
    env.load(task_value, var)
    for k in range(step_i):
        env.step(expert_actions[k])
    ob, _, _, _ = env.step(alt_action)
    return ob


def probe_alts_for_state(args_tuple) -> list[dict]:
    """ProcessPool worker: env-probe K alternative actions for one state.
    args_tuple = (task_value, variation_idx, expert_actions, step_i, alt_actions).
    Returns [{"action": alt, "next_state": ob}, ...]."""
    task_value, var, expert_actions, step_i, alt_actions = args_tuple
    env = _WORKER_ENV
    results = []
    for alt in alt_actions:
        env.load(task_value, var)
        for k in range(step_i):
            env.step(expert_actions[k])
        ob, _, _, _ = env.step(alt)
        results.append({"action": alt, "next_state": ob})
    return results


def query_admissible_at_clean_init(task_value: str, var: int) -> list[str]:
    """Worker variant of getValidActionObjectCombinations() right after load.
    Used to fill in admissible for step_i=0 states (where replay_full doesn't
    pre-record info.valid since it's pre-first-step)."""
    env = _WORKER_ENV
    env.load(task_value, var)
    return env.getValidActionObjectCombinations()


# -----------------------------------------------------------------------------
# DeepSeek client
# -----------------------------------------------------------------------------


def make_deepseek_client():
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def call_deepseek(client, system: str, user: str, temperature: float,
                  max_tokens: int = 2048,
                  frequency_penalty: float = 0.0) -> tuple[str, dict]:
    """Single DeepSeek call. Returns (content, usage_dict).
    frequency_penalty: pass 0.3-0.5 for long-form generation (reflection)
    to suppress DeepSeek's occasional duplicate-paragraph degradation.
    Only effective with thinking disabled (which we always do here)."""
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        extra_body={"thinking": {"type": "disabled"}},  # mandatory per TEAM_GUIDE §1.2
    )
    usage = resp.usage
    usage_dict = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    return resp.choices[0].message.content, usage_dict


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", default=DEFAULT_REPLAY)
    ap.add_argument("--rollout", default=DEFAULT_ROLLOUT)
    ap.add_argument("--raw-agenttraj", default="envs/scienceworld/data/raw/sciworld_train.json")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    ap.add_argument("--oversample-n", type=int, default=OVERSAMPLE_N)
    ap.add_argument("--target-k", type=int, default=TARGET_K)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and run env probing but DO NOT call the LLM.",
    )
    ap.add_argument("--llm-concurrency", type=int, default=DEFAULT_LLM_CONCURRENCY)
    ap.add_argument("--env-workers", type=int, default=DEFAULT_ENV_WORKERS)
    args = ap.parse_args()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load AgentGym REACT system prompt + extract action menu.
    with open(args.raw_agenttraj) as f:
        raw0 = json.load(f)[0]
    system_prompt_agentgym = raw0["conversations"][0]["value"]
    action_menu = extract_action_menu_from_system_prompt(system_prompt_agentgym)

    # Load replay_full into memory (we only need the filtered subset).
    print(f"loading replay from {args.replay}...")
    surviving = []
    for line in open(args.replay):
        r = json.loads(line)
        if r.get("final_done") and r.get("final_score") == 100:
            surviving.append(r)
    print(f"  surviving trajectories: {len(surviving)}")

    # Load initial_obs from iwm_rollout (we captured these via env.look()).
    print(f"loading initial_obs from {args.rollout}...")
    initial_obs_by_iid: dict[str, str] = {}
    for line in open(args.rollout):
        r = json.loads(line)
        if r.get("kind") == "initial":
            initial_obs_by_iid[r["item_id"]] = r["initial_obs"]
    print(f"  initial_obs records: {len(initial_obs_by_iid)}")

    # Sample N (trajectory, step) pairs deterministically.
    rng = random.Random(args.seed)
    pool = []
    for rec in surviving:
        n_steps = len(rec["replay_steps"])
        for step_i in range(n_steps):
            pool.append((rec["item_id"], step_i))
    samples_idx = rng.sample(range(len(pool)), args.n_samples)
    samples = [pool[i] for i in samples_idx]
    iid_to_rec = {r["item_id"]: r for r in surviving}
    print(f"  sampled {len(samples)} (item_id, step) pairs")

    # Pre-compute everything that doesn't need LLM — so we can §2-gate the LLM calls separately.
    prepared = []
    for iid, step_i in samples:
        rec = iid_to_rec[iid]
        thoughts = rec["agenttraj_thoughts"]
        actions = rec["agenttraj_actions"]
        replay_steps = rec["replay_steps"]
        observations = [s["observation"] for s in replay_steps]
        task_desc = replay_steps[0]["info"].get("taskDesc", "")
        initial_obs = initial_obs_by_iid.get(iid, "")

        # state_at_s_i = initial_obs if step_i == 0 else observations[step_i - 1]
        if step_i == 0:
            current_state = initial_obs
        else:
            current_state = observations[step_i - 1]

        # admissible at s_i = info.valid AFTER step_i-1 (for i >= 1), else needs querying at clean s_0
        # We need it for the filter. For i=0 we'll need to query env; but in the rollout we already
        # captured admissible-at-s_0 by querying env at clean s_0 inside rollout_iwm.py.
        # We don't have that data here directly — so we'll re-query env at runtime for i=0.
        # For i >= 1: use replay_steps[i-1]["info"]["valid"].
        if step_i == 0:
            admissible = None  # to be filled by env query at runtime
        else:
            admissible = replay_steps[step_i - 1]["info"].get("valid", [])

        expert_action = actions[step_i]
        expert_next_state = observations[step_i]

        proposer_user = PROPOSER_USER_TEMPLATE.format(
            task_description=task_desc,
            history_block=render_history_block(
                thoughts[:step_i], actions[:step_i], observations[:step_i]
            ),
            current_state=current_state,
            expert_action=expert_action,
            n=args.oversample_n,
        )
        proposer_system = PROPOSER_SYSTEM_PROMPT.format(
            action_menu=action_menu, n=args.oversample_n
        )

        situation_block = (
            f"Task:\n{task_desc}\n\n"
            f"Prior trajectory:\n{render_history_block(thoughts[:step_i], actions[:step_i], observations[:step_i])}\n\n"
            f"Current observation:\n{current_state}"
        )

        prepared.append(
            {
                "item_id": iid,
                "step": step_i,
                "task_key": rec["task_key"],
                "task_value": rec["task_value"],
                "variation_idx": rec["variation_idx"],
                "expert_actions": actions,
                "task_desc": task_desc,
                "current_state": current_state,
                "expert_action": expert_action,
                "expert_next_state": expert_next_state,
                "admissible": admissible,
                "proposer_system": proposer_system,
                "proposer_user": proposer_user,
                "situation_block": situation_block,
            }
        )

    # Estimate token volume — input only at this stage (no LLM call yet).
    approx_in = sum(len(p["proposer_system"]) + len(p["proposer_user"]) for p in prepared)
    print(f"\napprox proposer input chars: {approx_in:,} (~{approx_in//4:,} tokens)")
    if args.dry_run:
        print("\n--dry-run: skipping LLM calls. Writing prepared prompts to a debug file...")
        debug_path = out_path.with_suffix(".dryrun.jsonl")
        with open(debug_path, "w") as f:
            for p in prepared:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  -> {debug_path}")
        return

    # === LLM phase: §2 gate must have been approved before reaching this. ===
    print("\nInstantiating DeepSeek client...")
    client = make_deepseek_client()
    proposer_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    reflection_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    filter_status_total: dict[str, int] = {}
    k_final_dist: dict[int, int] = {}
    hybrid_split_dist: dict[tuple[int, int], int] = {}  # (n_valid_kept, n_invalid_kept) -> count
    pipeline_t_start = time.time()

    # -----------------------------------------------------------------------
    # Stage 0: fill in admissible for step_i=0 states via a small env-pool batch
    # -----------------------------------------------------------------------
    step0_indices = [i for i, p in enumerate(prepared) if p["admissible"] is None]
    if step0_indices:
        print(f"\n=== Stage 0: querying admissible at clean init for {len(step0_indices)} step_i=0 states ===")
        t0 = time.time()
        with ProcessPoolExecutor(
            max_workers=min(args.env_workers, len(step0_indices)),
            initializer=_worker_init,
        ) as pool:
            payloads = [(prepared[i]["task_value"], prepared[i]["variation_idx"]) for i in step0_indices]
            futures = {pool.submit(query_admissible_at_clean_init, *pl): i for pl, i in zip(payloads, step0_indices)}
            for fut in as_completed(futures):
                i = futures[fut]
                prepared[i]["admissible"] = fut.result()
        print(f"  done in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Stage 1: proposer LLM, concurrent via ThreadPool
    # -----------------------------------------------------------------------
    print(f"\n=== Stage 1: {len(prepared)} proposer calls (concurrency={args.llm_concurrency}) ===")
    t0 = time.time()
    proposer_outputs: list = [None] * len(prepared)
    proposer_errors: dict[int, str] = {}

    def _call_proposer(i: int):
        p = prepared[i]
        return call_deepseek(
            client, p["proposer_system"], p["proposer_user"],
            temperature=1.0, max_tokens=512,
        )

    with ThreadPoolExecutor(max_workers=args.llm_concurrency) as pool:
        futures = {pool.submit(_call_proposer, i): i for i in range(len(prepared))}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                text, usage = fut.result()
                proposer_outputs[i] = text
                proposer_usage["prompt_tokens"] += usage["prompt_tokens"]
                proposer_usage["completion_tokens"] += usage["completion_tokens"]
            except Exception as exc:  # noqa: BLE001
                proposer_errors[i] = repr(exc)
    print(f"  done in {time.time() - t0:.1f}s   (errors: {len(proposer_errors)})")

    # -----------------------------------------------------------------------
    # Stage 1.5: filter (CPU, sequential — fast)
    # -----------------------------------------------------------------------
    filter_state: list[dict] = []  # per-state {kept, trace, candidates}
    for i, p in enumerate(prepared):
        if i in proposer_errors:
            filter_state.append({"kept": [], "trace": [], "candidates": [], "error": proposer_errors[i]})
            continue
        candidates = parse_proposer_response(proposer_outputs[i], args.oversample_n)
        kept, trace = filter_candidates(candidates, p["expert_action"], p["admissible"], args.target_k)
        filter_state.append({"kept": kept, "trace": trace, "candidates": candidates})
        for t in trace:
            filter_status_total[t["status"]] = filter_status_total.get(t["status"], 0) + 1
        n_valid_kept = sum(1 for t in trace if t["status"] == "kept_valid")
        n_invalid_kept = sum(1 for t in trace if t["status"] == "kept_invalid_filler")
        k_final_dist[len(kept)] = k_final_dist.get(len(kept), 0) + 1
        hybrid_split_dist[(n_valid_kept, n_invalid_kept)] = (
            hybrid_split_dist.get((n_valid_kept, n_invalid_kept), 0) + 1
        )

    # -----------------------------------------------------------------------
    # Stage 2: env-probe alternatives, concurrent via ProcessPool
    # -----------------------------------------------------------------------
    print(f"\n=== Stage 2: env probing alt next-states (workers={args.env_workers}) ===")
    t0 = time.time()
    env_results: list = [None] * len(prepared)
    probe_jobs = [
        (i, (prepared[i]["task_value"], prepared[i]["variation_idx"],
             prepared[i]["expert_actions"], prepared[i]["step"], filter_state[i]["kept"]))
        for i in range(len(prepared))
        if filter_state[i]["kept"]
    ]
    if probe_jobs:
        with ProcessPoolExecutor(max_workers=args.env_workers, initializer=_worker_init) as pool:
            futures = {pool.submit(probe_alts_for_state, payload): i for i, payload in probe_jobs}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    env_results[i] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    env_results[i] = {"error": repr(exc)}
    # states with no kept alts -> empty env_results stays None
    print(f"  done in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Stage 3: reflection LLM, concurrent via ThreadPool — STREAMS to disk
    # -----------------------------------------------------------------------
    print(f"\n=== Stage 3: reflection calls (concurrency={args.llm_concurrency}, streaming to {out_path}) ===")
    t0 = time.time()
    reflection_outputs: list = [None] * len(prepared)
    reflection_prompts: list = [None] * len(prepared)
    reflection_errors: dict[int, str] = {}
    n_completed = 0

    def _call_reflection(i: int):
        p = prepared[i]
        alts = env_results[i]
        if not alts or isinstance(alts, dict):
            return None, None, None
        alt_block = "\n".join(
            f"  {j + 1}. Action a_i^{j + 1}: {a['action']}, resulting state s_i^{j + 1}: {a['next_state']}"
            for j, a in enumerate(alts)
        )
        refl_prompt = REFLECTION_PROMPT_TEMPLATE.format(
            situation=p["situation_block"],
            expert_action=p["expert_action"],
            expert_next_state=p["expert_next_state"],
            alternatives_block=alt_block,
        )
        text, usage = call_deepseek(
            client,
            "You are a careful self-reflecting reasoner.",
            refl_prompt,
            temperature=0.7,
            # NO hard max_tokens cap on reflection — length is controlled by the
            # prompt's soft "~500 word cap" guidance. A mid-sentence-truncated
            # reflection is worse training data than an overlong-but-complete one.
            max_tokens=2048,
            frequency_penalty=0.3,  # suppress DeepSeek's occasional duplicate-paragraph degradation
        )
        return text, usage, refl_prompt

    # Stream each completed reflection to output as soon as it's done.
    # If the process crashes mid-run, completed records are already on disk.
    with open(out_path, "w") as fout, ThreadPoolExecutor(max_workers=args.llm_concurrency) as pool:
        futures = {pool.submit(_call_reflection, i): i for i in range(len(prepared))}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                text, usage, refl_prompt = fut.result()
                reflection_outputs[i] = text
                reflection_prompts[i] = refl_prompt
                if usage:
                    reflection_usage["prompt_tokens"] += usage["prompt_tokens"]
                    reflection_usage["completion_tokens"] += usage["completion_tokens"]
            except Exception as exc:  # noqa: BLE001
                reflection_errors[i] = repr(exc)

            # Write THIS record immediately.
            p = prepared[i]
            fs = filter_state[i]
            record = {
                "item_id": p["item_id"],
                "step": p["step"],
                "task_key": p["task_key"],
                "expert_action": p["expert_action"],
                "expert_next_state": p["expert_next_state"],
                "proposer_raw": proposer_outputs[i],
                "proposer_parsed": fs.get("candidates", []),
                "filter_trace": fs["trace"],
                "filtered_alternatives": env_results[i] if env_results[i] else [],
                "k_final": len(fs["kept"]),
                "reflection_prompt": reflection_prompts[i],
                "reflection_cot": reflection_outputs[i],
                "proposer_error": proposer_errors.get(i),
                "reflection_error": reflection_errors.get(i),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            n_completed += 1
            if n_completed % 100 == 0 or n_completed == len(prepared):
                elapsed_now = time.time() - t0
                rate = n_completed / elapsed_now if elapsed_now > 0 else 0
                eta = (len(prepared) - n_completed) / rate if rate > 0 else 0
                print(
                    f"  [{n_completed:>6}/{len(prepared)}] elapsed={elapsed_now:7.1f}s "
                    f"rate={rate:5.2f}/s eta={eta:7.1f}s   errors={len(reflection_errors)}",
                    flush=True,
                )

    print(f"  done in {time.time() - t0:.1f}s   (errors: {len(reflection_errors)})")

    # Results list (kept for the in-memory aggregate stats below).
    elapsed = time.time() - pipeline_t_start
    # Output file was streamed during Stage 3 — no additional write needed here.
    print(f"\n=== pipeline complete in {elapsed:.1f}s ===")
    print(f"  output: {out_path}  ({out_path.stat().st_size:,} bytes)")
    print()

    # Aggregate stats
    print("=== filter status totals (across all candidates from all states) ===")
    for k, v in sorted(filter_status_total.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22}: {v}")
    print()
    print("=== k_final distribution (how many alternatives survived per state) ===")
    for k in sorted(k_final_dist.keys()):
        print(f"  k={k}: {k_final_dist[k]} states")
    print()
    print("=== hybrid composition: (n_valid_kept, n_invalid_kept) per state ===")
    for (nv, ni), c in sorted(hybrid_split_dist.items()):
        print(f"  ({nv} valid, {ni} invalid): {c} states  ({100*c/len(prepared):.1f}%)")
    n_all_valid = sum(c for (nv, ni), c in hybrid_split_dist.items() if ni == 0 and nv == args.target_k)
    print(f"  -> {n_all_valid}/{len(prepared)} states ({100*n_all_valid/len(prepared):.1f}%) have full K={args.target_k} valid alts (no invalid fillers needed)")
    print()
    print(f"=== token usage ===")
    print(
        f"  proposer  : in={proposer_usage['prompt_tokens']:>6}  "
        f"out={proposer_usage['completion_tokens']:>6}"
    )
    print(
        f"  reflection: in={reflection_usage['prompt_tokens']:>6}  "
        f"out={reflection_usage['completion_tokens']:>6}"
    )
    total_in = proposer_usage["prompt_tokens"] + reflection_usage["prompt_tokens"]
    total_out = proposer_usage["completion_tokens"] + reflection_usage["completion_tokens"]
    print(f"  TOTAL     : in={total_in:>6}  out={total_out:>6}  sum={total_in + total_out:>6}")


if __name__ == "__main__":
    main()

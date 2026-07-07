"""SR rollout: generate self-reflection CoTs for every expert state.

Inputs (joined per (item_id, step)):
  - raw textcraft_train.json: expert Thoughts (for prior-step history)
  - replay_full.jsonl: expert action + expert next-state observation
  - iwm_rollout.jsonl: alt records (alt action + alt next-state) — first 3
    alts per state are kept (deterministic by alt_idx).

For each expert state we call DeepSeek V4 Pro (thinking disabled) once and
ask it to produce a self-reflection monologue that arrives at the chosen
action. Prompt design follows METHOD.md §4.3 with the workspace-wide
leak-suppression rules (no "expert"/"selected"/... labels in output, no
numbered references to alternatives, convergence anchor clause) and
TextCraft's tighter length target.

Output: envs/textcraft/data/rollout/sr_rollout.jsonl
  One record per expert state:
    {
      "item_id", "step", "data_idx",
      "task_obs", "history_text",   # the rendered prompt context
      "expert_action", "expert_next_state",
      "alts": [{"action", "next_state"}, ...],  # up to 3
      "reflection_prompt": "<full prompt as sent to LLM>",
      "reflection_raw": "<LLM response>",
      "reflection_text": "<post-processed: trimmed, dedup-doubled-half if any>",
      "n_alts_used", "input_tokens", "output_tokens", "wall_time_s"
    }

Run modes:
  --dry-run        : build prompts only, no API calls. Prints token estimate.
  --limit N        : only process the first N expert states (smoke).
  --workers N      : ThreadPoolExecutor concurrency for live API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

K_SR = 3
RAW_DEFAULT = "envs/textcraft/data/raw/textcraft_train.json"
REPLAY_DEFAULT = "envs/textcraft/data/replay/replay_full.jsonl"
ROLLOUT_DEFAULT = "envs/textcraft/data/rollout/iwm_rollout.jsonl"
OUTPUT_DEFAULT = "envs/textcraft/data/rollout/sr_rollout.jsonl"

# DeepSeek API (TEAM_GUIDE §1.2)
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE = "https://api.deepseek.com"


SR_SYSTEM_PROMPT = """\
You are looking at a Minecraft crafting situation where an agent must decide \
its next action, given a list of available crafting recipes and the current \
state of its inventory and environment.

Your task is to write a self-reflection — an internal monologue that arrives \
at the action the agent takes by considering possible alternatives and the \
available evidence.

CRITICAL FRAMING:
- The agent has ALREADY CHOSEN the action listed under "The agent's \
chosen action". This is not a tentative attempt and the action's outcome \
has not yet happened from the agent's perspective.
- Your job is to write the reasoning the agent uses to ARRIVE AT that \
exact choice. The outcomes shown for the chosen action and the alternatives \
are forward-looking projections provided to help you write the rationale, \
not events that have already occurred.
- Failed-looking exploratory actions (e.g. trying to `get` an item that \
turns out not to be obtainable, attempting a `craft` to confirm what is \
missing) are valid choices in their own right. They are deliberate \
information-gathering moves. Your reflection must DEFEND why the agent \
decided on this exact action even if its projected outcome is a failure or \
non-progress.

The reflection becomes the agent's training target: at inference time the \
agent will produce this kind of monologue WITHOUT an external label telling \
it which action is the right one. Your reflection MUST therefore read as the \
agent's own reasoning that converges on the action.

Guidelines:
- Stay strictly within the provided information.
- Avoid meta-commentary about being an AI.
- Use natural, step-by-step reasoning.
- Focus on logical decision-making grounded in the recipe list and the \
agent's current inventory.

Constraints on phrasing (these matter — violating them corrupts the \
training signal):
- BANNED VOCABULARY in your output: "expert", "selected", "chosen", \
"correct", "right action", "best option", "optimal", "preferred". The \
monologue is the agent's own thinking; there is no privileged "expert" \
label for it to reference.
- DO NOT refer to alternatives by numbered or external labels (e.g. \
"Action 1", "Alternative 2", "option a"). Use natural inline phrasing: \
"I could try X", "Another option is to Y", "I'm not sure if Z would help".

Convergence (CRITICAL — the most common failure mode):
- The action the agent takes IS the action it will commit to; your job is \
to write reasoning that ARRIVES AT that action, not to evaluate whether it \
is the best one.
- The monologue MUST converge on the action the agent takes. Do NOT let \
the monologue end by recommending an alternative — even if the alternative \
appears more progress-making.
- When the agent's action is a state-checking action like `inventory` \
(which produces no progress toward the goal), the reflection MUST justify \
this caution. For example: "I'd like to act quickly, but committing to \
gathering before I know what I already have could be wasteful — better to \
check my inventory first." Do NOT pivot at the end to "I'll grab the X" \
just because gathering looks more goal-directed; the reasoning's job is \
to defend the cautious choice.
- The final sentence should explicitly name (or clearly reference) the \
action the agent takes, framed as the next step the agent will commit to.

Length:
- Target around 250 words; do not exceed 350 words.

Output: Directly write the self-reflection monologue. No extra headings, \
disclaimers, bullet points, or external notes.\
"""


def build_sr_user_prompt(
    task_obs: str,
    history_text: str,
    current_obs: str,
    expert_action: str,
    expert_next_state: str,
    alts: list,
) -> str:
    """Build the per-state user prompt for the reflection LLM.

    Note: the input slot for the expert action reads "The action the agent
    takes here" (not "Expert Action") to keep the LLM from echoing
    privileged labels — pitfalls.md "label leakage" entry.
    """
    pieces = [
        "Situation:",
        task_obs,
        "",
    ]
    if history_text.strip():
        pieces.append("History of prior reasoning and environment responses:")
        pieces.append(history_text)
        pieces.append("")
    pieces.append(f"Current observation: {current_obs}")
    pieces.append("")
    pieces.append(f"The agent's chosen action: {expert_action}")
    pieces.append(
        f"(For reference only — this action would produce the environment "
        f"response: {expert_next_state})"
    )
    pieces.append("")
    if alts:
        pieces.append(
            "For comparison, here are some other actions the agent could have "
            "considered, and what each would have produced:"
        )
        for alt in alts:
            pieces.append(f"  - {alt['action']}  -->  {alt['next_state']}")
        pieces.append("")
    pieces.append(
        "Write the self-reflection that arrives at the chosen action now. "
        "Remember: the chosen action has already been decided; your job is to "
        "reconstruct the reasoning that defends it, not to second-guess it."
    )
    return "\n".join(pieces)


def render_history(raw_conversations: list, replayed_observations: list, up_to_step: int) -> str:
    """Same shape as build_iwm_sft.render_history but without the leading task obs.

    Returns just the prior step blocks (Thought + Action + Obs) joined.
    Empty string if up_to_step == 0.
    """
    pieces = []
    for i in range(up_to_step):
        gpt_turn_idx = 2 * i + 3
        if gpt_turn_idx >= len(raw_conversations):
            break
        pieces.append(raw_conversations[gpt_turn_idx]["value"])
        if i < len(replayed_observations):
            pieces.append(f"Instruction:\n{replayed_observations[i]}")
    return "\n\n".join(pieces)


def estimate_tokens(text: str) -> int:
    """Crude token estimate: ~3.5 chars per token for English+code mix."""
    return max(1, int(len(text) / 3.5))


def detect_doubled(text: str, head_chars: int = 100, tol: int = 50) -> bool:
    """Heuristic from pitfalls: DeepSeek occasionally writes the same reflection
    twice in one response. Detect by finding the first `head_chars` substring
    appearing again later, with the second copy similar in length to the first.
    """
    if not text or len(text) < 2 * head_chars:
        return False
    head = text[:head_chars]
    rest = text[head_chars:]
    pos = rest.find(head)
    if pos < 0:
        return False
    first_len = head_chars + pos
    second_len = len(text) - first_len
    return abs(first_len - second_len) < tol


def maybe_dedup_doubled(text: str) -> tuple[str, bool]:
    """If text is a clean byte-doubled response, return the first half + True.
    Otherwise return text unchanged + False."""
    if not detect_doubled(text):
        return text, False
    half = len(text) // 2
    if text[:half] == text[half : 2 * half] or abs(len(text[:half]) - len(text[half:])) < 50:
        return text[:half].rstrip(), True
    return text, False


def call_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> dict:
    """One sync DeepSeek call. Returns dict with reflection_raw + usage."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "frequency_penalty": 0.3,
        "thinking": {"type": "disabled"},
    }
    t0 = time.time()
    r = requests.post(
        f"{DEEPSEEK_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=120,
    )
    r.raise_for_status()
    j = r.json()
    return {
        "reflection_raw": j["choices"][0]["message"]["content"],
        "input_tokens": j.get("usage", {}).get("prompt_tokens", 0),
        "output_tokens": j.get("usage", {}).get("completion_tokens", 0),
        "wall_time_s": time.time() - t0,
    }


def process_state(unit, dry_run: bool):
    """Process one (item_id, step) state. `unit` is a fully-prepared dict."""
    user_prompt = build_sr_user_prompt(
        task_obs=unit["task_obs"],
        history_text=unit["history_text"],
        current_obs=unit["current_obs"],
        expert_action=unit["expert_action"],
        expert_next_state=unit["expert_next_state"],
        alts=unit["alts"],
    )
    record = {
        "item_id": unit["item_id"],
        "step": unit["step"],
        "data_idx": unit["data_idx"],
        "task_obs": unit["task_obs"],
        "history_text": unit["history_text"],
        "expert_action": unit["expert_action"],
        "expert_next_state": unit["expert_next_state"],
        "alts": unit["alts"],
        "n_alts_used": len(unit["alts"]),
        "reflection_prompt_user": user_prompt,
        "reflection_prompt_system": SR_SYSTEM_PROMPT,
    }
    if dry_run:
        record["reflection_raw"] = None
        record["reflection_text"] = None
        record["doubled_dedup"] = None
        record["est_input_tokens"] = estimate_tokens(SR_SYSTEM_PROMPT) + estimate_tokens(user_prompt)
        return record

    resp = call_deepseek(SR_SYSTEM_PROMPT, user_prompt)
    refl_raw = resp["reflection_raw"]
    refl_text, doubled = maybe_dedup_doubled(refl_raw)
    record.update(
        {
            "reflection_raw": refl_raw,
            "reflection_text": refl_text,
            "doubled_dedup": doubled,
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "wall_time_s": resp["wall_time_s"],
        }
    )
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--replay", default=REPLAY_DEFAULT)
    ap.add_argument("--rollout", default=ROLLOUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N expert states (smoke).",
    )
    ap.add_argument(
        "--limit-traj",
        type=int,
        default=None,
        help="Process only the first N trajectories (sorted by item_id integer).",
    )
    ap.add_argument(
        "--include-iids",
        default="",
        help="Comma-separated item_ids to include IN ADDITION to --limit-traj's window. "
        "Useful for cherry-picking long trajectories. Example: "
        "'textcraft_523,textcraft_204'.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and emit token estimates; no API calls.",
    )
    args = ap.parse_args()

    raw_path = Path(args.raw).resolve()
    replay_path = Path(args.replay).resolve()
    rollout_path = Path(args.rollout).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"raw     : {raw_path}")
    print(f"replay  : {replay_path}")
    print(f"rollout : {rollout_path}")
    print(f"output  : {out_path}")
    print(f"workers : {args.workers}")
    print(f"dry_run : {args.dry_run}")

    print("loading raw textcraft_train.json...")
    with open(raw_path) as f:
        raw = json.load(f)
    raw_by_iid = {r["item_id"]: r for r in raw}

    print("loading replay_full.jsonl...")
    replay_by_iid = {json.loads(l)["item_id"]: json.loads(l) for l in open(replay_path)}

    print("loading iwm_rollout.jsonl...")
    rollout_initial: dict = {}
    rollout_alts: dict = defaultdict(list)
    for line in open(rollout_path):
        r = json.loads(line)
        if r["kind"] == "initial":
            rollout_initial[r["item_id"]] = r
        elif r["kind"] == "alt":
            rollout_alts[(r["item_id"], r["step"])].append(r)
    for k in rollout_alts:
        rollout_alts[k].sort(key=lambda x: x["alt_idx"])

    # Build units list: one per (item_id, step)
    print("building units...")
    units = []
    iids_sorted = sorted(replay_by_iid.keys(), key=lambda x: int(x.split("_")[1]))
    if args.limit_traj is not None:
        kept = iids_sorted[: args.limit_traj]
    else:
        kept = iids_sorted
    extra = [s.strip() for s in args.include_iids.split(",") if s.strip()]
    for iid in extra:
        if iid not in kept and iid in replay_by_iid:
            kept.append(iid)
    iids_sorted = kept
    print(f"  selected {len(iids_sorted)} traj"
          + (f" (limit={args.limit_traj}, extra={len(extra)})" if extra or args.limit_traj else ""))
    for iid in iids_sorted:
        replay = replay_by_iid[iid]
        if iid not in raw_by_iid:
            continue
        if iid not in rollout_initial:
            continue
        conv = raw_by_iid[iid]["conversations"]
        data_idx = replay["data_idx"]
        replayed_obs = [s["observation"] for s in replay["steps"]]
        actions_clean = [s["action"] for s in replay["steps"]]
        task_obs = rollout_initial[iid]["initial_obs"]
        # current_obs at step i = the env's response to step i-1 (for i==0 it's the task obs context)
        for i, expert_action in enumerate(actions_clean):
            history_text = render_history(conv, replayed_obs, up_to_step=i)
            current_obs = replayed_obs[i - 1] if i > 0 else "(start: no actions taken yet)"
            alts_full = rollout_alts.get((iid, i), [])
            alts = [
                {"action": a["action"], "next_state": a["next_state"]}
                for a in alts_full[:K_SR]
            ]
            units.append(
                {
                    "item_id": iid,
                    "step": i,
                    "data_idx": data_idx,
                    "task_obs": task_obs,
                    "history_text": history_text,
                    "current_obs": current_obs,
                    "expert_action": expert_action,
                    "expert_next_state": replayed_obs[i],
                    "alts": alts,
                }
            )

    if args.limit is not None:
        units = units[: args.limit]
        print(f"  limited to first {len(units)} units")
    print(f"  total units: {len(units)}")

    # In dry-run mode print token estimates and a sample prompt
    if args.dry_run:
        total_est = 0
        for u in units:
            rec = process_state(u, dry_run=True)
            total_est += rec["est_input_tokens"]
        avg_in = total_est / max(1, len(units))
        print(f"\n--- token estimates ---")
        print(f"  states                  : {len(units)}")
        print(f"  avg input tokens / state: ~{avg_in:.0f}")
        print(f"  total input tokens (est): ~{total_est:,}")
        print(f"  output target           : ~250 words ~ ~330 tokens/state")
        est_out = 330 * len(units)
        print(f"  total output tokens(est): ~{est_out:,}")
        # DeepSeek V4 Pro pricing (approx, May 2026): $0.27/M input + $1.1/M output
        cost = total_est / 1e6 * 0.27 + est_out / 1e6 * 1.1
        print(f"  cost estimate (DeepSeek V4 Pro): ~${cost:.2f}")
        # Write empty placeholder output for inspection?
        sample = units[0]
        rec = process_state(sample, dry_run=True)
        print(f"\n--- sample system prompt ({len(SR_SYSTEM_PROMPT)} chars) ---")
        print(SR_SYSTEM_PROMPT[:500] + "..." if len(SR_SYSTEM_PROMPT) > 500 else SR_SYSTEM_PROMPT)
        print(f"\n--- sample user prompt for {sample['item_id']} step {sample['step']} "
              f"({len(rec['reflection_prompt_user'])} chars) ---")
        print(rec["reflection_prompt_user"])
        return

    # Live mode
    print(f"\nrunning {len(units)} LLM calls with {args.workers} workers...")
    t0 = time.time()
    n_done = 0
    n_doubled = 0
    write_lock = threading.Lock()
    with open(out_path, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_state, u, False): (u["item_id"], u["step"]) for u in units}
        for fut in as_completed(futures):
            iid_step = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                print(f"  failed on {iid_step}: {exc!r}", flush=True)
                continue
            with write_lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            n_done += 1
            if rec.get("doubled_dedup"):
                n_doubled += 1
            if n_done % 20 == 0 or n_done == len(units):
                el = time.time() - t0
                rate = n_done / el if el > 0 else 0
                eta = (len(units) - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done:>4}/{len(units)}] elapsed={el:6.1f}s rate={rate:4.2f}/s "
                    f"eta={eta:6.1f}s doubled_dedup={n_doubled}",
                    flush=True,
                )
    el = time.time() - t0
    print(f"\nDONE in {el:.1f}s.")
    print(f"  output: {out_path}  ({out_path.stat().st_size:,} bytes)")
    print(f"  doubled_dedup observed: {n_doubled}")


if __name__ == "__main__":
    main()

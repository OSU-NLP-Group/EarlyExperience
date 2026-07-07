"""SR (self-reflection) smoke for AppWorld.

For a handful of expert states:
  1. Load expert_outcomes.jsonl (task, step) → expert_code + outcome_summary
  2. Load probe_full_summarized.jsonl → per-state alts with summaries
  3. Sub-sample K=3 alts per state (2 data + 1 http error, env-meaningful only)
  4. Build reflection prompt (METHOD.md §4.3 + scienceworld/bfcl guardrails)
  5. Call DeepSeek V4 Pro → first-person internal monologue converging on expert code
  6. Post-hoc leak check for banned vocab / numbered labels

Only Pro reflection calls here (no env, no Flash) — expert outcomes already
captured by run_expert_outcomes.py.
"""
import os, sys, json, time, random, collections
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
PRO = "deepseek-v4-pro"

EXPERT_PATH = Path("envs/appworld/data/rollout/expert_outcomes.jsonl")
ALTS_PATH = Path("envs/appworld/data/rollout/probe_full_summarized.jsonl")
OUT_PATH = Path("envs/appworld/data/_recon/smoke_sr.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

APPWORLD_ROOT = os.environ.get(
    "APPWORLD_ROOT",
    "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root",
)

TASK_IDS = None  # picked dynamically for coverage
N_STATES_TOTAL = int(os.environ.get("N_STATES_TOTAL", "20"))
K_SR = 0  # v5: NO alternatives — pure "why is this step reasonable" CoT.
          # AppWorld's action space is too large for random alts to be genuine
          # competitors; with alts we get hallucination, without we get honest
          # forward reasoning grounded in the real observed outcome (not STaR).


REFLECTION_SYSTEM = """You are helping build a training dataset that teaches an AI agent to reason well before it acts. Read this carefully — understanding WHAT we are building and WHY will let you produce exactly the right thing.

WHAT WE ARE BUILDING
Each example teaches the agent the `<think>` reasoning it should have at one step of a multi-step AppWorld task (APIs are called as `apis.<app>.<endpoint>(...)`). We show you a task, the history of steps already taken, and the single line of code the agent runs at THIS step. Your job is to write the inner monologue that leads to that line — the honest reasoning a capable agent would genuinely have at that moment.

THE ONE PRINCIPLE EVERYTHING FOLLOWS FROM
After training, the agent will stand exactly where this monologue stands: it will see the task and the history so far, and NOTHING else. It will not have run the line yet, so it won't know what the line returns. It won't be handed a list of options. It must generate reasoning like this on its own, then act.

So the monologue must be reproducible from that standpoint — everything in it has to be derivable from the task, the history, and ordinary knowledge of how these apps behave. The moment the reasoning leans on something the agent could not have at that instant, we are teaching it to depend on information it will never have, and at test time it will hallucinate to fill the gap. That is the failure we are designing this data to avoid. Keep the monologue strictly within what the agent legitimately knows and can predict.

Two consequences of this principle worth making explicit, because they are easy to slip on:
  • You (the author) are additionally shown the REAL outcome this line produced. Treat it as a private compass that reassures you the reasoning heads the right way — never as content. The agent has not seen it. So no exact counts ("19 songs"), no IDs ("alarm 122"), no returned names/titles, no "it worked". And do not describe the SHAPE of a response you haven't gotten — do not assert it "returns a dict with a `songs` key" or "each item has an `enabled` field". Speak in honest expectation ("this should give me the list of alarms", "I'll get the credentials back"), the way someone reasons before seeing the result.
  • Reason forward, in the voice of deciding — "I need X, so I'll do Y", "this should get me Z" — not backward from a known result.

WHAT MAKES THE REASONING GOOD (not just permitted)
We want the genuine decision reasoning a thoughtful agent has: where am I in the task, what do I need next, and why is THIS line the move that gets it — including what would go wrong if I skipped it. That is the transferable skill. A few things that dilute it:
  • Narrating the code instead of reasoning. If the monologue just re-says in English what the line literally does ("loops through pages 0-9 and collects items"), it teaches nothing. Explain intent and rationale; when the code is self-evident, say less.
  • Planning the whole rest of the episode. Justify THIS step. A short "so I can use it next" is natural; a roadmap of the next five steps is not — the agent decides those later, from their own results.
  • Treating a small setup line (a bare assignment, a temp variable) as beneath explanation. It is a real move; give its purpose in a clause ("so I have the id ready to use") and move on.
  • Staying honest about the apps: reason consistently with how endpoints were used in the history; where you're unsure of an endpoint's exact behavior, describe intent ("authenticate", "retrieve the list") rather than inventing a signature.
  • Never frame it as choosing a labeled "correct/best/optimal/right/expert" action — there is no external key; you are simply reasoning your way to the sensible move.

LENGTH: one tight paragraph, and only as long as the reasoning genuinely needs — most steps are 2-4 sentences, a trivial setup line one. Aim under ~120 words. Brevity is itself a quality signal here; do not pad.

OUTPUT: the monologue paragraph only. No code block, no restatement of the line — the code is attached separately."""


def build_reflection_user(task, history, expert_code, expert_outcome, alts=None):
    lines = [f"## Task\n{task}", ""]
    if history:
        lines.append("## Interaction history — steps ALREADY COMPLETED before this step")
        for i, h in enumerate(history):
            lines.append(f"### Step {i}")
            lines.append(f"Code:\n{h['code']}")
            lines.append(f"Result: {h['outcome']}")
        lines.append("")
    lines.append(f"## THIS STEP — the line of code the agent is about to run")
    lines.append(expert_code)
    lines.append(f"\n## [AUTHOR-ONLY, INVISIBLE TO THE AGENT] The real outcome this line produced")
    lines.append(f"(Use this ONLY to confirm your reasoning heads the right way. The agent has NOT run the line — do NOT quote counts, IDs, names, response fields, or success/failure from it.)")
    lines.append(expert_outcome)
    lines.append(f"\nWrite ONLY the agent's first-person inner monologue justifying why running this line is the sensible move at this point — "
                 f"forward/predictive voice, none of the author-only specifics, no code narration, no fabricated response fields. "
                 f"Output the monologue paragraph and nothing else (no code block).")
    return "\n".join(lines)


def reflect(task, history, expert_code, expert_outcome, alts=None):
    user = build_reflection_user(task, history, expert_code, expert_outcome, alts)
    r = client.chat.completions.create(
        model=PRO,
        messages=[{"role": "system", "content": REFLECTION_SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0.7,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return r.choices[0].message.content or "", user, {
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
    }


def check_leakage(text):
    # phrase-level: only flag when a normative label attaches to the action noun
    banned = [
        "expert action", "expert code", "expert snippet", "the expert",
        "correct action", "correct choice", "correct code", "correct snippet",
        "correct answer", "correct move",
        "the right action", "the right choice", "the right code",
        "the right move", "the right thing to do",
        "the best action", "the best choice", "the best code", "the best move",
        "the best option", "the best snippet",
        "the optimal action", "the optimal choice", "the optimal code",
        "the chosen action", "the chosen code", "the selected action",
        "action 1", "action 2", "action 3", "action 4", "action 5",
        "alternative 1", "alternative 2", "alternative 3",
        "option a", "option b", "option c",
    ]
    low = text.lower()
    return [w for w in banned if w in low]


def main():
    # Load expert per (task, step)
    expert_by_state = {}
    for line in EXPERT_PATH.open():
        r = json.loads(line)
        expert_by_state[(r["task_id"], r["step"])] = r

    # Load alts per (task, step)
    alts_by_state = collections.defaultdict(list)
    for line in ALTS_PATH.open():
        r = json.loads(line)
        alts_by_state[(r["task_id"], r["step"])].append(r)

    # Diverse task pick: sample tasks with varying required_app counts and lengths.
    from appworld.ground_truth import GroundTruth
    from appworld import load_task_ids
    all_train = load_task_ids("train")
    task_meta = []
    for tid in all_train:
        gt = GroundTruth.load(tid, mode="full")
        n_steps = max((k for (t, k) in expert_by_state if t == tid), default=-1) + 1
        if n_steps == 0: continue
        task_meta.append({"tid": tid, "n_apps": len(gt.required_apps), "n_steps": n_steps})
    rng = random.Random(42)
    rng.shuffle(task_meta)
    # take a spread: 3 short single-app, 3 mid two-app, 2 long three-app
    single = [t for t in task_meta if t["n_apps"] == 1][:4]
    dual   = [t for t in task_meta if t["n_apps"] == 2][:3]
    tri    = [t for t in task_meta if t["n_apps"] >= 3][:2]
    picked_tasks = single + dual + tri
    print(f"Picked {len(picked_tasks)} tasks: "
          + ", ".join(f"{t['tid']}({t['n_apps']}app/{t['n_steps']}step)" for t in picked_tasks))

    task_specs = {}
    for t in picked_tasks:
        specs = json.load((Path(APPWORLD_ROOT)/"data"/"tasks"/t["tid"]/"specs.json").open())
        task_specs[t["tid"]] = specs["instruction"]

    # Choose N_STATES_TOTAL states across trajectories, weighted by n_steps
    chosen = []
    for t in picked_tasks:
        # pick ~ 2-3 states per task at fractions 0.15, 0.45, 0.75
        n = t["n_steps"]
        for frac in (0.15, 0.45, 0.75):
            si = min(n-1, max(0, int(n*frac)))
            chosen.append((t["tid"], si))
    chosen = chosen[:N_STATES_TOTAL]
    print(f"Chosen {len(chosen)} states\n")

    results = []
    total_in, total_out = 0, 0

    for tid, si in chosen:
        # history = all expert steps < si with their summaries
        n = max(k for (t, k) in expert_by_state if t == tid) + 1
        history = [{"code": expert_by_state[(tid, j)]["expert_code"],
                    "outcome": expert_by_state[(tid, j)]["outcome_summary"]}
                   for j in range(si)]
        expert_code = expert_by_state[(tid, si)]["expert_code"]
        expert_outcome = expert_by_state[(tid, si)]["outcome_summary"]

        # v5: K=0 — no alternatives. Pure "why is this step reasonable" CoT.
        print(f"\n{'='*78}\n[{len(results)+1}/{len(chosen)}] {tid} step {si}/{n-1}\n{'='*78}")
        print(f"TASK: {task_specs[tid]}")
        print(f"\nEXPERT CODE:\n{expert_code}")
        print(f"\nEXPERT OUTCOME:\n{expert_outcome}")

        t0 = time.time()
        cot, prompt, usage = reflect(task_specs[tid], history, expert_code, expert_outcome)
        dt = time.time() - t0
        total_in += usage["prompt_tokens"]
        total_out += usage["completion_tokens"]

        # model outputs monologue only now; the SFT builder attaches expert code deterministically
        monologue = cot.strip()
        leak = check_leakage(monologue)
        wc = len(monologue.split())
        # doubling detector (DeepSeek mode-collapse, pitfalls.md)
        doubled = len(monologue) > 80 and monologue.count(monologue[:60]) >= 2
        # did the model wrongly emit a code block despite instruction?
        stray_code = "<code>" in cot
        print(f"\n--- monologue ({wc} words, leak={leak}, doubled={doubled}, "
              f"stray_code={stray_code}, {dt:.1f}s, "
              f"in={usage['prompt_tokens']} out={usage['completion_tokens']}) ---\n{monologue}")

        results.append({
            "task_id": tid, "step": si, "expert_code": expert_code,
            "expert_outcome": expert_outcome,
            "reflection": monologue, "word_count": wc, "leak": leak,
            "doubled": doubled, "stray_code": stray_code,
            "usage": usage,
        })

    # Summary
    cost = total_in*0.27/1e6 + total_out*1.10/1e6   # Pro pricing
    n_leak = sum(1 for r in results if r["leak"])
    n_doubled = sum(1 for r in results if r.get("doubled"))
    n_stray = sum(1 for r in results if r.get("stray_code"))
    wcs = [r["word_count"] for r in results]
    print(f"\n\n{'='*78}")
    print(f"n_states: {len(results)}   cost=${cost:.4f} (Pro)")
    print(f"Leak check (banned vocab): {n_leak}/{len(results)}")
    print(f"Doubled (mode-collapse):   {n_doubled}/{len(results)}")
    print(f"Stray code block:          {n_stray}/{len(results)}")
    print(f"Word counts: {sorted(wcs)}  (cap ~120)")

    json.dump({
        "n_states": len(results),
        "totals": {"input_tokens": total_in, "output_tokens": total_out, "cost_usd": round(cost, 4)},
        "n_leak": n_leak,
        "results": results,
    }, OUT_PATH.open("w"), indent=2, default=str)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()

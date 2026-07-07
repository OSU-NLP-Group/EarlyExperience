"""Regenerate ScienceWorld SR reflections with prompt v2 (leak-proof + honest +
length-adaptive + environment-facts), over the post-filter surviving states.

Reuses sr_rollout.jsonl's proposer candidates + env-probed next-states — only the
reflection LLM call is re-run. NO proposer re-run, NO env re-rollout.

Surviving states = all 39,700 D_expert states MINUS:
  - post-completion states (pre-action score already == 100)            [replay_full.jsonl]
  - env-flagged no-op states (expert_next_state contains "already")     [sr_rollout.jsonl]
  - k_final < 3 / proposer or reflection error / missing prompt

Output: envs/scienceworld/data/rollout/sr_rollout_v2.jsonl
  one line per state: {item_id, step, expert_action, reflection_cot_v2,
                       prompt_tokens, completion_tokens, error}
Resumable: re-running skips (item_id, step) already present in the output file.

Run (from workspace root, DEEPSEEK_API_KEY in env):
    python envs/scienceworld/scripts/regen_reflection_v2.py [--concurrency 100] [--limit N]
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

SR = "envs/scienceworld/data/rollout/sr_rollout.jsonl"
REPLAY = "envs/scienceworld/data/replay/replay_full.jsonl"
OUT = "envs/scienceworld/data/rollout/sr_rollout_v2.jsonl"
MODEL = "deepseek-v4-pro"
BASE_URL = "https://api.deepseek.com"

PROMPT_V2 = """\
You will be presented with a situation in which a text-based agent is acting. The agent ends up taking one specific action at this step; your task is to write the agent's internal monologue that would naturally arrive at that decision.

- Situation (s_i):
{situation}
- The action the agent takes here: {expert_action}
- Other actions the agent considered (but did not take):
{alternatives_block}

Reference outcomes — FOR YOUR PRIVATE JUDGMENT ONLY. Use these to understand which options help and which do not, but NEVER quote, paraphrase, or narrate any of them in the monologue (at decision time the agent has not executed anything yet and cannot have seen these):
{reference_outcomes_block}

Environment fact about waiting: `wait`/`wait1` only advance time and reveal nothing by themselves. Check the history: a wait is justified only if it shows a time-dependent process already in motion (a substance heating on an active source, or a just-completed circuit settling) that needs time to progress. If the history shows the task is essentially done or no such process is active, waiting accomplishes nothing — say so honestly, do not invent an effect. A door, once opened, stays open — it does not close on its own.

Your monologue should:
1. Analyze the situation and the goal.
2. Briefly consider each of the other actions in turn, working out why the agent does not take it — expressed as expectation ("X probably won't help because..."), never as an observed result.
3. Arrive at the action above being the right next step, justified by what it is EXPECTED to achieve toward the goal — not by any already-observed result.
4. Highlight any relevant clues, constraints, or consequences from the situation.

Guidelines:
- Stay strictly within the provided information. Reason ONLY from what the situation, history, and ordinary commonsense about a physical lab support. Do NOT invent environment mechanics, object properties, or facts that are not given. If the agent is acting on a hunch or to explore, say so honestly ("I'm not certain it's here, but it's worth checking") rather than fabricating a reason it must be right.
- You always know your current room, your inventory, and the latest observation from the history. State these as fact. Only express uncertainty about things you genuinely cannot know yet (e.g. what is inside an unopened room).
- Match length and confidence to the decision. Most steps — navigation, picking something up, a routine step in a known sequence — need only 2-4 sentences. Reserve longer reasoning (up to ~180 words) for genuinely non-obvious decisions (interpreting a test result, choosing among several substantive actions). Never pad a thin reason into a confident essay.
- Avoid meta-commentary about being an AI.

CRITICAL — this monologue becomes TRAINING DATA. At inference the agent has NO label distinguishing the action it takes from the ones it merely considered, and has NOT yet executed anything. Therefore:
- NEVER state, quote, paraphrase, or describe the outcome of ANY action (the one taken or the others) as already observed. Forbidden: "the resulting state shows...", "which results in...", "the observation says/shows...", "now I see/find...", "this gives me...", "I end up...". Use ONLY anticipatory language: "this should...", "I expect this will...", "to check whether...", "going there should let me...".
- Do NOT use the words "expert", "selected", "chosen", "correct choice", "right action", "best option", "optimal action", "best alternative". Do NOT say "the prior trajectory" or "the alternatives listed" — reason as the agent, not about the data.
- Do NOT refer to the other options by numbered labels ("Action 1", "Alternative 1"). Phrase them inline: "I could try X", "Another option is to Y".
- Write as if YOU are the agent freshly deliberating, with honest hedging where the situation warrants it, NOT a confident essay justifying a pre-known answer.
- Write as ONE continuous paragraph. No paragraph breaks, bullets, lists, or headings.

ALSO: the monologue must end by settling on the action the agent takes, never on a different action — but match how confident you sound to how much justification the situation actually supports. For a routine or exploratory action a brief honest rationale is correct and sufficient; do not inflate it.

Output: Directly write the monologue, no extra headings, disclaimers, or external notes.
"""

SYSTEM = "You are a careful self-reflecting reasoner."


def extract_situation(p: str):
    if "- Situation (s_i):\n" not in p:
        return None
    return p.split("- Situation (s_i):\n", 1)[1].split("\n- The action the agent takes here:", 1)[0]


def build_prompt(r):
    sit = extract_situation(r["reflection_prompt"])
    alts = r["filtered_alternatives"]
    ab = "\n".join(f"  - {x['action']}" for x in alts)
    ref = "\n".join(
        [f'  - taking "{r["expert_action"]}" leads to: {r["expert_next_state"]}']
        + [f'  - taking "{x["action"]}" would lead to: {x["next_state"]}' for x in alts]
    )
    return PROMPT_V2.format(
        situation=sit, expert_action=r["expert_action"],
        alternatives_block=ab, reference_outcomes_block=ref,
    )


def load_post_completion(path):
    pc = set()
    for line in open(path):
        rec = json.loads(line)
        if not (rec.get("final_done") and rec.get("final_score") == 100):
            continue
        steps = rec["replay_steps"]
        for i in range(len(steps)):
            prev = steps[i - 1].get("score") if i > 0 else 0
            if (prev or 0) >= 100:
                pc.add((rec["item_id"], i))
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)

    print("loading post-completion set from replay...", flush=True)
    pc = load_post_completion(REPLAY)
    print(f"  post-completion states: {len(pc):,}", flush=True)

    print("selecting surviving states from sr_rollout...", flush=True)
    todo = []
    for line in open(SR):
        r = json.loads(line)
        if r.get("k_final", 0) < 3 or r.get("reflection_error") or r.get("proposer_error"):
            continue
        if not r.get("reflection_prompt"):
            continue
        if (r["item_id"], r["step"]) in pc:
            continue
        if "already" in r["expert_next_state"].lower():   # env-flagged no-op
            continue
        todo.append(r)
    print(f"  surviving states: {len(todo):,}", flush=True)

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                d = json.loads(line)
                if not d.get("error"):
                    done.add((d["item_id"], d["step"]))
            except Exception:
                pass
        print(f"  resume: {len(done):,} already done", flush=True)
    todo = [r for r in todo if (r["item_id"], r["step"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"  to generate now: {len(todo):,}", flush=True)
    if not todo:
        print("nothing to do.", flush=True)
        return

    lock = threading.Lock()
    fout = open(OUT, "a")
    counts = {"ok": 0, "err": 0, "pin": 0, "pout": 0}
    t0 = time.time()

    def work(r):
        prompt = build_prompt(r)
        last = None
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}],
                    temperature=0.7, max_tokens=2048, frequency_penalty=0.3,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                return {
                    "item_id": r["item_id"], "step": r["step"],
                    "expert_action": r["expert_action"],
                    "reflection_cot_v2": resp.choices[0].message.content,
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "error": None,
                }
            except Exception as e:  # noqa
                last = repr(e)
                time.sleep(2 * (attempt + 1))
        return {"item_id": r["item_id"], "step": r["step"],
                "expert_action": r["expert_action"], "reflection_cot_v2": None,
                "prompt_tokens": 0, "completion_tokens": 0, "error": last}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, r): i for i, r in enumerate(todo)}
        for n, f in enumerate(as_completed(futs), 1):
            d = f.result()
            with lock:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                fout.flush()
                if d["error"]:
                    counts["err"] += 1
                else:
                    counts["ok"] += 1
                    counts["pin"] += d["prompt_tokens"]
                    counts["pout"] += d["completion_tokens"]
            if n % 500 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / el if el else 0
                eta = (len(todo) - n) / rate if rate else 0
                print(f"  {n:,}/{len(todo):,}  ok={counts['ok']:,} err={counts['err']}  "
                      f"in={counts['pin']:,} out={counts['pout']:,}  "
                      f"{rate:.1f}/s  eta {eta/60:.0f}min", flush=True)
    fout.close()
    print(f"\nDONE. ok={counts['ok']:,} err={counts['err']}  "
          f"tokens in={counts['pin']:,} out={counts['pout']:,}  "
          f"wall={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()

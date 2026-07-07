"""Build the final SFT JSONL files for AppWorld.

Consumes:
  - data/rollout/expert_outcomes.jsonl          (931 expert states: code + outcome_summary)
  - data/rollout/probe_full_summarized.jsonl    (84,863 alts: call + summary + bucket)
  - data/rollout/reflection_full.jsonl          (931 reflections)

Produces (data/sft/):
  - expert_sft.jsonl          imitation baseline: s_i -> expert code
  - reflection_sft.jsonl      SR: s_i -> <think>reflection</think><code>expert</code>
  - iwm_sft_balanced.jsonl    IWM controlled 70/30: per state 7 data + 2 http + 1 py + expert transition
  - iwm_sft_full.jsonl        IWM paper-faithful: every alt + expert transition (no filter)

Shared state serialization (identical s_i across all files):
  system = role prompt (AGENT for expert/reflection, WORLD-MODEL for iwm)
  user   = task + interaction history (prior expert code + outcome SUMMARY)
           (+ candidate action, iwm only)
IWM assistant = next-state summary. expert assistant = code. reflection assistant = think+code.

No LLM calls. Deterministic. Filters:
  - reflection: drop records with code_matches==False (empty/garbled <code>). Documented.
  - iwm_balanced: per-state stratified sample with fixed seed.
"""
import json, random, collections
from pathlib import Path

ROLL = Path("envs/appworld/data/rollout")
OUT = Path("envs/appworld/data/sft")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# ---------------- system prompts ----------------
AGENT_SYSTEM = """You are an AI assistant that solves multi-step tasks in an AppWorld REPL environment. You interact with apps through their APIs, called as `apis.<app>.<endpoint>(...)`. At each step you write ONE top-level Python statement; the environment executes it and returns its output, and based on that you write the next statement. When the task is complete you MUST call `apis.supervisor.complete_task(...)` (with an `answer=` argument if the task asks a question).

At each step you may reason inside a <think>...</think> block, then you emit the code inside a <code>...</code> block. If you have nothing to reason about, emit the <code> block directly."""

WM_SYSTEM = """You are a WORLD MODEL for an AppWorld REPL environment. You are given the state of an ongoing task — the task description, the interaction history of prior steps, and a candidate Python snippet that is about to be executed. Predict, in ONE concise sentence, what the environment will return when that snippet runs.

Rules:
- Describe only the environment's response — the state change, the data returned, or the error. State it factually, preserving specific identifiers/values.
- Do NOT solve the task, do NOT continue the trajectory, do NOT write code.
- Do NOT judge whether the snippet is a good choice.
- If the snippet errors, state the error factually (status code, exception type) without analyzing why.
Output ONLY the one-sentence prediction."""


# ---------------- state serialization ----------------
def render_history(history):
    """history: list of {code, outcome_summary}. Returns text block."""
    if not history:
        return ""
    parts = ["# Interaction history (previous steps and their results):"]
    for i, h in enumerate(history):
        parts.append(f"## Step {i}")
        parts.append(f"Code:\n{h['code']}")
        parts.append(f"Result: {h['outcome_summary']}")
    return "\n".join(parts)


def build_state_user(task, history):
    """The shared s_i serialization used by all three categories."""
    blocks = [f"# Task\n{task}"]
    hist = render_history(history)
    if hist:
        blocks.append(hist)
    blocks.append("# Now write the code for the next step.")
    return "\n\n".join(blocks)


def build_iwm_user(task, history, candidate_code):
    """s_i + the candidate action to predict (world-model input)."""
    blocks = [f"# Task\n{task}"]
    hist = render_history(history)
    if hist:
        blocks.append(hist)
    blocks.append(f"# Candidate code to predict the outcome of:\n{candidate_code}")
    return "\n\n".join(blocks)


def rec(system, user, assistant):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def main():
    # ---- load ----
    expert = {}
    for l in (ROLL / "expert_outcomes.jsonl").open():
        r = json.loads(l); expert[(r["task_id"], r["step"])] = r
    max_step = collections.defaultdict(int)
    for (tid, si) in expert:
        max_step[tid] = max(max_step[tid], si)

    alts_by_state = collections.defaultdict(list)
    for l in (ROLL / "probe_full_summarized.jsonl").open():
        r = json.loads(l); alts_by_state[(r["task_id"], r["step"])].append(r)

    reflections = {}
    for l in (ROLL / "reflection_full.jsonl").open():
        r = json.loads(l); reflections[(r["task_id"], r["step"])] = r

    # task instructions
    import os
    root = os.environ.get("APPWORLD_ROOT",
        "/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root")
    instr_cache = {}
    def instr(tid):
        if tid not in instr_cache:
            instr_cache[tid] = json.load(
                (Path(root)/"data"/"tasks"/tid/"specs.json").open())["instruction"]
        return instr_cache[tid]

    def history_for(tid, si):
        return [{"code": expert[(tid, j)]["expert_code"],
                 "outcome_summary": expert[(tid, j)]["outcome_summary"]}
                for j in range(si)]

    states = sorted(expert.keys())
    print(f"expert states: {len(states)}")

    # ============ 1. expert_sft ============
    n = 0
    with (OUT / "expert_sft.jsonl").open("w") as f:
        for (tid, si) in states:
            e = expert[(tid, si)]
            user = build_state_user(instr(tid), history_for(tid, si))
            assistant = f"<code>\n{e['expert_code']}\n</code>"
            f.write(json.dumps(rec(AGENT_SYSTEM, user, assistant)) + "\n"); n += 1
    print(f"expert_sft.jsonl: {n}")

    # ============ 2. reflection_sft ============
    # v5: reflection_raw is the monologue ONLY. The <code> is attached
    # deterministically from the known-correct expert code (never rely on the
    # generator to reprint). Safety filters: drop empty, dedup DeepSeek doubling.
    n = 0; drop_empty = 0; drop_doubled = 0
    with (OUT / "reflection_sft.jsonl").open("w") as f:
        for (tid, si) in states:
            rf = reflections.get((tid, si))
            if rf is None or not rf.get("reflection_raw"):
                drop_empty += 1; continue
            monologue = rf["reflection_raw"].strip()
            # strip any stray <code>…</code> the generator may have appended
            if "<code>" in monologue:
                monologue = monologue.split("<code>")[0].strip()
            # doubling detector (DeepSeek mode-collapse, pitfalls.md): drop if head repeats
            if len(monologue) > 80 and monologue.count(monologue[:60]) >= 2:
                drop_doubled += 1; continue
            if not monologue:
                drop_empty += 1; continue
            code = expert[(tid, si)]["expert_code"].strip()
            user = build_state_user(instr(tid), history_for(tid, si))
            assistant = f"<think>\n{monologue}\n</think>\n<code>\n{code}\n</code>"
            f.write(json.dumps(rec(AGENT_SYSTEM, user, assistant)) + "\n"); n += 1
    print(f"reflection_sft.jsonl: {n}  (dropped {drop_empty} empty, {drop_doubled} doubled)")

    # ============ 3. iwm builders ============
    def iwm_record(tid, si, candidate_code, next_state_summary):
        user = build_iwm_user(instr(tid), history_for(tid, si), candidate_code)
        return rec(WM_SYSTEM, user, next_state_summary)

    # ---- 3a. iwm_full: every alt + expert transition ----
    n = 0
    with (OUT / "iwm_sft_full.jsonl").open("w") as f:
        for (tid, si) in states:
            # expert transition
            e = expert[(tid, si)]
            f.write(json.dumps(iwm_record(tid, si, e["expert_code"], e["outcome_summary"])) + "\n"); n += 1
            # all alts
            for a in alts_by_state.get((tid, si), []):
                if not a.get("summary"): continue
                f.write(json.dumps(iwm_record(tid, si, a["call"], a["summary"])) + "\n"); n += 1
    print(f"iwm_sft_full.jsonl: {n}")

    # ---- 3b. iwm_balanced: per state 7 data + 2 http + 1 py + expert ----
    n = 0; short = 0
    with (OUT / "iwm_sft_balanced.jsonl").open("w") as f:
        for (tid, si) in states:
            e = expert[(tid, si)]
            f.write(json.dumps(iwm_record(tid, si, e["expert_code"], e["outcome_summary"])) + "\n"); n += 1
            pool = [a for a in alts_by_state.get((tid, si), []) if a.get("summary")]
            data = [a for a in pool if a["bucket"].startswith("data_")]
            http = [a for a in pool if a["bucket"].startswith("http_")]
            pyer = [a for a in pool if a["bucket"].startswith("py_")]
            st_rng = random.Random(f"{tid}/{si}/{SEED}")
            picked = (st_rng.sample(data, min(7, len(data)))
                    + st_rng.sample(http, min(2, len(http)))
                    + st_rng.sample(pyer, min(1, len(pyer))))
            if len(picked) < 10:
                short += 1  # some states lack enough of a bucket; accept as-is
            for a in picked:
                f.write(json.dumps(iwm_record(tid, si, a["call"], a["summary"])) + "\n"); n += 1
    print(f"iwm_sft_balanced.jsonl: {n}  (states short of full K=10: {short})")


if __name__ == "__main__":
    main()

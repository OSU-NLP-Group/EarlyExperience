"""Build reflection_sft.v2.jsonl from sr_rollout_v2.jsonl (prompt-v2 CoTs).

Multi-turn, eval-aligned (same shape as expert_sft.v2): instruction in user-0,
"OK..." ack as assistant-0, then the prior expert Thought/Action turns (KEPT as
the original short expert thoughts, masked as context), and the FINAL assistant
turn = "Thought:\\n<reflection v2>\\n\\nAction:\\n<expert_action>".

  ** TRAINING NOTE: loss must be on the final turn only -> set `mask_history: true`
     for this dataset in LLaMA-Factory. The data file cannot encode that; it is a
     trainer-side flag. Without it, the prior expert turns get trained too. **

States are already filtered upstream (the regen script skipped post-completion and
"already" no-op states). This build only adds the post-hoc QA filter:
  - dedup doubled reflections (byte-identical and paraphrase via half/half Jaccard)
  - drop banned-vocab leaks (expert/chosen/selected/...)
  - drop meta-reference leaks ("prior trajectory", "alternatives listed")
  - drop hard outcome-narration leaks ("the resulting state", "which results in")
    [NB: "the observation says/shows" is NOT dropped — it legitimately refers to the
     current observation s_i; dropping it was ~all false positives in smoke]
  - drop numbered-alternative labels
  - normalize paragraph breaks to a single space

Run (after regen):
    python envs/scienceworld/scripts/build_reflection_sft_v2.py [--limit N] [--out PATH]
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

REPLAY = "envs/scienceworld/data/replay/replay_full.jsonl"
ROLLOUT = "envs/scienceworld/data/rollout/iwm_rollout.jsonl"
SR_V2 = "envs/scienceworld/data/rollout/sr_rollout_v2.jsonl"
CONV = "envs/scienceworld/data/conversation_start_eval.json"
OUT = "envs/scienceworld/data/sft/reflection_sft.v2.jsonl"
TD_PREFIX = "Task Description:\n"

BANNED = ("expert", "selected action", "chosen action", "correct choice",
          "right action", "best option", "optimal action", "best alternative")
META = ("prior trajectory", "alternatives listed", "the action the agent takes")
HARD_LEAK = ("the resulting state", "which results in")
NUM_LABEL = re.compile(r"\b(Action [1-9]|Alternative [1-9]|a_i\^[1-9])\b")


def dedupe_doubled(text):
    """Return (text, status): 'unchanged' | 'byte_dedup' | 'para_dedup' | 'unsafe'."""
    if len(text) < 200:
        return text, "unchanged"
    # byte-identical doubling: first 100 chars reappear, halves similar length
    fp = text[:100]
    idx = text.find(fp, 100)
    if idx != -1:
        first, second = text[:idx].rstrip(), text[idx:].rstrip()
        if abs(len(first) - len(second)) <= 50:
            return first, "byte_dedup"
        return text, "unsafe"
    # paraphrase doubling: half-vs-half 4-gram Jaccard high
    w = text.split()
    if len(w) >= 120:
        mid = len(w) // 2
        def sh(ws): return set(tuple(ws[i:i+4]) for i in range(len(ws)-3))
        a, b = sh(w[:mid]), sh(w[mid:])
        if a and b and len(a & b) / len(a | b) > 0.45:
            return " ".join(w[:mid]).rstrip(), "para_dedup"
    return text, "unchanged"


def norm(s):
    s = re.sub(r"\n+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    conv = json.load(open(CONV))
    INSTR, ACK = conv["instruction"], conv["ack"]

    traj = {}
    for line in open(REPLAY):
        rec = json.loads(line)
        if rec.get("final_done") and rec.get("final_score") == 100:
            traj[rec["item_id"]] = rec
    initial_obs = {}
    for line in open(ROLLOUT):
        r = json.loads(line)
        if r["kind"] == "initial":
            initial_obs[r["item_id"]] = r["initial_obs"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stat = {"in": 0, "err": 0, "byte": 0, "para": 0, "unsafe": 0,
            "banned": 0, "meta": 0, "hardleak": 0, "numlabel": 0, "no_traj": 0, "out": 0}
    with open(out, "w") as fo:
        for line in open(SR_V2):
            r = json.loads(line)
            stat["in"] += 1
            cot = r.get("reflection_cot_v2")
            if r.get("error") or not cot:
                stat["err"] += 1
                continue
            cot, dup = dedupe_doubled(cot)
            if dup == "unsafe":
                stat["unsafe"] += 1; continue
            if dup == "byte_dedup": stat["byte"] += 1
            if dup == "para_dedup": stat["para"] += 1
            low = cot.lower()
            if any(w in low for w in BANNED): stat["banned"] += 1; continue
            if any(w in low for w in META): stat["meta"] += 1; continue
            if any(w in low for w in HARD_LEAK): stat["hardleak"] += 1; continue
            if NUM_LABEL.search(cot): stat["numlabel"] += 1; continue
            cot = norm(cot)

            iid, si = r["item_id"], r["step"]
            t = traj.get(iid)
            if t is None or iid not in initial_obs:
                stat["no_traj"] += 1; continue
            thoughts, actions = t["agenttraj_thoughts"], t["agenttraj_actions"]
            observations = [s["observation"] for s in t["replay_steps"]]
            td = t["replay_steps"][0]["info"].get("taskDesc", "")
            if td.startswith(TD_PREFIX):
                td = td[len(TD_PREFIX):]

            msgs = [
                {"role": "user", "content": INSTR},
                {"role": "assistant", "content": ACK},
                {"role": "user", "content": f"{td.rstrip()}\n{initial_obs[iid].rstrip()}"},
            ]
            for j in range(si):
                th = thoughts[j].strip() if j < len(thoughts) and thoughts[j] else ""
                act = actions[j].strip() if j < len(actions) else ""
                asst = f"Thought:\n{th}\n\nAction:\n{act}" if th else f"Action:\n{act}"
                msgs.append({"role": "assistant", "content": asst})
                msgs.append({"role": "user", "content": observations[j].rstrip()})
            msgs.append({"role": "assistant",
                         "content": f"Thought:\n{cot}\n\nAction:\n{r['expert_action'].strip()}"})
            fo.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            stat["out"] += 1
            if args.limit and stat["out"] >= args.limit:
                break
    print(json.dumps(stat, indent=2))
    print(f"output: {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

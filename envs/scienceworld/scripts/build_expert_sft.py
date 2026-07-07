"""Build expert_sft.jsonl from the replay output.

Filter applied (TEAM_GUIDE §3, approved 2026-05-16):
  Keep only trajectories where the replay reached `final_done == True`
  AND `final_score == 100` in unmodified scienceworld 1.1.3. This drops
  ~3.9% of AgentTraj-L (mostly task-3-3 book-conductivity trajectories
  whose gold paths depend on env semantics that no longer exist).

Output schema (one line per surviving trajectory):
    {"messages": [
        {"role": "system",    "content": "<AgentGym REACT system prompt>"},
        {"role": "user",      "content": "<task_description>\\n<initial obs>"},
        {"role": "assistant", "content": "Thought:\\n<thought>\\n\\nAction:\\n<action>"},
        {"role": "user",      "content": "<env obs after the action>"},
        {"role": "assistant", "content": "Thought:\\n<thought>\\n\\nAction:\\n<action>"},
        ...
    ]}

Sources for each piece:
  - System prompt: extracted verbatim from any AgentTraj-L trajectory's
    turn 0 (the canonical AgentGym REACT system prompt).
  - Task description + initial obs: obtained from the env at runtime
    (env.load(task,var); env.getTaskDescription(); env.step("look around"))
    so they reflect the unmodified env's current canonical naming.
  - Thoughts/actions: from normalize-pass output (which was the input to
    the replay, so they're stored as `agenttraj_thoughts` /
    `agenttraj_actions` in the replay record).
  - Per-step obs: from the env replay (replay_steps[i].observation).

The AgentTraj-L "OK, I'll follow your instructions..." ack turn is
dropped (was role=gpt loss=False; not training signal). All other
content is preserved.

Run (from workspace root):
    conda run -n agentenv-sciworld --no-capture-output python \\
        envs/scienceworld/scripts/build_expert_sft.py [--workers N]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPLAY_FULL_DEFAULT = "envs/scienceworld/data/replay/replay_full.jsonl"
RAW_AGENTTRAJ_DEFAULT = "envs/scienceworld/data/raw/sciworld_train.json"
OUTPUT_DEFAULT = "envs/scienceworld/data/sft/expert_sft.jsonl"


# Worker globals — initialized once per process to avoid reloading the JVM
# for every record.
_WORKER_ENV = None
_SYSTEM_PROMPT = None


def _worker_init(system_prompt: str):
    global _WORKER_ENV, _SYSTEM_PROMPT
    from scienceworld import ScienceWorldEnv

    _WORKER_ENV = ScienceWorldEnv()
    _SYSTEM_PROMPT = system_prompt


def _format_assistant(thought: str, action: str) -> str:
    """Render an (optional thought, action) pair into ReAct assistant content.
    Matches the format AgentGym's REACT prompt asks the model to produce."""
    thought = (thought or "").strip()
    action = (action or "").strip()
    if thought:
        return f"Thought:\n{thought}\n\nAction:\n{action}"
    return f"Action:\n{action}"


def build_one(record: dict) -> dict:
    """Construct the SFT messages list for one replay record."""
    task_value = record["task_value"]
    var = record["variation_idx"]

    _WORKER_ENV.load(task_value, var)
    task_desc = _WORKER_ENV.getTaskDescription()
    initial_obs, _, _, _ = _WORKER_ENV.step("look around")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{task_desc}\n{initial_obs}"},
    ]

    actions = record["agenttraj_actions"]
    thoughts = record["agenttraj_thoughts"]
    steps = record["replay_steps"]

    for i, action in enumerate(actions):
        thought = thoughts[i] if i < len(thoughts) else ""
        messages.append(
            {"role": "assistant", "content": _format_assistant(thought, action)}
        )
        # The env's response to this action goes in as the next user turn.
        # For the final action of a successful trajectory there will still be
        # an env response; we include it. There's no following assistant turn,
        # which is fine — the trainer just won't compute loss on a hanging
        # user turn.
        if i < len(steps):
            messages.append({"role": "user", "content": steps[i]["observation"]})

    return {"messages": messages}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", default=REPLAY_FULL_DEFAULT)
    ap.add_argument(
        "--raw-agenttraj",
        default=RAW_AGENTTRAJ_DEFAULT,
        help="Only used to extract the canonical AgentGym REACT system prompt from turn 0.",
    )
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N surviving trajectories (smoke).",
    )
    args = ap.parse_args()

    replay_path = Path(args.replay).resolve()
    raw_path = Path(args.raw_agenttraj).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"replay : {replay_path}")
    print(f"raw    : {raw_path}")
    print(f"output : {out_path}")
    print(f"workers: {args.workers}")

    # Extract canonical system prompt from any trajectory's turn 0.
    print("extracting AgentGym REACT system prompt from AgentTraj-L turn 0...")
    with open(raw_path) as f:
        raw = json.load(f)
    system_prompt = raw[0]["conversations"][0]["value"]
    print(f"  system prompt length: {len(system_prompt)} chars")
    del raw

    # Stream replay_full and collect surviving records.
    print("scanning replay_full.jsonl for done && score==100 trajectories...")
    surviving = []
    total_seen = 0
    for line in open(replay_path):
        rec = json.loads(line)
        total_seen += 1
        if rec.get("final_done") and rec.get("final_score") == 100:
            surviving.append(rec)
    print(f"  {total_seen} total records, {len(surviving)} pass filter")

    if args.limit is not None:
        surviving = surviving[: args.limit]
        print(f"  limit applied: keeping {len(surviving)} for this run")

    # Build SFT messages in parallel.
    t_start = time.time()
    n_done = 0
    with open(out_path, "w") as fout, ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(system_prompt,),
    ) as pool:
        futures = {pool.submit(build_one, rec): rec["item_id"] for rec in surviving}
        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  worker failed on {iid}: {exc!r}", flush=True)
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 200 == 0 or n_done == len(surviving):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(surviving) - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done:>5}/{len(surviving)}] elapsed={elapsed:6.1f}s "
                    f"rate={rate:5.1f}/s eta={eta:6.1f}s",
                    flush=True,
                )

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.1f}s. {n_done} SFT lines written.")
    print(f"  output : {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

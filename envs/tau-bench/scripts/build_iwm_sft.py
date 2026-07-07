"""Build iwm_sft.jsonl from iwm_rollout.jsonl.

One record per (s, a, s') triple = 31,494 records (1 expert + 5 alts per obs).

Single-turn (system, user, assistant) format — same design choice as SearchQA's
IWM SFT: the IWM training target is the env's NEXT-STATE, which would be a
`tool` or `user` turn in a multi-turn chat. SFT trainers compute loss only on
`assistant` turns, so we put the next-state as the assistant content of a
single-turn record. Unambiguous for any standard chat-format trainer.

Per record:
    system   = IWM_SYSTEM (signals "predict env, not act")
    user     = rendered conversation history + the action being probed
    assistant = next_obs string (raw env response, including any Error: ... )

Action rendering in the user content uses Hermes-style tags:
    tool:    <tool_call>{"name": "X", "arguments": {...}}</tool_call>
    respond: <response>TEXT</response>

Run:
    PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output python \\
        envs/tau-bench/scripts/build_iwm_sft.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from tau_bench.envs.retail.tools import ALL_TOOLS

D_EXPERT_DEFAULT = "envs/tau-bench/data/rollout/D_expert.jsonl"
IWM_IN_DEFAULT = "envs/tau-bench/data/rollout/iwm_rollout.jsonl"
OUT_DEFAULT = "envs/tau-bench/data/sft/iwm_sft.jsonl"

# Tool schemas — same 16 retail tools as expert/reflection, embedded per record
# so the IWM model has explicit knowledge of each tool's signature when learning
# (state, action) → next_obs. Cost ~150MB over the file; worth it for stronger
# prior on response format (esp. for long-tail tools / argument-edge errors).
TOOL_SCHEMAS = [t.get_info() for t in ALL_TOOLS]

IWM_SYSTEM = (
    "You are predicting how a retail customer-service environment will respond to an agent's next action. "
    "The environment contains a customer database, an order database, a product catalog, a customer simulator, "
    "and a set of API tools. "
    "Given the conversation so far and the next action the agent is taking, output the environment's response — "
    "this may be a tool output, a tool error message, an 'Unknown action' message, or the customer's next utterance. "
    "Output the response verbatim with no commentary."
)


def render_history(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        elif role == "user":
            lines.append(f"[customer] {m.get('content','')}")
        elif role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                fn = tcs[0]["function"]
                lines.append(f"[agent tool_call] {fn['name']}({fn['arguments']})")
            else:
                lines.append(f"[agent] {m.get('content','')}")
        elif role == "tool":
            name = m.get("name") or "?"
            lines.append(f"[tool: {name}] {m.get('content','')}")
    return "\n".join(lines)


def render_action(action: dict, kind: str) -> str:
    if kind == "tool":
        # name + arguments rendered as a JSON object in a Hermes-style tag
        payload = {"name": action["name"], "arguments": action.get("arguments", {})}
        return f"<tool_call>{json.dumps(payload)}</tool_call>"
    # respond
    content = action.get("arguments", {}).get("content", "")
    return f"<response>{content}</response>"


def build_user_content(history_text: str, action_text: str) -> str:
    return (
        f"{history_text}\n\n"
        f"=== Next action being taken ===\n{action_text}\n\n"
        f"=== Environment's response (write it verbatim) ==="
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-expert", default=D_EXPERT_DEFAULT)
    ap.add_argument("--iwm", default=IWM_IN_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    # Build traj map for context reconstruction
    d_traj: dict[tuple[int, int], list] = {}
    with open(args.d_expert) as f:
        for line in f:
            r = json.loads(line)
            d_traj[(r["task_id"], r["trial"])] = r["traj"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_total, n_expert, n_alt = 0, 0, 0
    n_alt_error = 0
    n_alt_llm, n_alt_fallback = 0, 0
    n_dropped_empty = 0
    with open(args.iwm) as fin, open(args.out, "w") as fout:
        for line in fin:
            obs = json.loads(line)
            key = (obs["task_id"], obs["trial"])
            traj = d_traj.get(key)
            if traj is None:
                continue
            history = traj[: obs["obs_idx"]]
            history_text = render_history(history)

            # Expert triple — drop if next_obs is empty (e.g. think tool returns "")
            exp = obs["expert"]
            if not exp["next_obs"]:
                n_dropped_empty += 1
            else:
                exp_action_text = render_action(exp["action"], obs["expert_kind"])
                user_content = build_user_content(history_text, exp_action_text)
                fout.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": IWM_SYSTEM},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": exp["next_obs"]},
                    ],
                    "tools": TOOL_SCHEMAS,
                    "metadata": {
                        "task_id": obs["task_id"], "trial": obs["trial"], "obs_idx": obs["obs_idx"],
                        "source": "expert", "kind": obs["expert_kind"],
                    },
                }) + "\n")
                n_total += 1
                n_expert += 1

            # Alt triples — alts are always tool actions; drop empty next_obs too
            for j, alt in enumerate(obs["alts"]):
                if not alt["next_obs"]:
                    n_dropped_empty += 1
                    continue
                alt_action_text = render_action(alt["action"], "tool")
                user_content = build_user_content(history_text, alt_action_text)
                fout.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": IWM_SYSTEM},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": alt["next_obs"]},
                    ],
                    "tools": TOOL_SCHEMAS,
                    "metadata": {
                        "task_id": obs["task_id"], "trial": obs["trial"], "obs_idx": obs["obs_idx"],
                        "source": "alt", "alt_idx": j, "alt_source": alt.get("source", "llm"),
                        "kind": "tool_alt",
                    },
                }) + "\n")
                n_total += 1
                n_alt += 1
                if alt.get("source") == "llm":
                    n_alt_llm += 1
                else:
                    n_alt_fallback += 1
                if alt["next_obs"].startswith("Error:") or alt["next_obs"].startswith("Unknown action"):
                    n_alt_error += 1

    sz = Path(args.out).stat().st_size
    print(f"iwm_sft: {n_total} triples ({n_expert} expert + {n_alt} alts)")
    print(f"  alt sources : {n_alt_llm} llm + {n_alt_fallback} fallback_random")
    print(f"  alt error responses (kept as IWM signal): {n_alt_error} / {n_alt} ({n_alt_error/max(n_alt,1)*100:.1f}%)")
    print(f"  output: {args.out}  ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

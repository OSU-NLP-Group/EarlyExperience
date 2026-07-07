"""
Shared helpers for build_*_sft.py.

Two output formats per category (paper §B.3 Open item #1 → dual emit):

  *_text.jsonl  : BFCL prompting-mode style — tool calls rendered as Python-syntax
                  text strings; tool results rendered as user messages with a
                  [tool_result] tag. Paper-faithful; works with any instruct base.

  *_fc.jsonl    : OpenAI-compatible tool-calls format — assistant messages carry
                  `tool_calls: [{type:"function", function:{name, arguments}}]`;
                  tool results carry role:tool + tool_call_id. Requires FC-capable
                  base model.

Both share the same source records and same user-content prefix (system + tools
+ history); only the message role/content shape for prior actions differs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "envs/bfcl_v4/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
FUNC_DOC = DATA / "multi_turn_func_doc"

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

# Per-category system prompt templates. Each SFT category has a distinct assistant
# target format, and the system prompt must match that format — otherwise training
# contradicts itself (e.g. system saying "output function calls" while the target
# is a natural-language next-state description).
#
# Tool schemas are injected via `{functions}` in each template.

# expert_sft: assistant emits function calls (or a brief natural-language wrap-up
# after the turn's work is done, matching Opus FC's actual behavior at inference).
# Adapted from BFCL `_DEFAULT_SYSTEM_PROMPT` template with the strict
# "only return function calls" clause softened to allow the turn-terminating text
# summary (which is what Opus actually emits at turn-end in the source trajectories).
SYSTEM_PROMPT_EXPERT_TEMPLATE = (
    "You are an expert function-calling agent in a multi-turn tool-use environment. "
    "Based on the user's requests across turns, you make one or more function/tool "
    "calls to accomplish each turn's goal. When the current turn's work is done, "
    "you may emit a brief natural-language summary of what was accomplished before "
    "the next turn begins.\n\n"
    "If none of the available functions can be used for a request, say so. If the "
    "user's request lacks parameters required by a function, ask for the missing "
    "information rather than inventing values.\n\n"
    "At each turn, complete the tasks requested by the user within the current "
    "turn. Continue calling functions until you have fulfilled the user's request "
    "to the best of your ability. Once you are done with the current turn, the "
    "system considers it complete and proceeds to the next turn.\n\n"
    "Here is a list of functions in JSON format that you can invoke:\n{functions}\n"
)

# iwm_sft: assistant is a WORLD MODEL — given history and a proposed action,
# describe what the environment would return. Do NOT emit the action itself.
SYSTEM_PROMPT_IWM_TEMPLATE = (
    "You are a world model for a multi-turn function-calling environment. You are "
    "given the conversation history so far and a proposed action (marked in the "
    "user's last message with the prefix `[probe action]`).\n\n"
    "Your job is to predict, in ONE concise sentence, what the environment would "
    "do or return if that action were executed at this point. Describe the "
    "outcome factually — the state change, the data returned, or the error "
    "message.\n\n"
    "STRICT rules for your response:\n"
    "  - Output ONLY the one-sentence outcome description. No preamble, no "
    "explanation, no quoting the action.\n"
    "  - Do NOT emit the action yourself — you are predicting its effect, not "
    "choosing to execute it.\n"
    "  - Do NOT output function-call syntax. No `name(args)`, no `tool_calls` "
    "structures.\n"
    "  - Do NOT mention the user or the broader task — just describe the "
    "environment's response.\n\n"
    "Here is the tool set defined in this environment (schemas provided so you "
    "can predict each tool's behavior faithfully):\n{functions}\n"
)

# reflection_sft: assistant reasons through the situation in a Thought paragraph,
# then emits the Action. Format is symmetric for text (Action: <call>) and FC
# (tool_calls structured); the phrasing here works for both.
SYSTEM_PROMPT_REFL_TEMPLATE = (
    "You are an expert function-calling agent in a multi-turn tool-use "
    "environment. Before each action, you reason through the situation in a "
    "single Thought paragraph, then emit the chosen action.\n\n"
    "Your response format:\n"
    "  Thought:\n"
    "  <one paragraph of internal reasoning about the situation, the state, "
    "and why the action you are about to take is fitting>\n\n"
    "  Action:\n"
    "  <the function call(s) to invoke — either as Python-syntax call strings "
    "or, if your interface supports it, as structured tool_calls>\n\n"
    "The Thought paragraph should sound like your own first-person deliberation. "
    "The Action should be a valid call from the available tool set below.\n\n"
    "Tools available:\n{functions}\n"
)


def _build_prompt(template: str, tool_schemas: list[dict]) -> str:
    return template.format(functions=json.dumps(tool_schemas, indent=2))


# BACK-COMPAT: any code still calling `SYSTEM_PROMPT_TEMPLATE` or the old
# `build_system_prompt(schemas)` gets the expert template (previous behavior).
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_EXPERT_TEMPLATE


def load_tool_schemas(involved_classes: list[str]) -> list[dict]:
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


def build_system_prompt(tool_schemas: list[dict]) -> str:
    """DEPRECATED: back-compat wrapper. Returns the expert-category prompt.
    Prefer `build_expert_system_prompt` / `build_iwm_system_prompt` /
    `build_reflection_system_prompt` explicitly."""
    return _build_prompt(SYSTEM_PROMPT_EXPERT_TEMPLATE, tool_schemas)


def build_expert_system_prompt(tool_schemas: list[dict]) -> str:
    """System prompt for expert_sft: agent that outputs function calls
    (+ optional turn-terminating text summary)."""
    return _build_prompt(SYSTEM_PROMPT_EXPERT_TEMPLATE, tool_schemas)


def build_iwm_system_prompt(tool_schemas: list[dict]) -> str:
    """System prompt for iwm_sft: world model that predicts one-sentence
    outcome of a [probe action]. Explicitly forbids emitting function
    calls in the response."""
    return _build_prompt(SYSTEM_PROMPT_IWM_TEMPLATE, tool_schemas)


def build_reflection_system_prompt(tool_schemas: list[dict]) -> str:
    """System prompt for reflection_sft: agent that emits Thought paragraph
    followed by Action (call). Works for both text and FC output formats."""
    return _build_prompt(SYSTEM_PROMPT_REFL_TEMPLATE, tool_schemas)


def parse_call_to_name_args(call_str: str) -> tuple[str | None, dict]:
    """Parse a Python-syntax call like `mv(source='a', destination='b/')` into
    (name, {arg: value}). Used for emitting FC-format tool_calls."""
    m = re.match(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s*\((.*)\)\s*$", call_str.strip())
    if not m:
        return None, {}
    name = m.group(1)
    rest = m.group(2)
    args = {}
    # naive kwarg parser — handles strings/numbers/bools/None; lists captured as text
    pattern = re.compile(
        r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*"
        r"('[^']*'|\"[^\"]*\"|\[[^\]]*\]|-?\d+(?:\.\d+)?|True|False|None|"
        r"[A-Za-z_][A-Za-z_0-9]*)"
    )
    for m2 in pattern.finditer(rest):
        k, raw = m2.group(1), m2.group(2)
        if raw.startswith(("'", '"')):
            v: Any = raw[1:-1]
        elif raw.startswith("["):
            # try to parse as list (best-effort)
            try:
                v = json.loads(raw.replace("'", '"'))
            except json.JSONDecodeError:
                v = raw
        elif raw in ("True", "False"):
            v = (raw == "True")
        elif raw == "None":
            v = None
        else:
            try:
                v = int(raw)
            except ValueError:
                try:
                    v = float(raw)
                except ValueError:
                    v = raw
        args[k] = v
    return name, args


def calls_to_text_block(call_list: list[str]) -> str:
    """Render a list of call strings as a single BFCL-style text emit.
    Single call: `cd(folder='X')`. Parallel calls: `[cd(...), ls()]`."""
    if len(call_list) == 1:
        return call_list[0]
    return "[" + ", ".join(call_list) + "]"


def calls_to_fc_tool_calls(call_list: list[str], turn_idx: int, step_idx: int) -> list[dict]:
    """Render call list as OpenAI-compatible `tool_calls` array."""
    out = []
    for j, c in enumerate(call_list):
        name, args = parse_call_to_name_args(c)
        if not name:
            continue
        out.append({
            "id": f"call_t{turn_idx}_s{step_idx}_{j}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        })
    return out


def render_history_messages_text(prior_steps: list[dict],
                                  current_user_msg: str | None,
                                  use_summaries_for_tool_results: bool = True) -> list[dict]:
    """Build the conversation history as message list for TEXT format.

    Tool results become `role:user` messages with a [tool_result] tag prefix
    (BFCL prompting mode merges these into user-side context).
    """
    msgs = []
    for r in prior_steps:
        if r.get("user_msg_for_turn"):
            msgs.append({"role": "user", "content": r["user_msg_for_turn"]})
        if r.get("step_type", "function_call") == "function_call":
            calls = r["expert_emit_decoded"]
            msgs.append({"role": "assistant", "content": calls_to_text_block(calls)})
            # Tool results: summary if available, else raw. Field name varies by source file:
            # rollout/iwm_full_summarized.jsonl uses `expert_tool_responses_recorded`;
            # parsed/opus_expert_steps.jsonl uses bare `tool_responses_recorded`.
            sums = r.get("expert_summaries") or [None] * len(calls)
            raws = (r.get("expert_tool_responses_recorded")
                    or r.get("tool_responses_recorded")
                    or [None] * len(calls))
            parts = []
            for s, raw in zip(sums, raws):
                if use_summaries_for_tool_results and s:
                    parts.append(s)
                elif raw:
                    parts.append(str(raw))
            if parts:
                msgs.append({"role": "user",
                             "content": "[tool_result] " + " | ".join(parts)})
        elif r.get("step_type") == "text_only":
            t = r["expert_emit_raw"]
            if isinstance(t, list) and t and isinstance(t[0], str):
                t = t[0]
            msgs.append({"role": "assistant", "content": str(t)})
    if current_user_msg:
        msgs.append({"role": "user", "content": current_user_msg})
    return msgs


def render_history_messages_fc(prior_steps: list[dict],
                                current_user_msg: str | None) -> list[dict]:
    """Build conversation history as messages for FC format.

    Prior assistant calls become structured `tool_calls`; tool results become
    `role:tool` messages bound to tool_call_id.
    """
    msgs = []
    for r in prior_steps:
        if r.get("user_msg_for_turn"):
            msgs.append({"role": "user", "content": r["user_msg_for_turn"]})
        if r.get("step_type", "function_call") == "function_call":
            calls = r["expert_emit_decoded"]
            tool_calls = calls_to_fc_tool_calls(calls, r["turn_idx"], r["step_idx"])
            msgs.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            # Summary preferred for tool content; falls back to raw tool response.
            sums = r.get("expert_summaries") or [None] * len(tool_calls)
            raws = (r.get("expert_tool_responses_recorded")
                    or r.get("tool_responses_recorded")
                    or [None] * len(tool_calls))
            for tc, s, raw in zip(tool_calls, sums, raws):
                content = s if s else (str(raw) if raw else "")
                msgs.append({"role": "tool",
                             "tool_call_id": tc["id"],
                             "content": content})
        elif r.get("step_type") == "text_only":
            t = r["expert_emit_raw"]
            if isinstance(t, list) and t and isinstance(t[0], str):
                t = t[0]
            msgs.append({"role": "assistant", "content": str(t)})
    if current_user_msg:
        msgs.append({"role": "user", "content": current_user_msg})
    return msgs


def merge_consecutive_user_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive user-role messages into one. Other consecutive same-role
    pairs (e.g. tool→tool from parallel function calls) are left untouched — those
    are valid per OpenAI tool-use spec (one tool message per tool_call_id).

    CRITICAL: must NOT mutate any of the input dicts. Callers in build_*_sft.py
    iterate a loop where the same `base_msgs` list (with shared dict references)
    gets re-fed into this function across iterations; if we mutate a shared dict
    here, content accumulates across loop iterations.

    Necessary because render_history renders tool results as user-role messages
    (for prompting-mode formats), which can collide with the next user message
    (next turn's user_msg, or an injected [probe action] line for IWM).
    """
    out: list[dict] = []
    for msg in messages:
        if (out and out[-1].get("role") == "user" and msg.get("role") == "user"):
            # mutate a fresh copy of the previous output entry; leaves caller's input untouched
            prev_copy = dict(out[-1])
            prev_content = prev_copy.get("content") or ""
            curr_content = msg.get("content") or ""
            sep = "\n\n" if prev_content and curr_content else ""
            prev_copy["content"] = (prev_content + sep + curr_content).strip()
            out[-1] = prev_copy
        else:
            # always shallow-copy so we never expose a mutable reference into caller's data
            out.append(dict(msg))
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def load_expert_ids() -> set[str]:
    p = REPO / "envs/bfcl_v4/data/split/expert_ids.json"
    return set(json.loads(p.read_text()))


def load_parsed_steps_by_case() -> dict[str, list[dict]]:
    from collections import defaultdict
    p = REPO / "envs/bfcl_v4/data/parsed/opus_expert_steps.jsonl"
    out: dict[str, list[dict]] = defaultdict(list)
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["case_id"]].append(r)
    for cid in out:
        out[cid].sort(key=lambda r: r["global_emit_idx"])
    return out

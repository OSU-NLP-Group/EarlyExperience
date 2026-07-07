"""Normalize AgentTraj-L action / thought text so the resulting trajectories
execute cleanly in unmodified scienceworld 1.1.3+.

Why this script exists
----------------------
AgentTraj-L was generated against a pre-public version of scienceworld whose
naming conventions differ from any PyPI-published version. Three classes of
mismatch cause 31.8% of trajectories to fail to replay (analysis recorded in
NOTES.md):

  (A) "green house" (two words) -> "greenhouse" (one word). 90.9% of failures.
  (B) "connect <non-wire-object> terminal 1 to <wire> terminal 2"
        ->  "connect <non-wire-object> to <wire> terminal 2".
      scienceworld grammar simplification: only wires/electrical components
      retain explicit terminal indices on the source side now. 7.9% of
      failures (book / glass-jar-as-electrical-component cases).
  (C) Plain typos in AgentTraj-L: "cachew" -> "cashew", "sandwhich" -> "sandwich".
      1.2% of failures. These are AgentTraj-L data bugs, not env changes.

We deliberately fix this in the DATA, not in the env, so that downstream users
who train on our SFT files can run the resulting model against an unmodified
AgentGym + scienceworld stack and have it just work.

Scope of edits
--------------
Only `gpt`-role turns in `conversations` get rewritten. The `human` turns
(AgentTraj-L's recorded environment observations) are left untouched so they
remain a forensic record of what the env looked like when AgentTraj-L was
generated.

Run
---
    conda run -n agentenv-sciworld --no-capture-output python \\
        envs/scienceworld/scripts/normalize_agenttraj.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Rule A: room rename
def _rewrite_green_house(s: str) -> tuple[str, int]:
    new = s.replace("green house", "greenhouse")
    return new, (1 if new != s else 0)


# Rule B: drop redundant "terminal 1" on the source side of connect actions
# when the source is not a wire / electrical-component with real terminals.
# Source is "wire-y" if it literally contains the word "wire" — in this env
# only wires use the "<source> terminal N" form for their endpoints; light
# bulbs and batteries use "anode in X" / "cathode in X" instead.
_CONNECT_RE = re.compile(r"\bconnect (.+?) terminal 1 to ", flags=re.IGNORECASE)


def _rewrite_terminal_1(s: str) -> tuple[str, int]:
    hits = [0]

    def repl(m):
        source = m.group(1)
        if "wire" in source.lower():
            return m.group(0)  # legitimate wire terminal, keep
        hits[0] += 1
        return f"connect {source} to "

    new = _CONNECT_RE.sub(repl, s)
    return new, hits[0]


# Rule C: AgentTraj-L's own typos
_TYPO_MAP = [
    ("cachew", "cashew"),
    ("sandwhich", "sandwich"),
]


def _rewrite_typos(s: str) -> tuple[str, int]:
    new = s
    n = 0
    for bad, good in _TYPO_MAP:
        if bad in new:
            n += new.count(bad)
            new = new.replace(bad, good)
    return new, n


def normalize_text(s: str) -> tuple[str, dict[str, int]]:
    stats = {}
    s, stats["A_green_house"] = _rewrite_green_house(s)
    s, stats["B_terminal_1"] = _rewrite_terminal_1(s)
    s, stats["C_typos"] = _rewrite_typos(s)
    return s, stats


def normalize_trajectory(traj: dict) -> tuple[dict, dict[str, int]]:
    """Return a deep-modified copy of `traj` with all gpt-turn `value`s
    normalized. Human-turn `value`s are left untouched."""
    convs = traj["conversations"]
    out_convs = []
    total_stats = Counter()
    for msg in convs:
        if msg.get("from") == "gpt" and isinstance(msg.get("value"), str):
            new_val, stats = normalize_text(msg["value"])
            for k, v in stats.items():
                total_stats[k] += v
            new_msg = dict(msg)
            new_msg["value"] = new_val
            out_convs.append(new_msg)
        else:
            out_convs.append(msg)
    out_traj = dict(traj)
    out_traj["conversations"] = out_convs
    return out_traj, total_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        default="envs/scienceworld/data/raw/sciworld_train.json",
    )
    ap.add_argument(
        "--output",
        default="envs/scienceworld/data/normalized/sciworld_train.json",
    )
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"input : {inp}")
    print(f"output: {out}")
    with open(inp) as f:
        data = json.load(f)
    print(f"trajectories: {len(data)}")
    print()

    rule_totals = Counter()
    trajs_touched_by_rule = Counter()
    n_trajs_touched = 0
    out_data = []
    for traj in data:
        new_traj, stats = normalize_trajectory(traj)
        out_data.append(new_traj)
        for k, v in stats.items():
            rule_totals[k] += v
            if v > 0:
                trajs_touched_by_rule[k] += 1
        if sum(stats.values()) > 0:
            n_trajs_touched += 1

    with open(out, "w") as f:
        json.dump(out_data, f, ensure_ascii=False)

    print("=== normalization stats ===")
    print(
        f"  trajectories touched (any rule)        : {n_trajs_touched:>5}  "
        f"({100*n_trajs_touched/len(data):.1f}%)"
    )
    for rule in ("A_green_house", "B_terminal_1", "C_typos"):
        print(
            f"  rule {rule:<14}: applied {rule_totals[rule]:>6} times, "
            f"on {trajs_touched_by_rule[rule]:>5} trajectories"
        )
    print()
    print(f"wrote {len(out_data)} trajectories to {out}")
    print(f"output size: {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

# Env-specific NOTES.md — Template

`envs/<env>/NOTES.md` records the decisions, artifacts, and gotchas particular to one environment. It is the single source of truth for anything that is *not* generic across envs (which lives in `METHOD.md` / `method_recap.md` / `pitfalls.md` instead).

Fill each section only if it applies to your env; delete the rest. The goal is orientation for the next reader, not exhaustive documentation.

---

# `<Env Name>`

> **Status**: recon / smoke / full-scale done, and any open decisions still pending.

## Env at a glance

- What kind of env: text-based / HTTP-service / in-process / JVM-backed / ...
- Action space shape: enumerable / structured-and-large / open-ended (Python / natural-language / ...)
- State representation: short structured text / long HTML / multi-modal / ...
- One-shot vs multi-step scoring, and what "done" means.

## Upstream repo & version pin

- Upstream URL:
- Fork URL (if applicable):
- Pinned commit / branch:
- Install path (conda env, setup command, any prerequisites):

## Expert data source (`D_expert`)

Where the expert trajectories come from — the env's own gold path, a released dataset, or LLM-generated (last resort). Include the URL / dataset name and the exact filter applied to arrive at the trajectories actually used.

## Modification strategy

Only what changed *relative to upstream*. High-level rationale, not code details. If nothing changed upstream, say so.

## Method mapping — `s_i`, `a_i^j`, `s_i^j`

- How `s_i` is serialized (must match the env's native evaluator input format).
- How alternative actions are drawn (enumerable admissible list / LLM-proposed / hybrid).
- How next state `s_i^j` is captured (raw observation / summarized / rendered from a template).
- Any deviations from the paper's default (e.g. K=0 for open action spaces).

## Key EE hyperparameters

- **K for IWM**:
- **K for SR**:
- **Alt-sampling strategy** (IWM):
- **Alt-sampling strategy** (SR):
- **Reflection length target**:

## Approved decisions

Any decision that was made through the team's approval protocol — filter rules, deployment checkpoints, deviations from paper. Format: what was decided, when, and why.

## Pipeline (in execution order)

Short listing of the actual scripts / stages, in the order they must run. Not a manual — a map so the next person knows where to look.

## Data output

Paths (or Drive URLs) for the final SFT files:

- `expert_sft*.jsonl` — N records, X MB
- `iwm_sft*.jsonl` — N records, X MB
- `reflection_sft*.jsonl` — N records, X MB

Plus intermediate artifacts, if kept.

## Things easy to get wrong here

Env-specific gotchas that don't generalize to other envs. Anything that took time to debug the first time and would waste time again next time. Cross-link to `pitfalls.md` if a gotcha is generalizable enough that it should live there instead.

## Open decisions / TODO

Anything still open that a new reader needs to know is unresolved.

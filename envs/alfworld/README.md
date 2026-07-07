# ALFWorld

Text-based household task env, paper §B.1. The agent is given a natural-language task (`Your task is to: put a clean fork in cabinet`) and steps a TextWorld-simulated home by emitting admissible commands (`go to countertop 1`, `pick up fork 1`, ...).

## Upstream

- [`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent) — a research training framework that ships a batteries-included ALFWorld env in `agent_system/environments/env_package/alfworld/`.

## Modification strategy

The upstream verl-agent focuses on **RL training loops** — it wraps the ALFWorld TextWorld backend and provides gym-style step/reset/reward. We keep that wrapper untouched and layer an **offline data-extraction pipeline** on top of it. Two conceptual pieces added:

1. **Expert harvester** — walks each task's gold walkthrough (which ALFWorld's TextWorld backend can produce natively for every game) and records `(state_i, expert_action_i)` as expert SFT records. No LLM needed; ALFWorld's own gold path is the expert.
2. **Alternative-action prober** — at each expert state, enumerates the env's admissible-commands list (which the gym exposes at every step) and steps *each non-expert admissible action* on a re-cloned env replayed to that state, capturing `(state_i, alternative_j, next_state)` triples for IWM.
3. **Reflection builder** — for each expert state, prompts an LLM (see *Reproducibility notes*) to produce a first-person reasoning trace that arrives at the expert action after considering the alternatives; used as SR data.

## Method mapping (EE ↔ ALFWorld)

| EE concept | ALFWorld realization |
|---|---|
| Expert action | one entry from admissible-commands at each state along the gold walkthrough |
| Alternative-action pool | the *rest* of admissible-commands at that state (fully enumerable — no LLM proposer needed) |
| Next state | textual observation returned by TextWorld after executing the alternative |
| Reflection target | expert action + LLM-authored monologue over the alt outcomes |

The **key insight for ALFWorld** is that the action space is already enumerable, so IWM never needs an LLM to invent alternatives — it just re-steps every alt admissible command from a re-synchronized env.

## Data output

Living on Google Drive (this repo does **not** vendor any SFT files):

```
ggdrive:Early-Experience-Reproduce/data/alfworld/
├── expert_sft.jsonl
├── iwm_sft.jsonl
└── reflection_sft.jsonl
```

## Reproducibility notes

- **Trajectory source**: ALFWorld's built-in TextWorld gold walkthrough for each training game.
- **K for IWM**: all admissible alternatives at each state (typically 5–15). No sampling — exhaustive.
- **K for SR**: 3 alternatives per state, chosen randomly from admissible.
- **CoT preservation**: expert records carry `<think>...</think><action>...</action>` markup.
- **Careful gotcha**: naïve batched alt probing across parallel envs can silently drift each alternative onto a *different game's* state. Any re-implementation must verify state re-synchronization (e.g. by comparing pre-step observations across all K rollouts) before trusting IWM data.

For a full reproduction, follow the modification strategy above against a fresh clone of upstream.

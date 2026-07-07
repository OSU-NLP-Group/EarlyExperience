---
name: early-experience-data
description: |
  Use this skill whenever the user asks to generate, collect, inspect, or
  prepare early-experience training data (Implicit World Modeling or
  Self-Reflection, in the sense of arXiv:2510.08558) for any environment
  under `envs/<env>/`. Trigger phrases include "rollout", "collect expert
  trajectories", "generate reflection", "smoke test the pipeline", "set up
  a new env for early experience", or any work that produces JSONL files
  under `envs/<env>/data/`. Do NOT trigger for model training, evaluation,
  or any work outside `envs/<env>/`.
---

# Early Experience Data Generation

This skill governs how data generation for the early-experience paradigm is done across all environments in this workspace. Read this entire file before starting work in `envs/<env>/`.

## What this workspace is for

The workspace produces SFT-ready training data for the two methods defined in `docs/paper.pdf` and recapped in `docs/METHOD.md`:

- **Implicit World Modeling (IWM)** — train the policy to predict the next state given the current state and an action.
- **Self-Reflection (SR)** — train the policy to produce a chain-of-thought reasoning over expert vs alternative actions, then the expert action.

The output of this workspace is JSONL files under `envs/<env>/data/sft/`. **Training itself is out of scope.** If a task requires running SFT, evaluating checkpoints, or tuning training hyperparameters, stop and escalate.

## Required reading order

Before any work in `envs/<env>/`, read in this order:

1. `docs/METHOD.md` — what IWM and SR actually are, and the reflection prompt template. This is the source of truth for the method.
2. `docs/TEAM_GUIDE.md` — global rules: LLM choice (DeepSeek V4 Pro, thinking disabled), API usage, the pre-call gate, the filter approval protocol, the deployment checkpoint, version control, submodule policy.
3. `.claude/skills/early-experience-data/method_recap.md` — short list of decisions where the right answer is non-obvious. Use as a runbook, not as a tutorial.
4. `.claude/skills/early-experience-data/pitfalls.md` — accumulated gotchas from previous envs. Skim before each new env in case something carries over.
5. `envs/<env>/NOTES.md` if it exists — env-specific decisions and orientation already recorded.

If any of these files is missing or empty, surface it as a question rather than guessing.

**On conflicts between layers.** For workspace-wide mechanics (LLM choice, gates, version control, submodule policy, output layout shape), `TEAM_GUIDE.md` and this skill win. For **env-specific** content (state representation, K, sampling strategy, prompt extensions, filtering rationale, anything tied to one env's interface or paper section), the env's `NOTES.md` is closer to truth than the generic guidance here. **But never silently resolve a conflict.** If you notice that NOTES.md says one thing and SKILL.md / TEAM_GUIDE.md / METHOD.md says another, stop and surface the conflict to the user before proceeding — the usual outcome is that NOTES.md needs a correction, and the user will edit it.

## Hard rules (non-negotiable)

These override any user request short of an explicit override.

1. **No LLM API call without confirmation.** Every batch — smoke or full, 5 samples or 50k — must go through the pre-call confirmation gate defined in `TEAM_GUIDE.md` §2. State what we're doing, how much data, and approximate token volume. Wait for explicit go-ahead. Silence is not consent.

2. **Default to not editing inside `envs/<env>/<upstream-name>/`.** Prefer external wrappers or config overrides from outside the submodule. If touching submodule internals is genuinely the simpler path, explain the reason to the user, get approval, and make the change. Mechanics live in `TEAM_GUIDE.md` §7.

3. **Env runs end-to-end before pipeline code is written.** For any new env: install per its README, run its own tests or a minimal `reset → step → done` loop, confirm it works. Pipeline code (rollout, reflection generation, SFT writers) does not get written until this gate passes. The point is to surface install/dependency/runtime issues before they get tangled with your own code.

4. **Smoke before scale.** No full-scale rollout or reflection generation before a smoke run (CC chooses the size) has produced data that passes hand inspection of at least 5 samples.

5. **Filter rules go through the approval protocol.** Default position for any new env is no filters. Any filter applied to data in `data/sft/` must go through the protocol in `TEAM_GUIDE.md` §3 — propose, quantify impact, wait for user approval, record in NOTES.md, then apply.

6. **Full-scale runs go through the deployment checkpoint.** Before transitioning from smoke to full scale on any env, run the deployment checkpoint in `TEAM_GUIDE.md` §5. Single message covering all the per-env decisions and projected costs; wait for explicit user go-ahead.

7. **Final SFT output layout is fixed in shape, not in count.** See "Output layout" below. The three SFT **categories** (expert / IWM / reflection), the filename prefix per category, the directory location, and the outer JSONL schema are not negotiable. Default is one file per category; produce more only when the env's standard pipeline (as captured in NOTES.md) prescribes multiple variants. This workspace targets the paper's standard pipeline per env, not ablations.

8. **No training code.** This workspace does not contain SFT scripts, RL loops, or evaluation harnesses. If asked to write training code, stop and escalate.

9. **SR reflection length is a per-env decision, confirmed with the user before any LLM batch.** Unconstrained, modern LLMs (DeepSeek V4 Pro and peers) emit ~500–1000 word reflections, with the longest cases also being the most pathological (model wrestling with itself, drifting to a different action). Workspace default: target 200–400 words with a soft cap around **500 words**, enforced **only through prompt instruction** — not via a `max_tokens` API limit. Hard token caps truncate mid-sentence and produce worse training data than overlong-but-complete reflections; rely on the prompt to nudge length, and accept the occasional outlier. Each env must surface its own length choice in `envs/<env>/NOTES.md` and have the user explicitly confirm it — sometimes a short-task env wants tighter (e.g. 200 target); sometimes a long-horizon complex-reasoning env wants more headroom.

10. **The prediction stage (IWM) must be format-separated from the imitation stages (expert / SR).** IWM trains the model to *predict a next state*; expert and SR train it to *produce an action*. These are opposite behaviors and must be distinguishable by the model itself — otherwise a checkpoint that inherits IWM weights (the paper's two-stage IWM continues imitation from the IWM stage, and any joint mix trains them together) learns to emit next-state / observation text where an action is expected. Give IWM its **own system prompt** (framed as world-modeling / next-state prediction, not "you are an agent who acts") **and a delimited next-state target** (e.g. `Observation:\n<state>`); keep the agent system prompt and the action-output format (`Thought:/Action:` or the env's equivalent) for expert and SR. Never let the prediction stage and the action stages share one system prompt with an undelimited target. See Output layout.

11. **The SFT message format must match the env's native evaluator's input structure.** Training and evaluation are out of scope, but the SFT *message shape* is our output, and a train/eval format mismatch degrades the trained model no matter how clean the data is. Before finalizing an env's SFT files, read how the env's native eval harness builds the policy's input — multi-turn vs single-turn, whether the task instruction sits in a `system` message or the first `user` turn, any fixed acknowledgement turns (e.g. an "OK, I'll follow…" turn), the exact instruction text, and how the state/task string is rendered — and match it in `expert_sft` and `reflection_sft` (the categories the policy is actually run in at eval). Record the eval-format reference (which file/function builds the input) in `envs/<env>/NOTES.md`.

## Output layout

Only the final SFT outputs have a fixed layout. Everything else is the env's choice.

### Required final outputs

`envs/<env>/data/sft/` must contain files in **three categories** — expert, IWM, reflection — all in OpenAI chat-messages JSONL format:

    {"messages": [{"role": "...", "content": "..."}, ...]}

Default is one file per category, using the canonical names below. Some envs' standard pipeline (as captured in `envs/<env>/NOTES.md`) prescribes more than one variant in a category — typically when the paper itself specifies different settings for different policy model sizes. In those cases produce each variant as its own file with a descriptive suffix and list them in NOTES.md.

- **`expert_sft.jsonl`** — imitation learning baseline.
  user content = state representation;
  assistant content = expert action.

- **`iwm_sft.jsonl`** — implicit world modeling stage.
  user content = state representation + the chosen action (expert or alternative);
  assistant content = next state (raw or summarized, env's choice).

- **`reflection_sft.jsonl`** — self-reflection stage.
  user content = state representation;
  assistant content = reflection chain-of-thought, followed by the expert action.

The string format inside each `content` field — how a "state" is serialized, how an "action" is encoded — is the env's choice. The serialization must be **consistent across all three categories for the same env**: a given `s_i` is rendered the same way in expert, IWM, and reflection files. Document the choice in `envs/<env>/NOTES.md`.

**Each file is single-purpose within its category.** Do not mix reflection samples into an expert file, or expert-only samples into a reflection file. Cross-category mixing at training time is the trainer's job.

**The prediction stage must be format-distinct from the imitation stages (hard rule 10).** Rendering `s_i` consistently does *not* make the three files interchangeable in shape. `iwm_sft` is a next-state **prediction** task and must carry a **system prompt** and an **assistant-target surface** that mark it as such — world-model framing plus a delimited next-state (e.g. `Observation:\n…`). `expert_sft` and `reflection_sft` are **action** tasks under the agent system prompt, with the action-output format. Concretely: if `iwm_sft` reuses the agent system prompt ("you are an agent, respond with an action") and puts the next state in a bare, undelimited assistant turn, then a policy that continues from IWM weights is being taught that "after `Action: X` comes a run of observation text" — and at inference it emits observation-style text instead of stopping after its action. The env-specific `s_i` serialization is still shared; the *system prompt and the target format* are what separate prediction from action.

### Intermediate artifacts

Anything that is not a final SFT file goes somewhere under `envs/<env>/data/` — but the subdirectory layout, file names, and formats are the env's choice. Pick whatever makes the env easy to work with and easy to regenerate from.

## Where you are when entering `envs/<env>/`

Use the directory's current state to figure out where you are:

- **NOTES.md absent** → recon mode. Read the upstream repo, get the env to run end-to-end (hard rule 3), draft NOTES.md, stop for user review. Do not write pipeline code yet.
- **NOTES.md present, no scripts/data** → implementation mode. Write the rollout / reflection / SFT-building code. Run a smoke. Stop for user review of smoke outputs.
- **NOTES.md present, smoke data exists, full data does not** → either iterate on smoke findings or run the deployment checkpoint and go to full scale. Discuss with the user before deciding.
- **NOTES.md present, full SFT data exists** → finishing mode. Verify against "What 'done' looks like" below.

## What "done" looks like for an env

A finished env has:

- A populated `envs/<env>/NOTES.md` documenting all env-specific decisions, the upstream repo and its pinned commit, and where intermediate artifacts live.
- A working isolated Python environment, set up however the upstream's install path requires, with the actual setup commands recorded in NOTES.md so the env can be reproduced.
- Submodule(s) under `envs/<env>/<upstream-name>/` pinned to known commits.
- Scripts in `envs/<env>/` for collecting expert trajectories, rolling out alternative actions, generating reflections, and a smoke test. Names and structure are the env's choice.
- `envs/<env>/data/sft/` containing at least one file in each of the three required categories (expert / IWM / reflection) at the agreed full scale, with all variants enumerated in NOTES.md.
- If anything env-specific was learned that other envs might hit, append it to `pitfalls.md`.

## When something is ambiguous

Default to **stopping and asking** rather than guessing. The cost of a clarification message is far below the cost of a wrong rollout batch or a wasted day building the wrong setup. Cases that always warrant asking:

- State representation (raw vs summary, what fields to include).
- K and sampling strategy — propose what you think makes sense, but the final value is the user's decision.
- Whether the env's reflection prompt needs env-specific extensions to the template in METHOD.md.
- Anything involving cost — token volume, GPU time, storage.
- Anything that requires editing inside a submodule.
- Upstream URL, fork URL, or pinned commit when setting up a new env. The user provides these — never fetch from GitHub by guessing or invent a URL.

Filters are not in this list because they have their own protocol (hard rule 5 → TEAM_GUIDE §3). Full-scale runs are not in this list because they have their own gate (hard rule 6 → TEAM_GUIDE §5).

## File map quick reference

```
docs/
├── paper.pdf                    original paper
├── METHOD.md                    method definitions (IWM, SR, formulas, prompt template)
└── TEAM_GUIDE.md                global rules (LLM, API, submodules, gates, protocols)

.claude/skills/early-experience-data/
├── SKILL.md                     this file (router + output format)
├── method_recap.md              short list of decisions easy to get wrong
└── pitfalls.md                  accumulated gotchas from previous envs

envs/<env>/
├── NOTES.md                     env-specific decisions, upstream pins, intermediate-artifact map
├── <upstream-name>/             git submodule (default: don't edit; see TEAM_GUIDE §7)
├── (env's own code + whatever the install path requires)
└── data/
    ├── sft/                     final training-ready JSONL — three categories, ≥1 file each
    │   ├── expert_sft*.jsonl
    │   ├── iwm_sft*.jsonl
    │   └── reflection_sft*.jsonl
    └── (intermediate artifacts, layout is env's choice)
```
---
name: early-experience-data
description: |
  Use this skill whenever the user asks to generate, collect, inspect, or
  prepare early-experience training data (Implicit World Modeling or
  Self-Reflection, in the sense of arXiv:2510.08558) for an agent
  environment. Trigger phrases include "rollout", "collect expert
  trajectories", "generate reflection", "smoke test the pipeline",
  "set up a new env for early experience", or any work that produces
  expert / IWM / reflection SFT JSONL files. Do NOT trigger for model
  training, evaluation, or unrelated work.
---

# Early Experience Data Generation

This skill governs how data generation for the early-experience paradigm is done. Read this entire file before starting work on a new env.

## What this skill is for

This skill produces SFT-ready training data for the two methods defined in the paper (https://arxiv.org/abs/2510.08558) and recapped in `METHOD.md`:

- **Implicit World Modeling (IWM)** — train the policy to predict the next state given the current state and an action.
- **Self-Reflection (SR)** — train the policy to produce a chain-of-thought reasoning over expert vs alternative actions, then the expert action.

The output of the workflow is JSONL files in three categories: expert, IWM, reflection. Where you put them and how you organize the surrounding pipeline scripts is up to your project — the skill is agnostic about layout. **Training itself is out of scope.** If a task requires running SFT, evaluating checkpoints, or tuning training hyperparameters, that belongs elsewhere — surface it to the user rather than acting on it inside this skill.

## Required reading order

Before starting a new env, read in this order:

1. `METHOD.md` — what IWM and SR actually are, and the reflection prompt template. This is the source of truth for the method.
2. `method_recap.md` — short list of decisions where the right answer is non-obvious. Use as a runbook, not as a tutorial.
3. `pitfalls.md` — accumulated gotchas from previous envs. Skim before each new env in case something carries over.
4. This env's own `NOTES.md` (if you or a previous agent already started one) — env-specific decisions already recorded. `NOTES_TEMPLATE.md` is the skeleton to copy when you're starting fresh.

If any of these files is missing or empty (aside from `NOTES.md` on a fresh env), surface it as a question rather than guessing.

**On conflicts between layers.** For skill-wide mechanics (gates, output shape), this skill wins. For **env-specific** content (state representation, K, sampling strategy, prompt extensions, filtering rationale, anything tied to one env's interface or paper section), the env's `NOTES.md` is closer to truth than the generic guidance here. **But never silently resolve a conflict.** If `NOTES.md` says one thing and `SKILL.md` / `METHOD.md` says another, stop and surface the conflict to the user before proceeding — the usual outcome is that `NOTES.md` needs a correction, and the user will edit it.

## Step 0: Ask the user which generator model to use

**Before writing any pipeline code or making any LLM API call, ask the user which model to use for the generation stages of this pipeline** — alternative-action proposal (when the action space isn't fully enumerable), state/observation summarization (when the raw next-state text is too noisy), and self-reflection CoT generation.

Explain the choice to the user:

- **The paper's method uses the base policy model itself** as the generator — the same model that will be trained. This is what "self" in Self-Reflection literally refers to: the model reflects on its own alternative actions using its own capacity. The paper's core claim is that base-model exploration and self-reflection alone can improve over training on expert trajectories, **without any external stronger model**. This matters especially in environments without verifiable rewards where RL is impractical.
- **Another model — stronger, or just different — is also a valid choice.** The pipeline works with any capable instruction-following LLM. Use a stronger one if you care more about data quality than paper-faithful reproduction, or a smaller / faster one if cost is the constraint.

Neither is more "correct" — the choice is up to the user. Record the decision in the env's `NOTES.md` under a "Model choice for data generation" line, then use that model consistently for all three generation stages in this env. If later stages want to switch, surface the change to the user.

## Hard rules (non-negotiable)

These override any user request short of an explicit override.

1. **No LLM API call without user confirmation.** Every batch — smoke or full, 5 samples or 50k — must be preceded by a message stating what will run, how much data, and approximate token / cost volume. Wait for explicit go-ahead. Silence is not consent.

2. **Default to not editing inside the upstream env repo.** Prefer external wrappers or config overrides from outside it. If touching upstream internals is genuinely the simpler path, explain the reason to the user, get approval, then make the change.

3. **Env runs end-to-end before pipeline code is written.** For any new env: install per its README, run its own tests or a minimal `reset → step → done` loop, confirm it works. Pipeline code (rollout, reflection generation, SFT writers) does not get written until this gate passes. The point is to surface install / dependency / runtime issues before they get tangled with your own code.

4. **Smoke before scale.** No full-scale rollout or reflection generation before a smoke run has produced data that passes hand inspection of at least 5 samples.

5. **Filter rules require explicit user approval.** Default position for any new env is no filters. Any filter applied to SFT data must first be proposed to the user with rationale and quantified impact (how many samples it drops, what kind); wait for approval, record the rule and its rationale in the env's `NOTES.md`, then apply.

6. **Before scaling from smoke to full, summarize and confirm.** Send a single message covering all the per-env decisions locked in during smoke (K, sampling regime, prompt structure, filter rules if any, projected token / time / cost). Wait for explicit user go-ahead before launching the full run.

7. **Final SFT output shape is fixed.** See "Output shape" below. The three SFT **categories** (expert / IWM / reflection), the filename prefix per category, and the outer JSONL schema are not negotiable. Default is one file per category; produce more only when the env's pipeline (as recorded in its `NOTES.md`) prescribes multiple variants.

8. **This skill's scope stops at SFT-ready files.** If a task requires running SFT, evaluating checkpoints, or tuning training hyperparameters, that is out of scope for the skill — surface it to the user rather than acting on it silently.

9. **SR reflection length is a per-env decision, confirmed with the user before any LLM batch.** Unconstrained, modern LLMs emit ~500–1000 word reflections, with the longest cases also being the most pathological (model wrestling with itself, drifting to a different action). Recommended default: target 200–400 words with a soft cap around **500 words**, enforced **only through prompt instruction** — not via a `max_tokens` API limit. Hard token caps truncate mid-sentence and produce worse training data than overlong-but-complete reflections; rely on the prompt to nudge length, and accept the occasional outlier. Each env must surface its own length choice in the env's `NOTES.md` and have the user explicitly confirm it — sometimes a short-task env wants tighter (e.g. 200 target); sometimes a long-horizon complex-reasoning env wants more headroom.

    **Note for smaller base policy models:** consider shortening the SR target further. The action tokens the model must actually emit at inference are a small fraction of any single reflection example, so long reflections dilute the imitation signal in the loss. Smaller models have less capacity to spare on background reasoning — a tighter reflection concentrates supervision on the action itself and often improves the downstream gain.

10. **The prediction stage (IWM) must be format-separated from the imitation stages (expert / SR).** IWM trains the model to *predict a next state*; expert and SR train it to *produce an action*. These are opposite behaviors and must be distinguishable by the model itself — otherwise a checkpoint that inherits IWM weights (the paper's two-stage IWM continues imitation from the IWM stage, and any joint mix trains them together) learns to emit next-state / observation text where an action is expected. Give IWM its **own system prompt** (framed as world-modeling / next-state prediction, not "you are an agent who acts") **and a delimited next-state target** (e.g. `Observation:\n<state>`); keep the agent system prompt and the action-output format (`Thought:/Action:` or the env's equivalent) for expert and SR. Never let the prediction stage and the action stages share one system prompt with an undelimited target. See Output layout.

11. **The SFT message format must match the env's native evaluator's input structure.** Training and evaluation are out of scope, but the SFT *message shape* is this skill's output, and a train/eval format mismatch degrades the trained model no matter how clean the data is. Before finalizing an env's SFT files, read how the env's native eval harness builds the policy's input — multi-turn vs single-turn, whether the task instruction sits in a `system` message or the first `user` turn, any fixed acknowledgement turns (e.g. an "OK, I'll follow…" turn), the exact instruction text, and how the state/task string is rendered — and match it in `expert_sft` and `reflection_sft` (the categories the policy is actually run in at eval). Record the eval-format reference (which file/function builds the input) in the env's `NOTES.md`.

## Output shape

Only the final SFT outputs have a fixed shape. Everything else — directory layout, intermediate artifact naming, pipeline code structure — is the project's choice.

### Required final outputs

Each env produces files in **three categories** — expert, IWM, reflection — all in OpenAI chat-messages JSONL format:

    {"messages": [{"role": "...", "content": "..."}, ...]}

Default is one file per category, using the canonical names below. Some envs' pipeline (as captured in the env's `NOTES.md`) prescribes more than one variant in a category — typically when the paper itself specifies different settings for different policy model sizes. In those cases produce each variant as its own file with a descriptive suffix and list them in the `NOTES.md`.

- **`expert_sft.jsonl`** — imitation learning baseline.
  user content = state representation;
  assistant content = expert action.

- **`iwm_sft.jsonl`** — implicit world modeling stage.
  user content = state representation + the chosen action (expert or alternative);
  assistant content = next state (raw or summarized, env's choice).

- **`reflection_sft.jsonl`** — self-reflection stage.
  user content = state representation;
  assistant content = reflection chain-of-thought, followed by the expert action.

The string format inside each `content` field — how a "state" is serialized, how an "action" is encoded — is the env's choice. The serialization must be **consistent across all three categories for the same env**: a given `s_i` is rendered the same way in expert, IWM, and reflection files. Document the choice in the env's `NOTES.md`.

**Each file is single-purpose within its category.** Do not mix reflection samples into an expert file, or expert-only samples into a reflection file. Cross-category mixing at training time is the trainer's job.

**The prediction stage must be format-distinct from the imitation stages (hard rule 10).** Rendering `s_i` consistently does *not* make the three files interchangeable in shape. `iwm_sft` is a next-state **prediction** task and must carry a **system prompt** and an **assistant-target surface** that mark it as such — world-model framing plus a delimited next-state (e.g. `Observation:\n…`). `expert_sft` and `reflection_sft` are **action** tasks under the agent system prompt, with the action-output format. Concretely: if `iwm_sft` reuses the agent system prompt ("you are an agent, respond with an action") and puts the next state in a bare, undelimited assistant turn, then a policy that continues from IWM weights is being taught that "after `Action: X` comes a run of observation text" — and at inference it emits observation-style text instead of stopping after its action. The env-specific `s_i` serialization is still shared; the *system prompt and the target format* are what separate prediction from action.

### Intermediate artifacts

Where you keep intermediate artifacts (raw rollouts, per-state alternative-action pools, summarizer outputs) and how you name them is up to the project. Pick whatever makes the env easy to work with and easy to regenerate from.

## Where you are when starting on an env

Use the env's current state to figure out where you are:

- **`NOTES.md` absent** → orientation stage. Read the upstream repo, get the env to run end-to-end (hard rule 3), draft `NOTES.md` (start from `NOTES_TEMPLATE.md`), stop for user review. Do not write pipeline code yet.
- **`NOTES.md` present, no scripts / data** → implementation stage. Write the rollout / reflection / SFT-building code. Run a smoke. Stop for user review of smoke outputs.
- **`NOTES.md` present, smoke data exists, full data does not** → either iterate on smoke findings or, if the smoke passes review, run the scale-up gate (hard rule 6) and go to full scale. Discuss with the user before deciding.
- **`NOTES.md` present, full SFT data exists** → finishing mode. Verify against "What 'done' looks like" below.

## What "done" looks like for an env

A finished env has:

- A populated `NOTES.md` documenting all env-specific decisions, the upstream repo and its pinned commit, and where intermediate artifacts live.
- A working isolated Python environment (or whatever the upstream's install path requires) with the actual setup commands recorded in `NOTES.md` so the env can be reproduced.
- The upstream env repo pinned to a known commit (submodule, vendored, or lockfile — whatever the project uses).
- Scripts for collecting expert trajectories, rolling out alternative actions, generating reflections, and a smoke test. Names and structure are the project's choice.
- SFT output files in the three required categories (expert / IWM / reflection) at the agreed full scale, with all variants enumerated in `NOTES.md`.
- If anything env-specific was learned that other envs might hit, append it to `pitfalls.md`.

## When something is ambiguous

Default to **stopping and asking** rather than guessing. The cost of a clarification message is far below the cost of a wrong rollout batch or a wasted day building the wrong setup. Cases that always warrant asking:

- State representation (raw vs summary, what fields to include).
- K and sampling strategy — propose what you think makes sense, but the final value is the user's decision.
- Whether the env's reflection prompt needs env-specific extensions to the template in `METHOD.md`.
- Anything involving cost — token volume, GPU time, storage.
- Anything that requires editing inside the upstream env repo.
- Upstream URL, pinned commit, or setup instructions when starting a new env. The user provides these — never fetch from GitHub by guessing or invent a URL.

Filters and full-scale launches are not in this list because they have their own hard rules (5 and 6 above).

## File map

Files in this skill:

```
skill/
├── SKILL.md                     this file (router + output shape)
├── METHOD.md                    method definitions (IWM, SR, formulas, prompt template)
├── method_recap.md              short list of decisions easy to get wrong
├── pitfalls.md                  accumulated gotchas from previous envs
├── NOTES_TEMPLATE.md            skeleton to copy when you start a new env's NOTES.md
└── paper.pdf                    original paper
```

Files this skill expects the project to produce per env:

- `NOTES.md` — env-specific decisions, upstream pins, intermediate-artifact map. Copy from `NOTES_TEMPLATE.md`.
- Three SFT files: `expert_sft.jsonl`, `iwm_sft.jsonl`, `reflection_sft.jsonl` (see "Output shape" above).
- Pipeline scripts and intermediate artifacts — the project decides how to organize these.
# Pitfalls — Accumulated Gotchas

This file collects env-specific surprises that turned out to apply, or
might apply, beyond the env where they were first hit. CC and the user
both append here.

Skim this file before starting a new env — if a previous env hit
something that sounds related, the new env may hit it too.

## When an entry is worth making

Use judgment. The point of this file is that the next env benefits from your scar tissue — that's the only test. Mid-work and finish-time additions are both fine. The lists below describe the kinds of things that have historically been worth writing down, and the kinds that turned out not to be; use them as prompts, not as rules.

Tends to be worth recording:

- An env behavior that contradicted what the README / paper / NOTES.md
  said it would do.
- A subtle data-quality bug that only showed up at scale.
- A concurrency / resource / state-management failure mode that took
  real time to diagnose.
- Something CC initially got wrong because it pattern-matched from
  another env, and the user had to catch it.

Usually not worth recording here:

- Generic engineering hygiene already covered by SKILL.md or
  TEAM_GUIDE.md.
- One-time mistakes with no general lesson.
- Documented behaviors of an env that anyone reading the env's README
  would already know.

The bar for an entry: would a CC starting on a *different* env benefit
from seeing this? If no, it goes in `envs/<env>/NOTES.md` instead.

## Entry format

Each entry is one section. Keep it short — one or two paragraphs.

```
### <short title — descriptive, scannable>

**Env where first hit**: <env name>
**Date**: <YYYY-MM-DD>

**What happened**: One or two sentences on the observed problem.

**Root cause**: One or two sentences on why it happened.

**How to avoid**: A specific, actionable check or change. If a NOTES.md
template should be updated to include this check by default, say so.

**Likely also affects**: List of envs that share the relevant
characteristic and may hit the same issue.
```

## Entries

### Upstream agent libraries silently un-pin their env packages

**Env where first hit**: scienceworld
**Date**: 2026-05-16

**What happened**: AgentGym's `agentenv-sciworld/pyproject.toml` listed its env dependency as just `"scienceworld"` (no version constraint). `pip install` pulled the latest published version (1.2.3 at the time of recon). The AgentTraj-L dataset bundled with AgentGym was generated against a much earlier, unpublished build of the env. The version drift between the two showed up as 31.8% of expert trajectories failing to replay end-to-end — actions named "open door to green house" (two words) were rejected because the env now spells the room "greenhouse" (one word). Other less-frequent drifts: book → book-with-title for circuit grammar, `cachew` vs `cashew` for object names.

**Root cause**: An agent library that's older than its underlying env, with no version pin to bridge the gap. The dataset and the env share a name but not a version.

**How to avoid**: Whenever the upstream agent library doesn't pin its env package version explicitly, **pin it yourself** in the fork before running anything at scale. If no PyPI version still aligns with the dataset (which is what we found — the older versions had a different API entirely), commit to a data-side `normalize_<env>.py` preprocessor that does string-level rewrites instead of trying to chase versions. NOTES.md template for new envs should include a "Upstream env package version pinned to X (rationale: ...)" line.

**Likely also affects**: any env where the agent framework and the env are maintained as separate packages — ALFWorld + alfworld-pip, WebShop + webshop-server, BabyAI + minigrid, etc.

### "Expert trajectory" datasets are sometimes gold-path-with-LLM-CoT, not agent rollouts

**Env where first hit**: scienceworld
**Date**: 2026-05-16

**What happened**: Treating AgentTraj-L's SciWorld split as "agent rollouts" led to the wrong design choices on K, sampling strategy, and CoT-consistency reasoning. Direct comparison showed AgentTraj-L's action sequences are byte-identical to `env.get_gold_action_sequence()` output (with some non-determinism in tie-breaking), and the `Thought:` text was clearly written by an LLM looking at the gold-path action — not produced by an agent during rollout. Also: AgentTraj-L's `item_id sciworld_N` maps **directly** to the env's `data_idx = N` in the games-list enumeration order (which we'd nearly missed and almost rebuilt with text matching).

**Root cause**: Dataset README didn't distinguish between gold paths, replayed gold paths, and agent rollouts. Local inspection of a handful of trajectories caught it.

**How to avoid**: At recon time for any env's expert dataset, sanity-check by extracting 2-3 trajectories' action sequences and comparing them byte-for-byte against `env.get_gold_action_sequence()` (or whatever the env's equivalent is). If they match, the dataset is gold-path-based and the (s_i, a_i) pairs are "optimal expert" not "agent rollouts" — this changes K-sampling design considerations and may obviate filter rules paper §B.6-style "drop trajectories that don't reach score=100" (gold paths basically always do, modulo env-version drift).

**Likely also affects**: any env in AgentGym (BabyAI, TextCraft) and any other framework that calls its bundled trajectories "expert demonstrations" without distinguishing source.

### JVM- or process-backed envs have no native state save/restore

**Env where first hit**: scienceworld
**Date**: 2026-05-16

**What happened**: For IWM / SR rollout we need K=3 alternative-action probes at every expert state s_i, then continue from s_{i+1}. The natural approach assumes the env supports "snapshot at s_i / restore after probing", but `ScienceWorldEnv` only exposes `load(task, var)` which resets to the variation's initial state. There is no native save/restore. We worked around it with `env.load + replay expert_actions[0..i-1]` to restore s_i, but each rebuild is O(i) env steps, so a trajectory of N steps with K probes per state costs roughly K·N²/2 env operations.

**Root cause**: ScienceWorld's `getRunHistory()` is for inspection only (returns the action log), and the env's internal Java state cannot be serialized back to disk. Multiple separate `ScienceWorldEnv()` instances each spawn their own JVM, so there is no in-process way to "branch" the env either.

**How to avoid**: At recon time, probe whether the env has `save_state` / `load_state` / `__deepcopy__` / `clone` / `fork` methods. If it has none, budget for the O(K·N²) rebuild cost up front — use process-pool parallelism and don't be surprised when a "60k-step probing run" actually executes ~1M env operations.

**Likely also affects**: any JVM-backed env (TextWorld via py4j, the older Jericho IF gamebase, any env that calls out to a binary JAR), any Docker-isolated env (WebArena, WebShop's HTTP service), and any env whose state lives outside the Python process.

### LLM reflection generators leak supervision labels into the CoT by default

**Env where first hit**: scienceworld
**Date**: 2026-05-17

**What happened**: The first version of the SR reflection prompt followed METHOD.md §4.3 closely — input slots labeled "Expert Action (a_i)" and "Alternative Actions: 1. ... 2. ...". The reflection LLM (DeepSeek V4 Pro) echoed those labels into its monologue ~16% of the time ("the expert action is X, which makes sense because...", "Action 1 would be less helpful because..."). This breaks the entire point of the SR CoT: at inference the trained model has no privileged "expert" label, and the CoT must be the model's *own* reasoning that arrives at the chosen action. Training on text that announces the expert label corrupts the supervision signal.

**Root cause**: LLMs default to "essay justifying a known answer" mode when handed a labeled answer in the input — this is the default training distribution. The model has no inherent reason to suppress labels just because the next-token target says it should.

**How to avoid**: Two prompt-design rules, both must be applied:
1. **Remove the word "expert" / "selected" / "chosen" / "correct" / "right" / "best" / "optimal" from the input slot names**. Renaming "Expert Action" → "The action the agent takes here" drops leakage to near-zero. Add a banned-vocab list to the Guidelines block as a safety net.
2. **Don't refer to alternatives by numbered labels** ("Action 1", "Action 2", "a_i^1") in the input either; if you must, instruct the model to use natural inline phrasing in the output ("I could try X", "Another option I have is to Y"). At inference the model doesn't see a pre-enumerated list — it considers options in its own head, and the training target must read that way.

Also add a post-hoc detector that grep's for the banned vocabulary as a last line of defense; expect a small residual rate (~0.3% in our SR run) that should be dropped.

**Likely also affects**: every env's SR pipeline. The leakage mode is LLM-class-wide; we observed it with DeepSeek V4 Pro, and the same default behavior is documented for GPT-4 / Claude family in similar setups.

### Long-form LLM generations occasionally duplicate the whole response

**Env where first hit**: scienceworld
**Date**: 2026-05-17

**What happened**: Across 39,700 SR reflections, ~2.9% (1,170) consisted of the same monologue written twice in a row — byte-identical first half, then a second identical copy starting immediately after the first (no separator, just `.I` joined). 100% of cases were exactly 2x repeats, never 3x or more. `frequency_penalty=0.3` reduced this rate from a higher baseline but didn't eliminate it; the failure mode is at the document level (the model fails to emit EOS after a complete answer and conditional on its own complete output decides "writing a similar answer" is the highest-probability continuation), and token-level penalties don't reach far enough to suppress it.

**Root cause**: LLM mode-collapse / EOS-failure at the end of long-form generation. Increasing `frequency_penalty` further (0.5–1.0) damages natural noun-repetition in the body of the text; raising sampling threshold compromises diversity.

**How to avoid**: Don't fight this in the prompt. Detect post-hoc: for each generation, check whether the first ~100 characters appear again later in the text with a similar-length second copy following — and dedup by keeping only the first copy. In our run this recovered 98.5% of doubled records (1,153 / 1,170) as clean complete reflections; the remaining 1.5% (length mismatch > 50 chars between halves) was safer to drop than auto-dedup.

**Likely also applies**: every env's SR or LLM-CoT-generation step. The duplicate-doubling mode has been observed across LLM families and is essentially independent of env.

### `build_*_sft.py` shared-dict mutation across loop iterations

**Env where first hit**: bfcl_v4
**Date**: 2026-06-02

**What happened**: In `iwm_sft_text.jsonl` (all 8,677 records), the `[tool_result]` user message at one position in the messages list contained 6–15 verbatim copies of the same content (the current turn's user message and prior `[probe action]` lines). Role alternation audit (`user→user`, `assistant→assistant`) reported 0% bad — the duplication is invisible at that level. Only a content-pattern check (any 60-char head appearing ≥3× in a single message) surfaced it, and only because an independent reviewer flagged a specific record (`recs[4506]`). Field-level identifiers / token counts in the data file looked normal; the bytes just included repeats of one sentence.

**Root cause**: A merge-helper (`merge_consecutive_user_messages`) mutated `prev["content"]` in place where `prev` was the same `dict` reference that lived inside the caller's `base_msgs` list. `build_iwm_sft.py` builds `base_text_msgs` once per state and then loops over (expert calls + K=10 alts) emitting one SFT record per iteration with `base_text_msgs + [user_msg_i, target_msg_i]`. After iteration `i`, `base_text_msgs[-1]["content"]` had been silently extended; iteration `i+1` saw the already-extended content and extended again. Content accumulated linearly across iterations.

**How to avoid**: Any helper that "merges" or otherwise post-processes a messages list MUST not mutate dict objects belonging to the caller. The minimal pattern: shallow-copy each dict as it enters the output list (`out.append(dict(msg))`) and shallow-copy again before writing into an already-output entry (`out[-1] = dict(prev_copy_with_changes)`). When auditing SFT output, run BOTH a role-alternation check AND a content-pattern check (e.g., does any 60-char substring of a message appear ≥3× within that same message). The role check alone is insufficient — accumulated duplicates pass it.

**Likely also affects**: every env's `build_*_sft.py` step that (a) builds a shared message-history prefix once per state and (b) loops over multiple per-state items emitting one SFT record per iteration. SciW IWM and Tau-Bench IWM both fit this pattern (multiple alts per state). Add the content-pattern audit to any new env's `build_*_sft.py` validation by default.

### A live env "expert/oracle" can be far worse than the env's own precomputed gold paths

**Env where first hit**: alfworld
**Date**: 2026-06-05

**What happened**: The first rollout drove the expert via ALFWorld's *live* handcoded `AlfredExpert` (the env's `config_tw` `expert_type`). It failed to solve ~21% of games (won rate 79%: wrong receptacle, picked unrelated objects, even emitted `help`), and even on solved games it wandered (avg 11.8 steps vs the gold 3.5). The user caught it by asking to read a failing trajectory and to verify the expert source.

**Root cause**: ALFWorld's "handcoded" expert is a brittle rule-based heuristic (receptacle search by string-match + `random.choice`), chosen for DAgger real-time *speed* over reliability — a documented limitation (alfworld issue #87; the paper's DAgger numbers ~61% seen corroborate). The correct expert source is the **offline-precomputed gold `walkthrough` field inside each `game.tw-pddl`** — byte-identical to the original verl-agent expert data, and what the paper's 21,031 pairs came from.

**How to avoid**: When an env ships BOTH a live expert/oracle AND precomputed gold solutions, check which the original/paper data used and prefer the precomputed gold path. Sanity-check the expert solve rate (gold ≈ 100%) and trajectory length (gold paths are short) before building anything. A 79% solve rate or long wandering trajectories is the tell.

**Likely also affects**: any verl-agent / embodied env with a "handcoded" vs "planner" expert split; any env whose game files embed a `walkthrough` / `policy_commands` / gold-plan distinct from a runtime solver (ALFWorld, TextWorld-based envs).

### SR reflections leak future info when the expert action changes location / opens a container

**Env where first hit**: alfworld
**Date**: 2026-06-05

**What happened**: SR grounds the reflection in the expert action's outcome `s_{i+1}`. For *location-changing* actions (`go to X`, `open Y`), `s_{i+1}` reveals the destination's contents — which the agent does NOT know at decision time. ~25% of navigation reflections (first smoke) justified the move with "indeed I found bowl 1 there" / "now I see cup 1 and cup 2" — training the model to hallucinate seeing a place before going there.

**Root cause**: `s_{i+1}` of a move/open is *post-decision* information; the reflector LLM naturally writes it as already-observed. In-place actions (take/move/use — target visible in the current obs) don't have this problem.

**How to avoid**: System-prompt rule: justify location-changing / container-opening moves with **anticipatory** language ("going there *should* let me find …"), never as already-seen; only the CURRENT observation is fact. Post-hoc detector should fire ONLY on `go to`/`open` records and must ignore "expect to see…" phrasing — otherwise it's ~98% false positives (in-place "now I'm here, I can see X" is legitimate; "my best move"/"first choice" are natural language, not label leaks). This cut the leak from ~25% to 0.23%; residual re-run (attempt-2) then dropped.

**Likely also affects**: every embodied/navigation env's SR — ScienceWorld navigation, WebShop page transitions, BabyAI, any env where an action moves the agent and `s_{i+1}` reveals new state the agent couldn't see when choosing.

### Regex "leak" scans and verbatim checks are proxies — they don't judge data quality

**Env where first hit**: appworld
**Date**: 2026-07-01

**What happened**: SR smoke printed 20/20 "clean" (0 banned-vocab hits, 20/20 verbatim code match) after a prompt fix. CC treated the numbers as proof the data was good. The user pushed back: "your scan only catches problems you imagined ahead of time — you never actually READ the reflections." True. The regex missed the actual quality dimensions that matter for training: whether the CoT reasoning is sound rather than generic parroting, whether it correctly interprets each alt's outcome as evidence, whether it doesn't invent facts, whether it converges on the committed snippet for the *right reason*. A pipeline that passes automated scans and fails on read-through is worse than one that visibly fails — training on 10k such records is money burned.

**Root cause**: Regex scans are dimensionality-collapsing. They flag one pattern of leakage (label vocabulary, forbidden phrases, code-not-matching), which the model can trivially route around without becoming more truthful. Real SR quality lives in *content* — coherence, faithfulness to inputs, non-genericity — none of which regex can see.

**How to avoid**: For every SR / reflection / summary batch, the smoke report is not complete until CC has **read at least 5-10 sampled records in full** and written a short paragraph of judgment ("this one reasons well from alt X's error", "this one just restates the outcome without reasoning", "this one invents an entity not in the input"). Automated scans are fine as *pre-filters* to catch known failure modes at scale — never as the sole quality signal. `method_recap.md` already says "look at the data you generated"; this pitfall is the reminder that "look at" means human reading, not regex scans. Add to smoke script templates: after auto stats, print sampled records for review — but treat CC's read-through as the gating step, not the stats.

**Likely also affects**: every env's SR / reflection / summary phase; every phase that produces free-form LLM text as training target. Especially dangerous when the automated checks look "clean" (0 hits) — that is exactly when quality could still be terrible in ways CC never imagined.
# ScienceWorld

## Project-side characterization

Closed-finite action space at every step. The underlying `scienceworld` Python package can enumerate the **valid (action-template, objects) combinations** at any state — paper §B.6 treats this enumeration as the admissible action list, and the expert's action is one item from it.

AgentGym's stock HTTP server does **not** expose that flat list. Its only related endpoint, `/action_hint`, returns `{"possible_actions", "possible_objects"}` — i.e. the templates and the objects as two separate lists. To recover the flat admissible list we patched the inner fork to expose a new `/admissible_actions` endpoint backed by ScienceWorld's `get_valid_action_object_combinations()` (snake_case API; the camelCase variant is deprecated in `scienceworld 1.2.3`). At the `boil`/var-0 reset state this endpoint returns 417 ready-to-execute action strings.

This env runs through the **AgentGym framework**, which serves ScienceWorld as an HTTP service. AgentGym is a sibling submodule that ships environment adapters for several envs; we only use the SciWorld slice here. Originally `WooooDyy/AgentGym`, now forked into `UlyssesXC/AgentGym` to support the `/admissible_actions` patch (see Upstream below).

State is short structured text (procedural lab readouts). IWM next-state prediction targets the raw next state, not a summary.

Expert trajectories come from AgentGym's **AgentTraj-L** dataset on HuggingFace (`AgentGym/AgentTraj-L`), SciWorld split `sciworld_train.json`. The paper §B.6 refers to this as "AGENTTRAJ" — same data, different label. **2120 trajectories, but actual SA-pair count is 42,178 (≈ 20 SA pairs per trajectory) — not the 14,506 the paper cites.** The discrepancy is unresolved; possibilities include the paper using a filtered subset (e.g. only successful trajectories, or only the first N pairs per trajectory) or a different dataset version. To be revisited before the deployment checkpoint.

Conversation format is ShareGPT-style: each trajectory is `{conversations: [...], item_id: "sciworld_N"}` with no explicit `task_name`/`variationIdx` label — the task must be recovered from the initial human turn's text. 95.2% of gpt turns use `Thought:\n<reasoning>\n\nAction:\n<action>` (the other 4.8% is the per-trajectory leading "OK..." acknowledgement turn). Action canonicalization (e.g. `drop` vs `put down`, `go to` vs `move`, `wait1` for single-step wait) matches AgentGym's `function_to_name` table in `agentenv/agentenv/envs/sciworld.py` — confirmed against expert turns.

## Scope and limits

- **Dataset scale**: 42,178 expert (state, action) pairs across 2120 trajectories from AgentTraj-L's `sciworld_train.json`. The paper §B.6 cites 14,506; discrepancy unresolved (see "Project-side characterization" above).
- **K for IWM**: 3. Sample 3 non-expert actions uniformly from the admissible list (excluding the expert), then include the expert action — each expert state contributes 4 triples to the IWM data. The "admissible list" comes from our patched `/admissible_actions` endpoint. [paper §B.6]
- **K for SR**: 3 by default. Drop to K=2 when the policy model is Llama-3.1-8B-Instruct. [paper §B.6]
- **SR alternative sampling**: policy proposes alternatives at temperature 1.0, **not** uniform-random from admissible. Canonicalize and dedup. **If a proposed action is not in the admissible list, discard it and refill from random admissible (unused entries only).** [paper §B.6]
- **One-shot example**: AgentGym's ScienceWorld prompt interface does **not** supply the §B.6 one-shot — its `conversation_start_dict` in `agentenv/agentenv/envs/sciworld.py` only contains the action menu + ReAct format instructions. We must add the §B.6 one-shot ourselves in our wrapper, used identically for SR data construction and (downstream) evaluation. The exact one-shot content is in the paper.

## Rough guide

The work is straightforward by the standards of this workspace. Note that **IWM and SR use different alternative-sampling pipelines** — don't conflate them:

1. **IWM data**: at each expert state, draw 3 non-expert actions uniformly from `admissible \ {expert}`. Step the env on each, plus on the expert action. Record the four (state, action, next_state) triples.
2. **SR data**: at each expert state, ask the policy (with the one-shot example) to propose 3 (or 2) alternative actions at T=1.0. Canonicalize, dedup, drop any not in admissible, refill from random admissible. Step the env on each. Pass everything through the reflection prompt from `METHOD.md` §4.

The HTTP-service architecture means env rollout is RPC-based, not in-process. Plan concurrency around that — see the corresponding note below.

## Things easy to get wrong here

- **Using policy-proposed alternatives for IWM, or admissible-random for SR.** The two pipelines use different alternative-sampling methods on purpose. [paper §B.6: behavior]

- **Skipping the admissible-list fallback in SR.** A proposed action not in admissible must be **replaced** by a random admissible draw, not just dropped. Otherwise K varies per state and edge states with all-invalid proposals contribute nothing. [paper §B.6]

- **Re-querying admissible on every step.** The admissible list is state-dependent (ScienceWorld's `get_valid_action_object_combinations()` is computed against the current world model state). Don't cache it on the client side across a trajectory.

- **Action string canonicalization.** AgentGym's client adapter [`function_to_name` in `agentenv/agentenv/envs/sciworld.py`] hardcodes specific surface forms (e.g. `drop` not `put down`, `wait1` for single-step wait). Spot-checked against expert turns in `sciworld_train.json` and they match. Our IWM/SR action strings must use the same forms so expert and rollout entries share a vocabulary.

- **Server concurrency.** AgentGym serves ScienceWorld as an HTTP service. Verify during smoke that real throughput scales with client concurrency. If not, run multiple server instances on different ports.

## Recon findings (resolved from code reading; smoke will retest)

- **CoT in AgentTraj-L sciworld_train.json**: **yes**, confirmed. 95.2% of gpt turns are `Thought:\n...\n\nAction:\n...` (the other 4.8% is each trajectory's leading "OK..." acknowledgement). Expert assistant content carries CoT — our `expert_sft.jsonl` should preserve it so `reflection_sft.jsonl` (which also ends with expert action + reasoning) stays comparable.
- **§B.6 one-shot in AgentGym prompt**: **no, not supplied**. We add it ourselves in the wrapper. See "Scope and limits" above.
- **Admissible-action endpoint**: `/action_hint` returns templates + objects separately, **not** a flat list. We patched the inner fork to add `GET /admissible_actions?id=` backed by snake_case `get_valid_action_object_combinations()`. Live at `UlyssesXC/AgentGym@scienceworld-ee` (commit `07b30b5`). Verified end-to-end: at `boil`/var-0 reset state the endpoint returns 417 entries.
- **Server concurrency**: default is a single uvicorn process; one JVM-backed ScienceWorld instance per `env_id`. Steps don't take a Python-side lock, so multi-id step calls can proceed concurrently within one process, modulo the shared JVM. For real parallelism, run multiple `sciworld` servers on different ports and have the client round-robin. Measure during smoke.

## Setup (reproducible commands)

```bash
# env server
conda create -n agentenv-sciworld python=3.8 -y
conda run -n agentenv-sciworld --no-capture-output python -m pip install -e \
    envs/scienceworld/agentgym/AgentGym/agentenv-sciworld/
# launch (chosen port 36010 — same as upstream BC scripts)
conda run -n agentenv-sciworld --no-capture-output sciworld --host 127.0.0.1 --port 36010
```

Installed versions: `scienceworld 1.2.3` (Java backend, runs on system OpenJDK 11), `agentenv_sciworld 0.0.2` (our patched version), `fastapi 0.124.4`, `uvicorn 0.33.0`, `py4j 0.10.9.9`. Java 1.8+ required; OpenJDK 11 works.

The pipeline-side Python env (DeepSeek SDK, rollout client, reflection generation) is **not** created yet; will be set up during implementation.

## Data

### Raw input
- `envs/scienceworld/data/raw/sciworld_train.json` — symlink to the HF cache for `AgentGym/AgentTraj-L`'s SciWorld split.
  - 2120 trajectories, 42,178 (state, action) pairs, 24.9 MB.
  - Reproduce: `huggingface_hub.hf_hub_download(repo_id="AgentGym/AgentTraj-L", filename="sciworld_train.json", repo_type="dataset")`.

### Normalized input (real pipeline source)
- `envs/scienceworld/data/normalized/sciworld_train.json` — produced by `envs/scienceworld/scripts/normalize_agenttraj.py`.
- AgentTraj-L was generated against a pre-public scienceworld version whose naming conventions differ from any PyPI release (verified by trying 1.0.x / 1.1.0 / 1.1.1 / 1.1.2 / 1.1.3 / 1.2.x — no public version has both the right API and the original object names). To make trajectories executable in `scienceworld 1.1.3` (and bit-identically in `1.2.x`), we apply three string-level rewrites to **the gpt-turn `value`s only** (human-turn observations untouched as a forensic record):
  - **(A)** `green house` → `greenhouse` (room rename). 1,448 substitutions on 626 trajectories.
  - **(B)** `connect <non-wire-source> terminal 1 to <wire> terminal N` → `connect <non-wire-source> to <wire> terminal N`. New scienceworld dropped the explicit source-side terminal index on non-wire objects. 752 substitutions on 750 trajectories.
  - **(C)** Plain AgentTraj-L typos: `cachew → cashew`, `sandwhich → sandwich`. 15 substitutions on 8 trajectories.
- **Design choice**: the rewrites live in the data, not the env. This keeps downstream-published SFT data self-contained — a user who trains on our files and evaluates against an unmodified AgentGym + scienceworld stack will produce canonical action strings the env accepts. No env patch is required at inference time.

### Replay against unmodified env
- `envs/scienceworld/data/replay/replay_full.jsonl` — produced by `envs/scienceworld/scripts/replay_agenttraj.py` on the normalized input.
- One line per trajectory, ~1.7 GB total. Includes for each step: the action, our env's observation/reward/score/done/info (the `info.valid` field is the admissible-action list, kept for downstream IWM K=3 alternative sampling).
- **Pass-rate distribution**:
  - 2,038 / 2,120 (**96.1%**) reach `done && score == 100`. Becomes our candidate `D_expert`.
  - 19 of 22 task types pass at 100%; 1-3 at 92.3% (12/13), 2-1 at 99.6% (269/270; the 1 miss is env non-determinism near a thermometer threshold, not a normalize bug), 3-3 at 82.2% (370/450; the 80 misses are conductivity-test trajectories where the *gold path's logic depends on `book` being a two-terminal electrical element*, which scienceworld 1.1.x+ removed — not recoverable by any string rewrite).
- `envs/scienceworld/data/replay_summary.jsonl` — one summary row per trajectory.
- Everything under `envs/scienceworld/data/` is gitignored; regenerate via the scripts.

## Approved decisions

### Filter rule (TEAM_GUIDE §3 protocol)

**Approved 2026-05-16.**

`D_expert` is the subset of normalized AgentTraj-L trajectories that, when replayed action-by-action against the unmodified `scienceworld 1.1.3` env via AgentGym, reach `final_done == True` and `final_score == 100`.

- **Effect**: keeps **2,038 / 2,120 = 96.1%** of trajectories; **39,700** (state, action) pairs. Drops 82 trajectories (80 task-3-3 conductivity tests whose gold paths depend on `book` being a two-terminal electrical element that no longer exists, 1 task-1-3 pre-existing failure, 1 task-2-1 trajectory flapping near a thermometer-reading threshold).
- **Rationale**: paper §B.6 says "These expert trajectories are optimal given the completeness of task solvability in the dataset" — i.e., the paper already filters by task solvability in their env. We mirror that, replacing "their env" with our pinned `scienceworld 1.1.3`. The dropped 3.9% are gold paths whose underlying env semantics have changed enough that they no longer solve the task; using them as IL targets would teach the policy actions that demonstrably don't work.
- **Reversibility**: the dropped trajectories are still present in `data/replay/replay_full.jsonl` (with their per-step env response captured); only their inclusion in `data/sft/expert_sft.jsonl` is suppressed.

### CoT in expert SFT

**Approved 2026-05-16.**

`expert_sft.jsonl`'s assistant content keeps AgentTraj-L's `Thought:` text alongside the action (normalized form). Rationale from the pre-normalize CoT/obs consistency analysis (100 successful replays × 471 mismatch steps): 0 cases where a thought referenced specific obs details that contradicted the env's current observation; the thoughts are task-level meta-reasoning, robust to the obs drift we normalized away. Post-normalize the situation is at least as consistent. Keeping CoT makes `expert_sft` structurally compatible with `reflection_sft` (both end with `Thought + Action`), which makes the training mix cleaner.

### Trajectory adapts to env, not the reverse

**Approved 2026-05-16.**

All downstream pipeline outputs (expert/iwm/reflection SFT files) must use the env's current canonical naming and grammar. The downstream user installs unmodified AgentGym + scienceworld and a model trained on our SFT files should produce actions the env accepts. Any further mismatch we discover gets fixed at the data layer (extend `normalize_agenttraj.py` or rewrite further down the pipeline), never by patching the env at inference time.

### Drop post-completion (already-solved) states from reflection + truncate expert at completion

**Approved 2026-06-30 (TEAM_GUIDE §3 protocol).**

Context: AgentTraj-L's generation agent did not reliably recognize task completion. ScienceWorld scores the task at the goal action, but the agent kept padding with filler (`wait1`/`look around`/`examine`) until the episode ended. Evidence over the 2,038 surviving trajectories: of 3,030 `wait` states, **0** ever raised the score, 44% are the literal last step, and **1,581 states (3.98% of 39,700)** have `score == 100 already BEFORE the action** (i.e. the task was already solved). Of those 1,581, 93% are `wait`/`wait1`, the rest `use`/`examine`/`look`.

- **Reflection (post-completion)**: do NOT generate/keep reflections for post-completion states (pre-action score already == 100). These have no decision to reason about; reflections there are necessarily contentless or confabulated. Detection is exact via the per-step `score` in `replay_full.jsonl` (not a fuzzy action-type rule). **Effect: drops 1,581 / 39,700 = 3.98%.** The goal-completing state itself is KEPT (its pre-action score is < 100). Mid-trajectory waits (1,562, score < 100 — e.g. circuit-settle after `connect`, heating waits) are KEPT and handled by the reflection prompt's environment-facts block.
- **Reflection (env-flagged no-op)**: ALSO drop states whose `expert_next_state` reports the action had no effect — the complete enumerable "already" class (one rule, `"already" in next_state`): `The door is already open.` (2,384), `The cupboard is already open.` (31), `The sink is already deactivated.` (16), `The freezer is already open.` (4) = **2,435 (6.13%)**. These are env-confirmed redundant actions; even an honest reflection can only justify "do the useless thing." Scan confirmed this is the *complete* set of no-op responses (`No known action matches that input.` = 0; no other "nothing-happened" template in the data). Disjoint from post-completion (which has no `open`/`de-activate` actions), so **combined reflection drop = 1,581 + 2,435 = 4,016 (10.1%)**. Semantically-misguided-but-env-accepted actions (going to a wrong room, acting on an unseen object) are deliberately NOT filtered — not enumerable; handled by the reflection prompt's honesty constraints instead.
- **Expert**: truncate each trajectory at the goal-completing action (drop the same trailing post-completion turns). Rationale: those turns teach the model to emit filler `wait` *after winning*, which feeds the eval-time over-generation / fail-to-stop failure mode. Truncating teaches "stop when the task is solved."
- **IWM**: NOT filtered — `(s, wait, "nothing changes")` post-completion triples are valid world-model dynamics signal.
- **Reversibility**: dropped states remain in `replay_full.jsonl` / `sr_rollout.jsonl`; only their inclusion in `reflection_sft.jsonl` / `expert_sft.jsonl` is suppressed.

## SR data

### Rollout

`envs/scienceworld/scripts/smoke_sr.py` (despite the name, also handles the production run via `--n-samples 39700`) runs the SR pipeline for every state in D_expert:

1. **Proposer** (LLM, DeepSeek V4 Pro, thinking disabled, temperature 1.0): oversample to **K+2 = 7** alternative actions. Prompt feeds the AgentGym REACT action menu + task description + serialized history + current state + the expert action, with explicit "Banned vocabulary in output" (no expert/selected/chosen/...labels) and "No numbered alternative references" guidelines.
2. **Hybrid filter** (CPU): canonicalize → dedup → drop=expert → split into valid (in admissible) / invalid (not in admissible) buckets → take up to K=3 from valid, then top up with invalid fillers if short. Rationale: prefer the rich "valid alt vs expert" comparison signal, keep variable-K resolved while still surfacing the "recovery from invalid attempts" signal in the minority of states where the LLM's valid proposals run short.
3. **Env probe** (process pool, 16 workers): for each kept alternative, env.load + replay 0..i-1 + env.step(alt) to capture s_i^j. Same pattern as IWM rollout.
4. **Reflection** (LLM, DeepSeek V4 Pro, thinking disabled, temperature 0.7, frequency_penalty 0.3): paper §B.6 / METHOD.md §4.3 prompt template, with three deliberate deviations recorded in `method_recap.md`:
   - Input labels read "The action the agent takes here: X" (not "Expert Action") to keep DeepSeek from echoing privileged labels in the output.
   - Output has a "convergence anchor" clause: "the monologue must converge on the action the agent takes ... do NOT let the monologue end by recommending a different action." Cuts the rate of "model decides differently than the gold path" cases from ~20% in earlier iterations down to a low single-digit percent.
   - Length is **prompt-soft-capped at ~500 words**, NOT via `max_tokens` (which would truncate mid-sentence and produce worse data than overlong-but-complete reflections). Per SKILL.md hard rule #9: SR length is a per-env decision; this env targets 300-400 words.

Concurrency: LLM stages use ThreadPoolExecutor (sync DeepSeek client × 100 threads); env stage uses ProcessPoolExecutor (16 workers, one JVM each). Output streams to disk per record so a mid-run crash loses only the in-flight batch.

Production run on 2026-05-17: **17,407 s (~4.8 hours)** wall, **$50.54** at DeepSeek V4 Pro pricing, **117.6M input + 17.1M output tokens**.

`envs/scienceworld/data/rollout/sr_rollout.jsonl` (400 MB) — one raw record per state with proposer_raw + proposer_parsed + filter_trace + filtered_alternatives + reflection_prompt + reflection_cot. K_final hit K=3 for 39,692 / 39,700 states (99.98%); the hybrid composition shows 75.3% pure-valid, 16.0% (2v+1i), 6.8% (1v+2i), 1.9% (0v+3i).

### reflection_sft.jsonl

`envs/scienceworld/scripts/build_reflection_sft.py` joins sr_rollout with replay_full + iwm_rollout's initial_obs, applies the post-hoc quality filters below, and renders one single-turn chat-messages record per kept state.

**Quality filters applied at build time**:
- **Dedup doubled reflections** (+1,157): DeepSeek's mode-collapse occasionally writes the same reflection twice in one response; detection is "first 100 chars appear again after char 100, second half length matches first within 50 chars"; recovery keeps the first complete copy.
- **Drop unsafe duplicates** (-48): cases flagged as doubled but where the two halves aren't a clean byte-identical pair — unsafe to auto-dedup.
- **Drop banned-word leak** (-97): records where `expert` / `selected action` / `chosen action` / `correct choice` / `right action` / `best option` / `optimal action` / `best alternative` appeared in the reflection. Despite the prompt's explicit ban, DeepSeek leaks one of these in ~0.3% of records.
- **Drop numbered-label leak** (-1): records that referenced alternatives as "Action 1" / "Alternative 1" / "a_i^1".
- **Drop k_final < 3** (-8): edge cases where the proposer's 7 candidates all collapsed.
- **Paragraph-break normalization** (10,565 records affected): `"\n+"` → `" "` in the reflection text. SR prompt asked for single-paragraph but ~27% of records still contained breaks; normalizing keeps the training target structurally uniform.

Total drop: 154 (0.39%). Total dedup recovery: 1,157. Final output: **39,546 reflection_sft records, 296 MB**.

**Format** (per record):
- `system` : AgentGym REACT prompt (same as expert_sft / iwm_sft).
- `user`   : task_desc + initial_obs + (prior expert Thought/Action + env-obs history rendered inline) + current state s_i.
- `assistant` : `"Thought:\n<reflection>\n\nAction:\n<expert_action>"`.

Note this is **single-turn** (one (system, user, assistant) triple per state, not multi-turn like expert_sft). The single-turn choice matches iwm_sft's pattern and lets the trainer treat each state's reflection as an independent SFT example, which is appropriate since each reflection is grounded in its own (state, expert, alternatives) tuple.

## IWM data

### Rollout

`envs/scienceworld/scripts/rollout_iwm.py` probes K=3 random non-expert alternative actions at every expert state in D_expert and records the env's response per alternative.

- For each state `s_i` in a surviving trajectory: sample K=3 distinct actions uniformly without replacement from `admissible(s_i) \ {expert_action_i}` and step the env on each.
- Alternative sampling is deterministic via `seed = md5("{item_id}/{step}")[:16]`, so reruns produce identical alternatives.
- ScienceWorld has no native state save/restore (only `env.load` to a fresh `(task, variation)`), so each probe requires `env.load + replay 0..i-1 expert actions` to restore `s_i`. The cost dominates wall-clock.
- Run details: 64 workers (process-parallel, one JVM each), ~61 minutes for the full 2,038 trajectories. Output is `envs/scienceworld/data/rollout/iwm_rollout.jsonl` (24 MB).
  - 2,038 "initial" records (one per trajectory, capturing `env.look()` at the clean `s_0`).
  - 119,095 "alt" records (5 fewer than the expected 119,100; in those 5 corner cases `|admissible \ {expert}|` was less than 3 — accepted as-is, no refill).
- Expert IWM triples `(s_i, a_i, s_{i+1})` are not re-probed here — they're already in `data/replay/replay_full.jsonl`.

### iwm_sft.jsonl

`envs/scienceworld/scripts/build_iwm_sft.py` joins `replay_full.jsonl` with `iwm_rollout.jsonl` and renders single-turn SFT records.

- **Format** (one record per IWM triple):
  - `system` : AgentGym REACT system prompt (same as `expert_sft.jsonl`)
  - `user`   : `<task_desc>\n<initial_obs>\n<Thought/Action/Obs history rendered inline>\n\nAction:\n<action_being_probed>`
  - `assistant` : env's response to that action (raw next-state observation)
- **Why single-turn**: the IWM target (next-state) is naturally the env's response, which in multi-turn agent chats would be a `user` turn — but SFT trainers compute loss on `assistant` turns. Putting the next-state as the assistant content in a single-turn (system, user, assistant) record sidesteps the loss-masking issue and is unambiguous for any standard chat-format trainer.
- **History rendering**: the prior expert Thought/Action and env observations are folded into the `user` content as plain text, in the natural AgentGym REACT chat flow. This carries enough context for the model to know the room layout, prior actions taken, and current state — necessary because raw post-step observations in ScienceWorld are often terse (e.g. `"The door is now open."`).
- **Counts**: 39,700 expert IWM records + 119,095 alternative IWM records = **158,795 total records**. File size: 922 MB.
- **Distribution of next-state outcomes**: includes successful state changes ("the door is now open") as well as env rejection responses ("No known action matches that input.") — both are valid IWM training signal per METHOD.md §3 (paper §5: "IWM keeps all rollout triples; invalid-action error messages are retained as training signal.").

## Upstream

- **Outer submodule** (we own this fork):
  - Fork URL: https://github.com/UlyssesXC/AgentGym-RL
  - Submodule path: `envs/scienceworld/agentgym/`
  - Pinned branch: `scienceworld-ee` (forked from `main`)
  - Pinned commit: `0641bf6` (bumps inner to the compat-pin commit)
- **Inner submodule** (we own this fork, as of 2026-05-16):
  - Fork URL: https://github.com/UlyssesXC/AgentGym
  - Submodule path: `envs/scienceworld/agentgym/AgentGym/`
  - Pinned branch: `scienceworld-ee` (forked from upstream commit `d014732` on `WooooDyy/AgentGym`'s `KYLN24-patch-1-3-gd014732` branch — same code, just under our control)
  - Pinned commit: `8526986`. Contains exactly two changes vs upstream, both **generation-time only** (downstream evaluators don't need this fork):
    - `07b30b5`: add `GET /admissible_actions` endpoint (admissible-action list per state for IWM K=3 alternative sampling).
    - `8526986`: pin `scienceworld==1.1.3` in pyproject.toml and `hasattr`-guard `init_env.close()` for 1.1.x compatibility; switch the new endpoint to the camelCase API which works on both 1.1.x and 1.2.x.

The underlying `scienceworld` Python package is installed as a transitive dependency via AgentGym's setup; not a separate submodule.
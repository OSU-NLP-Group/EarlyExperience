# Tau-Bench

Customer-service env from [Yao et al., 2024](https://arxiv.org/abs/2406.12045) (Sierra). Multi-turn tool use against typed APIs plus a LM-simulated customer plus a policy/wiki document the agent must adhere to. Paper §B.4 scopes us to the **Retail** subset.

This is a pure in-process Python env: the "world state" is just three dict-of-dicts (`users`, `orders`, `products`) loaded fresh each `reset()`; tools mutate this in place; reward = byte-exact comparison of (the DB the agent reached) vs (the DB the ground-truth action sequence would have reached). The user side is another LLM that holds its own message history and plays the customer; conversation ends when the simulator emits `###STOP###` or the agent triggers a terminate tool (`transfer_to_human_agents`).

## Project-side characterization

**Action space**: structured-but-large. Retail has 16 tools (14 real + `think` no-op + `transfer_to_human_agents` terminate). Tool **name** is enumerable, tool **arguments** are free-form (string/int/dict). So replacing an expert action with an alternative means LLM-proposing both name and args, optionally constrained to a reduced tool set. This is paper Table 2's "structured but large action sets" regime — different from ScienceWorld's pure-enumerable case.

**State save/restore**: free. `data` is a python dict, in-process, no JVM, no HTTP, no subprocess. `deepcopy(env.data)` snapshots; `env.data = deepcopy(snapshot)` restores. No JVM-style O(N²) replay cost we hit in ScienceWorld; alternative-action probing at K=5 per state will cost O(K) deepcopies per state, nothing more.

**Two LLMs in the loop during expert collection**: the env package itself embeds a user-simulator LLM via `litellm.completion` (`tau_bench/envs/user.py`). Both the agent and the user simulator are called per turn. **Once the expert trajectory is recorded, neither LLM is needed for downstream IWM/SR rollout** — the user's turns are baked into the recorded traj as plain text, and the env's DB can be deterministically reconstructed by replaying recorded tool calls on a fresh `load_data()`. This is the key reason this env is *cheaper* than its appearance suggests — only the expert-collection stage burns user-simulator tokens.

**Expert source**: there is no released expert dataset for tau-bench (the `historical_trajectories/` shipped in the repo are test-split gpt-4o/sonnet rollouts at 4 trial each — wrong split, wrong purpose). Paper §B.4 generated theirs by running a "high-performing instruction-tuned LLaMA-family model" against the env at temp=1.0, 4 trial per task, kept reward=1 with random tiebreak. **We did the same protocol with deepseek-v4-flash** (workspace LLM-uniformity rule + see [Approved decisions: LLM choice]).

**Tasks file count vs paper**: our `tasks_train.py` has **500** Task entries; paper §B.4 says "495". Diff of 5 unresolved — likely paper filtered 5 invalid tasks. Did not chase down; treated all 500 as candidates. Our 42 dropped (0-success) tasks include task IDs `[5, 7, 8, 10, 22, 34, 35, 42, 55, 64, 130, 142, 145, 149, 206, 244, 252, 267, 305, 321, ...]`; whoever wants to A/B against paper's exact split would need paper's source list.

## Scope and limits

- **Domain**: retail only. Airline subset not collected.
- **Dataset scale**: **458 expert trajectories, 5,249 (obs, action) pairs** — both slightly beat paper's 452 / 5,239.
- **K for IWM**: 5 (paper §B.4 explicit). LOCKED.
- **K for SR**: 3 (paper §B.4 explicit; subset of the 5 WM datapoints). TODO at SR stage.
- **Alt-action sampling regime**: LLM-proposed (paper says "use the target model to propose five action candidates", with the expert's tool removed from the toolset). DeepSeek-v4-flash plays the proposer (not the policy under training). LOCKED, see "Approved decisions: IWM proposer".
- **SR reflection length**: TODO — to be decided at SR stage per SKILL hard rule #9. Working assumption: target 200–400 words with soft cap ~500, prompt-enforced only.
- **`respond` turns**: paper's count math (SR 5,233 ≈ expert 5,239) implies SR covers nearly all expert observations including `respond` turns. LOCKED: same regime applies at respond obs — alternatives are tool calls drawn from the same 14-tool menu (no "expert tool to remove" since the expert action isn't a tool). All 6 triples per obs (1 expert + 5 alt) are kept in IWM. The expert-respond triple's next_obs is the recorded user reply — formally a user-simulator response rather than world dynamics, but its loss-time treatment is identical and it teaches the trained agent useful conversation priors.

## Rough guide

1. **Expert** — DONE. `expert_rollouts_raw.jsonl` (2000 rollouts) + `D_expert.jsonl` (458 trajectories, paper-filtered).
2. **IWM** — DONE. Two-stage pipeline:
   - **Stage 2a — proposer** (`scripts/propose_alternatives.py`): one DeepSeek call per obs, returns K=5 JSON tool-call candidates. System prompt carries WIKI + the 14-tool menu verbatim (cached across calls — 59% cache hit on full run, 94% steady-state after warm-up). User prompt has rendered conversation history + a strongly-worded "Forbidden tool" line. **Full run**: 5,249 calls / 3.7 min wall / 100% JSON parse / 0 errors / refill 0.06/obs / 20.4M input + 1.2M output tokens (12.0M cached).
   - **Stage 2b — env probe** (`scripts/iwm_rollout.py`): pure Python, no LLM. For each obs, replay recorded expert tool calls on a fresh `load_data()` to reach `s_i`, deepcopy, invoke each alt + record next_obs. Expert next_obs is read straight from the recorded traj (env is pinned + deterministic). **Full run**: 458 trajs / 5,249 obs / 26,245 alt probes / 43.4 min wall single-threaded / 0 unknown / 9,110 errors (34.7% — paper §5 valid IWM signal).
3. **SR** — DONE. Two-stage pipeline:
   - **Stage 3a — proposer** (`scripts/propose_reflections.py`): one DeepSeek v4-pro call per obs, returns reflection CoT. Uses **signal-contrast greedy alt selection** (pick 3 of 5 alts to maximize `(outcome_class, tool_family)` coverage, with a preference for at least one `ok` outcome as anchor). System prompt carries WIKI + leakage-prevention guidelines (banned vocab + no numbered labels + convergence anchor + 200–400 word soft cap, no `max_tokens`). `temperature=0.7`, `frequency_penalty=0.3` (per ScienceWorld pitfall). **Full run**: 5,249 calls / 10.0 min wall (128 workers, 9.4 req/s) / 0 errors / avg 359 words. 47% cache hit averaged (~95% steady).
   - **Stage 3b — rerun** (`scripts/rerun_reflections.py`): 145 records that tripped any quality flag (`is_doubled` / `has_banned_vocab` / `has_numbered_label` / `exceeded_soft_cap`) re-issued at `attempt=2` with same prompt (relying on sampling variation). **78 (53.8%) cleaned** by resampling alone; the remaining **67 flagged records were all confirmed false positives** by manual context inspection — driven by retail-domain words the obs forces: `gold` (color), `option N` (customer's choice from numbered list the agent presented), `expert` (jigsaw puzzle difficulty), `the chosen` (customer's chosen variants). Real label-leakage rate after merge = effectively **0%**.
4. **SFT build** — DONE. Three `scripts/build_*_sft.py` scripts produce the three required SFT files under `envs/tau-bench/data/sft/`. Serialization choices below.

## Things easy to get wrong here

- **Use the historical_trajectories shipped in the repo as `D_expert`** — they're test-split gpt-4o/sonnet rollouts (115 task × 4 trial = 460 records), wrong split and wrong purpose. Confirmed by inspection during recon.
- **Re-run user simulator during downstream rollout** — IWM/SR alternative-action probing only needs tool execution on a DB snapshot, not user simulation. Doing so would burn user-LLM tokens unnecessarily AND would diverge from the recorded user turns (which are baked into the traj).
- **Forget to deepcopy `data` before each alt probe** — tau-bench tools mutate `data` in place. Without a snapshot you'd contaminate the next probe.
- **Treat `respond` observations same as tool observations during IWM probing** — paper hints they're treated alike (count match), but the "remove expert's tool" rule doesn't apply at respond turns. Decide explicitly at IWM stage.
- **Trust `r._hidden_params["response_cost"]`** — litellm 1.86 does not have pricing for `deepseek-v4-flash` / `deepseek-v4-pro`; the field returns `None` (handled by tau-bench code as `or 0`). Cost must be tracked out-of-band.
- **`max_num_steps` cap = 30** — `ToolCallingAgent.solve(max_num_steps=30)` is hardcoded in the agent file; long-multi-action tasks (e.g., 3 cancels + 2 mods) can hit this if the agent dithers. In our smoke this never happened, but if it does in IWM/SR replay (it shouldn't, since we're replaying), it's a knob to remember.
- **Submodule's user.py uses `react`/`verify`/`reflection` user strategies that make extra LLM calls** — we use `llm` strategy (cheapest, 1 user-LLM call per user turn). Switching strategies multiplies user-LLM cost.

## Recon findings

- **All 16 retail tools work, schema is well-formed**. `ALL_TOOLS` enumerated, `get_info()` returns proper OpenAI tool schemas.
- **`Task` pydantic model** has fields `{user_id, instruction, actions, outputs}`. `tasks_train.py` passes `annotator=` as an extra kwarg; pydantic v2 default policy silently ignores it — no data loss, no error.
- **WIKI** (system prompt) is 5,718 chars / 81 lines of retail policy. Hard rule: agent must verify user id before any DB-mutating action and must explicitly confirm with the user before executing it.
- **DeepSeek v4 family** model IDs (`deepseek-v4-flash`, `deepseek-v4-pro`) work via litellm's `deepseek` provider. Litellm's cost map does not know these names but that doesn't block API calls. Thinking mode is **off** by default for v4-flash (confirmed by probe: `reasoning_content: None`).
- **Per-task hit rate is invariant flash vs pro**. Smoke A/B on tasks 0–19: both models hit 17/20 = 85.0%. Pro's only edge was per-trial pass^1 (0.775 vs 0.725) — useless to us since paper protocol keeps only 1 trial per task. The 3 hard tasks (5, 8, 10) defeated both models 0/4 across 8 total trials; these are intrinsic task failures, not model-skill issues.

## Setup (reproducible commands)

```bash
# 1) Submodule (already registered in .gitmodules, pinned at upstream main commit 59a200c)
git submodule update --init --recursive

# 2) Conda env. Python 3.10 covers tau-bench's package matrix (openai >=1.13, anthropic
#    >=0.26, google-generativeai >=0.5, litellm >=1.41 etc.). Use python -m pip from
#    inside the env — `conda run pip` will leak to user-site.
conda create -n tau-bench-ee python=3.10 -y
PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python -m pip install -e envs/tau-bench/tau-bench/
PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python -m pip install httpx idna     # litellm transitive deps litellm doesn't pull

# 3) Auth
export DEEPSEEK_API_KEY=...

# 4) Always run python inside the env with PYTHONNOUSERSITE=1 to avoid the user-site
#    leak we hit during setup (system-wide httpx without idna broke import). Convention
#    for every pipeline script in envs/tau-bench/scripts/.
```

Installed versions: `tau_bench==0.1.0` (editable), `litellm==1.86.0`, `openai==2.38`, `anthropic==0.104`, `pydantic==2.13`, `httpx==0.28`. Python 3.10.

## Data

### Raw rollouts
- **`envs/tau-bench/data/rollout/expert_rollouts_raw.jsonl`** (61.1 MB) — 2000 rollouts, one per line. Each line is the `EnvRunResult` schema: `{task_id, trial, reward, info, traj}`. `traj` is the full OpenAI-format messages list (system / user / assistant with `tool_calls` field / tool); `info.task` carries the ground-truth `actions` sequence and `info.reward_info.gt_data_hash` is the GT DB hash. This is the audit trail for everything downstream.
- Reproduce: 500 task × 4 trial via tau-bench's native `python run.py`. Single command at the bottom of this file. Wall ≈ 1h50min at `--max-concurrency 16`.

### Filtered D_expert
- **`envs/tau-bench/data/rollout/D_expert.jsonl`** (13.5 MB) — 458 rollouts, one per task, all reward=1. Built by `envs/tau-bench/scripts/build_d_expert.py` with `--seed 10` for reproducible random-tiebreak when multiple trials of a task succeed.
- Distribution of trials-passed for the 458 kept tasks:
  - 4/4 passed: 336 tasks (we picked 1 of 4)
  - 3/4 passed: 84 tasks (1 of 3)
  - 2/4 passed: 29 tasks (1 of 2)
  - 1/4 passed: 9 tasks (only 1 candidate — K=4 paid off here)
- 42 tasks dropped (0/4 across all trials). First 20 IDs: `[5, 7, 8, 10, 22, 34, 35, 42, 55, 64, 130, 142, 145, 149, 206, 244, 252, 267, 305, 321]`.
- (obs, action) count: **5,249 = 2,748 tool-call + 2,501 respond**. Both totals slightly beat paper's 5,239.

### IWM rollout (proposer alts + env-probed next-obs)
- **`envs/tau-bench/data/rollout/alternatives.jsonl`** (8.7 MB) — 5,249 lines, one per obs. Each line: `{task_id, trial, obs_idx, expert_kind, forbidden_tool, llm_raw, parse_ok, refills_needed, alts:[{name, arguments, source}×5], usage}`. Proposer output, no env probe yet.
- **`envs/tau-bench/data/rollout/iwm_rollout.jsonl`** (27.5 MB) — 5,249 lines, one per obs. Each line: `{task_id, trial, obs_idx, expert_kind, expert:{action, next_obs}, alts:[{action, next_obs, source}×5]}`. The full IWM dataset = 31,494 triples (1 expert + 5 alts per obs).
- Probe behavior breakdown (26,245 alts):
  - LLM-sourced alts (25,922 = 98.8%): 66.1% returned valid env response, 33.9% returned `Error:` (paper §5 retains as IWM signal).
  - Fallback-random alts (323 = 1.2%, empty args): 97.8% errored — expected, empty-args usually fails arg validation; still valid IWM signal.
  - 0 unknown-action responses (refill always draws from menu, so all tool names exist in `TOOLS_MAP`).

### SR rollout (reflection CoTs)
- **`envs/tau-bench/data/rollout/sr_rollout.jsonl`** (34 MB) — 5,249 records, one per obs, attempt 1. Each: `{task_id, trial, obs_idx, expert_kind, expert_action, expert_next_obs, picked_alts:[3 with outcome_class/tool_family], alt_selection_diagnostic, reflection_raw, flags:{is_doubled, has_banned_vocab, banned_vocab_hits, has_numbered_label, numbered_label_hit, has_paragraph_breaks, word_count, exceeded_soft_cap}, model:"deepseek-v4-pro", attempt:1, usage}`. Generated by `scripts/propose_reflections.py`. **No records dropped** at this stage; cleanup is downstream.
- **`envs/tau-bench/data/rollout/sr_rerun.jsonl`** (~1 MB) — 145 records that tripped a flag in attempt 1, re-run at attempt 2 via `scripts/rerun_reflections.py`. Same schema plus `prev_flags` for audit. Of the 145, 78 became clean and 81 had strictly-fewer flags (these are used at SFT-build); 67 still had the same flag (all confirmed false positives).

### Final SFT files (training-ready)

All three files use **native OpenAI Chat Completions format** with top-level `messages` + `tools` fields. This is the modern idiom matched by every tool-calling chat template (qwen2_tool, hermes2_tool, llama3_tool, etc.) and by `litellm.completion(messages=..., tools=...)` — i.e. exactly what tau-bench's eval loop produces at inference, so train-time and inference-time tokenized prompts are byte-identical. Sharegpt-only legacy trainers can convert with a 10-line `tools→system` flattening script.

- **`envs/tau-bench/data/sft/expert_sft.jsonl`** (126.0 MB) — **5,248 records** (2,748 tool + 2,500 respond). One per (obs, expert_action) pair. Multi-turn. The LAST `assistant` message is the training target. Avg 14.9 messages per record. Each record carries `tools` (16 retail tool schemas) at the top level.
- **`envs/tau-bench/data/sft/iwm_sft.jsonl`** (575.8 MB) — **31,490 records** (5,245 expert triples + 26,245 alt triples). One per (s, a, s') triple. **Single-turn `(system, user, assistant)`** — IWM's target is the env's next-state, which would be a `tool` or `user` role in a multi-turn chat, so we flatten and put it as the `assistant` content (same design rationale as SearchQA's IWM SFT). User content: rendered conversation history + the action being probed rendered as `<tool_call>{"name":...,"arguments":{...}}</tool_call>` or `<response>...</response>` (Hermes-style tags). System prompt explicitly signals "predict env, not act" to distinguish from agent-mode. 9,110 of the 26,245 alt records (34.7%) have `Error:` / `Unknown action` next_obs — retained per paper §5 as valid IWM signal. Tools schemas are embedded per record (~10 KB / record) — net +350 MB on top of the bare data, paid because the IWM model genuinely benefits from explicit tool signatures when learning (s,a)→s' (esp. for long-tail tools and argument-edge errors).
- **`envs/tau-bench/data/sft/reflection_sft.jsonl`** (139.6 MB) — **5,249 records** (2,748 tool + 2,501 respond). One per obs. Multi-turn. Merged from `sr_rollout.jsonl` + `sr_rerun.jsonl` preferring strictly-fewer-flag attempt: **5,168 records use attempt 1, 81 use attempt 2**. Target assistant message:
  - **tool turn**: `{content: <reflection_CoT>, tool_calls: [<expert_tool>]}` — reflection in OpenAI `content`, action in structured `tool_calls`. `qwen2_tool` / `hermes2_tool` / `llama3_tool` templates' `if/elif` logic uses `tool_calls` if present (skipping `content` in render) but still applies loss to the target assistant tokens including the reflection — so the reflection is learned as the "thinking" text that precedes the tool call.
  - **respond turn**: `{content: <reflection_CoT> + "\n\n" + <respond_text>}` — no `tool_calls` field; customer-facing reply appended after reflection in natural prose.
  - Final flag distribution: 0 flags = 5,182 (98.7%); 1 flag = 66; 2 flags = 1. The 67 remaining-flag records are confirmed regex false positives (`gold` color / `option N` customer choice / `expert` puzzle difficulty / `the chosen` variant), safe to use as-is.

### SFT build — design history + final shape (2026-06-02)

Three rounds of iteration before settling on the final design:

1. **Round 1 (initial build)**: bare OpenAI native format, no top-level `tools` field, content="" on tool-only assistants, LiteLLM metadata fields (`function_call`, `reasoning_content`, `provider_specific_fields`) preserved.

2. **Round 2 (reviewer flag — empty content bug under sharegpt)**: a reviewer agent noted that LLaMA-Factory's default `sharegpt` mode computes loss on every assistant `content` field and ignores `tool_calls`, so the 8,324 / 7,173 / 4 empty-content tool-only turns in expert / reflection / iwm would teach the model to emit empty strings under sharegpt. Patched by rendering `tool_calls` into `content` as Hermes-tagged text + replacing empty tool / respond contents with placeholders. **REVERTED in Round 3** — the patch was misdirected: the user is using `qwen2_tool`, whose `if/elif` template skips content when `tool_calls` is present, so the original empty content is correct; the patch's Hermes-tagged content would have appeared as DEAD DATA at training time (template skips content for tool-only turns, so the rendered text is unused) and as a SOURCE OF MISALIGNMENT relative to inference-time prompts.

3. **Round 3 (final — tools field + native format + cruft strip)**: settled on the format documented above.
   - Top-level `tools` field added per record (Option B from the A/B discussion). Mirrors `litellm.completion(messages, tools=tools_info)` exactly, so qwen2_tool template renders training-time and inference-time prompts identically.
   - LiteLLM metadata fields stripped from every message — only canonical OpenAI Chat Completions fields kept (`role`, `content`, `tool_calls` for assistant; `role`, `content`, `tool_call_id`, `name` for tool).
   - Empty-target drops kept (1 in expert from task 316 trial 1 obs 10 DeepSeek API anomaly; 4 in iwm from `think` tool's empty-string return — both are "produce nothing" targets, harmful under any template).
   - Empty-context cells (5 respond + 3 tool in expert and reflection) left as-is — qwen2_tool's `if/elif` skips them in render, and they don't contribute loss either way.

Post-final verification:
- All three files: **`tools` field present in 100% of records** (16 schemas each).
- All three files: **0 LiteLLM cruft fields** remaining.
- All three files: **0 empty TARGETS**.
- `expert_sft`: 5,248 records / 36,581 assistant turns / 17,902 tool messages.
- `iwm_sft`: 31,490 records / 31,490 assistant turns / 0 tool messages.
- `reflection_sft`: 5,249 records / 36,586 assistant turns / 17,905 tool messages.

### Why not Option A (tool schemas embedded in system content)?

Option A — concatenate the 16 tool schemas into each record's `system` content — is what bfcl_v4 uses, **but only because bfcl_v4's tools vary per task** so dataset-level `tools` wouldn't fit. tau-bench has a fixed 16-tool set, so Option B (top-level `tools` field) wins:

- `tau-bench`'s `ToolCallingAgent.solve()` calls `litellm.completion(messages=..., tools=self.tools_info)`. The `tools` parameter is rendered into the prompt by the model's chat template at a template-defined location.
- **Option B** lets the same chat template render `tools` at the same location during training — training prompt ≡ inference prompt, byte for byte.
- **Option A** would put tool schemas in `system` content at training time, but at inference the `tools` parameter still goes through the template's rendering path; without modifying tau-bench's eval loop to skip passing `tools`, you'd get **double rendering** (schemas in `system` content AND template-injected). Either you patch the eval loop (breaks portability) or accept the mismatch (degrades trained model performance).
- Releasing under Option B aligns the dataset with how every modern tool-calling SFT dataset is structured (Hermes-3, NousResearch, ToolACE, etc.) and matches the OpenAI Chat Completions API contract precisely.
- A 10-line `flatten_tools_to_system.py` transform is enough for legacy sharegpt-only trainers who insist on the embedded form.

Net data loss: 1 expert SA pair + 4 IWM triples = 5 records / 41,992 total = 0.012%. Negligible.

### Smoke (kept as baselines)
- `envs/tau-bench/data/smoke/tool-calling-deepseek-v4-flash-1.0_range_0--1_user-deepseek-v4-flash-llm_0524181148.json` — 1-task connectivity smoke
- `envs/tau-bench/data/smoke/tool-calling-deepseek-v4-flash-1.0_range_0--1_user-deepseek-v4-flash-llm_0524210856.json` — 20-task × 4-trial flash smoke (hit 17/20 = 85%)
- `envs/tau-bench/data/smoke/tool-calling-deepseek-v4-pro-1.0_range_0--1_user-deepseek-v4-pro-llm_0524212958.json` — 20-task × 4-trial pro A/B smoke (also hit 17/20 = 85%, decision to stay on flash)

Everything under `envs/tau-bench/data/` is gitignored; regenerate from the commands below.

## Approved decisions

### LLM choice — deepseek-v4-flash for both agent and user simulator (2026-05-24)

- **Agent** (expert generator): `deepseek-v4-flash`. Paper used "high-performing instruction-tuned LLaMA-family"; workspace LLM-uniformity rule says all data-generation calls go through DeepSeek. A/B vs deepseek-v4-pro on 20-task × 4-trial showed **zero gain** in per-task hit rate (both 17/20 = 85.0%), with pro saving zero hard tasks and costing 5–10× more.
- **User simulator**: same model. Paper didn't specify; tau-bench's default is gpt-4o. We deviate to keep workspace LLM-uniformity, single API key, single rate-limit pool. Since the user simulator is part of the env and **we don't train on it**, its quality drift relative to gpt-4o doesn't propagate into the SFT target.
- **Sampling**: `temperature=1.0` (paper §B.4), `thinking` mode off by default for flash (confirmed).

### Expert filter — paper §B.4 protocol (2026-05-24)

Standard paper rule, not a discretionary filter, so the §3 approval protocol does not apply:
- Per task: keep one trial with `reward == 1` chosen at random (seeded); drop the task entirely if 0 trials succeed.
- Effect: 458/500 = 91.6% of tasks contribute; 42 tasks dropped. Beats paper's 91.3%.
- Reversibility: the 1,542 non-kept and 42-task-worth of failed rollouts are all preserved verbatim in `expert_rollouts_raw.jsonl`; rerunning `build_d_expert.py` with a different seed or rule gives a different `D_expert.jsonl` without re-paying any LLM cost.

### K=4 trials per task (2026-05-24)

Paper protocol. Empirical from smoke: K=1 gives ~67.5% hit rate (loses ~25% vs paper), K=2 gives ~82.5% (loses ~9%), K=4 gives 85% (smoke) / 91.6% (full). Going beyond K=4 to chase >91.6% would violate paper protocol and have diminishing returns.

### IWM proposer — menu, prompt, smoke gate (2026-05-25)

**Menu = 14 tools** (the 16 retail tools minus `think` and `transfer_to_human_agents`).
- `think`: no-op, `invoke()` returns `""` — zero IWM signal. The expert used it 1 time across the full D_expert.
- `transfer_to_human_agents`: terminate tool with fixed/empty invoke output, also zero env-dynamics signal. Expert used it 10 times.
- Paper §B.4's "remove expert tool" rule is framed as "avoid repetitive signal"; excluding these is the same spirit applied to "avoid no signal". Defensible deviation.
- The rare expert turn that used one of these tools is still recorded in `D_expert.jsonl` and `expert_sft.jsonl`; only the K=5 *alternatives* at that obs are drawn from the 14-tool menu.

**Proposer**: DeepSeek `deepseek-v4-flash`, `temperature=1.0`, thinking off, `response_format=json_object`. One call per obs returning a JSON object `{"candidates":[{"name":..., "arguments":{...}}, ...]}` — paper-faithful "5 in one call" per `method_recap.md` (5× cheaper than 5 separate calls).

**System prompt** (cached): retail WIKI verbatim + 14 tool signatures + format rules. ~2.6k tokens. Identical across all 5,249 calls — confirmed 94% prompt-cache hit rate in v2 smoke after warm-up.

**Forbidden-tool enforcement**: a strongly-worded user-prompt block names the expert's tool and forbids any variant (different args, casing). The v1 prompt's polite wording let LLM cling to the same tool family ~50% of the time on tool obs; v2 explicit emphasis dropped refill rate from **0.52/obs → 0.05/obs** (10× improvement) on the same 20-traj benchmark. Locked at v2.

**Post-hoc filter** (in `propose_alternatives.py`):
1. Drop candidates whose name is the forbidden tool.
2. Drop candidates whose name is not in the 14-tool menu (catches LLM proposing excluded `think`).
3. Drop exact-duplicate (name + sorted-args) candidates.
4. Refill to K=5 with random tool names from the 14-tool menu (preferring unused names), empty `arguments` dict. Deterministic per (task_id, trial, obs_idx) for reproducibility.

**Smoke v2 results** (20 trajs, 224 obs):
- Parse OK: 224/224 (100%).
- Refill distribution: `{0: 216, 1: 6, 2: 1, 4: 1}` — 96% obs got 5 clean LLM alts with zero random fill.
- Per-kind refill: respond obs mean 0.000, tool obs mean 0.099.
- Tool-name distribution across LLM alts: all 14 tools used; top tool 13% (no domination).
- Prompt-cache hit: 94% (warm cache after ~30 calls).
- Wall: 11s for 224 calls at concurrency 64.

**Full-run projection** (5,249 obs):
- ~20M input + ~1.2M output tokens, ~95% cached → effective cost <$3.
- Proposer wall: ~5 min.
- Env probe wall: ~45 min single-threaded (deepcopy-bound).

**Full-run actuals** (2026-05-25):
- Proposer: 5,249 calls in **3.7 min** (23.7 req/s), 100% parse, 0 errors, refill 0.06/obs. Tokens: 20.4M in / 1.2M out / 12.0M cached (59% cache hit averaged across the whole run; ~94% after warm-up).
- Env probe: 458 trajs / 5,249 obs / **43.4 min** wall, 26,245 alt probes total, 9,110 errored alts (paper-valid signal), 0 unknown action.
- Final IWM dataset = **31,494 (s, a, s') triples** across 5,249 obs.

### SR generation — greedy alt selection + v4-pro + flag-no-drop (2026-05-25)

**Alt selection**: greedy diversity maximization on `(outcome_class, tool_family)` buckets, with priority for at least one `ok`-outcome anchor. Chosen over paper-literal random-3-of-5 because random leaves ~16% of obs with reflections that compare 3 essentially-identical alts (no learnable contrast). Greedy drops that to ~3.6% at the cost of a mild +2–4 pct bias toward rare categories (`business_rule`, `mutate_user`, `compute`). See "Approved decisions: SR alt selection" data analysis transcript for the trade-off rationale.

**Model**: `deepseek-v4-pro` for reflection writing (not flash). Reflection is long-form reasoning where pro's quality edge matters; cost differential (~5–8× flash) is acceptable for ~5k calls.

**Filter policy**: at rollout time, **all 5,249 records are written with annotated quality flags but NOT dropped**. Re-run pass handles the genuinely-bad ones via sampling-variation. The SFT-build script picks fewer-flag attempt per (task,trial,obs). This separates data generation from quality control and makes both reversible.

**Full-run actuals**:
- Proposer (attempt 1): 5,249 calls / **10.0 min** / 128 workers / 9.4 req/s sustained / 0 errors / avg 359 words / 21.3M in / 2.4M out / 10.0M cached (47% avg / ~95% steady).
- Rerun (attempt 2 on 145 flagged): 145 calls / **28 sec** / 128 workers / 0 errors / 78 cleaned (53.8%) / 67 still-flagged were all confirmed false positives.
- Final SR dataset (post-merge): **5,249 reflections**, 5,168 from attempt 1 + 81 from attempt 2; **0 flags** in 5,182 records (98.7%); 67 records have 1+ false-positive flag (retail-domain words like `gold`/`option N`/`expert`/`the chosen`).

### SFT serialization — three files, three target shapes (2026-05-25)

| File | Records | Format | Target |
|---|---:|---|---|
| `expert_sft.jsonl` | 5,249 | Multi-turn OpenAI with **native `tool_calls` structure** | Last `assistant` message of each record |
| `iwm_sft.jsonl` | 31,494 | **Single-turn** `(system, user, assistant)`; action rendered as Hermes-style `<tool_call>{name,arguments}</tool_call>` / `<response>...</response>` in user content | `assistant` message = raw env response string |
| `reflection_sft.jsonl` | 5,249 | Multi-turn OpenAI; **for tool turns**, target `assistant = {content: reflection, tool_calls: [expert]}`; **for respond turns**, target `assistant = {content: reflection + "\n\n" + respond_text}` | Last `assistant` message |

- **Why expert/reflection are multi-turn while IWM is single-turn**: IWM's training target is the env's next-state, which would naturally be a `tool` or `user` role in a multi-turn chat — but SFT trainers compute loss only on `assistant` turns. Putting the next-state as the `assistant` content of a single-turn record sidesteps loss-masking issues (same design rationale as SearchQA's IWM SFT). Expert and reflection both have `assistant`-role targets natively, so they can preserve the multi-turn structure.
- **Why native `tool_calls` is preserved rather than flattened to text**: modern training stacks (Hermes-2, Llama-3.1-Instruct, Mistral, LLaMA-Factory, TRL) handle OpenAI tool-calling messages directly. Preserving the structure keeps the data faithful to inference-time format (tau-bench's `ToolCallingAgent` emits `tool_calls`) and avoids text-parsing fragility. The Hermes-tagged text rendering is used ONLY in `iwm_sft.jsonl`'s user content because the alt action there is INPUT, not target.
- **`respond`-turn reflection format**: appending the customer-facing reply after the reflection in the same `content` field (no explicit separator). Trained model emits reflection followed by the actual reply as natural prose. Trivial heuristic at inference time: the last paragraph after a clear reasoning sweep is the message to the customer.
- **No additional filtering at SFT build**: all 5,249 SR records make it in; the 67 false-positive-flagged ones are preserved with their flag metadata for downstream inspection if needed.

## Reproduce the full run

```bash
# Expert collection (~1h50min wall, ~$? on flash)
cd envs/tau-bench/tau-bench
PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output python run.py \
    --env retail --task-split train \
    --start-index 0 --end-index 500 \
    --num-trials 4 \
    --model deepseek-v4-flash --model-provider deepseek \
    --user-model deepseek-v4-flash --user-model-provider deepseek \
    --user-strategy llm --max-concurrency 16 --temperature 1.0 \
    --log-dir /abs/path/to/envs/tau-bench/data/rollout/

# D_expert build (deterministic, ~5s)
cd /abs/path/to/repo/root
PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python envs/tau-bench/scripts/build_d_expert.py

# IWM proposer (~4 min, ~$2-3 on flash)
DEEPSEEK_API_KEY=... PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python envs/tau-bench/scripts/propose_alternatives.py

# IWM env probe (~45 min, no LLM)
PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python envs/tau-bench/scripts/iwm_rollout.py

# SR generation (~10 min on v4-pro, 128 workers; ~$25-80 cost range)
DEEPSEEK_API_KEY=... PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python envs/tau-bench/scripts/propose_reflections.py --max-workers 128

# SR rerun for any flagged record (~30 sec, <$0.50)
DEEPSEEK_API_KEY=... PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python envs/tau-bench/scripts/rerun_reflections.py --max-workers 128

# Build the three SFT files (deterministic, ~30 sec total, no LLM)
PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python envs/tau-bench/scripts/build_expert_sft.py
PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python envs/tau-bench/scripts/build_iwm_sft.py
PYTHONNOUSERSITE=1 conda run -n tau-bench-ee --no-capture-output \
    python envs/tau-bench/scripts/build_reflection_sft.py
```

## Upstream

- Fork URL: https://github.com/UlyssesXC/tau-bench
- Submodule path: `envs/tau-bench/tau-bench`
- Pinned branch: `main`
- Pinned commit: **`59a200c`** (unchanged from sierra-research/tau-bench upstream `main` as of 2026-05-24)
- Local patches: **none**. The fork is bit-identical to upstream; we have not edited a single line inside the submodule. All adaptation happens via CLI flags to `run.py` and post-processing scripts that live outside the submodule.

The upstream README explicitly warns "tasks in this repo are not updated — please use τ³-bench". We deliberately pin to this version to remain paper-faithful (paper used this version).

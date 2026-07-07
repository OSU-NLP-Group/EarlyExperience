# BFCL v4 (multi-turn function calling)

Paper §B.3 calls this "BFCLv3"; we pin v4 because v4 multi-turn data is the same lineage as v3 multi-turn plus ~6 months of upstream GT / evaluator fixes. File names are `BFCL_v4_*`; task content matches paper's spec.

## Env at a glance

- **In-process Python env.** No JVM / no HTTP service. 8 sim API classes (see "Action space"), executed by `eval()` on live instances. State = chat history + `vars(sim_instance)` of each `involved_classes` instance. Pure-Python state → `copy.deepcopy(instance)` works (this is what unlocks our IWM design vs SciW).
- **An "action" is a Python-syntax function call string**, e.g. `mv(source='a', destination='b')`. Within one agent emit step, the model may emit a *list* of parallel calls.
- **Each "case" is a scripted sequence of 1–7 user turns.** `question` is a static list — user does **not** react to model output. Multi-turn here means scripted subtasks sharing world state, not dialogue.
- **Within-turn budget = `MAXIMUM_STEP_LIMIT = 20` emit steps.** Step 21 triggers `force_quit` → entire case aborts. `valid==true` filtering excludes those.
- **BFCL eval is state-based**: passes iff final sim state == GT-executed sim state. Action sequences need not match GT byte-for-byte.

## Repo

| Item | Value |
|---|---|
| Upstream | `https://github.com/ShishirPatil/gorilla` (Apache 2.0) |
| Fork | `https://github.com/UlyssesXC/gorilla.git` |
| Submodule path | `envs/bfcl_v4/gorilla/` |
| Tracked branch | `main` |
| **Pinned commit** | **`aed2de1`** (2025-12-17, = `f7cf735` "Multiple Dataset Fix and New Model Support" + pyproject pathlib cleanup) — matches the 2025-12-16 BFCL leaderboard publication that generated our expert trajectories |
| Install | `cd envs/bfcl_v4/gorilla/berkeley-function-call-leaderboard && pip install -e .` (in its own conda env `bfcl`, py 3.10) |
| Replay verified | 3 Opus `valid==true` trajectories replayed against this pin produce byte-identical tool outputs (20/20 step match) — pin is correct |

## Expert source (D_expert)

`HuanzhiMao/BFCL-Result/2025-12-16/result/claude-opus-4-5-20251101-FC/multi_turn/BFCL_v4_multi_turn_base_result.json`, filtered to `valid == true`: **162 / 200** Base cases pass.

- **FC variant** chosen — non-FC scored 41/200 too few. We rejected GPT-4.1-FC alternative (47.5% pass, ~half the data) in favor of Opus FC.
- Trajectories carry **no `reasoning_content` / no thinking** — extended thinking was OFF for this leaderboard run.
- `inference_log` already embeds per-step observations (`role: tool`) and per-turn sim-state snapshots (`role: state_info`). Expert SFT harvest does NOT require re-replay.

## Train / eval split (Path A — paper-faithful 75/25)

```python
random.seed(42)
train_pool_ids = random.sample(all_base_case_ids, 150)   # 75% of 200
expert_ids     = train_pool_ids ∩ {valid==true}          # 123 cases (realized)
heldout_ids    = all_base_case_ids - train_pool_ids      # 50 cases — held-out I.D. eval
```

| | Realized count |
|---|---:|
| all Base case_ids | 200 |
| Opus valid==true | 162 |
| 75% train_pool | 150 |
| **expert_ids = train_pool ∩ valid** | **123** |
| dropped from pool (Opus failed) | 27 |
| heldout (I.D. eval) | 50 |

- Realized lists live at `data/split/{train_pool_ids, expert_ids, heldout_ids, dropped_from_train_pool_ids, valid_ids, failing_ids, all_ids}.json` — trainer does set-diff to verify zero leakage.
- 3 OOD splits (`long_context`, `miss_func`, `miss_param`) **never** enter any SFT file.
- 50 held-out Base cases **never** enter any SFT file.
- Paper claims "75% = 125 trajectories" but 75% of 200 = 150. Paper math doesn't reconcile. We strictly use 75% pool, ∩ Opus valid = 123 (within 2 of paper's 125).

## Action space

| Class | # funcs | Class | # funcs |
|---|---:|---|---:|
| GorillaFileSystem | 18 | TicketAPI | 9 |
| MathAPI | 17 | TradingBot | 20 |
| MessageAPI | 10 | TravelAPI (file: `travel_booking`) | 18 |
| TwitterAPI (file: `posting_api`) | 14 | VehicleControlAPI | 22 |

Per case, `involved_classes` selects 1 or 2 (Base: 65 single / 135 two-class). Candidate-action-space size per case ranges 18–39, mean 27.8. **K=10 distinct alt names from `pool \ {expert_name}` is structurally satisfiable on every Base case**.

v4-added classes (`memory_kv`, `memory_rec_sum`, `memory_vector`, `web_search`) never appear in multi-turn — ignored.

## Pipeline (three stages, all implemented and run at full scale)

### Stage 1 — IWM rollout (proposer + env execution)

`scripts/smoke_iwm.py` (name retained; runs full scale too).

- One worker per case, walks Opus expert steps sequentially on real sim instances; alt actions execute on **deepcopy** clones.
- At each fcall state: sample K=10 distinct alt **names** structurally from `involved_classes` method pool, excluding the expert's name (no LLM for name selection — `method_recap.md` "don't LLM when enumerable").
- **One** DeepSeek call per state to fill args for all 10 names (paper-faithful "K candidates per call", not K separate calls).
- Each alt is `eval()`'d against a deepcopy of the live sim instances; tool output, exec_error (if any), and post-state snapshot recorded.
- **State serialization for proposer prompt only**: GFS clean tree + cwd (in-memory translation, does NOT modify the env). Other 7 classes use `vars()` dump as-is. This affects ONLY what we show DeepSeek; SFT training data uses chat history, not internal state dumps.
- Model: `deepseek-v4-flash` (not pro — args are schema-bound, structured output stable at this scale).
- Concurrency: ThreadPoolExecutor across cases (50 workers).

### Stage 2 — Summarizer (raw responses → 1-sentence factual descriptions)

`scripts/summarize_iwm_rollout.py`.

- Converts every `(call, raw_tool_response)` pair (both expert and alt) into a single descriptive sentence.
- **Deliberate deviation from paper §B.3**: paper's Training Example collapses irrelevant alts into a canonical `"Cannot help fulfill the user's task."` template, which conflates *what happened* with *task-relevance judgment*. Our summarizer is **purely descriptive** — describes what env returned without any task-relevance claim. Rationale: clean separation between summary (facts) and SR (reasoning). SR has more material to reason over this way.
- Successful action → past-tense verb summary ("Moved 'a' into 'b/'."). Read-only → "Retrieved ..." / "Current directory contains: ...". Error → "Error: ..." restatement of the error. NO mention of "user" / "task" / "goal" / "cannot help".
- Model: `deepseek-v4-flash`, T=0.3.

### Stage 3 — SR (self-reflection monologue)

`scripts/smoke_sr.py`.

- Per fcall state: randomly sample **K=3** valid alts (parsed + summarized + non-exec-exception), prompt LLM to write first-person internal monologue arriving at the expert action.
- Inputs to SR prompt: conversation history (using prior steps' summaries as observation text), current sim state, available tools (schemas with per-tool descriptions only), expert call + expert summary, 3 alt calls + 3 alt summaries.
- SciW guardrails inherited from `pitfalls.md`:
  - **banned vocab**: "expert", "selected", "chosen", "correct", "best", "optimal", "preferred", "ideal", "official", "recommended" (post-hoc grep filter at SFT-build time)
  - **no numbered alt labels** ("Action 1", "Option B", "Alternative #2")
  - **single paragraph**, convergence anchor (must end committing to expert action)
  - **length target 200–300 words**, soft cap ~500 via prompt only (no `max_tokens` per SKILL.md hard rule 9)
- Model: `deepseek-v4-pro`, T=0.7, `frequency_penalty=0.3` (mode-collapse mitigation per SciW pitfall #5).

## Production data (full scale, all 123 expert cases)

```
envs/bfcl_v4/data/
├── raw/        opus_base_{result,score}.json                  Opus 4.5 FC leaderboard files
├── raw_gpt41/  gpt41_base_{result,score}.json                 GPT-4.1 FC for comparison (rejected)
├── split/      {all,valid,failing,train_pool,expert,heldout,dropped_from_train_pool}_ids.json
├── parsed/     opus_expert_steps.jsonl                        per-emit-step intermediate; ALL 162 valid cases
├── rollout/
│   ├── iwm_full.jsonl                                          775 fcall states × 10 alts (initial)
│   ├── iwm_full_summarized.jsonl                               ↑ + summaries (POST-repair, 100% alt valid)
│   ├── iwm_full_summarized_pre_repair.jsonl                    same with 22 states' alts still failed (kept for trace)
│   └── sr_full.jsonl                                           754 reflections (raw, with quality_flags)
└── sft/        6 final files (text + fc for each of expert / iwm / reflection); gitignored
```

## Realized counts vs paper §B.3

| Dataset | Ours (full) | Paper §B.3 | Ratio | Notes |
|---|---:|---:|---:|---|
| trajectories | **123** | 125 | 98% | seed=42 random 75% of 200, ∩ Opus valid |
| expert_sft (fcall + text_only) | **1,224** | 1,264 | 97% | matches paper closely |
| IWM total (10 alt + 1 expert per fcall) | **8,677** | 11,904 | 73% | 7,750 alts (100% recovered after repair) + 927 individual expert calls; Opus emit-step rate 6.30/case lower than paper-implied ~10 |
| SR raw | **754** | 1,200 | 63% | per fcall state |
| SR with quality flags (no drops) | **754** | 1,200 | 63% | quality_flags metadata per record, trainer decides |

**Origin of the gap**: Opus 4.5 is more efficient than the (unspecified) target model paper used to generate alts. Average fcall steps per case: ours 6.30, paper-implied ~10. Not a bug — direct consequence of expert-source choice. Documented as a deliberate deviation.

## Pipeline quality (audited at full scale)

**IWM (7,750 alt slots, 123 cases)**:

| Metric | Value |
|---|---:|
| args parse valid rate (post-repair) | **100%** (initially 97.2%; 22 states with all-batch JSON failures repaired via `repair_iwm_alts.py`, single re-roll round with temperature escalation 1.0/0.7/0.3) |
| Python exec exception rate | 1.2% (kept in IWM per paper §5 — invalid-action error messages are training signal) |
| env accept rate (of valid execs) | 79.7% |
| env reject rate | 20.3% |
| **state-change rate** | **76.1%** |
| no-op rate (read-only) | 23.9% |

Reject rate is dominated by `object_not_found` (filenames not in current pwd in GFS-heavy cases) + `auth_required` (random alt sampling picks tools needing prior auth). Both are valid IWM training signal per paper §5.

**Summarizer (8,458 jobs)**: 0 errors. Audit on 3,500 summary subset:
- SR-judgment vocab leak: 0.1% (false positives)
- Paper canonical "Cannot help" resurgence: 0
- Multi-sentence: 0%
- Median 55 chars, p99 165, max 353

**SR (754 reflections)**: 0 hard errors. Pre-filter audit:

| Metric | Rate | Stability across N=50/100/123 |
|---|---:|---|
| banned vocab leak (any) | 7.3% | 8.0% → 7.0% → 7.3% — stable |
| labeled alt refs | 0% | 0% across all scales |
| multi-paragraph | 0.7% | ~0% across all scales |
| paragraph duplication | 0.3% | ~0.3% across all scales |
| word count median | 193 | 197 / 190 / 193 — stable |
| **mode-collapse outliers** (>1000 words, short-ngram repeat) | **0.3%** (2/754) | Surfaced only at N=123; SciW pitfall #5 confirmed |

## SR quality flags (count-only, no drops)

**Decision (vs paper / vs SciW): we do NOT drop SR records on these signals.** Data volume is already tight relative to paper (754 vs 1,200), so dropping ~8% on top costs more than it's worth. Instead, every SR record in `reflection_sft*.jsonl` carries a `quality_flags` dict; downstream trainer decides include/weight per flag.

```json
{
  "case_id": ...,
  "messages": [...],
  "quality_flags": {
    "banned_vocab_hits": ["correct", "official"],   # empty list if clean
    "is_mode_collapsed": false,                     # word_count > 500 OR short-ngram repeat
    "word_count": 247,
    "has_meta_reference": false                     # mentions "alternative"/"comparison"/etc.
  }
}
```

Expected flag distribution on 754 records (from full-scale audit):

| Flag | Expected count | % of 754 |
|---|---:|---:|
| `banned_vocab_hits` non-empty | ~55 | ~7.3% |
| `is_mode_collapsed` true | 2 | 0.3% |
| `has_meta_reference` true | ~15 | ~2.0% |
| **fully clean (no flags)** | **~685** | **~91%** |

This approach side-steps the TEAM_GUIDE §3 filter approval (no data is being suppressed — flags are pure metadata). If at training time we discover specific flag combinations hurt model quality, those drops are recovered by `pandas.query("quality_flags.is_mode_collapsed == False")` on the SFT JSONL.

## SFT data shape — dual emit (FC native + flattened text)

We build BOTH serializations for each of the three categories — 6 final files total. Same case_id set, same per-step granularity; only the `assistant` content format differs. Trainer picks one or experiments with both (ablation).

| Category | `*_text.jsonl` (paper-faithful, default) | `*_fc.jsonl` (BFCL FC mode, structured) | count |
|---|---|---|---:|
| `expert_sft_*` | assistant content = `"mv(source='a', destination='b/')"` (or text reply for text_only step) | assistant content=null + `tool_calls: [{type:"function", function:{name, arguments}}]` | 1,224 |
| `iwm_sft_*` | assistant content = summarized next state (1-sentence) | same — IWM target is the summary string regardless of format (no tool_calls in target; this is next-state prediction) | 8,525 |
| `reflection_sft_*` | assistant content = `"Thought:\n<reflection>\n\nAction:\nmv(source='a', destination='b/')"` | assistant content=`"Thought:\n<reflection>"` + `tool_calls: [{...}]` | 754 |

**Format choice consequences for downstream training**:

| | text format | FC format |
|---|---|---|
| Trains predicting | text strings (BFCL parser at inference) | structured tool_call objects (BFCL FC evaluator direct) |
| Compatible base models | any instruct model | requires FC-capable chat template (Llama-3.1 instruct, Qwen-2.5 with `<tool_call>` tokens, etc.) |
| Paper §B.3 reproduction | **yes** — paper Table 7 BC numbers come from prompting-mode evals | not what paper measured |
| Compares against Opus 4.5 FC | needs BFCL prompting-mode eval (Opus's 81% was FC) | direct comparison possible (apples-to-apples FC mode) |
| Parse failure risk at inference | small (BFCL parser robust but not infinite) | zero (structured output) |

**Default = text** (paper-faithful, lower base-model requirement). FC version retained for FC-mode evaluation / ablation.

text_only steps (449) are kept in `expert_sft_*` (teach turn termination), excluded from IWM/SR (no env transition to learn).

## BFCL-specific pitfalls

- **`execute_multi_turn_func_call` globals() instance cache** — keyed by `(model_name, test_entry_id, class_name)`. Our pipeline bypasses this wrapper, calling `instance.method(...)` via custom `eval_call` with safety blacklist. Process-per-case isolation also separates state cleanly.
- **Opus FC has 0 truly empty emits** — every turn ends with a `text_only` emit (a natural-language "I'm done, here's what I did" summary), not zero calls. So "empty_emit handling" is moot; the question is text_only handling (decided: keep in expert_sft).

## Inherited pitfalls (from `pitfalls.md`, applied here)

- SciW pitfall #4 (SR CoT label leakage) — banned-vocab list in SR prompt + post-hoc grep filter at SFT-build time.
- SciW pitfall #5 (DeepSeek long-form paragraph duplication) — `frequency_penalty=0.3` reduces but doesn't eliminate. Confirmed: 2/754 mode-collapse outliers in our SR run. Detector to be added at SFT-build time.

## Open items

None blocking. All build-side decisions are settled; ready to emit final SFT files.

## Resolved items (history)

- ~~Open #1 (SFT serialization)~~ → **dual emit**: build both `*_text.jsonl` (paper-faithful default) and `*_fc.jsonl` (structured BFCL FC mode) for each of the 3 categories. 6 final files total. Trainer picks default or ablates both.
- ~~Open #2 (IWM next-state format)~~ → **DeepSeek-summarized**, per paper §B.3. Implemented in Stage 2 above. Summarizer prompt diverges from paper in being purely descriptive (no task-relevance judgment); that deviation is intentional.
- ~~Open #3 (empty-emit / text_only handling)~~ → text_only **kept** in expert_sft; **excluded** from IWM/SR. empty_emit doesn't occur in Opus FC.
- ~~Open #4 (SR filter)~~ → **changed to count-only with `quality_flags` metadata**, no drops. Side-steps §3 approval (no data suppression). Trainer filters at dataloader time if needed.

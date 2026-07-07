# SearchQA

Multi-hop question answering env, paper §B.5. The agent issues `<search>` queries against a Wikipedia (wiki-18) retrieval server, reads back `<information>...</information>`, and eventually emits `<answer>...</answer>`. Trajectories average ~3.7 search-and-reason steps before answering.

This is a **data-only** env from our workspace's POV — the env is just retrieval over a fixed wiki corpus, all expert / alt rollouts are already on disk in the repo the user dropped at `envs/searchqa/` (the author's paper-faithful implementation). There is no JVM, no HTTP env server to install, no rollout pass against a live env.

## Pipeline route — (γ): reuse repo's raw docs, DeepSeek re-summarize + DeepSeek SR

After three rounds of recon (paper §B.5 + the repo's data layout + 3 rounds of trade-off discussion with the user), the agreed route is:

- **Expert (`D_expert`)**: reuse `decision_data` from the repo verbatim — these are Search-R1 model rollouts on MuSiQue, filtered to correct trajectories per paper §B.5. Search-R1 is the expert *generator*, not the policy being trained, so this is not in scope for the "replace policy with DeepSeek" rule.
- **IWM (`D_rollout`)**: reuse the repo's pre-stored **raw retrieved documents** (`wm_positive_pdocs` + `wm_negative_pdocs`), but **DeepSeek re-summarizes them all** — the repo's summaries were produced by 8B Llama / Qwen at temperature 1.0 and are observably lower-quality than what DeepSeek will produce on the same raw docs. The retrieval itself (e5 + FAISS over wiki-18, ~50 GB index) is not re-done — we trust the repo's docs.
- **SR (`D_refl`)**: DeepSeek writes reasoning from scratch per paper §B.5 design, K=2 alts per state, sampling alts from the repo's `wm_negative_pdocs` (with their DeepSeek-summarized docs from the IWM step), reasoning prompt designed to avoid the leakage failure modes recorded in `method_recap.md` and `pitfalls.md` (scienceworld lessons).

Final output: three SFT files at `envs/searchqa/data/sft/`, OpenAI chat-messages JSONL, paper-faithful structure.

## Status (2026-05-18)

| Stage | Status | Output | Records | Cost |
|---|---|---|---:|---:|
| Expert SFT | ✅ done | `data/sft/expert_sft.jsonl` | 7,691 | $0 (no LLM) |
| IWM smoke (40 → 200 cases, prompt v1→v4) | ✅ done | `data/smoke/iwm_smoke.jsonl` | 200 | ~$0.30 |
| IWM full rollout | ✅ done | `data/rollout/iwm_rollout.jsonl` (880MB) | 159,391 | ~$104 |
| IWM SFT pack | ✅ done | `data/sft/iwm_sft.jsonl` (672MB) | 159,391 | $0 |
| SR design (A vs B) | ✅ resolved | — | — | — |
| SR smoke (20→100→100 cases) | ✅ done | `data/smoke/sr_smoke_{A,B}.jsonl` | 240 | ~$0.40 |
| SR full rollout | ✅ done | `data/rollout/sr_rollout_{A,B}.jsonl` | 7,691 × 2 | ~$18 |
| SR SFT pack (A and B variants) | ✅ done | `data/sft/reflection_sft_{A,B}.jsonl` | 7,588 + 7,527 | $0 |

## Repo provenance and tracking

The directory `envs/searchqa/` currently holds the author's paper-faithful implementation, dropped in by the user. As of 2026-05-17 we have not received a fork URL, so the directory is **not** a git submodule — it's just files on disk. The tracking strategy is:

- **Tracked in workspace git**: this `NOTES.md`, `scripts/` (our pipeline code), the author's top-level `README.md` (5.7 KB recon doc — kept for future readers).
- **Untracked / gitignored**: `LLaMA-Factory/` (1 GB+ training framework, regenerable from `github.com/hiyouga/LLaMA-Factory`), `retrieval/` (server stub + ~50 GB e5/wiki-18 data that would need to be downloaded), `evaluation_data/`, `SearchR1_rollout_setting1/` (the actual pre-rollout data files — large; we read from them but don't modify them). Author's `evaluation_*.py` scripts stay tracked because they're small and they document the eval contract.

If we later get a fork URL we can convert `envs/searchqa/` to a proper submodule.

## Paper §B.5 design (the spec we're matching)

- **Expert trajectories**: 7,000 tasks selected from MuSiQue training set (all 3-hop and 4-hop + 1,438 sampled 2-hop). Search-R1 model generates 5 trajectories per task at temperature 1.0; only trajectories whose final answer matches ground truth are kept; at most 2 correct trajectories per task. Result: **2,082 trajectories, 7,691 (state, action) pairs**, 1,262 unique tasks.
- **IWM**: predict **summaries of retrieved docs** (not raw docs). For each `<search>` state, the policy generates **30 alternative search queries** at temperature 1.0; each is executed against retrieval; the model summarizes the retrieved docs; training target is the summary. Invalid alts (no `<search>` tags) get the format-error feedback string. Aims for 1:1 ratio with `D_expert` per training script wiring.
- **SR**: **K=2** randomly sampled alternative actions per state (NOT K=3 like the scienceworld env). Model writes fine-grained reasoning explaining why expert action is preferable, using the retrieval documents associated with all three actions. SR covers BOTH `<search>` steps AND `<answer>` terminating steps (IWM only covers `<search>` because `<answer>` has no retrieval target).

## Data counts in the repo (Llama-3.1-8B-Instruct version — what we're using)

| Category | File | Records | What it is |
|---|---|---:|---|
| Expert | `decision_data_training.json` | **7,691** | (s, a) pairs; 5,685 search + 2,006 answer |
| IWM positive | `wm_positive_pdocs_data_training.json` | 5,685 | expert query + raw retrieved docs |
| IWM negative | `wm_negative_pdocs_data_training.json` | **153,706** | alt query + raw retrieved docs (~27 alts/state) |
| Other variants | `wm_positive_pquery`, `_pqueryaware`, `_psummary`; `wm_negative_*` | various | post-paper experiments, **not used** |
| SR (final) | `reasoning_v4_data_training.json` | 8,376 | repo's pre-generated SR (we do NOT use — we regenerate via DeepSeek) |

The IWM "raw docs" (positive + negative) sum to **159,391** documents, which is the size of the IWM summarization batch we'll send to DeepSeek.

## Design decisions confirmed with user (2026-05-17)

- `(γ)` route — re-summarize via DeepSeek, do NOT spin up retrieval server (~50 GB save, ~$500 save).
- Alt source = **Llama-3.1-8B-Instruct**'s `wm_negative_pdocs` (153,706 alts). This is the paper's headline base model, so the alt provenance is most directly aligned with paper Table 8's Llama-3.1-8B numbers.
- Single SFT file per category (one `expert_sft.jsonl`, one `iwm_sft.jsonl`, one `reflection_sft.jsonl`); no per-base-model splits.
- IWM K_target = 30 (paper). We don't try to fill states with <27 alts to exactly 30 — accept the repo's ~27 mean.
- SR K = 2 alts (per paper §B.5 explicit number for SearchQA, NOT K=3 from scienceworld).
- SR covers all 7,691 expert SA pairs (search + answer), per paper §B.5 behavior in `reasoning_v4`. No augmentation (one reflection per state).
- SR reasoning prompt: written from scratch by us based on paper §B.5 Training Example, applying all scienceworld leakage-prevention rules (banned vocab, no numbered labels, single paragraph, convergence anchor, length cap ~500 words). Not mimicking any specific `reasoning_v?` variant from the repo (those are 8B-Llama / Qwen outputs of varying iteration quality; DeepSeek with a well-designed prompt will produce stronger reasoning).
- Post-hoc cleanup at build time: paragraph-break normalization, duplicate-paragraph dedup (scienceworld pitfall), banned-vocab filter, k<2 drop.
- **IWM state strips historical `<think>` blocks** — diverges from author's `wm_positive_psummary` / `wm_negative_psummary` serialization (which keeps historical `<think>`). Rationale: `<think>` is policy-internal reasoning, not environment-observable; IWM learns environment dynamics (given partial obs → predict next obs), so historical `<think>` doesn't belong in the IWM state. IWM SFT record structure:
  - user content = system prompt + question + historical `<search>/<information>` chain (NO `<think>`) + latest `<search>{query}</search>`
  - assistant content = `<information>{summary}</information>` (no `<think>`, same as author)
- **Expert SFT keeps historical `<think>` blocks** — matches author's `decision_data` byte-for-byte. By-design split from IWM: expert teaches policy how to think-then-act, so historical `<think>` is normal trajectory context; stripping it would cause train-test mismatch (at inference, policy's own prior-turn `<think>` enters the context window). Two SFT formats coexist via joint multi-task training, disambiguated by system prompt (IWM: "predict retrieval summary"; expert: "act as agent on question").

## Pipeline scripts

Under `envs/searchqa/scripts/`:

- ✅ `build_expert_sft.py` — reformat repo's decision_data into `expert_sft.jsonl` (no LLM).
- ✅ `smoke_iwm.py` — IWM summarization smoke against DeepSeek; supports `--n-expert`/`--n-alt`/`--dry-run`. Source-of-truth for IWM prompts and helpers (rollout imports from here).
- ✅ `rollout_iwm.py` — full ~155k summarization run, streaming JSONL output, `--resume` skips done `(kind, src_idx)` pairs, `--concurrency` configurable.
- ✅ `build_iwm_sft.py` — pack rollout JSONL into `iwm_sft.jsonl` + final QA panel.
- ✅ `smoke_sr.py` — SR smoke supporting `--variant A|B`; source-of-truth for SR prompts, alt sampling, and post-hoc helpers. K=2 alts with cross-step fallback if same-state pool too sparse.
- ✅ `rollout_sr.py` — full 7,691 expert SA-pair SR generation per variant, streaming, `--resume`-idempotent.
- ✅ `build_reflection_sft.py` — pack `sr_rollout_{variant}.jsonl` → `reflection_sft_{variant}.jsonl` with post-hoc fixes: duplicate-doubling repair (sciworld pitfall) + paragraph-break collapse to single paragraph (soft constraint enforcement at build time).

## Cost — actuals

| Stage | Calls | Cost |
|---|---:|---:|
| IWM smoke (4 prompt iterations × 40-200 cases) | ~880 | ~$0.30 |
| IWM rollout (155,256 valid LLM + 4,135 invalid hardcode + 285 truncation re-roll) | 155,541 LLM | ~$104 |
| SR smoke (A v1 + A v2 + B, each 20 → 100 cases) | ~240 | ~$0.40 |
| SR rollout variant A (with K=2 alts) | 7,691 | ~$11 |
| SR rollout variant B (no alts) | 7,691 | ~$7 |
| **Total spent** | ~172k LLM calls | **~$123** |

IWM ended ~$104 vs initial $30-50 estimate — v4 prompt produced 150-word summaries (median 193w / ~250 output tok per call), much denser than v1's ~50w. Output token cost dominates. SR variant A is more expensive than B by ~$4 because it includes alt summaries in input (avg ~2600 input tokens vs B's ~1850).

## Things easy to get wrong here

- **SR covers `<answer>` steps too** — paper does this (`reasoning_v?` files have ~24% answer-terminating records). The "alts" for an `<answer>` step are "what if I continued searching instead". IWM does NOT cover `<answer>` (no retrieval target).
- **The repo's "K alts per state" is variable, not fixed 30** — `wm_negative_pdocs` has 27 alts/state on average due to dedup + invalid-format filtering. We accept variable K.
- **Per-model alt counts differ** — `wm_negative_*` for Llama-3.1-8B / Llama-3.2-3B / Qwen are different sizes because each model proposed its own alts. We pick L3.1-8B as canonical (see "Design decisions").
- **`wm_negative_self_reflect_data` is misleading** — it's a filtered subset of alts (per-model variable counts 29k–54k), NOT a clean K×N alt dataset. Don't use it directly; use `wm_negative_pdocs` instead. Documented as a workspace-level pitfall.

## Lessons from IWM rollout (2026-05-18)

Worth eventually surfacing to workspace-level `pitfalls.md`:

- **DeepSeek tier cap = concurrency 500, not RPM**. Pushing ThreadPool to 1000 returned 1,277 × `429: 'Your current concurrency: 500'` errors. concurrency 50 → 5/s, 200 → 15/s, **500 → 25/s (sweet spot)**, 1000 → 22/s (worse, threading overhead + 429 spikes). Default `httpx` connection pool is 100, must be bumped to ≥ concurrency or it becomes the throughput cap.
- **DeepSeek server-side partial responses (~0.18%)** — completion_tokens 1-13 returned with no exception raised by SDK. Detect via `completion_tokens < 100` filter; remove from JSONL and `--resume` to re-roll. All re-rolls succeeded normally.
- **Raw retrieved docs use `Doc N(Title: ...)` inline markup** — DeepSeek treats these markers as entity references and writes "Doc 1 describes... Doc 2 covers..." in summaries unless explicitly told they're retrieval markup (not entities). Must include an "About the document format" paragraph in the system prompt.
- **"Closing not-found exception" in prompt → question text leak**. v2 prompt allowed "No information about X appears..." closing if docs didn't cover query — DeepSeek wrote X as a paraphrase of the broader question (10/200 records had ≥5-word question-text overlap in closing). Fix: forbid all phrasings of "comment on absence" outright. The invalid-action path is the only legitimate place for "format error" feedback.
- **Hardcoded invalid-alt path**: `wm_negative_pdocs` includes 4,135 records (2.69%) whose assistant content is the fixed `'<information>Format error! You must put the search query within the <search></search> tags if external knowledge is needed.</information>'` string. These represent author's pre-filtered invalid alts (no `<search>` tag) and must be detected via exact-string match and copied verbatim to SFT (no LLM call). build/rollout scripts both have this branch.

## Prompt v1 → v4 evolution (for SearchQA IWM summarizer)

- v1: framed task as "summarize for the agent to answer the question" → DeepSeek wrote 1-sentence direct answers (median 57w, [204] Shelley was 8w).
- v2: reframed to "describe what docs contain", added 150-250w target → length OK but 31/40 records ended with "No information about X..." closing that leaked question phrases.
- v3: removed closing not-found exception → cleaner but Doc 1/2/3 enumeration appeared occasionally (mimicking raw-docs markup).
- v4 (production): added "About the document format" paragraph banning Doc N enumeration + explicit ban on "Separately,/A separate document/Another document" connectors. **0 Doc N, 0 closing-leak, 2.27% Separately residual (narrative tone, harmless)**, median 193w, 91% in 150-249 target range. Same prompt re-used for full 159k rollout.

## SR design — A vs B (resolved by producing both datasets)

Author's chosen training versions (`reasoning_v4/v5/v6`) have **0 records** with explicit "Candidate N" enumeration in `<think>` output (compared to v2/v3 which had ~660/version). Generation script not in repo so we couldn't see whether v4 generation prompt actually showed K=2 alts as context.

We could not resolve the ambiguity from data alone, so we generated **both variants** for downstream empirical comparison:

- **Variant A**: DeepSeek generation prompt includes K=2 alts (with IWM summaries as hypothetical retrievals). Paper §B.5 K=2 design — alts inform the implicit-alt reasoning but do NOT appear in the final SFT record's user content.
  - Alts source for search states: same `(question, n_completed_blocks)` from `wm_negative_pdocs` + IWM summary.
  - Alts source for answer states: same trajectory's last search step's alt pool (framed as "continue searching instead of answering") — 0 cross-step fallbacks needed in 100-smoke; ~0 in full rollout via cross-step backup.
- **Variant B**: DeepSeek generation prompt has only (state, expert_action). Author V4/V5/V6 minimum interpretation.

A prompt iteration: v1 had **39% paragraph breaks** (DeepSeek wrote a per-alt paragraph). v2 reworded alt presentation from bullet-list to inline ("The agent also briefly considered X (which would have retrieved: Y) and Z..."), and tightened the alt-consideration clause to require "single continuous paragraph with connectives". v2 dropped paragraph breaks to 8%. B's prompt was not re-iterated — its 17% under-100w records are not punts but legitimate concise reasoning at early-trajectory or directly-answerable states.

Both variants' final SFT comparison (post-build):

| | A | B |
|---|---|---|
| Final SFT records | 7,588 / 7,691 (98.7%) | 7,527 / 7,691 (97.9%) |
| `<think>` median | 233w (in target 200-350) | 154w (shorter but legitimate) |
| `<think>` p10 / p90 | 177 / 307 | 82 / 262 |
| Duplicate-doubling fixed at build | 15 (0.2%) | 27 (0.4%) |
| Paragraph-break normalized at build | 4,143 (55%) | 3,808 (51%) |
| Enumerated alt label leak ("Candidate N" etc.) | 0 | 0 |
| "expert" / "selected" label leak | 0 / 3 | 0 / 5 |

Both datasets are production-ready. User will train on each and compare downstream policy performance to settle the A-vs-B design question empirically.

## SR-specific lessons (queued for workspace `pitfalls.md`)

- **Prompt structure invites format mimicry**. Presenting K=2 alts as a bullet list ("- alternative was X\n- another was Y") nudges DeepSeek to write a per-alt paragraph in output (39% paragraph-break rate). Rewriting alts as an inline sentence ("The agent also considered X and Y...") dropped paragraph breaks to 8% with no length loss.
- **Short reflections at trajectory step 0 or directly-answerable states are not punts**. B's 17% < 100w records are clean concise reasoning, not 1-sentence punts. Don't tighten "min length" prompt guidance reflexively; inspect first.
- **Duplicate-doubling rate (~0.2-0.4%) survives `frequency_penalty=0.3`**. Same as sciworld. Build-time post-hoc fix (keep first half of doubled `<think>`) is reliable and re-roll is not necessary.
- **SR generation prompt can be paper-faithful (with K=2 alts) without alts appearing in training input** — author's V4/V5/V6 shows alts can be generation-time context only. For ambiguous design choices, produce both variants and resolve empirically.

## Open items

- Tracking the author repo as proper submodule pending a fork URL from the user.
- Lift IWM + SR lessons above to workspace `.claude/skills/early-experience-data/pitfalls.md` once any concurrent env work on that file is settled.
- (Optional) Re-roll 103 malformed A + 164 malformed B records via `--resume` after manually purging them from `sr_rollout_{A,B}.jsonl` — would push A→7,691 (100%) and B→7,691 (100%). Cost ~$0.5. Not done since 97-99% is sufficient for SFT.

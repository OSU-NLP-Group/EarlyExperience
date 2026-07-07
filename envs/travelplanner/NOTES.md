# TravelPlanner

Long-horizon travel-planning env from paper §B.7 (osu-nlp-group/TravelPlanner,
Xie et al. 2024, [arXiv:2402.01622](https://arxiv.org/abs/2402.01622)). The
agent fills a multi-day travel plan (transportation / 3 meals / attractions /
accommodation per day) under budget + constraint pressure.

## Status

- **2026-06-30** — Stage C v9 + Stage D v7: v8 trained model still
  underperformed IL on Final Pass. Hand-comparison of v7 vs v8 on
  246 cuisine-constrained meal records uncovered the real problem:
  v8's aggressive shortening (60-120 word target) dropped cuisine
  mention in 14% of records where v7 had it (208/246 → 178/246).
  In several cases v8 didn't just omit cuisine — it produced ACTIVELY
  WRONG reasoning: traj 29 d3 picks Haldiram's saying "doesn't conflict
  with any constraint" when the query explicitly required Italian/
  Mediterranean (Haldiram's serves tea/bakery); traj 19 d1 produced a
  meta-leak ("the decision already recorded shows Madly Bangalee was
  picked"). The shorter target was forcing the LLM to skip critical
  checks AND degrade reasoning quality.

  V9 keeps v8's short-CoT philosophy but adds two structural fixes
  that GUARANTEE the critical checks are present:

  1. `=== CITY SEQUENCE ===` block in `build_prompt_A` for
     transportation states. Computed from plan-so-far: lists visited
     destination cities with entry days, the still-need count, and
     names the final-day return constraint. Forces the reasoning to
     engage with high-level city routing — which was identified as a
     v7→v3 regression in downstream eval (v7 lost 9 city-route tasks
     vs v3).

  2. `=== CUISINE PREFERENCE ===` block in `build_prompt_A` for meal
     states when the query specifies cuisine. Surfaces the cuisine
     list explicitly and the TASK section makes cuisine match/mismatch
     a REQUIRED check (one sentence; explicit forced-mismatch when no
     option satisfies). Eliminates the v8 silent-drop failure mode.

  Word target bumped slightly to A: 100-160 (was 60-120) to absorb
  the now-mandatory checks without losing v8's overall tightness.
  No other data-side changes (wrong_city tag, K=8 stratified, attractions
  in SFT, sleep-city hint, accom-DEST-only all carry from v7).

  Full v9: 1,226 reflections, 200 s, **$1.21**, 37 banned-vocab flags;
  rerun cleared 30/37. Word distribution: median 124, mean 126
  (target 100-160 hit).

  Compression pass (`scripts/compress_sr.py`): single DeepSeek pass
  with explicit preservation directives (cuisine match/mismatch, city
  sequence, min-nights, budget delta, commitment) and explicit
  forbidden phrasings (no "the decision recorded", no "the annotation"
  meta-leaks). 1,226 records compressed in 101 s, $0.28.

  Verification (7 paired comparisons): cuisine mention preserved in
  all paired records that had it in v9-full; city-sequence preserved
  (sometimes paraphrased — e.g. "still need one more destination
  city" → "would break the three-city count"); forced cuisine
  mismatch acknowledgment preserved (Lutyens case); min-nights math
  preserved (7/30/4-night-min rejections + 2-night-min consecutive
  booking math intact).

  Two final SFT datasets shipped side-by-side for downstream A/B:
  - `reflection_sft_full.jsonl` — **1,225 records (5.5 MB)**, median **124 words**
  - `reflection_sft_compressed.jsonl` — **1,226 records (5.1 MB)**, median **50 words**

  `reflection_sft.jsonl` (default) is the FULL version.
  v8 outputs preserved as `sr_rollout_v8.jsonl` / `sr_rerun_v8.jsonl`.

- **2026-06-30** — Stage C v8 + Stage D v6: v7 trained model still lost to
  the expert-only IL baseline on Final Pass (v7: ~22pp vs IL: ~25pp).
  Downstream failure-mode analysis (separate eval, not in this repo):
  v7 lost mainly on (1) `invalid action` — JSON parsed but the
  `value` field was not in the current state's available_actions, so
  the adapter fell back to SKIP_* and triggered Complete-Information
  failures, and (2) `Reasonable City Route` — v7 actually regressed
  vs v3 on city-route by 9 tasks. Strict-JSON parse failures were
  NOT a meaningful loss source (94.7% of non-strict JSON still parsed
  to a valid action).
  Hypothesis: v7's CoT (median 174 words) was diluting the action-JSON
  format signal in the training loss. Action JSON occupies ~5-10% of
  output tokens; longer reasoning means the model spends proportionally
  less capacity on the action format. A second hypothesis: per-step
  micro-constraint enumeration crowded out high-level reasoning like
  city-sequence planning that v3 had room for.
  V8 directly regenerates with much shorter targets (no two-stage
  compression — single-pass is cleaner and roughly the same cost):
  Type A: 60-120 words (was 130-220); Type B/C: 50-90 (was 80-140);
  Type D: 50-80 (was 80-120); Type E: 100-150 (was 120-200). TASK
  sections explicitly tell the model to identify ONE OR TWO factors
  that genuinely drive THIS decision, list the candidate check
  categories (wrong-city / repeat / min-nights / mode-chain / city-
  sequence / mandatory-field), and ignore the rest. `CITY SEQUENCE`
  added as one of the listed checks for transportation states — not
  as a separate prompt block (it's just one consideration among
  several, woven into prose when the destination choice matters).
  No data-side fixes are added or removed; wrong_city tag, K=8
  stratified, attraction states, sleep-city hint, accom-DEST-only
  city all carry over from v7.
  Smoke (30 stratified): word distribution hit the target everywhere
  (overall median 81), recently observed wins: rec#641 transport
  weaves city-sequence into a 75-word decision ("I need a third Texas
  city, Lubbock is a natural next stop"); rec#92 SKIP_TRANSP correctly
  links to accommodation min-nights; rec#1064 accommodation rejects
  each alternative with a specific reason (30-night min / 365-night
  min / wrong city / pet-rule conflict / 3-night min) in 155 words.
  Full v8: 1,226 reflections, 185 s, **$1.14**, output tokens
  167k (v7 was 292k — 43% shorter), 40 banned-vocab flags; rerun
  cleared 33/40. Per-token flag rate roughly matches v7.
  reflection_sft.jsonl = **1,225 records (5.3 MB)** — same record
  count as v7 (1,226 - 1 still-flagged), 10% smaller on disk.
  Word distribution: median 95, 972/1225 in 60-120, 178 in 120-180,
  57 below 60, 18 above 180.
  v7 outputs preserved as `sr_rollout_v7.jsonl` / `sr_rerun_v7.jsonl`.

- **2026-06-29** — Stage C v7 + Stage D v5: SR model trained on v6 data
  underperformed v5 despite v6 closing the wrong_city / mandatory_skip /
  attraction-state coverage gaps the audit had flagged. Hand-reading 30
  v6 records revealed the regression's root cause:
  v6's mandatory 4-section scaffolding (City / Used / Min-nights / Day
  completeness + Decision) plus "(N/A — this step picks X, not Y)"
  filler rows had ballooned median reflection length from 170 → 320
  words. The structure also forced an explicit consecutive-night-count
  ("K") computation for every accommodation step, even when the plan-
  so-far did not actually reveal K — about a third of v6 accommodation
  reflections leaned on hypothetical phrasing ("I'll likely stay 2
  nights here", "feasible if I structure the itinerary that way"), and
  in one observed case (was-rec#132 Abilene 1-night-stay) the model
  computed K wrong, concluded "plan can't proceed", and still emitted
  the gold action — a true reasoning-vs-action contradiction in SFT.
  V7 keeps every v6 data-side improvement (wrong_city tag, K=8 stratified
  alts, attraction states in the SFT) but rolls the prompt rhetoric
  back to v5's natural first-person prose: TASK sections now request
  130–220-word inner monologues that weave in the relevant checks only
  WHEN they bear on the decision; no forced section headers, no N/A
  filler. Type E COMPLETE_PLAN reverts from a forced day-by-day field
  table to a natural prose review (still covers closed-circle,
  no-repeats, mode chain, min-nights, budget — just in conversational
  form). Additionally, the audit-led tightening:
  - `_allowed_cities_today` now takes a `field` argument. For
    accommodation it returns ONLY the destination of today's transport
    (where the agent sleeps tonight), not both endpoints. FROM-city
    accommodation alts on travel days now get `wrong_city` tagged.
  - `build_prompt_A` for accommodation states inserts an explicit
    `=== TONIGHT'S SLEEP CITY ===` block stating "tonight in <city>
    (the agent arrives there via today's transport from <FROM>)". This
    disambiguates late-evening-flight cases (e.g. Dallas→Houston 22:37
    flight where accom is in Houston) where the model's commonsense
    can otherwise contradict the gold (was-rec#1096 in v7-without-fix).
  Smoke (60 stratified + 15 random read by hand): clean natural prose,
  zero "(N/A …)" filler, zero hypothetical "I'll likely / it's feasible
  if I structure" phrasing, the was-1096 evening-flight case now reads
  cleanly. COMPLETE_PLAN reflections even include honest hedging when
  the gold has a soft-constraint miss (e.g. "none of these restaurants
  clearly serve American or French cuisine, which the query requested.
  That's a real problem I can't fix now"). Two iterations: v7.0 hit
  the evening-flight bug at ~6% rate; v7.1 with the accom hint fixes
  it (was-1096 now clean).
  Full v7.1: 1,226 reflections, 206 s, **$1.24** (24% cheaper than v6,
  33% faster), 21 banned-vocab flags; attempt-2 rerun cleared 20/21.
  reflection_sft.jsonl = **1,226 records (5.9 MB)** — same count as v6
  but 15% smaller on disk. Word distribution: median 174, 828/1226 in
  100–200, 320 in 200–300, 51 below 100, 27 above 300.
  v6 outputs preserved as `sr_rollout_v6.jsonl` / `sr_rerun_v6.jsonl`.

- **2026-06-26** — Stage C v6 + Stage D v4: SR v5-trained model still
  underperformed IWM on the *commonsense* axis (Diverse Restaurants
  47 fail vs IWM 12, Min Nights 64 vs 50, Within Current City 22 vs 13,
  Complete Information 48 vs 41) even though hard-constraint metrics
  were now ahead of IWM. Diagnosis: v5 reflections were natural-prose
  and rule-grounded but did **not** uniformly carry the cross-step
  state-tracking the four commonsense metrics demand. A coverage audit
  (`scripts/_audit_alt_coverage.py`) found three concrete data gaps:
  (1) `tag_alt` had NO `wrong_city` class — 57.6% of meal states and
  54.3% of attraction states actually have cross-city alts in the IWM
  pool, but those alts were either filtered out (no wrong tag → drop)
  or kept for the wrong reason (cuisine_violate masked the real city
  problem). (2) K=6 cap in `select_alts` truncated `mandatory_field_skip`
  out of 127 accommodation prompts and `cuisine_violate` out of 46 meal
  prompts. (3) Attraction states (221) had been intentionally excluded
  from `reflection_sft.jsonl` — but attraction is the cleanest signal
  for the city check the model was missing.
  V6 closes all three: adds `wrong_city` tag (uses `_allowed_cities_today`
  derived from start city + today's transport destination); changes
  `select_alts` to stratified K=8 (round 1 picks ≥1 per available
  failure-mode class, round 2 fills by priority); puts attraction states
  back into `reflection_sft`. Independently, every prompt builder is
  rewritten around a mandatory four-check scaffolding (City / Used /
  Min-nights / Day completeness) followed by a Decision paragraph, so
  the reflection EXTRACTS running state from plan-so-far on every step
  instead of pattern-matching the gold action. Type E gains a forced
  day-by-day completeness table with all six fields per day.
  Post-audit on v6: all `pre` = `shown` (0 truncation loss anywhere);
  `wrong_city` now teaches 46–60% of meal/attr/accom states; mandatory
  skip coverage on accom restored from 22.6% → 80.1%.
  Full v6: 1,226 reflections, 310 s, **$1.63**, 89 flagged; attempt-2
  rerun cleaned 81/89. Build keeps 1,225 (1 still-flagged drop).
  Sample audit (20 records): 15/15 carry all four checks, attraction
  states do explicit city-rejection reasoning, accommodation states do
  per-alt min_nights arithmetic, COMPLETE_PLAN reflections produce a
  day-by-day field table. Word count median 320 (target 200–320).
  Final reflection_sft.jsonl = **1,225 records** (7.0 MB) — +190 vs
  v5, includes 221 attraction states. v5 outputs preserved as
  `sr_rollout_v5.jsonl` / `sr_rerun_v5.jsonl`.
- **2026-06-24** — Stage C v5 + Stage D v3: SR trained on v3/v4 data
  evaluated *worse* than IL baseline. Root cause: reflections were citing
  prompt-internal scaffolding that does not exist at inference time —
  rule names (`is_not_absent`, `breaks_mode_chain`), tagged annotations
  (`× INVALID because: repeats_restaurant`), and verbatim section
  headers (`from the "Restaurants used so far" list`). Model trained on
  this learned to write phantom citations.
  V4 first attempt added an `ANTI-FABRICATION GUIDELINES` block and
  pre-filtered alts to only show truly-wrong ones, but the prompt still
  fed those `× INVALID because: <tag>` annotations into the alt list —
  so the model still parroted "the annotation says ... invalid" style.
  V5 rewrites the prompt to remove the leakage at the source:
  alts no longer carry `× INVALID` tags (just `→ would leave spent $X/$Y`);
  `HARD_CONSTRAINTS_BLOCK` rewritten as plain English prose, no
  function-style rule names; new `NO PROMPT-STRUCTURE LEAKAGE` +
  `NO FABRICATED PREFERENCES` block with explicit forbidden phrasings
  and bad-vs-good examples; per-type word targets tightened to
  100-180 (was 200-380). Smoke-40 v5: 39/40 clean. Full v5: 1,035
  reflections, 168 s, **$0.98**, 11 banned-vocab flags (vs ~100+ in v4);
  attempt-2 rerun cleared 10/11. Audit of 30 random v5 records (10
  shown in full): 0 rule-name leaks, 0 annotation references,
  0 section-name quoting, 0 reasoning/action mismatch. Mean reasoning
  170 words, 84% in 100-200 band. v4 outputs preserved as
  `sr_rollout_v4.jsonl` / `sr_rerun_v4.jsonl` for provenance.
  Final reflection_sft.jsonl = **1,035 records** (5.0 MB).
- **2026-05-27** — recon: submodule pinned, conda env up, expert replay
  passes 45/45 trajectories clean against canonical cost formulas. NOTES.md
  drafted.
- **2026-05-28** — Stage B IWM: refinfo (A, 29,767) and anywhere (C, 74,942)
  built. **Both later superseded by Stage B' once the paper author's gym was
  received** — see 2026-06-01 entry.
- **2026-06-23** — Stage C v3 + Stage D v2: complete evaluator audit found
  v1 prompt missed 5 ASR-killing rules (`is_valid_attractions` no-repeats,
  `is_valid_information_in_current_city`, `is_valid_visiting_city_number`,
  **`is_not_absent`** mandatory-fields on stay days, `is_valid_room_type`,
  and accommodation house_rules / max_occupancy semantics in the prompt).
  rollout_sr.py v3 expanded HARD_CONSTRAINTS_BLOCK from 4 to 11 rules,
  enriched history digest with attractions-used + visiting_city_number
  tracker, added accom_lookup (min_nights/room_type/max_occupancy/
  house_rules) and threaded into alt rendering. Smoke-40 verified all 11
  rules surface in reasoning (traj 25 d7 transport explicitly uses **all**
  of: closed-circle, mode chain, mandatory-fields, last-day no-accom,
  visiting_city_number target). Full v3 attempt-1: 1,149 reflections,
  $1.36, 8 min, 131 flagged. Attempt-2 rerun: 94/131 cleaned (72%), 37
  residual mostly adjectival "in the correct city" false positives.
  build_sft.py v2: context-aware filter (`is_real_label_leak` regex)
  distinguishes meta-leaks ("X was picked") from adjectival idioms
  ("correct city"). Final: 1,140 reflection_sft records (9 real-leak
  drops, 32 adjectival kept), 7.5% data growth in size vs v1 due to
  longer reasoning per record. Stage C+D total cost (v1+v3+rebuild):
  **$2.32**. v1 outputs preserved as `sr_rollout_v1.jsonl` /
  `sr_rerun_v1.jsonl` for provenance.
- **2026-06-02** — Stage D SFT build done via `build_sft.py`. Three final
  training-ready files under `data/sft/`:
  - `expert_sft.jsonl` — **1,370 records** (5.1 MB)
  - `iwm_sft.jsonl` — **52,754 records** (247.5 MB)
  - `reflection_sft.jsonl` — **1,148 records** (6.3 MB)
  All in OpenAI chat-messages format. State representation = paper gym's
  `_format_state_sft` (byte-identical to what the trained model sees at
  inference). expert/reflection share `SYS_AGENT` system prompt; IWM uses
  `SYS_WORLD` (predict-next-state task). IWM action compactly rendered
  paper-style (e.g. `SET_TRANSPORTATION (F3573659, $474)`); next-state =
  paper §B.7's budget delta + templated field-effect remark (no LLM cost).
  Reflection assistant content = `reflection_text + "\n\n" + action JSON`.
  221 attraction states intentionally absent from `reflection_sft` (SR
  scope decision: cost=$0, no grounded contrast).
- **2026-06-01** — Stage C SR: DeepSeek V4 Pro generated reflections for
  the 1,149-state SR scope (1,035 contrastive A + 44 SKIP_ACCOMMODATION B +
  24 budget-exhausted SKIP_MEAL C + 2 stay-day SKIP_TRANSPORTATION D + 44
  COMPLETE_PLAN E). Attempt 1: $0.78, 7.3 min, 25 flagged (2.2%, mostly
  "correct" used as geographic adjective vs label-leak ratio ~16:9). Attempt
  2 (rerun on the 25): 24/25 = 96% clean on resampling, $0.018. Final clean
  rate: 1,148/1,149 = 99.91% (the 1 remaining "my chosen action is SKIP"
  meta-leak gets dropped at SFT build). `data/rollout/sr_rollout.jsonl` +
  `sr_rerun.jsonl`. Total Stage C cost: **$0.80**. Next: Stage D — build the
  three final SFT files.
- **2026-06-01** — Stage B' IWM: paper author's gym
  (`gym_travelplanner-main.zip` in `envs/travelplanner-paper/`) installed in
  isolated conda env. Drove it across all 45 training trajectories via
  `rollout_iwm_paper.py`:
  - **52,754 (s,a,s') transitions over 1,370 states** (44 full + traj 41
    truncated to d1-d3).
  - paper gym semantics: transport bounded by ref_info-derived city_list (NOT
    transport-to-anywhere); attraction = ADD with `max_attractions_per_day=1`;
    `COMPLETE_PLAN` terminal action; over-budget alts filtered at
    `_is_action_valid`.
  - `fast_step` (state-equivalent to `env.step` minus reward eval) verified
    bit-identical to `env.step` via assertion on traj 0 and traj 22.
  - traj 41 truncated because its gold annotation places attractions in
    San Angelo/Houston on d4/d5 while `current_city` stays at San Antonio
    (geographically impossible); cutting after d3 preserves the consistent
    portion. No other trajectory has this issue (scanned).
  Old A and C marked **superseded**; legacy scripts moved to
  `scripts/_legacy/`. Next: SR design (needs pre-call gate).

## Project-side characterization

**Action space is enumerable but state-dependent**. At each state the agent
picks one value for one (day, field) slot:
- transportation: from query-bounded set of flights + (optional)
  self-driving + (optional) taxi between the two cities of that leg
- breakfast / lunch / dinner: from restaurants in the day's current city
- attraction: from attractions in the day's current city
- accommodation: from accommodations in the day's current city
- plus the implicit `SKIP_<FIELD>` action (gold plan uses `'-'` to indicate
  intentional skips, e.g. no transportation on rest days)

Per-state admissible cardinality:
- meals / attractions / accommodations ≈ 13–30 per city (from `ref_info`)
- transportation ≈ 1–5 flights + 0–1 self-drive + 0–1 taxi
- Aggregate across 45 queries × ~30 step states ≈ ~5,896 atomic candidates;
  paper's "~70,000 state-transition samples" comes from summing
  field-relevant subsets across all 1,350 expert states (estimate consistent).

**State machine is deterministic and self-contained**. Transition = `dict`
update + budget arithmetic. No JVM, no HTTP, no subprocess. `deepcopy(state)`
is the snapshot — branching at every state for IWM/SR alts costs O(K) per
state with zero env-side coordination. This is the easiest env in the
workspace from a probing perspective.

**Paper §B.7 implements a `gym.Env`**, but the wrapper is for downstream
training/eval. For early-experience data generation we only need three pure
functions (`step`, `serialize`, `enumerate_admissible`); a `gym.Env`
subclass would be ~20 extra lines on top of these and is properly out of
scope here per SKILL hard rule 8.

**Expert source is the official annotated training split** (45 plans on
HuggingFace `osunlp/TravelPlanner`, split=`train`). Same provenance as
the fork's `finetuning_data/std_travelplanner.jsonl`; the HF dataset adds
structured query fields (`org`, `dest`, `days`, `people_number`, `budget`,
`local_constraint`, `level`, `date`) and the canonical `annotated_plan`.

## Scope and limits

- **Dataset scale**: 45 expert trajectories → **1,350 (state, action) pairs**
  total under the (day × 6 fields) state machine (997 SET + 353 SKIP). Even
  15/15/15 split across 3/5/7-day trips → 18/30/42 steps per trip.
  Paper §B.7 reports 1,395 — the 3.3% gap likely comes from venue-splitting
  multi-attraction days (a single attraction-field value can contain up to
  3 venues joined by `;`). User-confirmed 2026-05-27 to keep the single-step
  semantics (1 SET_ATTRACTION per day even with multiple venues), trading
  full paper-numeric fidelity for cleaner state-machine alignment.
- **K for IWM**: **exhaustive — no cap** (per paper §B.7 "perform exhaustive
  augmentation by executing ALL available valid actions at each state").
  **Actual: 29,767 transitions** over ref_info (1,350 expert + 28,417 alt),
  not the paper's "over 70,000". Decision A (user-confirmed 2026-05-28): use
  ref_info, not global DB — see [IWM scale: ref_info vs global DB] below for
  the verification that settles the 70k discrepancy.
- **K for SR**: **up to 30** (per paper §B.7 "explore up to 30 alternative
  valid actions"). States with fewer than 30 admissible take all of them.
- **Eval out of scope** but design preserves compatibility: the `step` /
  `serialize` / `enumerate_admissible` functions we write are exactly what
  a downstream eval driver would wrap — see [Eval contract for downstream
  trainer] below.
- **Reflection length**: TBD at SR-design time per SKILL hard rule 9. Working
  default: 200–350 word target with soft cap ~450 (TravelPlanner action
  choices are constrained — budget compare, distance compare, cuisine
  match — so reasoning chains run shorter than scienceworld's procedural
  reflections). Confirm before SR smoke.

## Recon findings (resolved from code + replay)

- **Submodule contents**: pure mirror of `osu-nlp-group/TravelPlanner` `main`
  HEAD (`e52c87f`). The fork has **zero local commits**. No `gym.Env`
  wrapper — paper §B.7's gym was implemented by Bo Liu but never open-sourced
  in this repo. Available components useful to us:
  - `database/{train,validation,test}_ref_info.jsonl` — per-query candidate
    inventories (flights / restaurants / attractions / accommodations /
    self-driving / taxi) as clean JSON dicts.
  - `finetuning_data/std_travelplanner.jsonl` — 45 records of the gold plans
    in (system, user, assistant) message format (LLaMA-Factory finetuning
    layout from the original TravelPlanner SFT release).
  - `evaluation/{commonsense,hard}_constraint.py` — official constraint
    evaluators with `get_total_cost()` as the canonical cost formula.
  - `tools/{flights,restaurants,accommodations,attractions,googleDistanceMatrix}/apis.py`
    — pandas wrappers around the global database CSVs. **The CSVs themselves
    are NOT in the repo** (Google Drive download per upstream README). We
    don't need them — all prices we need are already in per-query `ref_info`.

- **HuggingFace dataset alignment**: `osunlp/TravelPlanner` (config=`train`,
  split=`train`, 45 records) is index-aligned with
  `database/train_ref_info.jsonl`. Verified by comparing record 0
  (St. Petersburg → Rockford, flight F3573659, $474, $1,700 budget) which
  is precisely the example used in paper §B.7. HF record provides clean
  structured fields; fork's ref_info provides clean JSON dicts for prices.

- **Cost formulas are byte-identical to canonical**. Spot-checked against
  submodule's `evaluation/hard_constraint.py:get_total_cost()`:
  - flight: `Price × people_number`
  - self-driving: `cost × ceil(people / 5)`
  - taxi: `cost × ceil(people / 4)`
  - meal (breakfast/lunch/dinner): `Average Cost × people_number`
  - accommodation: `price × ceil(people / maximum_occupancy)`, charged on
    each day the field is SET (i.e., per-night summation across the stay).
  Verified by replaying paper §B.7's example (St. Petersburg → Rockford
  3-day): step-0 cost $474, after-step-0 remaining $1226, final total
  $1,608 ≤ $1,700 ✓ — matches paper's narrated "(28% used)" exactly.

- **Replay budget compliance per level** (45 plans, single people: people=1
  cases on easy, multi-person mostly on medium/hard):
  | level   | within-budget | over-budget |
  |---------|---------------|-------------|
  | easy    | 9/15 (60%)    | 6/15        |
  | medium  | 7/15 (47%)    | 8/15        |
  | hard    | 8/15 (53%)    | 7/15        |
  | total   | 24/45 (53%)   | 21/45       |

  **The over-budget plans are a property of the dataset, not a replay bug**:
  paper §3 Table 9 reports the BC-trained model's Hard Constraint Macro at
  46.7% — gold plans themselves only satisfy hard constraints ~50% of the
  time. Smallest over: $10 (idx 44); largest: $2,145 (idx 35, a 5-day
  3-person multi-city hard trip with three $2,166 taxi legs).

- **City naming**: cross-city queries (e.g., `dest="Texas"` with
  `visiting_city_number=3`) use city names with state suffix in the gold
  plan, e.g. `"Boise(Idaho)"`. The submodule's
  `utils/func.get_valid_name_city` handles the same suffix via
  `extract_before_parenthesis`. Inlined the same regex in our `_split_name_city`
  rather than importing from the submodule (the submodule's imports also
  trigger `sys.path` and `os.chdir` side effects in
  `evaluation/hard_constraint.py` that pollute caller scope — best avoided).

## Setup (reproducible commands)

```bash
# 1) Submodule pinned at e52c87f (HEAD of fork's main as of 2026-05-27)
git submodule update --init envs/travelplanner/TravelPlanner

# 2) Conda env (lightweight: pandas for ref tables, openai for later LLM calls,
#    datasets for HuggingFace data load)
conda create -n travelplanner-ee python=3.10 -y
PYTHONNOUSERSITE=1 /home/ulss/miniconda3/envs/travelplanner-ee/bin/python \
    -m pip install pandas openai datasets huggingface_hub

# 3) Replay all 45 expert plans (pure Python, no LLM, no env-server)
PYTHONNOUSERSITE=1 /home/ulss/miniconda3/envs/travelplanner-ee/bin/python \
    envs/travelplanner/scripts/replay_expert.py
# → produces data/replay/{replay_full.jsonl, replay_summary.json}
```

Installed versions (2026-05-27): `pandas 2.3.3`, `openai 2.38.0`,
`datasets 4.4.1` (with `huggingface_hub` as transitive dep).
`PYTHONNOUSERSITE=1` avoids the user-site leak pattern seen in tau-bench
recon — system-wide `huggingface_hub`/`transformers` versions misalign with
this env's deps.

## Data

### Raw input (regenerable, gitignored)
- `data/replay/replay_full.jsonl` — 1,350 step records, one per (traj, step).
  Each line: `{traj_idx, step_idx, day, field, state_before, action,
  state_after}`. `state_*` = `{spent, cursor, done, plan}`; `action` =
  `{action_type, day, field, value, cost}`.
- `data/replay/replay_summary.json` — per-trajectory rollups (final spent,
  within-budget bool, step counts) plus aggregate counters.

### HuggingFace cache (regenerable, gitignored)
- `~/.cache/huggingface/datasets/osunlp___travelplanner/train/...` — auto
  downloaded by `datasets.load_dataset('osunlp/TravelPlanner', 'train')` on
  first run. ~6 MB.

### IWM rollout (Stage B', built 2026-06-01, gitignored)

Authoritative dataset, generated by driving the paper author's gym (Bo Liu's
`gym_travelplanner-main`, extracted under `envs/travelplanner-paper/`). Single
file, grouped one line per expert state:

```
{traj_idx, step_idx, day, field,
 state_before: {spent, current_day, current_field, field_idx,
                attraction_count, is_done, current_plan},
 budget, n_transitions,
 transitions: [{is_expert, action, spent_after}, ...]}
```

- **`data/rollout/iwm_rollout_paper.jsonl`** (10.9 MB) — **52,754 transitions
  across 1,370 states** (1,370 expert + 51,384 alt).
- Per-state avg: transportation 7.1, breakfast 60.3, lunch 56.7, dinner 55.1,
  attraction 40.8, accommodation 18.6, complete 1.0.
- traj 41 contributes only its first 18 states (d1-d3); other 44 traj run
  full + COMPLETE_PLAN.
- Reproduce:
  ```bash
  PYTHONNOUSERSITE=1 TRAVELPLANNER_DB=envs/travelplanner/global_db \
    /home/ulss/miniconda3/envs/travelplanner-paper/bin/python \
    envs/travelplanner/scripts/rollout_iwm_paper.py
  ```
  ~70 s wall on a single core. Use `--smoke` to run only traj 0+22 with
  `env.step` vs `fast_step` equivalence assertion (state bit-equal for 18
  steps each).

### Superseded variants (NOT used in final SFT)
The two earlier variants are kept in NOTES for provenance but their data
files have been removed:
- ~~`iwm_rollout_refinfo.jsonl` (A, 29,767)~~ — ref_info-only, missed paper
  gym's both-cities-on-travel-day meal enumeration.
- ~~`iwm_rollout_anywhere.jsonl` (C, 74,942)~~ — transport-to-anywhere; the
  paper's own SR/IWM examples in §B.7 show only `{gold flight, SKIP}` at
  transport states, contradicting transport-to-anywhere.
Legacy scripts: `scripts/_legacy/{admissible,rollout_iwm,transport_anywhere}.py`.

### Global DB (gitignored, needed only for variant C)
- `envs/travelplanner/global_db/` (294 MB) — the full TravelPlanner database
  (flights 305 MB CSV, attractions/restaurants/accommodations ~0.5 MB each,
  googleDistanceMatrix/distance.csv). Outside the submodule, gitignored.
- Reproduce: `pip install gdown && python -m gdown
  1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE -O /tmp/tp_database.zip && unzip
  /tmp/tp_database.zip -d /tmp/tp_db && cp -r /tmp/tp_db/database/*
  envs/travelplanner/global_db/`. (Drive link from upstream README.)
- Only `rollout_iwm.py --mode anywhere` reads this. Variant A and the whole
  SR/expert pipeline use only ref_info and never touch the global DB.

### Final SFT output (Stage D, not produced yet)
- `data/sft/expert_sft.jsonl` — 1,350 (state, expert_action) records
- `data/sft/iwm_sft_refinfo.jsonl` — A variant, 29,767 (state, action,
  next_state) records
- `data/sft/iwm_sft_anywhere.jsonl` — C variant, 74,942 records
- `data/sft/reflection_sft.jsonl` — 1,350 (state, reflection, expert_action)
  (SR built on ref_info alts; not affected by the A/C transportation split)

## IWM scale: ref_info vs global DB (verified 2026-05-28)

The paper §B.7 reports "over 70,000" IWM transitions; our exhaustive
enumeration over ref_info gives **29,767**. Investigated the 2.4× gap by
downloading the global DB (Google Drive, 59 MB zip → flights 305 MB CSV +
attractions/restaurants/accommodations ~0.5 MB each) and re-counting:

- **Global DB exhaustive = 29,601** — essentially identical to ref_info's
  29,767. For *visited* cities, ref_info candidate counts equal the global
  DB (Rockford 27=27 restaurants, 20=20 attractions). The attraction cap at
  20/city is a **global-DB property** (collection-time top-20), not ref_info
  truncation. So meals/attractions/accommodations are NOT the 70k source.
- **The only path to ~70k is transportation enumerating flights to ANY
  destination** (not just the gold route). The flights table has 3.8 M rows;
  counting all departing flights per travel-leg = 23,492 transportation
  transitions alone (vs 792 for gold-route-only), e.g. Las Vegas has 471
  departing flights on one date. global meals(21k) + attraction(4.6k) +
  accom(3.4k) + transport-to-anywhere(23.5k) ≈ 52k, +self-drive/taxi-to-
  anywhere ≈ 70k.
- **This contradicts the paper's own statement** that the action space is
  "based on available data from reference information" (which holds only the
  gold-route flights). The paper §B.7 is internally inconsistent on this
  point; we cannot reverse-engineer their exact procedure.

**Decision (user-confirmed 2026-05-28): build BOTH A and C as separate
datasets; user trains on each and compares downstream.**

- **A (refinfo, 29,767)** — transportation drawn only from ref_info (the
  gold-route flights + self-drive + taxi). Matches sole-planning inference:
  the eval feeds the agent only `reference_information`, verified
  byte-identical to our ref_info (HF `reference_information` == fork
  `database/train_ref_info.jsonl`). No train/eval distribution shift.
- **C (anywhere, 74,942)** — transportation enumerates every mode (flight /
  self-drive / taxi) to every reachable destination from the leg origin,
  via the global DB. Reproduces the paper's "over 70,000" (we verified the
  70k IS exactly transport-to-anywhere across all 3 modes; the precise
  74,942 is below the rough 78k estimate because multi-day drives — 'day'
  in duration — are excluded per apis.py). transportation is 61% of C and is
  mostly "travel to a city you're not visiting", which never appears at
  sole-planning inference and is highly redundant signal. Built anyway so
  the user can empirically test whether paper-scale IWM data trains better.

Self-drive/taxi cost formulas (exact, from apis.py:run_for_evaluation):
self-driving base = `int(distance_km * 0.05)`, taxi base = `int(distance_km)`,
then × ceil(people/5) and × ceil(people/4) respectively. Verified on the
Kansas City→Pensacola leg (1,433 km → self-drive $71, matches gold).

## Approved decisions

### Pure-function design, no gym.Env (user-confirmed 2026-05-27)

We **do not** implement a `gym.Env` subclass. The three pure functions
(`step`, `serialize`, `enumerate_admissible`) are sufficient for data
generation. A downstream trainer/evaluator that needs a gym interface
can wrap these three functions in ~20 lines.

Rationale: SKILL hard rule 8 forbids training/eval code; gym wrapper's
only purpose is to provide the OpenAI-Gym API for training loops, which
is downstream. Our SFT data shape (state-string → action-JSON) is
identical regardless.

### Expert source = HuggingFace osunlp/TravelPlanner (user-confirmed 2026-05-27)

`D_expert` = the 45 `annotated_plan`s in the HF train split, fully
decomposed into 1,350 (state, action) pairs by the (day × 6 fields)
state machine.

NOT regenerating expert with an LLM — these are the gold annotated plans
by the original paper authors (per method_recap.md "Find expert
trajectories before generating them" → option 1 wins).

### IWM = paper-exhaustive (user-confirmed 2026-05-27)

At every expert state, enumerate the full field-relevant admissible set
from per-query `ref_info`, step the deterministic state machine on each,
emit `(state, alt_action, next_state)` for every triple. No K cap.

Expected scale: ~70,000 transitions before any filtering (matching
paper's reported number).

### SR K up to 30 (user-confirmed 2026-05-27)

Per paper §B.7: up to 30 alternative valid actions per state. States with
`|admissible \ {expert}| < 30` contribute fewer alts — no padding.

### No-filter default

Per SKILL hard rule 5, default is no filters on `data/sft/`. The 21
over-budget gold trajectories are **kept in full** (an over-budget plan
still produces valid (s, a) pairs for IL and valid (s, a, s') for IWM —
the env response "spent now exceeds budget" is itself supervision signal
under the early-experience reward-free framing).

### Cost extraction from per-query ref_info, not global DB

We compute prices by lookup into the per-query `ref_info` (already
present in submodule `database/train_ref_info.jsonl`), not by importing
the canonical `evaluation/hard_constraint.py:get_total_cost()`. Reasons:
1. The canonical function depends on the global CSV database (not bundled
   in the fork — separate Google Drive download).
2. `ref_info` is exactly the candidate set the agent sees at inference
   time; using its prices keeps train/eval distributions aligned.
3. Cost formulas are byte-identical (verified spot-check above), so the
   numeric output matches.

### `(State)` city-name normalization

Multi-city queries embed state names in city values, e.g.
`"Apna Punjabi Zayka, Boise(Idaho)"`. Our `_split_name_city` strips the
`(...)` suffix when comparing against `ref_info`'s `City` / `city` field,
matching submodule `utils/func.extract_before_parenthesis` behavior.

## Eval contract for downstream trainer (not built here)

Per SKILL hard rule 8 we do not write the eval driver, but document the
interface so downstream consumers can wrap our three pure functions
without reverse engineering them. The driver loop is:

```python
from scripts.replay_expert import (
    TPState, step, hf_record_to_query)
# (serialize / enumerate_admissible will be added during Stage B.)

for hf_record in load_dataset("osunlp/TravelPlanner", "validation"):
    query = hf_record_to_query(hf_record)
    state = TPState.init(query)
    while not state.done:
        # 1) render state to the same string format the SFT data used
        action_json_str = trained_model.generate(serialize(state))
        # 2) parse the model's action JSON; on parse-fail, treat as SKIP
        action = json.loads(action_json_str)
        # 3) deterministic state transition
        state = step(state, action)
    # 4) extract the final plan (state.plan) and submit to the official
    #    evaluator: evaluation/eval.py with commonsense_constraint +
    #    hard_constraint
```

Compatibility guarantees we own:
- `serialize(state)` output format will be **frozen across all three SFT
  files** (consistent user-content rendering per SKILL output-layout rule).
- Action JSON schema: `{action_type, day, field, value, cost}` with
  `action_type` in `{SET_TRANSPORTATION, SET_BREAKFAST, SET_ATTRACTION,
  SET_LUNCH, SET_DINNER, SET_ACCOMMODATION, SKIP_<field>}`.
- `value` strings are byte-identical to ref_info entries' `Name`/`NAME`
  (with City suffix as in gold plans), so the official evaluator's
  `str.contains(re.escape(name))` matching keeps working.

## Things easy to get wrong here

- **Submodule has no gym wrapper**. `tools/planner/env.py:ReactEnv` is a
  scratchpad env for React/Reflexion agents in the original benchmark, NOT
  the paper §B.7 gym (which was never released). Don't import `ReactEnv`
  thinking it does what paper §B.7 describes — it doesn't.
- **Global database CSVs are not in the fork** (Google Drive link in
  upstream README). For data generation we don't need them; if a future
  Stage requires `evaluation/hard_constraint.py:get_total_cost()` directly,
  download the CSV bundle and place it at `TravelPlanner/database/{flights,
  restaurants, accommodations, attractions, googleDistanceMatrix}/`.
- **`evaluation/hard_constraint.py` does `os.chdir` on import** (line 17).
  Never `import` it from a long-lived process. Run via subprocess only,
  or re-implement the formulas (we did the latter — see "Cost extraction"
  decision).
- **City-name `(State)` suffix matters in multi-city queries**. Same-name
  cities across states (e.g. "Boise" appears in Idaho and Oklahoma) require
  the parenthesized state for disambiguation. Strip on lookup, never
  before display.
- **Many gold plans are over-budget** (47%). This is the dataset, not the
  replay — don't add a "drop over-budget trajectories" filter without going
  through the §3 filter-approval protocol (and even then it would drop
  half the training data).
- **`annotated_plan` is a stringified `[query_meta_dict, days_list]`**, NOT
  a bare days list. Use `ast.literal_eval(...)[1]` to extract the days.
  The trailing `{}, {}, {}, {}` fillers in days_list pad to 7 elements
  regardless of trip length — strip them with `[d for d in days if d]`.
- **HF dataset `local_constraint` and `date` fields are stringified Python
  literals** (not native JSON types). `ast.literal_eval` them on load.

## Inherited pitfalls from other envs

- **`pitfalls.md` "LLM reflection generators leak supervision labels"** —
  applies to SR prompt design when we get there. Same banned-vocab + grep
  defense as scienceworld / tau-bench / textcraft.
- **`pitfalls.md` "Long-form LLM generations occasionally duplicate"** —
  applies to SR; `frequency_penalty=0.3` + post-hoc dedup at SFT-build time
  is the established workspace recipe.
- **`pitfalls.md` "JVM- or process-backed envs have no native state
  save/restore"** does NOT apply here — TravelPlanner is pure-Python dict
  state, deepcopy is free, alt probing is O(K) per state with no overhead.
  This is one of the rare workspace envs where K=70k IWM exhaustion is
  cheap.

## Pipeline scripts (in execution order)

- `scripts/replay_expert.py` — Stage A. Reads HF train split + submodule
  ref_info; replays 45 gold plans through `step` state machine; writes
  per-step JSONL + summary. **DONE 2026-05-27** — 45/45 clean replay,
  24/45 within budget (matches dataset, not a bug). Also exports the pure
  functions (`TPState`, `step`, `derive_action`, `hf_record_to_query/plan`)
  reused downstream.
- `scripts/_legacy/admissible.py` — Stage B (A variant), **superseded**.
- `scripts/_legacy/rollout_iwm.py` — Stage B (A + C variants), **superseded**.
- `scripts/_legacy/transport_anywhere.py` — Stage B (C variant), **superseded**.
- `scripts/rollout_iwm_paper.py` — Stage B'. Drives the paper author's gym
  (`travelplanner.envs.travel_planner_sole_planning_env.TravelPlannerEnv`)
  with API singleton caching + `fast_step` (state-equivalent to `env.step`
  minus reward eval) + traj 41 truncation + `env.step` ↔ `fast_step` bit-
  equivalence assertion in smoke. **DONE 2026-06-01** — 52,754 transitions.

To be added in Stage C / D:
- `scripts/rollout_sr.py` — DeepSeek V4 Pro reflection generator (gated)
- `scripts/build_{expert,iwm,reflection}_sft.py` — final SFT writers

## Open questions for implementation / smoke

These will be answered explicitly here (and re-confirmed with the user
where they affect LLM cost) before the relevant Stage runs:

1. **State string format for SFT user content**. Paper §B.7 example shows:
   ```
   Total Days: D, Initial Budget: $B, Spent: $S, Remaining: $R
   Day d: <field>: PENDING|<value>, ...
   Next action required: Plan day d <field>
   Available: <admissible options>
   ```
   Need to confirm: should `Available:` enumerate full admissible (could be
   30+ lines) or just a hint? Paper's example shows the full list. Default:
   follow paper. Verify at Stage B smoke.

2. **Reflection prompt extensions**. Standard METHOD.md §4.3 template plus
   TravelPlanner-specific guideline ("consider budget remaining, minimum
   night stays, restaurant repetition, round-trip completeness"). To draft
   at SR design time, gated by pre-call rule.

3. **Reflection length target**. Working default 200–350 words soft cap;
   confirm before SR smoke.

4. **Expert SFT CoT**. Annotated plans have **no `Thought:` reasoning** —
   they're pure JSON plans. Decision: `expert_sft.jsonl` assistant = raw
   action JSON (no CoT). Structurally different from reflection_sft (which
   has CoT + action), but consistent with what the expert source actually
   contains. Same pattern as appworld (which also has no expert CoT).

5. **IWM sub-sampling at SFT-build time**. Exhaustive rollout will produce
   ~70k transitions. Whether to use all of them in `iwm_sft.jsonl` or
   sub-sample for class balance (e.g., cap per-(state,field) at 50) is a
   Stage D decision after we see the actual distribution.

## Upstream

- **Fork URL**: `https://github.com/UlyssesXC/TravelPlanner` (mirror of
  `https://github.com/OSU-NLP-Group/TravelPlanner`)
- **Submodule path**: `envs/travelplanner/TravelPlanner/`
- **Pinned branch**: `main`
- **Pinned commit**: `e52c87f4ac348a3410c46dc3553c519db5ec5e23`
  ("Merge pull request #53 from zlu/update-gitignore-v2", upstream's HEAD
  at recon time). Zero local commits in fork.

The upstream is a stable archival benchmark (paper published 2024; data
not updated since); pinning at HEAD is safe.

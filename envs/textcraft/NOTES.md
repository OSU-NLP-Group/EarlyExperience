# TextCraft

## Project-side characterization

Text-based Minecraft-style crafting game (originally from ADaPT,
https://github.com/archiki/ADaPT, served via the AgentGym framework as an
HTTP service). At each reset, the agent is given a goal item and a small set
of crafting recipes (target recipe + ≤10 distractors), and must reach the
goal via `get` (gather base items) and `craft` (combine ingredients per
recipe) actions.

The action grammar has 3 verbs (`get N <item>`, `craft <output> using
<ingredients>`, `inventory`) but the per-state admissible action set is
**combinatorial in the recipes given at reset**: typically 5–30 try-able
actions per state (1× `inventory`, one `get` per base item appearing in the
recipe list, one `craft` per recipe whose ingredients you already hold).
**Admissible actions are entirely client-derivable** from `(commands_list,
inventory)` — no server-side endpoint is needed, in contrast to ScienceWorld
which requires `/admissible_actions`.

State observations are **very short** structured text (`"Got 8 terracotta"`,
`"Crafted 1 minecraft:light_gray_dye"`, `"Inventory: ..."`, or a 1-line error
like `"Could not find oak planks"`). IWM next-state prediction can target
the raw next state directly; no summarizer is needed.

Reward is sparse 0/1: `reward=1, terminated=True` only when the agent
produces the goal item via either `get` (if base) or `craft` (if not). All
other steps return `reward=0`.

Expert trajectories come from AgentGym's **AgentTraj-L** dataset on
HuggingFace (`AgentGym/AgentTraj-L`), TextCraft split `textcraft_train.json`
(374 trajectories, 2,992 (s, a) pairs). Conversation format is ShareGPT-style
identical to the SciWorld split: 4-turn handshake (system prompt → "OK..."
acknowledgement → first human turn with `Crafting commands:` + `Goal:` →
first gpt `Thought:/Action:` turn), then alternating
`human (Instruction:<env response>)` ↔ `gpt (Thought:\n... \n\nAction:\n...)`.

**Not in the paper.** Paper Appendix B covers 8 envs (ALFWorld / WebShop /
BFCLv3 / Tau-Bench / SearchQA / ScienceWorld / TravelPlanner / WebArena);
TextCraft is not one of them. All per-env hyperparameters (K, sampling
strategy, length target, filter rules, CoT-preservation choice) are decided
**locally** following the workspace conventions, not borrowed from a paper
recipe.

## Scope and limits

- **Dataset scale**: 374 expert trajectories, **2,992** (state, action) pairs
  from AgentTraj-L's `textcraft_train.json`. 1–19 action turns per
  trajectory; median ≈ 6–7 actions. **All 374 retained** (no replay-based
  filter — user-decided 2026-05-25, see "Approved decisions").
- **Replay pass-rate**: **371 / 374 = 99.2 %** reach `(done=True,
  reward=1)` against the unmodified `agentenv-textcraft` shipped with our
  pinned AgentGym fork. The 3 failures share one root cause (gold-nugget
  version drift — see "Recon findings" below). Replay is kept as a
  diagnostic artifact in `data/replay/`; no trajectories are dropped on
  its basis.
- **K for IWM**: **K = 5** (user-confirmed 2026-05-25). For each expert
  state, sample K non-expert actions *from the client-derived admissible
  list* and step the env on each. Total alt rollouts ≈ 2,992 × 5 ≈
  14,960 (modulo states whose `|admissible \ {expert}|` is < 5).
- **K for SR**: **K = 3** (user-confirmed 2026-05-25).
- **SR alternative sampling strategy**: **open** — see "Open questions for
  implementation/smoke" below. Three viable candidates: (a) LLM-propose with
  admissible refill (sciworld-style), (b) admissible-random shared with IWM
  (cheaper), (c) hybrid. Decision deferred until smoke shows which produces
  the most informative `(s, a_alt, s_alt)` triples.
- **One-shot example in prompt**: not applicable — TextCraft is not in the
  paper, and AgentGym's bundled `TextCraftEnvClient.conversation_start`
  provides only the action menu + ReAct format instructions. We follow
  AgentGym's bundled prompt unmodified.

## Rough guide

1. **`D_expert`**: all **374** trajectories (2,992 (state, action) pairs).
   No pass-rate filter applied per user decision 2026-05-25 — `replay_full`
   is kept only as a diagnostic artifact.
2. **IWM data**: at each expert state, derive admissible from
   `(commands_list, inventory)` client-side, draw **K = 5** non-expert
   actions uniformly at random, step the env on each, plus the expert
   action.
3. **SR data**: pipeline to be finalized at smoke (proposer choice is the
   open question). **K = 3** alternatives per state. Reflection prompting
   follows METHOD.md §4.3 template with the two leak-suppression
   deviations baked in across this workspace (no `"expert"` labels, no
   `"Action 1"` labels — see method_recap.md and the SR section of
   sciworld NOTES).

The HTTP-service architecture means env rollout is RPC-based, not
in-process — but unlike sciworld, **each TextCraft env_id is a pure-Python
gym instance with no JVM**, so concurrency is bounded by Python thread/process
overhead, not by JVM/process count. Server-side already handles concurrent
ids via `threading.Lock`. Smoke will measure real throughput.

## Things easy to get wrong here

- **Treating action-verb count as action-space size.** There are 3 verbs but
  per-state admissible actions number 5–30. IWM/SR alt sampling must enumerate
  this space, not pick from {get, craft, inventory}.

- **Forgetting that `get` only works on base items.** Calling
  `get <craftable_item>` always returns `"Could not find <item>"` — this is an
  informative env response (teaches the base-vs-craftable distinction), but if
  not aware of this, an alt-action sampler might pick it as "valid" then be
  confused why every alt rollout fails.

- **Using `item_id "textcraft_N"` as the env's `data_idx`.** Verified false:
  the AgentTraj-L item_id numbering does NOT correspond to the env's
  data_idx-based goal-selection. Map by goal name instead (see "Recon
  findings" below; the map is bijective).

- **Trusting the recorded `commands_list`.** Even when goal items match, the
  distractor recipes in the env's reset response differ from what AgentTraj-L
  recorded (typically 8/22 differ for terracotta-class goals, similar for
  others). The **goal-path recipes are always present** (`create_recipe_set`
  always includes the full recipe tree for the goal), which is why replay
  pass-rate is 99.2 % — but our SFT records should use the **env's current
  commands_list**, not the recorded one. This matches sciworld's policy
  "Trajectory adapts to env, not the reverse" (downstream user installs
  unmodified AgentGym + agentenv-textcraft and reproduces env-canonical
  output).

- **Server concurrency limits we haven't measured yet.** Replay used 8
  threads to a single uvicorn process and finished 374 traj in 12 s — looks
  fine at that scale. Higher-concurrency stages (IWM rollout with K alts ×
  thousands of states) may need re-measurement; if a bottleneck shows up, run
  multiple `textcraft` server instances on different ports.

## Recon findings (resolved from code + replay)

- **Env package version**: `agentenv-textcraft 0.0.1` (no PyPI release;
  installed editable from our submodule). The crafting tree is built from
  `agentenv_textcraft/recipes/*.json` (860 recipe files) shipped inside the
  package. **Version drift caveat**: AgentTraj-L was generated against an
  older state of this `recipes/` directory; in particular, `gold_nugget` was
  apparently a base item then and is craftable now (`is_craftable=True`,
  appears in `itemid_recipes`). Same class of drift as sciworld's
  AgentTraj-L vs current env, just at the recipe-graph level instead of the
  string-rewrite level.

- **`item_id ↔ data_idx` mapping**: empirically **not** identity. Built by:
  ```
  for i, (item_id, _) in enumerate(sorted(tree.item_recipes_min_depth(1),
                                          key=lambda x: x[1])):
      name = item_id.replace("minecraft:", "").replace("_", " ")
      goal_to_idx.setdefault(name, i)
  ```
  This gives a **bijective** map: 374 traj → 374 distinct goal names → 374
  distinct `data_idx ∈ [0, 543]`. Verified: env's 544-goal universe has 0
  duplicate names, traj-side has 0 unmatched goals, 0 collisions (no two
  traj share a goal).

- **Replay infrastructure** (`envs/textcraft/scripts/replay_textcraft.py`):
  per traj, POST `/create` to get a fresh env_id, POST `/reset` with the
  mapped `data_idx`, step expert actions one by one, capture
  observation/reward/done per step. 8 concurrent threads, ~12 s for 374
  traj. Output: `data/replay/replay_full.jsonl` (one row per traj with full
  step trace), `data/replay/replay_summary.json` (aggregate counts).

- **Replay pass-rate breakdown**:
  | outcome                                       | count | pct    |
  |-----------------------------------------------|-------|--------|
  | `done=True, reward=1` (success)              | 371   | 99.2 % |
  | actions exhausted, never `done`              | 3     | 0.8 %  |

  All 3 failures: expert assumes `get N gold_nugget` succeeds (it doesn't in
  the current env's recipes/), which cascades into the gold-ingot/golden-X
  craft also failing.
  - `textcraft_43` (goal: golden carrot)
  - `textcraft_108` (goal: glistering melon slice)
  - `textcraft_288` (goal: golden apple)

- **Admissible enumeration approach**: client-side, no submodule patch.
  Pseudo-code (will be wrapped in `envs/textcraft/scripts/` during
  implementation):
  ```python
  def enumerate_admissible(commands_list, inventory):
      inputs, outputs = parse_recipe_inputs_and_outputs(commands_list)
      base_items = inputs - outputs
      adm = ["inventory"]
      for b in base_items: adm.append(f"get {N} {b}")  # N from recipe needs
      for recipe in commands_list:
          if ingredients_satisfied(recipe, inventory):
              adm.append(recipe.replace("craft ", "", 1).strip())  # the craft string itself
      return adm
  ```
  Per-state size empirically 5–30 actions. **Decision rule for "base item":**
  any item that appears as a recipe INPUT in `commands_list` but never as a
  recipe OUTPUT in `commands_list`. This is local to the given recipe set;
  matches the env's behavior because `is_craftable` checks the global tree
  which always includes whatever's in `commands_list`.

## Setup (reproducible commands)

```bash
# env server (pure Python, no JVM)
conda create -n agentenv-textcraft python=3.9 -y
conda run -n agentenv-textcraft --no-capture-output python -m pip install -e \
    envs/textcraft/agentgym/agentenv-textcraft/
# launch (port 36011 — picked to avoid conflict with sciworld's 36010)
cd envs/textcraft/agentgym/agentenv-textcraft && \
    conda run -n agentenv-textcraft --no-capture-output \
    textcraft --host 127.0.0.1 --port 36011
```

Installed versions (as of 2026-05-25):
- `agentenv-textcraft` 0.0.1 (editable from submodule)
- `fastapi` 0.128.8 · `uvicorn` 0.39.0 · `gymnasium` 1.1.1
- `transformers` 4.57.6 (dependency from setup.py; not actually used at runtime)

The pipeline-side Python env (DeepSeek SDK, rollout/reflection generation)
is **not** created yet; will be set up during implementation phase. The
existing `agentenv-textcraft` env can run replay/probe scripts because they
only need `requests` and the local `agentenv_textcraft` package.

## Data

### Raw input
- `envs/textcraft/data/raw/textcraft_train.json` — symlink to HF cache for
  `AgentGym/AgentTraj-L`'s TextCraft split.
  - 374 trajectories, 2,992 (state, action) pairs, ~2 MB.
  - Reproduce: `huggingface_hub.hf_hub_download(repo_id="AgentGym/AgentTraj-L",
    filename="textcraft_train.json", repo_type="dataset")`.

### Replay against env
- `envs/textcraft/data/replay/replay_full.jsonl` — produced by
  `envs/textcraft/scripts/replay_textcraft.py`.
- One line per trajectory; each row has `item_id`, mapped `data_idx`,
  `goal`, full step trace `[{action, observation, reward, done}, ...]`,
  `final_done`, `final_reward`, `failure_reason`.
- `envs/textcraft/data/replay/replay_summary.json` — aggregate counts.
- Everything under `envs/textcraft/data/` is gitignored; regenerate via
  the scripts. The HF symlink in `raw/` is also gitignored.

## SR data (built)

### Rollout
`envs/textcraft/scripts/rollout_sr.py` calls DeepSeek V4 Pro once per
expert state to produce a self-reflection CoT. Inputs joined per
(item_id, step) from raw `textcraft_train.json` (prior-step Thoughts),
`replay_full.jsonl` (replayed env responses), and `iwm_rollout.jsonl`
(K=3 reused alts deterministically taken as the first 3 by alt_idx).

Prompt structure (see `rollout_sr.py:SR_SYSTEM_PROMPT`):
- System: explicitly frames the chosen action as ALREADY-DECIDED and the
  outcomes (for both expert and alts) as forward-looking projections, not
  past events. This was iterated through smoke after round-1 LLM
  misinterpreted "Result of this action: ..." as a past failure and
  diverged toward an alt (1/22 strict divergence in round 1 → 0/22 in
  round 2 → 0/186 in expanded smoke).
- System: bans the workspace-wide leak vocabulary ("expert" / "selected" /
  "chosen" / "right action" / etc.) and numbered alt labels.
- System: target ~250 words, soft cap 350. No `max_tokens` API limit.
- User: task obs → prior Thought/Action/Obs history → current obs →
  "The agent's chosen action: <expert>" (NOT "Expert action") with
  forward-projected outcome → alts with their forward-projected outcomes.

API config: `model="deepseek-v4-pro"`, `temperature=0.7`,
`frequency_penalty=0.3`, `thinking={"type":"disabled"}`. Concurrency: 64
HTTP threads.

Smoke iteration:
- Round 1 (22 calls, $0.0107): 1 real divergence on TRIAL action; prompt
  reframed.
- Round 2 (22 calls, $0.0128): 0 divergence; convergence anchor passed.
- Round 3 expanded (186 calls including 5 longest traj, $0.1217): 0 real
  divergence; 9 substring-flagged false positives (manually cleared).

Full-scale run on 2026-05-25: **2,986 calls, 362 s wall-clock** (8.24
calls/s on 64 workers), **$1.96 actual cost**, 4.07 M input + 0.79 M
output tokens. Doubled-response auto-dedup observed: 1 (cleanly recovered).

Per-expert-state class distribution:
| class             | count | %     |
|-------------------|-------|-------|
| PROGRESS_craft    | 1,010 | 33.8% |
| PROGRESS_get      | 739   | 24.7% |
| TRIAL_inv         | 559   | 18.7% |
| TRIAL_get_fail    | 387   | 13.0% |
| TRIAL_craft_fail  | 285   | 9.5%  |
| malformed         | 6     | 0.2%  |

K_actual (alts per state, inherited from IWM rollout): K=3 for 2,624
states (87.9%), K=2 for 236, K=1 for 126.

Reflection length: median 217 words, max 448 words, 6 records (0.2%)
exceed 350 cap.

Output: `envs/textcraft/data/rollout/sr_rollout.jsonl` (32.4 MB).

### reflection_sft.jsonl

`envs/textcraft/scripts/build_reflection_sft.py` joins sr_rollout with
raw conversation thoughts and replayed env responses, renders one
single-turn chat-messages record per kept state. Build-time quality
filters (applied here, not at rollout, so sr_rollout.jsonl remains the
unfiltered raw):

| filter                  | dropped | notes                                              |
|-------------------------|---------|----------------------------------------------------|
| malformed expert action | 6       | first token ∉ {get, craft, inventory} — AgentTraj-L natural-language noise (e.g. "Look for ...", "pass", "I need ..."). User-decided 2026-05-25: drop only from reflection_sft, retain in expert_sft. |
| banned-vocab leak       | 5       | reflections containing "expert"/"selected action"/"chosen action"/etc. — DeepSeek echoed the prompt's "The agent's chosen action: ..." label in ~0.2% of outputs. |
| numbered alt labels     | 0       | (no leakage of "Action 1"/"Alternative 2") |
| 1 record overlaps both filters; total unique dropped = 11 (0.37%). |

**Format** (single-turn per state):
- `system`: AgentGym REACT prompt (same as `expert_sft.jsonl` and
  `iwm_sft.jsonl`).
- `user`: `"Instruction:\\n<task obs>\\n\\n"` + flat rendering of prior
  expert Thought/Action/Obs (same as iwm_sft history).
- `assistant`: `"Thought:\\n<reflection>\\n\\nAction:\\n<expert_action>"`
  — drop-in compatible with `expert_sft.jsonl`'s assistant content
  format, so the trainer can mix the two cleanly.

**Counts**: **2,975 records, 11.8 MB**.

Why single-turn (matching iwm_sft, not expert_sft's multi-turn): each
reflection is independently grounded in its own (state, expert, alts)
tuple. The trainer treats each as an independent SFT example, with loss
only on the assistant content (Thought + Action). Same reasoning as
sciworld reflection_sft.

## IWM data (built)

### Rollout
`envs/textcraft/scripts/rollout_iwm.py` walks each of 374 trajectories. At
every expert state `s_i`:
1. Query inventory via `/step inventory` (doesn't modify env state).
2. Compute `admissible(s_i)` via `admissible.AdmissibleEnumerator`
   (client-side, uses env's actual `CraftingTree`).
3. Sample up to K=5 alts uniformly from `admissible \ {expert_action_i}`
   with seed `md5(f"{item_id}/{step}")[:16]`.
4. For each alt: `/step alt`, capture next_state; then `/reset(data_idx)
   + replay expert[0..i-1]` to restore state for the next alt.
5. After all alts: `/step expert_action_i` to advance to `s_{i+1}`.

Run details (full 374 traj): 8 HTTP threads, **211 s wall-clock**. Output:
`envs/textcraft/data/rollout/iwm_rollout.jsonl` (2.7 MB).
- 374 `initial` records (env's reset observation + commands_list per traj).
- 13,214 `alt` records.

K_actual distribution (per expert state):
| K_actual | states | %    |
|----------|--------|------|
| 5        | 2,299  | 77.0 |
| 4        | 146    | 4.9  |
| 3        | 179    | 6.0  |
| 2        | 236    | 7.9  |
| 1        | 126    | 4.2  |
| total    | 2,986  | 100  |

23% shortfall expected and accepted (simple tasks like `data_idx=0` have
only 2 admissible actions at s_0).

Alt verb mix: 83.5% `get`, 10.2% `inventory`, 6.3% `craft`.
Alt env-response mix: 83.5% `Got X` (success get), 10.2% `Inventory:` (inspect),
6.3% `Crafted X` (success craft). **Zero invalid env responses**, validating
that `admissible.py` perfectly matches env's accept logic. The invalid-action
training signal is preserved separately via expert trajectories (experts
naturally try invalid `get <craftable>` and receive `"Could not find X"`).

### iwm_sft.jsonl
`envs/textcraft/scripts/build_iwm_sft.py` joins raw textcraft_train.json
(for Thoughts in prior-step history) + replay_full.jsonl (for replayed env
responses) + iwm_rollout.jsonl (for alts), and emits single-turn IWM
records.

**Format** (one record per IWM triple):
- `system`: AgentGym REACT system prompt (same as `expert_sft.jsonl`).
- `user`: `"Instruction:\\n<task obs>\\n\\n"` + prior expert `"Thought:\\n...\\n\\nAction:\\n..."` verbatim + `"\\n\\nInstruction:\\n<env obs>\\n\\n..."` history + `"\\n\\nAction:\\n<action_being_probed>"`.
- `assistant`: env's response (single line — TextCraft observations are short).

**Why single-turn**: same rationale as sciworld's iwm_sft — the IWM target
(next-state) is logically the env's response, which in multi-turn chat
would be a `user` turn but SFT trainers compute loss only on `assistant`.
Single-turn (system, user, assistant) puts the next-state as `assistant`
content, unambiguously training the IWM objective.

**Counts**: 2,986 expert IWM records + 13,214 alt records = **16,200 total**.
File size: 46.1 MB.

History rendering: prior-step env responses use REPLAYED observations
(current env's actual behavior, from replay_full.jsonl), not AgentTraj-L's
historical recordings. For the 371 replay-passing traj this is identical
to the recorded data. For the 3 gold-nugget-drift traj, history reflects
what current env returns (e.g., `"Could not find gold nugget"`), which may
be inconsistent with the expert's Thought text — accepted as a small noise
penalty in exchange for keeping the full 374 traj coverage.

## Approved decisions

### K values (user-confirmed 2026-05-25)

- **IWM K = 5**: at each expert state, sample 5 non-expert actions uniformly
  from `admissible(s_i) \ {expert}` (and step the env on each, plus on the
  expert). States whose admissible set has fewer than 5 non-expert options
  contribute fewer alts; no fallback (see "IWM alt sampling" below).
- **SR K = 3**: 3 alternative actions per state, sampling strategy TBD at
  smoke (see open question 1 below).

### SR alt sampling: reuse IWM alts (user-confirmed 2026-05-25)

For each expert state, the K=3 SR alternatives are drawn deterministically
from that state's iwm_rollout alt records (K=5 sampled). Selection is the
first 3 alts by their `alt_idx` (already deterministic via the IWM seed).
**No LLM proposer for SR.**

Rationale:
- TextCraft's action space is small and recipe-bounded; a policy
  proposer would mostly collapse to admissible anyway.
- Reusing IWM alts gives perfect (s_i, alt, s_i^alt) consistency across
  iwm_sft and reflection_sft — same state, same alt, same next-state.
- Zero additional LLM cost on the proposer side.
- Reflection generator (the one mandatory LLM call per state) still runs
  and goes through the pre-call gate.

K-shortfall propagation: states whose `K_actual` in iwm_rollout was < 3
(126 + 236 = 362 states with K_actual ∈ {1, 2}) get fewer than 3 alts in
SR too. Accepted on the same principle as IWM.

### SR reflection length target: 200–350 words soft cap (user-confirmed 2026-05-25)

Prompt-soft-capped at "around 250 words; do not exceed 350"; **no
`max_tokens` API cap**, per SKILL.md hard rule 9 (hard cap truncates
mid-sentence and produces worse data than overlong-but-complete reflection).
Tighter than sciworld's 300–400 because TextCraft state representation is
shorter (1-line obs) and reasoning over a fixed recipe list is less
discursive than sciworld's procedural-lab tasks.

### IWM alt sampling: uniform random from admissible (user-confirmed 2026-05-25)

No LLM. At each expert state, the alt set is sampled uniformly without
replacement from `admissible(s_i) \ {expert_action_i}`, where
`admissible(s_i)` is computed by `admissible.AdmissibleEnumerator` from
`(commands_list, inventory)` — same admissibility predicates the env's
`step()` uses internally.

Rationale:
- Matches the paper's default for enumerable-action envs (ALFWorld §B.1,
  ScienceWorld §B.6).
- The two earlier concerns (LLM proposing invalid-get / random craft hitting
  "missing ingredient") are both already covered:
  - Invalid-get signal (`"Could not find X"`) is present in `expert_sft.jsonl`
    itself — 374 expert traj contain natural exploration like `get light gray
    terracotta` / `get gold_nugget` that env rejects.
  - "Missing ingredient" failures are filtered out by `admissible.enumerate`'s
    inventory-satisfaction check on craft candidates → IWM alts never include
    them.
- Zero LLM cost; deterministic given a per-state seed.

Determinism: `seed = md5(f"{item_id}/{step}")[:16]` per state, identical to
sciworld's pattern, so reruns produce identical alt sets.

K-shortfall handling: when `|admissible(s_i) \ {expert}| < K=5`, accept
`K_actual` alts instead of padding with off-task fillers. Simple tasks
(`data_idx=0` has only 2 admissible actions at s_0) naturally fall in this
regime; the price is tiny.

### No replay-based filter on D_expert (user-confirmed 2026-05-25)

`D_expert = all 374 trajectories` from `textcraft_train.json`. The 3
gold-nugget-version-drift trajectories (`textcraft_43`, `_108`, `_288`)
are **retained** despite failing replay. `data/replay/replay_full.jsonl`
remains a diagnostic-only artifact, not a filter input.

Rationale: user prefers maximum (s, a) coverage; the 3 failing trajectories
contribute valid intermediate states even if their final craft fails. (Note:
when these states are used for IWM/SR rollout, the `gold_nugget`-related
expert actions will return `"Could not find gold nugget"` from the env —
those env responses themselves are still valid IWM training signal per
METHOD.md §3.)

### CoT in expert SFT (user-confirmed 2026-05-25)

`expert_sft.jsonl`'s assistant content keeps AgentTraj-L's `Thought:` text
alongside the action. Matches sciworld's decision and keeps `expert_sft`
structurally comparable to `reflection_sft` (both end with `Thought +
Action`), which makes any downstream training mix cleaner.

## Open questions for implementation / smoke

These are recorded so they don't get silently auto-resolved during
implementation. Each one should be answered explicitly in NOTES.md (with
user confirmation, where it affects an LLM call) before the relevant
pipeline stage runs.

1. **`textcraft-ee` branch push**: branch exists locally only. Submodule
   pin in `.gitmodules` declares `branch = textcraft-ee`, but the commit
   hash currently points at `main`'s tip (`c3b300f`). The submodule still
   resolves correctly because git tracks commits, not branch names —
   `submodule update --remote` will fail until the branch is pushed
   upstream. We don't need it pushed for any work in recon/smoke phases;
   need to push only when/if we actually patch the submodule.

## Upstream

- **Submodule** (we own this fork):
  - Fork URL: `https://github.com/UlyssesXC/AgentGym`
  - Submodule path: `envs/textcraft/agentgym/`
  - Pinned branch: `textcraft-ee` (forked from `main` at recon time;
    branch exists locally only as of 2026-05-25, push deferred)
  - Pinned commit: `c3b300f` (= `main`'s tip "Merge pull request #53 from
    MCU-UAV/main", 2025-09-11). No patches against upstream yet.

The underlying `agentenv-textcraft` package is installed editable from the
submodule (`pip install -e envs/textcraft/agentgym/agentenv-textcraft/`).
No separate PyPI package.

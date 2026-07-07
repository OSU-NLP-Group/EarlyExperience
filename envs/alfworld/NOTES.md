# ALFWorld — Env Notes

> Status: **recon complete, env runs end-to-end.** Pipeline code not yet written
> to the workspace standard. This env is a **special case**: an EE implementation
> was previously built directly on a `verl-agent` branch, *not* following the
> `early-experience-data` skill layout. We borrow that work's ideas (gold-path
> experts, all-admissible-action probing) but do **not** adopt its file/format
> conventions wholesale. Per user direction the existing training-JSONL format
> (`instruction`/`input`/`output`, LLaMA-Factory style) is **kept as-is** — we do
> not convert to the skill's OpenAI-`messages` 3-file layout.

---

## 1. Where the code lives

- Working code is **not** under `envs/alfworld/`. It lives under
  `envs/alfworld-paper/verl-agent-ee` — a **symlink** to
  `/mnt/data/xiangchao/verl-agent-ee`.
- That repo is the upstream **verl-agent** (branch `master`, HEAD `b7966f4`)
  with the EE work added on top as **untracked / working-tree-modified files**
  (it was never committed as a submodule on a fork).
- All of *our* new pipeline code goes in `envs/alfworld-paper/` *outside* the
  repo (e.g. `envs/alfworld-paper/scripts/`), importing `agent_system` from the
  symlinked repo via `sys.path`.
- **OPEN DECISION (needs user):** whether `/mnt/data/xiangchao/verl-agent-ee`
  should be turned into a pinned submodule/fork, and how to handle the existing
  in-tree edits to the env (see §6). For now it is used as a library via the
  symlink.
- Final data output location (per user): **`envs/alfworld-paper/`** (e.g.
  `envs/alfworld-paper/data/`).

## 2. Environment setup (reproducible)

- Conda env: **`verl-agent-alfworld`** (Python 3.12), pre-existing on this machine.
- ALFWorld data already downloaded at `~/.cache/alfworld/json_2.1.1/`
  (`train`, `valid_seen` [242 games], `valid_train`, `valid_unseen`).
- Required env var: `export ALFWORLD_DATA=$HOME/.cache/alfworld`
  (the config `config_tw.yaml` interpolates `$ALFWORLD_DATA`).
- Key versions: `alfworld 0.4.2`, `textworld 1.7.0`, `torch 2.11.0+cu130`,
  `gymnasium 0.29.1`.
- **Setup fix applied:** `verl-agent-alfworld` was missing `torchvision`, which
  `agent_system/.../alfworld/envs.py` imports at module top (only used on the
  Thor/image path we never hit). Installed with:
  `python -m pip install --no-deps torchvision==0.26.0`
  (`--no-deps` to avoid disturbing the cu130 torch; **must** use `python -m pip`
  — the env's bare `pip` resolved to `/usr/bin/pip` and installed a cp310 wheel
  into `~/.local`).

### End-to-end smoke (hard-rule-3 gate — PASSED)

- `envs/alfworld-paper/scripts/smoke_env.py` builds the env manager, resets,
  and steps admissible actions. Reset/step/observation/admissible/reward/done
  plumbing all work. (It does not reach `done` because it intentionally spams a
  no-op action; mechanics are confirmed.)
- Run with: `conda activate verl-agent-alfworld && export ALFWORLD_DATA=$HOME/.cache/alfworld && python scripts/smoke_env.py`.

## 3. Method mapping (how EE maps onto ALFWorld)

- **Expert source = ALFWorld TextWorld gold walkthrough.** This is the env's own
  optimal path (matches the skill's "find experts before generating" — no LLM
  needed). Tasks: 6 ALFRED types (pick&place, examine-in-light, clean, heat,
  cool, pick-two).
- **State `s_i` representation:** the existing multi-turn prompt format
  (`"You are an expert agent... Your task is to: <goal>. Prior to this step you
  have taken N steps. Below are the most recent observations and actions: [...]"`).
  **Kept per user direction.** Same serialization across expert / IWM / SR.
- **Alternative actions = enumerable.** At each `s_i` the env exposes a finite
  `admissible_commands` list; the expert action is one item. We probe by
  executing *the other admissible commands* — **no LLM proposer needed**.
- **K is not fixed** (per user): every admissible alternative at a state is
  probed; **all states/actions go into IWM data.**
- **Action encoding** (existing convention, kept):
  `<think>...</think><action>go to desk 1</action>`.

## 4. Data inventory (existing, in the repo)

| Artifact | What | Verdict |
|---|---|---|
| `alfworld_action_rollout_data_trial/` (1,462 tasks, 21,031 step JSONs) | per-step: `input_prompt` (s_i), `expert_action`, `expert_result.observation` (s_{i+1}), `other_actions[]` (alt action + observation + reward/done) | **expert part valid; alternatives CONTAMINATED — see §5** |
| `alfworld_action_rollout.json` (716 MB) | aggregated version of the above | same caveat |
| `alfworld_expert_decision_data_w_step_info_for_singleturn_train.json` (21,031 records) | expert SFT: `instruction`=s_i, `output`=expert action | **valid** (no observation used) |
| `alfworld_rollout.py` | the all-admissible parallel rollout script | has the sync bug (§5) |
| `alfworld_action_data*.py` | builders that produce the expert decision data | valid for expert; no IWM/SR builder exists |

- **No IWM next-state builder** and **no SR / reflection code** exist in this
  repo. (The paper's WM/SR data was built in a sibling `../LLaMA-Factory/` tree
  that is **not present** on this machine.) **No LLM-call code exists** anywhere
  in the EE scripts.

## 5. ⚠️ CRITICAL FINDING — alternative next-states are contaminated

`alfworld_rollout.py` probes alternatives by dispatching the expert action +
every other admissible command across `group_n` **parallel** envs in one
`env.step(...)`. But the parallel envs are each reset to their **own random
game** and the intended re-synchronization (`sync_environment_states`,
reset-to-gamefile + replay) does **not** actually align them. Result: each
alternative is executed against a **different game's state**, not the true `s_i`.

Quantified over a random sample of the stored data:

- **Expert branch (env 0): 0% "Nothing happens"** (377/377 `go to` actions land
  correctly) → `expert_result.observation` (`s_{i+1}`) is **clean and trustworthy**.
- **Alternative branch: 50–70% "Nothing happens"** on `go to` actions, at
  **every** step depth (not just step 1) → alternatives ran in the wrong game.
- `other_actions[].input_prompt_for_next_step` references a **different task** in
  ~98–100% of cases — independent confirmation of cross-env contamination.

**Consequence for EE data:**
- `expert_sft` (s_i → a_i): **OK.**
- IWM **expert** triple (s_i, a_i, s_{i+1}): **OK.**
- IWM/SR **alternative** triples (s_i, a_i^j, s_i^j): **UNUSABLE** as stored —
  regenerated by the corrected rollout below.

## 5b. Corrected rollout — `scripts/corrected_rollout.py` (DONE)

Regenerates clean `(s_i, a_i^j, s_i^j)` triples. Design, in order of how it was
arrived at:

- **Per-game env, expert walkthrough from ALFWorld's `AlfredExpert`.** Each game
  gets its own single-game TextWorld env (so reset is deterministic). The expert
  wrapper yields the gold walkthrough as clean admissible-command strings
  (the leading no-op `look` is used to prime the stateful expert, not recorded).
  `policy_commands` was rejected (raw PDDL entity names, not clean commands).
- **Probe via `os.fork()` snapshot — the key idea.** ALFWorld has no state
  snapshot and `reset()` costs ~1s of **pure CPU** (profiled: `wall == cpu`, so
  tmpfs / faster disk does **nothing**). Instead of reset+replay per alternative
  (O(K·N²) resets), a **synchronous** (forkable, in-process) probe env walks the
  expert path monotonically (1 reset + N steps, **no replay**); at each `s_i` we
  `fork()` a child per alternative — the child steps the alternative and reports
  its next-state, the parent stays at `s_i` (copy-on-write; verified the parent
  is never contaminated).
- **Parallel fork is essential.** Serial fork-wait = 156 ms/alt (the child's
  copy-on-write page faults during `step` don't overlap); forking a whole batch
  concurrently drops it to **~9 ms/alt** (faults spread across cores). Tunable
  via `FORK_BATCH` (default 24). The async expert-walk env is closed **before**
  forking (never fork a process holding async/multiprocessing resources).

**Validation:** alternative `go to` → "Nothing happens" = **0%** across 10k+
probed alternatives over multiple splits (was 50–70%); 0 probe errors.

**Performance:** ~77 s / 60 train games at `--workers 8` `FORK_BATCH=12`
(≈64–96 concurrent procs — the measured sweet spot; this is a virtualized box,
180 vCPUs, concurrency >~90 hits CPU contention). Extrapolated full splits:
valid_seen ~3 min, original-subset(1462) ~31 min, full train(3553) ~1.3 h.

**Output** mirrors the original per-step JSON schema (so the existing builders
still apply), but `other_actions[].observation` is now trustworthy. Lands in
`data/rollout/<split>/<task>/<trial>/stepN.json` + `walkthrough.json` + `.done`
(skip-marker for resume). Run:
`ALFWORLD_DATA=$HOME/.cache/alfworld FORK_BATCH=12 python scripts/corrected_rollout.py --split train --workers 8 --output_dir data/rollout/train`

This realizes the recorded pitfall's lesson (*"process-backed envs have no
native save/restore"*) with a fork-based snapshot rather than the O(K·N²)
reset+replay originally sketched.

### 5c. Expert source: gold walkthrough, NOT the live handcoded expert (corrected)

A first full run used ALFWorld's **live handcoded `AlfredExpert`** to generate the
walkthrough. That was a mistake (caught by the user): the handcoded expert is a
brittle rule-based heuristic — it searches receptacles by string-matching and uses
`random.choice`, and it **fails on ~21% of games** (won rate 79.2%, wrong receptacle,
picks unrelated objects, even emits `help`). This is the documented ALFWorld issue
[#87 "Following expert actions doesn't always lead to finishing the task"](https://github.com/alfworld/alfworld/issues/87);
the paper's DAgger numbers (~61% seen) corroborate. `config_tw.yaml` uses `handcoded`
for real-time DAgger *speed*, trading reliability ("the planner is very slow").

**Fix:** use each game's **built-in gold walkthrough** — the `walkthrough` field in
`game.tw-pddl`, an offline-precomputed correct solution. Verified byte-identical to
the original verl-agent `alfworld_expert_decision_data` action sequences (so the
ORIGINAL data was always gold-walkthrough-based; the handcoded detour was mine).
`policy_commands` was rejected (raw PDDL entity names, not clean commands).
`corrected_rollout.py` now reads `walkthrough` directly (a sync env walks it; no
expert wrapper). This is the same expert source the paper/original used.

**Full train run result (DONE, gold walkthrough):** 3553 games, **21,335 expert
steps** (≈ paper's 21,031 and the original's 21,031 — strong confirmation), won
rate **100%**, next-state correctness **0% dirty** (0/67,356 sampled), ~680 MB?,
in `data/rollout/train/`. Wall ~139 min. Per-step JSON holds `input_prompt` (s_i),
`expert_action`, `expert_result.observation` (s_{i+1}), and `other_actions[]` =
ALL admissible alternatives with clean next-states (a superset; builders subsample).
**won rate is 100%, so the earlier won-filter question is moot — no filter needed.**

## 5d. Data-building plan (per paper §B.1)

Paper §B.1 is the authority for K/sampling (do NOT copy other envs' choices):
- **D_expert**: 21,031 SA pairs (we have 21,335). expert_sft = (s_i → a_i).
- **IWM**: per state sample **8 non-expert** actions (admissible, uniform, no
  replacement) + the expert action = **9** → ~21k×9 ≈ 189,279 triples. Our rollout
  stored ALL admissible alts (superset); `build_iwm` subsamples 8 per state.
- **SR**: **K=3** alternatives, selected for **action-type diversity** (user's
  choice). Paper proposes them with the policy model (temp 1.0), invalid ones
  replaced by uniform-random admissible. We have no policy model; pure uniform
  random skews to `go to` (most admissibles are navigation). Instead we round-robin
  one alt per distinct action_type — substantive non-nav (take/open/use/examine) >
  nav (go to) > meta (inventory/look), random within type, seeded per state.
  Measured: 3.00/3 distinct action types, 0% all-go-to (vs uniform). This is
  actually CLOSER to the policy-propose behavior the paper intends (position-relevant
  actions; its example uses examine/examine/inventory). Reflection length **~120
  words** (paper example). Reflection generator = DeepSeek (workspace §1).
  - **Temporal-grounding constraint (ALFWorld-specific, added after smoke).** SR
    gives the reflector the expert action's outcome s_{i+1}. For *location-changing*
    actions (`go to`/`open`), s_{i+1} reveals the destination's contents — info the
    agent does NOT have when deciding. First smoke had ~25% of navigation reflections
    justify the move with post-hoc phrasing ("indeed I found bowl 1 there", "now I see
    cup 1 and cup 2"), which would train inference-time hallucination. Fixed with a
    system-prompt rule: justify location-changing moves with anticipatory language
    ("going there should let me find …"), never as already-seen. Post-smoke real
    temporal-leak rate **0/20** (detector only flags go-to/open and ignores
    "expect to see…"). In-place actions (take/move/use) describe the current obs
    legitimately and are not flagged.
  - **Smoke verified (38 reflections, 2 batches):** banned-vocab 0, numbered-label 0,
    doubled 0, temporal-leak 0, avg ~117 words; hand-read confirms first-person,
    convergent, observation-anchored, action-diverse, no hallucinated objects.

## 5e. Final SFT data — DONE

All three in `data/sft/` (LLaMA-Factory `instruction`/`input`/`output`; the
`instruction` = s_i = full `input_prompt` is identical across all three):
- **`expert_sft.json`** — 21,335 (≈ paper 21,031). input="", output=
  `<think>I will execute this action</think><action>{a_i}</action>`.
- **`iwm_sft.json`** — 192,008 (≈ paper 189,279). 9/state (expert + 8 uniform
  no-replacement non-expert). input=`<think>…</think><action>{a}</action>`,
  output=next-state obs (invalid "Nothing happens." kept as signal).
- **`reflection_sft.json`** — 21,331. input="", output=`<think>{reflection}</think><action>{a_i}</action>`.

Builders: `scripts/build_expert_sft.py`, `build_iwm_sft.py`, `build_reflection_sft.py`
(all no-LLM except the reflection text). Full SR generation: `propose_reflections.py`
(deepseek-v4-pro, T=0.7, freq_penalty=0.3, thinking off, 200 concurrency, 9 min,
28.5M in / 3.2M out tokens; 21,335 calls / 0 errors).

**Approved filter (TEAM_GUIDE §3) applied to reflection_sft only:** drop records
with a real temporal leak. Detectors were first tightened after audit (the raw
flags were ~98% false positives — "my best move"/"first choice" natural language;
true temporal leak only 50/21,335 = 0.23%). Those 50 were re-run (attempt-2,
`--rerun-leaked`); 46 cleared, 4 residual dropped (0.019%). banned/numbered/doubled
= 0 after tightening, nothing dropped for those. Pre-filter raw kept in
`data/rollout/sr_rollout.jsonl` (+ `sr_rerun.jsonl`) — reversible.

Env is **done** per the skill's "what done looks like": NOTES populated, isolated
conda env recorded, rollout + 3 SFT categories at full scale, builders in
`scripts/`. Remaining (await user): git-LFS track `data/**`, commit, and the
submodule/in-tree-edit decision (§6).
- **assistant format** (all three SFT files): `<think>{reasoning}</think><action>{a}</action>`
  — ALFWorld's policy output format (the original expert data uses it). expert_sft's
  `<think>` is the placeholder "I will execute this action"; reflection_sft's
  `<think>` is the real reflection CoT. That asymmetry IS the SR-vs-IL distinction.
- **user** content = the full `input_prompt` (identical across expert/iwm/reflection).
- Scripts: `propose_reflections.py` (raw SR, DONE) → `build_reflection_sft.py`
  (TODO) ; `build_expert_sft.py` + `build_iwm_sft.py` (TODO, no LLM).

## 6. Submodule / in-tree edits

`git status` in the repo shows working-tree edits to `agent_system/` and
`verl/`, plus a patch inside the alfworld submodule
(`alfred_tw_env.py`: adds `use_expert=False`, forcing `expert_plan=False`). These
are floating, uncommitted edits — a TEAM_GUIDE §7 warning sign. **OPEN DECISION:**
decide whether they belong in a wrapper (outside) or as a committed fork patch
before we depend on them.

## 7. Walkthrough regeneration (originals are gone)

The rollout reads expert walkthroughs from hardcoded
`/home/drogozhang/.../*_alfworld_walkthroughs.json`, which **do not exist** on
this machine. They are recoverable:
- TextWorld exposes `policy_commands` natively (verified: solves a game), **but**
  in raw PDDL entity names (`desk_bar__plus_02...`), not the clean
  admissible-command form ("go to desk 1").
- The clean form comes from ALFWorld's **AlfredExpert** wrapper (`expert_plan`
  extra), which the current patch disables. Regeneration = run with the expert
  wrapper enabled (or upstream alfworld) and dump `{task/trial: {solvable,
  walkthrough}}`. This is a next-phase task; mechanism is confirmed feasible.

## 8. SR / reflection (not started)

- No reflection prompt, no LLM call, no reflection data yet. Blocked on §5
  (needs clean alternative next-states).
- When built: DeepSeek V4 Pro, thinking off, **pre-call gate** required;
  reflection length and label-leakage guards per `method_recap` / `pitfalls`.
  Length target TBD (ALFWorld steps are short → likely tight, ~200 words).

## 9. Next steps

1. **(no LLM, no cost)** Decide submodule/edit handling (§6) + final data layout.
2. **(no LLM)** Regenerate walkthroughs via the AlfredExpert wrapper (§7).
3. **(no LLM)** Re-run rollout with a **corrected single-env reset+replay** so
   alternative next-states are clean (§5); re-validate with the "Nothing happens"
   audit (expert ≈ 0%, alternatives should drop to ≈ expert level).
4. **(no LLM)** Build IWM data (expert + clean alternatives) and confirm the
   expert decision data (21k) is what we keep for `expert_sft`.
5. **(LLM — pre-call gate)** Generate SR reflections on a smoke batch, inspect,
   then deployment-checkpoint → full scale.

## 10. Open decisions for the user

- Submodule/fork vs library-via-symlink; how to land the in-tree env edits (§6).
- Exact final data directory under `envs/alfworld-paper/` and whether to keep
  the existing 716 MB aggregate.
- Confirm the corrected single-env reset+replay rollout design (§5) before we
  spend wall-clock regenerating ~21k states × all-admissible probes.
- SR reflection length target and prompt extensions (§8).

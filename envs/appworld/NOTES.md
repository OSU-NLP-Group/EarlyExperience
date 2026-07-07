# AppWorld

Free-form code-action agent env, paper §B.4. The agent writes Python that runs in a sandboxed REPL with access to `apis.<app>.<endpoint>(...)` for a fixed set of mock apps (spotify, venmo, gmail, phone, file_system, amazon, todoist, simple_note, supervisor, ...). Each task is a natural-language instruction from a "supervisor"; the agent calls `apis.supervisor.complete_task(answer=...)` to finish.

This env is the **cleanest setup so far** because the expert trajectory comes from the env SDK itself — no external dataset, no version-drift issues like ScienceWorld.

## Status

- **2026-05-24** — recon: env runs end-to-end, NOTES.md drafted.
- **2026-05-25** — naive-alt baseline characterized (zero LLM); LLM proposer prompt-budget measured (zero LLM); batched-fill quality smoke (Design B, $0.04, 66 LLM calls) shows DeepSeek V4 Flash hits 100% on every quality metric at all tested batch sizes (10–100). Design locked: exhaustive per-state pool coverage, batch=10 per user decision.
- **2026-05-25** — **full LLM proposer pass DONE**: 931/931 states, $2.64, 29.9 min wall, output `data/rollout/proposer_full.jsonl` (16 MB, 84,819 filled calls).
- **2026-05-25** — probe-phase architecture validated: server has **no multi-tenancy** (global `world` in `appworld/serve/environment.py:91`), so `experiment_name` is just a label — must run 1 worker per server. 1-server c=8 smoke had 28% probe_crash; switching to 100 servers c=1 each gave 0% crash, 91% env-meaningful. Real per-server RAM under varied-task load: ~685 MB (vs ~110 MB when all servers share one task in smoke).
- **2026-05-25** — **full env-probe pass DONE**: 84,863 probes in 48.6 min (29.1 probe/s, 200 servers, ~137 GB RAM), output `data/rollout/probe_full.jsonl` (35 MB). Clean rate 99.96%, env-meaningful 85.08%. **All 931 states have ≥3 env-meaningful alts** (and 93.6% have ≥30) — ample headroom for any downstream K choice. Total IWM data cost: **$2.64** (env probes are free, just CPU/RAM).

## Env at a glance

- **HTTP-service architecture**, port-per-instance. `appworld serve environment --port N --root <appworld_root>` starts a uvicorn process; clients connect via `AppWorld(task_id, remote_environment_url="http://0.0.0.0:N")`. One port = one independent env state.
- **Action = free-form Python code** (REPL semantics). The agent's emit is one code block; `env.execute(code)` runs it and returns stdout / printed objects / exception messages as a string.
- **State**:
  - `env.task.instruction` — NL task description (string)
  - `env.task.supervisor` — `{first_name, last_name, email, phone_number}`
  - There is no admissible-action list (action space is open Python). State for SFT is built from the prompt template + observation history.
- **Per-task budget** = max emit steps (configurable; `appworld_expert_replay.py` defaults to 50).
- **No native state snapshot.** Same constraint as ScienceWorld: to probe alternatives at step `i`, must spin a fresh env + replay `expert_actions[0..i-1]` + step the alternative. O(K·N²) per task. The fresh-env-probe pattern is already implemented in the legacy `appworld_expert_replay.py`.
- **Completion**: `env.task_completed()` (bool, becomes true after `apis.supervisor.complete_task(...)`). Success: `env.evaluate().success`.

## Repo & versions

| Item | Value |
|---|---|
| Upstream (env SDK) | `https://github.com/StonyBrookNLP/appworld` |
| **Installed SDK version** | **appworld 0.2.0.dev0** (HEAD of upstream main as of 2026-05-24) |
| Outer fork (verl-agent wrapper) | `https://github.com/UlyssesXC/verl-agent` |
| Submodule path | `envs/appworld/verl-agent/` |
| Tracked branch | `appworld-ee` (forked from `master`) |
| **Pinned commit** | **`796ed31`** ("Update HGPO readme (#233)" — clean master at fork time) |
| Conda env | `appworld` (Python 3.12) |
| Pipeline-side env | not yet created — will be set up at implementation time (DeepSeek SDK + asyncio + the rollout client) |

The AppWorld package's pyproject uses `uv-build` as the build backend, so the install path is:
```bash
conda create -n appworld python=3.12 -y
/home/ulss/miniconda3/envs/appworld/bin/python -m pip install uv      # uv-build prereq
/home/ulss/miniconda3/envs/appworld/bin/python -m pip install 'git+https://github.com/StonyBrookNLP/appworld.git'
/home/ulss/miniconda3/envs/appworld/bin/appworld install              # unpacks app source + tests
/home/ulss/miniconda3/envs/appworld/bin/appworld download data        # ~198 MB tasks + base DBs
```

## Data layout (and the path conflict we resolved)

`appworld` SDK reads its data from `<APPWORLD_ROOT>/data/...`. **By default `APPWORLD_ROOT = cwd`**, which would put AppWorld's data at `envs/appworld/data/` — **directly colliding with this workspace's SKILL-mandated `envs/<env>/data/sft/` output layout**.

Resolution: AppWorld data is relocated to `envs/appworld/appworld_root/data/`. The split file lives at `envs/appworld/.env`:

```
APPWORLD_ROOT=/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root
```

**However**, `dotenv` autoloading is only honored by direct python imports; the `appworld` CLI's `serve environment` sub-command has its own `--root` flag (defaulting to `"."`) which overrides the env var. **All `appworld serve ...` invocations must pass `--root <APPWORLD_ROOT>` explicitly.** Python rollout scripts get the env var either via shell `export` or `os.environ["APPWORLD_ROOT"] = ...` *before* importing `appworld`.

Final directory layout:

```
envs/appworld/
├── NOTES.md                                  # this file
├── .env                                      # APPWORLD_ROOT pointer
├── verl-agent/                               # submodule, branch appworld-ee
├── appworld_root/                            # AppWorld SDK runtime data
│   └── data/
│       ├── tasks/<task_id>/                  # specs.json, ground_truth.{py,json}, ...
│       ├── api_docs/                         # human-readable API docs per app
│       ├── base_dbs/                         # initial DB snapshots per app
│       └── datasets/                         # split files (train/dev/test_*.txt)
├── scripts/                                  # OUR pipeline code (to be written)
│   └── _legacy/                              # snapshot of the earlier verl-agent EE scripts (see "Legacy reference")
└── data/                                     # SKILL-mandated outputs
    ├── sft/                                  # final 3 JSONLs (LFS-tracked)
    └── (intermediate artifacts, layout TBD)
```

## Task splits (verified via `load_task_ids`)

| Split | Tasks |
|---:|---:|
| train | **90** |
| dev | 57 |
| test_normal | 168 |
| test_challenge | 417 |
| **total** | **732** |

Task IDs follow `<scenario_hash>_<task_num>` (e.g. `82e2fac_1`, `82e2fac_2`, `82e2fac_3`) — each "scenario" expands into 3 variant tasks (different supervisors / app states but same NL template).

**Trajectory length** (audited over all 90 train tasks): min=6, median=9, max=19, **total 931 SA pairs (mean 10.3 / task)**.

**`required_apps` distribution** (90 train tasks):
- 1 app: 60 tasks (67%)
- 2 apps: 24 tasks (27%)
- 3 apps: 6 tasks (7%)

**App frequency** in train: spotify 48, phone 30, venmo 21, file_system 15, simple_note 12. spotify alone covers 47% of train.

**Pool size per task** (`required_apps` endpoints + always-available supervisor (6) + api_docs (4)):
- min=26 (single small app), median=100, max=115, mean=82.6.

**API counts per app**:
```
spotify=92   amazon=75   splitwise=65   todoist=56   venmo=54
gmail=48     phone=30    file_system=26  simple_note=17
supervisor=6 (always available)   api_docs=4 (always available)
```

## Expert source (D_expert) — built into the SDK

**YES, AppWorld ships ground-truth expert solutions for every task.** No external dataset to download, no version-drift normalization to chase.

For each `task_id`:

```python
from appworld.ground_truth import GroundTruth
gt = GroundTruth.load(task_id, mode="full")
gt.compiled_solution_code   # a Python function "def solution(apis, requester): ..."
gt.required_apps            # ["spotify"], ["spotify", "venmo"], ...
gt.answer                   # the expected answer (or a callable that computes it)
```

The `compiled_solution_code` is split into per-step executable code blocks by AST-walking the function body. Each top-level statement = one expert emit step. The legacy `appworld_expert_replay.py:parse_solution_code` does exactly this; reuse it.

The expert is also runnable end-to-end on a fresh env: confirmed on task `82e2fac_1` (10 steps, "What is the title of the most-liked song in my Spotify playlists.") — replay finishes with `task_completed() == True` and `evaluate().success == True`.

## Legacy reference

The previous (pre-Skill) AppWorld EE work is preserved at `envs/appworld/scripts/_legacy/`:

| File | Status for new pipeline |
|---|---|
| `appworld_expert_replay.py` | ✅ **Reusable verbatim**. AST solution-code parser + `SingleWorkerEnv` + `ReplayEnvManager` (prompt formatter). Strategy-A (fresh-env probe) replay loop. |
| `appworld_random_action_export.py` | ⚠️ Logic OK, but **alt source is a static 26-action pool** (`GENERIC_BRANCH_ACTIONS`) — violates `method_recap.md` "alt actions must be state-grounded" rule and paper §B.4 "target-model proposed". Need to replace the alt source with DeepSeek-proposed state-conditioned code blocks. |
| `appworld_random_action_export_parallel.py` | ✅ Multi-port shard launcher pattern is reusable. |
| `prompts/appworld.py` | ✅ `APPWORLD_TEMPLATE_NO_HIS` / `APPWORLD_TEMPLATE` with the verl-agent ReAct (`<think>...</think><code>...</code>`) format. Will likely be reused for SFT state serialization (TBD). |
| `start_appworld_server.sh` | ✅ Multi-port server launcher template. |

The big shape difference vs the SKILL contract: the legacy scripts produce **24-field "branch records"** (for world-model / DPO use cases); SKILL requires three SFT JSONLs (expert / iwm / reflection) with OpenAI chat-messages format. Legacy outputs would land in `data/rollout/` as intermediate artifacts; only the build_*_sft.py outputs go to `data/sft/`.

## Approach: exhaustive per-state pool coverage

AppWorld doesn't expose an admissible-action list (action = free-form Python code), and the action space is wide and arg-dependent. Two failure modes for naive alt sampling: (a) random arg literals → ~89% pure errors with low IWM signal density (verified in zero-LLM recon `scripts/recon_naive_alts.py`); (b) LLM proposing both names + args at once → uncontrolled diversity, repetition across steps.

**Decision**: instead of pre-deciding K, the proposer phase **exhaustively covers every endpoint in each state's pool**. Per state, the pool ≈ 100 endpoints (`required_apps` ∪ supervisor ∪ api_docs). DeepSeek fills args for all of them. Downstream IWM and SR steps sub-sample K from this pool with whatever error/data ratio we want.

Why this works for AppWorld specifically:
- Pool is **bounded and small** (median 100); not actually "infinite open code".
- Per-state LLM cost is one-time amortized across IWM + SR + any later ablation.
- "Informativeness" can be controlled by *selection from* the precomputed pool, not by trying to make every alt informative at generation time.

## Proposer phase design (locked 2026-05-25)

| Knob | Value | Rationale |
|---|---|---|
| Model | **DeepSeek V4 Flash** | TEAM_GUIDE §1.1 deviation; reasons below |
| Output format | JSON `{"calls": [{"endpoint": "...", "call": "apis.X.Y(...)"}, ...]}` | Unambiguous parsing; matches bfcl_v4 pattern |
| **Batch size** | **10 endpoints per LLM call** | User choice for safety. Smoke verified 100% quality up to batch=100, but conservative is fine: cost difference is tiny ($2.14 vs $3.62 full pipeline) |
| Coverage | full pool, every state | not K=3-style sub-sampling |
| Shuffling | `random.seed(42)` per state, then partition into chunks of 10 | deterministic, reproducible |
| Scope source | static AST analysis over expert prior code | no env-server needed during LLM phase; smoke confirmed quality unaffected |
| Concurrency | 20 ThreadPoolExecutor workers | DeepSeek Flash has no rate limit |
| Retry | 3× exponential backoff (1s, 3s, 8s) | per TEAM_GUIDE §1.4 |
| Resumability | output JSONL keyed by (task_id, step_idx); skip-if-exists | safe restart |

**Output**: `envs/appworld/data/rollout/proposer_full.jsonl`, one line per (task, step), containing all ~100 filled calls + per-state scope + token usage.

### Flash deviation from TEAM_GUIDE §1.1

TEAM_GUIDE §1.1 pins DeepSeek V4 Pro for all pipeline calls. We deviate to Flash for **both the proposer batch smoke and the full proposer pass** for AppWorld. Justification:

1. **Task is mechanical code-filling, not reasoning**: the LLM transforms an endpoint schema + scope dict into a syntactically valid Python call. Smoke (66 calls, $0.04) measured 100% on completeness / well-formed / correct-endpoint / required-param-coverage at batch sizes 10 / 30 / 50 / 80 / 100. Pro would not raise this ceiling.
2. **Cost matters at this scale**: full pass is ~10,200 calls. Flash is ~1/3 of Pro; total stays under $5 vs ~$15.
3. **No rate limit on Flash** enables 20-way concurrency → ~30 min wall vs hours.
4. **Precedent**: `envs/bfcl_v4` used Flash for both IWM proposer and summarizer (committed `cbcf50b`). AppWorld extends that practice.

SR (reflection) phase, when implemented, will follow the bfcl_v4 pattern: **Pro for reflection** (reasoning task), Flash for any auxiliary fill.

### Smoke results (Design B coverage, 2026-05-25)

State: task `82e2fac_1`, step 5 (mid-trajectory, scope has 5 user-defined vars). Pool size 102.

| batch | chunks | compl% | wf% | corr_ep% | scope% | req_cov% |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 33 | 100.0 | 100 | 100 | 78 | 100 |
| 30 | 12 | 100.0 | 100 | 100 | 72 | 100 |
| 50 | 9 | 100.0 | 100 | 100 | 81 | 100 |
| 80 | 6 | 100.0 | 100 | 100 | 71 | 100 |
| 100 | 6 | 100.0 | 100 | 100 | 85 | 100 |

`scope%` = % of returned calls that reference at least one scope variable; 70-85% range across batch sizes, no monotone degradation. The remaining 15-30% are endpoints whose REQUIRED params can't be filled from scope (e.g., `signup` needs new account info; `show_album(album_id=...)` needs an id not yet computed) — model uses literals or `1` as instructed.

Total: 66 calls, $0.039, 64s wall (8-way concurrent). Raw output: `data/_recon/smoke_batch_ceiling.json`.

### Cost projection (full proposer pass)

| Metric | Value |
|---|---|
| States | 931 |
| LLM calls (batch=10) | ~10,200 |
| Input tokens | ~9.0M |
| Output tokens | ~3.1M |
| **Cost (Flash)** | **~$2.15** |
| Wall (20 concurrent) | ~30 min |
| Output file | ~150-200 MB JSONL |

## Still-open decisions (after proposer pass produces the per-state pools)

1. **K for IWM**. With full pool per state, "K" is now "how many we sub-sample for IWM training records". Paper §B.4 not yet read carefully; can defer until we look at the proposer outputs and the env-probe responses (next phase).
2. **K for SR**. Same — defer.
3. **Env-probe phase concurrency**. The probe phase = run every (state, endpoint) call in fresh env to capture next-state. Estimate ~93k probes × ~7 HTTP calls (replay prefix) = ~650k HTTP. Need to size worker count vs server stability. Plan: one server per worker shard, ports 7000-7019, ~20 workers.
4. **IWM next-state format**. Raw `env.execute()` output (concise JSON / error message strings) vs summarized. Decide post-probe by inspecting actual response sizes.
5. **Error/data mix for IWM sub-sampling**. User target: ≤30% error in IWM training data. Achievable by selecting from the full pool — needs probe data first.
6. **Expert SFT CoT**. AppWorld GT has no `<think>` (raw code only). Default plan: (a) leave expert as raw code, accept structural mismatch with reflection_sft. Matches bfcl_v4 pattern.
7. **SR length target**. Per SKILL hard rule 9. Default proposal: 200–300 words target, soft cap ~500. Confirm before SR smoke.
8. **Reflection LLM** = DeepSeek V4 Pro (TEAM_GUIDE-compliant for reasoning tasks). Confirm at SR-design time.

## Setup (reproducible commands)

```bash
# one-time
conda create -n appworld python=3.12 -y
/home/ulss/miniconda3/envs/appworld/bin/python -m pip install uv
/home/ulss/miniconda3/envs/appworld/bin/python -m pip install 'git+https://github.com/StonyBrookNLP/appworld.git'
/home/ulss/miniconda3/envs/appworld/bin/appworld install

# data (relocate the default cwd-rooted data into appworld_root/)
cd envs/appworld
/home/ulss/miniconda3/envs/appworld/bin/appworld download data    # downloads into ./data (cwd default)
mkdir -p appworld_root
mv data appworld_root/data
echo "APPWORLD_ROOT=$PWD/appworld_root" > .env

# start server (any time, any port)
cd /tmp    # avoid cwd-default trap
APPWORLD_ROOT=/mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root \
  /home/ulss/miniconda3/envs/appworld/bin/appworld serve environment \
    --port 7050 \
    --root /mnt/data/xiangchao/verl-agent-ee-final/envs/appworld/appworld_root \
    --no-show-usage
```

## Things easy to get wrong here

- **`appworld serve` `--root` defaults to `.` (cwd) and silently overrides `APPWORLD_ROOT` env var.** Always pass `--root` explicitly to the CLI; for Python scripts, set `os.environ["APPWORLD_ROOT"] = ...` before `import appworld`.
- **`dotenv` autoload only works when cwd contains `.env`.** Scripts run from arbitrary cwds will silently fall back to cwd-as-root. Don't rely on `.env`; pass `APPWORLD_ROOT` explicitly.
- **No native state snapshot.** Branching at step `i` requires fresh-env + prefix replay (already handled by legacy `appworld_expert_replay.py`'s `SingleWorkerEnv` + `replay_trajectory()`). Don't try to "rewind" a live env — the SDK has no such method.
- **`experiment_name` collisions.** Two AppWorld clients with the same `experiment_name` on the same server will collide. Generate a UUID per probe (the legacy code already does this).
- **Server cwd matters when not passing `--root`.** Until the `appworld_root/` relocation, `appworld serve` would fall over with "Did not find any ./data" if launched from anywhere outside the AppWorld root.

## Inherited pitfalls from other envs (to remember)

- **`pitfalls.md` "JVM- or process-backed envs have no native state save/restore"** applies directly. Budget O(K·N²) env operations.
- **`pitfalls.md` "LLM reflection generators leak supervision labels into the CoT by default"** — applies to SR prompt design when we get there. Use the same banned-vocab + post-hoc grep defense as scienceworld.
- **`pitfalls.md` "Long-form LLM generations occasionally duplicate the whole response"** — applies to SR; same `frequency_penalty=0.3` + post-hoc dedup at SFT-build time.

## Pipeline scripts (in execution order)

- `scripts/recon_naive_alts.py` — zero-LLM. Naive baseline for 3 sample tasks (1/2/3-app). Output `data/_recon/naive_alts_report.json`.
- `scripts/recon_prompt_budget.py` — zero-LLM. Tiktoken-counts batched-fill prompts at multiple batch sizes.
- `scripts/smoke_batch_ceiling.py` — gated LLM smoke. $0.04, 66 calls. Established Flash 100% quality at batch 10–100. Output `data/_recon/smoke_batch_ceiling.json`.
- `scripts/run_proposer_full.py` — **DONE**. Full LLM proposer pass over all 931 train states. $2.64, 29.9 min, 84,819 filled calls. Output `data/rollout/proposer_full.jsonl`.
- `scripts/smoke_probe_one_task.py` — probe smoke on 1 task, 1-server. Demonstrated c=8 crash (28%) → root caused server-side global `world`.
- `scripts/smoke_probe_multi_server.py` — probe smoke on 1 task, 100-server. Validated 0% crash + 91% env-meaningful + 28.5 probe/s.
- `scripts/run_probe_full.py` — **DONE**. Full env-probe pass: 84,863 probes, 48.6 min, 200 servers, 99.96% clean, 85.08% env-meaningful. Output `data/rollout/probe_full.jsonl` (35 MB).

## Probe output schema (`data/rollout/probe_full.jsonl`)

One JSON line per probe:

```json
{
  "task_id": "82e2fac_1",
  "step": 5,
  "app": "spotify",
  "endpoint": "show_song",
  "call": "apis.spotify.show_song(song_id=1, access_token=access_token)",
  "response": "Execution failed. Traceback: ... Exception: Response status code is 404: ...",
  "response_len": 285,
  "error": null,
  "bucket": "http_404"
}
```

`bucket` is one of: `noop_success` (env "Execution successful." / json data), `data_json_list`, `data_json_dict`, `data_text`, `http_<code>` (HTTP-status-wrapped exception body), `py_typeerror`, `py_nameerror`, `py_syntaxerror`, `py_other`, `probe_crash` (client-side exception), `empty`.

## SR (self-reflection) design — K=0, IL+grounded-reflection variant (decided 2026-07-01)

**This is NOT paper-faithful SR.** AppWorld's action space is huge (~100 candidate endpoints/state, free-form args), so randomly-sampled or exhaustively-enumerated alternatives are almost never genuine competitors to the expert action — they're unrelated calls. We tried alt-comparison SR (K=3, then K=2 legit-only) and a first full run was trained; the trained SR model failed badly (worse than IL and worse than IWM): it guessed passwords, imagined the whole downstream episode in its CoT, and reversed think/code order. Root causes traced from the actual data:

- **Alt enumeration → hallucination.** 72% of the alt-comparison reflections walked through "I could do A but…, B but…, C but…" — options never present in the SFT user prompt. At inference (no option list) the model invents a menu → over-thinks / imagines episodes.
- **Fabricated-credential alts → password guessing.** 32% of states' reflections discussed alts the arg-filler had invented with placeholder creds (`password123`, `dummy_token`, `123456`). The model learned credential-guessing is a move to consider.
- **Schema fabrication + code-narration** in the CoT (asserting response field names it hadn't seen; translating the code line-by-line).

**Final design (v5):** K=0 — no alternatives at all. The reflection is a pure "why is this action the sensible move here" CoT, grounded in task + history + the *real observed outcome* (the generator/author sees the outcome to keep its reasoning on-track; it never surfaces in the monologue). This matches `method_recap.md` "Huge / open action spaces" and "Prefer principle over constraint".

- **Prompt:** principle-driven (explains that the trained agent will stand exactly where the monologue stands — only task+history, no run-yet outcome, no option list — so anything the monologue leans on that the agent couldn't have teaches hallucination). Strong generator (V4 Pro) responds far better to the principle than to a DON'T-list: schema-fabrication / code-narration / forward-planning dropped to near-zero once the "why" was explained. Prompt lives in `scripts/smoke_sr.py` / `scripts/run_sr_full.py` (`REFLECTION_SYSTEM`).
- **Generator LLM:** DeepSeek **V4 Pro**, temperature 0.7, thinking disabled.
- **Output:** monologue ONLY. The `<code>` is attached deterministically at SFT-build time (`<think>{monologue}</think>\n<code>{expert_code}</code>`) — never rely on the generator to reprint code (it omitted/garbled it in ~15% of cases; and it's known-correct anyway).
- **Length:** hard cap ~120 words, principle-enforced (brevity keeps the action signal clean). v5 smoke: 33–72 words, median ~55.
- **Post-hoc:** dedup DeepSeek's whole-response doubling (pitfalls.md), leak-grep as a safety net.
- **v5 smoke verdict (20 states, read in full, not scanned):** genuine forward reasoning (explains *why*: dedup with a set, unknown page count → paginate, auth-before-query), no schema fabrication, no outcome-quoting, no placeholder self-dismissal, 0 leak / 0 doubling.

## IWM SFT — use balanced, drop full (decided 2026-07-01)

Training showed **iwm_balanced beat IL; iwm_full was worse than IL.** Too many error-outcome transitions (the full 84k set is ~45% errors: 401/422/nameerror) dilute the world-model signal. `iwm_sft_balanced.jsonl` (per-state stratified: ~7 data + 2 http + 1 py, fixed seed) is the shipped IWM file. `iwm_sft_full.jsonl` is kept as an intermediate/ablation artifact, not the deliverable. See `method_recap.md` "Huge / open action spaces".

## think/code format consistency (symptom 3)

`expert_sft.jsonl` assistant = `<code>` only (GT has no reasoning); `reflection_sft.jsonl` = `<think>…</think><code>…</code>`. Mixing the two at train time can teach an inconsistent template (observed: trained model reversed think/code order). **User owns this fix** (either give expert a brief think, or handle the mix trainer-side). Data-side: both files use the same AGENT system prompt whose format clause already says "you may reason inside `<think>`, then emit `<code>`; if nothing to reason about, emit `<code>` directly."

## Next step

Full SR regen with the v5 K=0 principle-driven prompt (~931 monologues, V4 Pro, ~$0.5, ~6 min), then rebuild `reflection_sft.jsonl` with deterministic `<code>` attach + doubling dedup. `expert_sft.jsonl` and `iwm_sft_balanced.jsonl` already built.

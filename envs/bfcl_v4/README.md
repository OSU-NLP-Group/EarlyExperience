# BFCL v4 (multi-turn function calling)

Function-calling benchmark, paper §B.3 (paper calls this "BFCLv3"; we pin v4 because the multi-turn data lineage is identical plus ~6 months of upstream ground-truth / evaluator fixes). The agent handles a scripted 1–7 user-turn conversation with access to a small set of simulated API classes (file system, message API, trading, ticket, travel booking, vehicle control, ...) and completes each subtask by emitting Python-syntax function calls like `mv(source='a', destination='b')`.

## Upstream

- [`ShishirPatil/gorilla`](https://github.com/ShishirPatil/gorilla) (Apache 2.0) — the Berkeley Function Calling Leaderboard code and evaluator. Pin to the 2025-12-17 commit that matches the 2025-12-16 BFCL leaderboard publication (which generated our expert trajectories).

## Modification strategy

Upstream Gorilla is a full evaluator harness — we use it *as a library* for the sim classes and their state semantics, and layer an offline EE pipeline on top. Two important properties make BFCL an unusually clean EE target:

1. **State is `copy.deepcopy`-able**. All sim classes are pure Python instances, so we can snapshot / restore state without spinning up a fresh env. This makes alternative-action probing cheap: `O(K)` deepcopies per state, no re-replay.
2. **Expert already carries observations**. The published Opus-FC leaderboard result JSON embeds per-step tool responses in the `inference_log`. Expert SFT harvest does **not** require re-replay against the fork.

Conceptual pieces layered on top:

1. **Expert harvester** — parses the published Opus expert log to extract per-turn `(state, expert_calls, tool_responses)` tuples. No env execution required.
2. **Alternative-action prober** — at each fcall state, we sample K distinct alt *names* from the sim class's method pool (no LLM needed for names — the pool is structurally enumerable), then make **one** LLM call to fill args for all K names in a single response. Each alt is `eval`'d against a `deepcopy` of the live sim instance, and the tool output + post-state snapshot are recorded.
3. **Summarizer** — raw tool responses are noisy (large dicts, error strings, ...). A summarizer LLM condenses each response to a single descriptive sentence. Notable design choice: unlike the paper's canonical "cannot help fulfill the user's task" template for irrelevant alts, our summarizer stays **purely descriptive** ("Moved 'a' into 'b/'.", "Retrieved 5 records.", "Error: destination directory does not exist."), so SR later has more material to reason over.
4. **Reflection builder** — standard SR monologue arriving at expert calls given the alt summaries.

## Method mapping (EE ↔ BFCL)

| EE concept | BFCL realization |
|---|---|
| Expert action | one fcall (or a list of parallel fcalls) at each agent emit step |
| Alternative-action pool | K distinct method names from the involved sim classes, args filled by one LLM call |
| Next state | tool output + sim-instance vars() after `eval`'ing the alt against a deepcopy |
| Reflection target | expert calls + LLM-authored monologue over alt summaries |

**Key insight for BFCL**: because state is deepcopy-able, IWM is dramatically cheaper than any HTTP-service env — alt probing is just Python object copies.

## Data output

Living on Google Drive (this repo does **not** vendor any SFT files):

```
ggdrive:Early-Experience-Reproduce/data/bfcl_v4_v2/
├── expert_sft_text.jsonl        # paper-faithful plain-Python action format
├── expert_sft_fc.jsonl          # OpenAI FC schema variant
├── iwm_sft_text.jsonl
├── iwm_sft_fc.jsonl
├── reflection_sft_text.jsonl
└── reflection_sft_fc.jsonl
```

Both `text` and `fc` variants are provided so downstream training can pick the surface form that matches the target chat template.

## Reproducibility notes

- **Trajectory source**: `HuanzhiMao/BFCL-Result/2025-12-16/result/claude-opus-4-5-20251101-FC/multi_turn/BFCL_v4_multi_turn_base_result.json`, filtered to `valid == true` (162 / 200 Base cases).
- **Split**: 75/25 train/held-out by `case_id` with `random.seed(42)`. 3 OOD splits (`long_context`, `miss_func`, `miss_param`) never enter any SFT file.
- **K for IWM**: 10 alt names per state (structurally satisfiable on every Base case).
- **K for SR**: 3 alternatives per state.
- **Proposer / summarizer / reflection models**: DeepSeek V4 Flash (proposer, summarizer, args are schema-bound) / DeepSeek V4 Pro (reflection, needs more nuanced reasoning).
- **Anti-leak vocabulary filter for SR**: post-hoc grep filter against words like "expert" / "selected" / "chosen" / "correct" / "best" / "optimal" / "preferred" and numbered alt labels ("Action 1", "Alternative #2"). Any SR pipeline should apply an equivalent filter.

For a full reproduction, take the pipeline scripts under `scripts/` in this env as a reference, install the BFCL leaderboard package from upstream (pinned to the 2025-12-17 commit) in its own conda env, and follow the modification strategy above.

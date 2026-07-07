# AppWorld

Free-form code-action agent env. The agent solves a natural-language supervisor request (`"Deposit $5 in each of my spotify friends' Venmo accounts."`) by writing Python that runs in a sandboxed REPL with access to `apis.<app>.<endpoint>(...)` — a fixed catalog of mock apps (spotify, venmo, gmail, phone, file_system, amazon, todoist, simple_note, supervisor, ...).

## Upstream

- Env SDK: [`StonyBrookNLP/appworld`](https://github.com/StonyBrookNLP/appworld) — the AppWorld benchmark itself (HTTP-service, port-per-instance).
- Training wrapper: [`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent) — same family as ALFWorld/WebShop; provides a gym-style interface over the AppWorld SDK.

## Modification strategy

AppWorld is arguably the **cleanest EE candidate** because the expert trajectory comes from the SDK itself — every task ships with a ground-truth Python program that solves it, so we never pay LLM cost for expert data.

Two conceptual pieces layered on top of the verl-agent AppWorld wrapper:

1. **Expert extractor** — for each task, run the SDK's ground-truth program in a fresh env and record every `(state_i, expert_code_i)` step, along with the state observation the env returns after execution.
2. **Alternative-action generator** — because the action space is *open Python*, alternatives cannot be enumerated. We prompt a proposer LLM to produce K candidate code snippets per state (given the conversation history + available API schemas), then step each on a re-cloned env replayed to that state. The proposer runs **one call per state** filling K candidates in a single response (paper-faithful "K per call", not K separate calls).
3. **Reflection builder** — standard SR: LLM writes first-person reasoning that arrives at the expert action given the alternatives' outcomes.

## Method mapping (EE ↔ AppWorld)

| EE concept | AppWorld realization |
|---|---|
| Expert action | one Python emission from the SDK's ground-truth program at each step |
| Alternative-action pool | LLM-proposed K candidate Python emissions (fill mode, one call per state) |
| Next state | stdout / printed objects / exception messages returned by `env.execute(code)` |
| Reflection target | expert code + LLM-authored monologue |

**Key insight for AppWorld**: expert data is free (SDK ground truth); IWM cost is one proposer call per state + K env probes. Total dev cost measured at ~$3 for the full 931-state train split at K=10 with DeepSeek V4 Flash.

## Data output

Available in the [Hugging Face dataset](https://huggingface.co/datasets/osunlp/early-experience):

```
appworld/
├── expert_sft.jsonl
├── iwm_sft_full.jsonl         # all filled alts kept
├── iwm_sft_balanced.jsonl     # per-state K clipped for balanced training
└── reflection_sft.jsonl
```

Both `full` and `balanced` IWM variants are provided so downstream training can A/B them.

## Reproducibility notes

- **Trajectory source**: AppWorld SDK ground-truth programs for the `train` split (90 tasks, 931 SA pairs).
- **State re-synchronization**: AppWorld has no native snapshot API, so to probe alternatives at step `i` we spin up a fresh env, replay `expert_actions[0..i-1]`, then execute the alternative — the same `O(K · N²)` pattern any AppWorld EE implementation must follow.
- **Proposer**: DeepSeek V4 Flash with structured output; args and code candidates schema-bound so temperature-1 sampling is stable.
- **K for IWM**: the pipeline is designed so downstream K can be chosen at build time (data has ≥30 env-meaningful alternatives at 93.6% of states). `iwm_sft_balanced.jsonl` clips to a fixed per-state K.
- **Server concurrency caveat**: AppWorld's server has no multi-tenancy (a global `world` object). Any parallel probing must launch one server process per worker — running K workers against a shared server silently races and corrupts state.

For a full reproduction, install the AppWorld SDK per its own README and follow the modification strategy above against a fresh clone of upstream.

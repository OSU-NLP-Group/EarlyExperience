# AppWorld

Free-form code-action agent env. The agent solves a natural-language supervisor request (`"Deposit $5 in each of my spotify friends' Venmo accounts."`) by writing Python that runs in a sandboxed REPL with access to `apis.<app>.<endpoint>(...)` — a fixed catalog of mock apps (spotify, venmo, gmail, phone, file_system, amazon, todoist, simple_note, supervisor, ...).

**Not in the paper.** All per-env hyperparameters (K, sampling strategy, filter rules) are decided locally following the same conventions used for paper envs.

## Upstream

- Env SDK: [`StonyBrookNLP/appworld`](https://github.com/StonyBrookNLP/appworld) — the AppWorld benchmark itself (HTTP-service, port-per-instance).
- Training wrapper: [`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent) — same family as ALFWorld/WebShop; provides a gym-style interface over the AppWorld SDK.

## Modification strategy

The expert trajectory comes from the AppWorld SDK itself — every task ships with a ground-truth Python program that solves it, so we never pay LLM cost for expert data.

Four conceptual pieces layered on top of the verl-agent AppWorld wrapper:

1. **Expert extractor** — for each task, run the SDK's ground-truth program in a fresh env and record every `(state_i, expert_code_i)` step, along with the state observation the env returns after execution.
2. **Alternative-action generator** — because the action space is *open Python*, alternatives cannot be enumerated. A proposer LLM produces K candidate code snippets per state (given the conversation history + available API schemas), and each is stepped on a re-cloned env replayed to that state. The proposer runs **one call per state** filling K candidates in a single response (paper-faithful "K per call", not K separate calls).
3. **IWM data curation** — the raw alt-outcome distribution is dominated by invalid API calls (wrong endpoint or wrong args → exception messages). Training IWM on that raw mix would collapse the next-state target onto *"you got an error"* for most inputs, which is uninformative. The IWM SFT file therefore keeps all env-meaningful outcomes and only a bounded sample of invalid ones — the stage-1 next-state training distribution is deliberately rebalanced toward informative transitions.
4. **Reflection builder** — SR runs with **K=0**. Under an open-Python action space, sampled alternatives at a given state are rarely competitive attempts at the same subgoal: they collapse into unrelated API calls or fabricated-arg noise, and mixing those into SR teaches the model to imitate them. So the SR pipeline drops alternative-comparison entirely — for each expert state, it generates a first-person CoT that arrives at the expert action grounded in the task, history, and observed outcome. This is *not* paper-faithful SR (there is no alt comparison); it is the IL + grounded-reflection variant discussed in `skill/method_recap.md`.

## Method mapping (EE ↔ AppWorld)

| EE concept | AppWorld realization |
|---|---|
| Expert action | one Python emission from the SDK's ground-truth program at each step |
| Alternative-action pool (IWM only) | LLM-proposed K candidate Python emissions (fill mode, one call per state); post-filtered to rebalance valid vs invalid outcomes |
| Next state | stdout / printed objects / exception messages returned by `env.execute(code)` |
| Reflection target | expert code + LLM-authored monologue **grounded in task + history alone** (K=0, no alt comparison) |

**Key insight for AppWorld**: expert data is free (SDK ground truth); IWM cost is one proposer call per state + K env probes (scales linearly with states and K, plus the generator-model cost — see `skill/SKILL.md`). Two data-quality decisions this env forces relative to paper-faithful EE: (a) the IWM next-state distribution is rebalanced post-hoc away from the raw invalid-API-error majority; (b) SR runs at **K=0** because sampled alternatives under open Python are rarely competitive at the same subgoal — the SR data is IL + grounded reflection rather than paper-faithful alt-comparison SR.

## Data output

Available in the [Hugging Face dataset](https://huggingface.co/datasets/osunlp/early-experience) under `appworld/`:

```
appworld/
├── expert_sft.jsonl
├── iwm_sft.jsonl
└── reflection_sft.jsonl
```

## Reproducibility notes

- **Trajectory source**: AppWorld SDK ground-truth programs for the `train` split (90 tasks, 931 SA pairs).
- **State re-synchronization**: AppWorld has no native snapshot API, so to probe alternatives at step `i` we spin up a fresh env, replay `expert_actions[0..i-1]`, then execute the alternative — the same `O(K · N²)` pattern any AppWorld EE implementation must follow.
- **Proposer**: any LLM with structured output; args and code candidates schema-bound so temperature-1 sampling is stable. See `skill/SKILL.md` for the model-choice discussion.
- **K for IWM**: the raw pipeline retains ≥30 env-meaningful alternatives at 93.6% of states, so downstream K can be chosen at build time. Post-filter caps the invalid-API-error share so the IWM training distribution isn't dominated by `"you got an error"` transitions.
- **K for SR**: **0** (no alt comparison). Rationale: LLM-sampled alternatives under open Python are almost never *competitive* attempts at the same subgoal — they collapse into unrelated calls or fabricated-arg noise, and mixing them into SR teaches imitation of that noise. The SR data is IL + grounded-reflection, per the K=0 discussion in `skill/method_recap.md`; label it accordingly if your downstream training expects paper-faithful alt-comparison SR.
- **Server concurrency caveat**: AppWorld's server has no multi-tenancy (a global `world` object). Any parallel probing must launch one server process per worker — running K workers against a shared server silently races and corrupts state.

For a full reproduction, install the AppWorld SDK per its own README and follow the modification strategy above against a fresh clone of upstream.

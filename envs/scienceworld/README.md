# ScienceWorld

Interactive science-lab env, paper §B.6. The agent solves procedural science tasks (`"Your task is to boil water."`, `"...find an animal that eats plants..."`) by stepping through a JVM-backed simulator of a small lab world with 10+ rooms and 400+ manipulable objects.

## Upstream

- Env-server framework: [`WooooDyy/AgentGym`](https://github.com/WooooDyy/AgentGym) — the AgentGym family, which serves ScienceWorld as an HTTP service.
- Underlying env package: `scienceworld` (PyPI, Java 11 backend).

## Modification strategy

The paper's method requires an **admissible-action list at every state**. ScienceWorld's Python package can produce this natively (`get_valid_action_object_combinations()`), but AgentGym's stock HTTP server does **not** expose that flat list — it only ships an `/action_hint` endpoint that returns action templates and object names as two separate lists.

The one modification we made to AgentGym: **add an `/admissible_actions` endpoint** backed by ScienceWorld's own combination generator, so client-side EE code has O(1) access to the same admissible list the paper §B.6 protocol assumes.

Beyond that endpoint patch, everything else is a client-side EE data pipeline:

1. **Expert extractor** — reuses AgentGym's public `AgentTraj-L` dataset (SciWorld split, `sciworld_train.json` on Hugging Face). These are pre-computed expert trajectories.
2. **Trajectory normalizer** — AgentTraj-L was generated against a pre-public ScienceWorld version whose object naming differs slightly from any PyPI release. We apply three string-level rewrites to the assistant turns (e.g. `green house` → `greenhouse`; wire-terminal indexing simplification) so trajectories are byte-executable against unmodified `scienceworld 1.1.3+`. **Design choice**: the rewrites live in the *data*, not the env — downstream users who train on our SFT and evaluate against a stock AgentGym stack produce action strings the env accepts, no env patch required at inference time.
3. **Replay filter** — trajectories that fail to reach `done && score==100` when replayed step-by-step are dropped (2,038 / 2,120 = 96.1% pass).
4. **Alternative-action prober** — at each expert state, draw K non-expert actions uniformly from `admissible \ {expert}`, step the env on each (plus the expert), record next states.
5. **Reflection builder** — the paper §B.6 protocol: the *policy itself* proposes SR alternatives (not admissible-random), then any proposed action missing from the admissible list is discarded and refilled from random admissible.

## Method mapping (EE ↔ ScienceWorld)

| EE concept | ScienceWorld realization |
|---|---|
| Expert action | one action from AgentTraj-L's gpt turn at each state |
| IWM alternative pool | K uniform samples from `admissible \ {expert}` |
| SR alternative pool | K policy-proposed actions at T=1, refilled from admissible if invalid |
| Next state | short structured lab readout ("The door is now open.", "You are in the kitchen.", ...) |
| Reflection target | expert action + LLM-authored monologue over alt outcomes |

**Key gotcha, paper-cited**: IWM and SR use *different* alternative-sampling pipelines. Conflating them (using policy-proposed alts for IWM, or admissible-random for SR) is a common re-implementation bug.

## Data output

Available in the [Hugging Face dataset](https://huggingface.co/datasets/osunlp/early-experience) under `scienceworld/`:

```
scienceworld/
├── expert_sft.jsonl
├── iwm_sft.jsonl
└── reflection_sft.jsonl
```

## Reproducibility notes

- **Trajectory source**: `AgentGym/AgentTraj-L` (SciWorld split) on Hugging Face — 2,120 trajectories, 42,178 (state, action) pairs. Note: paper §B.6 cites 14,506 pairs; discrepancy unresolved (paper may have used a filtered subset).
- **K for IWM**: 3 non-expert alternatives per state (plus expert itself = 4 triples per state).
- **K for SR**: 3 alternatives (drop to 2 when the policy model is Llama-3.1-8B-Instruct, per paper §B.6).
- **One-shot example**: paper §B.6 specifies a one-shot in-context example. AgentGym's bundled prompt does *not* include it — add it in the client-side prompt, used identically at both SR generation and inference.
- **Admissible-list caching**: the admissible list is state-dependent (recomputed every step). Do not cache it across steps in the client.
- **Action canonicalization**: AgentGym's client adapter enforces specific surface forms (e.g. `drop` not `put down`, `wait1` for single-step wait). Any IWM/SR alt strings must match these forms so expert and rollout entries share a vocabulary.
- **Server concurrency**: default is a single uvicorn process backed by one JVM instance. For real parallelism, run multiple `sciworld` servers on different ports and round-robin.

For a full reproduction, install the AgentGym env-server package from upstream, apply the `/admissible_actions` endpoint patch described above, and follow the modification strategy.

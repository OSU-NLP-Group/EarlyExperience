# TravelPlanner

Long-horizon travel-planning env, paper §B.7 (originally from Xie et al. 2024, [arXiv:2402.01622](https://arxiv.org/abs/2402.01622)). The agent fills a multi-day travel plan — transportation / 3 meals / attractions / accommodation for each day — under budget and constraint pressure, then submits it in one shot.

## Upstream

- Paper benchmark suite: [`osu-nlp-group/TravelPlanner`](https://github.com/osu-nlp-group/TravelPlanner) — evaluator, database, and reference agents.
- **Paper gym**: the paper authors also ship a private `gym_travelplanner` Python package that exposes the same task as a gym-style step/reset API. This package is **not publicly released**; if you're re-implementing, either request it from the authors or re-derive an equivalent interface from the benchmark suite (the benchmark suite's evaluator gives you all the semantics you need — pricing, constraint checking, etc.).

## Modification strategy

The env is turn-based but "one turn" is enormous — the agent fills the entire multi-day plan before evaluation. That makes SR the dominant methodological challenge here: the reasoning trace must explain why a *particular slot* was filled a particular way, not just why the *whole plan* was chosen.

Three conceptual pieces layered on top of the paper gym:

1. **Expert extractor** — for each task, replay the paper's gold plan through the gym slot by slot, recording per-slot `(state_i, expert_choice_i)` pairs where `state_i` is the paper-formatted plan-in-progress + candidates + budget-remaining.
2. **IWM next-state predictor** — cheap: no LLM call. Next state is a rendered budget-delta line + a templated field-effect ("You now have $1,226 left of $1,700.", "Transportation for day 1 fixed."). Paper §B.7's IWM is exactly this deterministic-render setup.
3. **Reflection builder** — reflections must justify the specific slot against alternatives *from the same slot*. K=8 alternatives per slot are stratified across constraint dimensions (budget / cuisine / min-nights / city-sequence / mode-chain). The reflection prompt includes explicit structural blocks for `CITY SEQUENCE` (for transportation states) and `CUISINE PREFERENCE` (for meal states with cuisine constraints) so the critical checks appear in every reflection instead of being silently dropped.

## Method mapping (EE ↔ TravelPlanner)

| EE concept | TravelPlanner realization |
|---|---|
| Expert action | one slot-fill (transportation choice / meal choice / attraction / accommodation) at a specific state |
| Alternative-action pool | K=8 alternative slot fills, stratified across constraint dimensions to expose different failure modes |
| Next state | rendered budget-delta + templated field-effect (no LLM) |
| Reflection target | expert choice + LLM-authored monologue engaging with all K alt outcomes and the specific constraints in play |

**Key insight for TravelPlanner**: unlike other envs where the alt pool is enumerable or LLM-proposed, here the alt pool must be **stratified** — random alts underweight constraint violations that only surface for specific combinations (wrong-city, insufficient-nights, cuisine-mismatch). The reflection prompt must also structurally include the "required checks" or the LLM silently drops them.

## Data output

Available in the [Hugging Face dataset](https://huggingface.co/datasets/osunlp/early-experience) under `travelplanner/`:

```
travelplanner/
├── expert_sft.jsonl
├── iwm_sft.jsonl
└── reflection_sft.jsonl
```

## Reproducibility notes

- **Trajectory source**: paper's official gold plans through the paper gym.
- **K for IWM**: implicit (deterministic next-state render — no probing needed; every slot yields one training record).
- **K for SR**: 8 alternatives per slot, stratified across constraint dimensions.
- **State representation**: `_format_state_sft` from the paper gym (byte-identical to what the trained model sees at inference).
- **CoT length target**: aligns with paper §B.7's "extend max_gen to 8K" recommendation. See `skill/SKILL.md` hard rule 9 for the general length-choice discussion.
- **Anti-leak vocabulary filter**: same as other envs — no `"expert"`, `"chosen"`, `"correct"`, `"best"`, numbered alt labels.
- **Structural constraints in the prompt**: for transportation states, always include a `CITY SEQUENCE` block computed from the plan-so-far (visited destination cities with entry days, still-need count, final-day return constraint). For meal states with cuisine constraints, always include a `CUISINE PREFERENCE` block and make cuisine match/mismatch a required check.

For a full reproduction, obtain the paper gym (`gym_travelplanner`) from the authors or re-derive it from the public benchmark suite, and follow the modification strategy above.

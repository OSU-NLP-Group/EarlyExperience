# TravelPlanner

Long-horizon travel-planning env, paper §B.7 (originally from Xie et al. 2024, [arXiv:2402.01622](https://arxiv.org/abs/2402.01622)). The agent fills a multi-day travel plan — transportation / 3 meals / attractions / accommodation for each day — under budget and constraint pressure, then submits it in one shot.

## Upstream

- Paper benchmark suite: [`osu-nlp-group/TravelPlanner`](https://github.com/osu-nlp-group/TravelPlanner) — evaluator, database, and reference agents.
- **Paper gym**: the paper authors also ship a private `gym_travelplanner` Python package that exposes the same task as a gym-style step/reset API. This package is **not publicly released**; if you're re-implementing, you'll need to either request it from the authors or re-derive an equivalent interface from the benchmark suite (the benchmark suite's evaluator gives you all the semantics you need — pricing, constraint checking, etc.).

## Modification strategy

The env is turn-based but "one turn" is enormous — the agent fills the entire multi-day plan before evaluation. That makes SR the dominant methodological challenge here: the reasoning trace must explain why a *particular slot* was filled a particular way, not just why the *whole plan* was chosen.

Two conceptual pieces layered on top of the paper gym:

1. **Expert extractor** — for each task, replay the paper's gold plan through the gym slot by slot, recording per-slot `(state_i, expert_choice_i)` pairs where `state_i` is the paper-formatted plan-in-progress + candidates + budget-remaining. Total ~1,370 records across all training tasks.
2. **IWM next-state predictor** — cheap: no LLM call. Next state is a rendered budget-delta line + a templated field-effect ("You now have $1,226 left of $1,700.", "Transportation for day 1 fixed."). Paper §B.7's IWM is exactly this deterministic-render setup.
3. **Reflection builder** — this is where TravelPlanner is different. Reflections must justify the specific slot against alternatives *from the same slot*. K=8 alternatives per slot are stratified across constraint dimensions (budget / cuisine / min-nights / city-sequence / mode-chain). The reflection prompt evolved through multiple iterations because the first versions leaked "expert" labels and dropped required constraint checks. Final v9 adds explicit `CITY SEQUENCE` and `CUISINE PREFERENCE` structural blocks in the prompt to guarantee the critical checks appear in every reflection.
4. **Reflection compressor** — after v9 stabilized, an additional compression pass halves each reflection while preserving the paper-required constraint acknowledgments (cuisine match/mismatch, city sequence, min-nights math, budget delta, commitment).

## Method mapping (EE ↔ TravelPlanner)

| EE concept | TravelPlanner realization |
|---|---|
| Expert action | one slot-fill (transportation choice / meal choice / attraction / accommodation) at a specific state |
| Alternative-action pool | K=8 alternative slot fills, stratified across constraint dimensions to expose different failure modes |
| Next state | rendered budget-delta + templated field-effect (no LLM) |
| Reflection target | expert choice + LLM-authored monologue engaging with all K alt outcomes and the specific constraints in play |

**Key insight for TravelPlanner**: unlike other envs where the alt pool is enumerable or LLM-proposed, here the alt pool must be **stratified** — random alts underweight constraint violations that only surface for specific combinations (wrong-city, insufficient-nights, cuisine-mismatch). The reflection prompt must also structurally include the "required checks" or the LLM silently drops them.

## Data output

Available in the [Hugging Face dataset](https://huggingface.co/datasets/osunlp/early-experience):

```
travelplanner/
├── expert_sft.jsonl
├── iwm_sft.jsonl
└── (legacy reflection version)

travelplanner_v3/
├── reflection_sft_full.jsonl        # 1,225 records, median 124 words
└── reflection_sft_compressed.jsonl  # 1,226 records, median  50 words
```

Both `full` and `compressed` reflection variants ship so downstream training can A/B them.

## Reproducibility notes

- **Trajectory source**: paper's official gold plans through the private paper gym.
- **K for IWM**: implicit (deterministic next-state render — no probing needed; every slot yields one training record).
- **K for SR**: 8 alternatives per slot, stratified across constraint dimensions.
- **State representation**: `_format_state_sft` from the paper gym (byte-identical to what the trained model sees at inference).
- **CoT length target**: 100–160 words in `full`, ~50 words in `compressed`. The full version aligns with paper §B.7's "extend max_gen to 8K" recommendation; the compressed version is our own addition for tighter training.
- **Anti-leak vocabulary filter**: same as other envs — no `"expert"`, `"chosen"`, `"correct"`, `"best"`, numbered alt labels. TravelPlanner had 37 flags in v9-full (of 1,226 records); a rerun cleared 30/37.
- **Structural constraints in the prompt**: for transportation states, always include a `CITY SEQUENCE` block computed from the plan-so-far (visited destination cities with entry days, still-need count, final-day return constraint). For meal states with cuisine constraints, always include a `CUISINE PREFERENCE` block and make cuisine match/mismatch a required check.

For a full reproduction, obtain the paper gym (`gym_travelplanner`) from the authors or re-derive it from the public benchmark suite, and follow the modification strategy above.

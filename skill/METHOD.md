# METHOD: Early Experience for Language Agents

This is the distilled methods reference for the paper **"Agent Learning via Early Experience"** (arXiv:2510.08558, Zhang et al., 2025). The full paper PDF is included as `paper.pdf` for cross-reference. When in doubt, treat the paper as source of truth.

## 1. The Paradigm

Early experience is a middle ground between imitation learning and reinforcement learning, designed for environments that lack verifiable reward signals or where long-horizon rollouts make RL impractical. The agent collects its own interaction data by proposing non-expert actions at expert-visited states, and uses the **resulting next states** as supervision — without any reward.

There are two methods under this paradigm. Both share the same high-level structure — at expert states, collect alternative actions, observe their next states, and use those next states as supervision. The specific sampling and rollout details often differ between IWM and SR within a given environment, and across environments; treat the structure here as the shared backbone and work out the per-env specifics from the env itself and the user's requirements.

## 2. Notation

- `D_expert = {(s_i, a_i)}` — expert state-action pairs from existing demonstrations.
- For each expert state `s_i`, sample `K` alternative actions `{a_i^1, ..., a_i^K}` from the initial policy `π_θ(· | s_i)`. These must differ from the expert action `a_i`.
- Execute each `a_i^j` in the environment from state `s_i`, observe next state `s_i^j` sampled from the transition function `T(s_i, a_i^j)`.
- Also execute the expert action `a_i` to get the expert next state `s_{i+1}`.
- Collect `D_rollout = {(s_i, a_i^j, s_i^j)}` for all `i ∈ [N], j ∈ [K]`.

K is a per-env hyperparameter. Background and starting points are sketched in the env's `NOTES.md`; the final value comes from user input and what makes sense for the env.

## 3. Method 1: Implicit World Modeling (IWM)

**Idea.** Train the policy to predict the next state given the current state and an action. This grounds the model in environment dynamics before SFT. No separate world-model module — the same parameters `θ` handle both state prediction and action prediction.

**Training objective.** Next-token prediction over the next state, conditioned on the state-action pair:

```
L_IWM = - Σ over (s_i, a_i^j, s_i^j) ∈ D_rollout
          log p_θ(s_i^j | s_i, a_i^j)
```

**Two-stage training pipeline.**

1. **Stage 1 (IWM warm-up):** train for ~1 epoch on `D_rollout` with the next-state prediction loss above.
2. **Stage 2 (Imitation):** continue training on `D_expert` with the standard imitation learning loss `L_IL = - Σ log π_θ(a_i | s_i)`.

Training itself is out of scope for this skill — the two-stage description is here only so data producers know how the data is consumed downstream.

**Data composition for SFT framework.**
For each rollout triple, the SFT example is:
- **user / prompt:** representation of `s_i` + the action `a_i^j`
- **assistant / completion:** representation of `s_i^j` (the next state)

Some environments need to summarize the raw next state before training (e.g. WebShop uses a ~345-char text summary; SearchQA summarizes retrieved documents). Whether the raw next state is usable, or a summarizer is needed, depends on the env's state format; decide it from the env and confirm with the user, and document the decision in the env's `NOTES.md`.

## 4. Method 2: Self-Reflection (SR)

**Idea.** Have the agent compare its own non-expert actions with the expert action by looking at the resulting next states. Use an LLM to generate a chain-of-thought explanation `c_i^j` of why the expert action is preferable. Train the policy to produce both the reflection and the expert action.

**Reflection generation.** For each expert state `s_i`:
1. Execute the expert action `a_i` → get expert next state `s_{i+1}`.
2. For each alternative `a_i^j`, execute → get `s_i^j`.
3. Prompt a reflection LLM with `(s_i, a_i, s_{i+1}, [(a_i^j, s_i^j) for j in 1..K])` and get back a chain-of-thought `c_i^j` (one per alternative, or one summary covering all — paper uses per-alternative).

The standard reflection prompt template (verbatim from paper §4.3):

```
You will be presented with a situation where you need to choose between
multiple possible actions. Your task is to analyze the situation and provide
reasoning about why we decide to take the expert action.

- Situation Description (s_i): {situation}
- Expert Action (a_i): {expert_action}
- Expected Outcome (s_{i+1}): {expert_next_state}
- Alternative Actions:
  1. Action a_i^1: {alt_1}, resulting state s_i^1: {state_1}
  2. Action a_i^2: {alt_2}, resulting state s_i^2: {state_2}
  3. ...

Provide a detailed self-reflection as an internal monologue that demonstrates
your reasoning process for the current situation. Your monologue should:
1. Analyze the situation and the goal.
2. Compare the possible actions, explaining why each may be less optimal.
3. Justify why the expert action is most suitable, grounded in the expected outcome.
4. Highlight any relevant clues, constraints, or consequences from the situation.

Guidelines:
- Stay strictly within the provided information.
- Avoid meta-commentary about being an AI.
- Use natural, step-by-step reasoning.
- Focus on logical decision-making.

Output: Directly write the self-reflection monologue, no extra headings,
disclaimers, or external notes.
```

Each environment may extend this with environment-specific situation/action wording, but the four numbered analysis points and the four guidelines should remain. Document any deviation in the env's `NOTES.md`.

**Training objective.** Next-token prediction over the concatenated `c_i^j ◦ a_i` target, given `s_i`:

```
L_SR = - Σ over (s_i, a_i^j, c_i^j) ∈ D_refl
         log p_θ(c_i^j, a_i | s_i)
```

In the paper, training mixes `D_refl` with `D_expert` and runs a single next-token-prediction pass — but this mixing is a **training-time** operation. This skill's data-generation stage produces them as separate files (`expert_sft.jsonl` and `reflection_sft.jsonl`); whether and how to mix is the trainer's responsibility downstream.

Whether expert trajectories carry chain-of-thought reasoning matters for comparability between the IL baseline and SR. Check the actual expert data and align the assistant content across the two files consistently; the per-env choice and its rationale go in the env's `NOTES.md`.

**Data composition for SFT framework.**
- **user / prompt:** representation of `s_i`
- **assistant / completion:** `c_i^j` (the reflection) followed by `a_i` (the expert action)

`D_expert` examples have user = `s_i`, assistant = `a_i` (with original CoT if any).

## 5. Filtering in the paper

The paper applies several filtering steps:
- WebShop drops trajectories that finish in >15 steps before training.
- SR drops cases where the reflection LLM's concluded action doesn't match the expert.
- IWM keeps all rollout triples; invalid-action error messages are retained as training signal.

**These are the paper's choices, not standing rules for the skill.** Whether to apply any of them is a per-env decision — the default position for any new env is no filters. Add them only with explicit user approval, and record the rule and its rationale in the env's `NOTES.md`.

## 6. Alternative Action Sampling

The paper varies sampling strategy per environment because action spaces differ in structure. Two regimes that affect the strategy:

- **Enumerable action spaces** (e.g. admissible-action lists at each step): sample alternatives directly from the list. No LLM proposing needed.
- **Open or large action spaces** (free-form queries, web DOM interactions, typed tool calls): use a policy LLM to propose alternatives, often with temperature variation and deduplication.

The specific strategy for each env — K, temperature, fallback paths, env-specific constraints — depends on what the env actually exposes and what the user wants. the env's `NOTES.md` sketches the starting orientation per env, but the working choice is yours to make from the env's real interface and confirm with the user. Do not assume one env's strategy applies to another.

## 7. Hard constraint on data generation

Do not generate reflections without grounding them in actual observed next states. The paper compares to STaR (which generates rationales without environment grounding) and shows it can *degrade* performance. Every reflection in `reflection_sft.jsonl` must come from a real `(s_i, a_i^j, s_i^j)` triple where `s_i^j` was obtained by executing `a_i^j` in the env — never from the LLM imagining what the next state might be.

## 8. Reference

Zhang et al., "Agent Learning via Early Experience," arXiv:2510.08558, 2025. Full text at `paper.pdf`.
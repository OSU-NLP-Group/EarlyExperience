# Pitfalls — Accumulated Gotchas

Environment-specific failure modes observed while building early-experience data pipelines, filtered down to entries whose lesson generalizes beyond the env where they were first seen. Skim before starting a new env — if a previous entry sounds related, the new env may hit the same trap.

## Entry format

```
### <short title — descriptive, scannable>

**Env where observed**: <env name>

**What happened**: One or two sentences on the observed problem.

**Root cause**: One or two sentences on why it happened.

**How to avoid**: A specific, actionable check or change.

**Likely also affects**: List of envs that share the relevant characteristic and may hit the same issue.
```

## Entries

### JVM- or process-backed envs have no native state save/restore

**Env where observed**: ScienceWorld

**What happened**: For IWM / SR rollout the pipeline needs K alternative-action probes at every expert state `s_i`, then continue from `s_{i+1}`. The natural approach assumes the env supports "snapshot at `s_i` / restore after probing", but `ScienceWorldEnv` only exposes `load(task, var)` which resets to the variation's initial state. There is no native save/restore. The workaround is `env.load + replay expert_actions[0..i-1]` to reconstruct `s_i`, but each rebuild is O(i) env steps, so a trajectory of N steps with K probes per state costs roughly `K·N²/2` env operations.

**Root cause**: The env's internal state (Java process, Docker container, remote HTTP session, etc.) cannot be serialized back to disk, and multiple instances each spawn their own backend so there is no in-process way to "branch" the env either.

**How to avoid**: At recon time, probe whether the env has `save_state` / `load_state` / `__deepcopy__` / `clone` / `fork` methods. If it has none, budget for the O(K·N²) rebuild cost up front — use process-pool parallelism and don't be surprised when a "60k-step probing run" actually executes ~1M env operations.

**Likely also affects**: any JVM-backed env (TextWorld via py4j, Jericho IF gamebase, any env that calls out to a binary JAR), any Docker-isolated env (WebArena, WebShop's HTTP service), and any env whose state lives outside the Python process.

### LLM reflection generators leak supervision labels into the CoT by default

**Env where observed**: ScienceWorld

**What happened**: The paper's default reflection prompt (METHOD.md §4.3, quoted verbatim) uses input slots labeled "Expert Action (a_i)" and "Alternative Actions: 1. ... 2. ...". A reflection LLM handed such an input will echo those labels into its monologue at a non-trivial rate ("the expert action is X, which makes sense because…", "Action 1 would be less helpful because…") — observed ~16% in one batch. This breaks the entire point of the SR CoT: at inference the trained model has no privileged "expert" label, and the CoT must be the model's *own* reasoning that arrives at the chosen action. Training on text that announces the expert label corrupts the supervision signal.

**Root cause**: LLMs default to "essay justifying a known answer" mode when handed a labeled answer in the input — this is the default training distribution. The model has no inherent reason to suppress labels just because the next-token target says it should.

**How to avoid**: Two prompt-design rules, both must be applied:

1. **Remove the word "expert" / "selected" / "chosen" / "correct" / "right" / "best" / "optimal" from the input slot names**. Renaming "Expert Action" → "The action the agent takes here" drops leakage to near-zero. Add a banned-vocab list to the Guidelines block as a safety net.
2. **Don't refer to alternatives by numbered labels** ("Action 1", "Action 2", "a_i^1") in the input either; if the input must expose them, instruct the model to use natural inline phrasing in the output ("I could try X", "Another option I have is to Y"). At inference the model doesn't see a pre-enumerated list — it considers options in its own head, and the training target must read that way.

Also add a post-hoc detector that greps for the banned vocabulary as a last line of defense; expect a small residual rate (~0.3%) that should be dropped.

**Likely also affects**: every env's SR pipeline. The leakage mode is LLM-class-wide across providers.

### Long-form LLM generations occasionally duplicate the whole response

**Env where observed**: ScienceWorld

**What happened**: Across a full SR reflection batch (long-form CoTs 1.5k–3k chars each), ~2.9% of responses consisted of the same monologue written twice in a row — byte-identical first half, then a second identical copy starting immediately after the first (no separator, just joined mid-word). 100% of cases were exactly 2x repeats, never 3x or more. `frequency_penalty=0.3` reduced the rate from a higher baseline but didn't eliminate it; the failure mode is at the document level (the model fails to emit EOS after a complete answer and, conditional on its own complete output, decides "writing a similar answer" is the highest-probability continuation), and token-level penalties don't reach far enough to suppress it.

**Root cause**: LLM mode-collapse / EOS-failure at the end of long-form generation. Increasing `frequency_penalty` further (0.5–1.0) damages natural noun-repetition in the body of the text; raising sampling threshold compromises diversity.

**How to avoid**: Don't fight this in the prompt. Detect post-hoc: for each generation, check whether the first ~100 characters appear again later in the text with a similar-length second copy following — and dedup by keeping only the first copy. This recovers ~98% of doubled records as clean complete reflections; the residual (length mismatch > 50 chars between halves) is safer to drop than auto-dedup.

**Likely also applies**: every env's SR or LLM-CoT-generation step. The duplicate-doubling mode has been observed across LLM families and is essentially independent of env.

### SR reflections leak future info when the expert action changes location / opens a container

**Env where observed**: ALFWorld

**What happened**: SR grounds the reflection in the expert action's outcome `s_{i+1}`. For *location-changing* actions (`go to X`, `open Y`), `s_{i+1}` reveals the destination's contents — which the agent does NOT know at decision time. ~25% of navigation reflections (in one smoke) justified the move with "indeed I found bowl 1 there" / "now I see cup 1 and cup 2" — training the model to hallucinate seeing a place before going there.

**Root cause**: `s_{i+1}` of a move/open is *post-decision* information; the reflector LLM naturally writes it as already-observed. In-place actions (take/move/use — target visible in the current obs) don't have this problem.

**How to avoid**: System-prompt rule: justify location-changing / container-opening moves with **anticipatory** language ("going there *should* let me find …"), never as already-seen; only the CURRENT observation is fact. Post-hoc detector should fire ONLY on `go to`/`open` records and must ignore "expect to see…" phrasing — otherwise it's ~98% false positives (in-place "now I'm here, I can see X" is legitimate; "my best move"/"first choice" are natural language, not label leaks). This cuts the leak from ~25% to well under 1%.

**Likely also affects**: every embodied/navigation env's SR — ScienceWorld navigation, WebShop page transitions, BabyAI, any env where an action moves the agent and `s_{i+1}` reveals new state the agent couldn't see when choosing.

### Regex "leak" scans and verbatim checks are proxies — they don't judge data quality

**Env where observed**: AppWorld

**What happened**: An SR smoke printed 20/20 "clean" (0 banned-vocab hits, 20/20 verbatim code match) after a prompt fix. Treating the numbers as proof the data was good was the wrong move: the regex missed the actual quality dimensions that matter for training — whether the CoT reasoning is sound rather than generic parroting, whether it correctly interprets each alt's outcome as evidence, whether it doesn't invent facts, whether it converges on the committed action for the *right reason*. A pipeline that passes automated scans and fails on read-through is worse than one that visibly fails — training on 10k such records is money burned.

**Root cause**: Regex scans are dimensionality-collapsing. They flag one pattern of leakage (label vocabulary, forbidden phrases, code-not-matching), which the model can trivially route around without becoming more truthful. Real SR quality lives in *content* — coherence, faithfulness to inputs, non-genericity — none of which regex can see.

**How to avoid**: For every SR / reflection / summary batch, the smoke report is not complete until the agent has **read at least 5–10 sampled records in full** and written a short paragraph of judgment ("this one reasons well from alt X's error", "this one just restates the outcome without reasoning", "this one invents an entity not in the input"). Automated scans are fine as *pre-filters* to catch known failure modes at scale — never as the sole quality signal. `method_recap.md` says "look at the data you generated"; this pitfall is the reminder that "look at" means human (or careful agent) reading, not regex scans. Add to smoke script templates: after auto stats, print sampled records for review — but treat the read-through as the gating step, not the stats.

**Likely also affects**: every env's SR / reflection / summary phase; every phase that produces free-form LLM text as training target. Especially dangerous when the automated checks look "clean" (0 hits) — that is exactly when quality could still be terrible in ways automated checks never imagined.

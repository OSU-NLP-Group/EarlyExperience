# Method Recap

This is a short list of decisions where the right answer is non-obvious or runs against default coding instincts. Read `METHOD.md` for the formal definitions and the reflection prompt template; this file only covers what's easy to get wrong.

## Find expert trajectories before generating them

Don't reach for "let me generate experts with an LLM" first. Search in order:

1. Whether the env itself ships canonical solutions.
2. Whether the env's authors released expert data (repo, paper page, Hugging Face).
3. Whether the community has released expert data — follow-up papers or open datasets that reuse this env.
4. Only if none of the above: generate with an LLM, with explicit user confirmation before the run.

Generation is the last resort, not the first. Document the source and the URL in `envs/<env>/NOTES.md`.

## Filtering happens at data generation, after human approval

The three files under `envs/<env>/data/sft/` are the final, training-ready, already-filtered dataset. The trainer does not filter — it loads and trains.

But the filter rules themselves are not yours to invent. Before applying any filter, propose the rule to the user with the rationale and an estimate of its effect (how many samples it drops, what kind of samples). Wait for explicit approval. Approved filter rules go into `envs/<env>/NOTES.md`; samples that get dropped are still kept in intermediate artifacts (not in `sft/`) so the decision is reversible.

The starting position for any new env is **no filters**. Add them through this approval process only when there's evidence from the smoke run that a specific kind of bad sample is contaminating the data.

## Don't call an LLM when the action space is enumerable

Before reaching for an LLM to propose alternative actions, check what the action space actually looks like at this step.

If at state `s_i` the env exposes a finite admissible action list and the expert's action is one item from that list, sample alternatives by drawing from the list directly. No LLM call needed. Faster, cheaper, more reliably distinct than any LLM proposer.

LLM proposal is the right tool only when the action requires content that the env can't enumerate — a search query, a function call's argument string, a free-form text field.

A single env can mix both regimes within one trajectory: the first step might be a free-form search query (needs LLM) while every later step picks from a list of products and buttons (sample from the list). Don't apply one strategy uniformly — judge per step. Document the choice (per env, and per step type if mixed) in `envs/<env>/NOTES.md`.

## Look at the data you generated

A run is done when the pipeline finishes *and* the output has been read, not just when the script exits. A smoke run in particular should be aimed at as much state/action diversity from the env as is reasonable to cover — not just the easy or earliest parts of trajectories. After any batch (smoke or full), read enough of the produced JSONL by hand to form a real opinion of whether the data looks right. Anything that looks off or that you're not sure about goes to the user before scaling up or declaring the env done.

## Propose multiple alternative actions in a single LLM call

When the previous section's check says LLM proposal is needed, the naive approach is K separate API calls at high temperature. Don't do that. Make one call that explicitly asks for several distinct candidates, then deduplicate after canonicalization.

In-prompt diversity instruction is more reliable than temperature alone for getting actually-different candidates, and it's K-fold cheaper. Ask for a few extras so dedup doesn't leave you short. If after dedup there still aren't enough distinct alternatives, fall back — a second call, or random admissible actions. The fallback path exists; the loop-of-single-calls path does not.

## The reflection CoT must read as the model's own internal thinking

Reflection text in `reflection_sft.jsonl` becomes the model's training TARGET. At inference time the trained model has to produce this kind of text from scratch — with **no external label** distinguishing "expert" from "alternatives", **no pre-enumerated list** of options, and **no knowledge of any action's outcome** (it hasn't acted yet). Three default LLM behaviors that ruin SR data quality, all easy to miss:

1. **Don't let the CoT say "the expert action is X"** (or "the correct choice", "the right action", "the best option chosen by the expert"). The CoT is supposed to be the model's *own* reasoning that *arrives at* the chosen action. If the training target literally announces a privileged "expert" label, the model learns to expect a label it will never have at test time, and its inference-time CoT degrades into "the expert action is …" with no actual expert in sight.

2. **Don't let the CoT reference alternatives by external labels** ("Action 1", "Action 2", "Alternative a_i^1", etc.). At inference, the model is not picking from a labeled list — it's considering possibilities in its own head. The training text must use natural inline phrasing: "I could try X", "Another option I have is to Y", "I'm not sure if doing Z would help". Labels in the *input prompt* are fine (they're just for the reflection-generator LLM's convenience); labels in the *output CoT* are leaking the supervision into the training target.

3. **Don't let the CoT narrate the outcome of the action it settles on (or of any alternative) as already observed.** At decision time the agent has executed nothing, so it cannot know the resulting state. A CoT that says "the resulting state shows the door is now open", "using the thermometer, which reads 113°", "going to the kitchen, and now I see the fork", or "…which results in X" is leaking the post-decision next state into the target — training the model to hallucinate an outcome it will not have at inference. Every action, chosen or considered, must be reasoned about **anticipatorily**: "this should …", "I expect this will …", "to check whether …", never as a fact already seen. This leak is *induced by the reflection prompt itself* when the template hands the LLM the expert's next state and asks it to justify the action "grounded in its resulting state" (paper §4.3's own wording) — that instruction produces the narration directly. Fix: still give the generator the next states (it needs them to *judge* which option is better), but instruct it that those outcomes are for its private judgment only and must never be narrated; the monologue reasons toward the action anticipatorily. Note this is a different axis from the label leak — a reflection can be perfectly free of "expert"/"Action 1" and still leak the outcome.

Without an explicit Guideline forbidding all three, every capable LLM proposer defaults to "essay-justifying-a-known-answer" voice that includes them. The fix is one paragraph in the reflection prompt's Guidelines block plus a post-hoc check on a sample of the generated data: grep the CoTs for `expert`, `Action 1`, `Action 2`, and outcome-narration phrases like `the resulting state`, `which results in`, `now I see/find` — none should appear in correct SR data. One caution on the outcome check: a reference to the **current** observation `s_i` is legitimate ("the observation shows the door is closed" describes the state the agent is deciding from); only the **chosen action's result** is off-limits — so match phrasing that narrates a *post-action* state, not any mention of an observation.

## Prefer principle over constraint when the generator is a strong model

There are two ways to steer a CoT-generating LLM toward the data you want: **explain the principle** (what this data is for, why it must look a certain way, what the trained model will and won't have at inference) versus **enumerate constraints** (a checklist of "don't say X, don't do Y, banned phrase Z"). Both can appear in one prompt, but the balance should shift with the generator's capability:

- **Strong general-purpose LLM:** lead with the principle. A prompt that opens with "you are building training data; after training the agent will stand exactly where this monologue stands — it sees only the task and history, has not run the action, has no list of options, and must generate this reasoning itself; therefore anything the monologue leans on that the agent couldn't have at that instant teaches it to hallucinate" gets cleaner, more genuine output than the same behaviors listed as bare prohibitions. A capable model, told *why*, generalizes the intent to cases your checklist never enumerated — and the output reads as real reasoning rather than a model dancing around a rule list. In practice on an open-Python-action env, the same failure modes (schema-field fabrication, code-narration, forward-planning) that a long DON'T list only partially suppressed were nearly eliminated once the prompt explained the transfer principle behind them.

- **Weaker or smaller generator (including small models producing their own self-reflection):** lean more on explicit constraints — concrete banned phrases, WRONG-vs-RIGHT example pairs, a checklist — because a smaller model is less reliable at deriving the specific behaviors from an abstract principle. **But still state the principle**; constraints without the "why" produce brittle, gameable compliance. This is an author's judgment call per (env, model): decide how far to slide toward explicit constraints based on how much you trust the generator to reason from intent, and always smoke-read the output to see which register the model actually responds to.

The through-line: constraints tell the model what not to type; the principle tells it what the data is *for*. A strong model given the purpose writes better data than one handed only the rules. Encode the principle first; add constraints as needed for the model you're actually using.

## Huge / open action spaces change how many alternatives are worth collecting

Some environments have an action space so large that at any state there are effectively unbounded plausible API calls / argument combinations, and the overwhelming majority carry no useful signal (they error, or return data irrelevant to the task). AppWorld is the archetype: ~100 candidate endpoints per state, each with free-form args. Two consequences for how much alternative-action data to collect:

- **IWM: sample a subset, don't chase coverage.** The world model only needs enough (state, action, next-state) transitions to learn the dynamics; exhaustively probing every alternative mostly adds error-outcome transitions that dilute the signal. In practice on an open-action env, a full-coverage IWM set trained *worse* than a balanced subsample that capped the error fraction (and worse than plain IL), while the balanced subsample beat IL. Collect a bounded per-state sample with a sane data-vs-error mix; more is not better.

- **SR: K = 0 is a legitimate, sometimes best, choice.** SR's premise is that comparing the taken action against *genuinely competitive* alternatives yields insight. In a huge action space you usually cannot cheaply produce genuinely competitive alternatives — randomly sampled or exhaustively enumerated alternatives are almost never a real contender for the same subgoal, they're just unrelated calls. Feeding such alternatives into the reflection backfires two ways: with the alternatives named, the model learns to hallucinate a menu of options it was never given at inference; and (if the alternatives include fabricated-credential / placeholder calls the arg-filler invented) the model learns to *imitate* those bad calls — e.g. guessing passwords. The clean alternative is **K=0**: drop alternatives entirely and generate a pure "why is this action the sensible move here" CoT, grounded in the task, the history, and the real observed outcome (which the *author* sees to keep the reasoning on-track, but which never surfaces in the monologue). This is no longer paper-faithful SR (there is no alt-comparison), so **label it honestly in NOTES.md** as an IL+grounded-reflection variant — but for open-action-space envs it is often the only version that produces transferable reasoning instead of a hallucination-teaching artifact. Reserve K>0 for envs where you can actually construct plausible-but-suboptimal attempts at the *same* subgoal (different valid endpoint for the same data, different pagination, different operation order) — those give the reflection real material to compare.

## Some LLMs occasionally duplicate whole paragraphs in long generations

At temperature 0.7+ with long-form output (reflection CoTs run 1.5k–3k chars), some LLMs intermittently emit the same paragraph twice in a single response — observed ~4% rate in one run. This is a known LLM-class failure mode across major providers; if your provider doesn't suppress it server-side, handle it at post-processing time.

Mitigations, in order of cost:
- Add `frequency_penalty=0.3` to the API call. Note that some reasoning-mode configurations silently ignore sampling params — check your provider's docs and keep thinking / reasoning mode off if you want this to take effect.
- Post-hoc detector at the SFT-build step: if the first 100–150 chars of the CoT appear again in the second half, drop or re-roll. Cheap, near-zero false positives.
- Both is the prudent combination.

---

Use the available compute sensibly. A naive `for ... : client.chat.completions.create(...)` or `for ... : env.step(...)` loop is usually the first thing that comes to mind but rarely the right thing to ship — there's almost always free wall-clock on the table. What "sensibly" means here depends on the env's interface and the hardware actually available, and that's the agent's call.
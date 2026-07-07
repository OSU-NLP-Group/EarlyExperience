# WebShop

Simulated e-commerce shopping env, paper §B.2. The agent gets an instruction (`"Find me a lightweight running jacket under $50 in size medium"`), navigates a product-search interface with `search[...]` and `click[...]` actions, and eventually clicks the `buy` button on a chosen product.

## Upstream

- [`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent) — a research training framework that ships a batteries-included WebShop env in `agent_system/environments/env_package/webshop/`. This is the same family we base ALFWorld and AppWorld on.
- Underlying WebShop product database and gym: bundled with the verl-agent env package; do the upstream WebShop setup (product-index download, search-engine index build) per its own README.

## Modification strategy

Upstream verl-agent focuses on **RL training loops** — it wraps the WebShop gym with reward, done, and step semantics tuned for training. We keep that wrapper untouched and layer an **offline data-extraction pipeline** on top of it. Two conceptual pieces added:

1. **Expert replay** — WebShop's evaluation split ships gold trajectories (from human demonstrators). Replay each through the gym action-by-action, recording per-step `(state, expert_action)` pairs. The state is the paper-formatted search page or product page in text form; the expert action is one of `search[query]` / `click[item_id]` / `click[buy]`.
2. **Random-action prober** — at each expert state, sample K alternatives from the action space appropriate to that state (e.g. alternative `click[...]` targets on the current results page, or alternative `search[...]` queries for search states), step the gym on each from a re-cloned env replayed to that state, and record next states.
3. **Reflection builder** — standard SR: LLM writes a first-person reasoning trace that arrives at the expert action given the alt outcomes.

## Method mapping (EE ↔ WebShop)

| EE concept | WebShop realization |
|---|---|
| Expert action | one of `search[query]` / `click[item_id]` / `click[buy]` at each step of a gold trajectory |
| Alternative-action pool | K sampled alternatives from the state-appropriate action space |
| Next state | rendered next-page text (search results / product detail / buy confirmation) |
| Reflection target | expert action + LLM-authored monologue over alt outcomes |

**Key insight for WebShop**: the state-appropriate action space depends on the current page type. On a search-results page, alternatives are alternative `click[item_id]` targets from the results list. On a search state, alternatives are alternative `search[query]` reformulations. Sampling alternatives without conditioning on page type produces mostly invalid actions.

## Data output

Available in the [Hugging Face dataset](https://huggingface.co/datasets/osunlp/early-experience):

```
webshop_v2/
├── expert_sft.json
├── iwm_sft.json
└── reflection_sft.json
```

**⚠️ Format note**: `webshop_v2/` files use `.json` extension because they're prettified JSON arrays (not line-delimited JSONL). Downstream training loaders (e.g. LLaMA-Factory's default JSONL reader) need to preconvert to `.jsonl` first — trivial one-liner: `for r in json.load(open(...)): out.write(json.dumps(r) + '\n')`.

## Reproducibility notes

- **Trajectory source**: WebShop's shipped gold trajectories from human demonstrators (in the WebShop product database).
- **K for IWM**: sampled per state; the exact count varies with page type (search-results pages have more clickable alternatives than product-detail pages).
- **K for SR**: 3 alternatives per state.
- **State representation**: paper-formatted text rendering of the current WebShop page — same format the trained model sees at inference.
- **Anti-leak vocabulary filter for SR**: same as other envs.

For a full reproduction, install the verl-agent WebShop env package from upstream (including the search-engine index and product database bundled with it), and follow the modification strategy above.

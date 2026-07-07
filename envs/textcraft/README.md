# TextCraft

Text-based Minecraft-style crafting game (originally from [`archiki/ADaPT`](https://github.com/archiki/ADaPT)). At each reset the agent gets a goal item and a small set of recipes (target recipe + ≤10 distractors), and must reach the goal via `get` (gather base items) and `craft` (combine ingredients).

**Not in the paper.** Paper Appendix B covers 8 envs; TextCraft is not one of them. All per-env hyperparameters (K, sampling strategy, filter rules) are decided locally following the same conventions used for paper envs.

## Upstream

- Env-server framework: [`WooooDyy/AgentGym`](https://github.com/WooooDyy/AgentGym) — same family as ScienceWorld; TextCraft is a bundled env served over HTTP.

## Modification strategy

TextCraft needs **no server-side patches**. Two properties make it a cheap EE target:

1. **Admissible actions are client-derivable.** From `(commands_list, inventory)` alone we can enumerate all valid actions at any state (1× `inventory`, one `get` per base item in the recipe list, one `craft` per recipe whose ingredients are held). No server endpoint needed — in contrast to ScienceWorld's `/admissible_actions` patch.
2. **State observations are short structured text.** IWM next-state prediction can target the raw observation directly; no summarizer needed.

So the pipeline is purely client-side:

1. **Expert extractor** — reuses AgentGym's public `AgentTraj-L` dataset (TextCraft split, `textcraft_train.json` on Hugging Face; 374 trajectories, 2,992 (s,a) pairs).
2. **Alternative-action prober** — at each expert state, derive admissible client-side from `(commands_list, inventory)`, sample K non-expert actions uniformly, step the env on each. All 374 expert trajectories retained (no replay-based filter).
3. **Reflection builder** — standard SR monologue arriving at expert action given alt outcomes.

## Method mapping (EE ↔ TextCraft)

| EE concept | TextCraft realization |
|---|---|
| Expert action | one action from AgentTraj-L's gpt turn at each state |
| Alternative-action pool | K uniform samples from admissible-set (client-derived), minus the expert |
| Next state | short structured text (`"Got 8 terracotta"`, `"Crafted 1 minecraft:light_gray_dye"`, `"Could not find oak planks"`, ...) |
| Reflection target | expert action + LLM-authored monologue |

## Data output

Living on Google Drive (this repo does **not** vendor any SFT files):

```
ggdrive:Early-Experience-Reproduce/data/textcraft/
├── expert_sft.jsonl
├── (iwm — see textcraft_v2 for the latest iwm variant)
└── reflection_sft.jsonl

ggdrive:Early-Experience-Reproduce/data/textcraft_v2/
└── iwm_sft.v2.jsonl
```

## Reproducibility notes

- **Trajectory source**: `AgentGym/AgentTraj-L` (TextCraft split) on Hugging Face — all 374 trajectories retained; replay diagnostic shows 371/374 pass, but no pass-rate filter is applied.
- **K for IWM**: 5 non-expert alternatives per state, sampled uniformly from the client-derived admissible set.
- **K for SR**: 3 alternatives per state.
- **Concurrency**: TextCraft's env-server is pure Python (no JVM), so multi-worker probing scales with Python thread/process overhead — no separate-process-per-server needed.
- **Common re-implementation mistake**: treating "3 action verbs" as the action space. Per-state admissible actions number 5–30 because they combine with the current inventory and recipe list. IWM/SR alt sampling must enumerate this space, not pick from `{get, craft, inventory}`.

For a full reproduction, install the AgentGym env-server package from upstream and follow the modification strategy above.

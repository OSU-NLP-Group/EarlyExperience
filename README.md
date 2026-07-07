# Early Experience (ICML 2026)

<p align="center">
  <a href="https://arxiv.org/pdf/2510.08558"><img src="https://img.shields.io/badge/arXiv-2510.08558-b31b1b.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/🤗_HuggingFace-Coming_Soon-yellow" alt="Hugging Face"></a>
</p>

-----

## Introduction

Supported by [OSU NLP](https://github.com/OSU-NLP-Group) and [NeoCognition](https://neocognition.io/), this repo, done by [Xiangchao Chen](https://chasechen.xyz/), [Tinghui Zhu](https://darthzhu.github.io/), and [Kai Zhang](https://drogozhang.github.io/), contains open-source code and data reproductions for **"[Agent Learning via Early Experience](https://arxiv.org/pdf/2510.08558)"**.

The paper introduces a training paradigm sitting between imitation learning (IL) and reinforcement learning (RL) — one that is applicable to environments without verifiable reward signals or where long-horizon rollouts make RL impractical. The agent collects its own interaction data by proposing non-expert actions at expert-visited states, then uses the *resulting next states* as supervision — without any reward.

Two concrete methods under this paradigm:

- **Implicit World Modeling (IWM)** — train the policy to predict the next state given (state, action) on the rollout data, then continue with imitation on expert data. A two-stage pipeline.
- **Self-Reflection (SR)** — mix the reflection data (LLM-authored monologues comparing expert vs alternative actions) with expert data, and train with standard next-token prediction.

This repo ships:

- **A skill guide** (`skill/`) capturing the general methodology, method mapping, and pitfalls of generating EE data for any env — designed to be consumed by a code agent (any flavor) as a workflow reference.
- **Ten env-specific implementation notes** (`envs/<env>/`) with per-env modification strategies, data locations, and reproducibility notes.
- **Pre-generated SFT data** (expert / IWM / reflection) on Google Drive for every env we've stabilized.

## Data

All SFT files (expert / IWM / reflection) are available on [huggingface](https://huggingface.co/datasets/osunlp/early-experience):

## Using the skill

The [`skill/`](skill/) directory contains three files — `SKILL.md` (workflow), `method_recap.md` (design decisions), `pitfalls.md` (accumulated gotchas) — that together specify how to produce early-experience data for a new env.

To use with your code agent, copy the skill into your project and point the agent at `SKILL.md`:

```bash
cp -r skill /path/to/your/project/
```

Then instruct the agent: *"Read `skill/SKILL.md` before doing any early-experience data-generation work."* The skill walks the agent through method mapping, alternative-action sampling design, reflection generation, and the common re-implementation pitfalls.

## Main results

| Env | Model | IL | IWM | Δ vs IL | SR | Δ vs IL |
|---|---|---:|---:|---:|---:|---:|
| ALFWorld       | Qwen2.5 7B Instruct  | 58.6% | 68.8% | **+10.2%** | **74.2%** | **+15.6%** |
| WebShop        | Qwen2.5 7B Instruct  | 39.1% | 48.4% | **+9.3%**  | **55.5%** | **+16.4%** |
| BFCL           | Qwen2.5 7B Instruct  | 44.9% | **51.1%** | **+6.2%**  | 49.6% | +4.7% |
| TravelPlanner  | Qwen2.5 7B Instruct  | 25.0% | 29.4% | **+4.4%**  | **30.0%** | **+5.0%** |
| TextCraft      | Qwen2.5 7B Instruct  | 73.0% | **76.8%** | **+3.9%**  | 74.5% | +1.5% |
| ScienceWorld   | Qwen2.5 7B Instruct  | 65.4% | **68.6%** | **+3.2%**  | 66.0% | +0.6% |
| AppWorld       | Qwen2.5 14B Instruct | 43.7% | **59.7%** | **+16.0%** | 51.0% | +7.3% |

Note that this release uses a **unified training setup** across all envs — same base model class per env, same optimizer, same effective batch and learning rate — which differs slightly from the per-env hyperparameters described in the paper's Appendix B. All numbers above are the final checkpoint of a single one-shot training run (no ensembling, no re-runs).

## Environments

Ten envs are represented in this release. Seven have been stabilized and validated end-to-end (data → training → eval). Three are marked TBD — data ships as provisional, but the training-eval loop is not yet closed.

### ALFWorld — [envs/alfworld](envs/alfworld/)

Text-based household task env (paper §B.1). The agent solves ALFRED-style tasks like *"put a clean fork in cabinet"* by emitting admissible commands (`go to countertop 1`, `pick up fork 1`, ...) in a TextWorld-simulated home. IWM alternatives come from the fully-enumerable admissible-commands list — no LLM proposer needed. See [envs/alfworld/README.md](envs/alfworld/README.md) for the modification strategy and reproducibility notes.

### WebShop — [envs/webshop](envs/webshop/)

E-commerce shopping env (paper §B.2). The agent handles instructions like *"find a lightweight running jacket under $50"* through `search[...]` and `click[...]` actions. Alternatives at each expert step are sampled from the state-appropriate action space (alternative product clicks on a results page, alternative query reformulations for search states). See [envs/webshop/README.md](envs/webshop/README.md).

### BFCL v4 (multi-turn function calling) — [envs/bfcl_v4](envs/bfcl_v4/)

Multi-turn function-calling benchmark (paper §B.3). Because sim state is pure-Python and `copy.deepcopy`-able, alternative-action probing is exceptionally cheap: no re-replay, no HTTP env spin-up. IWM uses `K=10` alt names sampled structurally from the involved sim classes, with one LLM call filling all K argument sets per state. See [envs/bfcl_v4/README.md](envs/bfcl_v4/README.md).

### ScienceWorld — [envs/scienceworld](envs/scienceworld/)

Interactive science-lab env (paper §B.6). Requires the *admissible-action list at every state*; we patch AgentGym's HTTP env-server to expose an `/admissible_actions` endpoint backed by ScienceWorld's own combination generator. IWM samples `K=3` uniform non-expert alts; SR uses policy-proposed alts with refill-from-admissible. See [envs/scienceworld/README.md](envs/scienceworld/README.md).

### TravelPlanner — [envs/travelplanner](envs/travelplanner/)

Multi-day travel-planning env (paper §B.7). The whole plan is submitted in one turn, which makes SR the dominant methodological challenge: reflections must justify *specific slot fills* against constraint-stratified alternatives (budget / cuisine / min-nights / city-sequence / mode-chain). Two reflection variants ship: `full` (~124 words) and `compressed` (~50 words). See [envs/travelplanner/README.md](envs/travelplanner/README.md).

### WebArena — [envs/webarena](envs/webarena/) 🚧

Realistic full-stack web-navigation env (paper §B.8). **Status: TBD.** Provisional data ships on Drive; expert side is sourced from the [Agent Data Protocol (ADP)](https://arxiv.org/abs/2510.07059) paper's 4 WebArena-relevant expert sources. See [envs/webarena/README.md](envs/webarena/README.md).

### AppWorld — [envs/appworld](envs/appworld/)

Free-form code-action agent env. The agent writes Python that runs in a sandboxed REPL with access to mock apps (spotify, venmo, gmail, ...). The cleanest EE candidate we support — expert data is free (via the AppWorld SDK's ground-truth programs), and total pipeline cost measured at ~$3 for the full train split. Both `full` and `balanced` IWM variants ship. See [envs/appworld/README.md](envs/appworld/README.md).

### TextCraft — [envs/textcraft](envs/textcraft/)

Text-based Minecraft-style crafting game (not in paper). Included as an extra env exercising the same EE recipe on a simpler action space. Admissible actions are entirely client-derivable from `(commands_list, inventory)` — no server-side patch needed. See [envs/textcraft/README.md](envs/textcraft/README.md).

Tau-Bench and SearchQA will come soon 🚧!
---

For each env, the linked README describes:
- The upstream env repo (with license and version pin).
- Our high-level modification strategy — the *why* and *what* of the changes, not code-level detail. Enough to guide a reimplementation on your own fork.
- The method mapping (how EE's expert / IWM alt / SR reflection concepts realize concretely in this env).
- Data output locations on Google Drive.
- Key hyperparameters (K, sampling strategy) and re-implementation gotchas.

## Citation
If you find this code or data helpful, please cite:

```bibtex
@inproceedings{Ahang2026EarlyExperience,
title={Agent Learning via Early Experience},
author={Kai Zhang and Xiangchao Chen and Bo Liu and Tianci Xue and Zeyi Liao and Zhihan Liu and Xiyao Wang and Yuting Ning and Zhaorun Chen and Xiaohan Fu and Jian Xie and Yuxuan Sun and Boyu Gou and Qi Qi and Zihang Meng and Jianwei Yang and Ning Zhang and Xian Li and Ashish Shah and Dat Huynh and Hengduo Li and Zi Yang and Xuefei Cao and Lawrence Keunho Jang and Shuyan Zhou and Jiacheng Zhu and Huan Sun and Jason E Weston and Yu Su and Yifan Wu},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=N3dXUHY5dD}
}
```

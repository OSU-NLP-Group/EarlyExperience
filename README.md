# Early Experience (ICML 2026)

<p align="center">
  <a href="https://arxiv.org/pdf/2510.08558"><img src="https://img.shields.io/badge/arXiv-2510.08558-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/osunlp/early-experience"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow" alt="Hugging Face Dataset"></a>
</p>

---

## Introduction

Supported by [OSU NLP](https://github.com/OSU-NLP-Group) and [NeoCognition](https://neocognition.io/), this repo, done by [Xiangchao Chen](https://chasechen.xyz/), [Tinghui Zhu](https://darthzhu.github.io/), and [Kai Zhang](https://drogozhang.github.io/), contains open-source code and data reproductions for **"[Agent Learning via Early Experience](https://arxiv.org/pdf/2510.08558)"**.

The paper introduces a training paradigm sitting between imitation learning (IL) and reinforcement learning (RL) — one that is applicable to environments without verifiable reward signals or where long-horizon rollouts make RL impractical. The agent collects its own interaction data by proposing non-expert actions at expert-visited states, then uses the *resulting next states* as supervision — without any reward.

Two concrete methods under this paradigm:

- **Implicit World Modeling (IWM)** — train the policy to predict the next state given (state, action) on the rollout data, then continue with imitation on expert data. A two-stage pipeline.
- **Self-Reflection (SR)** — mix the reflection data (LLM-authored monologues comparing expert vs alternative actions) with expert data, and train with standard next-token prediction.

This repo ships:

- **A skill guide** (`skill/`) capturing the general methodology, method mapping, and pitfalls of generating EE data for any env — designed to be consumed by a code agent (any flavor) as a workflow reference.
- **Ten env-specific implementation notes** (`envs/<env>/`) that serve two purposes: (1) enough detail to reproduce our numbers on the paper's envs, and (2) worked examples of what "porting a new agentic env into the EE framework" actually looks like. Anyone can pick up the skill and add their own env by following the same recipe.
- **Pre-generated SFT data** (expert / IWM / reflection) on Hugging Face for every env in this release.

## Data

All SFT files (expert / IWM / reflection) are available on [huggingface](https://huggingface.co/datasets/osunlp/early-experience).

## Using the skill

The [`skill/`](skill/) directory provides a code-agent skill that lets any agentic environment quickly adopt the early-experience learning paradigm — both **Implicit World Modeling** and **Self-Reflection** (see [the paper](https://arxiv.org/pdf/2510.08558) for the method itself). More at [`skill/README.md`](skill/README.md).

Drop it into your project in one line and point the agent at `SKILL.md`:

```bash
git clone https://github.com/OSU-NLP-Group/EarlyExperience.git && cp -r EarlyExperience/skill /path/to/your/project/
```

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

> Note that this release uses a **unified training setup** across all envs — same base model class per env, same optimizer, same effective batch and learning rate — which differs slightly from the per-env hyperparameters described in the paper's Appendix B. All numbers above are the final checkpoint of a single one-shot training run (no ensembling, no re-runs).

## Environments

Ten envs in this release. Seven with the full loop shipped (data → training → eval); three more in progress.

### ALFWorld — [envs/alfworld](envs/alfworld/)

Text-based household task env (paper §B.1). The agent solves ALFRED-style tasks like *"put a clean fork in cabinet"* by emitting admissible commands (`go to countertop 1`, `pick up fork 1`, ...) in a TextWorld-simulated home. IWM alternatives come from the fully-enumerable admissible-commands list — no LLM proposer needed. See [envs/alfworld/README.md](envs/alfworld/README.md) for the modification strategy and reproducibility notes.

### WebShop — [envs/webshop](envs/webshop/)

E-commerce shopping env (paper §B.2). The agent handles instructions like *"find a lightweight running jacket under $50"* through `search[...]` and `click[...]` actions. Alternatives at each expert step are sampled from the state-appropriate action space (alternative product clicks on a results page, alternative query reformulations for search states). See [envs/webshop/README.md](envs/webshop/README.md).

### BFCL v4 (multi-turn function calling) — [envs/bfcl_v4](envs/bfcl_v4/)

Multi-turn function-calling benchmark (paper §B.3). Because sim state is pure-Python and `copy.deepcopy`-able, alternative-action probing needs no re-replay or HTTP env spin-up. IWM uses `K=10` alt names sampled structurally from the involved sim classes, with one LLM call filling all K argument sets per state. See [envs/bfcl_v4/README.md](envs/bfcl_v4/README.md).

### ScienceWorld — [envs/scienceworld](envs/scienceworld/)

Interactive science-lab env (paper §B.6). Requires the *admissible-action list at every state*; we patch AgentGym's HTTP env-server to expose an `/admissible_actions` endpoint backed by ScienceWorld's own combination generator. IWM samples `K=3` uniform non-expert alts; SR uses policy-proposed alts with refill-from-admissible. See [envs/scienceworld/README.md](envs/scienceworld/README.md).

### TravelPlanner — [envs/travelplanner](envs/travelplanner/)

Multi-day travel-planning env (paper §B.7). The whole plan is submitted in one turn, which makes SR the dominant methodological challenge: reflections must justify *specific slot fills* against constraint-stratified alternatives (budget / cuisine / min-nights / city-sequence / mode-chain). See [envs/travelplanner/README.md](envs/travelplanner/README.md).

### AppWorld — [envs/appworld](envs/appworld/)

Free-form code-action agent env (not in paper). The agent writes Python that runs in a sandboxed REPL with access to mock apps (spotify, venmo, gmail, ...). Expert data comes from the AppWorld SDK's ground-truth programs (no LLM cost), so IWM cost is one proposer call per state plus K env probes. See [envs/appworld/README.md](envs/appworld/README.md).

### TextCraft — [envs/textcraft](envs/textcraft/)

Text-based Minecraft-style crafting game (not in paper). Admissible actions are entirely client-derivable from `(commands_list, inventory)` — no server-side patch needed. See [envs/textcraft/README.md](envs/textcraft/README.md).

### Tau-Bench — [envs/tau-bench](envs/tau-bench/) 🚧

Customer-service env (paper §B.4). **Coming soon.** See [envs/tau-bench/README.md](envs/tau-bench/README.md).

### SearchQA — [envs/searchqa](envs/searchqa/) 🚧

Multi-hop question-answering env (paper §B.5). **Coming soon.** See [envs/searchqa/README.md](envs/searchqa/README.md).

### WebArena — [envs/webarena](envs/webarena/) 🚧

Realistic full-stack web-navigation env (paper §B.8). **In progress.** Expert side is sourced from the [Agent Data Protocol (ADP)](https://arxiv.org/abs/2510.07059) paper's 4 WebArena-relevant expert sources. See [envs/webarena/README.md](envs/webarena/README.md).

---

For each env, the linked README describes:
- The upstream env repo (with license and version pin).
- Our high-level modification strategy — the *why* and *what* of the changes, not code-level detail. Enough to guide a reimplementation on your own fork.
- The method mapping (how EE's expert / IWM alt / SR reflection concepts realize concretely in this env).
- Data output locations on Hugging Face.
- Key hyperparameters (K, sampling strategy) and re-implementation gotchas.

## Citation
If you find this code or data helpful, please cite:

```bibtex
@inproceedings{Zhang2026EarlyExperience,
title={Agent Learning via Early Experience},
author={Kai Zhang and Xiangchao Chen and Bo Liu and Tianci Xue and Zeyi Liao and Zhihan Liu and Xiyao Wang and Yuting Ning and Zhaorun Chen and Xiaohan Fu and Jian Xie and Yuxuan Sun and Boyu Gou and Qi Qi and Zihang Meng and Jianwei Yang and Ning Zhang and Xian Li and Ashish Shah and Dat Huynh and Hengduo Li and Zi Yang and Xuefei Cao and Lawrence Keunho Jang and Shuyan Zhou and Jiacheng Zhu and Huan Sun and Jason E Weston and Yu Su and Yifan Wu},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=N3dXUHY5dD}
}
```

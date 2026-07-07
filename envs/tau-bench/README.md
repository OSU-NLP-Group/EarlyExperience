# Tau-Bench

> **Status: TBD.** Reproduction is not yet stable enough to release. Provisional data is uploaded but not yet validated against a trained model.

Customer-service env, paper §B.4 (from [Yao et al., 2024](https://arxiv.org/abs/2406.12045)). Multi-turn tool use against typed APIs plus a LM-simulated customer plus a policy/wiki document the agent must adhere to. Paper §B.4 scopes to the **Retail** subset.

## Upstream

- Env: [`sierra-research/tau-bench`](https://github.com/sierra-research/tau-bench) (Sierra Research).

## Modification strategy (planned)

TBD — full description will be added once the Tau-Bench reproduction stabilizes.

## Data output (provisional)

Provisional files in the [Hugging Face dataset](https://huggingface.co/datasets/osunlp/early-experience) under `tau-bench/`, not yet validated end-to-end:

```
tau-bench/
├── expert_sft.jsonl
├── iwm_sft.jsonl
└── reflection_sft.jsonl
```

## Reproducibility notes

TBD — hyperparameters, K, alt-action sampling regime, and the LM-user-simulator handling will be documented once validated.

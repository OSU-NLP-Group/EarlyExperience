# SearchQA

> **Status: TBD.** Reproduction is not yet stable enough to release. Pipeline scripts and NOTES exist in this directory as work-in-progress reference; the data on Drive is provisional and not yet validated against a trained model.

Multi-hop question-answering env, paper §B.5. The agent issues `<search>` queries against a Wikipedia (wiki-18) retrieval server, reads back `<information>...</information>`, and eventually emits `<answer>...</answer>`.

## Upstream

- Paper reference implementation: SearchR1 (author's paper-faithful codebase).
- Underlying retrieval: e5 dense retrieval + FAISS index over wiki-18 (~50 GB index).

## Modification strategy (planned)

TBD — full description will be added once the SearchQA reproduction stabilizes.

## Data output (provisional)

```
ggdrive:Early-Experience-Reproduce/data/searchqa/
├── expert_sft.jsonl
├── iwm_sft.jsonl
├── reflection_sft_A.jsonl    # two reflection variants shipped for A/B; use A by default
└── reflection_sft_B.jsonl
```

## Reproducibility notes

TBD — hyperparameters, K, and the two reflection-variant design will be documented once validated.

# SearchQA

> **In progress.** Full pipeline description and results coming soon.

Multi-hop question-answering env, paper §B.5. The agent issues `<search>` queries against a Wikipedia (wiki-18) retrieval server, reads back `<information>...</information>`, and eventually emits `<answer>...</answer>`.

## Upstream

- Paper reference implementation: SearchR1 (author's paper-faithful codebase).
- Underlying retrieval: e5 dense retrieval + FAISS index over wiki-18 (~50 GB index).

## Modification strategy

Coming soon.

## Data output

Available in the [Hugging Face dataset](https://huggingface.co/datasets/osunlp/early-experience) under `searchqa/`:

```
searchqa/
├── expert_sft.jsonl
├── iwm_sft.jsonl
├── reflection_sft_A.jsonl    # two reflection variants shipped for A/B; use A by default
└── reflection_sft_B.jsonl
```

## Reproducibility notes

Coming soon.

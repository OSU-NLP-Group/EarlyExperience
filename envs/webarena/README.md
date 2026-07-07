# WebArena

> **Status: TBD.** Reproduction is not yet stable enough to release. Pipeline is not yet documented; provisional data is uploaded but not yet validated against a trained model.

Realistic web-navigation env, paper §B.8. The agent navigates a full-stack simulated web application (Gitlab, Reddit-like forum, e-commerce admin, maps, ...) using DOM interaction to complete natural-language tasks.

## Upstream

- Env: [`web-arena-x/webarena`](https://github.com/web-arena-x/webarena) — the WebArena benchmark, browser-backed via [BrowserGym](https://github.com/ServiceNow/BrowserGym).

## Modification strategy (planned)

TBD — full description will be added once the WebArena reproduction stabilizes.

## Data output (provisional)

Provisional files in the [Hugging Face dataset](https://huggingface.co/datasets/osunlp/early-experience) under `webarena/`, not yet validated end-to-end:

```
webarena/
├── iwm_sft.jsonl        # ⚠️ prettified JSON array, not JSONL — preconvert before loading
└── reflection_sft.jsonl
```

Expert data comes from the [ADP (Agent Data Protocol)](https://arxiv.org/abs/2510.07059) paper's 4 WebArena-relevant sources (`go-browse-wa`, `mind2web`, `nnetnav-live`, `nnetnav-wa`) rather than a single canonical expert file. Those source expert files are hosted separately.

## Reproducibility notes

TBD.

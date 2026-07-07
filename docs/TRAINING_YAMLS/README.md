# LLaMA-Factory Training Configs

每个 env 一个目录，按 paper Appendix B 的超参对齐。Base model 默认 `Qwen/Qwen3.5-9B`。

## 一、训练流程概念

Paper §3-§4 定义了三种方法，对应不同的训练 pipeline：

| Method | Stage 1 (warm-up SFT) | Stage 2 (final SFT) |
|---|---|---|
| **IL** (Imitation Learning / Behavior Cloning baseline) | — | `expert_sft.jsonl` |
| **IWM** (Implicit World Modeling) | `iwm_sft.jsonl` 1 epoch | `expert_sft.jsonl`，从 stage 1 ckpt 继续 |
| **SR** (Self-Reflection) | — | `expert_sft.jsonl` + `reflection_sft.jsonl` mix |

- IL：直接训 expert，作为 baseline
- IWM：两阶段 = warm-up（next-state prediction）+ continue（imitation），论文 §3 "two-stage pipeline"
- SR：一步 mix 训练，论文 §4.3 "we mix the self-reflection data D_refl with the expert dataset D_expert"

## 二、文件布局（per env）

```
docs/TRAINING_YAMLS/<env>/
├── dataset_info.json        # 注册 3 个数据集（expert, iwm, reflection）+ 必要 tags/columns
├── il.yaml                  # IL baseline
├── iwm_stage1.yaml          # IWM 阶段 1：训 iwm_sft.jsonl
├── iwm_stage2.yaml          # IWM 阶段 2：load stage1 ckpt，训 expert_sft.jsonl
└── sr.yaml                  # SR：mix expert + reflection
```

## 三、用法

### 准备数据

把对应 env 的 `dataset_info.json` 放在 LLaMA-Factory 的 `data/` 目录下（或在 yaml 里通过 `dataset_dir` 指定）。yaml 默认 `dataset_dir: /mnt/data/xiangchao/verl-agent-ee-final/envs/<env>/data/sft`，请按实际路径调整。

### 跑 IL（最简单）

```bash
llamafactory-cli train docs/TRAINING_YAMLS/scienceworld/il.yaml
```

### 跑 IWM（两阶段）

```bash
# stage 1: WM warm-up
llamafactory-cli train docs/TRAINING_YAMLS/scienceworld/iwm_stage1.yaml

# stage 2: 接着训 expert，先在 iwm_stage2.yaml 里改 model_name_or_path 指向 stage 1 输出
#   model_name_or_path: saves/scienceworld_iwm_stage1
llamafactory-cli train docs/TRAINING_YAMLS/scienceworld/iwm_stage2.yaml
```

### 跑 SR

```bash
llamafactory-cli train docs/TRAINING_YAMLS/scienceworld/sr.yaml
```

## 四、Mix 策略（SR）

paper §4.3 默认 `mix_strategy: concat`（直接拼接两个 jsonl）。如果 reflection 数远少于 expert，可以改成：

```yaml
mix_strategy: interleave_over
interleave_probs: "0.5,0.5"   # 顺序对应 dataset 字段里 dataset 的顺序
```

## 五、Per-env paper-aligned 超参速查

来源：paper Appendix B.1 ~ B.7。textcraft 不在 paper，沿用 ALFWorld 同配置。

| Env | IL epochs | IL lr | bs (effective) | ctx | 备注 |
|---|---:|---:|---:|---:|---|
| ScienceWorld | 1 | 5e-6 | 32 | 4096 | §B.6 |
| SearchQA | 3 | 1e-5 | 128 (4 GPU × 2 × 16 accum) | 8192 | §B.5 |
| BFCLv3 | 2 | 1e-5 | 16 | 8192 | §B.3，epochs 未指明，沿 ALFWorld |
| Tau-Bench | 6 | 1e-5 | 16 | 8192 | §B.4，IWM 阶段 1 用 lr=5e-6（特例） |
| TravelPlanner | 5 | 1e-5 | 128 (8 GPU × 16) | 32768 | §B.7，SR `cutoff_len` 同 32K，max_gen 8K |
| TextCraft | 2 | 1e-5 | 16 | 4096 | paper 外，沿 ALFWorld |

> Tau-Bench 的 IWM stage 1 用 lr **5e-6**（paper §B.4 明文），是所有 env 里唯一一个 IWM warm-up 跟 IL 用不同 lr 的；其他 env IL/IWM/SR 都用同一个 lr。

## 六、Base model + Template

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
template: qwen3_5_nothink   # paper-faithful: paper 用 Qwen2.5（无 thinking），我们禁用 thinking 对齐
```

LLaMA-Factory `src/llamafactory/data/template.py` 注册了两个 Qwen3.5 变体：

| Template | Thinking | Tool format | 用法 |
|---|---|---|---|
| `qwen3_5` | ON（默认） | XML（`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`） | 想用 thinking mode 时 |
| **`qwen3_5_nothink`** | **OFF** | XML（同上） | **paper-faithful 默认（我们用这个）** |

> Qwen3.5 的 tool 渲染是 **XML 格式**（跟 Qwen3-Coder 一致），跟 Qwen2.5 的 JSON 格式不同。`format_function` 会自动把 OpenAI FC 的 `tool_calls` 字段渲染成 XML，**不需要我们改数据**。

## 七、Tool 配置 per env（哪些 env 在 yaml/dataset_info 里加了 tool 字段）

| Env / 文件 | 顶层 `tools` 字段 | message `tool_calls` / `role=tool` | dataset_info 加了什么 | 备注 |
|---|---|---|---|---|
| tau-bench/expert,reflection | ✅ 16 schemas | ✅ | `columns.tools` + `function_tag` + `observation_tag` | qwen3_5 模板会自动把顶层 tools 注入 system message |
| tau-bench/iwm | ✅ 16 schemas | ❌（纯 single-turn 文本） | 同上 | tools 列保留也无害 |
| bfcl_v4/\*_fc | ❌（schema 在 system prompt 里手写） | ✅ tool_calls 在 | `function_tag` + `observation_tag`，不加 `columns.tools` | **⚠️ 训练目标 XML vs system prompt JSON 不一致；建议改用 text 版** |
| bfcl_v4/\*_text | ❌ | ❌（已 flatten 成 plain Python 字符串） | 无 tool 配置 | **推荐用这个**，跟 BFCL prompting-mode evaluator 兼容 |
| 其他 env | ❌ | ❌ | 无 tool 配置 | scienceworld / searchqa / textcraft / travelplanner — 都是 plain text |

**建议 bfcl_v4 用 text 版**：yaml 默认就是 `dataset: bfcl_expert_text`，不用改。`*_fc` 系列若想用，要么改下游 evaluator 接受 XML 格式，要么重 build 数据时把 system prompt 里的 JSON schema 删掉、改用顶层 `tools` 字段让模板自己注入。


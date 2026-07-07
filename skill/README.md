# Skill: Early Experience Data Generation

This directory is a self-contained workflow guide for a code agent to generate SFT-ready training data — expert, IWM (Implicit World Modeling), and reflection — for any agent environment, following the "Agent Learning via Early Experience" paper (arXiv:2510.08558).

## What's in here

- **`SKILL.md`** — the main workflow router. Hard rules, output shape, decision gates. Read this first.
- **`METHOD.md`** — paper method definitions (IWM, SR, formulas, reflection prompt template). Source of truth for the method.
- **`method_recap.md`** — short list of non-obvious decisions that are easy to get wrong.
- **`pitfalls.md`** — accumulated env-specific gotchas whose lesson generalizes across envs.
- **`NOTES_TEMPLATE.md`** — copy this when starting a new env's decision log.
- **`paper.pdf`** — the original paper, for cross-reference.

## Using it

Drop this directory into your project and point your code agent at `SKILL.md`:

```bash
git clone https://github.com/OSU-NLP-Group/EarlyExperience.git && cp -r EarlyExperience/skill /path/to/your/project/
```

Then tell the agent: *"Read `skill/SKILL.md` before doing any early-experience data-generation work."*

The skill walks the agent through method mapping, alternative-action sampling design, reflection generation, and the common re-implementation pitfalls — enough to produce EE data for a new env with the same recipe used for the paper's envs.

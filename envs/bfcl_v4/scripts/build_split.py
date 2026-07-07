"""
Realize the Path-A train/eval split that NOTES.md committed to.

  random.seed(42)
  train_pool_ids = random.sample(all_base_case_ids, 150)   # 75% of 200
  expert_ids     = train_pool_ids ∩ {Opus valid==true}     # ~122 cases
  heldout_ids    = all_base_case_ids - train_pool_ids      # 50 cases — held-out I.D. eval

Writes 5 JSON files (all_ids, valid_ids, failing_ids, train_pool_ids, expert_ids, heldout_ids)
under envs/bfcl_v4/data/split/. The lists are sorted by case index for human readability;
order is irrelevant to training but matters for diffability / reproducibility.

Run:
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/build_split.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = (
    REPO_ROOT
    / "envs/bfcl_v4/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
)
RAW_DIR = REPO_ROOT / "envs/bfcl_v4/data/raw"
SPLIT_DIR = REPO_ROOT / "envs/bfcl_v4/data/split"

SEED = 42
TRAIN_POOL_SIZE = 150  # 75% of 200


def _idx(case_id: str) -> int:
    return int(case_id.rsplit("_", 1)[-1])


def load_all_base_ids() -> list[str]:
    ids = []
    with (DATA_DIR / "BFCL_v4_multi_turn_base.json").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line)["id"])
    return sorted(ids, key=_idx)


def load_failing_ids() -> set[str]:
    """Opus FC's score file: line 0 = summary, lines 1+ = per-failed-case records."""
    failing = set()
    with (RAW_DIR / "opus_base_score.json").open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:
                continue
            o = json.loads(line)
            if not o.get("valid", True):
                failing.add(o["id"])
    return failing


def main() -> None:
    all_ids = load_all_base_ids()
    failing_ids = load_failing_ids()
    valid_ids = set(all_ids) - failing_ids
    assert len(all_ids) == 200, f"expected 200 Base cases, got {len(all_ids)}"
    assert len(failing_ids) == 38, f"expected 38 failing, got {len(failing_ids)}"
    assert len(valid_ids) == 162, f"expected 162 valid, got {len(valid_ids)}"

    rng = random.Random(SEED)
    train_pool_ids = sorted(rng.sample(all_ids, TRAIN_POOL_SIZE), key=_idx)
    heldout_ids = sorted(set(all_ids) - set(train_pool_ids), key=_idx)
    expert_ids = sorted(set(train_pool_ids) & valid_ids, key=_idx)
    dropped_in_pool = sorted(set(train_pool_ids) - valid_ids, key=_idx)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "all_ids.json": all_ids,
        "failing_ids.json": sorted(failing_ids, key=_idx),
        "valid_ids.json": sorted(valid_ids, key=_idx),
        "train_pool_ids.json": train_pool_ids,
        "expert_ids.json": expert_ids,
        "heldout_ids.json": heldout_ids,
        "dropped_from_train_pool_ids.json": dropped_in_pool,
    }
    for name, lst in out.items():
        (SPLIT_DIR / name).write_text(json.dumps(lst, indent=2))

    print(f"seed                              : {SEED}")
    print(f"all Base cases                    : {len(all_ids)}")
    print(f"Opus valid==true                  : {len(valid_ids)}")
    print(f"Opus failing                      : {len(failing_ids)}")
    print(f"75% train_pool                    : {len(train_pool_ids)}")
    print(f"  ∩ valid → expert_ids            : {len(expert_ids)}")
    print(f"  ∩ failing → dropped_from_pool   : {len(dropped_in_pool)}")
    print(f"25% heldout                       : {len(heldout_ids)}")
    print()
    print(f"split files written to: {SPLIT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

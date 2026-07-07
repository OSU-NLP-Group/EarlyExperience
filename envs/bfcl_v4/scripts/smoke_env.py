"""
Phase 0 env smoke for BFCL v4.

Verifies three things before any pipeline code is written:
1. GT end-to-end replay on multi_turn_base_0 (sanity that the sim API stack + evaluator works).
2. deepcopy probe on every multi-turn sim API class — our IWM design relies on per-state
   in-memory cloning, which SciW couldn't do (JVM); we MUST confirm BFCL allows it.
3. (handled by smoke_opus_replay.py) Opus trajectory replay.

Run with:
    conda run -n bfcl --no-capture-output python envs/bfcl_v4/scripts/smoke_env.py
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    CLASS_FILE_PATH_MAPPING,
    STATELESS_CLASSES,
    execute_multi_turn_func_call,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = (
    REPO_ROOT
    / "envs/bfcl_v4/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
)

# 8 sim API classes used by multi_turn cases (NOT the v4-only memory/web_search ones)
MULTI_TURN_CLASSES = [
    "GorillaFileSystem",
    "MathAPI",
    "MessageAPI",
    "TwitterAPI",
    "TicketAPI",
    "TradingBot",
    "TravelAPI",
    "VehicleControlAPI",
]


# ---------------------------------------------------------------------------
# (1) GT end-to-end on multi_turn_base_0
# ---------------------------------------------------------------------------


def load_case(case_id: str = "multi_turn_base_0") -> tuple[dict, dict]:
    """Return (question_entry, ground_truth_entry) for one case."""
    q_path = DATA_DIR / "BFCL_v4_multi_turn_base.json"
    a_path = DATA_DIR / "possible_answer/BFCL_v4_multi_turn_base.json"

    q_entry = None
    a_entry = None
    with q_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o["id"] == case_id:
                q_entry = o
                break
    with a_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o["id"] == case_id:
                a_entry = o
                break
    assert q_entry is not None, f"case {case_id} not found"
    assert a_entry is not None, f"case {case_id} possible_answer not found"
    return q_entry, a_entry


def probe_gt_end_to_end() -> bool:
    """Run multi_turn_base_0's ground truth through execute_multi_turn_func_call,
    confirm every call executes without raising, and dump final sim state."""
    print("\n=== Probe 1: GT end-to-end on multi_turn_base_0 ===")
    q, a = load_case("multi_turn_base_0")

    initial_config = q["initial_config"]
    involved_classes = q["involved_classes"]
    gt_turns: list[list[str]] = a["ground_truth"]

    print(f"  involved_classes: {involved_classes}")
    print(f"  # user turns: {len(gt_turns)}")
    print(f"  total GT calls: {sum(len(t) for t in gt_turns)}")

    all_ok = True
    last_instances = None

    for turn_idx, turn_calls in enumerate(gt_turns):
        try:
            results, instances = execute_multi_turn_func_call(
                func_call_list=turn_calls,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name="smoke_probe_gt",
                test_entry_id=q["id"],
                long_context=False,
                is_evaL_run=False,
            )
            print(f"  turn {turn_idx}: {len(turn_calls)} calls → {len(results)} results ✓")
            for c, r in zip(turn_calls, results):
                # truncate long results so output is readable
                rr = (r[:80] + "…") if isinstance(r, str) and len(r) > 80 else r
                print(f"      {c}  →  {rr}")
            last_instances = instances
        except Exception as e:  # noqa: BLE001
            print(f"  turn {turn_idx}: FAILED  ({type(e).__name__}: {e})")
            traceback.print_exc()
            all_ok = False
            break

    if all_ok and last_instances is not None:
        print("  final sim state (per class):")
        for cls_name, inst in last_instances.items():
            state = {
                k: v for k, v in vars(inst).items() if not k.startswith("_")
            }
            # truncate large state for readability
            s = json.dumps(state, default=str)
            s = s if len(s) < 400 else s[:400] + "…"
            print(f"    {cls_name}: {s}")
    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# (2) Deepcopy probe on each multi-turn class
# ---------------------------------------------------------------------------


def _representative_initial_config(class_name: str) -> dict:
    """Find a Base case that uses this class and return its initial_config slice.
    For stateless classes, returns {}.
    """
    if class_name in STATELESS_CLASSES:
        return {}
    q_path = DATA_DIR / "BFCL_v4_multi_turn_base.json"
    with q_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if class_name in o.get("involved_classes", []):
                cfg = o.get("initial_config", {}).get(class_name, {})
                return cfg
    return {}


def _instantiate(class_name: str):
    """Instantiate a class and load its initial scenario."""
    module = importlib.import_module(CLASS_FILE_PATH_MAPPING[class_name])
    cls = getattr(module, class_name)
    inst = cls()
    if class_name not in STATELESS_CLASSES:
        cfg = _representative_initial_config(class_name)
        inst._load_scenario(copy.deepcopy(cfg), long_context=False)
    return inst


def _public_state(inst) -> dict:
    return {k: v for k, v in vars(inst).items() if not k.startswith("_")}


def _mutate_in_place(inst, class_name: str) -> bool:
    """Pick a non-trivial mutation that's guaranteed to change visible state.
    Returns True if mutation was applied, False if no obvious mutation available."""
    if class_name == "GorillaFileSystem":
        # touch a file
        inst.touch(file_name="__smoke_probe__.txt")
        return True
    if class_name == "MessageAPI":
        # add a contact (most APIs expose this without auth)
        try:
            inst.add_contact(user_name="__smoke__")
            return True
        except Exception:
            pass
    if class_name == "TwitterAPI":
        try:
            inst.post_tweet(content="__smoke__", tags=[], mentions=[])
            return True
        except Exception:
            pass
    if class_name == "TicketAPI":
        try:
            inst.create_ticket(title="__smoke__", description="probe")
            return True
        except Exception:
            pass
    if class_name == "TradingBot":
        try:
            inst.add_to_watchlist(stock="NVDA")
            return True
        except Exception:
            pass
    if class_name == "TravelAPI":
        # mutate a public attribute directly as a fallback (we just need the state to diverge)
        if hasattr(inst, "credit_card_list"):
            inst.credit_card_list = dict(getattr(inst, "credit_card_list", {}))
            inst.credit_card_list["__smoke__"] = {"balance": 0}
            return True
    if class_name == "VehicleControlAPI":
        try:
            inst.activateParkingBrake(mode="engage")
            return True
        except Exception:
            pass
    # Generic fallback: flip the first public boolean / numeric attribute
    for k, v in list(vars(inst).items()):
        if k.startswith("_"):
            continue
        if isinstance(v, bool):
            setattr(inst, k, not v)
            return True
        if isinstance(v, (int, float)):
            setattr(inst, k, v + 1)
            return True
    return False


def probe_deepcopy() -> bool:
    """For each of 8 multi-turn classes: instantiate, deepcopy, mutate original,
    confirm clone state did NOT change. This is the central assumption of our IWM design."""
    print("\n=== Probe 2: deepcopy isolation on each multi-turn sim class ===")
    all_ok = True
    for cls_name in MULTI_TURN_CLASSES:
        try:
            inst = _instantiate(cls_name)
            clone = copy.deepcopy(inst)

            pre_state_clone = _public_state(clone)

            mutated = _mutate_in_place(inst, cls_name)
            if not mutated:
                # Can't even mutate the original → can't meaningfully test isolation
                # but at least confirm deepcopy itself didn't throw
                print(f"  {cls_name}: deepcopy OK, no obvious mutation available — partial pass")
                continue

            post_state_orig = _public_state(inst)
            post_state_clone = _public_state(clone)

            same_orig_vs_clone = post_state_orig == post_state_clone
            clone_unchanged = pre_state_clone == post_state_clone

            if clone_unchanged and not same_orig_vs_clone:
                print(f"  {cls_name}: ✓ original mutated, clone unchanged")
            elif clone_unchanged and same_orig_vs_clone:
                # mutation didn't actually change state (e.g. setter blocked)
                print(f"  {cls_name}: ⚠  clone unchanged but original also unchanged — "
                      "mutation didn't take effect; deepcopy not fully verified")
            else:
                print(f"  {cls_name}: ✗ clone state CHANGED after original mutation — "
                      "deepcopy did NOT isolate")
                all_ok = False
        except Exception as e:  # noqa: BLE001
            print(f"  {cls_name}: ✗ FAILED  ({type(e).__name__}: {e})")
            traceback.print_exc()
            all_ok = False

    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("BFCL v4 Phase 0 env smoke")
    print(f"  data dir: {DATA_DIR}")
    print(f"  multi-turn classes under test: {MULTI_TURN_CLASSES}")
    gt_ok = probe_gt_end_to_end()
    dc_ok = probe_deepcopy()
    overall = gt_ok and dc_ok
    print(f"\n{'='*60}\nOverall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

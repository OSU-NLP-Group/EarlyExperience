"""
AppWorld Expert Trajectory Replay Module
=========================================

Provides efficient expert trajectory replay for AppWorld, mirroring the
WebShop replay module pattern. Core capabilities:
  1. Ground truth loading: reads compiled_solution_code + specs from disk (no server needed)
  2. Solution code parsing: splits monolithic function into per-step executable blocks via ast
  3. Step-by-step replay: generator yields StepInfo before each expert action
  4. No Ray dependency: single-process, suitable for offline analysis / data generation

Usage
-----
>>> replay = AppWorldExpertReplay(dataset_name="train", port=8000)
>>> for task_id, code_blocks, metadata in replay.matched_tasks:
...     for step in replay.replay_trajectory(task_id, code_blocks):
...         # step.observation_text  — formatted prompt
...         # step.gold_action       — expert code block
...         your_model_predict(step.observation_text)
"""

import os
import sys
import json
import ast
import time
import uuid
import textwrap
from typing import List, Dict, Any, Tuple, Optional, Generator
from dataclasses import dataclass, field
from tqdm import tqdm

# ---------------------------------------------------------------------------
# AppWorld SDK imports
# ---------------------------------------------------------------------------
from appworld import AppWorld, load_task_ids
from appworld.common.path_store import path_store

# ---------------------------------------------------------------------------
# Import prompts via importlib to avoid pulling in ray/omegaconf/torch deps
# ---------------------------------------------------------------------------
import importlib.util as _ilu


def _import_from_file(module_name, file_path):
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prompt_mod = _import_from_file(
    "appworld_prompts",
    os.path.join(os.path.dirname(__file__), "prompts", "appworld.py"),
)
APPWORLD_TEMPLATE_NO_HIS = _prompt_mod.APPWORLD_TEMPLATE_NO_HIS
APPWORLD_TEMPLATE = _prompt_mod.APPWORLD_TEMPLATE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    """Per replay step snapshot exposed to the caller."""
    task_id: str
    step_idx: int
    total_steps: int
    task_description: str
    supervisor: dict          # {first_name, last_name, email, phone_number}
    observation_text: str     # formatted prompt (with template + history)
    raw_observation: str      # raw env output for the current state
    gold_action: str          # expert code block for this step
    all_gold_actions: List[str]
    is_first_step: bool
    is_last_step: bool
    info: dict


# ---------------------------------------------------------------------------
# Solution code parser
# ---------------------------------------------------------------------------

def parse_solution_code(compiled_solution_code: str) -> List[str]:
    """
    Parse compiled_solution_code into individual executable code blocks.

    The input is a function definition like:
        def solution(apis, requester):
            stmt1
            stmt2
            ...

    Returns a list of code strings, each executable via env.execute().
    Uses Python's ast module to identify top-level statements in the
    function body. Falls back to line-based splitting if ast fails.
    """
    try:
        return _parse_with_ast(compiled_solution_code)
    except Exception:
        return _parse_with_lines(compiled_solution_code)


def _parse_with_ast(source: str) -> List[str]:
    """AST-based parsing: each top-level statement in the function body = one block."""
    tree = ast.parse(source)

    # Find the function definition
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solution":
            func_def = node
            break
    if func_def is None:
        raise ValueError("No 'solution' function found in compiled_solution_code")

    if not func_def.body:
        return []

    # Get all source lines
    lines = source.splitlines()

    # The function body starts after the def line
    # Each statement has lineno (1-based) and end_lineno (1-based)
    body_first_line = func_def.body[0].lineno  # 1-based

    # Determine body indentation from the first statement
    first_body_line = lines[body_first_line - 1]
    indent = len(first_body_line) - len(first_body_line.lstrip())

    blocks = []
    for stmt in func_def.body:
        start = stmt.lineno - 1   # 0-based inclusive
        end = stmt.end_lineno      # 1-based inclusive -> 0-based exclusive

        # Extract source lines for this statement and dedent
        stmt_lines = lines[start:end]
        # Remove the function-body indentation
        dedented = []
        for line in stmt_lines:
            if line.strip() == "":
                dedented.append("")
            elif len(line) >= indent:
                dedented.append(line[indent:])
            else:
                dedented.append(line.lstrip())
        block = "\n".join(dedented).strip()
        if block:
            blocks.append(block)

    return blocks


def _parse_with_lines(source: str) -> List[str]:
    """Fallback: line-based parsing — split on the function body, group by indentation."""
    lines = source.splitlines()

    # Find the 'def solution(...):'  line
    body_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def solution(") and stripped.endswith(":"):
            body_start = i + 1
            break
    if body_start is None:
        # No function wrapper — treat each non-empty line as a block
        return [line.strip() for line in lines if line.strip()]

    # Determine indentation
    body_lines = lines[body_start:]
    if not body_lines:
        return []
    first_line = body_lines[0]
    indent = len(first_line) - len(first_line.lstrip())

    # Group lines into blocks: a new block starts at each line with exactly
    # the base indentation (continuation / nested lines have more indent)
    blocks = []
    current_block = []
    for line in body_lines:
        raw = line
        if raw.strip() == "":
            if current_block:
                current_block.append("")
            continue

        line_indent = len(raw) - len(raw.lstrip())
        if line_indent == indent and current_block:
            # New top-level statement — flush previous block
            block_text = "\n".join(current_block).strip()
            if block_text:
                blocks.append(block_text)
            current_block = []

        # Dedent
        if len(raw) >= indent:
            current_block.append(raw[indent:])
        else:
            current_block.append(raw.lstrip())

    # Flush last block
    if current_block:
        block_text = "\n".join(current_block).strip()
        if block_text:
            blocks.append(block_text)

    return blocks


# ---------------------------------------------------------------------------
# SingleWorkerEnv — lightweight AppWorld wrapper (no Ray)
# ---------------------------------------------------------------------------

class SingleWorkerEnv:
    """
    Wraps a single AppWorld instance. Mimics the interface of AppWorldWorker
    but runs in-process without Ray.
    """

    def __init__(self, port: int, experiment_name: str = "replay",
                 max_interactions: int = 50):
        self.port = port
        self.url = f"http://0.0.0.0:{port}"
        self.experiment_name = experiment_name
        self.max_interactions = max_interactions
        self.env: Optional[AppWorld] = None
        self.current_step = 0

    def reset(self, task_id: str) -> Tuple[str, dict]:
        """Close any existing env, create new AppWorld for task_id.
        Returns (instruction_text, info_dict)."""
        if self.env is not None:
            self.env.close()
            time.sleep(2)

        self.current_step = 0
        self.env = AppWorld(
            task_id=task_id,
            experiment_name=self.experiment_name,
            remote_environment_url=self.url,
        )
        obs = self.env.task.instruction
        info = {
            "task_id": task_id,
            "supervisor": dict(self.env.task.supervisor),
        }
        return obs, info

    def step(self, code: str) -> Tuple[str, float, bool, dict]:
        """Execute code via env.execute(), return (obs, reward, done, info)."""
        if self.env is None:
            raise RuntimeError("Environment not reset before step.")

        self.current_step += 1
        obs = self.env.execute(code)

        done = (self.env.task_completed()
                or self.current_step >= self.max_interactions)

        if done:
            is_success = self.env.evaluate().success
            reward = 10.0 if is_success else 0.0
            info = {"won": is_success, "step_count": self.current_step}
        else:
            reward = 0.0
            info = {"won": False, "step_count": self.current_step}

        return obs, reward, done, info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


# ---------------------------------------------------------------------------
# ReplayEnvManager — prompt formatting (matches AppWorldEnvironmentManager)
# ---------------------------------------------------------------------------

class ReplayEnvManager:
    """
    Lightweight version of AppWorldEnvironmentManager (no config/memory/torch deps).
    Provides reset / step / build_text_obs with identical prompt formatting.
    """

    def __init__(self, env: SingleWorkerEnv, history_length: int = 2):
        self.env = env
        self.history_length = history_length
        self.buffer: List[dict] = []        # [{action, text_obs}, ...]
        self.task: Optional[str] = None     # instruction text
        self.supervisor: Optional[dict] = None

    def reset(self, task_id: str) -> Tuple[str, str, dict]:
        """Reset env. Returns (formatted_obs, raw_obs, info)."""
        raw_obs, info = self.env.reset(task_id)
        self.task = raw_obs     # instruction text IS the raw obs at step 0
        self.supervisor = info["supervisor"]
        self.buffer = []
        text_obs = self._build_text_obs(raw_obs, init=True)
        return text_obs, raw_obs, info

    def step(self, code: str) -> Tuple[str, str, float, bool, dict]:
        """Execute code. Returns (formatted_obs, raw_obs, reward, done, info)."""
        raw_obs, reward, done, info = self.env.step(code)
        # Store AFTER executing — matches env_manager.py line 540:
        # self.memory.store({'text_obs': text_obs, 'action': actions})
        self.buffer.append({"action": code, "text_obs": raw_obs})
        text_obs = self._build_text_obs(raw_obs)
        return text_obs, raw_obs, reward, done, info

    def _build_text_obs(self, current_obs: str, init: bool = False) -> str:
        """Format observation — replicates env_manager.py:556-600 exactly."""
        if init:
            return APPWORLD_TEMPLATE_NO_HIS.format(
                supervisor_first_name=self.supervisor["first_name"],
                supervisor_last_name=self.supervisor["last_name"],
                supervisor_email=self.supervisor["email"],
                supervisor_phone_number=self.supervisor["phone_number"],
                task_description=self.task,
            )

        recent = self.buffer[-self.history_length:]
        valid_len = len(recent)
        start_idx = len(self.buffer) - valid_len
        action_history = ""
        for j, rec in enumerate(recent):
            step_number = start_idx + j + 1
            action = rec["action"]
            env_obs = rec["text_obs"]
            action_history += (
                f"\nCode {step_number}: \n{action}\n\n"
                f"Result {step_number}: \n{env_obs}\n"
            )

        if len(action_history) > 10000:
            action_history = "... " + action_history[-10000:]

        return APPWORLD_TEMPLATE.format(
            supervisor_first_name=self.supervisor["first_name"],
            supervisor_last_name=self.supervisor["last_name"],
            supervisor_email=self.supervisor["email"],
            supervisor_phone_number=self.supervisor["phone_number"],
            task_description=self.task,
            step_count=len(self.buffer),
            history_length=valid_len,
            action_history=action_history.strip(),
            current_step=len(self.buffer) + 1,
            current_observation=current_obs,
        )


# ---------------------------------------------------------------------------
# AppWorldExpertReplay — core orchestrator
# ---------------------------------------------------------------------------

class AppWorldExpertReplay:
    """
    AppWorld expert trajectory replay manager.

    Parameters
    ----------
    dataset_name : str
        AppWorld dataset split ("train", "test_normal", etc.)
    port : int
        HTTP port for the AppWorld remote environment server.
    history_length : int
        Number of recent steps to include in the prompt history.
    max_interactions : int
        Maximum steps per task (passed to SingleWorkerEnv).
    cache_dir : str | None
        Directory to cache parsed ground truth. None = no caching.
    """

    def __init__(
        self,
        dataset_name: str = "train",
        port: int = 8000,
        history_length: int = 2,
        max_interactions: int = 50,
        cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.port = port
        self.history_length = history_length
        self.max_interactions = max_interactions
        self.cache_dir = cache_dir

        # Load task IDs
        self.task_ids = load_task_ids(dataset_name)
        print(f"Loaded {len(self.task_ids)} task IDs from '{dataset_name}'")

        # Build ground truth index: task_id -> {code_blocks, metadata, ...}
        self._task_data: Dict[str, dict] = {}
        self._build_index()

    @property
    def matched_tasks(self) -> List[Tuple[str, List[str], dict]]:
        """Returns list of (task_id, code_blocks, metadata) for all valid tasks."""
        return [
            (tid, d["code_blocks"], d["metadata"])
            for tid, d in self._task_data.items()
        ]

    @property
    def num_matched(self) -> int:
        return len(self._task_data)

    # ---- core: step-by-step replay (generator) ----

    def replay_trajectory(
        self,
        task_id: str,
        code_blocks: List[str],
    ) -> Generator[StepInfo, None, None]:
        """
        Step-by-step replay of an expert trajectory.

        Yields StepInfo BEFORE executing each gold action. The caller can
        use this to record the observation, run a model, or insert probes.
        After yield, the gold action is executed internally to advance the env.

        Parameters
        ----------
        task_id : str
            AppWorld task ID.
        code_blocks : list[str]
            Parsed expert code blocks (from parse_solution_code).

        Yields
        ------
        StepInfo
        """
        env = self._make_env()
        try:
            text_obs, raw_obs, info = env.reset(task_id)
            supervisor = info["supervisor"]
            total = len(code_blocks)

            for step_idx, gold_code in enumerate(code_blocks):
                yield StepInfo(
                    task_id=task_id,
                    step_idx=step_idx,
                    total_steps=total,
                    task_description=env.task,
                    supervisor=supervisor,
                    observation_text=text_obs,
                    raw_observation=raw_obs,
                    gold_action=gold_code,
                    all_gold_actions=code_blocks,
                    is_first_step=(step_idx == 0),
                    is_last_step=(step_idx == total - 1),
                    info=info,
                )
                try:
                    text_obs, raw_obs, reward, done, info = env.step(gold_code)
                except Exception as e:
                    import traceback
                    tqdm.write(
                        f"WARNING: task={task_id} step={step_idx} "
                        f"gold_action={gold_code[:80]!r} failed: {e}"
                    )
                    traceback.print_exc()
                    return
                if done:
                    break
        finally:
            env.env.close()

    def replay_all(
        self,
        task_ids: Optional[List[str]] = None,
        step_callback=None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch replay of all matched tasks (or a specified subset).

        Parameters
        ----------
        task_ids : list[str] | None
            Tasks to replay. None = all matched tasks.
        step_callback : callable(StepInfo) -> Any | None
            Called at each step. Return value is recorded.
        show_progress : bool
            Whether to show tqdm progress bar.

        Returns
        -------
        list[dict] — per-task replay records.
        """
        pairs = self.matched_tasks
        if task_ids is not None:
            tid_set = set(task_ids)
            pairs = [(t, b, m) for t, b, m in pairs if t in tid_set]

        results = []
        it = tqdm(pairs, desc="Replaying", ncols=100) if show_progress else pairs

        for tid, blocks, meta in it:
            traj_record = {
                "task_id": tid,
                "total_steps": len(blocks),
                "metadata": meta,
                "steps": [],
            }
            for step in self.replay_trajectory(tid, blocks):
                traj_record["task_description"] = step.task_description
                cb_result = step_callback(step) if step_callback else None
                traj_record["steps"].append({
                    "step_idx": step.step_idx,
                    "gold_action": step.gold_action,
                    "observation_text": step.observation_text,
                    "callback_result": cb_result,
                })
            results.append(traj_record)

        return results

    # ---- internal methods ----

    def _make_env(self) -> ReplayEnvManager:
        """Create a new single-process env instance with unique experiment name."""
        name = f"replay_{uuid.uuid4().hex[:8]}"
        worker = SingleWorkerEnv(
            port=self.port,
            experiment_name=name,
            max_interactions=self.max_interactions,
        )
        return ReplayEnvManager(worker, history_length=self.history_length)

    def _build_index(self):
        """Load ground truth + task specs from disk for all task IDs.
        No AppWorld server needed — reads directly from data files."""

        # Try loading from cache
        if self.cache_dir:
            cache_file = os.path.join(
                self.cache_dir, f"gt_cache_{self.dataset_name}.json"
            )
            if os.path.exists(cache_file):
                print(f"Loading cached ground truth from {cache_file}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    self._task_data = json.load(f)
                print(f"Loaded {len(self._task_data)} tasks from cache")
                return

        # Load from disk
        from appworld.ground_truth import GroundTruth

        tasks_dir = os.path.join(path_store.data, "tasks")
        skipped = 0

        for task_id in tqdm(self.task_ids, desc="Loading ground truth"):
            try:
                # Load ground truth (no server needed)
                gt = GroundTruth.load(task_id, mode="full")

                # Load task specs (instruction + supervisor) from specs.json
                specs_path = os.path.join(tasks_dir, task_id, "specs.json")
                with open(specs_path, "r", encoding="utf-8") as f:
                    specs = json.load(f)

                instruction = specs["instruction"]
                supervisor_dict = specs.get("supervisor", specs.get("main_user", {}))

                # Parse solution code into executable steps
                if not gt.compiled_solution_code:
                    tqdm.write(f"WARNING: {task_id} has no compiled_solution_code, skipping")
                    skipped += 1
                    continue

                code_blocks = parse_solution_code(gt.compiled_solution_code)
                if not code_blocks:
                    tqdm.write(f"WARNING: {task_id} parsed to 0 code blocks, skipping")
                    skipped += 1
                    continue

                self._task_data[task_id] = {
                    "code_blocks": code_blocks,
                    "instruction": instruction,
                    "supervisor": supervisor_dict,
                    "metadata": {
                        "difficulty": gt.metadata.get("difficulty") if gt.metadata else None,
                        "num_apis": gt.metadata.get("num_apis") if gt.metadata else None,
                        "num_api_calls": gt.metadata.get("num_api_calls") if gt.metadata else None,
                        "required_apps": gt.required_apps,
                        "answer": gt.answer if not callable(gt.answer) else str(gt.answer),
                    },
                    "compiled_solution_code": gt.compiled_solution_code,
                }
            except Exception as e:
                tqdm.write(f"WARNING: failed to load GT for {task_id}: {e}")
                skipped += 1
                continue

        print(
            f"Index complete: {len(self._task_data)} tasks loaded, "
            f"{skipped} skipped"
        )

        # Save cache
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(
                self.cache_dir, f"gt_cache_{self.dataset_name}.json"
            )
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self._task_data, f, ensure_ascii=False)
            print(f"Cached ground truth to {cache_file}")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_replay(
    dataset_name: str = "train",
    port: int = 8000,
    history_length: int = 2,
    max_interactions: int = 50,
    cache_dir: Optional[str] = None,
) -> AppWorldExpertReplay:
    """Quick-create an AppWorldExpertReplay instance."""
    return AppWorldExpertReplay(
        dataset_name=dataset_name,
        port=port,
        history_length=history_length,
        max_interactions=max_interactions,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# CLI: self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AppWorld Expert Replay self-test")
    parser.add_argument("--dataset", default="train", help="Dataset split")
    parser.add_argument("--port", type=int, default=8000, help="AppWorld server port")
    parser.add_argument("--max-tasks", type=int, default=3, help="Max tasks to replay")
    parser.add_argument("--cache-dir", default=None, help="GT cache directory")
    args = parser.parse_args()

    replay = make_replay(
        dataset_name=args.dataset,
        port=args.port,
        cache_dir=args.cache_dir,
    )

    print(f"\n{'='*60}")
    print(f"Total {replay.num_matched} tasks have parsed expert trajectories")
    print(f"{'='*60}\n")

    for task_id, blocks, meta in replay.matched_tasks[: args.max_tasks]:
        print(f"Task {task_id}  ({len(blocks)} steps, difficulty={meta.get('difficulty')})")
        for step in replay.replay_trajectory(task_id, blocks):
            print(
                f"  Step {step.step_idx}/{step.total_steps}: "
                f"gold_action={step.gold_action[:80]!r}"
            )
        print()

"""Client-side admissible-action enumeration for TextCraft.

Unlike ScienceWorld, TextCraft does NOT need a server-side
`/admissible_actions` endpoint. The full information needed to determine
admissibility is already exposed via the env's reset observation
(`commands_list`) + the agent's tracked `inventory`:

  - `inventory` — always 1 action (any state).
  - `get N <item>` — admissible iff `item` is a base item (NOT craftable in
    the env's global crafting tree) AND a valid item id. The env accepts any
    positive integer N; we enumerate N's that appear as recipe input
    quantities in `commands_list`, plus N=1 as a safe default.
  - `craft <output> using <ingredients>` — admissible iff (a) the recipe
    string appears verbatim in `commands_list` (strict per the agent's
    prompt: "use ONLY these crafting commands provided"), and (b) all
    ingredient quantities are satisfied by the current `inventory`.

The "global" craftability check uses the env's actual `CraftingTree`
(loaded from `agentenv_textcraft/recipes/*.json`), which is the same object
the env's `step()` consults. This guarantees the predicate matches the
env's own admissibility judgment.

Usage:
    from admissible import AdmissibleEnumerator

    enum = AdmissibleEnumerator()  # loads tree once
    adm  = enum.enumerate(commands_list=[...], inventory={"oak log": 4})
    # adm is a sorted list of action strings, e.g.:
    #   ["inventory", "get 1 oak log", "get 4 oak log", "craft 3 oak wood using 4 oak log", ...]

Module-level smoke at the bottom: probes admissible at s_0 of a few
data_idx values and prints the size + a sample.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple


# Recipe-string parsing — mirrors env's TextCraftEnv action regexes and
# the canonical AgentGym recipe format ("craft N OUT using N1 IN1, N2 IN2").
_RECIPE_RE = re.compile(r"^craft\s+(\d+)\s+(.+?)\s+using\s+(.+)$")
_INGREDIENT_RE = re.compile(r"^(\d+)\s+(.+)$")


def _parse_recipe(recipe_str: str):
    """Parse 'craft N OUT using N1 IN1, N2 IN2, ...' into
    (out_count, out_item, [(N, IN), ...]). Returns None if format mismatches."""
    m = _RECIPE_RE.match(recipe_str.strip())
    if not m:
        return None
    out_count = int(m.group(1))
    out_item = m.group(2).strip()
    ingredients = []
    for part in m.group(3).split(","):
        mm = _INGREDIENT_RE.match(part.strip())
        if not mm:
            return None  # malformed → reject the whole recipe
        ingredients.append((int(mm.group(1)), mm.group(2).strip()))
    return (out_count, out_item, ingredients)


def _item_str_to_id(item_display: str) -> str:
    """Mirror env's TextCraftEnv.item_str_to_obj id form:
    'oak log' -> 'minecraft:oak_log'."""
    return "minecraft:" + item_display.replace(" ", "_")


class AdmissibleEnumerator:
    """Stateless enumerator (the only mutable state is the loaded tree)."""

    def __init__(self, minecraft_dir: str = None):
        """Load the env's CraftingTree exactly once.

        minecraft_dir: path to the dir that contains `recipes/`. Defaults to
        the agentenv-textcraft package install location.

        Note: importing `agentenv_textcraft` triggers a top-level
        `TextCraft_Wrapper()` in its `__init__.py`, which loads recipes from
        a cwd-relative path `agentenv_textcraft/recipes/`. We chdir to the
        package's parent directory for the duration of the import, then
        restore. Subsequent imports use Python's module cache and are not
        affected.
        """
        import importlib.util
        import os

        spec = importlib.util.find_spec("agentenv_textcraft")
        if spec is None or spec.origin is None:
            raise ImportError("agentenv_textcraft not installed (pip install -e ...)")
        pkg_dir = os.path.dirname(spec.origin)
        if minecraft_dir is None:
            minecraft_dir = pkg_dir

        old_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(pkg_dir))  # parent of the package
            from agentenv_textcraft.crafting_tree import CraftingTree

            self.tree = CraftingTree(minecraft_dir=minecraft_dir)
        finally:
            os.chdir(old_cwd)

    def _is_base_item(self, item_display: str) -> bool:
        """True iff env's `get` would accept this item.

        Env logic (from TextCraftEnv.step):
            if self.crafting_tree.is_craftable(item.name):  raise "Could not find"
            if self.crafting_tree.is_tag(item.item_id):     raise "Could not find"
            if not self.crafting_tree.is_valid_item(item.item_id): raise "Could not find"
        """
        item_id = _item_str_to_id(item_display)
        if self.tree.is_craftable(item_id):
            return False
        if self.tree.is_tag(item_id):
            return False
        if not self.tree.is_valid_item(item_id):
            return False
        return True

    def enumerate(
        self,
        commands_list: List[str],
        inventory: Dict[str, int],
    ) -> List[str]:
        """Return the sorted list of admissible action strings at this state."""
        parsed = [_parse_recipe(r) for r in commands_list]
        parsed = [p for p in parsed if p is not None]

        # ---- get actions ----
        # Collect (base_item_display_name, quantity) from recipe inputs.
        # For each distinct base item, always include N=1 as a baseline plus
        # every distinct N that appears in some recipe's ingredient list.
        per_item_Ns: Dict[str, Set[int]] = {}
        for _out_count, _out_item, ingredients in parsed:
            for n, item_display in ingredients:
                if self._is_base_item(item_display):
                    per_item_Ns.setdefault(item_display, set()).add(n)
        for item in per_item_Ns:
            per_item_Ns[item].add(1)  # always include the baseline

        get_actions: List[str] = []
        for item in sorted(per_item_Ns):
            for n in sorted(per_item_Ns[item]):
                get_actions.append(f"get {n} {item}")

        # ---- craft actions ----
        # Strict: only recipes from commands_list, and only those satisfied.
        craft_actions: List[str] = []
        for out_count, out_item, ingredients in parsed:
            satisfied = all(
                inventory.get(ing_item, 0) >= ing_n
                for ing_n, ing_item in ingredients
            )
            if not satisfied:
                continue
            ing_str = ", ".join(f"{n} {item}" for n, item in ingredients)
            craft_actions.append(f"craft {out_count} {out_item} using {ing_str}")

        return ["inventory"] + get_actions + craft_actions


# -----------------------------------------------------------------------------
# Module smoke: run a few data_idx values, print admissible-set sizes.
# -----------------------------------------------------------------------------
def _smoke():
    """Probe admissible at s_0 of a handful of data_idx values.
    Requires the textcraft server NOT to be running (we use the local CraftingTree
    directly to derive base-item-ness, but call requests to env for commands_list).
    """
    import requests

    enum = AdmissibleEnumerator()
    base = "http://127.0.0.1:36011"
    try:
        env_id = requests.post(f"{base}/create", json={}, timeout=5).json()["id"]
    except Exception:
        print("server unreachable; printing tree stats only")
        print(f"  tree.itemid_recipes (craftable items): {len(enum.tree.itemid_recipes)}")
        print(f"  tree.tag_set: {len(enum.tree.tag_set)}")
        print(f"  tree.itemid_set: {len(enum.tree.itemid_set)}")
        return

    for data_idx in [0, 100, 260, 373]:
        obs = requests.post(
            f"{base}/reset", json={"id": env_id, "data_idx": data_idx}, timeout=5
        ).json()["observation"]
        commands = [l for l in obs.split("\n") if l.startswith("craft ")]
        goal = obs.split("Goal: craft ")[-1].rstrip(".")
        adm = enum.enumerate(commands, inventory={})
        print(
            f"data_idx={data_idx:3d} goal=craft {goal!r}  "
            f"recipes={len(commands):2d} admissible@s0={len(adm):2d}"
        )
        for a in adm[:6]:
            print(f"   {a}")
        if len(adm) > 6:
            print(f"   ... (+ {len(adm)-6} more)")


if __name__ == "__main__":
    _smoke()

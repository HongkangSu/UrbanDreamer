"""HDV learning-state serialization helpers for SUMO evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_optional_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def _load_hdv_learning_state(state_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if state_path is None:
        return None
    if not state_path.exists():
        print(f"  [HDV state] input not found, skip restore: {state_path}")
        return None
    try:
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("top-level JSON is not an object")
        agents = state.get("agents", {})
        if not isinstance(agents, dict):
            raise ValueError("`agents` is not an object")
        print(f"  [HDV state] loaded {len(agents)} agent entries <- {state_path}")
        return state
    except Exception as exc:
        print(f"  [HDV state] WARNING: failed to load {state_path}: {exc}")
        return None


def _numeric_to_list(value: Any) -> Optional[list]:
    try:
        arr = np.asarray(value, dtype=np.float64)
        return arr.tolist()
    except Exception:
        return None


def _snapshot_hdv_learning_state(env) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for human in getattr(env, "human_agents", []):
        model = getattr(human, "model", None)
        if model is None:
            continue
        aid = str(getattr(human, "id", ""))
        if not aid:
            continue
        entry: Dict[str, Any] = {"model_class": model.__class__.__name__}
        if hasattr(model, "cost"):
            cost_list = _numeric_to_list(getattr(model, "cost"))
            if cost_list is not None:
                entry["cost"] = cost_list
        if hasattr(model, "memory"):
            mem_list = _numeric_to_list(getattr(model, "memory"))
            if mem_list is not None:
                entry["memory"] = mem_list
        if hasattr(model, "beta"):
            try:
                entry["beta"] = float(getattr(model, "beta"))
            except Exception:
                pass
        out[aid] = entry
    return out


def _merge_hdv_learning_state(
    base_state: Optional[Dict[str, Any]],
    update_agents: Dict[str, Dict[str, Any]],
    route_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged_agents: Dict[str, Dict[str, Any]] = {}
    if isinstance(base_state, dict):
        base_agents = base_state.get("agents", {})
        if isinstance(base_agents, dict):
            merged_agents.update(base_agents)
    merged_agents.update(update_agents)
    merged: Dict[str, Any] = {"version": 1, "agents": merged_agents}
    if isinstance(base_state, dict):
        if "route_policy" in base_state and isinstance(base_state["route_policy"], list):
            merged["route_policy"] = list(base_state["route_policy"])
    if route_state and isinstance(route_state.get("route_policy"), list):
        merged["route_policy"] = list(route_state["route_policy"])
    return merged


def _restore_hdv_learning_state(env, state: Optional[Dict[str, Any]], *, label: str) -> None:
    if not state:
        return
    agents_state = state.get("agents", {})
    if not isinstance(agents_state, dict):
        return

    humans = list(getattr(env, "human_agents", []))
    restored = 0
    missing = 0
    mismatched = 0
    no_model = 0

    for human in humans:
        aid = str(getattr(human, "id", ""))
        entry = agents_state.get(aid)
        if entry is None:
            missing += 1
            continue
        model = getattr(human, "model", None)
        if model is None:
            no_model += 1
            continue

        applied = False

        if "cost" in entry and hasattr(model, "cost"):
            try:
                cur = np.asarray(getattr(model, "cost"))
                new = np.asarray(entry["cost"], dtype=cur.dtype)
                if cur.shape == new.shape:
                    setattr(model, "cost", new.copy())
                    applied = True
                else:
                    mismatched += 1
            except Exception:
                mismatched += 1

        if "memory" in entry and hasattr(model, "memory"):
            try:
                cur_mem = np.asarray(getattr(model, "memory"))
                new_mem = np.asarray(entry["memory"], dtype=cur_mem.dtype)
                if cur_mem.shape == new_mem.shape:
                    setattr(model, "memory", new_mem.copy())
                    applied = True
                else:
                    mismatched += 1
            except Exception:
                mismatched += 1

        if "beta" in entry and hasattr(model, "beta"):
            try:
                setattr(model, "beta", float(entry["beta"]))
                applied = True
            except Exception:
                pass

        if applied:
            restored += 1

    print(
        f"  [HDV state/{label}] restored={restored}/{len(humans)} "
        f"missing={missing} mismatched={mismatched} no_model={no_model}",
        flush=True,
    )


def _save_hdv_learning_state(state_path: Optional[Path], state: Optional[Dict[str, Any]]) -> None:
    if state_path is None or not state:
        return
    agents = state.get("agents", {})
    if not isinstance(agents, dict):
        return
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f)
        print(f"  [HDV state] saved {len(agents)} agent entries -> {state_path}")
    except Exception as exc:
        print(f"  [HDV state] WARNING: failed to save {state_path}: {exc}")

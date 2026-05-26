"""
Shared helpers for RouteRL black-box HDV behavior.

This module centralizes the same per-human logic used by:
1) data collection (`RouteRLConnector`)
2) SUMO eval in MARL training (`SUMOEvalHarness`)

The RouteRL human model is treated as a black box:
  - action: `human.act(None)`
  - learning: `human.model.learn(None, action, reward)`
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


def _candidate_agent_keys(agent_id: Any) -> list[str]:
    """Return robust string keys for an agent id."""
    out: list[str] = []
    sid = str(agent_id)
    out.append(sid)
    try:
        out.append(str(int(float(sid))))
    except Exception:
        pass
    # Keep order but remove duplicates.
    return list(dict.fromkeys(out))


def build_human_agent_lookup(human_agents: Iterable[Any]) -> Dict[str, Any]:
    """
    Build `id(str) -> human agent object` lookup with tolerant id aliases.
    """
    lookup: Dict[str, Any] = {}
    for human in human_agents:
        hid = getattr(human, "id", None)
        if hid is None:
            continue
        for key in _candidate_agent_keys(hid):
            lookup[key] = human
    return lookup


def get_human_agent(human_lookup: Mapping[str, Any], agent_id: Any) -> Optional[Any]:
    """Resolve a human agent object from lookup with tolerant id casting."""
    for key in _candidate_agent_keys(agent_id):
        human = human_lookup.get(key)
        if human is not None:
            return human
    return None


def _rng_randint(rng: Any, upper_exclusive: int) -> int:
    """Sample integer in [0, upper_exclusive) from Random/Generator-like object."""
    if upper_exclusive <= 1:
        return 0
    if hasattr(rng, "randrange"):
        return int(rng.randrange(upper_exclusive))
    if hasattr(rng, "integers"):
        return int(rng.integers(0, upper_exclusive))
    if hasattr(rng, "randint"):
        try:
            return int(rng.randint(0, upper_exclusive - 1))
        except TypeError:
            return int(rng.randint(upper_exclusive))
    return int(random.randint(0, upper_exclusive - 1))


def choose_routerl_hdv_action(
    human_lookup: Mapping[str, Any],
    agent_id: Any,
    n_routes: int,
    rng: Any,
    episode_actions: Optional[MutableMapping[str, int]] = None,
) -> int:
    """
    Choose one RouteRL black-box HDV route index for an agent.

    If `episode_actions` is provided, stores the sampled action by string id.
    """
    n_routes = int(max(1, n_routes))
    route_idx = _rng_randint(rng, n_routes)

    human = get_human_agent(human_lookup, agent_id)
    if human is not None:
        try:
            proposed = int(human.act(None))
            route_idx = min(max(proposed, 0), n_routes - 1)
        except Exception:
            pass

    if episode_actions is not None:
        episode_actions[str(agent_id)] = int(route_idx)
    return int(route_idx)


def apply_routerl_hdv_learning(
    human_lookup: Mapping[str, Any],
    episode_actions: Mapping[str, int],
    travel_times: Mapping[str, float],
    *,
    travel_time_unit: str = "seconds",
) -> int:
    """
    Apply RouteRL black-box per-human learning updates.

    Args:
        human_lookup: id -> human object.
        episode_actions: id -> chosen action index in the same episode.
        travel_times: id -> travel time.
        travel_time_unit: "seconds" or "minutes" (case-insensitive).

    Returns:
        Number of human agents successfully updated.
    """
    unit = str(travel_time_unit).lower()
    if unit not in {"seconds", "minutes"}:
        raise ValueError(f"Unknown travel_time_unit='{travel_time_unit}', expected 'seconds' or 'minutes'")

    updated = 0
    for veh_id, action in episode_actions.items():
        tt_val: Optional[float] = None
        for key in _candidate_agent_keys(veh_id):
            if key in travel_times:
                try:
                    tt_val = float(travel_times[key])
                except Exception:
                    tt_val = None
                break
        if tt_val is None:
            continue

        human = get_human_agent(human_lookup, veh_id)
        if human is None or not hasattr(human, "model"):
            continue
        try:
            tt_min = tt_val if unit == "minutes" else (tt_val / 60.0)
            reward = -float(tt_min)
            human.model.learn(None, int(action), reward)
            updated += 1
        except Exception:
            continue
    return updated

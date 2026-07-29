from __future__ import annotations

from typing import Any

from urban_dreamer.traffic.environment import (
    ActionDecision,
    CandidatePathGenerator,
    RouteRLSumoSession,
)


def test_candidate_path_generator_uses_fixed_k4_penalty_dijkstra_and_cache() -> None:
    generator = CandidatePathGenerator(
        adjacency=[[1, 2], [3], [3], []],
        free_flow_costs=[1.0, 10.0, 2.0, 1.0],
    )

    first = generator.generate_paths(0, 3)
    second = generator.generate_paths(0, 3)
    other = generator.generate_paths(1, 3)
    third = generator.generate_paths(0, 3)

    assert [list(path) for path in first] == [
        [0, 2, 3],
        [0, 1, 3],
        [0, 2, 3],
        [0, 2, 3],
    ]
    assert second is first
    assert [list(path) for path in other] == [[1, 3]] * 4
    assert third is first
    assert len(generator.state_dict()["cache"]) == 2

    restored = CandidatePathGenerator(
        adjacency=[[1, 2], [3], [3], []],
        free_flow_costs=[1.0, 10.0, 2.0, 1.0],
    )
    restored.load_state_dict(generator.state_dict())
    assert restored.state_dict() == generator.state_dict()
    assert restored.generate_paths(0, 3) == first


def test_action_decision_builds_exact_one_hot_path_and_planned_flow() -> None:
    decision = ActionDecision.build(
        route_index=0,
        candidate_paths=((0, 2, 3), (0, 1, 3), (0, 2, 3), (0, 2, 3)),
        edge_order=(0, 1, 2, 3),
        edge_travel_times=(1.0, 10.0, 2.0, 1.0),
        decision_horizon_seconds=5.0,
    )

    assert decision.route_one_hot == (1, 0, 0, 0)
    assert decision.selected_path == (0, 2, 3)
    assert decision.planned_edge_flow == (0.0, 0.0, 1.0, 1.0)


def test_route_rl_session_advances_and_closes_fake_backend() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.closed = 0
            self.assigned_routes: dict[str, list[str]] = {}
            self._state = {
                "vehicles": {"veh-1": {"edge": "a"}},
                "edges": {"a": {"vehicle_count": 1.0, "mean_speed": 10.0}},
            }

        def reset(self) -> dict[str, Any]:
            return self._state

        def get_traffic_state(self) -> dict[str, Any]:
            return self._state

        def is_done(self) -> bool:
            return False

        def travel_times_seconds(self) -> dict[str, float]:
            return {"veh-1": 12.0}

        def hdv_route_choices(self) -> dict[str, int]:
            return {}

        def travel_time_route_choices(self) -> dict[str, int]:
            return {"veh-1": 1}

        def hdv_learning_state(self) -> dict[str, Any]:
            return {"version": 1, "route_policy": []}

        def restore_hdv_learning_state(self, state: Any) -> None:
            assert state is None

        def update_hdv_learning(self, travel_times_seconds: Any = None) -> int:
            assert travel_times_seconds is None
            return 0

        def set_vehicle_route(
            self,
            vehicle_id: str,
            route_index: int,
            route_edges: list[str],
        ) -> None:
            assert route_index == 1
            if vehicle_id == "bad-veh":
                raise RuntimeError("unsafe reroute")
            self.assigned_routes[vehicle_id] = list(route_edges)

        def advance(self, steps: int) -> None:
            assert steps == 3

        def get_active_vehicle_exposure(self) -> float:
            return 6.0

        def close(self) -> None:
            self.closed += 1

    fake_backend = FakeBackend()
    session = RouteRLSumoSession(
        edge_ids=("a", "b", "c"),
        backend_factory=lambda: fake_backend,
    )

    decision = ActionDecision.build(
        route_index=1,
        candidate_paths=(
            (0, 1),
            (0, 2),
            (0, 1),
            (0, 2),
        ),
        edge_order=(0, 1, 2),
        edge_travel_times=(1.0, 2.0, 1.0),
        decision_horizon_seconds=4.0,
    )

    assert session.reset() == fake_backend.get_traffic_state()
    assert session.is_done() is False
    assert session.travel_times_seconds() == {"veh-1": 12.0}
    assert session.hdv_route_choices() == {}
    assert session.travel_time_route_choices() == {"veh-1": 1}
    assert session.hdv_learning_state() == {"version": 1, "route_policy": []}
    assert session.update_hdv_learning() == 0
    session.set_routes_safely({"veh-1": decision, "bad-veh": decision})
    session.advance(3)
    assert session.get_traffic_state() == fake_backend.get_traffic_state()
    session.close()

    assert fake_backend.assigned_routes == {"veh-1": ["a", "c"]}
    assert session.last_active_vehicle_exposure == 6.0
    assert fake_backend.closed == 1
    session.close()
    assert fake_backend.closed == 1

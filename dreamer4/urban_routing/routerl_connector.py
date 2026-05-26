# routerl_connector.py
"""
RouteRL Environment Connector for Urban Routing (Phase 1)

Replaces SUMOConnector by wrapping routerl's TrafficEnvironment.
Provides the same interface expected by DataCollector:
  - reset() -> state dict
  - step(actions) -> (state, reward, done, info)
  - get_traffic_state() -> state dict
  - close()

RouteRL handles:
  - SUMO simulator lifecycle (start/step/stop)
  - Path generation via janux (pre-generated, see scripts/generate_routes.py)
  - Agent demand (origin-destination pairs, start times)

Usage:
    connector = RouteRLConnector(
        network_name="ing_small",
        network_folder="/path/to/OpenURB/networks/ing_small",
        origins=["edge1", "edge2"],
        destinations=["edge3", "edge4"],
        number_of_paths=4,
        num_agents=100,
        seed=42,
    )
    state = connector.reset()
    next_state, reward, done, info = connector.step(actions={})
"""

import os
import logging
import random as _random
import shlex
import types
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd

try:
    from routerl import TrafficEnvironment
    ROUTERL_AVAILABLE = True
except ImportError:
    ROUTERL_AVAILABLE = False
    print("Warning: routerl not available. Please install routerl.")

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

try:
    from .routerl_shared import create_routerl_environment
    from .routerl_hdv_blackbox import (
        build_human_agent_lookup,
        choose_routerl_hdv_action,
        apply_routerl_hdv_learning as apply_routerl_hdv_learning_shared,
    )
except ImportError:
    from routerl_shared import create_routerl_environment
    from routerl_hdv_blackbox import (
        build_human_agent_lookup,
        choose_routerl_hdv_action,
        apply_routerl_hdv_learning as apply_routerl_hdv_learning_shared,
    )


class RouteRLConnector:
    """
    Connector for SUMO traffic simulation using the RouteRL framework.

    Wraps routerl's TrafficEnvironment for SUMO lifecycle management
    and adds vehicles from agents.csv with Gawron-style route choices.
    """

    def __init__(
        self,
        network_name: str,
        network_folder: str,
        origins: list,
        destinations: list,
        number_of_paths: int = 4,
        path_gen_beta: float = -5.0,
        num_samples: int = 10,
        num_agents: int = 100,
        simulation_timesteps: int = 1800,
        records_folder: str = "routerl_records",
        seed: int = 42,
        use_gui: bool = False,
        use_routerl_hdv_blackbox: bool = False,
        sumo_depart_lane: str = "",
        sumo_depart_pos: str = "",
        sumo_depart_speed: str = "",
        sumo_time_to_teleport: str = "",
        sumo_eager_insert: bool = False,
        sumo_sloppy_insert: bool = False,
        sumo_emergency_insert: bool = False,
        sumo_extra_args: str = "",
        incident_edge: str = "",
        incident_start: float = 0.0,
        incident_end: float = 0.0,
        incident_speed: float = 0.0,
    ):
        if not ROUTERL_AVAILABLE:
            raise RuntimeError("routerl is not installed. Run: pip install routerl")

        self.network_name = network_name
        self.network_folder = os.path.abspath(network_folder)
        self.seed = seed
        self.use_gui = use_gui
        self.number_of_paths = number_of_paths
        self.use_routerl_hdv_blackbox = bool(use_routerl_hdv_blackbox)
        self._sumo_depart_lane = str(sumo_depart_lane or "").strip()
        self._sumo_depart_pos = str(sumo_depart_pos or "").strip()
        self._sumo_depart_speed = str(sumo_depart_speed or "").strip()
        self._sumo_time_to_teleport = str(sumo_time_to_teleport or "").strip()
        self._sumo_eager_insert = bool(sumo_eager_insert)
        self._sumo_sloppy_insert = bool(sumo_sloppy_insert)
        self._sumo_emergency_insert = bool(sumo_emergency_insert)
        self._sumo_extra_args = shlex.split(str(sumo_extra_args or ""))
        self._incident_edge = str(incident_edge or "").strip()
        self._incident_start = float(incident_start or 0.0)
        self._incident_end = float(incident_end or 0.0)
        self._incident_speed = float(incident_speed or 0.0)
        self._incident_lane_speeds: Dict[str, float] = {}
        self._incident_active = False
        self._skip_edge_subscriptions = str(
            os.environ.get("URB_SKIP_EDGE_SUBSCRIPTIONS", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._episode_count = 0
        self.origins = origins
        self.destinations = destinations

        os.makedirs(records_folder, exist_ok=True)
        self.records_folder = records_folder
        self._records_agents_csv = os.path.join(self.records_folder, "agents.csv")
        self._use_custom_agents = os.path.exists(self._records_agents_csv)

        self.env = create_routerl_environment(
            TrafficEnvironment,
            seed=seed,
            create_agents=not self._use_custom_agents,
            create_paths=False,  # paths pre-generated by scripts/generate_routes.py
            network_name=network_name,
            network_folder=self.network_folder,
            simulation_timesteps=simulation_timesteps,
            records_folder=records_folder,
            plots_folder=os.path.join(records_folder, "plots"),
            origins=origins,
            destinations=destinations,
            number_of_paths=number_of_paths,
            num_agents=num_agents if not self._use_custom_agents else None,
            new_machines_after_mutation=0,
            machine_behavior="selfish",
            machine_observation_type=None,
            save_every=1,
            phases=[0, 1],
            phase_names=["Data collection"],
            smooth_by=5,
            plot_choices="none",
            sumo_type="sumo-gui" if use_gui else "sumo",
            path_beta=path_gen_beta,
            path_num_samples=num_samples,
            visualize_paths=False,
            save_detectors_info=False,
        )
        self._patch_routerl_sumo_commands()

        # Load agent demand from agents.csv (RouteRL's generated agents)
        self._agents_df = self._load_agents()

        # Load available routes per OD pair from routes.csv
        self._od_routes = self._load_od_routes()
        self._od_route_paths = self._load_od_route_paths()

        # Load network geometry (sumolib)
        self.network = self._load_sumolib_network()

        # Populated after reset(): {vehicle_id: [route_edge_list, ...]}
        self.vehicle_routes: Dict[str, list] = {}
        self._episode_route_choices: Dict[str, int] = {}
        self._human_agents_by_id: Dict[str, Any] = {}
        self._human_episode_actions: Dict[str, int] = {}

        self._started = False
        self.current_step = 0

        print(f"RouteRLConnector initialized")
        print(f"  Network: {network_name} @ {self.network_folder}")
        print(f"  Agents: {len(self._agents_df)}, Paths per OD: {number_of_paths}")
        print(f"  Demand source: {'records_folder/agents.csv' if self._use_custom_agents else 'RouteRL-generated agents'}")
        if self.use_routerl_hdv_blackbox:
            print("  HDV routing: RouteRL black-box human model")
        if self._incident_enabled():
            print(
                "  Incident: "
                f"edge={self._incident_edge} "
                f"window=[{self._incident_start:.1f}, {self._incident_end:.1f}) "
                f"speed={self._incident_speed:.2f}m/s"
            )

    def _sumo_command_suffix(self) -> List[str]:
        """Return optional SUMO CLI args appended to RouteRL's base command."""
        suffix: List[str] = []
        if self._sumo_eager_insert:
            suffix.append("--eager-insert")
        if self._sumo_sloppy_insert:
            suffix.append("--sloppy-insert")
        if self._sumo_emergency_insert:
            suffix.append("--emergency-insert")
        suffix.extend(self._sumo_extra_args)
        return suffix

    def _apply_sumo_command_overrides(self, cmd: List[str]) -> List[str]:
        """Patch RouteRL's hard-coded SUMO command without modifying routerl."""
        out = list(cmd)
        if self._sumo_time_to_teleport:
            option = "--time-to-teleport"
            patched: List[str] = []
            idx = 0
            while idx < len(out):
                if out[idx] == option:
                    idx += 2
                    continue
                patched.append(out[idx])
                idx += 1
            out = patched + [option, self._sumo_time_to_teleport]
        out.extend(self._sumo_command_suffix())
        return out

    def _incident_enabled(self) -> bool:
        return (
            bool(self._incident_edge)
            and self._incident_end > self._incident_start
            and self._incident_speed > 0.0
        )

    def _capture_incident_lanes(self) -> None:
        self._incident_lane_speeds = {}
        self._incident_active = False
        if not self._incident_enabled():
            return
        try:
            edge_obj = self.network.getEdge(self._incident_edge)
        except Exception as exc:
            raise ValueError(f"Unknown incident edge: {self._incident_edge}") from exc
        for lane in edge_obj.getLanes():
            lane_id = str(lane.getID())
            try:
                self._incident_lane_speeds[lane_id] = float(traci.lane.getMaxSpeed(lane_id))
            except Exception:
                try:
                    self._incident_lane_speeds[lane_id] = float(lane.getSpeed())
                except Exception:
                    self._incident_lane_speeds[lane_id] = 13.9
        logging.info(
            "Configured SUMO incident edge=%s lanes=%d start=%.1f end=%.1f speed=%.2fm/s",
            self._incident_edge,
            len(self._incident_lane_speeds),
            self._incident_start,
            self._incident_end,
            self._incident_speed,
        )

    def _set_incident_lane_speed(self, speed: float) -> None:
        for lane_id in self._incident_lane_speeds:
            traci.lane.setMaxSpeed(str(lane_id), float(speed))

    def _restore_incident_lanes(self) -> None:
        for lane_id, speed in self._incident_lane_speeds.items():
            try:
                traci.lane.setMaxSpeed(str(lane_id), float(speed))
            except Exception:
                pass

    def _apply_incident_for_current_time(self) -> None:
        if not self._incident_lane_speeds:
            return
        try:
            sim_time = float(traci.simulation.getTime())
        except Exception:
            return
        should_activate = self._incident_start <= sim_time < self._incident_end
        if should_activate and not self._incident_active:
            self._set_incident_lane_speed(self._incident_speed)
            self._incident_active = True
            logging.info("Activated SUMO incident edge=%s at t=%.1f", self._incident_edge, sim_time)
        elif self._incident_active and not should_activate:
            self._restore_incident_lanes()
            self._incident_active = False
            logging.info("Cleared SUMO incident edge=%s at t=%.1f", self._incident_edge, sim_time)

    def _patch_routerl_sumo_commands(self) -> None:
        """Allow SUMO launch diagnostics despite RouteRL hard-coding core args."""
        if not (
            self._sumo_time_to_teleport
            or self._sumo_eager_insert
            or self._sumo_sloppy_insert
            or self._sumo_emergency_insert
            or self._sumo_extra_args
        ):
            return
        simulator = getattr(self.env, "simulator", None)
        if simulator is None:
            return

        def _start(sim_self) -> None:
            sim_self.runs += 1
            individual_sumo_stats_file = os.path.join(
                sim_self.sumo_save_path,
                f"detailed_sumo_stats_{sim_self.runs}.xml",
            )
            combined_sumo_stats_file = os.path.join(
                sim_self.sumo_save_path,
                f"sumo_stats_{sim_self.runs}.xml",
            )
            sumo_cmd = self._apply_sumo_command_overrides([
                sim_self.sumo_type,
                "--seed",
                str(sim_self.seed),
                "--net-file",
                sim_self.network_file_path,
                "--additional-files",
                f"{sim_self.det_xml_save_path},{sim_self.rou_xml_save_path}",
                "--no-step-log",
                "true",
                "--time-to-teleport",
                "-1",
                "--statistic-output",
                combined_sumo_stats_file,
                "--tripinfo-output",
                individual_sumo_stats_file,
            ])
            traci.start(sumo_cmd, label=sim_self.sumo_id)
            sim_self.sumo_connection = traci.getConnection(sim_self.sumo_id)

        def _reset(sim_self) -> dict:
            det_dict = {name: 0 for name in sim_self.detectors_name}
            if sim_self.save_detectors_info:
                for det_name in sim_self.detectors_name:
                    det_dict[det_name] = sim_self.sumo_connection.lanearea.getIntervalVehicleNumber(
                        f"{det_name}_det"
                    )

            sim_self.runs += 1
            individual_sumo_stats_file = os.path.join(
                sim_self.sumo_save_path,
                f"detailed_sumo_stats_{sim_self.runs}.xml",
            )
            combined_sumo_stats_file = os.path.join(
                sim_self.sumo_save_path,
                f"sumo_stats_{sim_self.runs}.xml",
            )
            sumo_cmd = self._apply_sumo_command_overrides([
                "--seed",
                str(sim_self.seed),
                "--net-file",
                sim_self.network_file_path,
                "--additional-files",
                f"{sim_self.det_xml_save_path},{sim_self.rou_xml_save_path}",
                "--no-step-log",
                "true",
                "--time-to-teleport",
                "-1",
                "--statistic-output",
                combined_sumo_stats_file,
                "--tripinfo-output",
                individual_sumo_stats_file,
            ])
            sim_self.sumo_connection.load(sumo_cmd)
            sim_self.timestep = 0
            sim_self.waiting_vehicles = {}
            return det_dict

        simulator.start = types.MethodType(_start, simulator)
        simulator.reset = types.MethodType(_reset, simulator)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_agents(self) -> pd.DataFrame:
        """Load agent demand from agents.csv."""
        agents_csv = (
            self._records_agents_csv
            if (self._use_custom_agents and os.path.exists(self._records_agents_csv))
            else os.path.join(self.network_folder, "agents.csv")
        )
        if os.path.exists(agents_csv):
            df = pd.read_csv(agents_csv)
            print(f"  Loaded {len(df)} agents from {agents_csv}")
            return df
        # Fallback: use RouteRL's generated agents
        for agent in self.env.all_agents:
            pass  # just ensure they're created
        rows = []
        for agent in self.env.all_agents:
            rows.append({
                "id": agent.id,
                "start_time": agent.start_time,
                "origin": agent.origin,
                "destination": agent.destination,
                "kind": agent.kind,
            })
        return pd.DataFrame(rows)

    def _load_od_routes(self) -> Dict[Tuple[int, int], int]:
        """Load route count per OD pair from routes.csv."""
        routes_csv = os.path.join(self.records_folder, "routes.csv")
        od_routes: Dict[Tuple[int, int], int] = {}
        if os.path.exists(routes_csv):
            df = pd.read_csv(routes_csv)
            for (o, d), group in df.groupby(["origins", "destinations"]):
                    od_routes[(int(o), int(d))] = len(group)
        return od_routes

    def _load_od_route_paths(self) -> Dict[Tuple[int, int], List[List[str]]]:
        """Load full path edge lists per OD pair from routes.csv."""
        routes_csv = os.path.join(self.records_folder, "routes.csv")
        od_paths: Dict[Tuple[int, int], List[List[str]]] = {}
        if not os.path.exists(routes_csv):
            return od_paths
        try:
            df = pd.read_csv(routes_csv)
        except Exception:
            return od_paths
        if not {"origins", "destinations", "path"}.issubset(df.columns):
            return od_paths
        for (o, d), group in df.groupby(["origins", "destinations"], sort=True):
            paths: List[List[str]] = []
            for path_str in group["path"].tolist():
                edges = [edge for edge in str(path_str).split() if edge]
                if edges:
                    paths.append(edges)
            if paths:
                od_paths[(int(o), int(d))] = paths
        return od_paths

    def _load_sumolib_network(self):
        """Load sumolib network for geometry."""
        try:
            import sumolib
            net_file = os.path.join(self.network_folder, f"{self.network_name}.net.xml")
            if os.path.exists(net_file):
                return sumolib.net.readNet(net_file)
        except Exception as e:
            logging.warning(f"Could not load sumolib network: {e}")
        return None

    def _add_all_vehicles(self, rng: _random.Random,
                          route_probs: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
                          ) -> None:
        """Add all vehicles to SUMO with route choices.

        Args:
            rng: Random number generator.
            route_probs: Optional per-OD route probabilities {(origin, dest): (K,)}.
                         If None, uniform random routes are used.
        """
        for _, row in self._agents_df.iterrows():
            agent_id = str(int(row["id"]))
            origin_idx = int(row["origin"])
            dest_idx = int(row["destination"])
            start_time = int(row["start_time"])
            kind = str(row.get("kind", "Human"))

            # Pick route for this OD pair
            n_routes = self._od_routes.get((origin_idx, dest_idx), 1)
            od_key = (origin_idx, dest_idx)
            if route_probs is not None and od_key in route_probs:
                probs = route_probs[od_key][:n_routes].copy()
                probs = np.clip(probs, 0, None)
                s = probs.sum()
                if s > 0:
                    probs /= s
                else:
                    probs = np.ones(n_routes) / n_routes
                route_idx = int(rng.choices(range(n_routes), weights=probs.tolist(), k=1)[0])
            elif self.use_routerl_hdv_blackbox and kind.lower().startswith("human"):
                route_idx = choose_routerl_hdv_action(
                    human_lookup=self._human_agents_by_id,
                    agent_id=agent_id,
                    n_routes=n_routes,
                    rng=rng,
                    episode_actions=self._human_episode_actions,
                )
            else:
                route_idx = rng.randint(0, max(0, n_routes - 1))
            route_id = f"{origin_idx}_{dest_idx}_{route_idx}"
            self._episode_route_choices[agent_id] = int(route_idx)
            self.vehicle_routes[agent_id] = [
                list(path) for path in self._od_route_paths.get(od_key, [])[:n_routes]
            ]

            add_kwargs = {
                "vehID": agent_id,
                "routeID": route_id,
                "depart": str(start_time),
                "typeID": kind,
            }
            if self._sumo_depart_lane:
                add_kwargs["departLane"] = self._sumo_depart_lane
            if self._sumo_depart_pos:
                add_kwargs["departPos"] = self._sumo_depart_pos
            if self._sumo_depart_speed:
                add_kwargs["departSpeed"] = self._sumo_depart_speed

            try:
                traci.vehicle.add(**add_kwargs)
            except Exception as e:
                logging.debug(f"Failed to add vehicle {agent_id}: {e}")

    def _refresh_human_agent_lookup(self) -> None:
        """Refresh map: vehicle-id string -> RouteRL human agent object."""
        self._human_agents_by_id = build_human_agent_lookup(
            getattr(self.env, "human_agents", [])
        )

    def _restore_human_learning_state(self, human_learning_state: Optional[Dict[str, Any]]) -> None:
        """Restore per-human RouteRL learning state before route choices are sampled."""
        if not human_learning_state:
            return
        agents_state = human_learning_state.get("agents", {})
        if not isinstance(agents_state, dict):
            return
        restored = 0
        for human in getattr(self.env, "human_agents", []):
            aid = str(getattr(human, "id", ""))
            if not aid:
                continue
            entry = agents_state.get(aid)
            if not isinstance(entry, dict):
                continue
            model = getattr(human, "model", None)
            if model is None:
                continue
            applied = False
            if "cost" in entry and hasattr(model, "cost"):
                try:
                    cur = np.asarray(getattr(model, "cost"))
                    new = np.asarray(entry["cost"], dtype=cur.dtype)
                    if cur.shape == new.shape:
                        setattr(model, "cost", new.copy())
                        applied = True
                except Exception:
                    pass
            if "memory" in entry and hasattr(model, "memory"):
                try:
                    cur = np.asarray(getattr(model, "memory"))
                    new = np.asarray(entry["memory"], dtype=cur.dtype)
                    if cur.shape == new.shape:
                        setattr(model, "memory", new.copy())
                        applied = True
                except Exception:
                    pass
            if "beta" in entry and hasattr(model, "beta"):
                try:
                    setattr(model, "beta", float(entry["beta"]))
                    applied = True
                except Exception:
                    pass
            if applied:
                restored += 1
        if restored > 0:
            logging.info("Restored RouteRL human learning state for %d agents", restored)

    def _setup_subscriptions(self) -> None:
        """Subscribe to edge attributes for efficient batch retrieval."""
        import traci.constants as tc
        if self._skip_edge_subscriptions:
            self._edge_ids = []
            return
        if self.network is None:
            return
        self._edge_ids = []
        for edge in self.network.getEdges():
            eid = edge.getID()
            if eid.startswith(":"):
                continue
            self._edge_ids.append(eid)
            traci.edge.subscribe(eid, [
                tc.LAST_STEP_OCCUPANCY,
                tc.LAST_STEP_MEAN_SPEED,
                tc.LAST_STEP_VEHICLE_NUMBER,
                tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
            ])

    def _build_state_from_traci(self, include_edges: bool = True) -> Dict[str, Any]:
        """
        Extract current traffic state from SUMO via traci.
        Uses subscriptions for edge data (batch retrieval, much faster).
        """
        import traci.constants as tc
        state: Dict[str, Any] = {"vehicles": {}, "edges": {}}

        if not TRACI_AVAILABLE:
            return state

        try:
            vehicle_ids = traci.vehicle.getIDList()
        except Exception:
            return state

        # --- Vehicle states (lightweight — no route computation) ---
        for veh_id in vehicle_ids:
            try:
                state["vehicles"][veh_id] = {
                    "position": traci.vehicle.getPosition(veh_id),
                    "speed": traci.vehicle.getSpeed(veh_id),
                    "edge": traci.vehicle.getRoadID(veh_id),
                    "lane": traci.vehicle.getLaneIndex(veh_id),
                    "angle": traci.vehicle.getAngle(veh_id),
                    "waiting_time": traci.vehicle.getWaitingTime(veh_id),
                }
            except Exception as e:
                logging.debug(f"Error getting vehicle {veh_id} state: {e}")

        # --- Edge states via subscriptions (single batch call) ---
        if include_edges and hasattr(self, '_edge_ids') and self._edge_ids:
            for eid in self._edge_ids:
                try:
                    sub = traci.edge.getSubscriptionResults(eid)
                    if sub:
                        state["edges"][eid] = {
                            "occupancy": sub.get(tc.LAST_STEP_OCCUPANCY, 0.0),
                            "mean_speed": sub.get(tc.LAST_STEP_MEAN_SPEED, 0.0),
                            "vehicle_number": sub.get(tc.LAST_STEP_VEHICLE_NUMBER, 0),
                            "halting_number": sub.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0),
                        }
                except Exception as e:
                    logging.debug(f"Error getting edge {eid}: {e}")
        elif include_edges and self._skip_edge_subscriptions:
            edge_speeds: Dict[str, List[float]] = {}
            edge_halting: Dict[str, int] = {}
            for veh in state["vehicles"].values():
                edge_id = str(veh.get("edge", ""))
                if (not edge_id) or edge_id.startswith(":"):
                    continue
                speed = float(veh.get("speed", 0.0) or 0.0)
                edge_speeds.setdefault(edge_id, []).append(speed)
                edge_halting[edge_id] = edge_halting.get(edge_id, 0) + int(speed <= 0.1)
            for edge_id, speeds in edge_speeds.items():
                state["edges"][edge_id] = {
                    "occupancy": 0.0,
                    "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                    "vehicle_number": int(len(speeds)),
                    "halting_number": int(edge_halting.get(edge_id, 0)),
                }

        return state

    # ------------------------------------------------------------------
    # Public interface (same as SUMOConnector)
    # ------------------------------------------------------------------

    def reset(self,
              route_probs: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
              human_learning_state: Optional[Dict[str, Any]] = None,
              ) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.

        Args:
            route_probs: Optional per-OD route probabilities for HDV route sampling.

        Returns:
            State dict with keys: vehicles, edges
        """
        self.vehicle_routes = {}
        self._episode_route_choices = {}
        self.current_step = 0

        if not self._started:
            self.env.start()
            self._started = True
        else:
            try:
                self.env.simulator.reset()
                try:
                    traci.switch(self.env.simulator.sumo_id)
                except Exception:
                    pass
            except Exception:
                try:
                    traci.close()
                except Exception:
                    pass
                self.env.simulator.start()
                try:
                    traci.switch(self.env.simulator.sumo_id)
                except Exception:
                    pass

        self._episode_count += 1
        self._capture_incident_lanes()
        self._human_episode_actions = {}
        self._refresh_human_agent_lookup()
        if human_learning_state is not None:
            self._restore_human_learning_state(human_learning_state)

        # Subscribe to edge attributes for efficient batch retrieval
        self._setup_subscriptions()

        # Add all vehicles with route choices (Gawron-updated or random)
        rng = _random.Random(self.seed + self._episode_count)
        self._add_all_vehicles(rng, route_probs=route_probs)

        # Warm up: advance so vehicles enter the network
        self.advance_steps(30)

        state = self._build_state_from_traci(include_edges=False)
        num_vehicles = len(state.get("vehicles", {}))
        print(f"  Episode {self._episode_count}: {num_vehicles} vehicles in network")
        return state

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Advance the simulation by one step.

        Args:
            actions: {vehicle_id: route_choice_index}

        Returns:
            (next_state, reward, done, info)
        """
        for vehicle_id, route_idx in actions.items():
            if vehicle_id in self.vehicle_routes:
                available = self.vehicle_routes[vehicle_id]
                if 0 <= route_idx < len(available):
                    try:
                        traci.vehicle.setRoute(vehicle_id, available[route_idx])
                    except Exception as e:
                        logging.debug(f"Re-route failed for {vehicle_id}: {e}")

        try:
            self._apply_incident_for_current_time()
            traci.simulationStep()
        except Exception as e:
            logging.warning(f"simulationStep failed: {e}")

        self.current_step += 1
        next_state = self._build_state_from_traci()
        reward = self._calculate_reward(next_state)

        try:
            min_expected = traci.simulation.getMinExpectedNumber()
        except Exception:
            min_expected = 0

        done = (
            self.current_step >= self.env.simulator.simulation_length
            or min_expected <= 0
        )

        info = {
            "step": self.current_step,
            "num_vehicles": len(next_state["vehicles"]),
            "avg_speed": self._get_avg_speed(next_state),
        }

        return next_state, reward, done, info

    def advance_steps(self, n_steps: int) -> List[Tuple[str, float]]:
        """Advance SUMO by ``n_steps`` without rebuilding Python-side state.

        Returns
            A list of ``(vehicle_id, sim_time)`` arrivals observed during the
            bulk advance. This allows callers to preserve per-step arrival
            accounting while only materializing full traffic state on decision
            boundaries.
        """
        n = max(int(n_steps), 0)
        arrival_events: List[Tuple[str, float]] = []
        for _ in range(n):
            try:
                self._apply_incident_for_current_time()
                traci.simulationStep()
            except Exception as e:
                logging.warning(f"simulationStep failed during bulk advance: {e}")
                break
            self.current_step += 1
            if TRACI_AVAILABLE:
                try:
                    sim_time = float(traci.simulation.getTime())
                    arrived_ids = traci.simulation.getArrivedIDList()
                except Exception:
                    sim_time = float(self.current_step)
                    arrived_ids = []
                for veh_id in arrived_ids:
                    arrival_events.append((str(veh_id), sim_time))
        return arrival_events

    def is_done(self) -> bool:
        """Return whether the current SUMO episode has terminated."""
        try:
            min_expected = traci.simulation.getMinExpectedNumber()
        except Exception:
            min_expected = 0
        return (
            self.current_step >= self.env.simulator.simulation_length
            or min_expected <= 0
        )

    def get_traffic_state(self, include_edges: bool = True) -> Dict[str, Any]:
        """Return the current traffic state without advancing the simulation."""
        return self._build_state_from_traci(include_edges=include_edges)

    def _calculate_reward(self, state: Dict[str, Any]) -> float:
        total_waiting = sum(v["waiting_time"] for v in state["vehicles"].values())
        avg_speed = self._get_avg_speed(state)
        return -total_waiting * 0.1 + avg_speed * 0.5

    def _get_avg_speed(self, state: Dict[str, Any]) -> float:
        if not state["vehicles"]:
            return 0.0
        return sum(v["speed"] for v in state["vehicles"].values()) / len(state["vehicles"])

    def apply_routerl_hdv_learning(self, travel_times_sec: Dict[str, float]) -> int:
        """
        Apply RouteRL human-model learning with per-agent travel times from the finished episode.

        Args:
            travel_times_sec: {vehicle_id: travel_time_seconds}

        Returns:
            Number of human agents updated.
        """
        if not self.use_routerl_hdv_blackbox:
            return 0
        if not self._human_episode_actions:
            return 0
        return apply_routerl_hdv_learning_shared(
            human_lookup=self._human_agents_by_id,
            episode_actions=self._human_episode_actions,
            travel_times=travel_times_sec,
            travel_time_unit="seconds",
        )

    def close(self):
        """Shut down the SUMO connection."""
        try:
            self._restore_incident_lanes()
        except Exception:
            pass
        try:
            traci.close()
        except Exception:
            pass
        self._started = False

    def __del__(self):
        self.close()

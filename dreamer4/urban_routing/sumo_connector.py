# sumo_connector.py
"""
SUMO Environment Connector for Urban Routing

This module provides an interface to connect with OpenURB/SUMO environments,
handling traffic state extraction and multi-discrete action spaces.
"""

import os
import sys
from typing import Dict, Tuple, List, Optional, Any
import numpy as np

# Try to import SUMO libraries
try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    import traci
    import sumolib
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    print("Warning: SUMO not available. Please install SUMO and set SUMO_HOME environment variable.")


class SUMOConnector:
    """
    Connector for SUMO traffic simulation environment.

    Provides unified interface for:
    - Resetting the environment
    - Stepping the simulation with multi-discrete actions
    - Extracting traffic state (vehicles, junctions, edges)
    """

    def __init__(
        self,
        config_file: str,
        use_gui: bool = False,
        step_length: float = 1.0,
        max_steps: int = 3600,
        seed: Optional[int] = None,
        enable_rerouting: bool = False,  # NEW: control dynamic rerouting
    ):
        """
        Initialize SUMO connector.

        Args:
            config_file: Path to SUMO configuration file (.sumocfg)
            use_gui: Whether to use SUMO GUI
            step_length: Simulation step length in seconds
            max_steps: Maximum number of simulation steps
            seed: Random seed for reproducibility
            enable_rerouting: Whether to enable dynamic rerouting during simulation
                             False = vehicles follow predefined routes (for data collection)
                             True = vehicles can be rerouted dynamically (for RL training)
        """
        if not SUMO_AVAILABLE:
            raise RuntimeError("SUMO is not available. Please install SUMO.")

        self.config_file = config_file
        self.use_gui = use_gui
        self.step_length = step_length
        self.max_steps = max_steps
        self.seed = seed
        self.enable_rerouting = enable_rerouting  # NEW

        # State tracking
        self.current_step = 0
        self.vehicles = {}
        self.junctions_info = {}
        self.edges_info = {}

        # Action space info (will be populated during reset)
        self.junction_ids = []
        self.vehicle_routes = {}  # {vehicle_id: available_routes}

        # Load network (must be after junction_ids initialization)
        self.network = None
        self._load_network()

    def _load_network(self):
        """Load SUMO network from config file."""
        # Parse config to get network file
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.config_file)
        root = tree.getroot()

        net_file = None
        for input_elem in root.findall('input'):
            net_file_elem = input_elem.find('net-file')
            if net_file_elem is not None:
                net_file = net_file_elem.get('value')
                break

        if net_file is None:
            raise ValueError(f"Cannot find network file in config: {self.config_file}")

        # Make path absolute relative to config file
        config_dir = os.path.dirname(self.config_file)
        net_file = os.path.join(config_dir, net_file)

        if not os.path.exists(net_file):
            raise FileNotFoundError(f"Network file not found: {net_file}")

        self.network = sumolib.net.readNet(net_file)
        print(f"Loaded SUMO network: {net_file}")
        print(f"  Nodes: {len(self.network.getNodes())}")
        print(f"  Edges: {len(self.network.getEdges())}")

        # Extract junction information
        for node in self.network.getNodes():
            if node.getType() != "dead_end":
                self.junction_ids.append(node.getID())

        print(f"  Junctions (traffic lights/intersections): {len(self.junction_ids)}")

    def reset(self) -> Dict[str, Any]:
        """
        Reset the SUMO environment.

        Returns:
            Dict containing initial state information:
                - vehicles: Dict of vehicle states
                - junctions: Dict of junction states
                - edges: Dict of edge states
        """
        # Close existing connection if any
        try:
            traci.close()
        except:
            pass

        # Start SUMO
        sumo_cmd = [
            "sumo-gui" if self.use_gui else "sumo",
            "-c", self.config_file,
            "--step-length", str(self.step_length),
            "--no-warnings", "true",
            "--duration-log.disable", "true",
            "--no-step-log", "true",
        ]

        if self.seed is not None:
            sumo_cmd.extend(["--seed", str(self.seed)])

        print(f"Starting SUMO with config: {self.config_file}")
        traci.start(sumo_cmd)

        self.current_step = 0
        self.vehicles = {}
        self.vehicle_routes = {}

        # Step more times to let vehicles enter the network
        for _ in range(30):  # Increased from 5 to 30
            traci.simulationStep()
            self.current_step += 1

        # Get initial state
        state = self.get_traffic_state()

        num_vehicles = len(state.get("vehicles", {}))
        print(f"Reset complete: {num_vehicles} vehicles in network")

        if num_vehicles == 0:
            print("Warning: No vehicles in network. This might be a configuration issue.")
            print("Check that the .rou.xml file contains valid vehicle definitions.")
            print(f"Simulation time: {traci.simulation.getTime()}s")
            print(f"Departed vehicles: {traci.simulation.getDepartedNumber()}")
            print(f"Expected vehicles: {traci.simulation.getMinExpectedNumber()}")

        return state

    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one simulation step with given actions.

        Args:
            actions: Dict mapping vehicle_id to route_choice_index
                    {vehicle_id: route_index, ...}
                    Only applied if enable_rerouting=True

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Apply actions (re-routing decisions) only if enabled
        if self.enable_rerouting and actions:
            for vehicle_id, route_idx in actions.items():
                if vehicle_id in self.vehicle_routes:
                    available_routes = self.vehicle_routes[vehicle_id]
                    if 0 <= route_idx < len(available_routes):
                        route_edges = available_routes[route_idx]
                        try:
                            traci.vehicle.setRoute(vehicle_id, route_edges)
                        except Exception as e:
                            # Suppress warning if rerouting is disabled
                            if self.enable_rerouting:
                                print(f"Warning: Failed to set route for {vehicle_id}: {e}")

        # Step simulation
        traci.simulationStep()
        self.current_step += 1

        # Get new state
        next_state = self.get_traffic_state()

        # Calculate reward (total travel time minimization)
        reward = self._calculate_reward(next_state)

        # Check if done
        done = (
            self.current_step >= self.max_steps or
            traci.simulation.getMinExpectedNumber() <= 0
        )

        # Additional info
        info = {
            "step": self.current_step,
            "num_vehicles": len(next_state["vehicles"]),
            "avg_speed": self._get_avg_speed(next_state),
            "rerouting_enabled": self.enable_rerouting,  # NEW: for debugging
        }

        return next_state, reward, done, info

    def get_traffic_state(self) -> Dict[str, Any]:
        """
        Get current traffic state from SUMO.

        Returns:
            Dict containing:
                - vehicles: {vehicle_id: {position, speed, edge, ...}}
                - junctions: {junction_id: {waiting_time, density, ...}}
                - edges: {edge_id: {occupancy, mean_speed, ...}}
        """
        state = {
            "vehicles": {},
            "junctions": {},
            "edges": {},
        }

        # Vehicle states
        vehicle_ids = traci.vehicle.getIDList()
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

                # Get available routes for this vehicle
                # Store multiple route choices to next junction for diversity
                current_edge = traci.vehicle.getRoadID(veh_id)
                if current_edge and not current_edge.startswith(":"):
                    try:
                        # Get current route to know the destination
                        current_route = traci.vehicle.getRoute(veh_id)

                        # If vehicle has remaining route, generate alternatives
                        if len(current_route) > 1:
                            current_idx = current_route.index(current_edge)

                            # Get destination from original route
                            destination_edge_id = current_route[-1]
                            destination_edge = self.network.getEdge(destination_edge_id)

                            edge_obj = self.network.getEdge(current_edge)
                            outgoing = edge_obj.getOutgoing()

                            # Generate alternative routes to destination
                            available_routes = []

                            for out_edge in list(outgoing.keys())[:4]:
                                try:
                                    # Find path from next edge to destination
                                    route_result = self.network.getShortestPath(
                                        out_edge,
                                        destination_edge
                                    )

                                    if route_result and len(route_result[0]) > 0:
                                        # Build complete route: current -> next -> ... -> destination
                                        route_edges = [current_edge] + [e.getID() for e in route_result[0]]
                                        available_routes.append(route_edges)
                                except:
                                    # Route not found, skip
                                    continue

                            # If we found valid routes, store them
                            if len(available_routes) > 0:
                                self.vehicle_routes[veh_id] = available_routes
                    except Exception as e:
                        # Silently skip if any error
                        pass

            except Exception as e:
                print(f"Warning: Error getting vehicle {veh_id} state: {e}")

        # Junction states (traffic density, waiting time)
        for junction_id in self.junction_ids[:50]:  # Limit to 50 junctions for now
            try:
                # Get edges connected to this junction
                node = self.network.getNode(junction_id)
                incoming_edges = [e.getID() for e in node.getIncoming()]

                total_vehicles = 0
                total_waiting = 0.0
                total_speed = 0.0

                for edge_id in incoming_edges:
                    try:
                        veh_count = traci.edge.getLastStepVehicleNumber(edge_id)
                        waiting = traci.edge.getWaitingTime(edge_id)
                        mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)

                        total_vehicles += veh_count
                        total_waiting += waiting
                        total_speed += mean_speed * veh_count if veh_count > 0 else 0
                    except:
                        pass

                avg_speed = total_speed / total_vehicles if total_vehicles > 0 else 0.0

                state["junctions"][junction_id] = {
                    "vehicle_count": total_vehicles,
                    "waiting_time": total_waiting,
                    "avg_speed": avg_speed,
                    "density": total_vehicles / max(1, len(incoming_edges)),
                }
            except Exception as e:
                print(f"Warning: Error getting junction {junction_id} state: {e}")

        # Edge states
        for edge in self.network.getEdges()[:100]:  # Limit to 100 edges
            edge_id = edge.getID()
            if edge_id.startswith(":"):  # Skip internal edges
                continue

            try:
                state["edges"][edge_id] = {
                    "occupancy": traci.edge.getLastStepOccupancy(edge_id),
                    "mean_speed": traci.edge.getLastStepMeanSpeed(edge_id),
                    "vehicle_number": traci.edge.getLastStepVehicleNumber(edge_id),
                    "halting_number": traci.edge.getLastStepHaltingNumber(edge_id),
                }
            except Exception as e:
                print(f"Warning: Error getting edge {edge_id} state: {e}")

        return state

    def _calculate_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate reward based on traffic state.

        Simple reward: negative total waiting time + positive average speed
        """
        total_waiting = sum(
            v["waiting_time"] for v in state["vehicles"].values()
        )

        avg_speed = self._get_avg_speed(state)

        # Reward = -(waiting time) + (speed bonus)
        reward = -total_waiting * 0.1 + avg_speed * 0.5

        return reward

    def _get_avg_speed(self, state: Dict[str, Any]) -> float:
        """Get average speed of all vehicles."""
        if len(state["vehicles"]) == 0:
            return 0.0

        total_speed = sum(v["speed"] for v in state["vehicles"].values())
        return total_speed / len(state["vehicles"])

    def close(self):
        """Close SUMO connection."""
        try:
            traci.close()
        except:
            pass

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Example usage
if __name__ == "__main__":
    # This is just for testing, requires a valid SUMO config file
    print("SUMOConnector module")
    print("Usage: connector = SUMOConnector(config_file='path/to/config.sumocfg')")
    print("       state = connector.reset()")
    print("       next_state, reward, done, info = connector.step(actions)")

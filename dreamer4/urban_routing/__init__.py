# urban_routing module for Dreamer4
"""
Urban Routing module for applying Dreamer4 world model to traffic simulation.

This module provides:
- SUMO environment connector (sumo_connector.py)
- Data collection pipeline (data_collector.py)
- Edge registry and GNN flow projector (flow_projector.py)
- World model inference interface (urban_world_model.py)
- Rollout buffer for history management (rollout_buffer.py)
"""

from .sumo_connector import SUMOConnector
from .data_collector import DataCollector

__version__ = "0.1.0"

__all__ = [
    "SUMOConnector",
    "DataCollector",
]

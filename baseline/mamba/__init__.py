"""UrbanDreamer MAMBA baseline core.

Adapted from jbr-ai-labs/mamba under the bundled MIT license.
"""

from .config import MambaConfig
from .learner import (
    MODEL_FAMILY,
    SCHEMA_VERSION,
    MambaLearner,
    masked_world_model_loss,
    observe_sparse_batch,
)
from .networks import (
    AgentAttentionEncoder,
    MambaActor,
    MambaCritic,
    MambaWorldModel,
    RSSMRepresentation,
    RSSMState,
    RSSMTransition,
)
from .replay import (
    SparseBatch,
    SparseEpisode,
    SparseFrame,
    SparseReplayBuffer,
    collate_sparse_windows,
    fixed_slot_agent_chunks,
    rotating_agent_chunks,
)

__all__ = [
    "MODEL_FAMILY",
    "SCHEMA_VERSION",
    "AgentAttentionEncoder",
    "MambaActor",
    "MambaConfig",
    "MambaCritic",
    "MambaLearner",
    "MambaWorldModel",
    "RSSMRepresentation",
    "RSSMState",
    "RSSMTransition",
    "SparseBatch",
    "SparseEpisode",
    "SparseFrame",
    "SparseReplayBuffer",
    "collate_sparse_windows",
    "fixed_slot_agent_chunks",
    "masked_world_model_loss",
    "observe_sparse_batch",
    "rotating_agent_chunks",
]

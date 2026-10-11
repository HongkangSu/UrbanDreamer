"""Bound transformer execution for remainders while keeping their joint loss."""
from functools import wraps

import torch

from urban_dreamer.models.world_model import ConditionEncoding, DynamicsOutput


def enable_pretrain_tail_execution(trainer, *, max_batch=32):
    dynamics = trainer.world_model.dynamics
    original = dynamics.forward
    full_batch = trainer.config.resources.world_model_initialization_batch_size

    @wraps(original)
    def forward(action, *, packed_latents, flow_time, condition=None, action_events=None):
        batch = action.shape[0]
        if (torch.is_grad_enabled() and dynamics.training and max_batch < batch < full_batch
                and condition is not None and action_events is None):
            outputs = []
            for start in range(0, batch, max_batch):
                stop = min(start + max_batch, batch)
                selected = ConditionEncoding(condition.edge_tokens[start:stop],
                    condition.reference_tokens[start:stop] if condition.reference_tokens is not None else None)
                outputs.append(original(action[start:stop], packed_latents=packed_latents[start:stop],
                    flow_time=flow_time[start:stop], condition=selected).latent_velocity)
            return DynamicsOutput(torch.cat(outputs, dim=0))
        return original(action, packed_latents=packed_latents, flow_time=flow_time,
                        condition=condition, action_events=action_events)

    dynamics.forward = forward

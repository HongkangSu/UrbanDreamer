import copy

import pytest
import torch
from torch import nn

from tools.run_partial_context_fm import frozen_digest, load_selected
from tools.train_partial_context_fm import allowed_name, selected_state, DirectResidual


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamics = nn.Module()
        self.dynamics.transformer = nn.Module()
        self.dynamics.transformer.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(6)])
        self.dynamics.reference_proj = nn.Linear(2, 2, bias=False)
        self.dynamics.flow_head = nn.Linear(2, 2)
        self.register_buffer('fixed_codec_buffer', torch.tensor([3.]))


def test_selected_checkpoint_restores_only_adapted_modules_and_residual():
    model, residual = Model(), DirectResidual()
    frozen = frozen_digest(model)
    saved = selected_state(model, residual)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if allowed_name(name):
                parameter.add_(7.)
        for parameter in residual.parameters():
            parameter.add_(9.)
    load_selected(model, residual, saved)
    assert frozen_digest(model) == frozen
    for name, value in saved['model'].items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for name, value in saved['residual'].items():
        torch.testing.assert_close(residual.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize('alteration', ('missing', 'frozen_key'))
def test_selected_checkpoint_rejects_wrong_module_scope_before_copying(alteration):
    model, residual = Model(), DirectResidual()
    state = selected_state(model, residual)
    if alteration == 'missing':
        state['model'].pop(next(iter(state['model'])))
    else:
        state['model']['dynamics.transformer.layers.0.weight'] = torch.zeros(2, 2)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='exactly adapted modules'):
        load_selected(model, residual, state)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

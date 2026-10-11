import pytest
import torch

from urban_dreamer.models.policy import ActionSelection


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('size', [0, 3])
def test_host_selection_preserves_indices_and_entropy(device, size):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'cuda':
        try:
            # Use the runtime's current visible device. A transient NVML count
            # of zero must not turn an available CUDA runtime into "cuda:-1".
            device = f'cuda:{torch.cuda.current_device()}'
            available = torch.cuda.mem_get_info(device)[0]
        except RuntimeError as error:
            if 'out of memory' not in str(error).lower():
                raise
            pytest.skip('CUDA context cannot initialize while device is occupied')
        if available < 512 * 1024**2:
            pytest.skip('CUDA device busy with running experiments')
    indices = torch.tensor([0, 7, 16777217][:size], dtype=torch.int64, device=device)
    entropy = torch.tensor([0., 0.123456789, 2.3456789][:size], device=device)
    selection = ActionSelection(indices, torch.zeros(size, device=device), entropy)
    assert selection.indices_and_entropy() == (indices.cpu().tolist(), entropy.cpu().tolist())

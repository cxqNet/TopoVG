"""The existing three-tensor CPU FP32 .pt contract."""
from ..config import FEATURE_KEYS
from .backend import LazyModule

torch = LazyModule('torch')

def validate_features(features):
    if set(features) != FEATURE_KEYS:
        raise ValueError(f'Feature keys must be exactly {sorted(FEATURE_KEYS)}')
    for key, shape in [('semantic_mask_8', (8, 8)),
                       ('semantic_mask_16', (16, 16)), ('self_64', (4096, 4096))]:
        tensor = features[key]
        if tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError(f'{key}: expected CPU FP32 {shape}, got {tensor.shape}/{tensor.dtype}')
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{key}: NaN/Inf in features')

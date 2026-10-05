"""Decode an existing .pt file and optionally save both visualizations."""
import math
from pathlib import Path
from ..decoding import tfbd
from ..decoding.tfbd import predict, prepare_decoder_image
from ..utils.features import validate_features
from .visualization import save_visualizations

def evaluate_and_save(item):
    sample, feature_root, results_root, save_images = item
    tfbd.load_backend()
    path = Path(feature_root) / f"{sample['id']}.pt"
    features = tfbd.torch.load(path, map_location='cpu', weights_only=True)
    validate_features(features)
    decoded = predict(features, sample, prepare_decoder_image(sample), return_heatmap=save_images)
    row, heatmap = decoded if save_images else (decoded, None)
    del features
    if not all(math.isfinite(float(row[key])) for key in ('iou', 'intersection', 'union')):
        raise ValueError(f"{sample['id']}: nonfinite evaluation result")
    if save_images:
        save_visualizations(sample, row, heatmap, results_root)
    row['pred_bbox_xyxy'] = [float(v) for v in row['bbox']]
    row['expression'] = sample['expression']
    return sample['id'], row

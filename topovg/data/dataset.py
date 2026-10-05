"""Read a complete split; generation only consumes images and expressions."""
import json
import math
from pathlib import Path
import re

def load_samples(data_root, split='test', require_gt=True):
    root = Path(data_root).expanduser().resolve()
    path = root / 'split' / f'{split}.json'
    with path.open(encoding='utf-8') as f:
        samples = json.load(f)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f'{path}: expected a nonempty list')
    seen = set()
    for sample in samples:
        sid = str(sample['id'])
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid in seen:
            raise ValueError(f'Invalid or duplicate sample id: {sid}')
        seen.add(sid)
        sample['id'] = sid
        if not isinstance(sample['expression'], str) or not sample['expression'].strip():
            raise ValueError(f'{sid}: empty expression')
        if require_gt:
            width, height = int(sample['width']), int(sample['height'])
            box = sample['bbox_xyxy']
            if (width <= 0 or height <= 0 or len(box) != 4
                    or not all(math.isfinite(float(v)) for v in box)
                    or not 0 <= box[0] < box[2] <= width
                    or not 0 <= box[1] < box[3] <= height):
                raise ValueError(f'{sid}: invalid dimensions/bbox_xyxy')
        # New split format: image is normally a stem, e.g. "00005".
        image = str(sample['image'])
        if Path(image).name != image:
            raise ValueError(f'{sid}: image must be a basename or stem')
        candidates = [root / 'JPEGImages' / image] if Path(image).suffix else [
            root / 'JPEGImages' / (image + ext) for ext in ('.jpg', '.png', '.jpeg')]
        found = [p for p in candidates if p.is_file()]
        if len(found) != 1:
            raise FileNotFoundError(f'{sid}: expected exactly one image: {candidates}')
        sample['_image_path'] = str(found[0])
    if not require_gt:
        return [{key: sample[key] for key in ('id', 'image', 'expression', '_image_path')}
                for sample in samples]
    return samples

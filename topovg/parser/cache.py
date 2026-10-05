"""Expression-keyed JSON cache and split expression grouping."""
from collections import OrderedDict
import json
from ..utils.text import normalize_text

def read_dictionary(path):
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected expression -> parsed graph dictionary')
    return value


def load_expressions(split_path):
    samples = json.loads(split_path.read_text(encoding='utf-8'))
    if not isinstance(samples, list) or not samples:
        raise ValueError('split JSON must be a nonempty list')
    groups = OrderedDict()
    for i, sample in enumerate(samples):
        text = sample.get('expression') if isinstance(sample, dict) else None
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError(f'split row {i}: missing/empty expression')
        text = text.strip()
        key = normalize_text(text)
        group = groups.setdefault(key, {'expression': text, 'count': 0})
        group['count'] += 1
    return samples, groups

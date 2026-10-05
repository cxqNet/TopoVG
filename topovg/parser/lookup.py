"""Resolve a parsed graph for every generation sample."""
import json
from pathlib import Path
from ..utils.text import normalize_text
from .schema import graph_error

def find_graphs(samples, parser_path):
    with Path(parser_path).expanduser().open(encoding='utf-8') as f:
        trees = json.load(f)
    if not isinstance(trees, dict):
        raise ValueError('Parser JSON must be expression -> parsed graph')
    normalized = {}
    for text, graph in trees.items():
        key = normalize_text(text)
        if key in normalized and normalized[key] != graph:
            raise ValueError(f'Conflicting normalized parser keys: {text}')
        normalized[key] = graph
    graphs, missing = {}, []
    for s in samples:
        text = s['expression'].strip()
        graph = trees.get(text, normalized.get(normalize_text(text)))
        if graph_error(graph) is not None:
            missing.append((s['id'], text))
        else:
            graphs[s['id']] = graph
    if missing:
        raise ValueError(f'{len(missing)} missing/invalid parser entries; first: {missing[:5]}')
    return graphs

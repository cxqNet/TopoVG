"""Graph validation, normalized lookup and coverage accounting."""
from collections import Counter
from ..utils.text import normalize_text

def graph_error(graph):
    """Check fields the SemanticExecutor actually uses; no semantic guessing."""
    def object_error(node, path):
        if not isinstance(node, dict):
            return f'{path} must be an object'
        words = node.get('compound_words')
        if not isinstance(words, list) or not words or not all(
                isinstance(w, str) and w.strip() for w in words):
            return f'{path}.compound_words must be a nonempty list of strings'
        attrs = node.get('attributes', [])
        if not isinstance(attrs, list) or not all(isinstance(v, str) for v in attrs):
            return f'{path}.attributes must be a list of strings'
        return None

    if not isinstance(graph, dict):
        return 'graph must be an object'
    error = object_error(graph.get('target'), 'target')
    if error:
        return error
    rel = graph.get('relation')
    if rel is None:
        return None
    if not isinstance(rel, dict):
        return 'relation must be an object or null'
    direction = rel.get('direction')
    if direction is not None and not isinstance(direction, str) and not (
            isinstance(direction, list) and all(isinstance(v, str) for v in direction)):
        return 'relation.direction must be a string/list of strings or null'
    anchor = rel.get('anchor')
    if anchor is None:
        return None
    # These two legacy shapes are supported by the current SemanticExecutor.
    if isinstance(anchor, str) and anchor.strip():
        return None
    if isinstance(anchor, list) and anchor and all(isinstance(v, str) and v.strip() for v in anchor):
        return None
    return object_error(anchor, 'relation.anchor')


def index_dictionary(trees):
    index = {}
    for text, graph in trees.items():
        if not isinstance(text, str):
            raise ValueError('Parser keys must be strings')
        key = normalize_text(text)
        if key in index and trees[index[key][0]] != graph:
            raise ValueError('Conflicting parser entries after text normalization: '
                             f'{index[key][0]!r} and {text!r}. Resolve these before filling.')
        index.setdefault(key, []).append(text)
    return index


def build_pending(groups, trees):
    index = index_dictionary(trees)
    counts = Counter({'valid_unique': 0, 'missing_unique': 0, 'invalid_unique': 0,
                      'valid_samples': 0, 'pending_samples': 0})
    pending = []
    for key, group in groups.items():
        aliases = index.get(key, [])
        if aliases and graph_error(trees[aliases[0]]) is None:
            counts['valid_unique'] += 1
            counts['valid_samples'] += group['count']
        else:
            kind = 'invalid' if aliases else 'missing'
            counts[kind+'_unique'] += 1
            counts['pending_samples'] += group['count']
            pending.append((key, group['expression'], aliases))
    return pending, dict(counts)

"""Incremental parser completion; existing valid graphs are skipped."""
from .schema import build_pending, graph_error

def fill_pending(trees, groups, parser, save, max_retries=2, max_new_tokens=512,
                 limit=0, prior_failures=None):
    """Testable update loop: keep valid entries; replace only missing/invalid."""
    pending, initial_counts = build_pending(groups, trees)
    failures = dict(prior_failures or {})
    attempted = successes = 0
    for key, expression, aliases in pending:
        if limit and attempted >= limit:
            break
        attempted += 1
        print(f'[{attempted}/{min(limit, len(pending)) if limit else len(pending)}] {expression}', flush=True)
        graph = None
        raw = ''
        reason = ''
        for attempt in range(max_retries):
            graph, raw = parser.parse(expression, max_new_tokens, attempt)
            reason = graph_error(graph)
            if reason is None:
                break
            print(f'  Invalid attempt {attempt+1}/{max_retries}: {reason}', flush=True)
        if reason is None:
            # Update aliases together so generate_features.py cannot see two
            # conflicting graphs for one normalized expression.
            for alias in aliases:
                trees[alias] = graph
            if not aliases:
                trees[expression] = graph
            failures.pop(key, None)
            successes += 1
            save(trees, failures)
            print('  Saved', flush=True)
        else:
            failures[key] = {'expression': expression, 'error': reason, 'raw_output': raw}
            save(None, failures)
            print('  Failed; recorded for the next run', flush=True)
    remaining, counts = build_pending(groups, trees)
    return {'initial': initial_counts, 'final': counts, 'attempted': attempted,
            'successes': successes, 'remaining_unique': len(remaining)}, failures

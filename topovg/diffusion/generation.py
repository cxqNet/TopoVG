"""Single-pass dataset traversal, with existing features protected by default."""
from pathlib import Path
from ..utils.file_io import atomic_features


def run_generation(samples, feature_root, engine, overwrite=False, save=atomic_features):
    """Extract features once for each pending sample."""
    feature_root = Path(feature_root)
    generated = skipped = 0
    for index, sample in enumerate(samples, 1):
        path = feature_root / f"{sample['id']}.pt"
        if path.is_file() and not overwrite:
            skipped += 1
            print(f"{index}/{len(samples)}  |  {sample['id']}: skipped (exists)", flush=True)
            continue
        features = None
        try:
            engine.prepare(sample)
            features = engine.draw(sample)
            save(path, features)
            generated += 1
            print(f"{index}/{len(samples)}  |  {sample['id']}: saved {path.name}", flush=True)
        finally:
            try:
                if features is not None:
                    engine.release(features)
            finally:
                engine.finish()
    return {'generated': generated, 'skipped': skipped, 'total': len(samples)}
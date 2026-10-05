"""Shared feature and result path conventions."""
from pathlib import Path


def feature_directory(data_root, feature_dir=None):
    return (Path(feature_dir) if feature_dir else Path(data_root) / 'features').expanduser().resolve()


def result_directory(data_root, results_dir=None):
    if results_dir:
        return Path(results_dir).expanduser().resolve()
    dataset = Path(data_root).expanduser().resolve().name
    return (Path('outputs') / f'{dataset}_evaluation').resolve()

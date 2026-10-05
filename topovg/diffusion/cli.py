"""Generate one feature file per image-expression sample for an entire split."""
import argparse
from pathlib import Path
from ..config import DEFAULT_DATA_ROOT, DEFAULT_PARSER_JSON, DEFAULT_SD_MODEL
from ..data.dataset import load_samples
from ..data.paths import feature_directory
from ..parser.lookup import find_graphs
from ..utils.file_io import feature_lock
from .pipeline import DiffusionEngine
from .generation import run_generation


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    cli.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    cli.add_argument('--parser-json', default=DEFAULT_PARSER_JSON)
    cli.add_argument('--model-path', default=DEFAULT_SD_MODEL,
                     help='Local SD1.5 directory; also settable via TOPOVG_SD_MODEL')
    cli.add_argument('--device', default='cuda:0')
    cli.add_argument('--timestep', type=int, default=100)
    cli.add_argument('--query-chunk', type=int, default=512,
                     help='Attention query slice; 0 disables slicing')
    cli.add_argument('--feature-dir', help='Default: data-root/features')
    cli.add_argument('--output', help='Legacy path: write to output/features/')
    cli.add_argument('--overwrite', action='store_true',
                     help='Explicitly replace existing .pt files with a new single draw')
    cli.add_argument('--check-only', action='store_true',
                     help='Check dataset and parser coverage without loading a model or writing files')
    args = cli.parse_args()
    if not 0 <= args.timestep < 1000 or args.query_chunk < 0:
        cli.error('--timestep must be in [0,999] and --query-chunk must be >= 0')
    if args.output and args.feature_dir:
        cli.error('Use either --feature-dir or the legacy --output, not both')
    samples = load_samples(args.data_root, args.split, require_gt=False)
    root = (Path(args.output).expanduser().resolve() / 'features' if args.output
            else feature_directory(args.data_root, args.feature_dir))

    def pending_samples():
        return [s for s in samples if args.overwrite or not (root / f"{s['id']}.pt").is_file()]

    pending = pending_samples()
    if args.check_only:
        find_graphs(pending, args.parser_json) if pending else None
        print(f'Samples: {len(samples)} | pending: {len(pending)} | features: {root}', flush=True)
        return 0
    if pending and not Path(args.model_path).expanduser().is_dir():
        cli.error(f'Local SD1.5 directory not found: {args.model_path}; set --model-path')
    root.mkdir(parents=True, exist_ok=True)
    print(f'Generating {len(samples)} samples | features: {root}', flush=True)
    try:
        with feature_lock(root):
            # Recheck after locking; completed files also provide interruption recovery.
            pending = pending_samples()
            graphs = find_graphs(pending, args.parser_json) if pending else {}
            engine = DiffusionEngine(args, graphs)
            counts = run_generation(samples, root, engine, overwrite=args.overwrite)
    except KeyboardInterrupt:
        print('\nInterrupted. Completed feature files have been retained.', flush=True)
        return 130
    print(
        f"Complete | generated: {counts['generated']} | "
        f"skipped: {counts['skipped']} | total: {counts['total']}",
        flush=True,
    )
    return 0
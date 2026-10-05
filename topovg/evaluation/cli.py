"""Evaluate an entire split from saved features, with live sample output."""
import argparse
from pathlib import Path
from ..config import DEFAULT_DATA_ROOT, SAVE_IMAGES_DEFAULT
from ..data.dataset import load_samples
from ..data.paths import feature_directory, result_directory
from .metrics import MetricBook
from .reporting import parse_bool, sample_message, final_message, save_evaluation
from .worker import evaluate_and_save
from .parallel import parallel_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--feature-dir', help='Default: data-root/features; read-only input')
    parser.add_argument('--output', help='Legacy input path: a directory containing features/')
    parser.add_argument('--results-dir', help='Default: ./outputs/<dataset>_evaluation')
    parser.add_argument('--save-images', type=parse_bool, default=SAVE_IMAGES_DEFAULT,
                        metavar='{true,false}', help='Save localization.png and heatmap.png; default true')
    parser.add_argument('--workers', type=int, default=4,
                        help='Each worker needs at least ~150 MiB CPU RAM; start with 1-4.')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Checkpoint evaluation.txt every N completed samples; default 50')
    args = parser.parse_args()
    if args.workers < 1 or args.save_every < 1:
        parser.error('--workers and --save-every must be >= 1')
    samples = load_samples(args.data_root, args.split)
    if args.output and args.feature_dir:
        parser.error('Use either --feature-dir or the legacy --output, not both')
    feature_root = (Path(args.output).expanduser().resolve() / 'features' if args.output
                    else feature_directory(args.data_root, args.feature_dir))
    results_root = result_directory(args.data_root, args.results_dir)
    if results_root == feature_root or feature_root in results_root.parents:
        parser.error('--results-dir must be outside the input features/ directory')
    missing = [s['id'] for s in samples if not (feature_root / f"{s['id']}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing feature files; first: {missing[:10]}')
    results_root.mkdir(parents=True, exist_ok=True)
    result_path = results_root / 'evaluation.txt'
    book = MetricBook(len(samples))
    lines = []
    jobs = [(s, str(feature_root), str(results_root), args.save_images) for s in samples]
    print(
        f'Evaluating {len(samples)} samples | workers={args.workers} | '
        f'save_images={str(args.save_images).lower()}',
        flush=True,
    )
    save_evaluation(result_path, lines, book, 'running')
    results = (map(evaluate_and_save, jobs) if args.workers == 1
               else parallel_results(jobs, args.workers))
    status = 'failed'
    try:
        for index, (sid, row) in enumerate(results, 1):
            book.update(sid, row)
            message = sample_message(index, len(samples), sid, row)
            print(message, flush=True)
            lines.append(message)
            if index % args.save_every == 0:
                save_evaluation(result_path, lines, book, 'running')
        status = 'complete'
    except KeyboardInterrupt:
        status = 'interrupted'
        print('\nInterrupted; saving completed sample results.', flush=True)
    finally:
        try:
            if hasattr(results, 'close'):
                results.close()
        finally:
            save_evaluation(result_path, lines, book, status)
    if status == 'complete':
        print(final_message(book.metrics()), flush=True)
    else:
        print(f'Incomplete: {len(book.rows)}/{book.total} | status={status}', flush=True)
    print(f'Results saved to: {result_path}', flush=True)
    return 0 if status == 'complete' else 130
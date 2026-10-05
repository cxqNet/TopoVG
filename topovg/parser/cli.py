"""Complete the parser cache from a dataset split using local Qwen."""
import argparse
import json
from pathlib import Path
from ..config import DEFAULT_QWEN_MODEL
from ..utils.file_io import atomic_json, backup_file, file_signature, output_lock
from .cache import read_dictionary, load_expressions
from .schema import build_pending, graph_error
from .local_llm import LocalQwenParser
from .fill import fill_pending

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--split-json', default='./Data/RRSIS-D/split/test.json')
    cli.add_argument('--parser-json', default='./JsonTree.json')
    cli.add_argument('--output-json', help='Omit to fill parser-json in place with an automatic backup.')
    cli.add_argument('--model-path', default=DEFAULT_QWEN_MODEL)
    cli.add_argument('--device', default='cuda:0')
    cli.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16', 'float32'], default='auto')
    cli.add_argument('--max-new-tokens', type=int, default=512)
    cli.add_argument('--max-retries', type=int, default=2, help='Total attempts per expression')
    cli.add_argument('--limit', type=int, default=0, help='0: all pending expressions; useful for a small smoke run')
    cli.add_argument('--check-only', action='store_true', help='Report coverage without model load or file writes')
    args = cli.parse_args()
    if args.max_new_tokens < 1 or args.max_retries < 1 or args.limit < 0:
        cli.error('max-new-tokens/max-retries must be positive and limit nonnegative')
    source = Path(args.parser_json).expanduser().resolve()
    output = Path(args.output_json).expanduser().resolve() if args.output_json else source
    split_path = Path(args.split_json).expanduser().resolve()
    if not source.is_file():
        cli.error(f'Existing parser file not found: {source}')
    if output == split_path:
        cli.error('output-json must not overwrite the dataset split')
    samples, groups = load_expressions(split_path)

    def merge_current():
        trees = read_dictionary(source)
        # A separate output file also acts as a resume checkpoint. Do not silently
        # overwrite an existing graph if it conflicts with the source dictionary.
        if output != source and output.exists():
            for text, graph in read_dictionary(output).items():
                if text in trees and trees[text] != graph and graph_error(trees[text]) is None:
                    raise ValueError(f'Output/source conflict for {text!r}; check the two files')
                trees[text] = graph
        return trees

    trees = merge_current()
    pending, counts = build_pending(groups, trees)
    print(f'Test samples={len(samples)}, normalized unique expressions={len(groups)}, '
          f'parser entries={len(trees)}', flush=True)
    print('Coverage:', json.dumps(counts, ensure_ascii=False), flush=True)
    print(f'Local LLM needs {len(pending)} unique expressions, representing '
          f"{counts['pending_samples']} sample rows.", flush=True)
    if args.check_only:
        print('Check only: no model loaded; no files modified.', flush=True)
        return 0 if not pending else 2
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path).expanduser().resolve()
    failure_path = output.with_name(output.stem+'.fill_failures.json')
    with output_lock(output):
        # Re-read after acquiring the lock; a prior worker may have finished.
        trees = merge_current()
        pending, counts = build_pending(groups, trees)
        target_signature = file_signature(output)
        original_signature = file_signature(source)
        backup = None
        first_write = True

        def save(value, failures):
            nonlocal target_signature, first_write, backup
            if file_signature(output) != target_signature:
                raise RuntimeError('Parser output was modified by another process; refusing to overwrite it')
            if output != source and file_signature(source) != original_signature:
                raise RuntimeError('Parser source changed during filling; rerun after checking it')
            if value is not None:
                if first_write:
                    backup = backup_file(output)
                    if backup:
                        print(f'Original file backed up: {backup}', flush=True)
                    first_write = False
                atomic_json(output, value)
                target_signature = file_signature(output)
            if failures:
                atomic_json(failure_path, failures)
            elif failure_path.exists():
                failure_path.unlink()

        if not pending:
            # Separate output still needs materialization even when every graph
            # already exists in the source. In-place coverage needs no write.
            if output != source:
                save(trees, {})
            print(f'Complete coverage. Ready for generate_features.py: {output}', flush=True)
            return 0
        if not model_path.is_dir():
            cli.error(f'Local model directory not found: {model_path}; set --model-path')
        parser = LocalQwenParser(model_path, args.device, args.dtype)
        prior_failures = read_dictionary(failure_path)
        try:
            report, _ = fill_pending(trees, groups, parser, save, args.max_retries,
                                     args.max_new_tokens, args.limit, prior_failures)
        except KeyboardInterrupt:
            print('\nInterrupted. All completed expressions were saved. Rerun the same command to resume.', flush=True)
            return 130
        print('Final coverage:', json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if report['remaining_unique']:
            print(f"Not yet complete: {report['remaining_unique']} expressions remain. "
                  'Rerun the same command; completed entries will be skipped.', flush=True)
            return 2
        print(f'All {len(samples)} test rows covered. Use --parser-json "{output}"', flush=True)
        return 0

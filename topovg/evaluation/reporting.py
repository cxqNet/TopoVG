"""Per-sample console lines and the atomic evaluation.txt report."""
import argparse
import json
import os
from pathlib import Path


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() == 'true':
        return True
    if str(value).lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Use true or false.')


def sample_message(index, total, sid, row):
    bbox_text = json.dumps([round(float(v), 6) for v in row['pred_bbox_xyxy']])
    expression = ' '.join(row['expression'].split())
    return (f"{index}/{total}  |  {sid}: IoU={row['iou']:.6f}, "
            f"pred_bbox_xyxy: {bbox_text}, {expression}")


def final_message(metrics):
    return 'Final metrics: ' + ' | '.join(
        f'{key}={metrics[key]:.4f}'
        for key in ('Pr@0.3', 'Pr@0.5', 'Pr@0.7', 'mIoU', 'oIoU'))


def save_evaluation(path, lines, book, status):
    complete = len(book.rows) == book.total and status == 'complete'
    footer = (final_message(book.metrics()) if complete
              else f'Incomplete: {len(book.rows)}/{book.total} | status={status}')
    text = '\n'.join(lines) + ('\n\n' if lines else '') + footer + '\n'
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
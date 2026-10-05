"""Continuous-coordinate bbox IoU and aggregate percentage metrics."""
def compute_iou(box1, box2):
    inter = max(0.0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(
        0.0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return (inter / union if union > 0 else 0.0), inter, union


class MetricBook:
    """Replace one sample in O(1), keeping the denominator the entire split."""
    def __init__(self, total):
        self.total = total
        self.rows = {}
        self.sum_iou = self.inter = self.union = 0.0
        self.hits = {0.3: 0, 0.5: 0, 0.7: 0}

    def update(self, sid, row):
        if sid in self.rows:
            self._add(self.rows[sid], -1)
        self.rows[sid] = row
        self._add(row, 1)

    def _add(self, row, sign):
        self.sum_iou += sign * row['iou']
        self.inter += sign * row['intersection']
        self.union += sign * row['union']
        for threshold in self.hits:
            self.hits[threshold] += sign * int(row['iou'] >= threshold)

    def metrics(self):
        return {'count': len(self.rows), 'total': self.total,
                'mIoU': 100.0 * self.sum_iou / self.total,
                'oIoU': 100.0 * self.inter / self.union if self.union > 0 else 0.0,
                **{f'Pr@{t:.1f}': 100.0 * h / self.total for t, h in self.hits.items()}}

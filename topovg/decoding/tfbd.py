"""Original attention fusion, sparse refinement and affinity-based bbox decoding."""
from ..utils import backend as _backend_environment
from ..config import (PAD_RATIO, BETA_C, BETA_H, LAMBDA_REG, CONF_ALPHA,
                      TOPK_PEAKS, PEAK_THRESH, MIN_PEAK_DIST)
from ..evaluation.metrics import compute_iou

_backend_ready = False

def load_backend():
    global torch, F, np, cv2, sp, splinalg, _backend_ready
    if not _backend_ready:
        import torch
        import torch.nn.functional as F
        import numpy as np
        import cv2
        import scipy.sparse as sp
        import scipy.sparse.linalg as splinalg
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        _backend_ready = True


def prepare_decoder_image(sample):
    """Prepare the fixed decoder image preprocessing."""
    load_backend()
    img = cv2.imread(sample['_image_path'])
    if img is None:
        raise ValueError(f"Unreadable image: {sample['_image_path']}")
    if img.shape[1] != int(sample['width']) or img.shape[0] != int(sample['height']):
        raise ValueError(f"{sample['id']}: split dimensions do not match image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    return cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32) / 255.0


def predict(features, sample, decoder_image=None, return_heatmap=False):
    load_backend()
    # This is the original per-scale projection -> resize -> normalize -> fuse.
    maps = []
    for res, weight in [(8, 0.15), (16, 0.85)]:
        mask = features[f'semantic_mask_{res}']
        mask = F.interpolate(mask[None, None], size=(64, 64), mode='bilinear',
                             align_corners=False).squeeze()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        maps.append(mask * weight)
    fused = torch.stack(maps).sum(0)
    self_att = features['self_64']
    response = (self_att @ fused.reshape(4096, 1)).reshape(64, 64).numpy()
    decoded = extract_bbox_sagfs_sata(response, self_att, sample['_image_path'],
                                  int(sample['width']), int(sample['height']),
                                  PAD_RATIO, decoder_image, return_heatmap=return_heatmap)
    bbox, heatmap = decoded if return_heatmap else (decoded, None)
    iou, inter, union = compute_iou(bbox, sample['bbox_xyxy'])
    row = {'bbox': bbox, 'iou': iou, 'intersection': inter, 'union': union}
    return (row, heatmap) if return_heatmap else row


def apply_morphology(binary_mask):
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)


def find_topk_local_peaks(heatmap_norm, k=3, peak_thresh=0.6, min_dist=15):
    kernel = np.ones((3, 3), np.uint8)
    heatmap_dilated = cv2.dilate(heatmap_norm, kernel)
    local_max_mask = (heatmap_norm == heatmap_dilated) & (heatmap_norm >= peak_thresh)
    peak_coords = np.argwhere(local_max_mask)
    if len(peak_coords) == 0: return []

    peak_coords = sorted(peak_coords, key=lambda p: float(heatmap_norm[p[0], p[1]]), reverse=True)
    selected = []
    for p in peak_coords:
        y, x = int(p[0]), int(p[1])
        keep = True
        for sx, sy in selected:
            if (x - sx) ** 2 + (y - sy) ** 2 < min_dist ** 2:
                keep = False
                break
        if keep: selected.append((x, y))
        if len(selected) >= k: break
    return selected


def extract_bbox_sagfs_sata(heatmap, self_att, image_path, img_w, img_h, pad_ratio=0.0, prepared_image=None, return_heatmap=False):
    WORK_RES = 256
    N = WORK_RES * WORK_RES

    if prepared_image is None:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unreadable image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (WORK_RES, WORK_RES))
        img_smoothed = cv2.GaussianBlur(img_resized, (3, 3), 0)
        img_norm = img_smoothed.astype(np.float32) / 255.0
    else:
        img_norm = prepared_image

    heatmap_resized = cv2.resize(heatmap, (WORK_RES, WORK_RES), interpolation=cv2.INTER_CUBIC)
    hm_min, hm_max = heatmap_resized.min(), heatmap_resized.max()
    hm_norm = (heatmap_resized - hm_min) / (hm_max - hm_min + 1e-8)


    diff_c_h = np.sum((img_norm[:, :-1, :] - img_norm[:, 1:, :]) ** 2, axis=-1)
    diff_h_h = (hm_norm[:, :-1] - hm_norm[:, 1:]) ** 2
    w_h = np.exp(-(BETA_C * diff_c_h + BETA_H * diff_h_h))

    diff_c_v = np.sum((img_norm[:-1, :, :] - img_norm[1:, :, :]) ** 2, axis=-1)
    diff_h_v = (hm_norm[:-1, :] - hm_norm[1:, :]) ** 2
    w_v = np.exp(-(BETA_C * diff_c_v + BETA_H * diff_h_v))

    idx_nodes = np.arange(N).reshape(WORK_RES, WORK_RES)
    edges_h = np.vstack([idx_nodes[:, :-1].ravel(), idx_nodes[:, 1:].ravel()])
    edges_v = np.vstack([idx_nodes[:-1, :].ravel(), idx_nodes[1:, :].ravel()])

    row = np.concatenate([edges_h[0], edges_h[1], edges_v[0], edges_v[1]])
    col = np.concatenate([edges_h[1], edges_h[0], edges_v[1], edges_v[0]])
    data = np.concatenate([w_h.ravel(), w_h.ravel(), w_v.ravel(), w_v.ravel()])

    W_mat = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
    D_vec = np.array(W_mat.sum(axis=1)).ravel()
    L = sp.diags(D_vec) - W_mat

    q = 1.0 + CONF_ALPHA * (np.abs(hm_norm - 0.5) * 2.0) ** 2
    Q = sp.diags(q.ravel())

    try:
        x = splinalg.spsolve(L + LAMBDA_REG * Q, LAMBDA_REG * (Q @ hm_norm.ravel()))
        refined_field = x.reshape(WORK_RES, WORK_RES)
    except Exception:
        refined_field = hm_norm

    binary_mask = (refined_field >= 0.6).astype(np.uint8) * 255
    binary_mask = apply_morphology(binary_mask)


    peaks = find_topk_local_peaks(hm_norm, k=TOPK_PEAKS, peak_thresh=PEAK_THRESH, min_dist=MIN_PEAK_DIST)
    valid_peaks = []

    if peaks:
        primary_peak = peaks[0]
        valid_peaks.append(primary_peak)
        grid_x0 = min(int(primary_peak[0] / WORK_RES * 64), 63)
        grid_y0 = min(int(primary_peak[1] / WORK_RES * 64), 63)
        idx0 = grid_y0 * 64 + grid_x0

        AFFINITY_THRESHOLD = 2.0 / 4096.0
        for i in range(1, len(peaks)):
            grid_xi = min(int(peaks[i][0] / WORK_RES * 64), 63)
            grid_yi = min(int(peaks[i][1] / WORK_RES * 64), 63)
            idxi = grid_yi * 64 + grid_xi
            if float(self_att[idx0, idxi]) > AFFINITY_THRESHOLD:
                valid_peaks.append(peaks[i])
    else:
        py, px = np.unravel_index(np.argmax(hm_norm), hm_norm.shape)
        valid_peaks.append((px, py))


    num_labels, labels_im = cv2.connectedComponents(binary_mask)
    selected_labels = set()

    for px, py in valid_peaks:
        lbl = labels_im[py, px]
        if lbl > 0:
            selected_labels.add(lbl)


    if len(selected_labels) == 0:
        binary_mask_fallback = (hm_norm >= 0.60).astype(np.uint8) * 255
        num_labels, labels_im = cv2.connectedComponents(binary_mask_fallback)
        py, px = np.unravel_index(np.argmax(hm_norm), hm_norm.shape)
        lbl = labels_im[py, px]
        if lbl > 0: selected_labels.add(lbl)


    xs_all, ys_all = [], []
    for lbl in selected_labels:
        ys, xs = np.where(labels_im == lbl)
        xs_all.extend(xs)
        ys_all.extend(ys)

    if not xs_all:
        bbox = [0.0, 0.0, float(img_w), float(img_h)]
        return (bbox, refined_field) if return_heatmap else bbox

    xmin_base, xmax_base = min(xs_all), max(xs_all)
    ymin_base, ymax_base = min(ys_all), max(ys_all)

    scale_x = img_w / WORK_RES
    scale_y = img_h / WORK_RES

    xmin = xmin_base * scale_x
    ymin = ymin_base * scale_y
    xmax = (xmax_base + 1) * scale_x
    ymax = (ymax_base + 1) * scale_y

    pad_x = img_w * pad_ratio
    pad_y = img_h * pad_ratio

    bbox = [float(max(0, xmin - pad_x)), float(max(0, ymin - pad_y)),
            float(min(img_w, xmax + pad_x)), float(min(img_h, ymax + pad_y))]
    return (bbox, refined_field) if return_heatmap else bbox

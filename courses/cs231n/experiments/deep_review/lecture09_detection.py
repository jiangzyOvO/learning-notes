"""Continuous xyxy IoU, class-aware NMS and toy matching; no detector training.

Simplified evaluation: one class, no crowd/ignore/max-detection conventions.
NMS suppresses IoU > threshold. Evaluation accepts IoU >= threshold.
"""
import numpy as np


def pairwise_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 4 or b.shape[1] != 4:
        raise ValueError('Expected arrays shaped (N,4) and (M,4)')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Coordinates must be finite')
    if np.any(a[:, 2:] < a[:, :2]) or np.any(b[:, 2:] < b[:, :2]):
        raise ValueError('Reversed box endpoints')
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                    - np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def class_aware_nms(boxes, scores, labels, threshold=0.5):
    order = np.argsort(-scores, kind='stable')
    kept = []
    while len(order):
        top = order[0]
        kept.append(int(top))
        remaining = order[1:]
        overlap = pairwise_iou(boxes[[top]], boxes[remaining])[0]
        suppress = (labels[remaining] == labels[top]) & (overlap > threshold)
        order = remaining[~suppress]
    return np.array(kept, dtype=int)


def evaluate_one_class(boxes, scores, ground_truth, threshold=0.5):
    """Toy score-ordered greedy matching, a GT can be used at most once."""
    order = np.argsort(-scores, kind='stable')
    used = np.zeros(len(ground_truth), dtype=bool)
    tp = np.zeros(len(order), dtype=int)
    for rank, idx in enumerate(order):
        overlaps = pairwise_iou(boxes[[idx]], ground_truth)[0]
        available = np.where(~used, overlaps, -1)
        if len(available):
            match = int(np.argmax(available))
            if available[match] >= threshold:
                tp[rank] = 1
                used[match] = True
    cum_tp = np.cumsum(tp)
    precision = cum_tp / np.arange(1, len(order) + 1)
    recall = cum_tp / len(ground_truth) if len(ground_truth) else np.zeros(len(order))
    return order, tp, precision, recall, int((~used).sum())


def smooth_l1(errors):
    absolute = np.abs(errors)
    return np.where(absolute < 1, 0.5 * errors**2, absolute - 0.5)


def main():
    np.testing.assert_allclose(pairwise_iou([[0, 0, 2, 2]], [[1, 1, 3, 3]]), [[1/7]])
    np.testing.assert_allclose(pairwise_iou([[0, 0, 2, 2]], [[0, 0, 2, 2], [2, 0, 4, 2]]), [[1, 0]])
    np.testing.assert_allclose(pairwise_iou([[0, 0, 0, 0]], [[0, 0, 0, 0]]), [[0]])
    assert pairwise_iou(np.empty((0, 4)), np.zeros((2, 4))).shape == (0, 2)
    boxes = np.array([[0.,0.,2.,2.], [.1,.1,2.1,2.1], [5.,5.,7.,7.]])
    scores = np.array([.9, .8, .7])
    labels = np.array([0, 0, 0])
    kept = class_aware_nms(boxes, scores, labels)
    np.testing.assert_array_equal(kept, [0, 2])
    np.testing.assert_array_equal(class_aware_nms(boxes, scores, np.array([0, 1, 0])), [0, 1, 2])
    gt = boxes[[0, 2]]
    order, tp, precision, recall, fn = evaluate_one_class(boxes, scores, gt)
    np.testing.assert_array_equal(tp, [1, 0, 1])
    np.testing.assert_allclose(precision, [1, .5, 2/3])
    np.testing.assert_allclose(recall, [.5, .5, 1])
    assert fn == 0
    filtered = evaluate_one_class(boxes[kept], scores[kept], gt)
    np.testing.assert_allclose(filtered[2][-1], 1)
    np.testing.assert_allclose(filtered[3][-1], 1)
    # Only positive candidates participate in this teaching localization loss.
    errors = np.array([[.2, -.4, 1.5, 0.], [100., 100., 100., 100.]])
    positive = np.array([True, False])
    box_loss = smooth_l1(errors[positive]).sum()
    np.testing.assert_allclose(box_loss, 1.1)
    print('IoU hand example:', 1/7)
    print('Duplicate overlap:', pairwise_iou(boxes[[0]], boxes[[1]])[0, 0])
    print('NMS retained indices:', kept)
    print('Before NMS TP flags / precision / recall / FN:', tp, precision, recall, fn)
    print('After NMS final precision / recall:', filtered[2][-1], filtered[3][-1])
    print('Positive-only summed Smooth L1 loss:', box_loss)
    print('All constructed checks passed; not a benchmark evaluation')


if __name__ == '__main__':
    main()

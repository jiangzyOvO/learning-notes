"""Lecture 1 supplement: pixel representation, layouts and a toy translation.

Run: python3 experiments/deep_review/lecture01_images.py
Requires NumPy. No image dataset or classifier is trained here.
"""
import numpy as np


def main():
    # HWC: height, width, color channels. A constructed 2x2 RGB image.
    image_hwc = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 255]],
    ], dtype=np.uint8)
    image_chw = image_hwc.transpose(2, 0, 1)
    wrong_chw = image_hwc.reshape(3, 2, 2)
    assert image_hwc.shape == (2, 2, 3)
    assert image_chw.shape == (3, 2, 2)
    assert np.array_equal(image_chw[0], [[255, 0], [0, 255]])
    assert not np.array_equal(wrong_chw, image_chw)
    assert np.array_equal(image_chw.transpose(1, 2, 0), image_hwc)
    print("HWC shape:", image_hwc.shape)
    print("Top-left pixel RGB:", image_hwc[0, 0].tolist())
    print("Correct red channel:", image_chw[0].tolist())
    print("Wrong red channel after reshape:", wrong_chw[0].tolist())

    # Repetition only illustrates the batch axis; it is not a training dataset.
    batch_chw = np.stack([image_chw, image_chw], axis=0)
    normalized = batch_chw.astype(np.float32) / 255.0
    flat = normalized.reshape(normalized.shape[0], -1)
    assert batch_chw.shape == (2, 3, 2, 2)
    assert flat.shape == (2, 12)
    assert normalized.dtype == np.float32
    assert np.array_equal(flat.reshape(normalized.shape), normalized)
    print("Batch NCHW:", batch_chw.shape)
    print("Normalized type and range:", normalized.dtype,
          float(normalized.min()), float(normalized.max()))
    print("Flattened per sample:", flat.shape)

    gray = np.zeros((4, 4), dtype=np.uint8)
    gray[1, 1:3] = 255
    shifted = np.zeros_like(gray)
    shifted[:, 1:] = gray[:, :-1]  # right by one pixel, zero-filled left edge
    difference = gray.astype(np.float32) - shifted.astype(np.float32)
    l1 = np.abs(difference).sum()
    assert float(l1) == 510.0
    print("Toy pattern before translation:", gray.tolist())
    print("Toy pattern after translation:", shifted.tolist())
    print("L1 pixel distance:", float(l1))
    print("All representation checks passed; no classifier was trained.")


if __name__ == "__main__":
    main()

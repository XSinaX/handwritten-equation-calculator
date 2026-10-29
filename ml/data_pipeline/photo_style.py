"""
Photo-style training images: training equations exactly as the API sees them
after a phone photo.

Each generated training equation (ink) is drawn as a simulated phone photo
(photo_simulation.py), JPEG-compressed, then run through the same
preprocessing the API applies to uploads (shared/image_preprocessing.py).
Training on these teaches the model what thresholded, cropped photos really
look like - broken or ragged strokes, uneven thickness, leftover specks -
instead of only clean renders.

Run from the project root, after equation_generator.py has written
ml/data/generated/train:

    python -m ml.data_pipeline.photo_style

Output: ml/data/photo_generated/train/X.npy, y.npy (the trainer adds it to
the training data automatically when it exists).
"""

import multiprocessing
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from ml.data_pipeline.photo_simulation import jpeg_round_trip, render_photo
from shared.config import IMAGE_HEIGHT, IMAGE_WIDTH
from shared.image_preprocessing import PreprocessingError, preprocess_gray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INK_FOLDER = PROJECT_ROOT / "ml" / "data" / "generated" / "train"
OUTPUT_FOLDER = PROJECT_ROOT / "ml" / "data" / "photo_generated" / "train"

SEED = 7


def read_ink(path):
    root = ET.parse(path).getroot()

    label = None
    strokes = []

    for element in root.iter():
        if element.tag.endswith("annotation") and element.attrib.get("type") == "normalizedLabel":
            label = element.text

        elif element.tag.endswith("trace") and element.text:
            points = [tuple(map(float, raw.split()[:2])) for raw in element.text.strip().split(",") if raw.strip()]

            if points:
                strokes.append(points)

    return label, strokes


def make_photo_style(job):
    """(index, path) -> (index, 128x512 image or None, label)."""

    index, path = job
    rng = np.random.default_rng([SEED, index])

    label, strokes = read_ink(path)

    if not label or not strokes:
        return index, None, label

    photo = render_photo(strokes, rng)
    encoded, _ = jpeg_round_trip(photo, rng)
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

    try:
        return index, preprocess_gray(gray), label
    except PreprocessingError:
        return index, None, label


def main():
    paths = sorted(INK_FOLDER.glob("*.inkml"))

    if not paths:
        print(f"No InkML files in {INK_FOLDER}. Run equation_generator.py first.")
        return

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    X = np.lib.format.open_memmap(
        OUTPUT_FOLDER / "X.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(paths), IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    labels = [None] * len(paths)
    kept = np.zeros(len(paths), dtype=bool)

    with multiprocessing.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for done, (index, image, label) in enumerate(
            pool.imap_unordered(make_photo_style, enumerate(paths), chunksize=64), start=1
        ):
            if image is not None:
                X[index] = image
                labels[index] = label
                kept[index] = True

            if done % 5000 == 0:
                print(f"  {done}/{len(paths)}", flush=True)

    X.flush()
    del X

    # Drop the few images preprocessing rejected
    if not kept.all():
        X = np.load(OUTPUT_FOLDER / "X.npy")[kept]
        np.save(OUTPUT_FOLDER / "X.npy", X)

    np.save(OUTPUT_FOLDER / "y.npy", np.array([label for label, keep in zip(labels, kept) if keep]))

    print(f"Saved {int(kept.sum())} photo-style images to {OUTPUT_FOLDER} ({int((~kept).sum())} rejected)")


if __name__ == "__main__":
    main()

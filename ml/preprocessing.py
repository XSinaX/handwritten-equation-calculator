import argparse
import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ml.config import IMAGE_HEIGHT, IMAGE_WIDTH, SYMBOL_WIDTH

EQUATION_PADDING = 5
DATASETS_ROOT = Path(__file__).resolve().parent / "datasets"
SPLITS = ("train", "validation", "test")
OUTPUT_FILENAME = "preprocessed.npz"
NUM_WORKERS = min(6, os.cpu_count() or 1)
WORK_BATCH_SIZE = 256
WORK_CHUNK_SIZE = 8
BACKGROUND_KERNEL_RATIO = 0.10
DARK_BORDER_LEVEL = 50
DARK_BORDER_MIN_AREA_RATIO = 0.005
TINY_WIDTH_RATIO = 0.30
TINY_HEIGHT_RATIO = 0.30
MAX_NOISE_DISTANCE_RATIO = 0.70
SYMBOL_PADDING = 5


def preprocess_symbol_image(image: np.ndarray) -> np.ndarray:
    """Center black ink on white, preserving aspect ratio and 5px minimum padding.

    Input: 2-D uint8 image with black ink (0) and white background (255).
    Output: 2-D uint8 image of shape (IMAGE_HEIGHT, SYMBOL_WIDTH).
    """
    if image.ndim != 2 or image.dtype != np.uint8 or not image.size:
        raise ValueError("Expected a non-empty 2-D uint8 symbol image")

    ys, xs = np.nonzero(image < 245)
    if not len(xs):
        raise ValueError("Symbol contains no dark ink pixels")

    crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    max_height = IMAGE_HEIGHT - 2 * SYMBOL_PADDING
    max_width = SYMBOL_WIDTH - 2 * SYMBOL_PADDING
    if max_height < 1 or max_width < 1:
        raise ValueError("Symbol dimensions are too small for the requested padding")

    height, width = crop.shape
    scale = min(max_height / height, max_width / width)
    scaled_height = max(1, min(max_height, round(height * scale)))
    scaled_width = max(1, min(max_width, round(width * scale)))
    resized = cv2.resize(
        crop,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    canvas = np.full((IMAGE_HEIGHT, SYMBOL_WIDTH), 255, dtype=np.uint8)
    top = (IMAGE_HEIGHT - scaled_height) // 2
    left = (SYMBOL_WIDTH - scaled_width) // 2
    canvas[top : top + scaled_height, left : left + scaled_width] = resized
    return canvas


def image_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale, converting transparent pixels to white."""
    if image.ndim == 2:
        return image.astype(np.uint8)

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected a grayscale, RGB, or RGBA image array.")

    if image.shape[2] == 4:
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    else:
        rgb = image.astype(np.uint8)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def remove_dark_borders(grayscale: np.ndarray) -> np.ndarray:
    """Paint over substantial near-black regions touching the image edges."""
    dark = (grayscale < DARK_BORDER_LEVEL).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    height, width = grayscale.shape
    border_ids = []

    for component_id in range(1, count):
        x, y, w, h, area = stats[component_id]
        touches_edge = x == 0 or y == 0 or x + w == width or y + h == height
        if touches_edge and area >= DARK_BORDER_MIN_AREA_RATIO * grayscale.size:
            border_ids.append(component_id)

    if not border_ids:
        return grayscale.copy()

    mask = np.isin(labels, border_ids).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8)) > 0
    paper_color = np.median(grayscale[~mask]) if (~mask).any() else 255

    cleaned = grayscale.copy()
    cleaned[mask] = paper_color
    return cleaned


def binarize_image(grayscale: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting with a morphological background and apply Otsu."""
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    kernel_size = max(15, int(min(grayscale.shape) * BACKGROUND_KERNEL_RATIO))
    kernel_size |= 1  # Morphology and blur use an odd-sized kernel.

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.dilate(blurred, kernel)
    background = cv2.blur(background, (kernel_size, kernel_size))

    normalized = cv2.divide(blurred, background, scale=255)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def remove_isolated_noise(binary: np.ndarray) -> np.ndarray:
    """Remove tiny, distant noise from the binary image."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((binary == 0).astype(np.uint8), connectivity=8)
    if count <= 2:
        return binary.copy()

    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    reference_width = float(np.percentile(widths, 75))
    reference_height = float(np.percentile(heights, 75))

    tiny = (widths < TINY_WIDTH_RATIO * reference_width) & (heights < TINY_HEIGHT_RATIO * reference_height)
    tiny_ids = np.flatnonzero(tiny) + 1
    anchor_ids = np.flatnonzero(~tiny) + 1
    if not len(tiny_ids) or not len(anchor_ids):
        return binary.copy()

    # For each pixel, measure distance to the nearest pixel in a large component.
    anchor_mask = np.ones(binary.shape, dtype=np.uint8)
    anchor_mask[np.isin(labels, anchor_ids)] = 0
    distances = cv2.distanceTransform(anchor_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    max_distance = max(2.0, MAX_NOISE_DISTANCE_RATIO * reference_height)
    cleaned = binary.copy()
    for component_id in tiny_ids:
        x, y, w, h, _ = stats[component_id]
        component = labels[y : y + h, x : x + w] == component_id
        closest = float(distances[y : y + h, x : x + w][component].min())
        if closest > max_distance:
            cleaned[y : y + h, x : x + w][component] = 255

    return cleaned


def crop_to_ink(binary: np.ndarray) -> np.ndarray:
    """Crop to the smallest rectangle containing all remaining black pixels."""
    points = cv2.findNonZero(255 - binary)
    if points is None:
        raise ValueError("No handwriting was found in the image.")

    x, y, width, height = cv2.boundingRect(points)
    return binary[y : y + height, x : x + width]


def resize_and_pad(binary: np.ndarray) -> np.ndarray:
    """Fit ink inside config width/height without distortion; centre on white.

    Reserve at least EQUATION_PADDING pixels on all four sides. Height
    determines the scale unless this would exceed the available width.
    """
    height, width = binary.shape
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty image")
    available_height = IMAGE_HEIGHT - 2 * EQUATION_PADDING
    available_width = IMAGE_WIDTH - 2 * EQUATION_PADDING
    if available_height <= 0 or available_width <= 0:
        raise ValueError("IMAGE_HEIGHT and IMAGE_WIDTH must exceed twice the padding")

    scale = min(available_height / height, available_width / width)
    new_height = max(1, min(available_height, round(height * scale)))
    new_width = max(1, min(available_width, round(width * scale)))
    resized = cv2.resize(
        binary,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    remaining_height = IMAGE_HEIGHT - new_height
    remaining_width = IMAGE_WIDTH - new_width
    top = remaining_height // 2
    bottom = remaining_height - top
    left = remaining_width // 2
    right = remaining_width - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Run the full preprocessing pipeline on a handwritten equation image."""
    grayscale = image_to_grayscale(image)
    grayscale = remove_dark_borders(grayscale)
    binary = binarize_image(grayscale)
    binary = remove_isolated_noise(binary)
    binary = crop_to_ink(binary)
    return resize_and_pad(binary)


def read_split_samples(split_folder: Path) -> list[tuple[Path, str, str]]:
    """Collect labelled PNG paths from each dataset folder's CSV."""
    samples = []
    for folder in sorted(split_folder.iterdir()):
        if not folder.is_dir():
            continue
        csv_path = folder / "labels.csv"
        if not csv_path.is_file():
            csv_files = sorted(folder.glob("*.csv"))
            if not csv_files:
                print(f"  Skipping {folder.name}: no CSV file")
                continue
            if len(csv_files) > 1:
                raise ValueError(f"Multiple CSV files found in {folder}; expected labels.csv")
            csv_path = csv_files[0]

        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"image", "equation"}.issubset(reader.fieldnames):
                raise ValueError(f"Expected image and equation columns in {csv_path}")
            for row in reader:
                filename = (row.get("image") or "").strip().replace("\\", "/")
                equation = (row.get("equation") or "").strip()
                if not filename:
                    continue
                samples.append((folder / filename, equation, (row.get("type") or "unknown").strip()))
    return samples


def configure_worker() -> None:
    """Avoid OpenCV starting additional CPU threads inside each worker."""
    cv2.setNumThreads(1)


def process_sample(sample: tuple[Path, str, str]) -> tuple[np.ndarray | None, str | None]:
    """Load and preprocess one image; return its pixels or a skip reason."""
    image_path, equation, _ = sample
    try:
        if not equation:
            raise ValueError("Empty equation label")
        with Image.open(image_path) as image:
            rgba = "A" in image.getbands() or "transparency" in image.info
            rgb = image.convert("RGBA" if rgba else "RGB")
            source_image = np.asarray(rgb, dtype=np.uint8)
        processed = preprocess_image(source_image)
        if processed.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"Unexpected output size {processed.shape}")
        return processed, None
    except (OSError, ValueError, cv2.error) as error:
        return None, str(error)


def preprocess_split(split: str, datasets_root: Path = DATASETS_ROOT, workers: int = NUM_WORKERS) -> tuple[int, int]:
    """Combine folders into one .npz, preprocessing images in parallel.

    Workers read and process independent PNGs. The main process alone writes
    the memory-mapped image array and collects matching labels in CSV order.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")

    split_folder = Path(datasets_root) / "equation" / split
    if not split_folder.is_dir():
        raise FileNotFoundError(f"Missing equation split folder: {split_folder}")
    samples = read_split_samples(split_folder)
    if not samples:
        raise ValueError(f"No labelled images found in {split_folder}")

    equations, paths, types = [], [], []
    skipped = 0
    print(f"{split}: preprocessing {len(samples):,} images with {workers} worker(s)")

    with tempfile.TemporaryDirectory(prefix="preprocess_", dir=split_folder) as temp:
        images = np.lib.format.open_memmap(
            Path(temp) / "images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(samples), IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=configure_worker) as pool:
                # Submit in bounded batches so completed images don't pile up in RAM.
                for start in range(0, len(samples), WORK_BATCH_SIZE):
                    batch = samples[start : start + WORK_BATCH_SIZE]
                    results = pool.map(process_sample, batch, chunksize=WORK_CHUNK_SIZE)

                    for (image_path, equation, sample_type), (processed, error) in zip(batch, results):
                        if error is not None:
                            skipped += 1
                            if skipped <= 5:
                                print(f"  Skipped {image_path}: {error}")
                            continue

                        index = len(equations)
                        images[index] = processed
                        equations.append(equation)
                        paths.append(image_path.relative_to(split_folder).as_posix())
                        types.append(sample_type)

            count = len(equations)
            if not count:
                raise ValueError(f"No valid images were processed for {split}")

            images.flush()
            output = split_folder / OUTPUT_FILENAME
            np.savez_compressed(
                output,
                images=images[:count],
                equations=np.asarray(equations, dtype=np.str_),
                sources=np.asarray(paths, dtype=np.str_),
                types=np.asarray(types, dtype=np.str_),
            )
        finally:
            # On Windows the temporary memmap must be released before cleanup.
            del images

    print(f"{split}: saved {count:,} images to {output} (skipped {skipped:,})")
    return count, skipped


def process_symbol_sample(sample):
    """Load and preprocess one rendered CROHME symbol."""
    path, label = sample

    try:
        if not label:
            raise ValueError("Empty symbol label")

        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8)

        processed = preprocess_symbol_image(grayscale)
        return processed, None

    except (OSError, ValueError, cv2.error) as error:
        return None, str(error)


def preprocess_symbol_split(split: str, workers: int = NUM_WORKERS):
    """Combine preprocessed symbols into one NPZ per split."""
    split_folder = DATASETS_ROOT / "symbol" / split
    png_folder = split_folder / "CROHME23" / "PNG"
    csv_path = png_folder / "labels.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing symbol labels: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"image", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected image,label columns in {csv_path}")

        samples = [(png_folder / row["image"], row["label"].strip()) for row in reader]

    if not samples:
        raise ValueError(f"No symbols found in {csv_path}")

    labels, sources = [], []
    skipped = 0

    with tempfile.TemporaryDirectory(dir=split_folder) as temp:
        images = np.lib.format.open_memmap(
            Path(temp) / "images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(samples), IMAGE_HEIGHT, SYMBOL_WIDTH),
        )

        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=configure_worker,
            ) as pool:
                for start in range(0, len(samples), WORK_BATCH_SIZE):
                    batch = samples[start : start + WORK_BATCH_SIZE]

                    results = pool.map(
                        process_symbol_sample,
                        batch,
                        chunksize=WORK_CHUNK_SIZE,
                    )

                    for (path, label), (processed, error) in zip(batch, results):
                        if error is not None:
                            skipped += 1
                            if skipped <= 5:
                                print(f"Skipped {path}: {error}")
                            continue

                        index = len(labels)
                        images[index] = processed
                        labels.append(label)
                        sources.append(path.relative_to(split_folder).as_posix())

            count = len(labels)
            if not count:
                raise ValueError(f"No valid symbols in {split}")

            images.flush()
            output = split_folder / OUTPUT_FILENAME

            np.savez_compressed(
                output,
                images=images[:count],
                labels=np.asarray(labels, dtype=np.str_),
                sources=np.asarray(sources, dtype=np.str_),
            )

        finally:
            del images

    print(f"Symbol {split}: saved {count:,} images to {output} (skipped {skipped:,})")


def main() -> None:
    """Preprocess each split into one NumPy archive using multiple workers."""
    parser = argparse.ArgumentParser(description="Preprocess equation images in parallel")
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS, help=f"Number of image-processing workers (default: {NUM_WORKERS})"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    for split in SPLITS:
        preprocess_split(split, workers=args.workers)
        preprocess_symbol_split(split, workers=args.workers)


if __name__ == "__main__":
    main()

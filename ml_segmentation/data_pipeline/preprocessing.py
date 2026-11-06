"""Preprocess synthetic equation photographs AND their pixel-level annotations.

This module contains the preprocessing rules locally inside ``ml_segmentation``.
It does not import or execute preprocessing code from outside this experimental folder.

Input (one sample):
  Synthetic/<image>.png, <image>_instances.png, <image>_noise.png,
  <image>_symbols.json

Output (one split):
  ml_segmentation/datasets/equation/<split>/preprocessed.npz

NPZ arrays (no pickle): images (uint8), instance_masks (uint16), and
noise_masks (uint8) are flat, concatenated pixels. pixel_offsets (int64,
N+1 entries) and canvas_widths (int32, N entries) locate each 128-high image.
equations, sources, types, and symbol_annotations are N Unicode strings;
symbol_annotations contains a JSON list of symbol instance annotations.
sample_indices (int32) records the original labels.csv row for each sample.
Variable widths and all original pixel coordinates are preserved.

The binarization and ink/noise cleanup are unchanged. Height-only scaling
removes the old width-limited downscaling. Symbol and true-noise ownership are
corrected using the known binarization Gaussian, then scaled by weighted
one-hot ownership votes using the SAME interpolation as the image.
Any surviving black foreground not attributable to a symbol is possible noise.
After preprocessing all black pixels belong to a real instance or noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import multiprocessing as mp
from collections import Counter
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Local copies of the equation preprocessing rules used by this experimental
# pipeline. Nothing in ``ml_segmentation`` imports or executes code from the production
# ``ml`` package.
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1024
EQUATION_PADDING = 5
BACKGROUND_KERNEL_RATIO = 0.10
DARK_BORDER_LEVEL = 50
DARK_BORDER_MIN_AREA_RATIO = 0.005
TINY_WIDTH_RATIO = 0.30
TINY_HEIGHT_RATIO = 0.30
MAX_NOISE_DISTANCE_RATIO = 0.70


def image_to_grayscale(image: np.ndarray) -> np.ndarray:
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
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    kernel_size = max(15, int(min(grayscale.shape) * BACKGROUND_KERNEL_RATIO))
    kernel_size |= 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.dilate(blurred, kernel)
    background = cv2.blur(background, (kernel_size, kernel_size))
    normalized = cv2.divide(blurred, background, scale=255)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def remove_isolated_noise(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary == 0).astype(np.uint8), connectivity=8
    )
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
    points = cv2.findNonZero(255 - binary)
    if points is None:
        raise ValueError("No handwriting was found in the image.")
    x, y, width, height = cv2.boundingRect(points)
    return binary[y : y + height, x : x + width]


def _resize_geometry(height: int, width: int, resize_mode: str = "height_only") -> tuple[int, int, int, int, int, float]:
    """One *uniform* scale for both axes; optionally drop the width cap.

    Returns (new_height, new_width, canvas_width, top, left, scale).
    In height_only mode the canvas grows horizontally for long equations.
    Long equations are NEVER squeezed, clipped or dropped to fit IMAGE_WIDTH.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty image")
    available_height = IMAGE_HEIGHT - 2 * EQUATION_PADDING
    if resize_mode == "height_only":
        scale = available_height / height
        new_height = available_height
        new_width = max(1, round(width * scale))
        canvas_width = max(IMAGE_WIDTH, new_width + 2 * EQUATION_PADDING)
    elif resize_mode == "legacy_fit":
        available_width = IMAGE_WIDTH - 2 * EQUATION_PADDING
        scale = min(available_height / height, available_width / width)
        new_height = max(1, min(available_height, round(height * scale)))
        new_width = max(1, min(available_width, round(width * scale)))
        canvas_width = IMAGE_WIDTH
    else:
        raise ValueError(f"Unknown resize mode: {resize_mode}")
    top = (IMAGE_HEIGHT - new_height) // 2
    left = (canvas_width - new_width) // 2
    return new_height, new_width, canvas_width, top, left, scale


def resize_and_pad(binary: np.ndarray, resize_mode: str = "height_only") -> np.ndarray:
    """Aspect-preserving scale and white pad; height-only can exceed 1024 px."""
    height, width = binary.shape
    new_h, new_w, canvas_w, top, left, scale = _resize_geometry(height, width, resize_mode)
    resized = cv2.resize(binary, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return cv2.copyMakeBorder(
        resized, top, IMAGE_HEIGHT - new_h - top,
        left, canvas_w - new_w - left, cv2.BORDER_CONSTANT, value=255
    )


SPLITS = ("train", "validation", "test")
DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "equation"
OVERLAP_ID = 65535
NUM_WORKERS = min(6, os.cpu_count() or 1)
WORK_CHUNK_SIZE = 4


def _bbox(mask: np.ndarray) -> list[int] | None:
    """Return half-open [x1, y1, x2, y2], or None when invisible."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _align_gaussian_ownership(
    final_binary: np.ndarray,
    instance_mask: np.ndarray,
    true_noise_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Attribute binarization-created foreground using the actual 3x3 Gaussian.

    The renderer's instance map already carries its photographic Gaussian
    contribution. Binarization itself uses an additional 3x3 Gaussian. For
    black pixels outside the carried ownership, blur each instance's source
    support with that same 3x3 kernel. Competing owners use the largest
    Gaussian weight (higher instance ID wins exact ties). The protected
    injected-noise seeds and their Gaussian response also compete; protected
    seeds ALWAYS remain noise.

    A zero owner on a black pixel is *defined* as possible noise, never an
    annotation gap. Actual output noise is finalized after resizing.
    """
    foreground = final_binary == 0
    if np.any(instance_mask == OVERLAP_ID):
        raise ValueError(
            "Legacy 65535 overlap ID found. Regenerate synthetic images using "
            "the new renderer; a mixed pixel cannot be attributed reliably."
        )
    labels = instance_mask.copy()
    labels[~foreground] = 0
    protected = (true_noise_mask > 0) & foreground
    labels[protected] = 0
    candidates = foreground & (labels == 0) & (~protected)
    candidates_before = int(np.count_nonzero(candidates))

    # GaussianBlur(mask, (3,3), 0) is exactly the 3x3 Gaussian operation
    # present at the start of binarize_image(). Keep this source operation
    # independent of output threshold, so the vote adapts to stroke shape.
    max_weight = np.zeros(final_binary.shape, dtype=np.float32)
    inferred = np.zeros(final_binary.shape, dtype=np.uint16)
    for ident in np.unique(instance_mask):
        ident = int(ident)
        if ident == 0:
            continue
        support = (instance_mask == ident).astype(np.float32)
        weight = cv2.GaussianBlur(support, (3, 3), sigmaX=0)
        better = candidates & (weight > 0) & (weight >= max_weight)
        inferred[better] = ident
        max_weight[better] = weight[better]

    # Original true-noise seeds are protected. Their 3x3 Gaussian halo
    # competes just as symbols do; a strong noise contribution can defeat a
    # weaker neighbouring symbol halo, without forcing all nearby ink to noise.
    noise_response = cv2.GaussianBlur((true_noise_mask > 0).astype(np.float32),
                                       (3, 3), sigmaX=0)
    noise_wins = candidates & (noise_response > max_weight)
    inferred[noise_wins] = 0
    labels[candidates] = inferred[candidates]
    labels[protected] = 0
    labels[~foreground] = 0
    return labels, protected.astype(np.uint8), {
        "unowned_before_binarization_correction": candidates_before,
        "pixels_reassigned_by_binarization_gaussian": int(np.count_nonzero(
            candidates & (labels > 0)
        )),
        "possible_noise_after_binarization": int(np.count_nonzero(
            foreground & (labels == 0) & (~protected)
        )),
    }


def _resize_ownership_votes(
    binary_crop: np.ndarray,
    instance_crop: np.ndarray,
    true_noise_crop: np.ndarray,
    resize_mode: str = "height_only",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize per-owner float coverage using image's exact mapping/filter.

    Each source-black pixel casts its vote into a one-hot ownership layer.
    Noise is also a vote; explicitly injected noise is tracked separately for
    diagnostics. The winner for every FINAL black pixel is determined by
    maximum interpolated contribution, with later symbol ID breaking symbol
    ties. No nearest-neighbour label sampling or iterative flood-fill.
    """
    height, width = binary_crop.shape
    new_height, new_width, canvas_width, top, left, scale = _resize_geometry(
        height, width, resize_mode,
    )
    dsize = (new_width, new_height)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    foreground = binary_crop == 0

    # Keep the binarization/resize path untouched. Work with float one-hot
    # layers and the exact same size/interpolation; never resize integer IDs.
    resized_black = cv2.resize(
        binary_crop, dsize, interpolation=interpolation,
    )
    final_black = resized_black < 128
    true_pixels = foreground & (true_noise_crop > 0)
    possible_pixels = foreground & (instance_crop == 0) & (~true_pixels)
    noise_support = foreground & (instance_crop == 0)

    def resize_vote(layer: np.ndarray) -> np.ndarray:
        return cv2.resize(layer.astype(np.float32), dsize, interpolation=interpolation)

    strongest = resize_vote(noise_support)
    winners = np.zeros((new_height, new_width), dtype=np.uint16)
    true_vote = resize_vote(true_pixels)
    possible_vote = resize_vote(possible_pixels)
    for ident in np.unique(instance_crop):
        ident = int(ident)
        if ident == 0:
            continue
        weight = resize_vote(foreground & (instance_crop == ident))
        better = (weight > 0) & (weight >= strongest)
        strongest[better] = weight[better]
        winners[better] = ident
    winners[~final_black] = 0
    noise = np.where(final_black & (winners == 0), 255, 0).astype(np.uint8)
    true_output = (final_black & (winners == 0) &
                   (true_vote >= possible_vote) & (true_vote > 0))
    true_output = true_output.astype(np.uint8) * 255

    instance_canvas = np.zeros((IMAGE_HEIGHT, canvas_width), dtype=np.uint16)
    noise_canvas = np.zeros((IMAGE_HEIGHT, canvas_width), dtype=np.uint8)
    true_canvas = np.zeros((IMAGE_HEIGHT, canvas_width), dtype=np.uint8)
    instance_canvas[top:top+new_height, left:left+new_width] = winners
    noise_canvas[top:top+new_height, left:left+new_width] = noise
    true_canvas[top:top+new_height, left:left+new_width] = true_output
    return instance_canvas, noise_canvas, true_canvas


def preprocess_image_and_masks(
    image: np.ndarray,
    instance_mask: np.ndarray,
    noise_mask: np.ndarray,
    true_noise_mask: np.ndarray | None = None,
    resize_mode: str = "height_only",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Keep original image preprocessing; repair and scale ownership jointly.

    Masks use 0=not a symbol and 1..N=instances. For any FINAL black pixel
    with no symbol owner, noise_mask=255. The separate protected true-noise
    layer is only used during correction and discarded in final NPZ output.
    """
    if image.ndim not in (2, 3) or image.dtype != np.uint8:
        raise ValueError("Expected uint8 grayscale or RGB/RGBA equation image")
    h, w = image.shape[:2]
    if (instance_mask.shape != (h, w) or instance_mask.ndim != 2
            or instance_mask.dtype != np.uint16):
        raise ValueError("Instance mask must be HxW uint16 and match the source image")
    if noise_mask.shape != (h, w) or noise_mask.ndim != 2 or noise_mask.dtype != np.uint8:
        raise ValueError("Noise mask must be HxW uint8 and match the source image")
    if true_noise_mask is None:
        # Backward-compatible fallback for legacy *injected-noise-only* masks.
        # Proper reprocessing should use new source files with _true_noise.png.
        true_noise_mask = noise_mask
    if true_noise_mask.shape != (h, w) or true_noise_mask.dtype != np.uint8:
        raise ValueError("True-noise mask must be matching HxW uint8")

    gray = image_to_grayscale(image)
    border_clean = remove_dark_borders(gray)
    thresholded = binarize_image(border_clean)
    cleaned = remove_isolated_noise(thresholded)
    ys, xs = np.where(cleaned == 0)
    if not xs.size:
        raise ValueError("No handwriting found after preprocessing")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    cropped = crop_to_ink(cleaned)
    assert cropped.shape == (y2 - y1, x2 - x1)
    processed = resize_and_pad(cropped, resize_mode=resize_mode)
    assert processed.shape[0] == IMAGE_HEIGHT

    known = instance_mask > 0
    symbol_removed_by_binarization = int(np.count_nonzero(known & (thresholded != 0)))
    symbol_removed_by_noise_cleanup = int(np.count_nonzero(
        known & (thresholded == 0) & (cleaned != 0)
    ))
    noise_removed_by_cleanup = int(np.count_nonzero(
        (true_noise_mask > 0) & (thresholded == 0) & (cleaned != 0)
    ))

    aligned_instances, protected_true, correction_stats = _align_gaussian_ownership(
        cleaned, instance_mask, true_noise_mask,
    )
    result_instances, result_noise, result_true = _resize_ownership_votes(
        cropped,
        aligned_instances[y1:y2, x1:x2],
        protected_true[y1:y2, x1:x2],
        resize_mode=resize_mode,
    )
    # image geometry must match even if image library's rounded threshold
    # differs from the float branch at exactly 127.5.
    result_instances[processed != 0] = 0
    result_noise[processed != 0] = 0
    result_true[processed != 0] = 0
    # Exhaustively classify any remaining black output pixels as noise.
    # This catches rare numerical interpolation ties / JPEG artifacts.
    remaining = (processed == 0) & (result_instances == 0)
    result_noise[remaining] = 255
    result_noise[result_instances != 0] = 0
    result_true[result_noise == 0] = 0

    unassigned = (processed == 0) & (result_instances == 0) & (result_noise == 0)
    overlaps = (result_instances != 0) & (result_noise != 0)
    if np.any(unassigned) or np.any(overlaps) or np.any(result_instances == OVERLAP_ID):
        raise AssertionError("Final mask ownership invariant failed")
    stats = {
        "input_labelled_pixels": int(np.count_nonzero(known)),
        "removed_symbol_pixels_after_border_and_binarization": symbol_removed_by_binarization,
        "removed_symbol_pixels_at_noise_cleanup": symbol_removed_by_noise_cleanup,
        "removed_injected_noise_pixels_at_cleanup": noise_removed_by_cleanup,
        **correction_stats,
        "final_labelled_symbol_pixels": int(np.count_nonzero(result_instances)),
        "final_injected_noise_pixels": int(np.count_nonzero(result_true)),
        "final_noise_pixels": int(np.count_nonzero(result_noise)),
        "final_possible_noise_pixels": int(np.count_nonzero(result_noise & (result_true == 0))),
        "final_unassigned_black_pixels": int(np.count_nonzero(unassigned)),
        "final_overlap_pixels": 0,
        "source_crop_xyxy": [x1, y1, x2, y2],
        "processed_canvas_width": int(processed.shape[1]),
    }
    return processed, result_instances, result_noise, stats


def _read_sample(sample: tuple[str, str, str, str]) -> tuple[dict | None, str | None]:
    """Top-level, spawn-safe worker. Never writes shared output files."""
    path_string, equation, kind, resize_mode = sample
    path = Path(path_string)
    stem = path.with_suffix("")
    instance_path = stem.with_name(stem.name + "_instances.png")
    noise_path = stem.with_name(stem.name + "_noise.png")
    annotation_path = stem.with_name(stem.name + "_symbols.json")
    try:
        if not equation:
            raise ValueError("Empty equation label")
        # Load RGB/RGBA before applying the local preprocessing path.
        with Image.open(path) as im:
            rgba = "A" in im.getbands() or "transparency" in im.info
            source = np.asarray(im.convert("RGBA" if rgba else "RGB"), dtype=np.uint8)
        instance_mask = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
        noise_mask = cv2.imread(str(noise_path), cv2.IMREAD_UNCHANGED)
        if instance_mask is None or noise_mask is None:
            raise ValueError("Missing instance/noise mask")
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        symbols = annotation.get("symbols", [])
        if not isinstance(symbols, list) or not symbols:
            raise ValueError("No symbol annotations")
        expected_ids = list(range(1, len(symbols) + 1))
        ids = [int(symbol["instance_id"]) for symbol in symbols]
        if ids != expected_ids:
            raise ValueError("Symbol instance IDs are not sequential 1..N")
        if "".join(symbol["token"] for symbol in symbols) != equation:
            raise ValueError("Symbol annotation sequence differs from labels.csv")

        true_noise_path = stem.with_name(stem.name + "_true_noise.png")
        true_noise = cv2.imread(str(true_noise_path), cv2.IMREAD_UNCHANGED) if true_noise_path.is_file() else None
        image, instance, noise, stats = preprocess_image_and_masks(
            source, instance_mask, noise_mask, true_noise,
            resize_mode=resize_mode,
        )
        rendered = []
        vanished = 0
        for symbol in symbols:
            ident = int(symbol["instance_id"])
            visible = instance == ident
            bbox = _bbox(visible)
            vanished += bbox is None
            rendered.append({
                "instance_id": ident,
                "token": symbol["token"],
                "bbox_xyxy": bbox,
                "visible_pixels": int(np.count_nonzero(visible)),
                "source_bbox_xyxy": symbol.get("bbox_xyxy"),
            })
        stats["vanished_symbols"] = vanished
        return {
            "image": image,
            "instance_mask": instance,
            "noise_mask": noise,
            "annotations": json.dumps(rendered, ensure_ascii=False, separators=(",", ":")),
            "stats": stats,
            "equation": equation,
            "kind": kind,
            "source": path.name,
        }, None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, cv2.error) as exc:
        return None, str(exc)


def configure_worker() -> None:
    cv2.setNumThreads(1)


def _samples(split: str, root: Path) -> list[tuple[str, str, str]]:
    folder = root / split / "Synthetic"
    label_file = folder / "labels.csv"
    if not label_file.is_file():
        raise FileNotFoundError(f"Missing {label_file}; run equation_generator first")
    with label_file.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"image", "equation"}.issubset(reader.fieldnames):
            raise ValueError(f"Missing image/equation columns in {label_file}")
        samples = []
        for row in reader:
            filename = (row.get("image") or "").strip()
            if not filename or Path(filename).name != filename:
                raise ValueError(f"Invalid image filename in {label_file}: {filename}")
            samples.append((str(folder / filename), (row.get("equation") or "").strip(), (row.get("type") or "synthetic").strip()))
    if not samples:
        raise ValueError(f"No samples listed in {label_file}")
    return samples



def _save_preview(path: Path, image: np.ndarray, instances: np.ndarray, noise: np.ndarray) -> None:
    """Write a compact three-panel visual alignment check (no source changes)."""
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labelled = np.full(base.shape, 255, dtype=np.uint8)
    noise_pixels = noise > 0
    labelled[noise_pixels] = (0, 145, 255)  # orange: injected specks
    for ident in np.unique(instances):
        ident = int(ident)
        if ident == 0:
            continue
        if ident == OVERLAP_ID:
            colour = (225, 0, 225)  # magenta: conflicting symbol instances
        else:
            colour = (
                40 + (ident * 67) % 185,
                40 + (ident * 139) % 185,
                40 + (ident * 91) % 185,
            )
        labelled[instances == ident] = colour
    overlay = cv2.addWeighted(base, 0.3, labelled, 0.7, 0)
    overlay[(instances == 0) & (~noise_pixels)] = base[(instances == 0) & (~noise_pixels)]
    output = np.concatenate((base, labelled, overlay), axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"Could not save visual preview: {path}")


def stream_samples(npz: Path, indices: set[int] | None = None, limit: int = 0):
    """Read the single NPZ directly, yielding one equation at a time.

    Large pixel arrays are decompressed sequentially to bound memory usage.
    Indices count saved equations; limit counts yielded equations.
    """
    npz = Path(npz)
    if npz.is_dir():
        npz = npz / "preprocessed.npz"
    if indices is not None and not indices:
        return
    with np.load(npz, allow_pickle=False) as data:
        offsets = data["pixel_offsets"]
        widths = data["canvas_widths"]
        equations = data["equations"]
        annotations = data["symbol_annotations"]
        sources = data["sources"]
    with ZipFile(npz) as archive, ExitStack() as stack:
        streams = []
        for name, dtype in (("images", np.uint8), ("instance_masks", np.uint16),
                            ("noise_masks", np.uint8)):
            member = stack.enter_context(archive.open(name + ".npy"))
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, fortran, stored_dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, fortran, stored_dtype = np.lib.format.read_array_header_2_0(member)
            else:
                raise ValueError(f"Unsupported NPY header version: {version}")
            if fortran or stored_dtype != np.dtype(dtype) or shape != (int(offsets[-1]),):
                raise ValueError(f"Invalid pixel array {name} in {npz}")
            streams.append((member, stored_dtype))
        yielded = 0
        for idx, width in enumerate(widths):
            count = int(offsets[idx + 1] - offsets[idx])
            if count != IMAGE_HEIGHT * int(width):
                raise ValueError(f"Invalid image dimensions at sample {idx}")
            arrays = []
            for member, dtype in streams:
                raw = member.read(count * dtype.itemsize)
                arrays.append(np.frombuffer(raw, dtype=dtype).reshape(IMAGE_HEIGHT, int(width)).copy())
            if indices is not None and idx not in indices:
                continue
            yield (idx, *arrays, str(equations[idx]), json.loads(str(annotations[idx])), str(sources[idx]))
            yielded += 1
            if limit and yielded >= limit:
                break


def preprocess_split(
    split: str, workers: int = NUM_WORKERS, root: Path = DATASETS_ROOT,
    preview: int = 3, resize_mode: str = "height_only",
) -> tuple[int, int]:
    """Create one compressed NPZ; publish it only after a successful reload.

    Temporary raw pixel files keep RAM bounded even for the 30,000-equation
    training split. Existing source images and processed shards are untouched.
    """
    root = Path(root).resolve()
    if not root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Dataset root must stay inside ml_segmentation/")
    if split not in SPLITS or workers < 1 or preview < 0:
        raise ValueError("Invalid split, workers, or preview count")
    samples = _samples(split, root)
    split_dir = root / split
    output = split_dir / "preprocessed.npz"
    if output.exists():
        raise FileExistsError(f"Preserving existing dataset: {output}. Use a different --root to regenerate.")
    metadata = {name: [] for name in ("equations", "sources", "types", "symbol_annotations",
                                     "sample_indices", "canvas_widths")}
    offsets = [0]
    totals, skip_reasons = Counter(), Counter()
    pixel_types = {"images": np.uint8, "instance_masks": np.uint16, "noise_masks": np.uint8}
    print(f"{split}: preprocessing {len(samples):,} images into {output.name}", flush=True)
    with TemporaryDirectory(prefix=".preprocessing_", dir=split_dir) as temporary:
        temporary = Path(temporary)
        with ExitStack() as stack:
            files = {name: stack.enter_context((temporary / name).open("wb")) for name in pixel_types}
            executor = stack.enter_context(ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("spawn"), initializer=configure_worker))
            tasks = [(*sample, resize_mode) for sample in samples]
            for idx, (data, error) in enumerate(executor.map(_read_sample, tasks, chunksize=WORK_CHUNK_SIZE)):
                if error:
                    skip_reasons[error] += 1
                    if sum(skip_reasons.values()) <= 5:
                        print(f"  Skipped {Path(samples[idx][0]).name}: {error}", flush=True)
                    continue
                for name, key in (("images", "image"), ("instance_masks", "instance_mask"),
                                  ("noise_masks", "noise_mask")):
                    files[name].write(data[key].tobytes())
                for name, key in (("equations", "equation"), ("types", "kind"),
                                  ("symbol_annotations", "annotations")):
                    metadata[name].append(data[key])
                metadata["sources"].append(f"Synthetic/{data['source']}")
                metadata["sample_indices"].append(idx)
                metadata["canvas_widths"].append(data["image"].shape[1])
                offsets.append(offsets[-1] + data["image"].size)
                totals.update({k: v for k, v in data["stats"].items() if isinstance(v, int)})
                if idx < preview:
                    _save_preview(temporary / f"preview_{idx}.png", data["image"], data["instance_mask"], data["noise_mask"])
                if (idx + 1) % 500 == 0:
                    print(f"  {split}: {idx + 1:,}/{len(samples):,} scanned", flush=True)
        saved = len(metadata["equations"])
        if not saved:
            raise ValueError(f"No usable preprocessed equations in {split}")
        payload = {name: np.asarray(values, dtype=np.int32 if name in ("sample_indices", "canvas_widths") else np.str_)
                   for name, values in metadata.items()}
        payload["pixel_offsets"] = np.asarray(offsets, dtype=np.int64)
        mapped = {name: np.memmap(temporary / name, mode="r", dtype=dtype, shape=(offsets[-1],))
                  for name, dtype in pixel_types.items()}
        archive = temporary / "preprocessed.npz"
        try:
            np.savez_compressed(archive, **mapped, **payload)
        finally:
            for array in mapped.values():
                array._mmap.close()
        reloaded = sum(1 for _ in stream_samples(archive))
        if reloaded != saved:
            raise ValueError("Preprocessed NPZ failed its reload check")
        os.replace(archive, output)
        preview_dir = split_dir / "preprocessing_previews_npz"
        for path in temporary.glob("preview_*.png"):
            preview_dir.mkdir(exist_ok=True)
            os.replace(path, preview_dir / path.name)
    report = {"split": split, "total_in_csv": len(samples), "saved": saved,
              "skipped": len(samples) - saved, "skip_reasons": dict(skip_reasons),
              "totals": dict(totals), "resize_mode": resize_mode, "output": output.name}
    (split_dir / "preprocessing_npz_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{split}: saved and reloaded {saved:,} equations -> {output}", flush=True)
    return saved, len(samples) - saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess rendered equations into one compressed NPZ per split")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--preview", type=int, default=3)
    parser.add_argument("--resize-mode", choices=("height_only", "legacy_fit"), default="height_only")
    parser.add_argument("--root", type=Path, default=DATASETS_ROOT)
    args = parser.parse_args()
    for split in SPLITS if args.split == "all" else (args.split,):
        preprocess_split(split, workers=args.workers, preview=args.preview,
                         root=args.root, resize_mode=args.resize_mode)


if __name__ == "__main__":
    main()

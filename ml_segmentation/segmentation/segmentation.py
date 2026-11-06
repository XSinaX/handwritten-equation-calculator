from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Segment:
    image: np.ndarray
    bbox: tuple[int, int, int, int]


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected a 2-D uint8 image")
    if not np.all((image == 0) | (image == 255)):
        raise ValueError("Expected a binary image containing only 0 and 255")
    return image == 0


def connected_component_segmentation(image: np.ndarray) -> list[Segment]:
    """Return every 8-connected black-ink component without filtering."""
    foreground = _foreground_mask(image)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    segments: list[Segment] = []
    for component_id in range(1, count):
        x, y, width, height, _area = stats[component_id]
        x, y, width, height = map(int, (x, y, width, height))
        component_pixels = labels[y : y + height, x : x + width] == component_id
        crop = np.where(component_pixels, 0, 255).astype(np.uint8)
        segments.append(Segment(crop, (x, y, x + width, y + height)))
    return sorted(segments, key=lambda segment: (segment.bbox[0], segment.bbox[1]))


METHODS = ("baseline", "skeleton", "stroke_width")


def _skeleton_zhang_suen(binary_ink: np.ndarray) -> np.ndarray:
    img = binary_ink.astype(np.uint8).copy()
    h, w = img.shape
    if h < 3 or w < 3:
        return img.astype(bool)
    for _ in range(128):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            N = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            A = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            for a, b in ((p3, p4), (p4, p5), (p5, p6), (p6, p7), (p7, p8), (p8, p9), (p9, p2)):
                A += ((a == 0) & (b == 1)).astype(np.uint8)
            base = (img == 1) & (N >= 2) & (N <= 6) & (A == 1)
            if step == 0:
                mask = base & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                mask = base & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if np.any(mask):
                img[mask] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def _crop_part(foreground: np.ndarray, origin_x: int, origin_y: int) -> Segment:
    ys, xs = np.nonzero(foreground)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pixels = np.where(foreground[y1:y2, x1:x2], 0, 255).astype(np.uint8)
    return Segment(pixels, (origin_x + x1, origin_y + y1, origin_x + x2, origin_y + y2))


def _weak_seam(
    segment: Segment, method: str, median_component_width: float, max_relative_ink: float, min_side_fraction: float
):
    """Return a vertical split coordinate + diagnostic, or None."""
    ink = segment.image == 0
    h, w = ink.shape
    ink_count = int(ink.sum())
    if w < max(14, round(1.25 * median_component_width)) or h < 5 or ink_count < 18:
        return None
    lo = max(3, int(round(w * 0.20)))
    hi = min(w - 3, int(round(w * 0.80)))
    if hi <= lo:
        return None

    binary = ink.astype(np.uint8)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    typical_stroke = max(1.0, 2.0 * float(np.median(distance[ink])))
    occupancy = ink.sum(axis=0).astype(np.float64)
    cost = ((2.0 * distance) ** 2 * ink).sum(axis=0)
    positive_cost = cost[cost > 0]
    if len(positive_cost) < 3:
        return None
    typical_cost = max(1.0, float(np.median(positive_cost)))
    skeleton = _skeleton_zhang_suen(binary) if method == "skeleton" else None
    skel_count = skeleton.sum(axis=0) if skeleton is not None else None

    prefix = np.cumsum(occupancy)
    candidates = []
    for x in range(lo, hi + 1):
        if occupancy[x] == 0:
            continue
        left_area = float(prefix[x - 1])
        right_area = float(ink_count - left_area)
        if min(left_area, right_area) < min_side_fraction * ink_count:
            continue
        left_width = x
        right_width = w - x
        if min(left_width, right_width) < max(3.0, 0.26 * median_component_width):
            continue

        relative_cost = float(cost[x] / typical_cost)
        relative_ink = float(occupancy[x] / max(1.0, typical_stroke))
        if method == "skeleton":
            if skel_count[x] != 1 or relative_ink > max(2.5, 4.0 * max_relative_ink):
                continue
            if relative_cost > max(0.65, max_relative_ink):
                continue
            quality = relative_cost + 0.12 * relative_ink
        else:
            if relative_cost > max_relative_ink:
                continue
            if relative_ink > 2.25:
                continue
            quality = relative_cost

        quality += 0.06 * abs((x / w) - 0.5)
        candidates.append((quality, x, relative_cost, relative_ink))

    if not candidates:
        return None
    _, x, cost_ratio, ink_ratio = min(candidates)
    return x, {"local_cost_ratio": float(cost_ratio), "local_ink_ratio": float(ink_ratio)}


def _split_components(
    image: np.ndarray,
    method: str,
    max_relative_ink: float = 0.55,
    min_side_fraction: float = 0.18,
    max_cuts_per_component: int = 1,
):
    """Run original CCL, then optionally split weak bridges into MLP candidates."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if not 0 < max_relative_ink <= 2 or not 0 < min_side_fraction < 0.5:
        raise ValueError("Invalid bridge sensitivity")
    if max_cuts_per_component < 1:
        raise ValueError("max_cuts_per_component must be positive")
    parts = connected_component_segmentation(image)
    original_n = len(parts)
    if method == "baseline" or not parts:
        return parts, {
            "original_components": original_n,
            "components": original_n,
            "proposed_cuts": 0,
            "attempted_cuts": 0,
            "successful_splits": 0,
            "cuts": [],
        }

    reference_widths = [
        p.bbox[2] - p.bbox[0] for p in parts if (p.bbox[3] - p.bbox[1]) >= 6 and np.count_nonzero(p.image == 0) >= 10
    ]
    typical_width = max(4.0, float(np.median(reference_widths))) if reference_widths else 12.0
    output = []
    cuts = []
    attempted_cuts = 0
    for part in parts:
        pending = [(part, 0)]
        while pending:
            current, depth = pending.pop(0)
            cut = (
                _weak_seam(current, method, typical_width, max_relative_ink, min_side_fraction)
                if depth < max_cuts_per_component
                else None
            )
            if cut is None:
                output.append(current)
                continue
            attempted_cuts += 1
            split_x, info = cut
            fg = current.image == 0
            x0, y0, _, _ = current.bbox
            left = np.zeros_like(fg)
            left[:, :split_x] = fg[:, :split_x]
            right = np.zeros_like(fg)
            right[:, split_x:] = fg[:, split_x:]
            if not np.any(left) or not np.any(right):
                output.append(current)
                continue
            left_part = _crop_part(left, x0, y0)
            right_part = _crop_part(right, x0, y0)
            if np.count_nonzero(left_part.image == 0) + np.count_nonzero(right_part.image == 0) != int(fg.sum()):
                raise AssertionError("Proposed split unexpectedly lost ink")
            cuts.append({"bbox": current.bbox, "cut_x": int(x0 + split_x), **info})
            pending.extend([(left_part, depth + 1), (right_part, depth + 1)])
    output.sort(key=lambda part: (part.bbox[0], part.bbox[1]))
    return output, {
        "original_components": original_n,
        "components": len(output),
        "proposed_cuts": len(cuts),
        "attempted_cuts": attempted_cuts,
        "successful_splits": len(cuts),
        "cuts": cuts,
    }


def baseline_segmentation(
    image: np.ndarray, max_relative_ink: float = 0.55, min_side_fraction: float = 0.18, max_cuts_per_component: int = 1
):
    """Plain CCL with the same (candidates, diagnostics) interface as splitters."""
    return _split_components(image, "baseline", max_relative_ink, min_side_fraction, max_cuts_per_component)


def skeleton_segmentation(
    image: np.ndarray, max_relative_ink: float = 0.55, min_side_fraction: float = 0.18, max_cuts_per_component: int = 1
):
    """Split narrow medial-path crossings without removing any ink."""
    return _split_components(image, "skeleton", max_relative_ink, min_side_fraction, max_cuts_per_component)


def stroke_width_segmentation(
    image: np.ndarray, max_relative_ink: float = 0.55, min_side_fraction: float = 0.18, max_cuts_per_component: int = 1
):
    """Split low distance-transform-cost seams without removing any ink."""
    return _split_components(image, "stroke_width", max_relative_ink, min_side_fraction, max_cuts_per_component)


def propose_components(
    image: np.ndarray,
    method: str,
    max_relative_ink: float = 0.55,
    min_side_fraction: float = 0.18,
    max_cuts_per_component: int = 1,
):
    """Shared entry point used by existing merger and classifier experiments."""
    if method not in SEGMENTATION_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {tuple(SEGMENTATION_METHODS)}")
    return SEGMENTATION_METHODS[method](image, max_relative_ink, min_side_fraction, max_cuts_per_component)


SEGMENTATION_METHODS = {
    "baseline": baseline_segmentation,
    "skeleton": skeleton_segmentation,
    "stroke_width": stroke_width_segmentation,
}

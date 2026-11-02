from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Segment:
    image: np.ndarray
    bbox: tuple[int, int, int, int]


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    """Validate the shared input contract and return foreground pixels."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected a 2-D uint8 image; call preprocess_image() first")
    if not np.all((image == 0) | (image == 255)):
        raise ValueError("Expected a binary image containing only 0 and 255")
    return image == 0


def connected_component_segmentation(image: np.ndarray) -> list[Segment]:
    """Return one segment for each 8-connected foreground component.

    No components are removed, so tiny dots are preserved. A multi-part
    symbol (e.g. '=') will produce multiple segments; touching symbols
    will produce one segment. This is the unmodified CCA baseline.
    """
    foreground = _foreground_mask(image)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)

    segments = []
    for component_id in range(1, count):  # 0 is the white background
        x, y, width, height, _area = stats[component_id]
        x, y, width, height = map(int, (x, y, width, height))

        # Keep only this component, even if other bounding boxes overlap.
        component_pixels = labels[y : y + height, x : x + width] == component_id
        crop = np.where(component_pixels, 0, 255).astype(np.uint8)
        segments.append(Segment(crop, (x, y, x + width, y + height)))

    return sorted(segments, key=lambda segment: (segment.bbox[0], segment.bbox[1]))


def vertical_projection_segmentation(image: np.ndarray) -> list[Segment]:
    """Split wherever an entire image column has no foreground pixels.

    Each run of non-empty columns becomes a segment, tightly cropped in Y.
    Touching symbols cannot be separated, and disconnected parts whose
    X-ranges do not overlap may be split. This is the projection baseline.
    """
    foreground = _foreground_mask(image)
    occupied_columns = np.any(foreground, axis=0)

    segments = []
    start = None

    # A sentinel empty column flushes the final run.
    for x, occupied in enumerate(np.append(occupied_columns, False)):
        if occupied and start is None:
            start = x
        elif not occupied and start is not None:
            end = x
            rows = np.flatnonzero(np.any(foreground[:, start:end], axis=1))
            y_min, y_max = int(rows[0]), int(rows[-1]) + 1
            segments.append(
                Segment(
                    image[y_min:y_max, start:end].copy(),
                    (start, y_min, end, y_max),
                )
            )
            start = None

    return segments


def augmented_CCL(
    image: np.ndarray,
    overlap_threshold: float = 0.60,
    vertical_ratio_threshold: float = 1.50,
    tiny_width_ratio: float = 0.30,
    tiny_height_ratio: float = 0.30,
    max_merge_distance_ratio: float = 0.70,
) -> list[Segment]:
    """Group disconnected CCA components using overlap and tiny-fragment rules.

    Pass 1: merge components with sufficient X overlap, unless BOTH are tall.
    Pass 2: merge tiny components (small in width AND height relative to the
    image's 75th-percentile component dimensions) to the closest non-tiny
    component, measured by foreground-pixel Euclidean distance.

    Distant tiny artifacts are removed before grouping. Tiny components can be
    merged only within a scale-relative distance, including during overlap
    grouping. Larger legitimate components still follow the original rules.
    Components never get split; grouped crops contain only selected ink pixels.
    """
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in [0, 1]")
    if vertical_ratio_threshold <= 0 or not 0 < tiny_width_ratio < 1 or not 0 < tiny_height_ratio < 1:
        raise ValueError("Aspect and tiny-size ratios must be positive; tiny ratios must be < 1")

    components = connected_component_segmentation(image)
    if len(components) <= 1:
        return components

    boxes = [seg.bbox for seg in components]
    widths = np.array([x2 - x1 for x1, _, x2, _ in boxes], dtype=float)
    heights = np.array([y2 - y1 for _, y1, _, y2 in boxes], dtype=float)
    is_tall = heights / np.maximum(widths, 1) >= vertical_ratio_threshold

    reference_width = float(np.percentile(widths, 75))
    reference_height = float(np.percentile(heights, 75))
    tiny = (widths < tiny_width_ratio * reference_width) & (heights < tiny_height_ratio * reference_height)
    max_merge_distance = max(2.0, max_merge_distance_ratio * reference_height)

    def bbox_gap(a, b):
        """Minimum edge-to-edge separation of two rectangles."""
        dx = max(0, a[0] - b[2], b[0] - a[2])
        dy = max(0, a[1] - b[3], b[1] - a[3])
        return float(np.hypot(dx, dy))

    parent = list(range(len(components)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    # 1. Horizontal-overlap grouping.
    for i, (x1, _, x2, _) in enumerate(boxes):
        for j in range(i + 1, len(components)):
            if is_tall[i] and is_tall[j]:
                continue
            # Overlapping X coordinates alone must not join a tiny distant dot
            # to a symbol many rows above or below it.
            if (tiny[i] or tiny[j]) and bbox_gap(boxes[i], boxes[j]) > max_merge_distance:
                continue
            u1, _, u2, _ = boxes[j]
            overlap = max(0, min(x2, u2) - max(x1, u1))
            fraction = overlap / min(widths[i], widths[j])
            if fraction >= overlap_threshold:
                union(i, j)

    # 2. Assign tiny fragments to the nearest non-tiny ink component.
    anchors = [int(i) for i in np.flatnonzero(~tiny)]

    # Handle each unanchored tiny GROUP once. If overlap already attached a
    # tiny component to a normal component, do not assign it somewhere else.
    anchored_roots = {find(i) for i in anchors}
    pending_groups: dict[int, list[int]] = {}
    for i in np.flatnonzero(tiny):
        i = int(i)
        root = find(i)
        if root not in anchored_roots:
            pending_groups.setdefault(root, []).append(i)

    if anchors and pending_groups:
        distance_maps = {}
        for j in anchors:
            x1, y1, x2, y2 = boxes[j]
            # Distance transform: each pixel's distance to closest zero
            source = np.ones(image.shape, dtype=np.uint8)
            region = source[y1:y2, x1:x2]
            region[components[j].image == 0] = 0
            distance_maps[j] = cv2.distanceTransform(source, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        for members in pending_groups.values():
            ink_coordinates = []
            for i in members:
                x1, y1, _, _ = boxes[i]
                local_ys, local_xs = np.nonzero(components[i].image == 0)
                ink_coordinates.append((local_ys + y1, local_xs + x1))
            ys = np.concatenate([p[0] for p in ink_coordinates])
            xs = np.concatenate([p[1] for p in ink_coordinates])

            closest_anchor = min(
                anchors,
                key=lambda j: float(distance_maps[j][ys, xs].min()),
            )
            closest_distance = float(distance_maps[closest_anchor][ys, xs].min())
            if closest_distance <= max_merge_distance:
                union(members[0], closest_anchor)

    # 3. Rebuild each group using ONLY its members' black pixels.
    groups: dict[int, list[Segment]] = {}
    for i, component in enumerate(components):
        groups.setdefault(find(i), []).append(component)

    output = []
    for members in groups.values():
        x1 = min(seg.bbox[0] for seg in members)
        y1 = min(seg.bbox[1] for seg in members)
        x2 = max(seg.bbox[2] for seg in members)
        y2 = max(seg.bbox[3] for seg in members)
        crop = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
        for member in members:
            mx1, my1, mx2, my2 = member.bbox
            target = crop[my1 - y1 : my2 - y1, mx1 - x1 : mx2 - x1]
            np.minimum(target, member.image, out=target)
        output.append(Segment(crop, (x1, y1, x2, y2)))

    return sorted(output, key=lambda seg: (seg.bbox[0], seg.bbox[1]))


SEGMENTATION_METHODS = {
    "cca": connected_component_segmentation,
    "vertical_projection": vertical_projection_segmentation,
    "augmented_CCL": augmented_CCL,
}

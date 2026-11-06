from __future__ import annotations

import json
import math
import random
from functools import lru_cache
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

import numpy as np

LAYOUT_STATS_PATH = Path(__file__).resolve().parents[1] / "source" / "layout_stats.json"
REFERENCE_DIGIT_HEIGHT = 100.0  # Normalization used by layout_stats.py.
MIN_PERCENTILE = 10
MAX_PERCENTILE = 90
MAX_SYMBOL_ROTATION_DEGREES = 5.0
MAX_EQUATION_ROTATION_DEGREES = 6.0
MIN_RENDERED_INK_GAP_PIXELS = 6

DIGITS = set("0123456789")
PARENTHESES = {"(", ")"}
OPERATORS = {"+", "-", "=", r"\times", r"\div"}
INKML_NAMESPACE = "http://www.w3.org/2003/InkML"


def load_layout_statistics(path: Path = LAYOUT_STATS_PATH) -> dict:
    """Precompute trimmed samples once, not for each generated symbol.

    For a [gap, vertical_offset] sample, both dimensions must lie inside
    their respective P10-P90 intervals, retaining the original paired values.
    """
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)

    for token_group in data["symbol_sizes"].values():
        for token, observations in token_group.items():
            if not observations:
                raise ValueError(f"No size statistics for {token!r}")
            lower, upper = np.percentile(observations, (MIN_PERCENTILE, MAX_PERCENTILE))
            token_group[token] = [v for v in observations if lower <= v <= upper]

    for category, observations in data["relationships"].items():
        if not observations:
            raise ValueError(f"No relationship statistics for {category!r}")
        array = np.asarray(observations, dtype=float)
        lower = np.percentile(array, MIN_PERCENTILE, axis=0)
        upper = np.percentile(array, MAX_PERCENTILE, axis=0)
        data["relationships"][category] = [
            pair for pair in observations if all(lower[j] <= pair[j] <= upper[j] for j in (0, 1))
        ]
        if not data["relationships"][category]:
            raise ValueError(f"No relationship samples after trimming: {category}")

    return data


@lru_cache(maxsize=4096)
def _read_symbol_file(path: Path) -> tuple:
    """Parse each InkML file once per worker (bounded cache)."""
    root = ET.parse(path).getroot()
    label = next(
        (
            (child.text or "").strip()
            for child in root
            if child.tag.endswith("annotation") and child.get("type") == "label"
        ),
        None,
    )
    trace_format = next((child for child in root if child.tag.endswith("traceFormat")), None)
    channels = (
        [channel.get("name") for channel in trace_format if channel.tag.endswith("channel")]
        if trace_format is not None
        else ["X", "Y"]
    )
    if "X" not in channels or "Y" not in channels:
        raise ValueError(f"InkML is missing X/Y channels: {path}")
    x_index, y_index = channels.index("X"), channels.index("Y")
    time_index = channels.index("T") if "T" in channels else None

    traces = []
    for element in root:
        if not element.tag.endswith("trace"):
            continue
        points = []
        for raw_point in (element.text or "").split(","):
            values = raw_point.strip().split()
            if not values:
                continue
            if len(values) <= max(x_index, y_index):
                raise ValueError(f"Incomplete InkML trace point in {path}")
            x, y = float(values[x_index]), float(values[y_index])
            t = float(values[time_index]) if time_index is not None and len(values) > time_index else 0.0
            if not all(math.isfinite(value) for value in (x, y, t)):
                raise ValueError(f"Non-finite trace coordinate in {path}")
            points.append((x, y, t))
        if points:
            traces.append(points)

    if not traces:
        raise ValueError(f"No strokes in {path}")
    normalized, width, height = shift_traces_to_origin(traces)
    return label, normalized, width, height


def load_symbol_label(path: Path) -> str | None:
    """Read the symbol label from the same cached InkML parse as its strokes."""
    return _read_symbol_file(Path(path))[0]


def load_symbol_traces(symbol: dict) -> dict:
    """Return independent stroke lists, safe to rescale or rotate per equation."""
    _, normalized, width, height = _read_symbol_file(Path(symbol["file_path"]))
    return {
        **symbol,
        "traces": [trace.copy() for trace in normalized],
        "width": width,
        "height": height,
    }


def shift_traces_to_origin(traces: list) -> tuple[list, float, float]:
    """Translate a collection of strokes to origin; return its X/Y extents."""
    points = [point for trace in traces for point in trace]
    if not points:
        raise ValueError("Cannot lay out an empty set of strokes")
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    width = max(p[0] for p in points) - min_x
    height = max(p[1] for p in points) - min_y
    return (
        [[(x - min_x, y - min_y, t) for x, y, t in trace] for trace in traces],
        width,
        height,
    )


def size_symbol(symbol: dict, stats: dict) -> dict:
    """Sample the token's measured height/width and uniformly rescale strokes."""
    token = symbol["token"]
    measurement = "width" if token in OPERATORS else "height"
    sample = random.choice(stats["symbol_sizes"][measurement][token])
    target_size = sample * REFERENCE_DIGIT_HEIGHT / 100.0
    original_size = symbol[measurement]
    if original_size <= 0:
        raise ValueError(f"Cannot scale zero-{measurement} token {token!r}: {symbol['file_path']}")

    factor = target_size / original_size
    symbol["traces"] = [[(x * factor, y * factor, t) for x, y, t in trace] for trace in symbol["traces"]]
    symbol["width"] *= factor
    symbol["height"] *= factor
    return symbol


def rotate_symbol(symbol: dict) -> dict:
    """Rotate one glyph slightly, then recalculate its exact bounding box."""
    angle = math.radians(random.uniform(-MAX_SYMBOL_ROTATION_DEGREES, MAX_SYMBOL_ROTATION_DEGREES))
    cosine, sine = math.cos(angle), math.sin(angle)
    cx, cy = symbol["width"] / 2, symbol["height"] / 2
    rotated = [
        [((x - cx) * cosine - (y - cy) * sine + cx, (x - cx) * sine + (y - cy) * cosine + cy, t) for x, y, t in trace]
        for trace in symbol["traces"]
    ]
    symbol["traces"], symbol["width"], symbol["height"] = shift_traces_to_origin(rotated)
    return symbol


def relationship_category(left: dict, right: dict) -> tuple[str | None, int]:
    """Return the statistics category and orientation of its stored Y offset.

    +1 means the recorded offset is right_center - left_center.
    -1 means it is the reverse, because the stats always describe
    operator_center - digit_center, paren_center - digit_center, etc.
    """
    a, b = left["token"], right["token"]
    if left.get("is_sign") and b in DIGITS:
        return "sign_digit", -1
    if a in DIGITS and b in DIGITS:
        return "digit_digit", 1
    if a in DIGITS and b in OPERATORS:
        return "digit_operator", 1
    if a in OPERATORS and b in DIGITS:
        return "digit_operator", -1
    if a in DIGITS and b in PARENTHESES:
        return "digit_parenthesis", 1
    if a in PARENTHESES and b in DIGITS:
        return "digit_parenthesis", -1
    if a in OPERATORS and b in PARENTHESES:
        return "operator_parenthesis", 1
    if a in PARENTHESES and b in OPERATORS:
        return "operator_parenthesis", -1
    return None, 1


def position_symbol(symbol: dict, previous: dict | None, stats: dict) -> dict:
    """Place a glyph relative to the previous glyph using *paired* gap/Y data."""
    if previous is None:
        symbol["x"] = 0.0
        symbol["y"] = 0.0
        symbol["requested_ink_gap_units"] = 0.0
        return symbol

    category, y_direction = relationship_category(previous, symbol)
    if category is None:
        gap, vertical_offset = 30.0, 0.0
    else:
        gap, vertical_offset = random.choice(stats["relationships"][category])

    symbol["requested_ink_gap_units"] = max(0.0, float(gap) * REFERENCE_DIGIT_HEIGHT / 100.0)
    symbol["x"] = previous["x"] + previous["width"] + symbol["requested_ink_gap_units"]
    previous_center_y = previous["y"] + previous["height"] / 2
    symbol_center_y = previous_center_y + y_direction * vertical_offset * REFERENCE_DIGIT_HEIGHT / 100.0
    symbol["y"] = symbol_center_y - symbol["height"] / 2
    return symbol


def combine_symbols(symbols: list[dict]) -> list:
    """Merge positioned glyph strokes without joining distinct pen strokes."""
    combined = []
    for symbol in symbols:
        for trace in symbol["traces"]:
            combined.append([(x + symbol["x"], y + symbol["y"], t) for x, y, t in trace])
    return combined


def rotate_equation(traces: list) -> list:
    """Apply one global rotation around the equation's centre and re-normalize."""
    points = [point for trace in traces for point in trace]
    if not points:
        raise ValueError("Equation contains no strokes")
    cx = (min(p[0] for p in points) + max(p[0] for p in points)) / 2
    cy = (min(p[1] for p in points) + max(p[1] for p in points)) / 2
    angle = math.radians(random.uniform(-MAX_EQUATION_ROTATION_DEGREES, MAX_EQUATION_ROTATION_DEGREES))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated = [
        [((x - cx) * cosine - (y - cy) * sine + cx, (x - cx) * sine + (y - cy) * cosine + cy, t) for x, y, t in trace]
        for trace in traces
    ]
    return shift_traces_to_origin(rotated)[0]


INK_COLOURS = (
    (30, 30, 30),
    (110, 40, 20),
    (60, 60, 60),
    (60, 20, 15),
)
WRITING_HEIGHT_RANGE = (120, 200)
JPEG_QUALITY_RANGE = (65, 95)
PAGE_MARGIN = 35


NOISE_SPECK_COUNT_RANGE = (0, 3)
NOISE_SPECK_RADIUS_RANGE = (1, 2)
OVERLAP_ID = 65535
RENDER_AUDIT_THRESHOLD = 127
MIN_BLUR_CONTRIBUTION = 0.25


def make_paper(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Create off-white paper with subtle texture and uneven lighting."""
    paper_colour = np.array(
        [rng.uniform(220, 245), rng.uniform(225, 250), rng.uniform(230, 255)],
        dtype=np.float32,
    )
    paper = np.empty((height, width, 3), dtype=np.float32)
    paper[:] = paper_colour

    # Simulate a mild shadow across the paper.
    ys, xs = np.indices((height, width), dtype=np.float32)
    angle = rng.uniform(0, 2 * math.pi)
    direction = xs * math.cos(angle) + ys * math.sin(angle)
    direction -= direction.min()
    direction /= max(float(direction.max()), 1.0)
    shadow_strength = rng.uniform(0.05, 0.22)
    paper *= 1 - shadow_strength * direction[..., None]

    # Fine paper grain / camera sensor noise.
    paper += rng.normal(0, rng.uniform(1.5, 4.0), paper.shape).astype(np.float32)
    return paper


def layout_symbols(selected_symbols: list[dict], stats: dict) -> list[dict]:
    """Produce globally transformed, individually identified symbol strokes."""
    prepared = [rotate_symbol(size_symbol(load_symbol_traces(item), stats)) for item in selected_symbols]
    for index, symbol in enumerate(prepared):
        position_symbol(symbol, prepared[index - 1] if index else None, stats)

    counts = [len(symbol["traces"]) for symbol in prepared]
    positioned = combine_symbols(prepared)
    rotated = rotate_equation(positioned)
    offset = 0
    for instance_id, (symbol, count) in enumerate(zip(prepared, counts), start=1):
        symbol["instance_id"] = instance_id
        symbol["traces"] = rotated[offset : offset + count]
        offset += count
    assert offset == len(rotated)
    return prepared


def _draw_stroke(canvas: np.ndarray, points: np.ndarray, line_width: int, value: int = 255) -> None:
    """Exactly reproduce the previous renderer's antialiased polyline/dot rule."""
    extent = np.ptp(points, axis=0)
    if len(points) == 1 or max(extent) <= line_width:
        centre = tuple(np.round(points.mean(axis=0)).astype(int))
        cv2.circle(canvas, centre, max(1, line_width // 2), value, -1, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [points], False, value, line_width, cv2.LINE_AA)


def _rasterized_ink_x_bounds(symbol: dict, scale: float, line_width: int) -> tuple[float, float]:
    """Horizontal extent of actual pen pixels for a transformed symbol."""
    points = [(x, y) for trace in symbol["traces"] for x, y, _ in trace]
    lo_x = min(p[0] for p in points)
    hi_x = max(p[0] for p in points)
    lo_y = min(p[1] for p in points)
    hi_y = max(p[1] for p in points)
    pad = max(8, line_width + 4)
    width = max(1, math.ceil((hi_x - lo_x) * scale)) + 2 * pad + 3
    height = max(1, math.ceil((hi_y - lo_y) * scale)) + 2 * pad + 3
    layer = np.zeros((height, width), dtype=np.uint8)
    for trace in symbol["traces"]:
        coords = np.asarray(
            [[round((x - lo_x) * scale) + pad, round((y - lo_y) * scale) + pad] for x, y, _ in trace],
            dtype=np.int32,
        )
        _draw_stroke(layer, coords, line_width)
    _, ink_xs = np.nonzero(layer > 0)
    if not len(ink_xs):
        raise ValueError(f"No rasterized ink for {symbol.get('token', '<unknown>')!r}")
    return lo_x * scale + float(ink_xs.min() - pad), lo_x * scale + float(ink_xs.max() - pad)


def enforce_visible_ink_spacing(
    symbols: list[dict],
    scale: float,
    line_width: int,
    minimum_gap_pixels: int = MIN_RENDERED_INK_GAP_PIXELS,
) -> None:
    """Move later tokens right until their rendered ink clears earlier ink."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if minimum_gap_pixels < 1:
        raise ValueError("minimum_gap_pixels must be >= 1")
    rightmost_ink: float | None = None
    for symbol in symbols:
        left, right = _rasterized_ink_x_bounds(symbol, scale, line_width)
        if rightmost_ink is not None:
            requested_pixels = float(symbol.get("requested_ink_gap_units", 0.0)) * scale
            target = max(float(minimum_gap_pixels), requested_pixels)
            shift_pixels = max(0, math.ceil(rightmost_ink + target + 1 - left))
            if shift_pixels:
                dx = shift_pixels / scale
                symbol["traces"] = [[(x + dx, y, t) for x, y, t in trace] for trace in symbol["traces"]]
                left += shift_pixels
                right += shift_pixels
        rightmost_ink = right if rightmost_ink is None else max(rightmost_ink, right)


def _draw_specks(shape: tuple[int, int], foreground: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate separately labelled dark artifacts, without overwriting real ink."""
    height, width = shape
    result = np.zeros(shape, dtype=np.uint8)
    count = int(rng.integers(NOISE_SPECK_COUNT_RANGE[0], NOISE_SPECK_COUNT_RANGE[1] + 1))
    if not count:
        return result
    forbidden = cv2.dilate((foreground > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    for _ in range(count):
        for _attempt in range(24):
            x = int(rng.integers(1, max(width - 1, 2)))
            y = int(rng.integers(1, max(height - 1, 2)))
            radius = int(rng.integers(NOISE_SPECK_RADIUS_RANGE[0], NOISE_SPECK_RADIUS_RANGE[1] + 1))
            y1, y2 = max(0, y - radius), min(height, y + radius + 1)
            x1, x2 = max(0, x - radius), min(width, x + radius + 1)
            if not forbidden[y1:y2, x1:x2].any() and not result[y1:y2, x1:x2].any():
                cv2.circle(result, (x, y), radius, 255, -1, cv2.LINE_AA)
                break
    return result


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _draw_symbol_layer(
    symbol: dict, *, shape: tuple[int, int], min_x: float, min_y: float, scale: float, line_width: int
) -> np.ndarray:
    """Rasterize a single instance exactly as it contributes to the photo."""
    layer = np.zeros(shape, dtype=np.uint8)
    for trace in symbol["traces"]:
        coords = np.array(
            [[round((x - min_x) * scale) + PAGE_MARGIN, round((y - min_y) * scale) + PAGE_MARGIN] for x, y, _ in trace],
            dtype=np.int32,
        )
        _draw_stroke(layer, coords, line_width)
    return layer


def _blur_source_contributions(
    symbols: list[dict],
    *,
    shape: tuple[int, int],
    min_x: float,
    min_y: float,
    scale: float,
    line_width: int,
    perspective: np.ndarray,
    sigma: float,
    true_noise_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse-blur attribution with one float accumulator per input image."""
    height, width = shape
    strongest = np.zeros(shape, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.uint16)
    for symbol in symbols:
        drawn = _draw_symbol_layer(
            symbol,
            shape=shape,
            min_x=min_x,
            min_y=min_y,
            scale=scale,
            line_width=line_width,
        )
        warped = cv2.warpPerspective(
            drawn,
            perspective,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        weight = cv2.GaussianBlur(warped.astype(np.float32), (0, 0), sigmaX=sigma)
        wins = (weight >= MIN_BLUR_CONTRIBUTION) & (weight >= strongest)
        labels[wins] = int(symbol["instance_id"])
        strongest[wins] = weight[wins]

    warped_noise = cv2.warpPerspective(
        true_noise_alpha,
        perspective,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    noise_weight = cv2.GaussianBlur(warped_noise.astype(np.float32), (0, 0), sigmaX=sigma)
    noise_wins = (noise_weight >= MIN_BLUR_CONTRIBUTION) & (noise_weight > strongest)
    labels[noise_wins] = 0
    protected = (
        cv2.warpPerspective(
            (true_noise_alpha >= 128).astype(np.uint8),
            perspective,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    labels[protected] = 0
    return labels, protected.astype(np.uint8) * 255


def render_equation(
    selected_symbols: list[dict],
    stats: dict,
    output_path: str | Path,
    line_width: int = 4,
    seed: int | None = None,
) -> Path:
    """Render photo and fully attributable symbol/noise masks."""
    if line_width < 1:
        raise ValueError("line_width must be at least 1 pixel")
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("Output image must use the .png extension")

    rng = np.random.default_rng(seed)
    symbols = layout_symbols(selected_symbols, stats)
    if not symbols:
        raise ValueError("No symbols selected")
    if len(symbols) >= OVERLAP_ID:
        raise ValueError("Too many instances for uint16 labels")
    xy = [(x, y) for symbol in symbols for trace in symbol["traces"] for x, y, _ in trace]
    min_y, max_y = min(p[1] for p in xy), max(p[1] for p in xy)
    writing_height = rng.uniform(*WRITING_HEIGHT_RANGE)
    scale = writing_height / max(max_y - min_y, 1.0)
    enforce_visible_ink_spacing(symbols, scale, line_width)
    xy = [(x, y) for symbol in symbols for trace in symbol["traces"] for x, y, _ in trace]
    min_x, max_x = min(p[0] for p in xy), max(p[0] for p in xy)
    min_y, max_y = min(p[1] for p in xy), max(p[1] for p in xy)
    width = max(1, math.ceil((max_x - min_x) * scale)) + 2 * PAGE_MARGIN
    height = max(1, math.ceil((max_y - min_y) * scale)) + 2 * PAGE_MARGIN
    shape = (height, width)
    paper = make_paper(width, height, rng)
    ink = np.zeros(shape, dtype=np.uint8)
    for symbol in symbols:
        np.maximum(
            ink,
            _draw_symbol_layer(
                symbol,
                shape=shape,
                min_x=min_x,
                min_y=min_y,
                scale=scale,
                line_width=line_width,
            ),
            out=ink,
        )

    speck_rng = np.random.default_rng(None if seed is None else (int(seed) ^ 0x5CA1AB1E))
    noise_alpha = _draw_specks(shape, ink, speck_rng)
    colour = np.array(INK_COLOURS[rng.integers(len(INK_COLOURS))], dtype=np.float32)
    ink_opacity = rng.uniform(0.88, 1.0)
    alpha = (np.maximum(ink, noise_alpha).astype(np.float32) / 255 * ink_opacity)[..., None]
    photo = paper * (1 - alpha) + colour * alpha

    jitter = 0.015 * min(width, height)
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    target = source + rng.uniform(-jitter, jitter, source.shape).astype(np.float32)
    perspective = cv2.getPerspectiveTransform(source, target)
    photo = cv2.warpPerspective(photo, perspective, (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Identical sigma and (auto-selected) kernel for photo, symbols and specks.
    sigma = float(rng.uniform(0.3, 1.0))
    instance_ids, true_noise_mask = _blur_source_contributions(
        symbols,
        shape=shape,
        min_x=min_x,
        min_y=min_y,
        scale=scale,
        line_width=line_width,
        perspective=perspective,
        sigma=sigma,
        true_noise_alpha=noise_alpha,
    )
    photo = cv2.GaussianBlur(photo, (0, 0), sigmaX=sigma)
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    quality = int(rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1))
    ok, jpeg = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to simulate JPEG compression")
    photo = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    noise_mask = np.where(
        ((gray <= RENDER_AUDIT_THRESHOLD) & (instance_ids == 0)) | (true_noise_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    noise_mask[instance_ids != 0] = 0
    # true noise must be kept separate for the next Gaussian correction stage
    true_noise_mask[instance_ids != 0] = 0
    if np.any((gray <= RENDER_AUDIT_THRESHOLD) & (instance_ids == 0) & (noise_mask == 0)):
        raise AssertionError("Renderer emitted unexplained dark foreground")
    if np.any(instance_ids == OVERLAP_ID):
        raise AssertionError("Renderer emitted obsolete overlap ID")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    instance_path = stem.with_name(stem.name + "_instances.png")
    noise_path = stem.with_name(stem.name + "_noise.png")
    true_noise_path = stem.with_name(stem.name + "_true_noise.png")
    meta_path = stem.with_name(stem.name + "_symbols.json")

    metadata = {
        "image": output_path.name,
        "instance_mask": instance_path.name,
        "noise_mask": noise_path.name,
        "true_noise_mask": true_noise_path.name,
        "width": width,
        "height": height,
        "instance_background_id": 0,
        "instance_overlap_id": None,
        "noise_mask_semantics": "255 = rendered non-symbol dark pixel, including injected noise; 0 otherwise",
        "true_noise_mask_semantics": "255 = protected injected-noise seed, 0 otherwise",
        "blur_sigma": sigma,
        "symbols": [],
    }
    for symbol in symbols:
        ident = int(symbol["instance_id"])
        metadata["symbols"].append(
            {
                "instance_id": ident,
                "token": symbol["token"],
                "bbox_xyxy": _bbox_from_mask((instance_ids == ident) & (gray <= RENDER_AUDIT_THRESHOLD)),
                "source_inkml": str(symbol["file_path"]),
                "trace_count": len(symbol["traces"]),
            }
        )

    for path, data in (
        (output_path, photo),
        (instance_path, instance_ids),
        (noise_path, noise_mask),
        (true_noise_path, true_noise_mask),
    ):
        if not cv2.imwrite(str(path), data):
            raise OSError(f"Cannot save {path}")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path

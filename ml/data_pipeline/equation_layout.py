"""Arrange handwritten symbol InkML into one equation for photo_renderer.

This module owns all stroke geometry. The equation generator supplies an
ordered list of dictionaries containing token, type, file_path, and is_sign.

The input layout_stats.json has already had 1.5x-IQR outliers removed.
P10-P90 trimming is applied here *only for sampling*, leaving that JSON intact.
Run equation_generator.py as a module to create the image dataset.
"""

from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


LAYOUT_STATS_PATH = Path(__file__).with_name("layout_stats.json")
REFERENCE_DIGIT_HEIGHT = 100.0  # Normalization used by layout_stats.py.
MIN_PERCENTILE = 10
MAX_PERCENTILE = 90
MAX_SYMBOL_ROTATION_DEGREES = 5.0
MAX_EQUATION_ROTATION_DEGREES = 6.0

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
            pair for pair in observations
            if all(lower[j] <= pair[j] <= upper[j] for j in (0, 1))
        ]
        if not data["relationships"][category]:
            raise ValueError(f"No relationship samples after trimming: {category}")

    return data


def load_symbol_traces(symbol: dict) -> dict:
    """Read X/Y and optional T from symbol InkML; move its bounds to (0, 0)."""
    path = Path(symbol["file_path"])
    root = ET.parse(path).getroot()
    trace_format = next((child for child in root if child.tag.endswith("traceFormat")), None)
    channels = (
        [channel.get("name") for channel in trace_format if channel.tag.endswith("channel")]
        if trace_format is not None else ["X", "Y"]
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
    return {**symbol, "traces": normalized, "width": width, "height": height}


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
    symbol["traces"] = [
        [(x * factor, y * factor, t) for x, y, t in trace]
        for trace in symbol["traces"]
    ]
    symbol["width"] *= factor
    symbol["height"] *= factor
    return symbol


def rotate_symbol(symbol: dict) -> dict:
    """Rotate one glyph slightly, then recalculate its exact bounding box."""
    angle = math.radians(random.uniform(-MAX_SYMBOL_ROTATION_DEGREES, MAX_SYMBOL_ROTATION_DEGREES))
    cosine, sine = math.cos(angle), math.sin(angle)
    cx, cy = symbol["width"] / 2, symbol["height"] / 2
    rotated = [
        [
            ((x - cx) * cosine - (y - cy) * sine + cx,
             (x - cx) * sine + (y - cy) * cosine + cy, t)
            for x, y, t in trace
        ]
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
        return "sign_digit", -1  # stored: sign_center - digit_center
    if a in DIGITS and b in DIGITS:
        return "digit_digit", 1
    if a in DIGITS and b in OPERATORS:
        return "digit_operator", 1  # stored: operator_center - digit_center
    if a in OPERATORS and b in DIGITS:
        return "digit_operator", -1
    if a in DIGITS and b in PARENTHESES:
        return "digit_parenthesis", 1  # stored: paren_center - digit_center
    if a in PARENTHESES and b in DIGITS:
        return "digit_parenthesis", -1
    if a in OPERATORS and b in PARENTHESES:
        return "operator_parenthesis", 1  # stored: paren_center - operator_center
    if a in PARENTHESES and b in OPERATORS:
        return "operator_parenthesis", -1
    return None, 1


def position_symbol(symbol: dict, previous: dict | None, stats: dict) -> dict:
    """Place a glyph relative to the previous glyph using *paired* gap/Y data."""
    if previous is None:
        symbol["x"] = 0.0
        symbol["y"] = 0.0
        return symbol

    category, y_direction = relationship_category(previous, symbol)
    if category is None:
        # Rare unsupported pairs (e.g. parenthesis->parenthesis).
        gap, vertical_offset = 30.0, 0.0
    else:
        gap, vertical_offset = random.choice(stats["relationships"][category])

    symbol["x"] = previous["x"] + previous["width"] + gap * REFERENCE_DIGIT_HEIGHT / 100.0
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
        [
            ((x - cx) * cosine - (y - cy) * sine + cx,
             (x - cx) * sine + (y - cy) * cosine + cy, t)
            for x, y, t in trace
        ]
        for trace in traces
    ]
    return shift_traces_to_origin(rotated)[0]


def layout_equation(selected_symbols: list[dict], stats: dict) -> ET.Element:
    """Perform all layout steps and return in-memory InkML for render_photo()."""
    prepared = [rotate_symbol(size_symbol(load_symbol_traces(item), stats)) for item in selected_symbols]
    for index, symbol in enumerate(prepared):
        position_symbol(symbol, prepared[index - 1] if index else None, stats)
    traces = rotate_equation(combine_symbols(prepared))

    root = ET.Element(f"{{{INKML_NAMESPACE}}}ink")
    fmt = ET.SubElement(root, f"{{{INKML_NAMESPACE}}}traceFormat")
    for channel in ("X", "Y", "T"):
        ET.SubElement(fmt, f"{{{INKML_NAMESPACE}}}channel", name=channel, type="decimal")
    for index, trace in enumerate(traces):
        element = ET.SubElement(root, f"{{{INKML_NAMESPACE}}}trace", id=str(index))
        element.text = ",".join(f"{x:.5f} {y:.5f} {t:.3f}" for x, y, t in trace)
    return root

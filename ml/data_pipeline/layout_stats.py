"""
Real handwriting layout statistics.

The generators used to place symbols with hand-picked rules (every operator
0.8 digit-heights wide, brackets exactly digit height, gaps 0.15-0.6 digit
heights). Real handwriting is different - e.g. minus signs are short and float
at mid-height, brackets are taller than digits, digits inside a number sit
closer together than symbols around an operator.

This module measures those things from real human-written equations
(MathWriting, train split only - the valid/test splits stay unseen) and lets
both generators sample sizes, vertical positions and gaps from the measured
distributions.

    python -m ml.data_pipeline.layout_stats     # re-measure -> ml/data_pipeline/layout_stats.json

The measured statistics are committed with the code (layout_stats.json next
to this file), so the generators work without the MathWriting human data.
Re-measuring is only needed to change how the statistics are measured.
"""

import json
import random
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HUMAN_INK_FOLDER = PROJECT_ROOT / "ml" / "data" / "mathwriting-2024" / "human"
STATS_PATH = Path(__file__).resolve().with_name("layout_stats.json")

# Only measure inks from this original MathWriting split
MEASURE_SPLIT = "train"

DIGITS = set("0123456789")
FLAT_TOKENS = {"-", "="}  # sized by width; their height is too small to be reliable

# Fewer measured samples than this for a token -> use the rule-based size/position
MIN_SAMPLES = 15

# Fewest samples that may remain after trimming outliers
MIN_TRIMMED_SAMPLES = 8

_cache = {"stats": None}


# -------------------------
# Measuring
# -------------------------


def read_ink(path):
    root = ET.parse(path).getroot()

    label = split = None
    strokes = []

    for element in root.iter():
        if element.tag.endswith("annotation"):
            if element.attrib.get("type") == "normalizedLabel":
                label = element.text
            elif element.attrib.get("type") == "splitTagOriginal":
                split = element.text

        elif element.tag.endswith("trace") and element.text:
            points = []

            for raw in element.text.strip().split(","):
                values = raw.split()

                if len(values) >= 2:
                    points.append((float(values[0]), float(values[1])))

            if points:
                strokes.append(points)

    return label, split, strokes


def group_strokes(strokes, return_members=False):
    """
    Groups strokes into symbols: strokes whose horizontal ranges overlap
    (by more than half of the narrower one) belong to the same symbol.
    Works for simple left-to-right equations ("+", "=", "÷", two-stroke digits).

    Returns a list of bounding boxes (x0, y0, x1, y1), left to right - and,
    with return_members=True, also the stroke indices in each symbol.
    """

    boxes = []

    for index, stroke in enumerate(strokes):
        xs = [p[0] for p in stroke]
        ys = [p[1] for p in stroke]
        boxes.append(([min(xs), min(ys), max(xs), max(ys)], index))

    boxes.sort(key=lambda item: (item[0][0] + item[0][2]) / 2)

    groups = []
    members = []

    for box, index in boxes:
        if groups:
            last = groups[-1]
            overlap = min(last[2], box[2]) - max(last[0], box[0])
            narrower = max(1e-6, min(last[2] - last[0], box[2] - box[0]))

            if overlap > 0.5 * narrower:
                last[0] = min(last[0], box[0])
                last[1] = min(last[1], box[1])
                last[2] = max(last[2], box[2])
                last[3] = max(last[3], box[3])
                members[-1].append(index)
                continue

        groups.append(list(box))
        members.append([index])

    if return_members:
        return groups, members

    return groups


def gap_category(left_token, right_token):
    left_is_value = left_token in DIGITS or left_token == ")"
    right_is_value = right_token in DIGITS or right_token == "("

    if left_token in DIGITS and right_token in DIGITS:
        return "digit_digit"

    if left_token == "(" or right_token == ")":
        return "inside_bracket"

    if left_is_value and not right_is_value:
        return "before_operator"

    if not left_is_value and right_is_value:
        return "after_operator"

    return "other"


def measure(ink_folder=HUMAN_INK_FOLDER, split=MEASURE_SPLIT):
    from ml.models.ctc.tokenizer import CTCTokenizer

    tokenizer = CTCTokenizer()

    sizes = defaultdict(list)  # token -> [(width/H, height/H, centre offset/H)]
    gaps = defaultdict(list)  # category -> [gap/H]

    used = skipped = 0

    for path in sorted(Path(ink_folder).glob("*.inkml")):
        label, ink_split, strokes = read_ink(path)

        if ink_split != split or not label or not strokes:
            continue

        try:
            tokens = [tokenizer.tokens[i] for i in tokenizer.encode(label)]
        except ValueError:
            skipped += 1
            continue

        groups = group_strokes(strokes)

        # Only trust equations where grouping found exactly one box per symbol
        if len(groups) != len(tokens):
            skipped += 1
            continue

        digit_boxes = [box for box, token in zip(groups, tokens) if token in DIGITS]

        if not digit_boxes:
            skipped += 1
            continue

        digit_height = statistics.median(b[3] - b[1] for b in digit_boxes)
        digit_centre = statistics.median((b[1] + b[3]) / 2 for b in digit_boxes)

        if digit_height <= 0:
            skipped += 1
            continue

        used += 1

        for box, token in zip(groups, tokens):
            width = (box[2] - box[0]) / digit_height
            height = (box[3] - box[1]) / digit_height
            centre = ((box[1] + box[3]) / 2 - digit_centre) / digit_height
            sizes[token].append((width, height, centre))

        for (left_box, left_token), (right_box, right_token) in zip(
            zip(groups, tokens), zip(groups[1:], tokens[1:])
        ):
            gap = (right_box[0] - left_box[2]) / digit_height
            gaps[gap_category(left_token, right_token)].append(gap)

    stats = {
        "source": f"MathWriting human inks, {split} split",
        "equations_used": used,
        "equations_skipped": skipped,
        "sizes": dict(sizes),
        "gaps": dict(gaps),
    }

    return stats


def main():
    stats = measure()

    # 3 decimals is far finer than the measurements are accurate
    stats["sizes"] = {t: [[round(v, 3) for v in row] for row in rows] for t, rows in stats["sizes"].items()}
    stats["gaps"] = {c: [round(v, 3) for v in values] for c, values in stats["gaps"].items()}

    with open(STATS_PATH, "w", encoding="utf-8") as file:
        json.dump(stats, file)

    print(f"Measured {stats['equations_used']} equations ({stats['equations_skipped']} skipped) -> {STATS_PATH}")

    for token, values in sorted(stats["sizes"].items()):
        w = statistics.median(v[0] for v in values)
        h = statistics.median(v[1] for v in values)
        c = statistics.median(v[2] for v in values)
        print(f"  {token:>7}: n={len(values):4d}  width {w:.2f}  height {h:.2f}  centre offset {c:+.2f}  (x digit height)")

    for category, values in sorted(stats["gaps"].items()):
        q = statistics.quantiles(values, n=10) if len(values) > 1 else values * 9
        print(f"  gap {category:<16}: n={len(values):4d}  10%={q[0]:+.2f}  median={statistics.median(values):+.2f}  90%={q[-1]:+.2f}")


# -------------------------
# Sampling (used by the generators)
# -------------------------


# Samples outside this percentile range are ignored when sampling. A few
# equations get mis-grouped (e.g. a "5" measured 0.0 tall, a bracket 3.8x
# digit height); trimming the tails removes those without losing real variety.
TRIM_PERCENTILE = 5


def _trim(values, columns):
    """Keeps rows whose every column lies within the trimmed percentile range."""

    if len(values) < MIN_SAMPLES:
        return values

    array = np.asarray(values, dtype=np.float64).reshape(len(values), columns)
    low = np.percentile(array, TRIM_PERCENTILE, axis=0)
    high = np.percentile(array, 100 - TRIM_PERCENTILE, axis=0)
    keep = np.all((array >= low) & (array <= high), axis=1)

    return [values[i] for i in np.nonzero(keep)[0]]


def load_stats():
    if _cache["stats"] is None:
        if not STATS_PATH.exists():
            raise FileNotFoundError(
                f"{STATS_PATH} is missing. It is part of the code; restore it, or re-create it with "
                "`python -m ml.data_pipeline.layout_stats` (needs the MathWriting human inks)."
            )

        with open(STATS_PATH, encoding="utf-8") as file:
            stats = json.load(file)

        stats["sizes"] = {token: _trim(values, 3) for token, values in stats["sizes"].items()}
        stats["gaps"] = {category: _trim(values, 1) for category, values in stats["gaps"].items()}

        _cache["stats"] = stats

    return _cache["stats"]


def sample_size_and_offset(token):
    """
    Returns (target width or None, target height or None, centre offset) in
    digit heights, sampled from real handwriting - or None if there isn't
    enough data for this token. Flat symbols ("-", "=") get a width, the rest
    a height.
    """

    stats = load_stats()

    if len(stats["sizes"].get(token, [])) < MIN_TRIMMED_SAMPLES:
        return None

    width, height, centre = random.choice(stats["sizes"][token])

    if token in FLAT_TOKENS:
        return width, None, centre

    return None, height, centre


def sample_gap(left_token, right_token):
    """Gap in digit heights between two neighbouring symbols, or None."""

    stats = load_stats()

    values = stats["gaps"].get(gap_category(left_token, right_token), [])

    if len(values) < MIN_TRIMMED_SAMPLES:
        return None

    return random.choice(values)


if __name__ == "__main__":
    main()

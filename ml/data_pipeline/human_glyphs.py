"""
Cuts individual symbols out of real handwritten equations.

MathWriting only has ~30 drawings per symbol in symbols/. Its human-written
equations contain many more, in the natural style people use when writing a
whole equation. For simple left-to-right equations made only of our symbols,
strokes can be grouped into symbols by horizontal overlap (layout_stats.py);
when the number of groups matches the label, each group is saved as its own
symbol file.

Only the MathWriting *train* split is used - valid/test writers stay unseen.
The ink generator adds these drawings to its training pool only, so the
validation and test pools are unchanged.

Run from the project root (after data_cleaning.py):

    python -m ml.data_pipeline.human_glyphs

Output: ml/data/mathwriting-2024/human_symbols/*.inkml (one symbol each,
with a "label" annotation like the files in symbols/).
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from ml.data_pipeline.layout_stats import HUMAN_INK_FOLDER, MEASURE_SPLIT, group_strokes
from ml.models.ctc.tokenizer import CTCTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FOLDER = PROJECT_ROOT / "ml" / "data" / "mathwriting-2024" / "human_symbols"


def read_strokes(path):
    """Returns (label, split, strokes) with strokes as lists of (x, y, t)."""

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
                    t = float(values[2]) if len(values) > 2 else 0.0
                    points.append((float(values[0]), float(values[1]), t))

            if points:
                strokes.append(points)

    return label, split, strokes


def save_symbol(strokes, token, path):
    ink = ET.Element("ink", xmlns="http://www.w3.org/2003/InkML")
    ET.SubElement(ink, "annotation", type="label").text = token
    ET.SubElement(ink, "annotation", type="inkCreationMethod").text = "humanEquationSegment"

    for index, stroke in enumerate(strokes):
        trace = ET.SubElement(ink, "trace", id=str(index))
        trace.text = ",".join(f"{x:.2f} {y:.2f} {t:.1f}" for x, y, t in stroke)

    ET.ElementTree(ink).write(path, encoding="utf-8", xml_declaration=True)


def main():
    tokenizer = CTCTokenizer()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    for old in OUTPUT_FOLDER.glob("*.inkml"):
        old.unlink()

    counts = Counter()
    used = skipped = 0

    for path in sorted(HUMAN_INK_FOLDER.glob("*.inkml")):
        label, split, strokes = read_strokes(path)

        if split != MEASURE_SPLIT or not label or not strokes:
            continue

        try:
            tokens = [tokenizer.tokens[i] for i in tokenizer.encode(label)]
        except ValueError:
            skipped += 1
            continue

        groups, members = group_strokes([[(x, y) for x, y, _ in s] for s in strokes], return_members=True)

        # Only trust equations where grouping found exactly one group per symbol
        if len(groups) != len(tokens):
            skipped += 1
            continue

        used += 1

        for position, (token, stroke_indices) in enumerate(zip(tokens, members)):
            symbol_strokes = [strokes[i] for i in sorted(stroke_indices)]
            save_symbol(symbol_strokes, token, OUTPUT_FOLDER / f"{path.stem}_{position:02d}.inkml")
            counts[token] += 1

    print(f"Used {used} equations ({skipped} skipped) -> {sum(counts.values())} symbols in {OUTPUT_FOLDER}")
    print("  " + ", ".join(f"{token}: {count}" for token, count in sorted(counts.items())))


if __name__ == "__main__":
    main()

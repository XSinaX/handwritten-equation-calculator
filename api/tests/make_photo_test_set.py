"""
Builds a "phone photo" test set from real handwritten equations.

Each MathWriting human ink is drawn onto a simulated photo of paper
(ml/data_pipeline/photo_simulation.py): uneven lighting, paper texture, ruled
lines, ink colours, slight rotation/perspective, blur and JPEG compression.

The images are then fed through the real API by evaluate_api.py, so the score
reflects the full path a photo takes: upload -> API preprocessing -> model
-> decoding -> calculation.

Run from the project root:

    python -m api.tests.make_photo_test_set --splits valid test
    python -m api.tests.make_photo_test_set --splits train valid test --output api/tests/photos_all

Writes <output>/*.jpg and <output>/labels.csv (filename, equation, split).
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ml.data_pipeline.photo_simulation import jpeg_round_trip, render_photo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INK_FOLDER = PROJECT_ROOT / "ml" / "data" / "mathwriting-2024" / "human"
DEFAULT_OUTPUT = PROJECT_ROOT / "api" / "tests" / "photos"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ink-folder", type=Path, default=DEFAULT_INK_FOLDER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"], help="Original MathWriting splits to use")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    rng = np.random.default_rng(arguments.seed)
    arguments.output.mkdir(parents=True, exist_ok=True)

    rows = []

    for path in sorted(arguments.ink_folder.glob("*.inkml")):
        label, split, strokes = read_ink(path)

        if split not in arguments.splits or not label or not strokes:
            continue

        photo = render_photo(strokes, rng)
        filename = f"{path.stem}.jpg"
        encoded, _ = jpeg_round_trip(photo, rng)
        (arguments.output / filename).write_bytes(encoded.tobytes())

        rows.append({"filename": filename, "equation": label, "split": split})

    with open(arguments.output / "labels.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "equation", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} photo-like images to {arguments.output}")


if __name__ == "__main__":
    main()

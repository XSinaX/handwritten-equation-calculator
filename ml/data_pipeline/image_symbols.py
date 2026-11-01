"""
Image-based equation generator.

The ink generator (equation_generator.py) only has ~30 handwritten drawings
per symbol, so the model sees very few handwriting styles. This module builds
equations from image datasets of individual handwritten symbols (thousands
per symbol) instead:

    Xainano_Raw/dataset/<symbol>/*.jpg   45x45, thin pen  (all 17 symbols)
    sagyam_raw/dataset/<symbol>/*.png    ~150x135, marker (no brackets)

Equation structure (equation_generator.py) and symbol sizes, vertical
positions and gaps (measured from real handwriting, layout_stats.py) are shared
with the ink generator, so both produce the same kinds of equations. Only the
final step differs: symbol images are pasted onto a canvas instead of ink
strokes being drawn.

Every symbol is cleaned (black/white, cropped, stray lines removed) and thinned
to a 1-pixel centre line; a random pen width is then applied to the whole
equation, matching the range of stroke widths in the ink-built images.

Run from the project root:

    python -m ml.data_pipeline.image_symbols

Output (same format as the ink pipeline):
    ml/data/image_generated/train/X.npy, y.npy
    ml/data/image_generated/test/X.npy, y.npy
"""

import math
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import ml.data_pipeline.equation_generator as ink_generator
from ml.data_pipeline.preprocessing import resize_image
from ml.data_pipeline.thinning import thin
from shared.config import BLANK_TOKEN, IMAGE_HEIGHT, IMAGE_WIDTH, TOKENS

# -------------------------
# Locations
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / "ml" / "data"

# Folder holding Xainano_Raw/ and sagyam_raw/
IMAGE_DATASET_FOLDER = DATA_FOLDER

OUTPUT_FOLDER = DATA_FOLDER / "image_generated"
BANK_CACHE = DATA_FOLDER / "image_symbol_bank.pkl"

# Dataset folder name for each of our tokens (None = not in that dataset)
SOURCES = {
    "xainano": {
        "folder": "Xainano_Raw/dataset",
        "tokens": {
            **{digit: digit for digit in "0123456789"},
            "+": "plus",
            "-": "minus",
            "=": "equals",
            "(": "lparen",
            ")": "rparen",
            "\\times": "multiply",
            "\\div": "divide",
        },
    },
    "sagyam": {
        "folder": "sagyam_raw/dataset",
        "tokens": {
            **{digit: digit for digit in "0123456789"},
            "+": "add",
            "-": "sub",
            "=": "eq",
            "\\times": "mul",
            "\\div": "div",
        },
    },
}

# -------------------------
# Settings
# -------------------------

# Cap per symbol per dataset (keeps memory use and the symbol mix balanced)
MAX_IMAGES_PER_SYMBOL = 4000

# Share of each symbol's images held back for the image-based test set
TEST_FRACTION = 0.15

SPLIT_SEED = 42

SAMPLES_PER_SPLIT = {
    "train": 30000,
    "test": 3000,
}

# Chance that an equation uses sagyam (marker) symbols instead of Xainano.
# Symbols sagyam doesn't have (brackets) always come from Xainano.
SAGYAM_PROBABILITY = 0.3

# Height (pixels) of a typical digit on the working canvas
CANVAS_NUMBER_HEIGHT = 64

# Pen width as a fraction of digit height (log-uniform between these), like
# the ink pipeline where the pen is fixed relative to the writing - so long
# equations, which get shrunk more to fit, end up with thinner lines.
MIN_PEN_TO_DIGIT_HEIGHT = 0.04
MAX_PEN_TO_DIGIT_HEIGHT = 0.18

# Never thinner than this in the final 128x512 image (pixels)
MIN_FINAL_STROKE_WIDTH = 1.5

# Blank border around the equation, as a fraction of digit height
PADDING_RATIO = 0.3


# -------------------------
# Cleaning
# -------------------------

"""
Zhang-Suen thinning: reduces strokes in a binary image to 1-pixel-wide lines.

Used to give symbols from different image datasets (thin pen vs thick marker)
the same starting thickness before a random pen width is applied.
"""

import numpy as np


def thin(mask):
    """
    Parameters
    ----------
    mask : np.ndarray (H, W)
        Non-zero where there is ink.

    Returns
    -------
    np.ndarray (H, W) uint8
        1 on the 1-pixel-wide centre line of each stroke, 0 elsewhere.
    """

    image = np.pad((mask > 0).astype(np.uint8), 1)

    while True:
        changed = False

        for step in (0, 1):
            # Neighbours, clockwise from the pixel above (P2) to above-left (P9)
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]

            centre = image[1:-1, 1:-1]

            neighbours = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(n.astype(np.int16) for n in neighbours)

            # Number of 0 -> 1 changes going round the neighbours
            transitions = sum(
                ((neighbours[i] == 0) & (neighbours[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8)
            )

            if step == 0:
                side = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                side = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            remove = (centre == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & side

            if remove.any():
                centre[remove] = 0
                changed = True

        if not changed:
            break

    return image[1:-1, 1:-1]


def load_grayscale(path):
    image = Image.open(path)

    # Transparent images: put them on white first
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

    return np.asarray(image.convert("L"), dtype=np.uint8)


def clean_symbol(path, token, source):
    """
    Returns the symbol as a cropped, 1-pixel-wide binary mask (uint8 0/1),
    or None if the image should be skipped.
    """

    gray = load_grayscale(path)

    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = ink.shape

    if ink.sum() == 0 or ink.mean() > 0.5:
        return None

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    keep = np.zeros(count, dtype=bool)

    min_area = 2 if source == "xainano" else max(3, int(0.0002 * height * width))

    for index in range(1, count):
        x, y, w, h, area = stats[index]

        if area < min_area:
            continue

        # sagyam: drop ruler/border lines - long thin marks touching the edge
        if source == "sagyam":
            touches_edge = x == 0 or y == 0 or x + w == width or y + h == height
            long_and_thin = max(w, h) > 0.5 * max(width, height) and min(w, h) < 0.12 * max(w, h)

            if touches_edge and long_and_thin:
                continue

        keep[index] = True

    if not keep.any():
        return None

    ink = keep[labels].astype(np.uint8)

    ys, xs = np.nonzero(ink)
    ink = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    pieces = cv2.connectedComponents(ink, connectivity=8)[0] - 1

    # Xainano's "divide" folder also contains "/" slashes; a real ÷ has 3 parts
    if token == "\\div" and pieces < 3:
        return None

    # "=" is two separate lines
    if token == "=" and pieces < 2:
        return None

    return thin(ink)


# -------------------------
# Symbol bank
# -------------------------


def build_symbol_bank():
    """
    Cleans every image once and caches the result.

    Returns
    -------
    dict: split -> source -> token -> list of masks
    """

    if BANK_CACHE.exists():
        with open(BANK_CACHE, "rb") as file:
            return pickle.load(file)

    rng = random.Random(SPLIT_SEED)

    bank = {"train": {}, "test": {}}

    for source, info in SOURCES.items():
        for split in bank:
            bank[split][source] = {}

        for token, folder_name in info["tokens"].items():
            folder = IMAGE_DATASET_FOLDER / info["folder"] / folder_name

            files = sorted(path for path in folder.iterdir() if path.is_file() and not path.name.startswith("."))
            rng.shuffle(files)
            files = files[:MAX_IMAGES_PER_SYMBOL]

            masks = []
            skipped = 0

            for path in files:
                try:
                    mask = clean_symbol(path, token, source)
                except OSError:
                    mask = None

                if mask is None:
                    skipped += 1
                else:
                    masks.append(mask)

            test_count = max(1, round(len(masks) * TEST_FRACTION))

            bank["test"][source][token] = masks[:test_count]
            bank["train"][source][token] = masks[test_count:]

            print(
                f"  {source:>7} {token:>7}: {len(masks):5d} kept "
                f"({len(masks) - test_count} train / {test_count} test), {skipped} skipped",
                flush=True,
            )

    BANK_CACHE.parent.mkdir(parents=True, exist_ok=True)

    with open(BANK_CACHE, "wb") as file:
        pickle.dump(bank, file)

    return bank


def symbol_choices(bank_split, source):
    """
    Maps each token to the list the ink generator's add_symbol() picks from.
    Each entry is (source, index). Tokens the source lacks come from Xainano.
    """

    choices = {}

    for token in TOKENS:
        if token == BLANK_TOKEN:
            continue

        from_source = source if token in bank_split[source] else "xainano"
        choices[token] = [(from_source, index) for index in range(len(bank_split[from_source][token]))]

    return choices


# -------------------------
# Equation assembly
# -------------------------


def resize_mask(mask, width, height):
    """Resizes a thin binary mask without breaking its lines."""

    width = max(1, int(round(width)))
    height = max(1, int(round(height)))

    shrinking = width < mask.shape[1] or height < mask.shape[0]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR

    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=interpolation)

    return (resized > 0.15).astype(np.uint8)


def slant_and_rotate(canvas):
    """Same random shear and rotation as the ink generator, applied to an image."""

    angle = math.radians(random.uniform(-ink_generator.MAX_ROTATION_DEGREES, ink_generator.MAX_ROTATION_DEGREES))
    shear = random.uniform(-ink_generator.MAX_SHEAR, ink_generator.MAX_SHEAR)

    height, width = canvas.shape

    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    # Shear then rotate (as in the ink generator), about the canvas centre
    matrix = np.array(
        [
            [cos_angle, cos_angle * shear - sin_angle],
            [sin_angle, sin_angle * shear + cos_angle],
        ]
    )

    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float64)
    moved = corners @ matrix.T

    min_xy = moved.min(axis=0)
    max_xy = moved.max(axis=0)

    affine = np.hstack([matrix, -min_xy.reshape(2, 1)])
    new_width = int(math.ceil(max_xy[0] - min_xy[0]))
    new_height = int(math.ceil(max_xy[1] - min_xy[1]))

    warped = cv2.warpAffine(canvas.astype(np.float32), affine, (new_width, new_height), flags=cv2.INTER_LINEAR)

    return (warped > 0.15).astype(np.uint8)


def compose_equation(selected_symbols, bank_split):
    """
    Builds one equation image from symbols chosen by the ink generator's
    generate_equation_file_list(). Returns (uint8 image 128x512, label).
    """

    prepared = []

    for symbol in selected_symbols:
        source, index = symbol["file_path"]
        mask = bank_split[source][symbol["token"]][index]

        prepared.append(
            {
                **symbol,
                "mask": mask,
                "width": mask.shape[1],
                "height": mask.shape[0],
                "scale_factor": -1,
            }
        )

    positions = real_handwriting_layout(prepared)

    canvas = paste_symbols(prepared, positions)

    return finish_equation_image(canvas), ink_generator.build_equation_label(selected_symbols)


def real_handwriting_layout(prepared):
    """
    Sizes, vertical positions and gaps sampled from real handwriting (the
    ink generator's plan_*_from_real_handwriting functions). Resizes each
    mask and returns (left, top) for each symbol.
    """

    prepared = ink_generator.plan_sizes_from_real_handwriting(prepared, CANVAS_NUMBER_HEIGHT)

    for symbol in prepared:
        scale = symbol["scale_factor"]
        symbol["mask"] = resize_mask(symbol["mask"], symbol["width"] * scale, symbol["height"] * scale)
        symbol["height"], symbol["width"] = symbol["mask"].shape

    return ink_generator.plan_positions_from_real_handwriting(prepared, CANVAS_NUMBER_HEIGHT)


def paste_symbols(prepared, positions):
    """Pastes every resized symbol mask onto one canvas at its (left, top)."""

    padding = int(PADDING_RATIO * CANVAS_NUMBER_HEIGHT)

    min_left = min(left for left, _ in positions)
    min_top = min(top for _, top in positions)
    max_right = max(left + s["width"] for (left, _), s in zip(positions, prepared))
    max_bottom = max(top + s["height"] for (_, top), s in zip(positions, prepared))

    canvas_width = int(math.ceil(max_right - min_left)) + 2 * padding
    canvas_height = int(math.ceil(max_bottom - min_top)) + 2 * padding

    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    for (left, top), symbol in zip(positions, prepared):
        x = int(round(left - min_left)) + padding
        y = int(round(top - min_top)) + padding
        h, w = symbol["mask"].shape
        canvas[y : y + h, x : x + w] = np.maximum(canvas[y : y + h, x : x + w], symbol["mask"])

    return canvas


def finish_equation_image(canvas):
    """Applies a random pen width and fits the canvas into a 128x512 image."""

    canvas = slant_and_rotate(canvas)

    # Pen width relative to digit height, but never so thin that it vanishes
    # once the canvas is shrunk to 128x512
    final_scale = min(IMAGE_WIDTH / canvas.shape[1], IMAGE_HEIGHT / canvas.shape[0])
    pen_ratio = math.exp(random.uniform(math.log(MIN_PEN_TO_DIGIT_HEIGHT), math.log(MAX_PEN_TO_DIGIT_HEIGHT)))
    pen_width = max(pen_ratio * CANVAS_NUMBER_HEIGHT, MIN_FINAL_STROKE_WIDTH / final_scale)
    pen = max(1, int(round(pen_width)))

    if pen > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pen, pen))
        canvas = cv2.dilate(canvas, kernel)

    # Black ink on white, then the same fit-and-centre as the ink pipeline
    image = np.where(canvas > 0, 0, 255).astype(np.uint8)
    return resize_image(image, (IMAGE_HEIGHT, IMAGE_WIDTH))


def generate_split(bank, split, count, output_folder):
    bank_split = bank[split]

    choices = {source: symbol_choices(bank_split, source) for source in SOURCES}

    generation_stats = {
        "total_files": 0,
        "token_counts": {token: 0 for token in TOKENS if token != BLANK_TOKEN},
    }

    output_folder.mkdir(parents=True, exist_ok=True)

    X = np.lib.format.open_memmap(
        output_folder / "X.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(count, IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    labels = []

    for index in range(count):
        source = "sagyam" if random.random() < SAGYAM_PROBABILITY else "xainano"

        selected, generation_stats = ink_generator.generate_equation_file_list(
            choices[source],
            generation_stats,
        )

        X[index], label = compose_equation(selected, bank_split)
        labels.append(label)

        if (index + 1) % 5000 == 0:
            print(f"  {split}: {index + 1}/{count}", flush=True)

    X.flush()
    del X

    np.save(output_folder / "y.npy", np.array(labels))

    print(f"Saved {count} {split} equations to {output_folder}")


def main():
    random.seed(SPLIT_SEED)

    print("Cleaning symbol images (cached after the first run)...")
    bank = build_symbol_bank()

    for split, count in SAMPLES_PER_SPLIT.items():
        generate_split(bank, split, count, OUTPUT_FOLDER / split)


if __name__ == "__main__":
    main()

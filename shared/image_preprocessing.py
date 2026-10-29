"""
Turns an uploaded image (phone photo, scan, screenshot or canvas drawing)
into the 128x512 black-on-white image the model expects.

Shared by the API (api/app/services/preprocessing.py) and the training data
pipeline (ml/data_pipeline/photo_style.py), so the model can be trained on
exactly what it will see in use.

Steps (preprocess_bytes / preprocess_gray):
  1. load as greyscale; transparent areas get the paper colour (photos) or
     white (canvas drawings)
  2. flip light-on-dark images
  3. paint over near-black borders at the photo's edge
  4. even out the paper's brightness, then one automatic threshold
  5. remove ruled lines and specks
  6. crop to the writing (ignoring stray marks) and fit into 128x512
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from shared.config import IMAGE_HEIGHT, IMAGE_WIDTH


class PreprocessingError(ValueError):
    """The image can't be turned into model input (e.g. no handwriting found)."""


# Ink blobs smaller than this fraction of the image area are treated as noise
MIN_COMPONENT_AREA_RATIO = 0.00005

# Near-black areas (below this grey level) that touch the photo's edge and
# cover at least this fraction of it are treated as borders, not ink
DARK_BORDER_LEVEL = 50
DARK_BORDER_MIN_AREA_RATIO = 0.005

# Grey levels a pixel must differ from the background by to count as writing
# when deciding whether the image is light-on-dark
INK_CONTRAST = 40

# Size of the paper-brightness filter, as a fraction of the image's shorter side
BACKGROUND_KERNEL_RATIO = 0.1

# Ruled lines: horizontal runs spanning at least this fraction of the photo
# width are removed (ruled lines cross the whole page; a handwritten stroke,
# even the bar of a lone "7" in a narrow photo, does not). Gaps up to
# RULED_LINE_BRIDGE_RATIO of the width are bridged first (faint lines turn
# into dashes after thresholding).
RULED_LINE_MIN_WIDTH_RATIO = 0.7
RULED_LINE_BRIDGE_RATIO = 0.02

# Ink blobs at least this fraction of the largest blob's area define the
# writing; smaller marks are only kept if they are near it (see writing_box)
MAJOR_BLOB_AREA_RATIO = 0.15
# Small marks are kept within this many writing-heights of the writing
NEARBY_MARK_REACH_SIDEWAYS = 3.0
NEARBY_MARK_REACH_VERTICAL = 0.5

# Training images have a small white border around tightly cropped ink.
# Recreate that here (as a fraction of the cropped ink height).
CROP_PADDING_RATIO = 0.10


def resize_image(image: np.ndarray) -> np.ndarray:
    target_h = IMAGE_HEIGHT
    target_w = IMAGE_WIDTH

    pil_image = Image.fromarray(image.astype(np.uint8))

    original_w, original_h = pil_image.size

    scale = min(
        target_w / original_w,
        target_h / original_h,
    )

    new_w = round(original_w * scale)
    new_h = round(original_h * scale)

    if new_w == 0 or new_h == 0:
        raise PreprocessingError("Image aspect ratio is too extreme to preprocess.")

    resized = pil_image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        mode="L",
        size=(target_w, target_h),
        color=255,
    )

    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2

    canvas.paste(resized, (left, top))

    return np.asarray(canvas, dtype=np.uint8)


def load_grayscale(contents: bytes) -> Image.Image:
    pil_image = Image.open(BytesIO(contents))

    # Transparent areas: convert("L") would turn them black, so fill them first.
    #  - Canvas drawings (mostly transparent, opaque strokes): fill with white.
    #  - Photos with transparent corners from rotating/cropping (mostly
    #    opaque): fill with the photo's own paper colour. White would leave a
    #    hard edge that thresholding turns into a bar the model reads as "1".
    if pil_image.mode in ("RGBA", "LA") or "transparency" in pil_image.info:
        rgba = np.asarray(pil_image.convert("RGBA"))
        gray = np.asarray(Image.fromarray(rgba[..., :3]).convert("L"), dtype=np.float32)
        alpha = rgba[..., 3].astype(np.float32) / 255

        opaque = alpha > 0.99

        if opaque.mean() > 0.5:
            fill = float(np.median(gray[opaque]))
        else:
            fill = 255.0

        blended = gray * alpha + fill * (1 - alpha)

        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    return pil_image.convert("L")


def fill_dark_borders(gray: np.ndarray) -> np.ndarray:
    """
    Paints over near-black areas touching the edge of a photo (left behind by
    cropping or rotating) with the paper colour. Otherwise their inner edge
    survives thresholding as a long bar that the model reads as a "1".
    Ink strokes are thin and don't touch the edge, so they are kept.
    """

    dark = (gray < DARK_BORDER_LEVEL).astype(np.uint8)

    if not dark.any():
        return gray

    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)

    height, width = gray.shape
    min_area = DARK_BORDER_MIN_AREA_RATIO * gray.size
    border = np.zeros(count, dtype=bool)

    for index in range(1, count):
        x, y, w, h, area = stats[index]
        touches_edge = x == 0 or y == 0 or x + w == width or y + h == height

        if touches_edge and area >= min_area:
            border[index] = True

    if not border.any():
        return gray

    mask = border[labels]

    # Grow slightly so the blurred edge of the band goes too
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    paper = np.median(gray[~mask]) if (~mask).any() else 255

    cleaned = gray.copy()
    cleaned[mask] = paper

    return cleaned


def is_light_on_dark(gray: np.ndarray) -> bool:
    """
    True when the writing is lighter than its background.

    The background is the typical (median) colour. Writing is whatever
    differs clearly from it: if those pixels are mostly brighter, it is light
    writing on a dark background. Unlike a fixed brightness cut-off, this
    isn't fooled by dim grey paper, and a black band at the edge of a photo
    only adds darker pixels, so it can't trigger a false inversion.
    """

    background = float(np.median(gray))
    brighter = int((gray > background + INK_CONTRAST).sum())
    darker = int((gray < background - INK_CONTRAST).sum())

    return brighter > darker


def _flatten_and_threshold(blurred: np.ndarray, size: int) -> np.ndarray:
    size += 1 - size % 2  # odd

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.dilate(blurred, kernel)
    background = cv2.blur(background, (size, size))

    flattened = np.clip(blurred.astype(np.float32) / np.maximum(background, 1) * 255, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def binarise(gray: np.ndarray) -> np.ndarray:
    """
    Black ink on white from a photo or drawing.

    The paper's brightness is estimated everywhere (a large max-filter erases
    the ink, then it is smoothed) and divided out, so dim paper, shadows and
    uneven light all become evenly white. One automatic (Otsu) threshold then
    separates ink from paper.

    The filter must be several stroke widths wide or thick strokes come out
    hollow. Stroke width is measured on a first pass with the standard filter
    size; if the strokes are thicker than that allows, a second pass uses a
    larger filter (capped at half the image's shorter side).
    """

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    height, width = gray.shape
    shorter_side = min(height, width)

    size = int(max(15, BACKGROUND_KERNEL_RATIO * shorter_side))
    binary = _flatten_and_threshold(blurred, size)

    ink = (binary == 0).astype(np.uint8)

    if ink.any():
        distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
        stroke_width = 2 * float(np.percentile(distance[ink > 0], 95))
        needed = int(min(3 * stroke_width, 0.5 * shorter_side))

        if needed > size:
            binary = _flatten_and_threshold(blurred, needed)

    return binary


def remove_lines_and_specks(binary: np.ndarray) -> np.ndarray:
    """
    Cleans a thresholded photo (black ink on white):
      - removes ruled/printed lines: horizontal ink runs longer than
        RULED_LINE_MIN_WIDTH_RATIO of the image width - far longer than any
        handwritten minus sign. Faint lines often break into dashes after
        thresholding, so small breaks are bridged first.
      - removes specks (paper texture, sensor noise) smaller than
        MIN_COMPONENT_AREA_RATIO of the image, which the model could otherwise
        read as the dots of a division sign.
    """

    ink = (binary < 128).astype(np.uint8)
    height, width = ink.shape

    bridge = max(5, int(width * RULED_LINE_BRIDGE_RATIO))
    bridged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, bridge), np.uint8))

    line_length = max(40, int(width * RULED_LINE_MIN_WIDTH_RATIO))
    lines = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, np.ones((1, line_length), np.uint8))

    if lines.any():
        # Cover the line's full thickness, then erase it
        lines = cv2.dilate(lines, np.ones((3, 1), np.uint8))
        ink[lines > 0] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(4, binary.size * MIN_COMPONENT_AREA_RATIO)
    small = np.nonzero(stats[:, cv2.CC_STAT_AREA] < min_area)[0]
    small = small[small != 0]

    if len(small):
        ink[np.isin(labels, small)] = 0

    return np.where(ink > 0, 0, 255).astype(np.uint8)


def writing_box(boxes) -> tuple:
    """
    Bounding box (left, top, right, bottom) of the handwriting.

    Starts from the substantial ink blobs (digits, brackets, operators), then
    adds smaller marks only if they are near the writing, so the dots of a
    "÷" and a widely spaced "=" are kept but a stray speck or pen mark
    elsewhere on the photo doesn't stretch the crop. "Near" allows more room
    beside the writing (symbols can be spaced out) than above or below it
    (where only specks and marks are found).
    """

    largest = max(box[cv2.CC_STAT_AREA] for box in boxes)

    def edges(box):
        x, y, w, h = box[cv2.CC_STAT_LEFT], box[cv2.CC_STAT_TOP], box[cv2.CC_STAT_WIDTH], box[cv2.CC_STAT_HEIGHT]
        return x, y, x + w, y + h

    included = [box for box in boxes if box[cv2.CC_STAT_AREA] >= MAJOR_BLOB_AREA_RATIO * largest]
    others = [box for box in boxes if box[cv2.CC_STAT_AREA] < MAJOR_BLOB_AREA_RATIO * largest]

    left = min(edges(b)[0] for b in included)
    top = min(edges(b)[1] for b in included)
    right = max(edges(b)[2] for b in included)
    bottom = max(edges(b)[3] for b in included)

    added = True
    while added and others:
        added = False
        reach_x = NEARBY_MARK_REACH_SIDEWAYS * (bottom - top)
        reach_y = NEARBY_MARK_REACH_VERTICAL * (bottom - top)

        remaining = []

        for box in others:
            x0, y0, x1, y1 = edges(box)
            gap_x = max(0, left - x1, x0 - right)
            gap_y = max(0, top - y1, y0 - bottom)

            if gap_x <= reach_x and gap_y <= reach_y:
                left, top, right, bottom = min(left, x0), min(top, y0), max(right, x1), max(bottom, y1)
                added = True
            else:
                remaining.append(box)

        others = remaining

    return left, top, right, bottom


def crop_to_ink(binary: np.ndarray) -> np.ndarray:
    """
    Crops a black-on-white image to the handwriting, ignoring small specks,
    and adds a small white border - the same framing the training images have.
    Without this, any empty space around the equation in the upload makes the
    symbols much smaller than anything the model was trained on.
    """

    ink = (binary < 128).astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    min_area = max(4, binary.size * MIN_COMPONENT_AREA_RATIO)

    boxes = [stats[index] for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= min_area]

    if not boxes:
        raise PreprocessingError("No handwriting was found in the image.")

    left, top, right, bottom = writing_box(boxes)

    cropped = binary[top:bottom, left:right]

    padding = max(2, int(round((bottom - top) * CROP_PADDING_RATIO)))

    return cv2.copyMakeBorder(
        cropped,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    """Greyscale uint8 image (any size) -> 128x512 uint8, black ink on white."""

    image = np.asarray(gray, dtype=np.uint8)

    # Light writing on a dark background (e.g. a dark-mode canvas): flip it
    # so it matches the black-on-white training images
    if is_light_on_dark(image):
        image = 255 - image

    image = fill_dark_borders(image)
    image = binarise(image)
    image = remove_lines_and_specks(image)
    image = crop_to_ink(image)

    return resize_image(image)


def preprocess_bytes(contents: bytes) -> np.ndarray:
    """Encoded image file (PNG/JPG/...) -> 128x512 uint8, black ink on white."""

    return preprocess_gray(np.asarray(load_grayscale(contents), dtype=np.uint8))

"""
Simulated phone photos of handwriting.

Draws ink strokes onto a "photo" of paper: off-white paper with uneven
lighting and texture, sometimes faint ruled lines, blue, black or grey ink,
random pen width, the writing placed anywhere on a larger page, slight
rotation and perspective, blur and JPEG compression.

Used for the photo-like test set (api/tests/make_photo_test_set.py) and for
photo-style training images (photo_style.py).
"""

import math

import cv2
import numpy as np

# Writing height on the page, in pixels (before the photo is resized)
MIN_WRITING_HEIGHT = 80
MAX_WRITING_HEIGHT = 220

RULED_LINE_PROBABILITY = 0.3
INK_COLOURS = [(30, 30, 30), (110, 40, 20), (60, 60, 60), (60, 20, 15)]  # BGR: black, blue, grey, navy

JPEG_QUALITY_RANGE = (60, 92)


def paper(width, height, rng):
    """Off-white paper with uneven lighting, texture and maybe ruled lines (BGR float)."""

    base = np.array([rng.uniform(215, 250), rng.uniform(220, 252), rng.uniform(225, 255)])
    page = np.ones((height, width, 3)) * base

    # Lighting: a gentle gradient across the page, darker on one side
    ys, xs = np.mgrid[0:height, 0:width]
    angle = rng.uniform(0, 2 * math.pi)
    direction = (xs * math.cos(angle) + ys * math.sin(angle)) / max(width, height)
    lighting = 1 - rng.uniform(0.05, 0.30) * (direction - direction.min()) / max(1e-6, np.ptp(direction))
    page *= lighting[..., None]

    # Paper texture / sensor noise
    page += rng.normal(0, rng.uniform(2, 6), page.shape)

    if rng.random() < RULED_LINE_PROBABILITY:
        spacing = int(rng.uniform(40, 90))
        colour = np.array([rng.uniform(190, 230), rng.uniform(170, 210), rng.uniform(140, 190)])  # faint blue

        for y in range(int(rng.uniform(0, spacing)), height, spacing):
            page[y : y + 2] = page[y : y + 2] * 0.5 + colour * 0.5

    return page


def render_photo(strokes, rng):
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]

    ink_width = max(xs) - min(xs)
    ink_height = max(ys) - min(ys)

    writing_height = rng.uniform(MIN_WRITING_HEIGHT, MAX_WRITING_HEIGHT)
    scale = writing_height / max(ink_height, 1e-6)

    # Very wide equations: keep the page a sensible size
    if ink_width * scale > 1800:
        scale = 1800 / ink_width

    writing_w = ink_width * scale
    writing_h = ink_height * scale

    # Page with random margins, writing placed anywhere on it
    margin_x = rng.uniform(0.1, 0.6) * writing_w + 40
    margin_y = rng.uniform(0.3, 2.0) * writing_h + 40
    width = int(writing_w + 2 * margin_x)
    height = int(writing_h + 2 * margin_y)

    offset_x = rng.uniform(0.3, 1.7) * margin_x
    offset_y = rng.uniform(0.3, 1.7) * margin_y

    page = paper(width, height, rng)

    pen = max(1, int(round(writing_h * rng.uniform(0.02, 0.06))))
    colour = np.array(INK_COLOURS[rng.integers(len(INK_COLOURS))], dtype=np.float64)
    strength = rng.uniform(0.85, 1.0)

    ink = np.zeros((height, width), dtype=np.uint8)

    for stroke in strokes:
        points = np.array(
            [[(x - min(xs)) * scale + offset_x, (y - min(ys)) * scale + offset_y] for x, y in stroke],
            dtype=np.int32,
        )

        if len(points) == 1:
            cv2.circle(ink, tuple(points[0]), max(1, pen // 2), 255, -1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(ink, [points], False, 255, pen, lineType=cv2.LINE_AA)

    alpha = (ink.astype(np.float64) / 255 * strength)[..., None]
    photo = page * (1 - alpha) + colour * alpha

    # Small rotation and perspective, as if the phone wasn't held quite straight
    jitter = 0.03 * min(width, height)
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = source + rng.uniform(-jitter, jitter, source.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    centre = (width / 2, height / 2)
    rotation = np.vstack([cv2.getRotationMatrix2D(centre, rng.uniform(-5, 5), 1.0), [0, 0, 1]])
    photo = cv2.warpPerspective(
        photo.astype(np.float32),
        (rotation @ matrix).astype(np.float64),
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )

    photo = cv2.GaussianBlur(photo, (0, 0), rng.uniform(0.4, 1.4))

    return np.clip(photo, 0, 255).astype(np.uint8)


def jpeg_round_trip(photo, rng):
    """Compresses and decompresses a photo as JPEG, like a phone camera would."""

    quality = int(rng.uniform(*JPEG_QUALITY_RANGE))
    _, encoded = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, quality])

    return encoded, quality

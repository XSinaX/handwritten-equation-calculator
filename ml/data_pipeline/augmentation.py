"""
Random image augmentation applied on the fly during training.

The generated images are very clean: tightly cropped, centred, one pen width,
no noise. Photos and drawings sent to the API are not. These augmentations
make every epoch look slightly different so the model learns the symbols
rather than the exact pixels of the training images.

Images are uint8, shape (IMAGE_HEIGHT, IMAGE_WIDTH), black ink on white.
"""

import random

import cv2
import numpy as np

# Probability of each augmentation being applied
THICKNESS_PROBABILITY = 0.5
PLACEMENT_PROBABILITY = 0.7
ROTATION_PROBABILITY = 0.3
BLUR_PROBABILITY = 0.3
NOISE_PROBABILITY = 0.3

# How small the equation may be shrunk inside the canvas (1.0 = fills it)
MIN_CONTENT_SCALE = 0.6

MAX_ROTATION_DEGREES = 3
MAX_NOISE_STD = 12


def change_thickness(ink):
    kernel = np.ones((2, 2), np.uint8)

    if random.random() < 0.5:
        return cv2.dilate(ink, kernel, iterations=random.randint(1, 2))

    thinner = cv2.erode(ink, kernel, iterations=1)

    # Don't let erosion wipe out thin strokes completely
    if thinner.sum() < 0.5 * ink.sum():
        return ink

    return thinner


def random_placement(ink):
    """
    Shrinks the equation by a random amount and puts it at a random position,
    so the model doesn't depend on content always being centred and filling
    the canvas.
    """

    height, width = ink.shape

    ys, xs = np.nonzero(ink > 32)

    if len(xs) == 0:
        return ink

    content = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    content_h, content_w = content.shape

    fit_scale = min(width / content_w, height / content_h)
    scale = fit_scale * random.uniform(MIN_CONTENT_SCALE, 1.0)

    new_w = max(1, min(width, int(round(content_w * scale))))
    new_h = max(1, min(height, int(round(content_h * scale))))

    resized = cv2.resize(content, (new_w, new_h), interpolation=cv2.INTER_AREA)

    left = random.randint(0, width - new_w)
    top = random.randint(0, height - new_h)

    canvas = np.zeros_like(ink)
    canvas[top : top + new_h, left : left + new_w] = resized

    return canvas


def rotate(ink):
    height, width = ink.shape

    angle = random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

    return cv2.warpAffine(ink, matrix, (width, height), borderValue=0)


def augment_image(image):
    """
    Returns a randomly augmented copy of a uint8 black-on-white image.
    """

    # Work with ink = bright so empty space is 0 for the transforms
    ink = 255 - image

    if random.random() < THICKNESS_PROBABILITY:
        ink = change_thickness(ink)

    if random.random() < PLACEMENT_PROBABILITY:
        ink = random_placement(ink)

    if random.random() < ROTATION_PROBABILITY:
        ink = rotate(ink)

    if random.random() < BLUR_PROBABILITY:
        ink = cv2.GaussianBlur(ink, (3, 3), 0)

    image = 255 - ink

    if random.random() < NOISE_PROBABILITY:
        noise = np.random.normal(0, random.uniform(0, MAX_NOISE_STD), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image

import cv2
import numpy as np

from shared.config import IMAGE_HEIGHT

EQUATION_PADDING = 5
BACKGROUND_KERNEL_RATIO = 0.10
DARK_BORDER_LEVEL = 50
DARK_BORDER_MIN_AREA_RATIO = 0.005
TINY_WIDTH_RATIO = 0.30
TINY_HEIGHT_RATIO = 0.30
MAX_NOISE_DISTANCE_RATIO = 0.70


def image_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale, converting transparent pixels to white."""
    if image.ndim == 2:
        return image.astype(np.uint8)

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected a grayscale, RGB, or RGBA image array.")

    if image.shape[2] == 4:
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    else:
        rgb = image.astype(np.uint8)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def remove_dark_borders(grayscale: np.ndarray) -> np.ndarray:
    """Paint over substantial near-black regions touching the image edges."""
    dark = (grayscale < DARK_BORDER_LEVEL).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    height, width = grayscale.shape
    border_ids = []

    for component_id in range(1, count):
        x, y, w, h, area = stats[component_id]
        touches_edge = x == 0 or y == 0 or x + w == width or y + h == height
        if touches_edge and area >= DARK_BORDER_MIN_AREA_RATIO * grayscale.size:
            border_ids.append(component_id)

    if not border_ids:
        return grayscale.copy()

    mask = np.isin(labels, border_ids).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8)) > 0
    paper_color = np.median(grayscale[~mask]) if (~mask).any() else 255

    cleaned = grayscale.copy()
    cleaned[mask] = paper_color
    return cleaned


def binarize_image(grayscale: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting with a morphological background and apply Otsu."""
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    kernel_size = max(15, int(min(grayscale.shape) * BACKGROUND_KERNEL_RATIO))
    kernel_size |= 1  # Morphology and blur use an odd-sized kernel.

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.dilate(blurred, kernel)
    background = cv2.blur(background, (kernel_size, kernel_size))

    normalized = cv2.divide(blurred, background, scale=255)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def remove_isolated_noise(binary: np.ndarray) -> np.ndarray:
    """Remove tiny, distant noise from the binary image."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((binary == 0).astype(np.uint8), connectivity=8)
    if count <= 2:
        return binary.copy()

    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    reference_width = float(np.percentile(widths, 75))
    reference_height = float(np.percentile(heights, 75))

    tiny = (widths < TINY_WIDTH_RATIO * reference_width) & (heights < TINY_HEIGHT_RATIO * reference_height)
    tiny_ids = np.flatnonzero(tiny) + 1
    anchor_ids = np.flatnonzero(~tiny) + 1
    if not len(tiny_ids) or not len(anchor_ids):
        return binary.copy()

    # For each pixel, measure distance to the nearest pixel in a large component.
    anchor_mask = np.ones(binary.shape, dtype=np.uint8)
    anchor_mask[np.isin(labels, anchor_ids)] = 0
    distances = cv2.distanceTransform(anchor_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    max_distance = max(2.0, MAX_NOISE_DISTANCE_RATIO * reference_height)
    cleaned = binary.copy()
    for component_id in tiny_ids:
        x, y, w, h, _ = stats[component_id]
        component = labels[y : y + h, x : x + w] == component_id
        closest = float(distances[y : y + h, x : x + w][component].min())
        if closest > max_distance:
            cleaned[y : y + h, x : x + w][component] = 255

    return cleaned


def crop_to_ink(binary: np.ndarray) -> np.ndarray:
    """Crop to the smallest rectangle containing all remaining black pixels."""
    points = cv2.findNonZero(255 - binary)
    if points is None:
        raise ValueError("No handwriting was found in the image.")

    x, y, width, height = cv2.boundingRect(points)
    return binary[y : y + height, x : x + width]


def resize_and_pad(binary: np.ndarray) -> np.ndarray:
    """Fit content to config height, preserve aspect ratio, then add 5px padding."""
    height, width = binary.shape
    content_height = IMAGE_HEIGHT - 2 * EQUATION_PADDING
    scale = content_height / height
    content_width = max(1, round(width * scale))

    resized = cv2.resize(
        binary,
        (content_width, content_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    return cv2.copyMakeBorder(
        resized,
        EQUATION_PADDING,
        EQUATION_PADDING,
        EQUATION_PADDING,
        EQUATION_PADDING,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Run the full preprocessing pipeline on a handwritten equation image."""
    grayscale = image_to_grayscale(image)
    grayscale = remove_dark_borders(grayscale)
    binary = binarize_image(grayscale)
    binary = remove_isolated_noise(binary)
    binary = crop_to_ink(binary)
    return resize_and_pad(binary)

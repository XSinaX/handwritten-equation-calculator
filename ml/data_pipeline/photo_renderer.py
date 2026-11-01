import math
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

# Colours are BGR because OpenCV uses BGR images.
INK_COLOURS = (
    (30, 30, 30),  # Black
    (110, 40, 20),  # Dark blue
    (60, 60, 60),  # Dark grey
    (60, 20, 15),  # Navy
)

WRITING_HEIGHT_RANGE = (120, 200)
JPEG_QUALITY_RANGE = (65, 95)
PAGE_MARGIN = 35


def read_strokes(inkml: str | Path | ET.Element) -> list[list[tuple[float, float]]]:
    """Read X/Y points from a generated InkML file or in-memory XML element."""
    if isinstance(inkml, ET.Element):
        root = inkml
    else:
        root = ET.parse(inkml).getroot()

    # Match InkML elements with or without an explicit XML namespace.
    strokes = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "trace":
            continue

        points = []
        for raw_point in (element.text or "").split(","):
            values = raw_point.strip().split()
            if not values:
                continue
            if len(values) < 2:
                raise ValueError("InkML trace point is missing its X/Y coordinates")
            points.append((float(values[0]), float(values[1])))

        if points:
            strokes.append(points)

    if not strokes:
        raise ValueError("InkML contains no handwriting strokes")

    return strokes


def make_paper(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Create off-white paper with subtle texture and uneven lighting.

    Deliberately excludes ruled lines: our dataset does not require them.
    """
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


def render_photo(
    inkml: str | Path | ET.Element,
    output_path: str | Path,
    line_width: int = 4,
    seed: int | None = None,
) -> Path:
    """Render InkML strokes as a photo-like PNG and return its saved path.

    `line_width` is the pen width *in rendered pixels*; the equation generator
    can choose it randomly. No model preprocessing, PNG label metadata, or CSV
    creation happens here.
    """
    if line_width < 1:
        raise ValueError("line_width must be at least 1 pixel")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("The output filename must end in .png")

    rng = np.random.default_rng(seed)
    strokes = read_strokes(inkml)

    x_coords = [x for stroke in strokes for x, _ in stroke]
    y_coords = [y for stroke in strokes for _, y in stroke]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Height is random (camera distance); width follows the aspect ratio.
    writing_height = rng.uniform(*WRITING_HEIGHT_RANGE)
    scale = writing_height / max(max_y - min_y, 1.0)
    width = max(1, math.ceil((max_x - min_x) * scale)) + 2 * PAGE_MARGIN
    height = max(1, math.ceil((max_y - min_y) * scale)) + 2 * PAGE_MARGIN

    paper = make_paper(width, height, rng)
    ink = np.zeros((height, width), dtype=np.uint8)

    for stroke in strokes:
        points = np.array(
            [[round((x - min_x) * scale) + PAGE_MARGIN, round((y - min_y) * scale) + PAGE_MARGIN] for x, y in stroke],
            dtype=np.int32,
        )

        # A dot may have a single point or several very closely spaced points.
        # Draw it as a circle so division-symbol dots do not disappear.
        extent = np.ptp(points, axis=0)
        if len(points) == 1 or max(extent) <= line_width:
            centre = tuple(np.round(points.mean(axis=0)).astype(int))
            cv2.circle(ink, centre, max(1, line_width // 2), 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(ink, [points], False, 255, line_width, cv2.LINE_AA)

    colour = np.array(INK_COLOURS[rng.integers(len(INK_COLOURS))], dtype=np.float32)
    ink_opacity = rng.uniform(0.88, 1.0)
    alpha = (ink.astype(np.float32) / 255 * ink_opacity)[..., None]
    photo = paper * (1 - alpha) + colour * alpha

    # Small camera perspective shift. Rotation is handled by the generator.
    jitter = 0.015 * min(width, height)
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    target = source + rng.uniform(-jitter, jitter, source.shape).astype(np.float32)
    perspective = cv2.getPerspectiveTransform(source, target)
    photo = cv2.warpPerspective(photo, perspective, (width, height), borderMode=cv2.BORDER_REPLICATE)

    photo = cv2.GaussianBlur(photo, (0, 0), sigmaX=rng.uniform(0.3, 1.0))
    photo = np.clip(photo, 0, 255).astype(np.uint8)

    # Reproduce mild phone JPEG artifacts, but SAVE the result as a PNG.
    quality = int(rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1))
    success, jpeg = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Failed to simulate JPEG compression")
    photo = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), photo):
        raise OSError(f"Could not save PNG: {output_path}")
    return output_path

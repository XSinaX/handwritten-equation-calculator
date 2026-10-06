from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import csv
from PIL import Image, ImageDraw
import random

import matplotlib.pyplot as plt

from shared.config import IMAGE_HEIGHT, IMAGE_WIDTH

MIN_LINE_WIDTH = 8
MAX_LINE_WIDTH = 14

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INK_FOLDER = PROJECT_ROOT / "ml" / "data" / "generated"


def render_ink(file_path, padding=20, line_width=10):
    """
    Render an InkML file as a grayscale NumPy array.

    Returns
    -------
    image : np.ndarray | None
        Grayscale image with shape (H, W), dtype uint8.
        None if no valid strokes could be extracted.

    target : str | None
        Value of the normalizedLabel annotation.

    log : str
        "OK" if no problems were found, otherwise contains issue messages.
    """

    tree = ET.parse(file_path)
    root = tree.getroot()

    log_messages = []
    strokes = []
    target = None

    # Extract target label
    for element in root.iter():
        if element.tag.endswith("annotation"):
            if element.attrib.get("type") == "normalizedLabel":
                target = element.text
                break

    if target is None:
        log_messages.append("Missing normalizedLabel.")

    # Extract strokes
    for element in root.iter():
        # Skip non-trace elements
        if not element.tag.endswith("trace"):
            continue

        trace_id = element.attrib.get("id", "unknown")

        if not element.text:
            log_messages.append(f"Trace {trace_id}: trace is empty.")
            continue

        points = []

        raw_points = element.text.strip().split(",")

        for point_index, raw_point in enumerate(raw_points):
            values = raw_point.strip().split()

            if len(values) < 2:
                log_messages.append(f"Trace {trace_id}, point {point_index}: " "missing X or Y dimension.")
                continue

            try:
                x = float(values[0])
                y = float(values[1])

            except ValueError:
                log_messages.append(f"Trace {trace_id}, point {point_index}: " "X or Y contains a non-numeric value.")
                continue

            points.append((x, y))

        if points:
            strokes.append(points)

    # Cannot render an image without valid strokes
    if not strokes:
        log_messages.append("No valid strokes found.")

        return None, target, "\n".join(log_messages)

    # Gather every point to determine the image boundaries
    all_points = []

    for stroke in strokes:
        for point in stroke:
            all_points.append(point)

    xs = []
    ys = []

    for point in all_points:
        x = point[0]
        y = point[1]

        xs.append(x)
        ys.append(y)

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    width = max(1, int(np.ceil(max_x - min_x)) + 2 * padding)
    height = max(1, int(np.ceil(max_y - min_y)) + 2 * padding)

    background = 255
    foreground = 0

    image = Image.new(
        mode="L",
        size=(width, height),
        color=background,
    )

    draw = ImageDraw.Draw(image)

    for stroke in strokes:

        shifted = []

        for x, y in stroke:
            shifted_x = int(round(x - min_x + padding))
            shifted_y = int(round(y - min_y + padding))

            shifted.append((shifted_x, shifted_y))

        unique_points = set(shifted)

        # Draw traces that collapse to one coordinate as a dot
        if len(unique_points) == 1:
            x, y = next(iter(unique_points))

            radius = max(1, line_width // 2)

            draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ),
                fill=foreground,
            )

        # Draw normal strokes
        else:
            draw.line(
                shifted,
                fill=foreground,
                width=line_width,
                joint="curve",
            )

    image = np.asarray(image, dtype=np.uint8)

    if not log_messages:
        log = "OK"
    else:
        log = "\n".join(log_messages)

    return image, target, log


def resize_image(
    image,
    target_size,
):
    """
    Resize a grayscale image while preserving its aspect ratio,
    then centre it on a fixed-size canvas.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W).
    target_size : tuple[int, int]
        Desired output size as (height, width).

    Returns
    -------
    np.ndarray
        Resized/padded image with shape target_size and dtype uint8.
    """

    target_h, target_w = target_size

    pil_image = Image.fromarray(image.astype(np.uint8))

    original_w, original_h = pil_image.size

    scale = min(
        target_w / original_w,
        target_h / original_h,
    )

    new_w = max(1, int(round(original_w * scale)))
    new_h = max(1, int(round(original_h * scale)))

    resized = pil_image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )

    background = 255
    canvas = Image.new(
        mode="L",
        size=(target_w, target_h),
        color=background,
    )

    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2

    canvas.paste(resized, (left, top))

    return np.asarray(canvas, dtype=np.uint8)


def preprocess_ink(path):
    line_width = random.randint(
        MIN_LINE_WIDTH,
        MAX_LINE_WIDTH,
    )
    image, target, log = render_ink(path, line_width=line_width)

    # Rendering failed completely
    if image is None:
        return None, target, log

    image = resize_image(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    return image.astype(np.uint8), target, log


ink_files = sorted(INK_FOLDER.glob("*.inkml"))

if not ink_files:
    print(f"No InkML files found in {INK_FOLDER}.")
    exit(1)

processed = []
targets = []
log_rows = []

for path in ink_files:
    try:
        image, target, log = preprocess_ink(path)

        # Only save the image if it was successfully processed
        if image is not None and log == "OK":
            processed.append(image)
            targets.append(target)

        # Record files that actually had an issue
        if log != "OK":
            log_rows.append(
                {
                    "filename": path.name,
                    "message": log,
                }
            )

    except Exception as error:
        log_rows.append(
            {
                "filename": path.name,
                "message": f"Preprocessing failed: {error}",
            }
        )

# -------------------------
# Visualize a few processed images for sanity check
# -------------------------
# if processed:

#     GRID_ROWS = 3
#     GRID_COLUMNS = 3
#     NUM_WINDOWS = 5

#     samples_per_window = GRID_ROWS * GRID_COLUMNS
#     num_samples = min(
#         samples_per_window * NUM_WINDOWS,
#         len(processed),
#     )

#     for start_index in range(
#         0,
#         num_samples,
#         samples_per_window,
#     ):

#         fig, axes = plt.subplots(
#             GRID_ROWS,
#             GRID_COLUMNS,
#             figsize=(10, 10),
#         )

#         axes = axes.flatten()

#         for grid_index, axis in enumerate(axes):

#             sample_index = start_index + grid_index

#             # Hide unused cells in the final window
#             if sample_index >= num_samples:
#                 axis.axis("off")
#                 continue

#             axis.imshow(
#                 processed[sample_index],
#                 cmap="gray",
#                 vmin=0,
#                 vmax=255,
#             )

#             axis.set_title(
#                 f"Target: {targets[sample_index]}",
#                 fontsize=9,
#             )

#             axis.axis("off")

#         fig.suptitle(
#             f"Samples {start_index + 1}-" f"{min(start_index + samples_per_window, num_samples)}",
#             fontsize=14,
#         )

#         plt.tight_layout()
#         plt.show()


# Save processed images
X = np.stack(processed)
np.save(INK_FOLDER / "X.npy", X)

# Save corresponding target labels
y = np.array(targets)
np.save(INK_FOLDER / "y.npy", y)


# Save preprocessing log
log_path = INK_FOLDER / "log.csv"

with open(log_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "message"],
    )

    writer.writeheader()
    writer.writerows(log_rows)


print(f"Files found: {len(ink_files)}")
print(f"Successfully processed: {len(processed)}")
print(f"Dataset shape: {X.shape}")
print(f"Saved to: {INK_FOLDER / 'X.npy'}")

if log_rows:
    print(f"WARNING: {len(log_rows)} file(s) had preprocessing issues. " f"See {log_path}")

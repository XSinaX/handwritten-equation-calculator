from __future__ import annotations

import csv
import io
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import math
import re
import tarfile
import zipfile
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, PngImagePlugin

from ml.config import BLANK_TOKEN, IMAGE_HEIGHT, SYMBOL_WIDTH, TOKENS

# ============================== Configuration ==============================

ML_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ML_ROOT / "datasets"
CROHME_ARCHIVE = DATASETS / "original" / "CROHME23.zip"
MATHWRITING_ARCHIVE = DATASETS / "original" / "mathwriting-2024.tgz"

WORKERS = 6
MATHWRITING_PENDING = WORKERS * 4

PADDING = 5
LINE_WIDTH = 4  # Stroke width when rendering MathWriting InkML as a PNG.
INK = "http://www.w3.org/2003/InkML"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

# Use the same vocabulary as the recognition model (excluding CTC blank).
VALID_TOKENS = set(TOKENS) - {BLANK_TOKEN}
TOKEN_PATTERN = re.compile("|".join(re.escape(token) for token in sorted(VALID_TOKENS, key=len, reverse=True)))

# Shared by CROHME equation and symbol extraction.
CROHME_SPLITS = {
    "train": ("train/CROHME2019", "train/CROHME2023_train"),
    "validation": ("val/CROHME2016_test", "val/CROHME2023_val"),
    "test": ("test/CROHME2019_test", "test/CROHME2023_test"),
}

# The supplied CROHME archive puts some 2023-training PNGs in the 2013 folder.
# Try the corresponding IMG path first, then this explicit alternate location.
CROHME_IMAGE_ALTERNATES = {
    "train/CROHME2023_train": "train/CROHME2013_train",
}

MATHWRITING_SPLITS = {
    "train": "train",
    "valid": "validation",
    "validation": "validation",
    "val": "validation",
    "test": "test",
}

# Never extract CROHME symbols that belong to these MathML structures.
EXCLUDED_MATHML = {
    "mfrac",
    "msqrt",
    "mroot",
    "msub",
    "msup",
    "msubsup",
    "munder",
    "mover",
    "munderover",
    "mmultiscripts",
    "mtable",
    "mtr",
    "mtd",
}


# ================================= Utilities ================================


def tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def get_annotation(element: ET.Element, annotation_type: str) -> str:
    """Read an annotation directly belonging to this element."""
    for child in element:
        if tag_name(child.tag) == "annotation" and child.get("type") == annotation_type:
            return (child.text or "").strip()
    return ""


def normalize_label(value: str, *, equation: bool) -> str | None:
    """Normalize and validate a symbol or a complete equation using TOKENS.

    Return None when the label contains unsupported characters or constructs.
    """
    if not value:
        return None

    label = value.strip().strip("$").strip()
    label = label.replace("−", "-").replace("×", r"\times").replace("÷", r"\div")

    if not equation:
        return label if label in VALID_TOKENS else None

    label = label.replace(r"\left", "").replace(r"\right", "")
    label = re.sub(r"\s+", "", label)
    label = label.replace("{", "").replace("}", "")

    if not any(character.isdigit() for character in label):
        return None

    # Tokenize and make sure no unsupported text remains between tokens.
    matched_tokens = TOKEN_PATTERN.findall(label)
    if "".join(matched_tokens) != label:
        return None

    return label


def read_traces(root: ET.Element) -> dict[str, list[tuple[float, float, float]]]:
    """Read InkML X/Y strokes, with T timestamps if present (otherwise 0)."""
    channels = ["X", "Y"]
    for child in root:
        if tag_name(child.tag) == "traceFormat":
            channels = []
            for channel in child:
                if tag_name(channel.tag) == "channel":
                    channels.append(channel.get("name"))
            break

    if "X" not in channels or "Y" not in channels:
        raise ValueError("InkML trace format has no X/Y coordinates")

    x_index = channels.index("X")
    y_index = channels.index("Y")
    time_index = channels.index("T") if "T" in channels else None

    traces = {}
    for element in root:
        if tag_name(element.tag) != "trace":
            continue

        trace_id = element.get("id")
        if not trace_id:
            raise ValueError("InkML trace has no id")

        points = []
        for entry in (element.text or "").split(","):
            fields = entry.split()
            if not fields:
                continue

            x = float(fields[x_index])
            y = float(fields[y_index])
            t = float(fields[time_index]) if time_index is not None else 0.0
            if not all(math.isfinite(value) for value in (x, y, t)):
                raise ValueError("Invalid InkML coordinate")
            points.append((x, y, t))

        if not points:
            raise ValueError("Empty InkML trace")
        traces[trace_id] = points

    if not traces:
        raise ValueError("No InkML traces found")
    return traces


def render_inkml(
    traces: dict[str, list[tuple[float, float, float]]],
    target_width: int | None = None,
) -> Image.Image:
    """Render MathWriting strokes at IMAGE_HEIGHT, preserving aspect ratio.

    Small point-like traces are filled circles, preserving division dots.
    """
    points = [point for trace in traces.values() for point in trace]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    content_width = max_x - min_x
    content_height = max_y - min_y

    scale = (IMAGE_HEIGHT - 2 * PADDING - LINE_WIDTH) / max(content_height, 1.0)

    if target_width is not None:
        scale = min(
            scale,
            (target_width - 2 * PADDING - LINE_WIDTH) / max(content_width, 1.0),
        )

    width = (
        target_width
        if target_width is not None
        else max(2 * PADDING + 1, math.ceil(content_width * scale) + 2 * PADDING + LINE_WIDTH)
    )

    image = Image.new("L", (width, IMAGE_HEIGHT), 255)
    canvas = ImageDraw.Draw(image)
    offset_x = (width - content_width * scale) / 2 if target_width is not None else PADDING + LINE_WIDTH / 2
    offset_y = (IMAGE_HEIGHT - content_height * scale) / 2

    for trace in traces.values():
        stroke = []
        for x, y, _ in trace:
            px = round((x - min_x) * scale + offset_x)
            py = round((y - min_y) * scale + offset_y)
            stroke.append((px, py))

        stroke_width = max(x for x, _ in stroke) - min(x for x, _ in stroke)
        stroke_height = max(y for _, y in stroke) - min(y for _, y in stroke)

        if max(stroke_width, stroke_height) <= LINE_WIDTH:
            center_x = round(sum(x for x, _ in stroke) / len(stroke))
            center_y = round(sum(y for _, y in stroke) / len(stroke))
            radius = max(1, LINE_WIDTH // 2)
            canvas.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=0,
            )
        else:
            canvas.line(stroke, fill=0, width=LINE_WIDTH, joint="curve")

    return image


def save_equation(image: Image.Image, dataset: str, split: str, source: str, label: str) -> dict[str, str]:
    """Save the equation PNG and return its row for labels.csv."""
    folder = DATASETS / "equation" / split / dataset
    folder.mkdir(parents=True, exist_ok=True)

    file_id = uuid5(NAMESPACE_URL, f"{dataset}/equation/{split}/{source}")
    filename = f"{file_id}.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("equation", label)
    image.save(folder / filename, pnginfo=metadata)

    return {
        "image": filename,
        "equation": label,
        "type": "human",
        "dataset": dataset,
    }


def write_equation_labels(dataset: str, records: dict[str, list[dict[str, str]]]) -> None:
    """Write one CSV per dataset and split, replacing it on every run."""
    for split, rows in records.items():
        folder = DATASETS / "equation" / split / dataset
        folder.mkdir(parents=True, exist_ok=True)

        with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("image", "equation", "type", "dataset"))
            writer.writeheader()
            writer.writerows(rows)


def save_symbol(
    split: str, source: str, index: int, label: str, strokes: list[list[tuple[float, float, float]]], writer: str
) -> dict[str, str]:
    """Save the labeled InkML and PNG; return CSV metadata without reopening PNG."""
    folder = DATASETS / "symbol" / split / "CROHME23" / "InkML"
    folder.mkdir(parents=True, exist_ok=True)
    filename = uuid5(NAMESPACE_URL, f"CROHME23/symbol/{split}/{source}#{index}")

    root = ET.Element(f"{{{INK}}}ink")
    ET.SubElement(root, f"{{{INK}}}annotation", {"type": "label"}).text = label
    if writer:
        ET.SubElement(root, f"{{{INK}}}annotation", {"type": "writer"}).text = writer

    trace_format = ET.SubElement(root, f"{{{INK}}}traceFormat")
    for channel in ("X", "Y", "T"):
        ET.SubElement(trace_format, f"{{{INK}}}channel", {"name": channel, "type": "decimal"})

    for number, trace in enumerate(strokes):
        trace_element = ET.SubElement(root, f"{{{INK}}}trace", {"id": str(number)})
        trace_element.text = ",".join(f"{x:g} {y:g} {t:g}" for x, y, t in trace)

    output = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    (folder / f"{filename}.inkml").write_bytes(output)

    png_folder = DATASETS / "symbol" / split / "CROHME23" / "PNG"
    png_folder.mkdir(parents=True, exist_ok=True)
    symbol_traces = {str(index): trace for index, trace in enumerate(strokes)}
    image = render_inkml(symbol_traces, target_width=SYMBOL_WIDTH)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("label", label)
    image.save(png_folder / f"{filename}.png", pnginfo=metadata)

    return {"image": f"{filename}.png", "label": label}


def write_symbol_labels(records: dict[str, list[dict[str, str]]]) -> None:
    """Write a labels.csv from the records returned by workers, not PNG metadata."""
    for split in CROHME_SPLITS:
        folder = DATASETS / "symbol" / split / "CROHME23" / "PNG"
        folder.mkdir(parents=True, exist_ok=True)
        rows = sorted(records[split], key=lambda row: row["image"])
        with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("image", "label"))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Symbol {split}: {len(rows):,} PNGs")


# ================================= Equations =================================


def extract_crohme_equation(
    root: ET.Element, archive: zipfile.ZipFile, image_members: dict[str, str], source: str, split: str
) -> dict[str, str] | None:
    """Save a supported CROHME equation using its existing PNG, if present."""
    label = normalize_label(get_annotation(root, "truth"), equation=True)
    if label is None:
        return None

    expected_png = source.replace("/INKML/", "/IMG/").rsplit(".", 1)[0] + ".png"
    png_member = image_members.get(expected_png.lower())
    if png_member is None:
        archive_root, relative = source.split("/INKML/", 1)
        parts = relative.split("/", 2)
        if len(parts) == 3:
            source_folder = f"{parts[0]}/{parts[1]}"
            alternate_folder = CROHME_IMAGE_ALTERNATES.get(source_folder)
            if alternate_folder:
                image_filename = parts[2].rsplit(".", 1)[0] + ".png"
                alternate_path = f"{archive_root}/IMG/{alternate_folder}/{image_filename}"
                png_member = image_members.get(alternate_path.lower())

    if png_member is None:
        return None

    with Image.open(io.BytesIO(archive.read(png_member))) as image:
        return save_equation(image, "CROHME23", split, source, label)


def extract_mathwriting_equation(root: ET.Element, source: str, split: str) -> dict[str, str] | None:
    label = normalize_label(get_annotation(root, "normalizedLabel"), equation=True)
    if label is None:
        return None

    image = render_inkml(read_traces(root))
    return save_equation(image, "MathWriting2024", split, source, label)


# ================================== Symbols ==================================


def extract_crohme_symbols(root: ET.Element, source: str, split: str, records: list[dict[str, str]]) -> None:
    """Extract only supported symbols outside fractions and stacked constructs."""
    math_root = None
    for annotation in root:
        if tag_name(annotation.tag) != "annotationXML":
            continue
        for element in annotation.iter():
            if tag_name(element.tag) == "math":
                math_root = element
                break
        if math_root is not None:
            break

    if math_root is None:
        return

    permitted_ids = set()

    def collect_ids(element: ET.Element) -> None:
        if tag_name(element.tag) in EXCLUDED_MATHML:
            return
        identifier = element.get(XML_ID)
        if identifier:
            permitted_ids.add(identifier)
        for child in element:
            collect_ids(child)

    collect_ids(math_root)
    if not permitted_ids:
        return

    traces = read_traces(root)
    writer = get_annotation(root, "writer")
    for index, group in enumerate(root.iter()):
        if tag_name(group.tag) != "traceGroup":
            continue

        views = []
        mathml_references = []
        for child in group:
            if tag_name(child.tag) == "traceView":
                views.append(child)
            elif tag_name(child.tag) == "annotationXML":
                mathml_references.append((child.get("href") or "").lstrip("#"))

        if not views:
            continue

        label = normalize_label(get_annotation(group, "truth"), equation=False)
        if label is None:
            continue
        if not any(ref in permitted_ids for ref in mathml_references):
            continue

        strokes = []
        for view in views:
            trace_id = (view.get("traceDataRef") or "").lstrip("#")
            if trace_id not in traces:
                raise ValueError(f"Symbol references unknown trace: {trace_id}")
            strokes.append(traces[trace_id])

        records.append(save_symbol(split, source, index, label, strokes, writer))


# ==================================== Main ===================================


# Each worker owns an independent ZIP reader (safe with Windows spawn).
_CROHME_READER: zipfile.ZipFile | None = None
_CROHME_IMAGES: dict[str, str] = {}


def init_crohme_worker(image_members: dict[str, str]) -> None:
    global _CROHME_READER, _CROHME_IMAGES
    _CROHME_READER = zipfile.ZipFile(CROHME_ARCHIVE)
    _CROHME_IMAGES = image_members


def process_crohme_member(job: tuple[str, str]):
    """Read one CROHME InkML and write its equation/symbol files."""
    name, split = job
    assert _CROHME_READER is not None
    try:
        root = ET.fromstring(_CROHME_READER.read(name))
    except (ET.ParseError, ValueError, OSError):
        return split, None, []

    equation = None
    symbols: list[dict[str, str]] = []
    # Preserve independent handling of invalid equations and symbols.
    try:
        equation = extract_crohme_equation(root, _CROHME_READER, _CROHME_IMAGES, name, split)
    except (ValueError, OSError, OverflowError):
        pass

    try:
        extract_crohme_symbols(root, name, split, symbols)
    except (ValueError, OSError, OverflowError):
        pass

    return split, equation, symbols


def process_mathwriting_member(job: tuple[str, str, bytes]):
    """Parse and render one MathWriting equation in a worker."""
    name, split, contents = job
    try:
        root = ET.fromstring(contents)
        record = extract_mathwriting_equation(root, name, split)
    except (ET.ParseError, ValueError, OSError, OverflowError):
        record = None
    return split, record


def extract_crohme() -> None:
    records = {split: [] for split in CROHME_SPLITS}
    symbol_records = {split: [] for split in CROHME_SPLITS}

    source_splits = {}
    for split, folders in CROHME_SPLITS.items():
        for folder in folders:
            source_splits[folder] = split

    with zipfile.ZipFile(CROHME_ARCHIVE) as archive:
        image_members = {}
        inkml_members = []
        for name in archive.namelist():
            if "/IMG/" in name and name.lower().endswith(".png"):
                image_members[name.lower()] = name
            elif "/INKML/" in name and name.lower().endswith(".inkml"):
                inkml_members.append(name)

    def jobs():
        for name in sorted(inkml_members):
            relative = name.split("/INKML/", 1)[1]
            parts = relative.split("/")
            source_folder = "/".join(parts[:2])
            split = source_splits.get(source_folder)
            if split is not None:
                yield name, split

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_crohme_worker, initargs=(image_members,)) as pool:
        for split, equation, symbols in pool.map(process_crohme_member, jobs(), chunksize=8):
            if equation is not None:
                records[split].append(equation)
            symbol_records[split].extend(symbols)

    write_equation_labels("CROHME23", records)
    write_symbol_labels(symbol_records)


def extract_mathwriting() -> None:
    records = {split: [] for split in ("train", "validation", "test")}

    # This archive is a streaming gzip TAR. Read sequentially in the parent,
    # while workers parse and render a bounded number of extracted members.
    with tarfile.open(MATHWRITING_ARCHIVE, "r|gz") as archive, ProcessPoolExecutor(max_workers=WORKERS) as pool:
        pending = deque()

        def collect_one():
            split, record = pending.popleft().result()
            if record is not None:
                records[split].append(record)

        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".inkml"):
                continue

            folders = Path(member.name).parts
            if "symbols" in folders or "synthetic" in folders:
                continue

            split = None
            for folder in folders:
                if folder in MATHWRITING_SPLITS:
                    split = MATHWRITING_SPLITS[folder]
                    break
            if split is None:
                continue

            stream = archive.extractfile(member)
            if stream is None:
                continue

            pending.append(pool.submit(process_mathwriting_member, (member.name, split, stream.read())))
            if len(pending) >= MATHWRITING_PENDING:
                collect_one()

        while pending:
            collect_one()

    write_equation_labels("MathWriting2024", records)


def main() -> None:
    for path in (CROHME_ARCHIVE, MATHWRITING_ARCHIVE):
        if not path.is_file():
            raise FileNotFoundError(f"Original dataset archive missing: {path}")

    extract_crohme()
    extract_mathwriting()
    print("Dataset extraction complete.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
CROHME_ARCHIVE = DATASETS_ROOT / "original" / "CROHME23.zip"
OUTPUT_PATH = Path(__file__).with_name("layout_stats.json")
# These are the only original training collections used for layout measurements.
TRAIN_FOLDERS = ("CROHME2019", "CROHME2023_train")

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
DIGITS = set("0123456789")
PARENTHESES = {"(", ")"}
OPERATORS = {"+", "-", "=", r"\times", r"\div"}
SUPPORTED = DIGITS | PARENTHESES | OPERATORS

# A nonlinear MathML node and its entire subtree must not contribute to
# baseline symbol sizes, reference digit height, or neighbour relationships.
# In particular, scripts use msup/msub tags even without LaTeX commands.
NONLINEAR_STRUCTURES = {
    "mfrac",
    "msqrt",
    "mroot",
    "msup",
    "msub",
    "msubsup",
    "munder",
    "mover",
    "munderover",
    "mmultiscripts",
    "mtable",
    "mtr",
    "mtd",
    "mstack",
    "mlongdiv",
    "menclose",
    "maction",
}
SYMBOL_ELEMENTS = {"mn", "mo", "mi", "mtext"}

RELATION_CATEGORIES = (
    "digit_digit",
    "digit_operator",
    "digit_parenthesis",
    "operator_parenthesis",
    "sign_digit",
)


def name(tag: str) -> str:
    """Return an XML element's local tag name, independent of namespace."""
    return tag.rsplit("}", 1)[-1]


def normalize_token(label: str) -> str:
    """Normalize the supported CROHME symbol labels."""
    label = label.strip()
    return {
        "−": "-",
        "–": "-",
        "×": r"\times",
        "÷": r"\div",
    }.get(label, label)


def read_training_equations(archive: zipfile.ZipFile):
    """Yield InkML files from the two original CROHME training folders."""
    train_root = "TC11_CROHME23/INKML/train/"
    members = sorted(archive.namelist())

    for folder in TRAIN_FOLDERS:
        folder_path = f"{train_root}{folder}/"
        for member in members:
            if member.startswith(folder_path) and member.lower().endswith(".inkml"):
                yield member, archive.read(member)


def parse_equation(xml_data: bytes) -> list[dict | None]:
    """Map MathML-ordered symbols to their annotated InkML traces and bounds.

    None is a boundary for unsupported/unresolved content or an excluded
    nonlinear subtree. This keeps remaining true neighbours from being joined.
    Validity of other tokens is assigned separately in filter_symbols().
    """
    root = ET.fromstring(xml_data)

    # Read raw traces by their original IDs. Respect the InkML X/Y channels.
    channels = ["X", "Y"]
    for child in root:
        if name(child.tag) == "traceFormat":
            channels = [ch.get("name") for ch in child if name(ch.tag) == "channel"]
            break
    if "X" not in channels or "Y" not in channels:
        raise ValueError("InkML is missing the X/Y trace channels")
    x_index, y_index = channels.index("X"), channels.index("Y")

    traces: dict[str, list[tuple[float, float]]] = {}
    for element in root:
        if name(element.tag) != "trace":
            continue
        trace_id = element.get("id") or element.get(XML_ID)
        if not trace_id:
            continue

        points = []
        for entry in (element.text or "").split(","):
            values = entry.strip().split()
            if not values:
                continue
            if len(values) <= max(x_index, y_index):
                raise ValueError("Invalid trace point")
            x, y = float(values[x_index]), float(values[y_index])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("Non-finite trace coordinates")
            points.append((x, y))
        if points:
            traces[trace_id] = points

    # Connect each symbol's MathML ID to its traceGroup and bounding box.
    symbols_by_id: dict[str, dict | None] = {}
    for group in root.iter():
        if name(group.tag) != "traceGroup":
            continue
        views = [child for child in group if name(child.tag) == "traceView"]
        if not views:
            continue  # Outer grouping, not a symbol.

        label = None
        mathml_id = None
        for child in group:
            if name(child.tag) == "annotation" and child.get("type") == "truth":
                label = normalize_token(child.text or "")
            elif name(child.tag) == "annotationXML" and child.get("href"):
                mathml_id = child.get("href").lstrip("#")

        if not mathml_id:
            continue

        symbol_traces = []
        for view in views:
            trace_id = (view.get("traceDataRef") or "").lstrip("#")
            if trace_id in traces:
                symbol_traces.append(traces[trace_id])

        # Incomplete stroke references cannot produce reliable geometry.
        if len(symbol_traces) != len(views) or not label:
            continue

        points = [point for trace in symbol_traces for point in trace]

        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        token = {
            "id": mathml_id,
            "token": label,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "valid": True,
        }
        # Repeated references to one MathML ID are ambiguous; don't guess.
        if mathml_id in symbols_by_id:
            symbols_by_id[mathml_id] = None
        else:
            symbols_by_id[mathml_id] = token

    # The ground-truth MathML, not stroke drawing order, defines neighbours.
    math_root = None
    for annotation in root:
        if name(annotation.tag) != "annotationXML":
            continue
        for element in annotation.iter():
            if name(element.tag) == "math":
                math_root = element
                break
        if math_root is not None:
            break
    if math_root is None:
        raise ValueError("No ground-truth MathML found")

    ordered: list[dict | None] = []
    excluded_positions: list[int] = []

    def walk(element: ET.Element) -> None:
        kind = name(element.tag)

        if kind in NONLINEAR_STRUCTURES:
            # Do not visit any of the structure's children. This prevents
            # numerator-to-denominator and baseline-to-script neighbours.
            excluded_positions.append(len(ordered))
            ordered.append(None)
            return

        if kind == "semantics":
            # Use only the presentation MathML; annotations aren't symbols.
            if len(element):
                walk(element[0])
            return

        if kind in SYMBOL_ELEMENTS:
            symbol = symbols_by_id.get(element.get(XML_ID, ""))
            if symbol is None:
                ordered.append(None)
            else:
                ordered.append(symbol)
            return

        # Ordinary containers (mrow, mfenced, etc.) remain traversable.
        children = [c for c in element if name(c.tag) not in {"annotation", "annotationXML"}]
        if not children:
            # An unrecognised leaf cannot be treated as empty whitespace:
            # retain a boundary rather than join symbols on either side.
            if kind not in {"math", "mrow", "mstyle"}:
                ordered.append(None)
            return

        for child in children:
            walk(child)

    walk(math_root)

    # Even when the fraction itself is excluded, parentheses surrounding it
    # can be unusually tall. Exclude matching pairs that enclose a nonlinear
    # structure, while retaining ordinary linear parentheses such as (1+2).
    opening: list[int] = []
    for index, symbol in enumerate(ordered):
        if symbol is None:
            continue
        if symbol["token"] == "(":
            opening.append(index)
        elif symbol["token"] == ")" and opening:
            start = opening.pop()
            if any(start < pos < index for pos in excluded_positions):
                ordered[start] = None
                ordered[index] = None

    return ordered


def filter_symbols(tokens: list[dict | None]) -> None:
    """Mark a token valid only when supported and geometrically measurable."""
    for symbol in tokens:
        if symbol is None:
            continue
        x0, y0, x1, y1 = symbol["bbox"]
        symbol["valid"] = symbol["token"] in SUPPORTED and x1 > x0 and (y1 > y0 or symbol["token"] in OPERATORS)


def relationship_category(left: dict, right: dict, previous: dict | None) -> str | None:
    """Classify an original neighbouring token pair, ignoring direction."""
    a, b = left["token"], right["token"]

    if a == "-" and b in DIGITS:
        # Unary minus begins an expression or follows an operator/open bracket.
        if previous is None or previous["token"] in OPERATORS | {"("}:
            return "sign_digit"

    if a in DIGITS and b in DIGITS:
        return "digit_digit"
    if (a in DIGITS and b in OPERATORS) or (a in OPERATORS and b in DIGITS):
        return "digit_operator"
    if (a in DIGITS and b in PARENTHESES) or (a in PARENTHESES and b in DIGITS):
        return "digit_parenthesis"
    if (a in OPERATORS and b in PARENTHESES) or (a in PARENTHESES and b in OPERATORS):
        return "operator_parenthesis"
    return None


def measure_equation(
    tokens: list[dict | None],
    results: dict,
) -> None:
    """Append normalized measurements from one equation to results."""
    valid = [s for s in tokens if s is not None and s["valid"]]
    digits = [s for s in valid if s["token"] in DIGITS and s["bbox"][3] > s["bbox"][1]]
    # One digit cannot provide a reliable equation-level height reference.
    if len(digits) < 2:
        return

    heights = [s["bbox"][3] - s["bbox"][1] for s in digits]
    reference = statistics.median(heights)
    if reference <= 0:
        return
    scale = 100.0 / reference

    for symbol in valid:
        x0, y0, x1, y1 = symbol["bbox"]
        token = symbol["token"]
        if token in DIGITS:
            results["symbol_sizes"]["height"][token].append(round((y1 - y0) * scale, 3))
        elif token in PARENTHESES:
            results["symbol_sizes"]["height"][token].append(round((y1 - y0) * scale, 3))
        elif token in OPERATORS:
            results["symbol_sizes"]["width"][token].append(round((x1 - x0) * scale, 3))

    results["height_consistency"]["digits"].append([len(digits), round(max(heights) / min(heights), 5)])

    paren_heights = [s["bbox"][3] - s["bbox"][1] for s in valid if s["token"] in PARENTHESES]
    if len(paren_heights) >= 2 and min(paren_heights) > 0:
        results["height_consistency"]["parentheses"].append(
            [len(paren_heights), round(max(paren_heights) / min(paren_heights), 5)]
        )

    # Preserve original MathML token order. Never create neighbours by
    # removing unsupported tokens, unresolved annotations, or barriers.
    for index in range(len(tokens) - 1):
        left, right = tokens[index], tokens[index + 1]
        if left is None or right is None or not left["valid"] or not right["valid"]:
            continue

        # Generated equations end at '='; never learn spacing after it.
        if left["token"] == "=":
            continue

        previous = tokens[index - 1] if index > 0 else None
        if previous is not None and not previous["valid"]:
            previous = None
        category = relationship_category(left, right, previous)
        if category is None:
            continue

        lx0, ly0, lx1, ly1 = left["bbox"]
        rx0, ry0, rx1, ry1 = right["bbox"]
        gap = (rx0 - lx1) * scale
        left_center = (ly0 + ly1) / 2
        right_center = (ry0 + ry1) / 2

        # Mixed categories always use a fixed offset direction regardless of
        # whether a digit or operator appears first in the written equation.
        if category in {"digit_operator", "sign_digit"}:
            digit_center = left_center if left["token"] in DIGITS else right_center
            operator_center = left_center if left["token"] in OPERATORS else right_center
            offset = (operator_center - digit_center) * scale
        elif category == "digit_parenthesis":
            digit_center = left_center if left["token"] in DIGITS else right_center
            paren_center = left_center if left["token"] in PARENTHESES else right_center
            offset = (paren_center - digit_center) * scale
        elif category == "operator_parenthesis":
            operator_center = left_center if left["token"] in OPERATORS else right_center
            paren_center = left_center if left["token"] in PARENTHESES else right_center
            offset = (paren_center - operator_center) * scale
        else:
            offset = (right_center - left_center) * scale

        # Do not learn spacing from overlapping bounding boxes.
        if gap < 0:
            continue

        results["relationships"][category].append([round(gap, 3), round(offset, 3)])


def iqr_filter(values: list, fields: tuple[int, ...] | None = None) -> list:
    """Remove observations beyond 1.5×IQR, calculating fences on original data.

    For paired values, remove the entire observation if any selected field is
    an outlier. This keeps horizontal gaps and vertical offsets matched.
    """
    if len(values) < 4:
        return values

    fences = []
    for field in fields if fields is not None else (None,):
        numbers = values if field is None else [row[field] for row in values]
        q1, _, q3 = statistics.quantiles(numbers, n=4, method="inclusive")
        margin = 1.5 * (q3 - q1)
        fences.append((field, q1 - margin, q3 + margin))

    return [
        row
        for row in values
        if all(lower <= (row if field is None else row[field]) <= upper for field, lower, upper in fences)
    ]


def apply_iqr_filter(stats: dict) -> int:
    """Clean final output without applying generator-specific percentile cuts."""
    removed = 0

    for measurements in stats["symbol_sizes"].values():
        for token, values in measurements.items():
            cleaned = iqr_filter(values)
            removed += len(values) - len(cleaned)
            measurements[token] = cleaned

    for category, pairs in stats["relationships"].items():
        cleaned = iqr_filter(pairs, fields=(0, 1))
        removed += len(pairs) - len(cleaned)
        stats["relationships"][category] = cleaned

    for category, pairs in stats["height_consistency"].items():
        # [number_of_symbols, height_ratio]: only the ratio is continuous.
        cleaned = iqr_filter(pairs, fields=(1,))
        removed += len(pairs) - len(cleaned)
        stats["height_consistency"][category] = cleaned

    return removed


def collect_layout_stats(
    archive_path: Path = CROHME_ARCHIVE,
    skip_counts: dict[str, int] | None = None,
) -> dict:
    """Collect CROHME training observations, optionally reporting skipped files."""
    results = {
        "symbol_sizes": {
            "height": {token: [] for token in "0123456789()"},
            "width": {token: [] for token in ("+", "-", "=", r"\times", r"\div")},
        },
        "relationships": {category: [] for category in RELATION_CATEGORIES},
        "height_consistency": {"digits": [], "parentheses": []},
    }

    files_seen = 0
    parsing_errors = 0
    insufficient_digits = 0
    with zipfile.ZipFile(archive_path) as archive:
        for _, xml_data in read_training_equations(archive):
            files_seen += 1
            try:
                tokens = parse_equation(xml_data)
                filter_symbols(tokens)
            except (ET.ParseError, ValueError, OverflowError, ZeroDivisionError):
                # A malformed annotation must not contribute unverified geometry.
                parsing_errors += 1
                continue

            valid_digit_count = sum(
                symbol is not None and symbol["valid"] and symbol["token"] in DIGITS for symbol in tokens
            )
            if valid_digit_count < 2:
                insufficient_digits += 1
                continue

            measure_equation(tokens, results)

    if files_seen == 0:
        raise ValueError("No CROHME2019/CROHME2023_train InkML found in archive")

    if skip_counts is not None:
        skip_counts.update(
            {
                "parsing_errors": parsing_errors,
                "insufficient_digits": insufficient_digits,
            }
        )
    return results


def main() -> None:
    if not CROHME_ARCHIVE.is_file():
        raise FileNotFoundError(f"CROHME23 archive not found: {CROHME_ARCHIVE}")

    skip_counts: dict[str, int] = {}
    stats = collect_layout_stats(skip_counts=skip_counts)
    removed = apply_iqr_filter(stats)
    with OUTPUT_PATH.open("w", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2, ensure_ascii=False)

    print(f"Layout statistics saved to {OUTPUT_PATH}")
    print(f"IQR outliers removed: {removed:,}")
    print(f"Skipped due to parsing errors: {skip_counts['parsing_errors']:,}")
    print(f"Skipped with fewer than two valid digits: {skip_counts['insufficient_digits']:,}")


if __name__ == "__main__":
    main()

import math
import random
import uuid
import xml.etree.ElementTree as ET
import statistics
from pathlib import Path

from ml.data_pipeline import layout_stats
from shared.config import BLANK_TOKEN, TOKENS

# -------------------------
# Generation limits
# -------------------------

# Each split is built from its own, non-overlapping pool of symbol drawings.
# This stops the validation/test scores from rewarding memorised strokes:
# the model is scored on handwriting it has never seen, like in real use.
SAMPLES_PER_SPLIT = {
    "train": 20000,
    "val": 3000,
    "test": 3000,
}

SYMBOL_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

SYMBOL_SPLIT_SEED = 42

MAX_OPERANDS = 4
MAX_DIGITS_PER_NUMBER = 3
MAX_EXPRESSION_IN_OPERANDS = 2


# -------------------------
# Operand probabilities
# -------------------------

NUMBER_PROBABILITY = 0.60
SIGNED_NUMBER_PROBABILITY = 0.20
PARENTHESIZED_EXPRESSION_PROBABILITY = 0.20


# -------------------------
# Supported symbols
# -------------------------

NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
OPERATORS = ["+", "-", "\\times", "\\div"]


# -------------------------
# Symbol positioning
# -------------------------
# Sizes, vertical positions and gaps come from real handwriting
# (layout_stats.py). These values cover what the real data can't:

# A negative sign is written close to its number ("-2")...
MIN_SIGN_TO_NUMBER_GAP_RATIO = 0.02
MAX_SIGN_TO_NUMBER_GAP_RATIO = 0.12

# ...and an operator before a negative sign keeps a clear gap, so "5 - -2"
# reads as "5, minus, negative 2" rather than three marks in a row
MIN_OPERATOR_TO_SIGN_GAP_RATIO = 0.35

# Symbols / gaps with no real measurements (e.g. "÷"), in digit heights
FALLBACK_NUMBER_HEIGHT = 1.0
FALLBACK_PARENTHESIS_HEIGHT = 1.0
FALLBACK_OPERATOR_WIDTH = 0.8
MIN_FALLBACK_SIZE_VARIATION = 0.85
MAX_FALLBACK_SIZE_VARIATION = 1.15
MIN_FALLBACK_GAP_RATIO = 0.15
MAX_FALLBACK_GAP_RATIO = 0.60

# Whole-equation distortions (people write uphill/downhill and slanted).
# Benchmarks showed the wider ranges generalise better than narrower ones.
MAX_ROTATION_DEGREES = 6
MAX_SHEAR = 0.30

# Baseline drift: how far (in digit heights) the line wanders over the equation
MAX_BASELINE_DRIFT_RATIO = 0.30

# -------------------------
# File paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYMBOL_FOLDER = PROJECT_ROOT / "ml" / "data" / "mathwriting-2024" / "symbols"

# Extra real drawings cut from human equations (human_glyphs.py). Added to the
# training pool only, so validation/test handwriting stays the same.
HUMAN_SYMBOL_FOLDER = PROJECT_ROOT / "ml" / "data" / "mathwriting-2024" / "human_symbols"
OUTPUT_FOLDER = PROJECT_ROOT / "ml" / "data" / "generated"


def create_symbol_file_dictionary(folder=None):
    """
    Returns:
        dict: Maps each supported token to a list of matching InkML file paths
        in folder (default: SYMBOL_FOLDER).
    """

    symbol_files = {}

    for token in TOKENS:
        if token != BLANK_TOKEN:
            symbol_files[token] = []

    for file_path in (folder or SYMBOL_FOLDER).glob("*.inkml"):
        tree = ET.parse(file_path)
        root = tree.getroot()

        label = None

        # Get the label from the file
        for element in root.iter():
            if element.tag.endswith("annotation") and element.attrib.get("type") == "label":
                label = element.text
                break

        if label and label in symbol_files:
            symbol_files[label].append(file_path)

    return symbol_files


def split_symbol_files(symbol_files):
    """
    Splits each token's drawings into separate train/val/test pools.

    Returns:
        dict: split name -> {token: [file paths]}
    """

    rng = random.Random(SYMBOL_SPLIT_SEED)
    split_names = list(SYMBOL_SPLIT_RATIOS)

    split_files = {split: {} for split in split_names}

    for token, files in symbol_files.items():
        files = sorted(files)
        rng.shuffle(files)

        if len(files) < len(split_names):
            raise ValueError(
                f"Token {token!r} has only {len(files)} drawing(s); "
                f"need at least {len(split_names)} to give every split its own."
            )

        start = 0

        for split_index, split in enumerate(split_names):
            if split_index == len(split_names) - 1:
                end = len(files)
            else:
                count = max(1, round(len(files) * SYMBOL_SPLIT_RATIOS[split]))
                # Leave at least one drawing for each remaining split
                remaining_splits = len(split_names) - split_index - 1
                end = min(start + count, len(files) - remaining_splits)

            split_files[split][token] = files[start:end]
            start = end

    print("\nSymbol drawings per split:")
    for token in symbol_files:
        counts = ", ".join(f"{split}={len(split_files[split][token])}" for split in split_names)
        print(f"  {token:>7}: {counts}")

    return split_files


def generate_number(symbol_files, generation_stats):
    """
    Generates a number containing between 1 and MAX_DIGITS_PER_NUMBER digits.

    Multi-digit numbers cannot start with zero.

    Returns:
        list: Ordered list of digit symbols forming the number.
    """

    selected_symbols = []

    number_length = random.randint(
        1,
        MAX_DIGITS_PER_NUMBER,
    )

    for digit_index in range(number_length):

        # Prevent leading zero in multi-digit numbers
        if digit_index == 0 and number_length > 1:
            digit = random.choice(NUMBERS[1:])
        else:
            digit = random.choice(NUMBERS)

        add_symbol(
            digit,
            "number",
            selected_symbols,
            symbol_files,
            generation_stats,
        )

    return selected_symbols


def generate_signed_number(symbol_files, generation_stats):
    """
    Generates a negative number, written without brackets.

    People don't write "+6" for a positive number or put a single number in
    brackets, so the only sign is "-". It can follow any operator
    ("5+-2", "5--2", "6\\times-2") or start the equation or a bracketed group.

    Examples:
        -3
        -27
    """

    selected_symbols = []

    add_symbol(
        "-",
        "operator",
        selected_symbols,
        symbol_files,
        generation_stats,
    )

    # Marks this "-" as a sign (not subtraction) so it is spaced tightly
    # against its number
    selected_symbols[-1]["is_sign"] = True

    number_symbols = generate_number(
        symbol_files,
        generation_stats,
    )

    selected_symbols.extend(number_symbols)

    return selected_symbols


def generate_parenthesized_expression(
    symbol_files,
    generation_stats,
):
    """
    Generates a parenthesized expression containing multiple operands.

    Examples:
        (3+7)
        (-3+7)
        (3--7)
        (-12\\times4)
    """

    selected_symbols = []

    add_symbol(
        "(",
        "parenthesis",
        selected_symbols,
        symbol_files,
        generation_stats,
    )

    number_of_operands = random.randint(
        2,
        MAX_EXPRESSION_IN_OPERANDS,
    )

    for operand_index in range(number_of_operands):

        operand = generate_operand(
            symbol_files,
            generation_stats,
            allow_expression=False,
        )

        selected_symbols.extend(operand)

        # Add an operator between operands
        if operand_index < number_of_operands - 1:
            operator = random.choice(OPERATORS)

            add_symbol(
                operator,
                "operator",
                selected_symbols,
                symbol_files,
                generation_stats,
            )

    add_symbol(
        ")",
        "parenthesis",
        selected_symbols,
        symbol_files,
        generation_stats,
    )

    return selected_symbols


def generate_operand(
    symbol_files,
    generation_stats,
    allow_expression=True,
):
    """
    Generates one operand based on the configured probabilities.

    Parenthesized expressions can be disabled to prevent nested expressions.
    Signed numbers are always negative and never bracketed.
    """

    operand_types = [
        "number",
        "signed_number",
    ]

    probabilities = [
        NUMBER_PROBABILITY,
        SIGNED_NUMBER_PROBABILITY,
    ]

    if allow_expression:
        operand_types.append("expression")
        probabilities.append(PARENTHESIZED_EXPRESSION_PROBABILITY)

    operand_type = random.choices(
        operand_types,
        weights=probabilities,
        k=1,
    )[0]

    if operand_type == "number":
        return generate_number(
            symbol_files,
            generation_stats,
        )

    if operand_type == "signed_number":
        return generate_signed_number(
            symbol_files,
            generation_stats,
        )

    return generate_parenthesized_expression(
        symbol_files,
        generation_stats,
    )


def add_symbol(
    token,
    symbol_type,
    selected_symbols,
    symbol_files,
    generation_stats,
):
    """
    Selects a random InkML file for a token, adds it to the generated
    symbol sequence, and updates the token usage statistics.
    """

    file_path = random.choice(symbol_files[token])

    selected_symbols.append(
        {
            "token": token,
            "type": symbol_type,
            "file_path": file_path,
        }
    )

    generation_stats["token_counts"][token] += 1


def generate_equation_file_list(symbol_files, generation_stats):
    """
    Generates the high-level structure of one equation.

    The equation contains between 2 and MAX_OPERANDS operands.
    Each operand is generated according to the configured probability system,
    and arithmetic operators are inserted between consecutive operands.

    Example structure:
        OPERAND + OPERAND \times OPERAND

    Returns:
        list: Ordered list of selected symbol metadata.
        dict: Updated generation statistics.
    """

    number_of_operands = random.randint(2, MAX_OPERANDS)

    selected_symbols = []

    for operand_index in range(number_of_operands):

        operand = generate_operand(
            symbol_files,
            generation_stats,
        )

        selected_symbols.extend(operand)

        # Add an arithmetic operator between consecutive operands
        if operand_index < number_of_operands - 1:
            operator = random.choice(OPERATORS)

            add_symbol(
                operator,
                "operator",
                selected_symbols,
                symbol_files,
                generation_stats,
            )

    # Randomly decide whether to add an equals sign at the end of the equation
    if random.random() < 0.5:
        add_symbol(
            "=",
            "operator",
            selected_symbols,
            symbol_files,
            generation_stats,
        )

    return selected_symbols, generation_stats


def read_traces(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    traces = []

    for element in root.iter():
        if not element.tag.endswith("trace"):
            continue

        trace = []

        for raw_point in element.text.strip().split(","):
            values = raw_point.strip().split()

            x = float(values[0])
            y = float(values[1])
            t = float(values[2])

            trace.append((x, y, t))

        traces.append(trace)

    return traces


def plan_sizes_from_real_handwriting(prepared_symbols, unit):
    """
    Sets each symbol's scale_factor so its size matches real handwriting
    (measured by layout_stats.py), where unit is the digit height in the
    symbols' own coordinates. Symbols without real data (e.g. "÷") use the
    FALLBACK_* sizes. Also stores each symbol's vertical centre offset.
    """

    for symbol in prepared_symbols:
        sample = layout_stats.sample_size_and_offset(symbol["token"])

        if sample is not None:
            target_width, target_height, centre = sample
        else:
            # No real measurements for this symbol (e.g. "÷")
            centre = 0.0
            target_width = target_height = None
            variation = random.uniform(MIN_FALLBACK_SIZE_VARIATION, MAX_FALLBACK_SIZE_VARIATION)

            if symbol["type"] == "number":
                target_height = FALLBACK_NUMBER_HEIGHT * variation
            elif symbol["type"] == "parenthesis":
                target_height = FALLBACK_PARENTHESIS_HEIGHT * variation
            else:
                target_width = FALLBACK_OPERATOR_WIDTH * variation

        if target_width is not None:
            scale_factor = target_width * unit / max(symbol["width"], 1e-6)
        else:
            scale_factor = target_height * unit / max(symbol["height"], 1e-6)

        symbol["scale_factor"] = scale_factor
        symbol["centre_offset"] = centre * unit

    return prepared_symbols


def plan_positions_from_real_handwriting(prepared_symbols, unit):
    """
    Returns (left, top) for each already-scaled symbol: real-handwriting gaps
    between neighbours and real vertical positions, plus baseline drift.
    Negative signs hug their number and keep a clear gap from an operator
    before them.
    """

    baseline_drift_per_symbol = (
        random.uniform(-MAX_BASELINE_DRIFT_RATIO, MAX_BASELINE_DRIFT_RATIO)
        * unit
        / max(1, len(prepared_symbols) - 1)
    )

    positions = []
    right_edge = 0.0

    for index, symbol in enumerate(prepared_symbols):
        if index == 0:
            left = 0.0
        else:
            previous = prepared_symbols[index - 1]

            if previous.get("is_sign"):
                # Sign hugs its number: use the gap between digits of a number
                gap = layout_stats.sample_gap("1", "2")
                if gap is None:
                    gap = random.uniform(MIN_SIGN_TO_NUMBER_GAP_RATIO, MAX_SIGN_TO_NUMBER_GAP_RATIO)
                gap = min(max(gap, 0.0), MAX_SIGN_TO_NUMBER_GAP_RATIO)
            else:
                gap = layout_stats.sample_gap(previous["token"], symbol["token"])
                if gap is None:
                    gap = random.uniform(MIN_FALLBACK_GAP_RATIO, MAX_FALLBACK_GAP_RATIO)

                if symbol.get("is_sign") and previous["type"] == "operator":
                    gap = max(gap, MIN_OPERATOR_TO_SIGN_GAP_RATIO)

            left = right_edge + gap * unit

        top = symbol.get("centre_offset", 0.0) - symbol["height"] / 2 + baseline_drift_per_symbol * index

        positions.append((left, top))
        right_edge = left + symbol["width"]

    return positions


def shift_traces_to_origin(traces):
    """
    Shifts all trace coordinates so the symbol starts at the origin (0, 0).

    Args:
        traces (list): List of traces, where each trace is a list of (x, y, t) tuples.

    Returns:
        list: Shifted traces with minimum x and y coordinates equal to 0.
    """

    x_values = []
    y_values = []

    for trace in traces:
        for x, y, t in trace:
            x_values.append(x)
            y_values.append(y)

    min_x = min(x_values)
    min_y = min(y_values)

    max_x = max(x_values)
    max_y = max(y_values)

    symbol_width = max_x - min_x
    symbol_height = max_y - min_y

    shifted_traces = []

    for trace in traces:
        shifted_trace = []

        for x, y, t in trace:
            shifted_x = x - min_x
            shifted_y = y - min_y

            shifted_trace.append((shifted_x, shifted_y, t))

        shifted_traces.append(shifted_trace)

    return shifted_traces, symbol_width, symbol_height


def scale_symbols(prepared_symbols):
    """
    Applies each symbol's scale factor to its traces and updates
    the stored width and height.

    Returns:
        list: Prepared symbols with scaled traces and dimensions.
    """

    for symbol in prepared_symbols:

        scale_factor = symbol["scale_factor"]
        if scale_factor <= 0:
            raise ValueError(f"Invalid scale factor {scale_factor} for symbol {symbol['token']}")
        scaled_traces = []

        for trace in symbol["traces"]:
            scaled_trace = []

            for x, y, t in trace:
                scaled_x = x * scale_factor
                scaled_y = y * scale_factor

                scaled_trace.append((scaled_x, scaled_y, t))

            scaled_traces.append(scaled_trace)

        symbol["traces"] = scaled_traces

        symbol["width"] *= scale_factor
        symbol["height"] *= scale_factor

    return prepared_symbols


def combine_equation_files(selected_symbols):

    prepared_symbols = []

    # Read and normalize all symbols first so their heights are available
    for symbol in selected_symbols:

        traces = read_traces(symbol["file_path"])

        shifted_traces, symbol_width, symbol_height = shift_traces_to_origin(traces)

        prepared_symbols.append(
            {
                **symbol,
                "traces": shifted_traces,
                "width": symbol_width,
                "height": symbol_height,
                "scale_factor": -1,  # Set by plan_sizes_from_real_handwriting()
            }
        )

    # Sizes, positions and gaps measured from real handwriting
    unit = statistics.median(s["height"] for s in prepared_symbols if s["type"] == "number")

    prepared_symbols = plan_sizes_from_real_handwriting(prepared_symbols, unit)
    prepared_symbols = scale_symbols(prepared_symbols)
    positions = plan_positions_from_real_handwriting(prepared_symbols, unit)

    combined_traces = []

    for symbol, (left, top) in zip(prepared_symbols, positions):
        for trace in symbol["traces"]:
            combined_traces.append([(x + left, y + top, t) for x, y, t in trace])

    combined_traces = apply_random_slant_and_rotation(combined_traces)

    return combined_traces, build_equation_label(selected_symbols)


def apply_random_slant_and_rotation(combined_traces):
    """
    Applies one random shear (italic slant) and rotation (writing uphill
    or downhill) to the whole equation.
    """

    angle = math.radians(random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES))
    shear = random.uniform(-MAX_SHEAR, MAX_SHEAR)

    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    transformed_traces = []

    for trace in combined_traces:
        transformed_trace = []

        for x, y, t in trace:
            # Shear first (slant), then rotate
            sheared_x = x + shear * y

            rotated_x = sheared_x * cos_angle - y * sin_angle
            rotated_y = sheared_x * sin_angle + y * cos_angle

            transformed_trace.append((rotated_x, rotated_y, t))

        transformed_traces.append(transformed_trace)

    return transformed_traces


def build_equation_label(selected_symbols):
    """
    Returns:
        str: Equation label such as '12+3='
    """

    equation_label = ""

    for symbol in selected_symbols:
        equation_label += symbol["token"]

    return equation_label


def save_combined_traces_to_inkml(combined_traces, equation_label, output_folder):
    """
    Saves the combined traces as a new InkML file.

    Returns:
        Path: Path of the saved InkML file.
    """

    output_folder.mkdir(parents=True, exist_ok=True)

    sample_id = uuid.uuid4().hex[:16]
    output_path = output_folder / f"{sample_id}.inkml"

    ink = ET.Element("ink", xmlns="http://www.w3.org/2003/InkML")

    annotation_label = ET.SubElement(
        ink,
        "annotation",
        type="label",
    )
    annotation_label.text = equation_label

    annotation_normalized = ET.SubElement(
        ink,
        "annotation",
        type="normalizedLabel",
    )
    annotation_normalized.text = equation_label

    annotation_creation = ET.SubElement(
        ink,
        "annotation",
        type="inkCreationMethod",
    )
    annotation_creation.text = "synthetic"

    annotation_sample_id = ET.SubElement(
        ink,
        "annotation",
        type="sampleId",
    )
    annotation_sample_id.text = sample_id

    trace_format = ET.SubElement(ink, "traceFormat")

    ET.SubElement(trace_format, "channel", name="X", type="decimal")
    ET.SubElement(trace_format, "channel", name="Y", type="decimal")
    ET.SubElement(trace_format, "channel", name="T", type="decimal", units="ms")

    for trace_index, trace in enumerate(combined_traces):
        trace_element = ET.SubElement(
            ink,
            "trace",
            id=str(trace_index),
        )

        point_strings = []

        for x, y, t in trace:
            point_strings.append(f"{x:.2f} {y:.2f} {t:.1f}")

        trace_element.text = ",".join(point_strings)

    tree = ET.ElementTree(ink)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return output_path


if __name__ == "__main__":

    all_symbol_files = create_symbol_file_dictionary()
    split_symbol_pools = split_symbol_files(all_symbol_files)

    if HUMAN_SYMBOL_FOLDER.exists():
        human_symbol_files = create_symbol_file_dictionary(HUMAN_SYMBOL_FOLDER)

        for token, files in human_symbol_files.items():
            split_symbol_pools["train"][token].extend(files)

        print(f"\nAdded {sum(map(len, human_symbol_files.values()))} drawings from human equations to the training pool")

    for split, num_samples in SAMPLES_PER_SPLIT.items():

        symbol_files = split_symbol_pools[split]
        output_folder = OUTPUT_FOLDER / split

        generation_stats = {
            "total_files": 0,
            "token_counts": {},
        }

        for token in TOKENS:
            if token != BLANK_TOKEN:
                generation_stats["token_counts"][token] = 0

        for _ in range(num_samples):

            selected_symbols, generation_stats = generate_equation_file_list(
                symbol_files,
                generation_stats,
            )

            combined_traces, equation_label = combine_equation_files(selected_symbols)

            save_combined_traces_to_inkml(
                combined_traces,
                equation_label,
                output_folder,
            )

            generation_stats["total_files"] += 1

        print(f"\nGeneration stats ({split} -> {output_folder}):")
        print(generation_stats)

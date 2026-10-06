import random
import uuid
import xml.etree.ElementTree as ET
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from shared.config import BLANK_TOKEN, TOKENS

# -------------------------
# Generation limits
# -------------------------

NUM_SAMPLES_TO_GENERATE = 20000

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

MIN_GAP_RATIO = 0.25
MAX_GAP_RATIO = 0.30

MIN_VERTICAL_SHIFT_RATIO = -0.05
MAX_VERTICAL_SHIFT_RATIO = 0.05

PARENTHESIS_HEIGHT_TO_NUMBER_HEIGHT_RATIO = 1.0
OPERATOR_WIDTH_TO_NUMBER_HEIGHT_RATIO = 0.8

MIN_SCALE_VARIATION = 0.95
MAX_SCALE_VARIATION = 1.05

MIN_ZERO_HEIGHT_TO_NUMBER_HEIGHT_RATIO = 0.75
MAX_ZERO_HEIGHT_TO_NUMBER_HEIGHT_RATIO = 1.00
# -------------------------
# File paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYMBOL_FOLDER = PROJECT_ROOT / "ml" / "data" / "mathwriting-2024" / "symbols"
OUTPUT_FOLDER = PROJECT_ROOT / "ml" / "data" / "generated"


def create_symbol_file_dictionary():
    """
    Returns:
        dict: Maps each supported token to a list of matching InkML file paths.
    """

    symbol_files = {}

    for token in TOKENS:
        if token != BLANK_TOKEN:
            symbol_files[token] = []

    for file_path in SYMBOL_FOLDER.glob("*.inkml"):
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


def generate_parenthesized_signed_number(
    symbol_files,
    generation_stats,
):
    """
    Generates a signed number enclosed in parentheses.

    Examples:
        (-3)
        (+27)

    Returns:
        list: Ordered list of symbols forming the signed number.
    """

    selected_symbols = []

    # Opening parenthesis
    add_symbol(
        "(",
        "parenthesis",
        selected_symbols,
        symbol_files,
        generation_stats,
    )

    # Randomly choose positive or negative sign
    sign = random.choice(["+", "-"])

    add_symbol(
        sign,
        "operator",
        selected_symbols,
        symbol_files,
        generation_stats,
    )

    # Generate the number
    number_symbols = generate_number(
        symbol_files,
        generation_stats,
    )

    selected_symbols.extend(number_symbols)

    # Closing parenthesis
    add_symbol(
        ")",
        "parenthesis",
        selected_symbols,
        symbol_files,
        generation_stats,
    )

    return selected_symbols


def generate_signed_number(symbol_files, generation_stats):
    """
    Generates a signed number without surrounding parentheses.

    Examples:
        -3
        +27
    """

    selected_symbols = []

    sign = random.choice(["+", "-"])

    add_symbol(
        sign,
        "operator",
        selected_symbols,
        symbol_files,
        generation_stats,
    )

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
        (3+(-7))
        (-12\\times(+4))
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
            is_first_position=(operand_index == 0),
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
    is_first_position=False,
):
    """
    Generates one operand based on the configured probabilities.

    Parenthesized expressions can be disabled to prevent nested expressions.
    Signed numbers are only written without parentheses when they are the
    first operand of the current expression.
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

        if is_first_position:
            return generate_signed_number(
                symbol_files,
                generation_stats,
            )

        return generate_parenthesized_signed_number(
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
            is_first_position=(operand_index == 0),
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


def calculate_vertical_shift(
    symbol_token,
    symbol_type,
    symbol_height,
    reference_number_height,
):

    # Special alignment for zeros
    if symbol_token == "0" and reference_number_height > symbol_height:

        height_difference = reference_number_height - symbol_height

        center_shift = height_difference / 2
        bottom_shift = height_difference

        # Bias the position toward bottom alignment
        vertical_shift = random.triangular(
            center_shift,
            bottom_shift,
            bottom_shift,
        )

        # Add normal handwriting variation
        jitter = (
            random.uniform(
                MIN_VERTICAL_SHIFT_RATIO,
                MAX_VERTICAL_SHIFT_RATIO,
            )
            * symbol_height
        )

        vertical_shift += jitter

        # Keep zero between center alignment and bottom alignment
        vertical_shift = max(
            center_shift,
            min(vertical_shift, bottom_shift),
        )

        return vertical_shift

    # Normal numbers and parentheses
    if symbol_type in ["number", "parenthesis"]:
        return (
            random.uniform(
                MIN_VERTICAL_SHIFT_RATIO,
                MAX_VERTICAL_SHIFT_RATIO,
            )
            * symbol_height
        )

    # Operators
    max_shift = reference_number_height - symbol_height

    if max_shift <= 0:
        return 0

    middle = max_shift / 2
    standard_deviation = max_shift / 6

    vertical_shift = random.gauss(
        middle,
        standard_deviation,
    )

    vertical_shift = max(
        0,
        min(vertical_shift, max_shift),
    )

    return vertical_shift


def find_reference_number_height(
    prepared_symbols,
    symbol_index,
    exclude_zero=False,
):
    """
    Finds the nearest number to the current symbol.

    If exclude_zero is True, zero symbols are ignored.
    If both sides are equally close, the left side is preferred.
    """

    for distance in range(1, len(prepared_symbols)):

        left_index = symbol_index - distance
        right_index = symbol_index + distance

        if left_index >= 0:
            symbol = prepared_symbols[left_index]

            if symbol["type"] == "number":
                if not exclude_zero or symbol["token"] != "0":
                    return symbol["height"]

        if right_index < len(prepared_symbols):
            symbol = prepared_symbols[right_index]

            if symbol["type"] == "number":
                if not exclude_zero or symbol["token"] != "0":
                    return symbol["height"]

    return 0


def calculate_horizontal_shift(combined_traces, symbol_heights):
    """
    Calculates the amount to shift the traces to the right based on the maximum x-coordinate
    plus a random offset based on the average height of the symbols.

    Returns:
        float: The amount to shift the traces to the right.
    """
    x_values = []
    for trace in combined_traces:
        for x, y, t in trace:
            x_values.append(x)

    max_x = max(x_values)

    average_height = sum(symbol_heights) / len(symbol_heights)

    gap_ratio = random.uniform(
        MIN_GAP_RATIO,
        MAX_GAP_RATIO,
    )

    gap = average_height * gap_ratio

    return max_x + gap


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


def calculate_symbol_scale_factors(prepared_symbols):

    number_heights = []
    operator_widths = []

    for symbol in prepared_symbols:

        if symbol["type"] == "number":
            number_heights.append(symbol["height"])

        elif symbol["type"] == "operator":
            operator_widths.append(symbol["width"])

    median_number_height = statistics.median(number_heights)

    target_parenthesis_height = median_number_height * PARENTHESIS_HEIGHT_TO_NUMBER_HEIGHT_RATIO

    target_operator_width = median_number_height * OPERATOR_WIDTH_TO_NUMBER_HEIGHT_RATIO

    for symbol in prepared_symbols:

        if symbol["type"] == "number":
            # Scale zeros differently to make them more visually consistent with human handwriting
            if symbol["token"] == "0":
                zero_height_ratio = random.uniform(
                    MIN_ZERO_HEIGHT_TO_NUMBER_HEIGHT_RATIO,
                    MAX_ZERO_HEIGHT_TO_NUMBER_HEIGHT_RATIO,
                )

                target_height = median_number_height * zero_height_ratio
                scale_factor = target_height / symbol["height"]
            else:
                scale_factor = median_number_height / symbol["height"]

        elif symbol["type"] == "parenthesis":
            scale_factor = target_parenthesis_height / symbol["height"]

        elif symbol["type"] == "operator":
            scale_factor = target_operator_width / symbol["width"]

        scale_variation = random.uniform(0.95, 1.05)

        symbol["scale_factor"] = scale_factor * scale_variation

    return prepared_symbols


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
                "scale_factor": -1,  # For detecting mistakes in the scale factor calculation
            }
        )

    # Calculate the scale factor and apply it to each symbol so they are all sized appropriately
    prepared_symbols = calculate_symbol_scale_factors(prepared_symbols)
    prepared_symbols = scale_symbols(prepared_symbols)

    combined_traces = []
    symbol_heights = []

    # Position and combine the prepared symbols
    for i in range(len(prepared_symbols)):

        symbol = prepared_symbols[i]

        symbol_type = symbol["type"]
        symbol_height = symbol["height"]
        shifted_traces = symbol["traces"]

        # Operators use the nearest number as their vertical reference
        if symbol_type == "operator":
            reference_number_height = find_reference_number_height(
                prepared_symbols,
                i,
            )
        elif symbol["token"] == "0":
            reference_number_height = find_reference_number_height(
                prepared_symbols,
                i,
                exclude_zero=True,
            )
        else:
            reference_number_height = 0

        vertical_shift = calculate_vertical_shift(
            symbol["token"],
            symbol_type,
            symbol_height,
            reference_number_height,
        )

        # First symbol starts at x = 0
        if i == 0:
            horizontal_shift = 0
        else:
            horizontal_shift = calculate_horizontal_shift(
                combined_traces,
                symbol_heights,
            )

        for trace in shifted_traces:

            positioned_trace = []

            for x, y, t in trace:
                positioned_x = x + horizontal_shift
                positioned_y = y + vertical_shift

                positioned_trace.append((positioned_x, positioned_y, t))

            combined_traces.append(positioned_trace)

        symbol_heights.append(symbol_height)

    equation_label = build_equation_label(selected_symbols)

    return combined_traces, equation_label


def build_equation_label(selected_symbols):
    """
    Returns:
        str: Equation label such as '12+3='
    """

    equation_label = ""

    for symbol in selected_symbols:
        equation_label += symbol["token"]

    return equation_label


def visualize_combined_traces(
    combined_traces,
    equation_label,
    selected_symbols,
):

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw the equation
    for trace in combined_traces:

        x_values = []
        y_values = []

        for x, y, t in trace:
            x_values.append(x)
            y_values.append(y)

        unique_points = set(zip(x_values, y_values))

        # Draw traces containing only one unique coordinate as dots
        if len(unique_points) == 1:
            x, y = next(iter(unique_points))

            ax.scatter(
                x,
                y,
                color="black",
                s=30,
            )

        # Draw normal strokes
        else:
            ax.plot(
                x_values,
                y_values,
                color="black",
                linewidth=2,
            )

    file_information = ""

    for symbol in selected_symbols:
        file_information += f"{symbol['token']} -> " f"{symbol['file_path'].name}\n"

    ax.invert_yaxis()
    ax.axis("equal")

    ax.set_title(f"Generated Equation: {equation_label}")

    fig.subplots_adjust(bottom=0.42)

    fig.text(
        0.5,
        0.02,
        file_information,
        ha="center",
        va="bottom",
        fontsize=8,
    )

    plt.show()


def save_combined_traces_to_inkml(combined_traces, equation_label):
    """
    Saves the combined traces as a new InkML file.

    Returns:
        Path: Path of the saved InkML file.
    """

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    sample_id = uuid.uuid4().hex[:16]
    output_path = OUTPUT_FOLDER / f"{sample_id}.inkml"

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

    symbol_files = create_symbol_file_dictionary()

    generation_stats = {
        "total_files": 0,
        "token_counts": {},
    }

    for token in TOKENS:
        if token != BLANK_TOKEN:
            generation_stats["token_counts"][token] = 0

    for _ in range(NUM_SAMPLES_TO_GENERATE):

        selected_symbols, generation_stats = generate_equation_file_list(
            symbol_files,
            generation_stats,
        )

        combined_traces, equation_label = combine_equation_files(selected_symbols)

        # visualize_combined_traces(combined_traces, equation_label, selected_symbols)

        saved_path = save_combined_traces_to_inkml(
            combined_traces,
            equation_label,
        )

        generation_stats["total_files"] += 1

    print("\nGeneration stats:")
    print(generation_stats)

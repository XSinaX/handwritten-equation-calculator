"""Build random arithmetic equations and render one synthetic PNG per sample.

Run from the repository root:
    python -m ml.data_pipeline.equation_generator

For each split, uses ONLY that split's CROHME23 symbol InkML drawings:
    ml/datasets/symbol/{train,validation,test}/CROHME23/InkML/*.inkml

Writes PNG images and labels.csv (image,equation,type) to:
    ml/datasets/equation/{train,validation,test}/Synthetic/

Equation grammar and token/file choices live here; stroke geometry lives in
ml.data_pipeline.equation_layout; photo appearance lives in photo_renderer.
"""

from __future__ import annotations

import csv
import random
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from ml.data_pipeline.equation_layout import (
    layout_equation,
    load_layout_statistics,
    load_symbol_traces,
)
from ml.data_pipeline.photo_renderer import render_photo

SAMPLES_PER_SPLIT = {"train": 20000, "validation": 3000, "test": 3000}
MAX_OPERANDS = 4
MAX_DIGITS_PER_NUMBER = 3
MAX_EXPRESSION_IN_OPERANDS = 2

NUMBER_PROBABILITY = 0.60
SIGNED_NUMBER_PROBABILITY = 0.20
PARENTHESIZED_EXPRESSION_PROBABILITY = 0.20
EQUALS_SIGN_PROBABILITY = 0.50

DIGITS = "0123456789"
OPERATORS = ("+", "-", r"\times", r"\div")
TOKENS = set(DIGITS) | {"(", ")", "="} | set(OPERATORS)

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
SYMBOL_DATASET = "CROHME23"
SYNTHETIC_FOLDER_NAME = "Synthetic"
MIN_PEN_WIDTH = 3
MAX_PEN_WIDTH = 6


def create_symbol_file_dictionary(split: str) -> dict[str, list[Path]]:
    """Map each token to usable symbol drawings from *this split only*.

    Invalid geometries are filtered out up front so generation never picks a
    digit/parenthesis with zero height or an operator with zero width.
    """
    folder = DATASETS_ROOT / "symbol" / split / SYMBOL_DATASET / "InkML"
    if not folder.is_dir():
        raise FileNotFoundError(f"Symbol folder not found: {folder}")

    symbol_files = {token: [] for token in TOKENS}
    skipped_invalid = {token: 0 for token in TOKENS}

    for path in sorted(folder.glob("*.inkml")):
        root = ET.parse(path).getroot()
        # The extracted symbols have one top-level <annotation type="label">.
        label = next(
            (
                (child.text or "").strip()
                for child in root
                if child.tag.endswith("annotation") and child.get("type") == "label"
            ),
            None,
        )
        if label not in symbol_files:
            continue

        try:
            geometry = load_symbol_traces({"token": label, "file_path": path})
        except (OSError, ET.ParseError, ValueError):
            skipped_invalid[label] += 1
            continue

        if label in set(OPERATORS) | {"="}:
            is_valid = geometry["width"] > 0
        else:
            is_valid = geometry["height"] > 0

        if is_valid:
            symbol_files[label].append(path)
        else:
            skipped_invalid[label] += 1

    missing = sorted(token for token, files in symbol_files.items() if not files)
    if missing:
        raise ValueError(f"No usable {split} symbol InkML available for: {', '.join(missing)}")

    skipped_total = sum(skipped_invalid.values())
    if skipped_total:
        details = ", ".join(f"{token}: {count}" for token, count in sorted(skipped_invalid.items()) if count)
        print(f"{split}: skipped {skipped_total} invalid symbol files ({details})")

    return symbol_files


def add_symbol(token: str, symbol_type: str, result: list[dict], symbol_files: dict, is_sign: bool = False) -> None:
    """Choose one source drawing; no traces or geometry are handled here."""
    result.append(
        {
            "token": token,
            "type": symbol_type,
            "file_path": random.choice(symbol_files[token]),
            "is_sign": is_sign,
        }
    )


def generate_number(symbol_files: dict) -> list[dict]:
    """Generate 1..MAX_DIGITS_PER_NUMBER digits, without leading zeros."""
    result = []
    length = random.randint(1, MAX_DIGITS_PER_NUMBER)
    for index in range(length):
        digit = random.choice(DIGITS[1:] if index == 0 and length > 1 else DIGITS)
        add_symbol(digit, "number", result, symbol_files)
    return result


def generate_signed_number(symbol_files: dict) -> list[dict]:
    """Produce an initial signed operand, such as -23 or +7."""
    result = []
    add_symbol(random.choice(("+", "-")), "operator", result, symbol_files, is_sign=True)
    result.extend(generate_number(symbol_files))
    return result


def generate_parenthesized_signed_number(symbol_files: dict) -> list[dict]:
    """Produce a signed operand safe after a binary operator, e.g. (-3)."""
    result = []
    add_symbol("(", "parenthesis", result, symbol_files)
    result.extend(generate_signed_number(symbol_files))
    add_symbol(")", "parenthesis", result, symbol_files)
    return result


def generate_parenthesized_expression(symbol_files: dict) -> list[dict]:
    """Generate a flat parenthesized expression without nested expressions."""
    result = []
    add_symbol("(", "parenthesis", result, symbol_files)
    count = random.randint(2, MAX_EXPRESSION_IN_OPERANDS)
    for index in range(count):
        result.extend(generate_operand(symbol_files, allow_expression=False, is_first_position=index == 0))
        if index < count - 1:
            add_symbol(random.choice(OPERATORS), "operator", result, symbol_files)
    add_symbol(")", "parenthesis", result, symbol_files)
    return result


def generate_operand(symbol_files: dict, allow_expression: bool = True, is_first_position: bool = False) -> list[dict]:
    """Choose a number, signed number, or (non-nested) expression."""
    choices = ["number", "signed"]
    weights = [NUMBER_PROBABILITY, SIGNED_NUMBER_PROBABILITY]
    if allow_expression:
        choices.append("expression")
        weights.append(PARENTHESIZED_EXPRESSION_PROBABILITY)

    selected = random.choices(choices, weights=weights, k=1)[0]
    if selected == "number":
        return generate_number(symbol_files)
    if selected == "signed":
        if is_first_position:
            return generate_signed_number(symbol_files)
        return generate_parenthesized_signed_number(symbol_files)
    return generate_parenthesized_expression(symbol_files)


def generate_equation(symbol_files: dict) -> tuple[str, list[dict]]:
    """Return (equation_label, ordered_token_metadata) without rendering."""
    result = []
    count = random.randint(2, MAX_OPERANDS)
    for index in range(count):
        result.extend(generate_operand(symbol_files, is_first_position=index == 0))
        if index < count - 1:
            add_symbol(random.choice(OPERATORS), "operator", result, symbol_files)
    # Sometimes the expression ends without an equals sign.
    if random.random() < EQUALS_SIGN_PROBABILITY:
        add_symbol("=", "operator", result, symbol_files)
    return "".join(symbol["token"] for symbol in result), result


def generate_split(split: str, count: int, stats: dict) -> None:
    """Generate images and append their labels alongside them in each split."""
    symbol_files = create_symbol_file_dictionary(split)
    output_dir = DATASETS_ROOT / "equation" / split / SYNTHETIC_FOLDER_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "labels.csv"

    # Appending preserves earlier runs, and filenames use UUIDs to avoid clashes.
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "equation", "type"))
        if csv_path.stat().st_size == 0:
            writer.writeheader()

        for _ in range(count):
            equation, selected_symbols = generate_equation(symbol_files)
            inkml = layout_equation(selected_symbols, stats)
            filename = f"synthetic_{uuid.uuid4().hex}.png"
            render_photo(inkml, output_dir / filename, line_width=random.randint(MIN_PEN_WIDTH, MAX_PEN_WIDTH))
            # Relative image path resolves from the labels.csv directory.
            writer.writerow({"image": filename, "equation": equation, "type": "synthetic"})

    print(f"{split}: generated {count:,} PNGs in {output_dir}")


def main() -> None:
    stats = load_layout_statistics()
    for split, count in SAMPLES_PER_SPLIT.items():
        if count > 0:
            generate_split(split, count, stats)


if __name__ == "__main__":
    main()

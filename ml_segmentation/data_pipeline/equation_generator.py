from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
import random
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from ml_segmentation.data_pipeline.equation_renderer import (
    render_equation,
    load_layout_statistics,
    load_symbol_label,
    load_symbol_traces,
)

SAMPLES_PER_SPLIT = {"train": 30000, "validation": 5000, "test": 5000}
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
SYMBOL_DATASETS_ROOT = Path(__file__).resolve().parents[2] / "ml" / "datasets" / "symbol"
SYMBOL_DATASET = "CROHME23"
SYNTHETIC_FOLDER_NAME = "Synthetic"
MIN_PEN_WIDTH = 3
MAX_PEN_WIDTH = 6
NUM_WORKERS = 8
WORK_BATCH_SIZE = 256
WORK_CHUNK_SIZE = 8

_worker_symbol_files = None
_worker_stats = None
_worker_output_dir = None


def create_symbol_file_dictionary(split: str) -> dict[str, list[Path]]:
    """Map each token to usable symbol drawings from *this split only*."""
    folder = SYMBOL_DATASETS_ROOT / split / SYMBOL_DATASET / "InkML"
    if not folder.is_dir():
        raise FileNotFoundError(f"Symbol folder not found: {folder}")

    symbol_files = {token: [] for token in TOKENS}
    skipped_invalid = {token: 0 for token in TOKENS}

    for path in sorted(folder.glob("*.inkml")):
        label = None
        try:
            label = load_symbol_label(path)
            if label not in symbol_files:
                continue
            geometry = load_symbol_traces({"token": label, "file_path": path})
        except (OSError, ET.ParseError, ValueError):
            if label in skipped_invalid:
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
    if random.random() < EQUALS_SIGN_PROBABILITY:
        add_symbol("=", "operator", result, symbol_files)
    return "".join(symbol["token"] for symbol in result), result


def initialize_generator_worker(symbol_files: dict, stats: dict, output_dir: Path) -> None:
    """Initialize one worker and seed it independently."""
    global _worker_symbol_files, _worker_stats, _worker_output_dir
    _worker_symbol_files = symbol_files
    _worker_stats = stats
    _worker_output_dir = output_dir
    random.seed(int.from_bytes(os.urandom(16), "big"))


def generate_one_image(_index: int) -> tuple[str, str]:
    """Generate and save one image; return its CSV row to the parent process."""
    equation, selected_symbols = generate_equation(_worker_symbol_files)
    filename = f"synthetic_{uuid.uuid4().hex}.png"
    render_equation(
        selected_symbols,
        _worker_stats,
        _worker_output_dir / filename,
        line_width=random.randint(MIN_PEN_WIDTH, MAX_PEN_WIDTH),
    )
    return filename, equation


def generate_split(split: str, count: int, stats: dict, output_root: Path | None = None) -> None:
    """Generate equations in four workers and replace this split's labels.csv."""
    symbol_files = create_symbol_file_dictionary(split)
    output_dir = (output_root or (DATASETS_ROOT / "equation")) / split / SYNTHETIC_FOLDER_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "labels.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "equation", "type"))
        writer.writeheader()

        with ProcessPoolExecutor(
            max_workers=NUM_WORKERS,
            initializer=initialize_generator_worker,
            initargs=(symbol_files, stats, output_dir),
        ) as pool:
            for start in range(0, count, WORK_BATCH_SIZE):
                indices = range(start, min(start + WORK_BATCH_SIZE, count))
                for filename, equation in pool.map(generate_one_image, indices, chunksize=WORK_CHUNK_SIZE):
                    writer.writerow({"image": filename, "equation": equation, "type": "synthetic"})

    print(f"{split}: generated {count:,} PNGs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic equations and pixel ownership masks")
    parser.add_argument(
        "--split",
        choices=("all", *SAMPLES_PER_SPLIT),
        default="all",
        help="Generate a single split or all splits (default: all)",
    )
    parser.add_argument(
        "--count", type=int, default=None, help="Override sample count per selected split, useful for a small pilot"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATASETS_ROOT / "equation",
        help="Alternative dataset root inside ml_segmentation/, to avoid overwriting existing data",
    )
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("--count must be positive")
    root = args.output_root.resolve()
    if not root.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("--output-root must stay inside the ml_segmentation/ directory")
    stats = load_layout_statistics()
    splits = SAMPLES_PER_SPLIT if args.split == "all" else (args.split,)
    for split in splits:
        count = args.count if args.count is not None else SAMPLES_PER_SPLIT[split]
        if count > 0:
            generate_split(split, count, stats, output_root=root)


if __name__ == "__main__":
    main()

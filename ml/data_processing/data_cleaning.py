import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "mathwriting-2024"

HUMAN_SPLITS = [
    "train",
    "valid",
    "test",
]

HUMAN_OUTPUT = DATASET_PATH / "human"
SYNTHETIC_PATH = DATASET_PATH / "synthetic"


def is_valid_equation(equation):
    allowed_commands = ["\\left", "\\right", "\\frac", "\\cdot"]

    commands = re.findall(r"\\[a-zA-Z]+", equation)

    for command in commands:
        if command not in allowed_commands:
            return False

    # Remove LaTeX commands
    cleaned_equation = re.sub(r"\\[a-zA-Z]+", "", equation)

    # Must contain at least one digit
    if not re.search(r"\d", cleaned_equation):
        return False

    # Remove allowed characters
    digit_equation = re.sub(r"[+=(){} -]", "", cleaned_equation)

    # Anything remaining must be a digit or decimal point
    if re.search(r"[^0-9.]", digit_equation):
        return False

    return True


def get_normalized_label(inkml_path):
    tree = ET.parse(inkml_path)
    root = tree.getroot()

    for annotation in root.findall(".//{*}annotation"):
        if annotation.attrib.get("type") == "normalizedLabel":
            return annotation.text

    return None


def get_safe_destination(destination_folder, filename, split_name):
    """
    Prevent accidental overwriting if two splits contain files
    with the same filename.
    """

    destination = destination_folder / filename

    if not destination.exists():
        return destination

    original = Path(filename)

    return destination_folder / (f"{original.stem}_{split_name}{original.suffix}")


def process_human_data():
    HUMAN_OUTPUT.mkdir(exist_ok=True)

    total_moved = 0
    total_deleted = 0
    total_errors = 0

    for split in HUMAN_SPLITS:
        split_path = DATASET_PATH / split

        print(f"\nProcessing human split: {split}")

        moved = 0
        deleted = 0
        errors = 0

        inkml_files = list(split_path.glob("*.inkml"))

        for index, inkml_path in enumerate(inkml_files, start=1):
            try:
                equation = get_normalized_label(inkml_path)

                if equation is None:
                    print(f"Missing label: {inkml_path.name}")
                    errors += 1
                    total_errors += 1
                    continue

                if is_valid_equation(equation):
                    destination = get_safe_destination(
                        HUMAN_OUTPUT,
                        inkml_path.name,
                        split,
                    )

                    shutil.move(
                        str(inkml_path),
                        str(destination),
                    )

                    moved += 1
                    total_moved += 1

                else:
                    inkml_path.unlink()

                    deleted += 1
                    total_deleted += 1

            except ET.ParseError:
                print(f"Could not parse: {inkml_path.name}")
                errors += 1
                total_errors += 1

            if index % 10000 == 0:
                print(f"  Processed: {index:,} | " f"Moved: {moved:,} | " f"Deleted: {deleted:,}")

        print(f"{split} complete | " f"Moved: {moved:,} | " f"Deleted: {deleted:,} | " f"Errors: {errors:,}")

    return total_moved, total_deleted, total_errors


def process_synthetic_data():
    print("\nProcessing synthetic data")

    kept = 0
    deleted = 0
    errors = 0

    inkml_files = list(SYNTHETIC_PATH.glob("*.inkml"))

    for index, inkml_path in enumerate(inkml_files, start=1):
        try:
            equation = get_normalized_label(inkml_path)

            if equation is None:
                print(f"Missing label: {inkml_path.name}")
                errors += 1
                continue

            if is_valid_equation(equation):
                kept += 1

            else:
                inkml_path.unlink()
                deleted += 1

        except ET.ParseError:
            print(f"Could not parse: {inkml_path.name}")
            errors += 1

        if index % 10000 == 0:
            print(f"  Processed: {index:,} | " f"Kept: {kept:,} | " f"Deleted: {deleted:,}")

    print(f"Synthetic complete | " f"Kept: {kept:,} | " f"Deleted: {deleted:,} | " f"Errors: {errors:,}")

    return kept, deleted, errors


def clean_dataset():
    print("========== HUMAN DATA ==========")

    human_moved, human_deleted, human_errors = process_human_data()

    print("\n========== SYNTHETIC DATA ==========")

    synthetic_kept, synthetic_deleted, synthetic_errors = process_synthetic_data()

    print("\n========== COMPLETE ==========")

    print(f"Human files moved:       {human_moved:,}")
    print(f"Human files deleted:     {human_deleted:,}")
    print(f"Human errors:            {human_errors:,}")

    print(f"Synthetic files kept:    {synthetic_kept:,}")
    print(f"Synthetic files deleted: {synthetic_deleted:,}")
    print(f"Synthetic errors:        {synthetic_errors:,}")


if __name__ == "__main__":
    clean_dataset()

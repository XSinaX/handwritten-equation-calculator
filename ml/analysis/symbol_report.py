from pathlib import Path
import xml.etree.ElementTree as ET

from ml.config import TOKENS, BLANK_TOKEN

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INK_FOLDER = PROJECT_ROOT / "ml" / "data" / "mathwriting-2024" / "symbols"


counts = {}

for token in TOKENS:
    if token != BLANK_TOKEN:
        counts[token] = 0


human_count = 0
non_human_count = 0


def get_file_info(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    label = None
    creation_method = None

    for element in root.iter():
        if not element.tag.endswith("annotation"):
            continue

        annotation_type = element.attrib.get("type")

        if annotation_type == "label":
            label = element.text

        elif annotation_type == "inkCreationMethod":
            creation_method = element.text

    return label, creation_method


ink_files = INK_FOLDER.glob("*.inkml")

total_files = 0

for file_path in ink_files:
    total_files += 1

    try:
        label, creation_method = get_file_info(file_path)

        if label in counts:
            counts[label] += 1

        if creation_method == "human":
            human_count += 1
        else:
            non_human_count += 1

    except Exception as error:
        print(f"Failed to read {file_path.name}: {error}")


print("\nSymbol counts:")
print("-------------------------")

for symbol, count in counts.items():
    print(f"{symbol}: {count}")

print("-------------------------")
print(f"Total InkML files scanned: {total_files}")
print(f"Matching symbol files: {sum(counts.values())}")

print("\nCreation method:")
print(f"Human: {human_count}")
print(f"Non-human: {non_human_count}")

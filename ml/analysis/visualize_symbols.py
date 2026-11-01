import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------

SYMBOL_TO_VISUALIZE = r"\div"

GRID_ROWS = 3
GRID_COLUMNS = 3

DOT_MARKER_SIZE = 35


# -------------------------
# File paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL_FOLDER = PROJECT_ROOT / "ml" / "data" / "mathwriting-2024" / "symbols"


def get_label(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for element in root.iter():
        if element.tag.endswith("annotation") and element.attrib.get("type") == "label":
            return element.text

    return None


def read_traces(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    traces = []

    for element in root.iter():
        if not element.tag.endswith("trace"):
            continue

        if not element.text:
            continue

        trace = []

        for raw_point in element.text.strip().split(","):
            values = raw_point.strip().split()

            x = float(values[0])
            y = float(values[1])
            t = float(values[2])

            trace.append((x, y, t))

        if trace:
            traces.append(trace)

    return traces


def draw_traces(traces, axis):

    for trace in traces:

        x_values = []
        y_values = []

        for x, y, t in trace:
            x_values.append(x)
            y_values.append(y)

        unique_points = set(
            zip(
                x_values,
                y_values,
            )
        )

        # Draw traces with one unique coordinate as a dot
        if len(unique_points) == 1:

            x, y = next(iter(unique_points))

            axis.scatter(
                x,
                y,
                color="black",
                s=DOT_MARKER_SIZE,
            )

        else:
            axis.plot(
                x_values,
                y_values,
                color="black",
                linewidth=2,
            )


def visualize_file_batch(file_paths):

    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLUMNS,
        figsize=(12, 9),
    )

    axes = axes.flatten()

    for index, axis in enumerate(axes):

        # Hide unused cells in the final batch
        if index >= len(file_paths):
            axis.axis("off")
            continue

        file_path = file_paths[index]

        traces = read_traces(file_path)

        draw_traces(
            traces,
            axis,
        )

        axis.invert_yaxis()
        axis.axis("equal")

        axis.set_title(
            file_path.name,
            fontsize=9,
        )

        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        f"Symbol: {SYMBOL_TO_VISUALIZE}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    selected_files = []

    for file_path in SYMBOL_FOLDER.glob("*.inkml"):

        label = get_label(file_path)

        if label == SYMBOL_TO_VISUALIZE:
            selected_files.append(file_path)

    selected_files.sort()

    print(f"Found {len(selected_files)} samples " f"for '{SYMBOL_TO_VISUALIZE}'.")

    files_per_window = GRID_ROWS * GRID_COLUMNS

    # Display 9 files at a time
    for start_index in range(
        0,
        len(selected_files),
        files_per_window,
    ):

        batch = selected_files[start_index : start_index + files_per_window]

        print(f"Showing samples " f"{start_index + 1}-" f"{start_index + len(batch)}")

        visualize_file_batch(batch)

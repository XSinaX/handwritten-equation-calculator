"""
Preprocessing Pipeline for Handwritten Digit Recognition (Task 1)
===================================================================
Implements and compares 5 preprocessing techniques:
1. Grayscaling
2. Resizing / Normalization
3. Binarization
4. Noise Removal
5. Deskewing

Usage:
    python preprocessing.py --input <folder_or_image> --output <output_folder>

If no input is given, it generates a synthetic sample image so you can
test the pipeline immediately, then swap in your real dataset later.
"""

import cv2
import numpy as np
import os
import argparse
import re
from typing import Optional
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# FILENAME LABEL PARSING (for datasets where class is encoded in the filename,
# e.g. "Eight_full (1).jpg", "Zero_full (999).jpg", "Nine_full (12).jpg")
# ---------------------------------------------------------------------------
WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def parse_label_from_filename(fname: str) -> Optional[str]:
    """Extract the digit class (0-9) from a filename whose class is spelled
    out as a word prefix, e.g. 'Eight_full (1).jpg' -> '8'.
    Returns None if no recognizable digit word is found."""
    first_word = re.split(r"[_\s(]", fname)[0].lower()
    return WORD_TO_DIGIT.get(first_word)


# ---------------------------------------------------------------------------
# 1. GRAYSCALING
# ---------------------------------------------------------------------------
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR/color image to single-channel grayscale.
    If already grayscale, returns it unchanged."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# 2. RESIZING / NORMALIZATION
# ---------------------------------------------------------------------------
def resize_and_normalize(image: np.ndarray, size=(28, 28)) -> np.ndarray:
    """Resize image to a fixed size and normalize pixel values to [0, 1].
    Returns a float32 array. Use `denormalize()` before saving/displaying."""
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def denormalize(image: np.ndarray) -> np.ndarray:
    """Convert a normalized [0,1] float image back to uint8 [0,255] for display/saving."""
    return (image * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. BINARIZATION
# ---------------------------------------------------------------------------
def binarize(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Convert grayscale image to binary (black/white) using one of:
    - 'otsu'      : Otsu's global thresholding (good default)
    - 'adaptive'  : Adaptive mean thresholding (good for uneven lighting)
    - 'fixed'     : Simple fixed threshold at 127
    """
    if image.dtype != np.uint8:
        image = denormalize(image)

    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    elif method == "fixed":
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return binary


# ---------------------------------------------------------------------------
# 4. NOISE REMOVAL
# ---------------------------------------------------------------------------
def remove_noise(image: np.ndarray, method: str = "median") -> np.ndarray:
    """Remove noise using one of:
    - 'median'    : Median blur (good for salt-and-pepper noise, preserves edges)
    - 'gaussian'  : Gaussian blur (smoother, can blur fine strokes)
    - 'morph'     : Morphological opening (removes small speckle noise from binary images)
    """
    if image.dtype != np.uint8:
        image = denormalize(image)

    if method == "median":
        return cv2.medianBlur(image, 3)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (3, 3), 0)
    elif method == "morph":
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    else:
        raise ValueError(f"Unknown noise removal method: {method}")


# ---------------------------------------------------------------------------
# 5. DESKEWING
# ---------------------------------------------------------------------------
def deskew(image: np.ndarray) -> np.ndarray:
    """Correct slant/skew in a binary image using image moments.
    Works best on a single-character binary image (post-binarization)."""
    if image.dtype != np.uint8:
        image = denormalize(image)

    m = cv2.moments(image)
    if abs(m["mu02"]) < 1e-2:
        # No meaningful skew detected
        return image.copy()

    skew = m["mu11"] / m["mu02"]
    h, w = image.shape
    M = np.array([[1, skew, -0.5 * w * skew], [0, 1, 0]], dtype=np.float32)
    deskewed = cv2.warpAffine(
        image, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )
    return deskewed


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------
def preprocess_pipeline(image: np.ndarray, target_size=(28, 28),
                         binarize_method="otsu", noise_method="median",
                         apply_deskew=True) -> dict:
    """
    Runs the full preprocessing pipeline step by step and returns
    every intermediate result, so you can visually compare each stage.
    """
    steps = {}
    steps["original"] = image.copy()

    gray = to_grayscale(image)
    steps["grayscale"] = gray

    resized_norm = resize_and_normalize(gray, size=target_size)
    steps["resized_normalized"] = denormalize(resized_norm)  # for display only

    denoised = remove_noise(denormalize(resized_norm), method=noise_method)
    steps["noise_removed"] = denoised

    binary = binarize(denoised, method=binarize_method)
    steps["binarized"] = binary

    if apply_deskew:
        final = deskew(binary)
        steps["deskewed"] = final
    else:
        steps["deskewed"] = binary

    return steps


# ---------------------------------------------------------------------------
# VISUALIZATION / COMPARISON
# ---------------------------------------------------------------------------
def visualize_steps(steps: dict, save_path: Optional[str] = None):
    """Plot each pipeline stage side by side for visual comparison."""
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    axes = np.atleast_1d(axes)

    for ax, (name, img) in zip(axes, steps.items()):
        cmap = "gray" if len(img.shape) == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(name.replace("_", " ").title(), fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison figure to {save_path}")
    plt.close(fig)


def compare_binarization_methods(gray_image: np.ndarray, save_path: Optional[str] = None):
    """Side-by-side comparison of the 3 binarization methods on the same image."""
    methods = ["otsu", "adaptive", "fixed"]
    results = {m: binarize(gray_image, method=m) for m in methods}

    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(3 * (len(methods) + 1), 3))
    axes[0].imshow(gray_image, cmap="gray")
    axes[0].set_title("Grayscale (input)")
    axes[0].axis("off")

    for ax, (name, img) in zip(axes[1:], results.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(name.title())
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved binarization comparison to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SYNTHETIC TEST IMAGE (used when no dataset is supplied yet)
# ---------------------------------------------------------------------------
def make_synthetic_sample() -> np.ndarray:
    """Creates a fake 'handwritten digit' image (a slanted stroke on noisy
    background) purely so the pipeline can be test-run before real data
    is wired in."""
    canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.ellipse(canvas, (50, 50), (25, 35), 15, 0, 360, (0, 0, 0), 4)
    noise = np.random.randint(0, 40, canvas.shape, dtype=np.uint8)
    noisy = cv2.subtract(canvas, noise)
    return noisy


# ---------------------------------------------------------------------------
# MAIN / CLI
# ---------------------------------------------------------------------------
def process_folder(input_dir: str, output_dir: str, target_size=(28, 28),
                    max_per_class: Optional[int] = None):
    """Runs the pipeline on every image in a FLAT folder where the class label
    is encoded in the filename (e.g. 'Eight_full (1).jpg'). Processed images
    are sorted into class subfolders under output_dir, and one comparison
    figure is saved per class (not just the first 3 files alphabetically)."""
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(valid_ext)]

    if not files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(files)} image(s). Processing...")

    per_class_count = {}
    comparison_saved = set()
    total_processed = 0
    total_skipped_unlabeled = 0
    total_failed = 0

    for fname in files:
        label = parse_label_from_filename(fname)
        if label is None:
            total_skipped_unlabeled += 1
            continue

        if max_per_class is not None and per_class_count.get(label, 0) >= max_per_class:
            continue

        path = os.path.join(input_dir, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"  Skipping unreadable file: {fname}")
            total_failed += 1
            continue

        steps = preprocess_pipeline(image, target_size=target_size)

        class_out_dir = os.path.join(output_dir, label)
        os.makedirs(class_out_dir, exist_ok=True)
        out_path = os.path.join(class_out_dir, fname)
        cv2.imwrite(out_path, steps["deskewed"])

        per_class_count[label] = per_class_count.get(label, 0) + 1
        total_processed += 1

        if label not in comparison_saved:
            fig_path = os.path.join(output_dir, f"comparison_class{label}_{os.path.splitext(fname)[0]}.png")
            visualize_steps(steps, save_path=fig_path)
            comparison_saved.add(label)

    print(f"\nDone. Processed {total_processed} image(s) across {len(per_class_count)} classes: {sorted(per_class_count.keys())}")
    if total_skipped_unlabeled:
        print(f"Skipped {total_skipped_unlabeled} file(s) with no recognizable class label in filename.")
    if total_failed:
        print(f"Failed to read {total_failed} file(s).")
    print(f"Output saved to: {output_dir} (sorted into class subfolders 0-9)")


def process_class_folders(input_dir: str, output_dir: str, target_size=(28, 28),
                           max_per_class: Optional[int] = None, save_comparisons_per_class: int = 1):
    """
    Runs the pipeline on a dataset organized as one subfolder per class label,
    e.g.:
        dataset/0/*.jpg
        dataset/1/*.jpg
        ...
        dataset/9/*.jpg

    This matches the digit-folder structure (0-9) most handwritten digit
    datasets use. Processed images are saved into the same class-folder
    structure under `output_dir`, so labels are never lost.

    Args:
        input_dir: path to the parent 'dataset' folder containing 0-9 subfolders.
        output_dir: where processed images will be written (mirrors input structure).
        target_size: size to resize each image to (default 28x28).
        max_per_class: optional cap on how many images to process per class
                        (useful for a quick test run before processing everything).
        save_comparisons_per_class: how many comparison figures to save per class
                                     folder (for visual sanity-checking).
    """
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    class_folders = [d for d in sorted(os.listdir(input_dir))
                      if os.path.isdir(os.path.join(input_dir, d))]

    if not class_folders:
        print(f"No class subfolders found in {input_dir}")
        return

    print(f"Found {len(class_folders)} class folders: {class_folders}")

    total_processed = 0
    total_failed = 0

    for label in class_folders:
        class_in_dir = os.path.join(input_dir, label)
        class_out_dir = os.path.join(output_dir, label)
        os.makedirs(class_out_dir, exist_ok=True)

        files = [f for f in sorted(os.listdir(class_in_dir)) if f.lower().endswith(valid_ext)]
        if max_per_class:
            files = files[:max_per_class]

        print(f"  Class '{label}': {len(files)} image(s)")

        for i, fname in enumerate(files):
            path = os.path.join(class_in_dir, fname)
            image = cv2.imread(path)
            if image is None:
                print(f"    Skipping unreadable file: {fname}")
                total_failed += 1
                continue

            steps = preprocess_pipeline(image, target_size=target_size)

            out_path = os.path.join(class_out_dir, fname)
            cv2.imwrite(out_path, steps["deskewed"])
            total_processed += 1

            if i < save_comparisons_per_class:
                fig_path = os.path.join(
                    output_dir, f"comparison_class{label}_{os.path.splitext(fname)[0]}.png"
                )
                visualize_steps(steps, save_path=fig_path)

    print(f"\nDone. Processed {total_processed} image(s), {total_failed} failed.")
    print(f"Output saved to: {output_dir} (same class-folder structure as input)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwritten digit preprocessing pipeline")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to a folder of images. If omitted, runs on a synthetic test image.")
    parser.add_argument("--output", type=str, default="/home/claude/preprocessing_output",
                         help="Folder to save processed images and comparison figures.")
    parser.add_argument("--size", type=int, default=28,
                         help="Target width/height in pixels (default 28, MNIST-style).")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Limit images processed per class folder (for a quick test run).")
    parser.add_argument("--flat", action="store_true",
                         help="Force flat-folder mode even if subfolders are detected.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.input and os.path.isdir(args.input):
        # Auto-detect class-folder structure (e.g. dataset/0/, dataset/1/, ...)
        subdirs = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        if subdirs and not args.flat:
            print(f"Detected {len(subdirs)} subfolders — treating as class-labeled dataset.")
            process_class_folders(args.input, args.output, target_size=(args.size, args.size),
                                   max_per_class=args.max_per_class)
        else:
            process_folder(args.input, args.output, target_size=(args.size, args.size),
                            max_per_class=args.max_per_class)
    else:
        print("No valid --input folder given. Running on a synthetic sample image instead.")
        sample = make_synthetic_sample()
        steps = preprocess_pipeline(sample, target_size=(args.size, args.size))
        visualize_steps(steps, save_path=os.path.join(args.output, "synthetic_comparison.png"))

        gray = to_grayscale(sample)
        compare_binarization_methods(gray, save_path=os.path.join(args.output, "binarization_comparison.png"))

        print(f"Synthetic test complete. Check: {args.output}")

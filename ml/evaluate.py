"""
Scores a trained model on data it never trained on.

    python -m ml.evaluate
    python -m ml.evaluate --image-model cnn --sequence-model gru

Reports:
  - generated/test: synthetic equations built from symbol drawings that
    were held out of training (unseen handwriting, synthetic layout)
  - human_test: real handwritten equations from MathWriting
    (unseen handwriting AND real layout - the closest to real use)
"""

import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from ml.models.ctc.decoder import constrained_decode, greedy_decode
from ml.models.ctc.tokenizer import CTCTokenizer
from ml.models.evaluation.evaluation import edit_distance
from shared.constrained_decoding import is_valid_equation

ML_ROOT = Path(__file__).resolve().parent

HELD_OUT_SETS = {
    "Synthetic test (unseen symbol drawings)": ML_ROOT / "data" / "generated" / "test",
    "Real handwriting (MathWriting human)": ML_ROOT / "data" / "human_test",
}

BATCH_SIZE = 64
EXAMPLES_TO_SHOW = 15


DECODERS = {
    "greedy": greedy_decode,
    "constrained": constrained_decode,
}

# Equation lengths (in symbols) grouped for the per-length breakdown
LENGTH_GROUPS = [(1, 4), (5, 9), (10, 14), (15, 19), (20, 99)]


def score_predictions(predictions, labels, tokenizer):
    correct = 0
    token_errors = 0
    target_tokens = 0
    mistakes = []
    confusions = Counter()
    skipped = 0

    # Accuracy on labels that are complete, valid equations (some human labels
    # are fragments like "1+" or "=0", which a calculator can't use anyway)
    complete_correct = 0
    complete_total = 0

    by_length = {group: [0, 0] for group in LENGTH_GROUPS}

    for prediction, label in zip(predictions, labels):
        try:
            target_ids = tokenizer.encode(str(label))
        except ValueError:
            skipped += 1
            continue

        target = tokenizer.decode(target_ids)
        predicted_ids = tokenizer.encode(prediction)

        errors = edit_distance(predicted_ids, target_ids)
        token_errors += errors
        target_tokens += len(target_ids)

        is_correct = prediction == target

        if is_valid_equation([tokenizer.tokens[i] for i in target_ids]):
            complete_total += 1
            complete_correct += is_correct

        for group in LENGTH_GROUPS:
            if group[0] <= len(target_ids) <= group[1]:
                by_length[group][0] += is_correct
                by_length[group][1] += 1

        if is_correct:
            correct += 1
        else:
            mistakes.append((target, prediction))

            # Same length -> count which symbols get swapped
            if len(predicted_ids) == len(target_ids):
                for target_id, predicted_id in zip(target_ids, predicted_ids):
                    if target_id != predicted_id:
                        confusions[(tokenizer.tokens[target_id], tokenizer.tokens[predicted_id])] += 1

    total = len(labels) - skipped

    return {
        "total": total,
        "sequence_accuracy": correct / max(1, total),
        "token_error_rate": token_errors / max(1, target_tokens),
        "complete_total": complete_total,
        "complete_accuracy": complete_correct / max(1, complete_total),
        "by_length": by_length,
        "mistakes": mistakes,
        "confusions": confusions,
        "skipped": skipped,
    }


def evaluate_folder(model, folder, tokenizer, device, decoders=("greedy", "constrained")):
    """Returns {decoder name: results} for every decoder in decoders."""

    X = np.load(folder / "X.npy")
    y = np.load(folder / "y.npy", allow_pickle=True)

    model.eval()

    predictions = {name: [] for name in decoders}

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            images = torch.from_numpy(X[start : start + BATCH_SIZE].astype(np.float32) / 255.0)
            images = images.unsqueeze(1).to(device)

            logits = model(images)

            for name in decoders:
                predictions[name].extend(DECODERS[name](logits, tokenizer))

    return {name: score_predictions(predictions[name], y, tokenizer) for name in decoders}


def print_report(name, result):
    print(f"\n===== {name} =====")
    print(f"Samples:           {result['total']}")
    print(f"Sequence accuracy: {result['sequence_accuracy'] * 100:.2f}%")
    print(f"Token error rate:  {result['token_error_rate'] * 100:.2f}%")

    if result["complete_total"] < result["total"]:
        print(
            f"Sequence accuracy on the {result['complete_total']} complete equations: "
            f"{result['complete_accuracy'] * 100:.2f}%"
        )

    lengths = [
        f"{low}-{high if high < 99 else '+'}: {right / count * 100:.0f}% (n={count})"
        for (low, high), (right, count) in result["by_length"].items()
        if count
    ]
    print("By length (symbols): " + " | ".join(lengths))

    if result["skipped"]:
        print(f"Skipped (unsupported label): {result['skipped']}")

    if result["confusions"]:
        print("Most common symbol swaps (true -> predicted):")
        for (true_token, predicted_token), count in result["confusions"].most_common(8):
            print(f"  {true_token:>7} -> {predicted_token:<7} x{count}")

    if result["mistakes"]:
        print("Example mistakes (true | predicted):")
        for target, prediction in result["mistakes"][:EXAMPLES_TO_SHOW]:
            print(f"  {target}  |  {prediction}")


def report_on_held_out_sets(model, tokenizer, device):
    for name, folder in HELD_OUT_SETS.items():
        if not (folder / "X.npy").exists():
            print(f"\n{name}: not found at {folder}, skipping.")
            continue

        for decoder, result in evaluate_folder(model, folder, tokenizer, device).items():
            print_report(f"{name} - {decoder} decoding", result)


def main():
    from ml.trainer import (
        IMAGE_MODEL,
        IMAGE_MODEL_CHOICES,
        SEQUENCE_MODEL,
        SEQUENCE_MODEL_CHOICES,
        choose_device,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-model", choices=IMAGE_MODEL_CHOICES, default=IMAGE_MODEL)
    parser.add_argument("--sequence-model", choices=SEQUENCE_MODEL_CHOICES, default=SEQUENCE_MODEL)
    arguments = parser.parse_args()

    model_path = ML_ROOT / "trained_models" / f"{arguments.image_model}-{arguments.sequence_model}.pt"

    device = choose_device()

    model = torch.load(model_path, map_location=device, weights_only=False)

    report_on_held_out_sets(model, CTCTokenizer(), device)


if __name__ == "__main__":
    main()

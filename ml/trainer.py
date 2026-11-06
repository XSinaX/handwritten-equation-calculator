from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.end_to_end.ctc.ctc_loss import EquationCTCLoss
from ml.end_to_end.ctc.decoder import greedy_decode
from ml.end_to_end.ctc.tokenizer import CTCTokenizer
from ml.end_to_end.equation_recognition_model import EquationRecognitionModel
from ml.end_to_end.evaluation import edit_distance

MODEL_PAIRS = {
    "cnn-gru": ("cnn", "gru"),
    "resnet-bilstm": ("resnet", "bilstm"),
    "vit-transformer": ("vision_transformer", "transformer"),
}
MODEL_CHOICES = {**MODEL_PAIRS}
DEFAULT_MODEL = "cnn-gru"

IMAGE_MODEL, SEQUENCE_MODEL = MODEL_PAIRS[DEFAULT_MODEL]
IMAGE_MODEL_CHOICES = [pair[0] for pair in MODEL_PAIRS.values()]
SEQUENCE_MODEL_CHOICES = [pair[1] for pair in MODEL_PAIRS.values()]

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001
GRADIENT_CLIP_NORM = 5.0
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 2
MIN_LEARNING_RATE = 1e-5
NUM_WORKERS = 0

ML_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = ML_ROOT / "datasets" / "equation"
MODEL_FOLDER = ML_ROOT / "trained_models"
RESULTS_FILE = MODEL_FOLDER / "results.json"
PREDICTIONS_FILE = MODEL_FOLDER / "test_predictions.csv"


class EquationDataset(Dataset):
    """Read preprocessed fixed-size images and raw equation labels from one split."""

    def __init__(self, split: str):
        path = DATASETS_ROOT / split / "preprocessed.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} dataset: {path}\n" "Run python -m ml.preprocess_image first.")

        with np.load(path, allow_pickle=False) as archive:
            self.images = archive["images"]
            self.equations = archive["equations"]
            self.sources = (
                archive["sources"]
                if "sources" in archive.files
                else np.asarray([f"{split}/{index}.png" for index in range(len(self.equations))], dtype=str)
            )

        if self.images.ndim != 3 or len(self.images) != len(self.equations) or len(self.sources) != len(self.images):
            raise ValueError(f"Invalid images/equations arrays in {path}")
        if len(self.images) == 0:
            raise ValueError(f"Empty {split} dataset: {path}")
        if self.images.dtype != np.uint8:
            raise ValueError(f"Expected uint8 images in {path}, got {self.images.dtype}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        # [H, W] uint8 -> [1, H, W] float32, with white=1 and black=0.
        image = torch.from_numpy(self.images[index].astype(np.float32) / 255.0)
        return image.unsqueeze(0), str(self.equations[index])


def sequence_accuracy(logits: torch.Tensor, labels, tokenizer: CTCTokenizer) -> int:
    """Count exact-match equations after CTC greedy decoding."""
    predictions = greedy_decode(logits.detach(), tokenizer)
    return sum(predicted == tokenizer.decode(tokenizer.encode(label)) for predicted, label in zip(predictions, labels))


def train_one_epoch(model, loader, tokenizer, criterion, optimizer, device) -> float:
    """Return online sequence accuracy across the training batches."""
    model.train()
    correct, total = 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        targets, target_lengths = tokenizer.encode_batch(labels)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets, target_lengths)

        with torch.no_grad():
            correct += sequence_accuracy(logits, labels, tokenizer)
            total += len(labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

    return correct / total


@torch.no_grad()
def evaluate(
    model,
    loader,
    tokenizer,
    criterion,
    device,
    with_loss: bool,
    with_token_error: bool = False,
    collect_predictions: bool = False,
) -> tuple[float, float | None, float | None, list[str]]:
    """Return exact-sequence accuracy, optional CTC loss, and optional token error rate."""
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    token_errors, target_tokens = 0, 0
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        predictions = greedy_decode(logits, tokenizer)
        if collect_predictions:
            all_predictions.extend(predictions)
        for prediction, label in zip(predictions, labels):
            target_ids = tokenizer.encode(label)
            correct += prediction == tokenizer.decode(target_ids)
            if with_token_error:
                predicted_ids = tokenizer.encode(prediction)
                token_errors += edit_distance(predicted_ids, target_ids)
                target_tokens += len(target_ids)
        total += len(labels)

        if with_loss:
            targets, target_lengths = tokenizer.encode_batch(labels)
            loss = criterion(logits, targets.to(device), target_lengths)
            loss_sum += loss.item() * len(labels)

    return (
        correct / total,
        loss_sum / total if with_loss else None,
        token_errors / target_tokens if with_token_error and target_tokens else None,
        all_predictions,
    )


def choose_device(gpu_index: int | None = None) -> torch.device:
    """Choose a GPU, preferring the device with most compute units."""
    if not torch.cuda.is_available():
        if gpu_index is not None:
            raise ValueError("--gpu was specified, but CUDA/ROCm is not available")
        print("Using CPU (PyTorch cannot access a CUDA/ROCm GPU).")
        return torch.device("cpu")

    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append((properties.multi_processor_count, index, properties.name))
        print(f"GPU {index}: {properties.name}")

    if gpu_index is not None:
        if not 0 <= gpu_index < len(gpus):
            raise ValueError(f"GPU index {gpu_index} is invalid")
        selected = gpu_index
    else:
        _, selected, _ = max(gpus)

    torch.cuda.set_device(selected)
    print(f"Using GPU {selected}")
    return torch.device(f"cuda:{selected}")


def train_end_to_end(model_name: str, device: torch.device) -> None:
    if model_name not in MODEL_PAIRS:
        raise ValueError(f"Choose one of: {', '.join(MODEL_PAIRS)}")
    image_name, sequence_name = MODEL_PAIRS[model_name]

    train_dataset = EquationDataset("train")
    validation_dataset = EquationDataset("validation")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    tokenizer = CTCTokenizer()
    model = EquationRecognitionModel(image_model=image_name, sequence_model=sequence_name).to(device)
    criterion = EquationCTCLoss(blank_index=tokenizer.blank_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    checkpoint_path = MODEL_FOLDER / f"{image_name}-{sequence_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_checkpoint_loss = float("inf")
    best_epoch = 0
    best_stopping_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Model: {model_name} | Train: {len(train_dataset):,} | " f"Validation: {len(validation_dataset):,}")

    for epoch in range(1, EPOCHS + 1):
        train_accuracy = train_one_epoch(model, train_loader, tokenizer, criterion, optimizer, device)
        validation_accuracy, validation_loss, _, _ = evaluate(
            model, validation_loader, tokenizer, criterion, device, with_loss=True
        )

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Seq Acc: {train_accuracy * 100:6.2f}% | "
            f"Val Seq Acc: {validation_accuracy * 100:6.2f}% | "
            f"Val Loss: {validation_loss:.4f}"
        )

        improved_accuracy = validation_accuracy > best_accuracy
        better_checkpoint = improved_accuracy or (
            validation_accuracy == best_accuracy and validation_loss < best_checkpoint_loss
        )
        if better_checkpoint:
            best_accuracy = validation_accuracy
            best_checkpoint_loss = validation_loss
            best_epoch = epoch
            torch.save(model, checkpoint_path)

        scheduler.step(validation_loss)
        if validation_loss < best_stopping_loss - EARLY_STOPPING_MIN_DELTA or improved_accuracy:
            best_stopping_loss = min(best_stopping_loss, validation_loss)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"\nBest epoch: {best_epoch} | Validation Seq Acc: {best_accuracy * 100:.2f}%")
    print(f"Best checkpoint: {checkpoint_path}")

    # Load the selected checkpoint and evaluate the held-out test split ONCE.
    best_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    test_dataset = EquationDataset("test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    # Full metrics at the selected validation checkpoint, not the last epoch.
    validation_accuracy, validation_loss, validation_ter, _ = evaluate(
        best_model,
        validation_loader,
        tokenizer,
        criterion,
        device,
        with_loss=True,
        with_token_error=True,
    )
    test_accuracy, test_loss, test_token_error_rate, predictions = evaluate(
        best_model,
        test_loader,
        tokenizer,
        criterion,
        device,
        with_loss=True,
        with_token_error=True,
        collect_predictions=True,
    )
    print(f"Test Seq Acc (best epoch {best_epoch}, n={len(test_dataset):,}): " f"{test_accuracy * 100:.2f}%")
    print(f"Test Token Error Rate: {test_token_error_rate * 100:.2f}%")

    result = {
        "architecture": "end_to_end",
        "model": model_name,
        "image_height": int(test_dataset.images.shape[1]),
        "image_width": int(test_dataset.images.shape[2]),
        "batch_size": BATCH_SIZE,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "validation_sequence_accuracy": validation_accuracy,
        "validation_loss": validation_loss,
        "validation_loss_type": "ctc",
        "validation_token_error_rate": validation_ter,
        "test_sequence_accuracy": test_accuracy,
        "test_loss": test_loss,
        "test_token_error_rate": test_token_error_rate,
        "checkpoint": str(checkpoint_path.relative_to(ML_ROOT)),
    }
    save_model_result(model_name, result)
    write_test_predictions(model_name, test_dataset.sources, test_dataset.equations, predictions, normalize=tokenizer)
    print(f"Metrics: {RESULTS_FILE} | Predictions: {PREDICTIONS_FILE}")


def save_model_result(model_name: str, metrics: dict) -> None:
    """Update only one model's record in the consolidated results JSON."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8")) if RESULTS_FILE.exists() else {}
    results[model_name] = metrics
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_test_predictions(model_name, sources, equations, predictions, normalize=None) -> None:
    """Merge prediction column on source path; overwrite same model on reruns.

    Preserve all other model columns, even when evaluated in a different order.
    """
    if not (len(sources) == len(equations) == len(predictions)):
        raise ValueError("Prediction CSV row count does not match test dataset")
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    other_columns = []
    if PREDICTIONS_FILE.exists():
        with PREDICTIONS_FILE.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not {"image", "equation"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Existing CSV has unexpected columns: {PREDICTIONS_FILE}")
            other_columns = [field for field in reader.fieldnames if field not in ("image", "equation")]
            for row in reader:
                if row["image"] in rows:
                    raise ValueError(f"Duplicate image key {row['image']!r} in {PREDICTIONS_FILE}")
                rows[row["image"]] = row

    seen = set()
    for path, label, prediction in zip(sources, equations, predictions):
        image_key = str(path).replace("\\", "/")
        if image_key in seen:
            raise ValueError(f"Duplicate image key in current dataset: {image_key}")
        seen.add(image_key)
        ground_truth = str(label)
        if normalize is not None:
            ground_truth = normalize.decode(normalize.encode(ground_truth))
        row = rows.setdefault(image_key, {"image": image_key, "equation": ground_truth})
        if row.get("equation") != ground_truth:
            raise ValueError(f"Conflicting reference equation for {image_key!r}")
        row[model_name] = prediction

    columns = ["image", "equation", *other_columns]
    if model_name not in columns:
        columns.append(model_name)
    # Missing predictions from runs using different source subsets are empty.
    with PREDICTIONS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows.values())


def main(model_names: list[str], gpu_index: int | None = None) -> None:
    bad = sorted(set(model_names) - set(MODEL_CHOICES))
    if bad:
        raise ValueError(f"Unknown model(s): {bad}; choose from {list(MODEL_CHOICES)}")
    device = choose_device(gpu_index)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    for selection in dict.fromkeys(model_names):
        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        print(f"\n{'='*65}\nSelected: {selection}\n{'='*65}", flush=True)
        if selection in MODEL_PAIRS:
            train_end_to_end(selection, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train end-to-end models")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=tuple(MODEL_CHOICES), help="Train one selected model")
    selection.add_argument("--models", nargs="+", choices=tuple(MODEL_CHOICES), help="Train selections sequentially")
    parser.add_argument("--all", action="store_true", help="Run all model selections")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA/ROCm GPU index")
    args = parser.parse_args()
    main(list(MODEL_CHOICES) if args.all else (args.models or [args.model or DEFAULT_MODEL]), gpu_index=args.gpu)

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from ml.models.equation_recognition_model import EquationRecognitionModel
from ml.models.ctc.tokenizer import CTCTokenizer
from ml.models.ctc.ctc_loss import EquationCTCLoss
from ml.models.evaluation.evaluation import evaluate_batch
from ml.data_pipeline.augmentation import augment_image
from ml.evaluate import report_on_held_out_sets

IMAGE_MODEL = "cnn"
SEQUENCE_MODEL = "bilstm"

IMAGE_MODEL_CHOICES = ["cnn", "resnet", "vision_transformer"]
SEQUENCE_MODEL_CHOICES = ["bilstm", "gru", "transformer"]

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# Stabilisers: cap the size of each update, and halve the learning rate when
# validation loss hasn't improved for a couple of epochs.
GRADIENT_CLIP_NORM = 5.0
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 2
MIN_LEARNING_RATE = 1e-5

# Background processes that prepare (and augment) batches while the GPU trains
NUM_WORKERS = min(4, os.cpu_count() or 1)

ML_ROOT = Path(__file__).resolve().parent
DATA_FOLDER = ML_ROOT / "data" / "generated"

# Train and validation are built from different symbol drawings
# (see equation_generator.py), so validation measures unseen handwriting.
TRAIN_FOLDER = DATA_FOLDER / "train"
VALIDATION_FOLDER = DATA_FOLDER / "val"

# Extra training data, each used when present (validation stays on the
# ink-built set):
#   image_generated - equations built from image datasets (image_symbols.py)
#   photo_generated - equations as the API sees them after a photo (photo_style.py)
EXTRA_TRAIN_FOLDERS = [
    ML_ROOT / "data" / "image_generated" / "train",
    ML_ROOT / "data" / "photo_generated" / "train",
]


class EquationDataset(Dataset):
    def __init__(self, folder, augment=False):
        self.x_path = folder / "X.npy"
        self.y = np.load(folder / "y.npy", allow_pickle=True)
        self.augment = augment

        # Images are opened lazily (memory-mapped) so each DataLoader worker
        # reads from disk instead of receiving its own copy of the whole array.
        self.X = None

    def __len__(self):
        return len(self.y)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["X"] = None
        return state

    def __getitem__(self, index):
        if self.X is None:
            self.X = np.load(self.x_path, mmap_mode="r")

        image = np.array(self.X[index])
        label = self.y[index]

        if self.augment:
            image = augment_image(image)

        # uint8 [0, 255] -> float32 [0, 1]
        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image)

        # Add grayscale channel:
        # [128, 128] -> [1, 128, 128]
        image = image.unsqueeze(0)

        return image, label


def train_one_epoch(
    model,
    dataloader,
    tokenizer,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0

    for images, labels in dataloader:

        images = images.to(device)

        # Convert target equations into CTC token IDs
        targets, target_lengths = tokenizer.encode_batch(labels)

        targets = targets.to(device)

        # Forward pass
        logits = model(images)

        loss = criterion(
            logits,
            targets,
            target_lengths,
        )

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


def validate(
    model,
    dataloader,
    tokenizer,
    criterion,
    device,
):
    model.eval()

    total_loss = 0

    correct_sequences = 0
    total_sequences = 0

    total_token_errors = 0
    total_target_tokens = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            targets, target_lengths = tokenizer.encode_batch(labels)

            targets = targets.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                targets,
                target_lengths,
            )

            total_loss += loss.item()

            (
                batch_correct_sequences,
                batch_total_sequences,
                batch_token_errors,
                batch_target_tokens,
            ) = evaluate_batch(
                logits,
                labels,
                tokenizer,
            )

            correct_sequences += batch_correct_sequences

            total_sequences += batch_total_sequences

            total_token_errors += batch_token_errors

            total_target_tokens += batch_target_tokens

    average_loss = total_loss / len(dataloader)

    sequence_accuracy = correct_sequences / total_sequences

    token_error_rate = total_token_errors / total_target_tokens

    return (
        average_loss,
        sequence_accuracy,
        token_error_rate,
    )


def choose_device(gpu_index=None):
    """
    Picks the GPU with the most compute units, so a dedicated graphics card
    wins over the small GPU built into the processor. (Memory size can't be
    used: a built-in GPU reports shared system RAM, which often looks bigger.)
    NVIDIA (CUDA) and AMD (ROCm) builds of PyTorch both report their GPUs
    through torch.cuda. Pass gpu_index to force a specific GPU.
    """

    if not torch.cuda.is_available():
        print("=" * 70)
        print("WARNING: no GPU available to PyTorch - training on the CPU.")
        print("This is typically 20-50x slower than a GPU.")
        print(f"Installed PyTorch: {torch.__version__}")
        print("If this PC has a GPU, install the GPU build of PyTorch (see README).")
        print("=" * 70)
        return torch.device("cpu")

    gpus = []

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append((properties.multi_processor_count, index, properties.name))
        print(
            f"Found GPU {index}: {properties.name} "
            f"({properties.multi_processor_count} compute units, "
            f"{properties.total_memory / 1024**3:.1f} GB)"
        )

    if gpu_index is not None:
        if not 0 <= gpu_index < len(gpus):
            raise ValueError(f"--gpu {gpu_index} does not exist; found {len(gpus)} GPU(s)")

        _, best_index, best_name = gpus[gpu_index]
    else:
        _, best_index, best_name = max(gpus)

    torch.cuda.set_device(best_index)
    print(f"Using device: cuda:{best_index} ({best_name})")

    return torch.device(f"cuda:{best_index}")


def main(
    image_model=IMAGE_MODEL,
    sequence_model=SEQUENCE_MODEL,
    gpu_index=None,
):

    torch.manual_seed(RANDOM_SEED)

    # -------------------------
    # Device
    # -------------------------
    device = choose_device(gpu_index)

    # Images are always the same size, so let the GPU library time its
    # options once and keep the fastest (convolutions and LSTM/GRU).
    torch.backends.cudnn.benchmark = True

    model_path = ML_ROOT / "trained_models" / f"{image_model}-{sequence_model}.pt"

    # -------------------------
    # Dataset
    # -------------------------

    train_dataset = EquationDataset(TRAIN_FOLDER, augment=True)

    extra_datasets = [
        EquationDataset(folder, augment=True) for folder in EXTRA_TRAIN_FOLDERS if (folder / "X.npy").exists()
    ]

    if extra_datasets:
        for folder in EXTRA_TRAIN_FOLDERS:
            if (folder / "X.npy").exists():
                print(f"Adding extra training data: {folder}")

        train_dataset = ConcatDataset([train_dataset, *extra_datasets])

    validation_dataset = EquationDataset(VALIDATION_FOLDER)

    training_size = len(train_dataset)
    validation_size = len(validation_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Training samples:   {training_size}")
    print(f"Validation samples: {validation_size}")

    # -------------------------
    # Tokenizer
    # -------------------------

    tokenizer = CTCTokenizer()

    # -------------------------
    # Model
    # -------------------------

    model = EquationRecognitionModel(
        image_model=image_model,
        sequence_model=sequence_model,
    )

    model = model.to(device)

    # -------------------------
    # CTC Loss
    # -------------------------

    criterion = EquationCTCLoss(
        blank_index=tokenizer.blank_index,
    )

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    # -------------------------
    # Training
    # -------------------------

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        training_loss = train_one_epoch(
            model,
            train_loader,
            tokenizer,
            criterion,
            optimizer,
            device,
        )

        (
            validation_loss,
            sequence_accuracy,
            token_error_rate,
        ) = validate(
            model,
            validation_loader,
            tokenizer,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {training_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Sequence Accuracy: {sequence_accuracy * 100:.2f}% | "
            f"Token Error Rate: {token_error_rate * 100:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.0e} | "
            f"Time: {time.time() - epoch_start:.0f}s"
        )

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss - EARLY_STOPPING_MIN_DELTA:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(
                model,
                model_path,
            )

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining complete.")
    print(f"Best model saved to: {model_path}")

    # Score the best model on data never used for training or model selection
    best_model = torch.load(model_path, map_location=device, weights_only=False)
    report_on_held_out_sets(best_model, tokenizer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-model",
        choices=IMAGE_MODEL_CHOICES,
        default=IMAGE_MODEL,
    )

    parser.add_argument(
        "--sequence-model",
        choices=SEQUENCE_MODEL_CHOICES,
        default=SEQUENCE_MODEL,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Index of the GPU to use (as listed in 'Found GPU N'). Default: the most powerful one.",
    )

    arguments = parser.parse_args()

    main(
        image_model=arguments.image_model,
        sequence_model=arguments.sequence_model,
        gpu_index=arguments.gpu,
    )

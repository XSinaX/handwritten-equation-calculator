from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from ml.models.equation_recognition_model import EquationRecognitionModel
from ml.models.ctc.tokenizer import CTCTokenizer
from ml.models.ctc.ctc_loss import EquationCTCLoss
from ml.models.evaluation.evaluation import evaluate_batch

IMAGE_MODEL = "cnn"
SEQUENCE_MODEL = "bilstm"

VALIDATION_RATIO = 0.2

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

ML_ROOT = Path(__file__).resolve().parent
MODEL_PATH = ML_ROOT / "trained_models" / f"{IMAGE_MODEL}-{SEQUENCE_MODEL}.pt"
DATA_FOLDER = ML_ROOT / "data" / "generated"

X_PATH = DATA_FOLDER / "X.npy"
Y_PATH = DATA_FOLDER / "y.npy"


class EquationDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path, allow_pickle=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]

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


def main():

    torch.manual_seed(RANDOM_SEED)

    # -------------------------
    # Device
    # -------------------------
    # TODO: CHECK THIS BEFORE RUNNING
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -------------------------
    # Dataset
    # -------------------------

    dataset = EquationDataset(
        X_PATH,
        Y_PATH,
    )

    validation_size = int(len(dataset) * VALIDATION_RATIO)
    training_size = len(dataset) - validation_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
        image_model=IMAGE_MODEL,
        sequence_model=SEQUENCE_MODEL,
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

    # -------------------------
    # Training
    # -------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
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
            f"Token Error Rate: {token_error_rate * 100:.2f}%"
        )

        if validation_loss < best_validation_loss - EARLY_STOPPING_MIN_DELTA:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(
                model,
                MODEL_PATH,
            )

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining complete.")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

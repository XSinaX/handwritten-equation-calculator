"""CNN/ResNet18 models, oracle-crop training, and full-pipeline inference."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from ml_segmentation.data_pipeline.preprocessing import stream_samples
from ml_segmentation.segmentation.evaluate_segmentation import require_aligned
from ml_segmentation.merger import merger as merging
from ml_segmentation.merger.evaluate_merger import evaluate_equation

DATA_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "equation"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
CLASSES = list("0123456789") + ["+", "-", "=", "(", ")", r"\times", r"\div", "NOISE"]
GEOMETRY_DIM = 8
IMAGE_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 100
BATCH_SIZE = 128
INFERENCE_BATCH_SIZE = 256
CNN_LEARNING_RATE = 1e-3
RESNET_LEARNING_RATE = 3e-4
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4
SEED = 42
CROP_CHUNK_SIZE = 2000


class Conv4(nn.Module):
    """Exactly four trainable convolution layers, followed by a classifier."""

    def __init__(self, num_classes: int, geometry_dim: int = 0):
        super().__init__()
        channels = [1, 32, 64, 96, 128]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(128 * 2 * 2 + geometry_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )
        self.geometry_dim = geometry_dim

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        visual = self.encoder(image)
        if self.geometry_dim:
            visual = torch.cat((visual, geometry), dim=1)
        return self.head(visual)


class ResNet18Classifier(nn.Module):
    """Grayscale ResNet18 trained from scratch; no pretrained external weights."""

    def __init__(self, num_classes: int, geometry_dim: int = 0):
        super().__init__()
        try:
            from torchvision.models import resnet18
        except Exception as exc:
            raise ImportError("ResNet18 requires torchvision compatible with torch") from exc
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.geometry_dim = geometry_dim
        self.head = nn.Sequential(
            nn.Linear(512 + geometry_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        visual = self.backbone(image)
        if self.geometry_dim:
            visual = torch.cat((visual, geometry), dim=1)
        return self.head(visual)


def center_resize(crop: np.ndarray, size: int, margin: int = 4) -> np.ndarray:
    """Tight bbox, aspect-preserving resize, white padding; white=255, black=0."""
    if size <= 2 * margin + 2:
        raise ValueError("size must exceed 2*margin+2")
    foreground = crop == 0
    if not np.any(foreground):
        return np.full((size, size), 255, dtype=np.uint8)
    yy, xx = np.nonzero(foreground)
    tight = crop[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    h, w = tight.shape
    scale = min((size - 2 * margin) / h, (size - 2 * margin) / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(tight, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out = np.full((size, size), 255, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    return out


def geometry_features(seg: Segment, full_shape: tuple[int, int]) -> np.ndarray:
    """Only observable geometry; no owner/noise labels or GT information."""
    h_img, w_img = full_shape
    x1, y1, x2, y2 = seg.bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    ink = (seg.image == 0).astype(np.uint8)
    count = int(ink.sum())
    components = cv2.connectedComponents(ink, connectivity=8)[0] - 1
    return np.asarray(
        [
            w / max(1, w_img),
            h / max(1, h_img),
            math.log1p(w),
            math.log1p(h),
            math.log1p(count),
            count / (w * h),
            min(components, 12) / 12.0,
            math.log((w + 1) / (h + 1)),
        ],
        dtype=np.float32,
    )


def training_crops(image, gt, noise, annotations, segmentation):
    """Keep the existing oracle-grouped, clean-complete symbol/noise targets."""
    anns = annotations
    require_aligned(image, gt, noise)
    candidate = merging.generate_candidates(image, gt, noise, annotations, segmentation, merging.TOLERANCE)
    if not candidate["recoverable"]:
        return
    groups = merging.group_by_owner(candidate["owners"], discard_unowned=False)
    final_segments = merging.groups_to_segments(candidate["segments"], groups)
    per_id = Counter()
    group_info = []
    for seg in final_segments:
        x1, y1, x2, y2 = seg.bbox
        fg = seg.image == 0
        owners = gt[y1:y2, x1:x2][fg]
        symbol_ids, counts = np.unique(owners[owners > 0], return_counts=True)
        foreign_noise = int(np.count_nonzero(owners == 0))
        if len(symbol_ids) == 0:
            group_info.append((seg, "NOISE", 0, 0))
        elif len(symbol_ids) == 1:
            ident = int(symbol_ids[0])
            purity = 1.0 - foreign_noise / max(1, len(owners))
            group_info.append((seg, str(anns[ident - 1]["token"]), ident, int(counts[0]) if purity >= 0.99 else 0))
            per_id[ident] += 1
        else:
            group_info.append((seg, None, -1, 0))
    total_ink = np.bincount(gt.ravel(), minlength=len(anns) + 1)
    for seg, token, ident, good_pixels in group_info:
        if token is None:
            continue
        if ident == 0:
            yield seg, "NOISE", 0
        elif per_id[ident] == 1 and good_pixels >= 0.98 * max(1, int(total_ink[ident])):
            yield seg, token, ident


def prepare_training_data(split: str, segmentation: dict) -> dict:
    packed, geometry, labels = [], [], []
    for _, image, gt, noise, _, annotations, _ in stream_samples(DATA_ROOT / split / "preprocessed.npz"):
        for segment, token, _ in training_crops(image, gt, noise, annotations, segmentation):
            crop = center_resize(segment.image, IMAGE_SIZE)
            packed.append(np.packbits(crop.reshape(-1) == 0, bitorder="little"))
            geometry.append(geometry_features(segment, image.shape))
            labels.append(CLASSES.index(token))
    if not labels:
        raise ValueError(f"No clean oracle-grouped training crops in {split}")
    return {
        "images": np.stack(packed),
        "geometry": np.asarray(geometry, np.float32),
        "labels": np.asarray(labels, np.int64),
    }


class PackedCrops(IterableDataset):
    """Keep crops bit-packed in memory, unpacking one original-sized chunk."""

    def __init__(self, data: dict, shuffle: bool, mean, std):
        self.data, self.shuffle, self.mean, self.std = data, shuffle, mean, std
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(SEED + self.epoch)
        starts = list(range(0, len(self.data["labels"]), CROP_CHUNK_SIZE))
        if self.shuffle:
            rng.shuffle(starts)
        for start in starts:
            end = start + CROP_CHUNK_SIZE
            bits = np.unpackbits(self.data["images"][start:end], axis=1, bitorder="little")
            images = bits[:, : IMAGE_SIZE * IMAGE_SIZE].reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
            geometry = (self.data["geometry"][start:end] - self.mean) / self.std
            labels = self.data["labels"][start:end]
            order = np.arange(len(labels))
            if self.shuffle:
                rng.shuffle(order)
            for i in order:
                yield images[i], geometry[i], int(labels[i])


def run_epoch(model, loader, criterion, optimizer, device, amp_enabled=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n = 0
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for images, geometry, labels in loader:
        images = images.to(device)
        geometry = geometry.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images, geometry)
                loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        bs = len(labels)
        total_loss += float(loss.item()) * bs
        n += bs
        y = labels.detach().cpu().numpy()
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        np.add.at(cm, (y, pred), 1)
    return total_loss / max(1, n), cm


def train_classifier(name: str, train: dict, validation: dict, output: Path, segmentation: dict) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device(DEVICE)
    model = CLASSIFICATION_METHODS[name](len(CLASSES)).to(device)
    geom = train["geometry"].astype(np.float64)
    mean = geom.mean(axis=0).astype(np.float32)
    std = np.sqrt(np.maximum(1e-6, (geom * geom).mean(axis=0) - geom.mean(axis=0) ** 2)).astype(np.float32)
    train_data = PackedCrops(train, True, mean, std)
    validation_data = PackedCrops(validation, False, mean, std)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, num_workers=0)
    counts = np.bincount(train["labels"], minlength=len(CLASSES)).astype(np.float64)
    median = float(np.median(counts[counts > 0]))
    weights = np.clip(np.where(counts > 0, np.sqrt(median / np.maximum(1, counts)), 1.0), 0.5, 4.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    lr = CNN_LEARNING_RATE if name == "cnn" else RESNET_LEARNING_RATE
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, meaningful, stale, history = math.inf, math.inf, 0, []
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        train_data.epoch = epoch
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _ = run_epoch(model, validation_loader, criterion, None, device)
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model": name,
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": IMAGE_SIZE,
                    "geometry_dim": GEOMETRY_DIM,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "training_source": "candidate_oracle",
                    "segmentation": dict(segmentation),
                    "best_epoch": epoch,
                    "validation_loss": best,
                },
                output / "model.pt",
            )
        if val_loss < meaningful - EARLY_STOPPING_MIN_DELTA:
            meaningful, stale = val_loss, 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_weighted_ce": train_loss,
                "validation_weighted_ce": val_loss,
                "no_improvement_epochs": stale,
            }
        )
        print(f"{name} epoch {epoch}: train {train_loss:.4f}, validation {val_loss:.4f}", flush=True)
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return {
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": lr,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "seed": SEED,
        "source": "candidate_oracle",
        "train_crops": len(train["labels"]),
        "validation_crops": len(validation["labels"]),
    }


def load_classifier(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing classifier {path}; run --split all first")
    saved = torch.load(path, map_location=DEVICE, weights_only=True)
    if saved.get("model") != name or saved.get("classes") != CLASSES or saved.get("image_size") != IMAGE_SIZE:
        raise ValueError(f"Classifier architecture, vocabulary, or crop size mismatch: {path}")
    model = CLASSIFICATION_METHODS[name](len(CLASSES)).to(DEVICE)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, saved


def collect_split(
    split: str, segmentation: dict, merger_name: str, merger_model, raster: int, threshold: float, classifier
) -> dict:
    """Predict every learned-merged crop; annotations affect only labels/flags."""
    from ml_segmentation.classification.post_classification import geometric_features, clean_label

    features, probabilities, labels, bboxes, stacked = [], [], [], [], []
    offsets, equations, pending, pending_geometry = [0], [], [], []

    def flush():
        if not pending:
            return
        raw = np.stack(pending).astype(np.float32)
        geometry = np.stack(pending_geometry).astype(np.float32)
        with torch.inference_mode():
            probs = (
                torch.softmax(
                    classifier(torch.from_numpy(raw[:, None]).to(DEVICE), torch.from_numpy(geometry).to(DEVICE)), dim=1
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        probabilities.extend(probs)
        pending.clear()
        pending_geometry.clear()

    merge = merging.MERGER_METHODS[merger_name]
    for idx, image, gt, noise, text, annotations, source in stream_samples(DATA_ROOT / split / "preprocessed.npz"):
        candidate = merging.generate_candidates(image, gt, noise, annotations, segmentation, merging.TOLERANCE)
        groups = merge(merger_model, candidate["segments"], image.shape, threshold, DEVICE, raster)
        row = evaluate_equation(
            idx=idx,
            split=split,
            source=source,
            equation=text,
            image=image,
            gt=gt,
            noise=noise,
            annotations=annotations,
            candidate=candidate,
            groups=groups,
        )
        row["reference_tokens"] = [str(a["token"]) for a in annotations]
        equations.append(row)
        segments = sorted(
            merging.groups_to_segments(candidate["segments"], groups), key=lambda s: (s.bbox[0], s.bbox[1])
        )
        symbol_counts = np.bincount(gt.ravel(), minlength=len(annotations) + 1) if split != "test" else None
        for segment in segments:
            ink = (center_resize(segment.image, IMAGE_SIZE) == 0).astype(np.uint8)
            geom, info = geometric_features(ink, segment.bbox, image.shape, 0.50, 2)
            features.append(geom)
            labels.append(clean_label(segment, gt, noise, annotations, symbol_counts) if split != "test" else -1)
            bboxes.append(segment.bbox)
            stacked.append(info["horizontally_stacked_components"])
            pending.append(ink)
            pending_geometry.append(geometry_features(segment, image.shape))
            if len(pending) >= INFERENCE_BATCH_SIZE:
                flush()
        offsets.append(len(features))
        if (idx + 1) % 500 == 0:
            print(f"{split}: {idx+1:,} equations, {len(features):,} crops", flush=True)
    flush()
    return {
        "features": np.asarray(features, np.float32).reshape(-1, 152),
        "probabilities": np.asarray(probabilities, np.float32).reshape(-1, len(CLASSES)),
        "labels": np.asarray(labels, np.int16),
        "offsets": np.asarray(offsets, np.int64),
        "eq_indices": np.asarray([r["sample_index"] for r in equations], np.int32),
        "equations": equations,
        "bboxes": np.asarray(bboxes, np.int32).reshape(-1, 4),
        "stacked": np.asarray(stacked, np.int16),
    }


def predict_segments(segments, image_shape: tuple[int, int], classifier) -> dict:
    """Prepare annotation-free inference using the same crop/features as collect_split."""
    from ml_segmentation.classification.post_classification import geometric_features

    features, crops, geometry, boxes, stacked = [], [], [], [], []
    for segment in sorted(segments, key=lambda s: (s.bbox[0], s.bbox[1])):
        ink = (center_resize(segment.image, IMAGE_SIZE) == 0).astype(np.uint8)
        feature, info = geometric_features(ink, segment.bbox, image_shape, 0.50, 2)
        features.append(feature)
        crops.append(ink)
        geometry.append(geometry_features(segment, image_shape))
        boxes.append(segment.bbox)
        stacked.append(info["horizontally_stacked_components"])
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(crops), INFERENCE_BATCH_SIZE):
            images = np.asarray(crops[start : start + INFERENCE_BATCH_SIZE], np.float32)
            geom = np.asarray(geometry[start : start + INFERENCE_BATCH_SIZE], np.float32)
            logits = classifier(torch.from_numpy(images[:, None]).to(DEVICE), torch.from_numpy(geom).to(DEVICE))
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
    return {
        "features": np.asarray(features, np.float32).reshape(-1, 152),
        "probabilities": np.asarray(probabilities, np.float32).reshape(-1, len(CLASSES)),
        "offsets": np.asarray([0, len(crops)], np.int64),
        "eq_indices": np.asarray([0], np.int32),
        "bboxes": np.asarray(boxes, np.int32).reshape(-1, 4),
        "stacked": np.asarray(stacked, np.int16),
    }


CLASSIFICATION_METHODS = {"cnn": Conv4, "resnet18": ResNet18Classifier}

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml_segmentation.data_pipeline.preprocessing import stream_samples
from ml_segmentation.segmentation.evaluate_segmentation import (
    analyze,
    PIXEL_TOLERANCE,
    COMPONENT_FRACTION,
    SYMBOL_FRACTION,
)
from ml_segmentation.segmentation.segmentation import Segment, SEGMENTATION_METHODS

DATA_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "equation"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4
MAX_TRAIN_PAIRS = 250_000
MAX_VALIDATION_PAIRS = 60_000
NEGATIVE_RATIO = 3.0
TRAIN_SEED = 1337
VALIDATION_SEED = 2027
RASTER_SIZE = 48
GEOMETRY_DIM = 26
THRESHOLDS = tuple(round(i / 100, 2) for i in range(70, 101))
SEGMENTATION_PARAMETERS = {"ratio": 0.35, "min_side_fraction": 0.18, "max_cuts": 1}
TOLERANCE = {
    "pixel_tolerance": PIXEL_TOLERANCE,
    "component_fraction": COMPONENT_FRACTION,
    "symbol_fraction": SYMBOL_FRACTION,
}


class GeometryMLP(nn.Module):
    def __init__(self, geometry_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geometry_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, geometry: torch.Tensor, pair_images: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(geometry).squeeze(1)


class PairCNNEncoder(nn.Module):
    """Small encoder trained for fragment affinity, not symbol classification."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(96 * 2 * 2, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class CNNGeometryAffinity(nn.Module):
    def __init__(self, geometry_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.encoder = PairCNNEncoder(embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(geometry_dim + embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, geometry: torch.Tensor, pair_images: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(pair_images)
        return self.head(torch.cat([geometry, embedding], dim=1)).squeeze(1)


def merge_segments(segments: list[Segment], members: list[int]) -> Segment:
    """Union disjoint candidate foreground pixels without adding box background."""
    if not members:
        raise ValueError("Cannot merge an empty group")
    chosen = [segments[i] for i in members]
    x1 = min(s.bbox[0] for s in chosen)
    y1 = min(s.bbox[1] for s in chosen)
    x2 = max(s.bbox[2] for s in chosen)
    y2 = max(s.bbox[3] for s in chosen)
    crop = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
    for s in chosen:
        sx1, sy1, sx2, sy2 = s.bbox
        target = crop[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1]
        np.minimum(target, s.image, out=target)
    return Segment(crop, (int(x1), int(y1), int(x2), int(y2)))


def groups_to_segments(segments: list[Segment], groups: list[list[int]]) -> list[Segment]:
    output = [merge_segments(segments, members) for members in groups if members]
    return sorted(output, key=lambda s: (s.bbox[0], s.bbox[1]))


def bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    dx = max(0, a[0] - b[2], b[0] - a[2])
    dy = max(0, a[1] - b[3], b[1] - a[3])
    return float(np.hypot(dx, dy))


def geometry_features(a: Segment, b: Segment, image_shape: tuple[int, int]) -> np.ndarray:
    """Scale-normalized pair geometry; no token labels or ground truth are inputs."""
    h_img, w_img = image_shape
    ax1, ay1, ax2, ay2 = map(float, a.bbox)
    bx1, by1, bx2, by2 = map(float, b.bbox)
    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    ref = max(4.0, (ah + bh) / 2)
    x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    y_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    x_gap = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    y_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
    area_a = float(np.count_nonzero(a.image == 0))
    area_b = float(np.count_nonzero(b.image == 0))
    density_a = area_a / max(1.0, aw * ah)
    density_b = area_b / max(1.0, bw * bh)
    ux1, uy1, ux2, uy2 = min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)
    feats = np.asarray(
        [
            aw / ref,
            ah / ref,
            bw / ref,
            bh / ref,
            area_a / (ref * ref),
            area_b / (ref * ref),
            density_a,
            density_b,
            (bcx - acx) / ref,
            (bcy - acy) / ref,
            abs(bcx - acx) / ref,
            abs(bcy - acy) / ref,
            x_gap / ref,
            y_gap / ref,
            x_overlap / min(aw, bw),
            y_overlap / min(ah, bh),
            np.hypot(x_gap, y_gap) / ref,
            np.log((aw + 1) / (bw + 1)),
            np.log((ah + 1) / (bh + 1)),
            np.log((area_a + 1) / (area_b + 1)),
            acx / max(1.0, w_img),
            bcx / max(1.0, w_img),
            acy / max(1.0, h_img),
            bcy / max(1.0, h_img),
            (ux2 - ux1) / ref,
            (uy2 - uy1) / ref,
        ],
        dtype=np.float32,
    )
    return feats


def pair_raster(a: Segment, b: Segment, size: int = 48, padding: int = 3) -> np.ndarray:
    """Two-channel binary crop preserving A/B shape and relative arrangement."""
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    x1, y1, x2, y2 = min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        raise ValueError("Invalid pair bounding box")
    layers = np.zeros((2, h, w), dtype=np.uint8)
    layers[0, ay1 - y1 : ay2 - y1, ax1 - x1 : ax2 - x1][a.image == 0] = 1
    layers[1, by1 - y1 : by2 - y1, bx1 - x1 : bx2 - x1][b.image == 0] = 1
    available = max(1, size - 2 * padding)
    scale = min(available / h, available / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    out = np.zeros((2, size, size), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    for c in range(2):
        resized = cv2.resize(layers[c], (nw, nh), interpolation=cv2.INTER_NEAREST)
        out[c, top : top + nh, left : left + nw] = resized
    return out


def all_pairs(n: int):
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a

    def groups(self) -> list[list[int]]:
        out = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return list(out.values())


def simple_threshold_groups(n: int, scores: dict[tuple[int, int], float], threshold: float) -> list[list[int]]:
    uf = UnionFind(n)
    for (i, j), score in scores.items():
        if score >= threshold:
            uf.union(i, j)
    return uf.groups()


def agglomerative_groups(
    n: int,
    scores: dict[tuple[int, int], float],
    threshold: float,
    linkage: str = "complete",
    min_cross_affinity: float | None = None,
) -> list[list[int]]:
    """Constrained agglomeration using learned pair affinities."""
    if linkage not in {"complete", "average"}:
        raise ValueError("linkage must be complete or average")
    clusters = [[i] for i in range(n)]
    floor = threshold if min_cross_affinity is None and linkage == "complete" else (min_cross_affinity or 0.0)

    def pair_score(i: int, j: int) -> float:
        return float(scores.get((min(i, j), max(i, j)), 0.0))

    def cluster_score(a: list[int], b: list[int]) -> tuple[float, float]:
        vals = [pair_score(i, j) for i in a for j in b]
        if not vals:
            return 0.0, 0.0
        return (min(vals) if linkage == "complete" else float(np.mean(vals)), min(vals))

    while True:
        best = None
        best_score = threshold
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score, minimum = cluster_score(clusters[i], clusters[j])
                if minimum < floor:
                    continue
                if score >= best_score:
                    best_score = score
                    best = (i, j)
        if best is None:
            break
        i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    return clusters


def group_by_owner(owners: list[int], discard_unowned: bool = True) -> list[list[int]]:
    """Ground-truth grouping for labelled classifier crops; not a merger method."""
    grouped = defaultdict(list)
    extras = []
    for i, owner in enumerate(owners):
        if owner > 0:
            grouped[owner].append(i)
        elif not discard_unowned:
            extras.append([i])
    return list(grouped.values()) + extras


def score_pairs(model, kind: str, raster_size: int, segments, image_shape, device: str) -> dict[tuple[int, int], float]:
    pairs = list(all_pairs(len(segments)))
    if not pairs:
        return {}
    geometry = np.stack([geometry_features(segments[i], segments[j], image_shape) for i, j in pairs])
    with torch.no_grad():
        g = torch.from_numpy(geometry).to(device)
        if kind == "cnn":
            px = np.stack([pair_raster(segments[i], segments[j], raster_size) for i, j in pairs]).astype(np.float32)
            p = torch.from_numpy(px).to(device)
        else:
            p = None
        scores = torch.sigmoid(model(g, p)).cpu().numpy()
    return {pair: float(score) for pair, score in zip(pairs, scores)}


def generate_candidates(
    image: np.ndarray, gt: np.ndarray, noise: np.ndarray, annotations: list, segmentation: dict, tolerance: dict
) -> dict:
    """Run the selected CCL-based method and match ink to annotated owners."""
    segments, proposals = SEGMENTATION_METHODS[segmentation["method"]](
        image, segmentation["ratio"], segmentation["min_side_fraction"], segmentation["max_cuts"]
    )
    metrics = analyze(
        image,
        gt,
        noise,
        annotations,
        segments,
        tolerance["pixel_tolerance"],
        tolerance["component_fraction"],
        tolerance["symbol_fraction"],
    )
    return {
        "segments": segments,
        "owners": [int(c["dominant_instance_id"]) for c in metrics["components"]],
        "candidate_metrics": metrics,
        "proposals": proposals,
        "recoverable": bool(metrics["grouping_recoverable"]),
    }


class PairDataset(Dataset):
    def __init__(self, data: dict, raster_size: int):
        self.geometry = data["geometry"].astype(np.float32)
        self.labels = data["labels"].astype(np.float32)
        self.packed = data["packed_pair_images"] if "packed_pair_images" in data else None
        self.raster_size = raster_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        geom = torch.from_numpy(self.geometry[index])
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        if self.packed is None:
            image = torch.empty(0, dtype=torch.float32)
        else:
            bits = np.unpackbits(self.packed[index], bitorder="little")[: 2 * self.raster_size * self.raster_size]
            image = torch.from_numpy(bits.reshape(2, self.raster_size, self.raster_size).astype(np.float32))
        return geom, image, label


def make_model(kind: str, raster_size: int):
    if kind == "geometry":
        return GeometryMLP(GEOMETRY_DIM)
    if kind == "cnn":
        return CNNGeometryAffinity(GEOMETRY_DIM, 64)
    raise ValueError(kind)


def generate_pairs(npz: Path, segmentation: dict, kind: str, max_pairs: int, seed: int) -> dict:
    """Generate the existing clean positive/negative pair sampling in memory."""
    tolerance, raster_size = TOLERANCE, RASTER_SIZE
    split = Path(npz).parent.name
    with_pixels = kind == "cnn"
    negative_ratio = NEGATIVE_RATIO
    rng = np.random.default_rng(seed)
    geometry = np.empty((max_pairs, GEOMETRY_DIM), np.float32)
    labels = np.empty((max_pairs,), np.uint8)
    packed = None
    packed_width = (2 * raster_size * raster_size + 7) // 8
    if with_pixels:
        packed = np.empty((max_pairs, packed_width), np.uint8)
    n = 0
    equations_seen = equations_used = 0
    recoverable_equations = 0
    clean_components_seen = 0
    excluded_components = 0
    for idx, image, gt, noise, equation, anns, source in stream_samples(npz, indices=None, limit=0):
        equations_seen += 1
        candidate = generate_candidates(image, gt, noise, anns, segmentation, tolerance)
        recoverable_equations += int(candidate["recoverable"])
        clean_ids = {
            i
            for i, c in enumerate(candidate["candidate_metrics"]["components"])
            if not c["significant_wrong_merge"] and int(c["overlap_pixels"]) == 0
        }
        clean_components_seen += len(clean_ids)
        excluded_components += len(candidate["segments"]) - len(clean_ids)
        positives = []
        negatives = []
        for i, j in all_pairs(len(candidate["segments"])):
            if i not in clean_ids or j not in clean_ids:
                continue
            oi, oj = candidate["owners"][i], candidate["owners"][j]
            if oi > 0 and oi == oj:
                positives.append((i, j, 1))
            elif oi != oj and (oi > 0 or oj > 0):
                negatives.append((i, j, 0))
        rng.shuffle(negatives)
        keep_neg = min(len(negatives), max(4, int(round(max(1, len(positives)) * negative_ratio))))
        chosen = positives + negatives[:keep_neg]
        rng.shuffle(chosen)
        if chosen:
            equations_used += 1
        for i, j, label in chosen:
            if n >= max_pairs:
                break
            a, b = candidate["segments"][i], candidate["segments"][j]
            geometry[n] = geometry_features(a, b, image.shape)
            labels[n] = label
            if with_pixels:
                bits = pair_raster(a, b, raster_size).reshape(-1)
                packed[n] = np.packbits(bits, bitorder="little")[:packed_width]
            n += 1
        if n >= max_pairs:
            break
        if equations_seen % 1000 == 0:
            print(f"  pairs {split}: {equations_seen:,} equations, {n:,}/{max_pairs:,} pairs", flush=True)
    if n == 0:
        raise ValueError(
            "No supervised pairs were produced from the training split. "
            f"Read {equations_seen} equations and found {clean_components_seen} clean "
            f"components ({excluded_components} excluded). Check that "
            "ml_segmentation/datasets/equation/<split>/preprocessed.npz contains correctly "
            "aligned nonzero symbol instance IDs; inspect candidate metrics if "
            "all components are marked contaminated."
        )
    summary = {
        "equations_seen": equations_seen,
        "equations_used": equations_used,
        "pairs": n,
        "positive_pairs": int(labels[:n].sum()),
        "negative_pairs": int(n - labels[:n].sum()),
        "recoverable_equations_seen": recoverable_equations,
        "clean_components_seen": clean_components_seen,
        "excluded_contaminated_components": excluded_components,
    }
    data = {"geometry": geometry[:n], "labels": labels[:n], "summary": summary}
    if with_pixels:
        data["packed_pair_images"] = packed[:n]
    return data


def train_model(
    kind: str, train_pairs: dict, validation_pairs: dict, checkpoint: Path, segmentation: dict, merger: str
) -> Path:
    """Train in-memory pairs; save the lowest validation-loss checkpoint."""
    raster_size, epochs, batch_size = RASTER_SIZE, MAX_EPOCHS, BATCH_SIZE
    lr, device = LEARNING_RATE, DEVICE
    early_stopping_patience = EARLY_STOPPING_PATIENCE
    early_stopping_min_delta = EARLY_STOPPING_MIN_DELTA
    train = PairDataset(train_pairs, raster_size)
    val = PairDataset(validation_pairs, raster_size)
    model = make_model(kind, raster_size).to(device)
    positives = float(train.labels.sum())
    negatives = len(train) - positives
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    last_meaningful_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    print(
        f"Training {kind} affinity model for up to {epochs} epochs "
        f"(early stopping: patience={early_stopping_patience}, "
        f"min_delta={early_stopping_min_delta:g}; monitor=validation weighted BCE)",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for geom, pixels, y in train_loader:
            geom, y = geom.to(device), y.to(device)
            pixels = pixels.to(device) if kind == "cnn" else None
            optimizer.zero_grad(set_to_none=True)
            logits = model(geom, pixels)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(y)
            count += len(y)
        model.eval()
        vloss = 0.0
        vcount = 0
        correct = 0
        with torch.no_grad():
            for geom, pixels, y in val_loader:
                geom, y = geom.to(device), y.to(device)
                pixels = pixels.to(device) if kind == "cnn" else None
                logits = model(geom, pixels)
                loss = criterion(logits, y)
                vloss += float(loss.item()) * len(y)
                vcount += len(y)
                correct += int(((torch.sigmoid(logits) >= 0.5) == (y >= 0.5)).sum().item())
        vloss /= max(1, vcount)
        vacc = correct / max(1, vcount)
        train_loss = total / max(1, count)
        improved = vloss < best_loss
        meaningful_improvement = vloss < last_meaningful_loss - early_stopping_min_delta
        if improved:
            best_loss = vloss
            best_epoch = epoch
            torch.save(
                {
                    "kind": kind,
                    "model_state_dict": model.state_dict(),
                    "geometry_dim": GEOMETRY_DIM,
                    "raster_size": raster_size,
                    "best_epoch": best_epoch,
                    "validation_loss": best_loss,
                    "segmentation": dict(segmentation),
                    "merger": merger,
                },
                checkpoint,
            )
        if meaningful_improvement:
            last_meaningful_loss = vloss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": round(train_loss, 8),
                "val_weighted_bce": round(vloss, 8),
                "val_pair_accuracy": round(vacc, 8),
                "checkpoint_improved": int(improved),
                "meaningful_improvement": int(meaningful_improvement),
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        print(
            f"Epoch {epoch:03d}/{epochs} | train loss {train_loss:.4f} "
            f"| val loss {vloss:.4f} | pair acc {vacc*100:.2f}% "
            f"| no meaningful improvement {epochs_without_improvement}/{early_stopping_patience}",
            flush=True,
        )
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}; best epoch {best_epoch}, " f"best val loss {best_loss:.6f}.",
                flush=True,
            )
            break

    history_path = checkpoint.parent / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(
        f"Best affinity checkpoint: {checkpoint} " f"(epoch {best_epoch}, val loss {best_loss:.6f})",
        flush=True,
    )
    print(f"Training history: {history_path}", flush=True)
    return checkpoint


def load_affinity(checkpoint: Path, device: str = DEVICE, segmentation: dict | None = None):
    """Load trained weights, checking candidate settings when supplied."""
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if segmentation is not None and saved.get("segmentation") != segmentation:
        raise ValueError(f"Checkpoint segmentation differs from the requested settings: {checkpoint}")
    kind, raster = saved["kind"], int(saved["raster_size"])
    model = make_model(kind, raster).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, kind, raster, saved


def geometry_mlp_simple(model, segments, image_shape, threshold, device=DEVICE, raster_size=RASTER_SIZE, scores=None):
    """Geometry MLP affinity followed by transitive simple grouping."""
    if scores is None:
        scores = score_pairs(model, "geometry", raster_size, segments, image_shape, device)
    return simple_threshold_groups(len(segments), scores, threshold)


def cnn_mlp_simple(model, segments, image_shape, threshold, device=DEVICE, raster_size=RASTER_SIZE, scores=None):
    """Shared CNN + geometry MLP affinity followed by simple grouping."""
    if scores is None:
        scores = score_pairs(model, "cnn", raster_size, segments, image_shape, device)
    return simple_threshold_groups(len(segments), scores, threshold)


def cnn_mlp_agglomerative(model, segments, image_shape, threshold, device=DEVICE, raster_size=RASTER_SIZE, scores=None):
    """The same CNN + MLP architecture with complete-linkage agglomeration."""
    if scores is None:
        scores = score_pairs(model, "cnn", raster_size, segments, image_shape, device)
    return agglomerative_groups(len(segments), scores, threshold)


MERGER_METHODS = {
    "geometry_mlp_simple": geometry_mlp_simple,
    "cnn_mlp_simple": cnn_mlp_simple,
    "cnn_mlp_agglomerative": cnn_mlp_agglomerative,
}

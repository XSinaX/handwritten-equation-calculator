"""Operator MLP, syntax repair, and end-only equals adjustment."""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_segmentation.classification.model import CLASSES

INDEX = {s: i for i, s in enumerate(CLASSES)}
GROUPS = {"operators": ["+", r"\div", "="]}
DIGITS = frozenset("0123456789")
OPERATORS = frozenset({"+", "-", r"\times", r"\div"})
PARENS_ONE = frozenset({"1", "(", ")"})
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 60
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 1e-4
TRAIN_EQUATION_LIMIT = 8000
VALIDATION_EQUATION_LIMIT = 2000
MIN_TRAINING_CROPS = 50
SYNTAX_SETTINGS = {"max_flips": 3, "max_positions": 12, "min_alt_prob": 0.01, "edit_penalty": 0.15}


def _safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def x_overlap_fraction(a: dict, b: dict) -> float:
    """Intersection of horizontal spans / narrower width (vertical order irrelevant)."""
    overlap = max(0.0, min(a["right"], b["right"]) - max(a["left"], b["left"]))
    return _safe_ratio(overlap, min(a["width"], b["width"]))


def geometric_features(
    ink: np.ndarray,
    bbox: tuple[int, int, int, int],
    equation_shape: tuple[int, int],
    overlap: float = 0.50,
    min_area: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Observable connected-component geometry; 1=ink on 64x64 CNN input."""
    ink = (np.asarray(ink) > 0).astype(np.uint8)
    if ink.ndim != 2:
        raise ValueError("Expected 2D binary CNN input")
    h, w = ink.shape
    n, labels, stats, centers = cv2.connectedComponentsWithStats(ink, connectivity=8)
    items: list[dict[str, float]] = []
    for i in range(1, n):
        left, top, ww, hh, area = [int(v) for v in stats[i]]
        items.append(
            dict(
                left=left,
                top=top,
                right=left + ww,
                bottom=top + hh,
                width=ww,
                height=hh,
                area=area,
                cx=float(centers[i, 0]),
                cy=float(centers[i, 1]),
            )
        )
    good = [r for r in items if r["area"] >= min_area]
    ranked = sorted(good, key=lambda r: (-r["area"], r["top"], r["left"]))
    anchor = ranked[0] if ranked else None
    core = [anchor] if anchor is not None else []
    if anchor is not None:
        core += [r for r in ranked[1:] if x_overlap_fraction(anchor, r) >= overlap]
    core = sorted(core, key=lambda r: (r["cy"], r["cx"]))
    k = len(core)
    total_area = float(ink.sum())
    core_area = sum(float(r["area"]) for r in core)
    x1, y1, x2, y2 = bbox
    original_w, original_h = max(1, x2 - x1), max(1, y2 - y1)
    eq_h, eq_w = equation_shape
    ys, xs = np.nonzero(ink)
    if len(xs):
        fw, fh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        lx, rx, ty, by = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        ink_region = ink[ty:by, lx:rx]
    else:
        fw, fh = 0, 0
        ink_region = ink
    row_sum = ink_region.sum(axis=1).astype(float)
    col_sum = ink_region.sum(axis=0).astype(float)
    row_max = float(row_sum.max()) if len(row_sum) else 0
    col_max = float(col_sum.max()) if len(col_sum) else 0
    features: list[float] = [
        math.log1p(original_w),
        math.log1p(original_h),
        _safe_ratio(original_w, eq_w),
        _safe_ratio(original_h, eq_h),
        math.log((original_w + 1) / (original_h + 1)),
        _safe_ratio(fw, w),
        _safe_ratio(fh, h),
        _safe_ratio(fw, fh),
        _safe_ratio(total_area, w * h),
        math.log1p(total_area),
        min(len(items), 12) / 12,
        min(len(good), 12) / 12,
        min(k, 12) / 12,
        min(max(0, len(good) - k), 12) / 12,
        _safe_ratio(core_area, total_area),
        _safe_ratio(anchor["area"], total_area) if anchor else 0,
        _safe_ratio(sum(r["area"] for r in ranked[1:]), total_area),
        _safe_ratio(sum(r["area"] for r in items if r["area"] < min_area), total_area),
        _safe_ratio(sum(row_sum >= 0.60 * row_max), len(row_sum)) if row_max else 0,
        _safe_ratio(sum(col_sum >= 0.60 * col_max), len(col_sum)) if col_max else 0,
    ]
    for r in ranked[:4]:
        x_overlap = x_overlap_fraction(anchor, r) if anchor else 0
        features.extend(
            [
                _safe_ratio(r["area"], total_area),
                _safe_ratio(r["width"], w),
                _safe_ratio(r["height"], h),
                _safe_ratio(r["cx"], w),
                _safe_ratio(r["cy"], h),
                _safe_ratio(r["width"], r["height"]),
                x_overlap,
                _safe_ratio(r["cy"] - anchor["cy"], h) if anchor else 0,
            ]
        )
    features.extend([0.0] * (8 * (4 - min(len(ranked), 4))))
    for r in core[:4]:
        features.extend(
            [
                _safe_ratio(r["area"], total_area),
                _safe_ratio(r["width"], w),
                _safe_ratio(r["height"], h),
                _safe_ratio(r["cx"], w),
                _safe_ratio(r["cy"], h),
            ]
        )
    features.extend([0.0] * (5 * (4 - min(k, 4))))
    # Pairwise x-overlap and vertical separation of the leading core items.
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        if k > j:
            a, b = core[i], core[j]
            vertical_gap = max(0, b["top"] - a["bottom"])
            features.extend([x_overlap_fraction(a, b), _safe_ratio(vertical_gap, h), _safe_ratio(b["cy"] - a["cy"], h)])
        else:
            features.extend([0.0] * 3)

    for band in np.array_split(ink_region, 3, axis=0):
        features.append(_safe_ratio(float(band.sum()), total_area))
    features.extend(
        [
            _safe_ratio(row_max, fw),
            _safe_ratio(col_max, fh),
            _safe_ratio(float(np.count_nonzero(row_sum)), fh),
            _safe_ratio(float(np.count_nonzero(col_sum)), fw),
        ]
    )
    if ink_region.size and ink_region.shape[0] and ink_region.shape[1]:
        shape_grid = cv2.resize(ink_region.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)
    else:
        shape_grid = np.zeros((8, 8), dtype=np.float32)
    features.extend(float(v) for v in shape_grid.ravel())
    arr = np.asarray(features, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise AssertionError("Nonfinite component feature")
    return arr, {
        "total_components": len(items),
        "size_filtered_components": len(good),
        "horizontally_stacked_components": k,
        "discarded_components": len(good) - k,
        "overlap_threshold": overlap,
    }


class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.12), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def clean_label(seg, gt: np.ndarray, noise: np.ndarray, anns: list[dict], symbol_counts: np.ndarray) -> int:
    """Train/validation ONLY. Reject mixed, incomplete, or stray segments."""
    x1, y1, x2, y2 = (int(v) for v in seg.bbox)
    ink = seg.image == 0
    if not np.any(ink):
        return -1
    owned = gt[y1:y2, x1:x2][ink]
    valid = owned[(owned > 0) & (owned <= len(anns))]
    if not len(valid):
        return -1
    ids, counts = np.unique(valid, return_counts=True)
    j = int(np.argmax(counts))
    owner = int(ids[j])
    if float(counts[j]) / len(owned) < 0.94:
        return -1
    if float(counts[j]) / max(1, int(symbol_counts[owner])) < 0.85:
        return -1
    label = str(anns[owner - 1]["token"])
    return INDEX.get(label, -1)


def apply_operator_mlp(
    original: np.ndarray, probs: np.ndarray, rerank: np.ndarray, alpha: float, margin: float
) -> np.ndarray:
    """Only modifies existing operator predictions among +, division and =."""
    new = original.copy()
    cls_ids = np.asarray([INDEX[t] for t in GROUPS["operators"]], dtype=np.int32)
    base_group = probs[:, cls_ids]
    base_group = base_group / np.clip(base_group.sum(axis=1, keepdims=True), 1e-10, None)
    mixture = (1 - alpha) * base_group + alpha * rerank
    selected_id = cls_ids[np.argmax(mixture, axis=1)]
    allowed = np.isin(original, cls_ids)
    old_pos = np.argmax(cls_ids[None, :] == original[:, None], axis=1)
    old_p = mixture[np.arange(len(original)), old_pos]
    new_p = mixture.max(axis=1)
    do_change = allowed & (selected_id != original) & ((new_p - old_p) >= margin)
    new[do_change] = selected_id[do_change]
    return new


def valid_expression(tokens: list[str]) -> bool:
    """Expression grammar, not arithmetic equality. Optional trailing '='."""
    if tokens and tokens[-1] == "=":
        tokens = tokens[:-1]
    if not tokens or "=" in tokens:
        return False
    pos = 0

    def factor() -> bool:
        nonlocal pos
        while pos < len(tokens) and tokens[pos] in ("+", "-"):
            pos += 1
        if pos >= len(tokens):
            return False
        if tokens[pos] == "(":
            pos += 1
            if not expr() or pos >= len(tokens) or tokens[pos] != ")":
                return False
            pos += 1
            return True
        if tokens[pos] in DIGITS:
            while pos < len(tokens) and tokens[pos] in DIGITS:
                pos += 1
            return True
        return False

    def term() -> bool:
        nonlocal pos
        if not factor():
            return False
        while pos < len(tokens) and tokens[pos] in (r"\times", r"\div"):
            pos += 1
            if not factor():
                return False
        return True

    def expr() -> bool:
        nonlocal pos
        if not term():
            return False
        while pos < len(tokens) and tokens[pos] in ("+", "-"):
            pos += 1
            if not term():
                return False
        return True

    return expr() and pos == len(tokens)


def balanced_parentheses(tokens: list[str]) -> bool:
    depth = 0
    for tok in tokens:
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def repair_one_parentheses(
    tokens: list[str],
    probs: np.ndarray,
    max_flips: int = 3,
    max_positions: int = 12,
    min_alt_prob: float = 0.01,
    edit_penalty: float = 0.15,
) -> tuple[list[str], list[dict]]:
    """Repair invalid syntax via up to max_flips replacements of 1, ( or )."""
    if probs.shape != (len(tokens), len(CLASSES)):
        raise ValueError("Probabilities must have one 18-class row per token")
    if max_flips < 1 or max_positions < 1 or not (0 <= min_alt_prob <= 1) or edit_penalty < 0:
        raise ValueError("Invalid syntax-repair configuration")
    if not tokens or valid_expression(tokens) or not any(t in ("(", ")") for t in tokens):
        return tokens.copy(), []
    eps = 1e-12
    positions = [i for i, token in enumerate(tokens) if token in PARENS_ONE]
    alternatives = {}
    for pos in positions:
        original = tokens[pos]
        choices = [t for t in ("1", "(", ")") if t != original]
        scored = []
        old_p = float(probs[pos, INDEX[original]])
        for other in choices:
            new_p = float(probs[pos, INDEX[other]])
            if new_p >= min_alt_prob:
                scored.append((other, math.log(max(new_p, eps)) - math.log(max(old_p, eps)) - edit_penalty))
        if scored:
            alternatives[pos] = scored
    rank = sorted(alternatives, key=lambda i: (-max(score for _, score in alternatives[i]), tokens[i] == "1", i))[
        :max_positions
    ]
    rank.sort()
    best_key = None
    best_changes = None
    for n_changes in range(1, min(max_flips, len(rank)) + 1):
        for selected in itertools.combinations(rank, n_changes):
            for options in itertools.product(*(alternatives[pos] for pos in selected)):
                candidate = tokens.copy()
                change_list = []
                score = 0.0
                for pos, (alt, delta) in zip(selected, options):
                    candidate[pos] = alt
                    change_list.append((pos, alt))
                    score += delta
                if not balanced_parentheses(candidate) or not valid_expression(candidate):
                    continue
                key = (score, -n_changes, tuple(-pos for pos, _ in change_list))
                if best_key is None or key > best_key:
                    best_key = key
                    best_changes = change_list
    if best_changes is None:
        return tokens.copy(), []
    result = tokens.copy()
    changes = []
    for pos, alt in best_changes:
        before = tokens[pos]
        result[pos] = alt
        changes.append(
            {
                "token_position": pos,
                "from": before,
                "to": alt,
                "original_probability": float(probs[pos, INDEX[before]]),
                "alternative_probability": float(probs[pos, INDEX[alt]]),
            }
        )
    return result, changes


def fused_operator_scores(
    resnet_probs: np.ndarray,
    operator_probs: np.ndarray,
    alpha: float,
    classes: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Make a ranking distribution consistent with operator MLP's mixture."""
    arr = np.asarray(resnet_probs, dtype=np.float64)
    rerank = np.asarray(operator_probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(classes):
        raise ValueError("ResNet posterior shape must be (n_crops, n_classes)")
    if rerank.shape != (len(arr), 3):
        raise ValueError("Operator-MLP posterior must have three columns: +, div, =")
    if not 0 <= alpha <= 1:
        raise ValueError("Operator mixture alpha must be in [0, 1]")
    ids = [classes.index("+"), classes.index(r"\div"), classes.index("=")]
    group = arr[:, ids]
    mass = group.sum(axis=1, keepdims=True)
    conditional = group / np.clip(mass, 1e-10, None)
    fused = arr.copy()
    fused[:, ids] = mass * ((1 - alpha) * conditional + alpha * rerank)
    return fused


def apply_equals_at_end_only(
    predictions: np.ndarray,
    probs: np.ndarray,
    offsets: np.ndarray,
    equation_indices: np.ndarray,
    classes: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, list[dict]]:
    """For each non-final visible '=', choose strongest alternative != '='."""
    original = np.asarray(predictions)
    probs = np.asarray(probs)
    offsets = np.asarray(offsets)
    indices = np.asarray(equation_indices)
    if original.ndim != 1 or probs.shape != (len(original), len(classes)):
        raise ValueError("Predictions and probability arrays have incompatible shapes")
    if len(offsets) != len(indices) + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != len(original):
        raise ValueError("Invalid equation offsets")
    equals_id, noise_id = classes.index("="), classes.index("NOISE")
    changed = original.copy()
    edits: list[dict] = []
    for eq_n, eq_id in enumerate(indices):
        lo, hi = int(offsets[eq_n]), int(offsets[eq_n + 1])
        visible = [i for i in range(lo, hi) if int(original[i]) != noise_id]
        for pos in visible[:-1]:
            if int(original[pos]) != equals_id:
                continue
            scores = probs[pos].copy()
            scores[equals_id] = -np.inf
            replacement = int(np.argmax(scores))
            changed[pos] = replacement
            edits.append(
                {
                    "equation_index": int(eq_id),
                    "crop_index": int(pos),
                    "segment_index": int(pos - lo),
                    "from": "=",
                    "to": classes[replacement],
                    "equals_probability": float(probs[pos, equals_id]),
                    "replacement_probability": float(probs[pos, replacement]),
                }
            )
    return changed, edits


def take_equations(data: dict, limit: int) -> dict:
    count = min(limit, len(data["equations"])) if limit else len(data["equations"])
    end = int(data["offsets"][count])
    result = dict(data)
    result["equations"] = data["equations"][:count]
    result["offsets"] = data["offsets"][: count + 1]
    result["eq_indices"] = data["eq_indices"][:count]
    for field in ("features", "probabilities", "labels", "bboxes", "stacked"):
        result[field] = data[field][:end]
    return result


def operator_features(data: dict) -> np.ndarray:
    logprobs = np.log(np.clip(data["probabilities"], 1e-7, 1))
    return np.column_stack((data["features"], logprobs)).astype(np.float32)


def train_operator(train: dict, validation: dict, name: str, output: Path) -> dict:
    """Fit the existing three-class reranker using clean train/validation crops."""
    train = take_equations(train, TRAIN_EQUATION_LIMIT)
    val = take_equations(validation, VALIDATION_EQUATION_LIMIT)
    subset = GROUPS["operators"]
    local_idx = {INDEX[token]: i for i, token in enumerate(subset)}
    good_train = np.flatnonzero(np.isin(train["labels"], list(local_idx)))
    good_val = np.flatnonzero(np.isin(val["labels"], list(local_idx)))
    if len(good_train) < max(MIN_TRAINING_CROPS, len(subset) * 5) or len(good_val) < len(subset) * 2:
        raise ValueError(f"Insufficient clean operator crops: train={len(good_train)}, validation={len(good_val)}")
    X_tr, X_val = operator_features(train)[good_train], operator_features(val)[good_val]
    y_tr = np.asarray([local_idx[int(y)] for y in train["labels"][good_train]], np.int64)
    y_val = np.asarray([local_idx[int(y)] for y in val["labels"][good_val]], np.int64)
    mu, sigma = X_tr.mean(axis=0), np.maximum(X_tr.std(axis=0), 1e-3)
    X_tr = np.clip((X_tr - mu) / sigma, -8, 8)
    X_val = np.clip((X_val - mu) / sigma, -8, 8)
    observed = np.bincount(y_tr, minlength=len(subset)).astype(np.float64)
    if np.any(observed == 0):
        raise ValueError("Operator training needs examples of +, division, and equals")
    model = SmallMLP(X_tr.shape[1], len(subset)).to(DEVICE)
    weights = (observed.sum() / observed) ** 0.35
    weights /= float(weights.mean())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=BATCH_SIZE, shuffle=True
    )
    val_x, val_y = torch.from_numpy(X_val).to(DEVICE), torch.from_numpy(y_val).to(DEVICE)
    best_loss, best_epoch, bad, history = math.inf, 0, 0, []
    best_weights = None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(val_x)
            val_loss = float(criterion(logits, val_y).item())
            val_acc = float((logits.argmax(1) == val_y).float().mean().item())
        history.append({"epoch": epoch, "val_loss": val_loss, "val_crop_accuracy": val_acc})
        if val_loss < best_loss - EARLY_STOPPING_MIN_DELTA:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= EARLY_STOPPING_PATIENCE:
                break
    if best_weights is None:
        raise ValueError("Operator MLP training did not produce a checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": name,
            "model_state_dict": best_weights,
            "input_dim": int(X_tr.shape[1]),
            "class_names": subset,
            "mean": mu.tolist(),
            "std": sigma.tolist(),
            "feature_set": "combined",
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
        },
        output / "operator_mlp.pt",
    )
    with (output / "operator_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return load_operator(output / "operator_mlp.pt", name)


def load_operator(path: Path, name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing learned postprocessor {path}; train with --post operator_mlp --split all")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved.get("model") != name
        or saved.get("class_names") != GROUPS["operators"]
        or saved.get("feature_set") != "combined"
    ):
        raise ValueError(f"Operator MLP model, classes, or features mismatch: {path}")
    model = SmallMLP(int(saved["input_dim"]), len(GROUPS["operators"]))
    model.load_state_dict(saved["model_state_dict"])
    return {
        "model": model.to(DEVICE).eval(),
        "mean": np.asarray(saved["mean"], np.float32),
        "std": np.asarray(saved["std"], np.float32),
    }


def operator_probabilities(data: dict, learned: dict) -> np.ndarray:
    X = operator_features(data)
    X = np.clip((X - learned["mean"]) / learned["std"], -8, 8)
    out = []
    with torch.inference_mode():
        for start in range(0, len(X), BATCH_SIZE):
            tensor = torch.from_numpy(X[start : start + BATCH_SIZE]).to(DEVICE)
            out.append(torch.softmax(learned["model"](tensor), dim=1).cpu().numpy())
    return np.vstack(out) if out else np.empty((0, 3), np.float32)


def tune_operator(validation: dict, learned: dict) -> dict:
    """Select mixing parameters on validation only, with the original tie rules."""
    from ml_segmentation.classification.evaluate_classification import prediction_rows

    data = take_equations(validation, VALIDATION_EQUATION_LIMIT)
    base = data["probabilities"].argmax(axis=1)
    rerank = operator_probabilities(data, learned)
    best = None
    for alpha in (0.25, 0.5, 0.75, 1.0):
        for margin in (0.00, 0.05, 0.10, 0.20):
            candidate = apply_operator_mlp(base, data["probabilities"], rerank, alpha, margin)
            rows = prediction_rows(data, candidate)
            key = (
                sum(row["sequence_correct"] for row in rows),
                -sum(row["token_edit_distance"] for row in rows),
                -int(np.count_nonzero(candidate != base)),
            )
            if best is None or key > best[0]:
                best = (
                    key,
                    {"alpha": alpha, "margin": margin, "selected_on": "validation", "validation_equations": len(rows)},
                )
    return best[1]


def operator_mlp(predictions: np.ndarray, data: dict, settings: dict, learned: dict | None):
    if learned is None:
        raise ValueError("operator_mlp requires a trained operator checkpoint")
    params = settings["operator_mlp"]
    rerank = operator_probabilities(data, learned)
    result = apply_operator_mlp(predictions, data["probabilities"], rerank, params["alpha"], params["margin"])
    data["alternative_probabilities"] = fused_operator_scores(data["probabilities"], rerank, params["alpha"], CLASSES)
    changes = [
        {"crop_index": int(i), "from": CLASSES[int(predictions[i])], "to": CLASSES[int(result[i])]}
        for i in np.flatnonzero(result != predictions)
    ]
    return result, changes


def syntax(predictions: np.ndarray, data: dict, settings: dict, learned: dict | None):
    result, edits = predictions.copy(), []
    for k, equation_index in enumerate(data["eq_indices"]):
        lo, hi = map(int, (data["offsets"][k], data["offsets"][k + 1]))
        visible = [i for i in range(lo, hi) if CLASSES[int(predictions[i])] != "NOISE"]
        tokens = [CLASSES[int(predictions[i])] for i in visible]
        if not tokens:
            continue
        edited, changes = repair_one_parentheses(tokens, data["probabilities"][visible], **settings["syntax"])
        for change in changes:
            i = visible[change["token_position"]]
            result[i] = INDEX[edited[change["token_position"]]]
            edits.append({"equation_index": int(equation_index), "crop_index": i, "segment_index": i - lo, **change})
    return result, edits


def equals(predictions: np.ndarray, data: dict, settings: dict, learned: dict | None):
    scores = data.get("alternative_probabilities", data["probabilities"])
    return apply_equals_at_end_only(predictions, scores, data["offsets"], data["eq_indices"], CLASSES)


def selected_methods(names: list[str]) -> list[str]:
    if names == ["none"]:
        return []
    if names == ["all"]:
        return list(POST_CLASSIFICATION_METHODS)
    if not names or any(name not in POST_CLASSIFICATION_METHODS for name in names):
        raise ValueError("Use post method names, 'all', or 'none'; do not mix all/none with method names")
    return [name for name in POST_CLASSIFICATION_METHODS if name in names]


def apply_methods(data: dict, methods: list[str], settings: dict, learned: dict | None) -> dict:
    # Dictionary order is the established operator -> syntax -> equals order.
    working = dict(data)
    working.pop("alternative_probabilities", None)
    current = data["probabilities"].argmax(axis=1)
    stages = {"base": current}
    for name in selected_methods(methods) if methods else []:
        current, _ = POST_CLASSIFICATION_METHODS[name](current, working, settings, learned)
        stages[name] = current
    return stages


POST_CLASSIFICATION_METHODS = {
    "operator_mlp": operator_mlp,
    "syntax": syntax,
    "equals": equals,
}

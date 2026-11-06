from collections import Counter

import numpy as np

OVERLAP_ID = 65535
PIXEL_TOLERANCE = 2
COMPONENT_FRACTION = 0.01
SYMBOL_FRACTION = 0.10


def equation_flags(sample_index: int, split: str, metrics: dict) -> dict:
    """Shared equation identity and annotation-based segmentation status."""
    recoverable = bool(metrics["grouping_recoverable"])
    return {
        "equation_id": f"{split}:{sample_index}",
        "sample_index": int(sample_index),
        "split": split,
        "segmentation_recoverable": recoverable,
        "segmentation_unrecoverable": not recoverable,
    }


def require_aligned(image: np.ndarray, mask: np.ndarray, noise: np.ndarray):
    if image.ndim != 2 or image.dtype != np.uint8 or not np.all((image == 0) | (image == 255)):
        raise ValueError("Expected binary uint8 equation image")
    if mask.shape != image.shape or mask.dtype != np.uint16:
        raise ValueError("Instance-ID mask must be uint16 and match image shape")
    if noise.shape != image.shape or noise.dtype != np.uint8:
        raise ValueError("Noise mask must be uint8 and match image shape")
    if np.any((mask != 0) & (image != 0)):
        raise ValueError("Annotation mismatch: labelled instance pixels exist on white image pixels")
    if np.any((noise != 0) & (image != 0)):
        raise ValueError("Annotation mismatch: noise labels exist on white image pixels")
    if np.any((noise != 0) & (mask != 0)):
        raise ValueError("Annotation mismatch: noise and instance masks overlap")
    if np.any(mask == OVERLAP_ID):
        raise ValueError("Legacy overlap ownership (65535); regenerate and preprocess")
    if np.any((image == 0) & (mask == 0) & (noise == 0)):
        raise ValueError("Dataset ownership error: unassigned black pixels. Reprocess with the new pipeline")


def thresholded_mix(
    foreign: int,
    labeled: int,
    minority_fraction: float,
    pixel_tolerance: int,
    comp_fraction: float,
    symbol_fraction: float,
) -> bool:
    """Tolerate tiny cross-symbol contamination while protecting small symbols."""
    if foreign <= pixel_tolerance or labeled == 0:
        return False
    return foreign / labeled > comp_fraction or minority_fraction > symbol_fraction


def thresholded_non_symbol_mix(non_symbol: int, symbol: int, pixel_tolerance: int, comp_fraction: float) -> bool:
    """Whether zero-owned ink materially contaminates a real-symbol group."""
    if symbol <= 0 or non_symbol <= pixel_tolerance:
        return False
    return non_symbol / max(1, symbol + non_symbol) > comp_fraction


def analyze(
    image: np.ndarray,
    gt: np.ndarray,
    noise: np.ndarray,
    anns: list,
    segments: list,
    pixel_tolerance: int,
    comp_fraction: float,
    symbol_fraction: float,
):
    """Measure component ownership directly from foreground pixels."""
    require_aligned(image, gt, noise)
    n_ids = len(anns)
    if n_ids == 0:
        raise ValueError("No instance annotations")
    if [int(s["instance_id"]) for s in anns] != list(range(1, n_ids + 1)):
        raise ValueError("Symbol IDs must be ordered 1..N")
    legal_gt = (gt == 0) | (gt == OVERLAP_ID) | ((gt >= 1) & (gt <= n_ids))
    if not np.all(legal_gt):
        raise ValueError("Instance mask contains IDs absent from symbol_annotations")

    visible = np.bincount(gt[(gt != 0) & (gt != OVERLAP_ID)].ravel(), minlength=n_ids + 1)
    results = []
    covered = np.zeros_like(image, dtype=np.uint8)
    own_ink = Counter()
    total_symbol_foreign = 0
    total_non_symbol_in_symbol_groups = 0
    total_noise_in_symbol_groups = 0
    total_unassigned_in_symbol_groups = 0
    mixed_strict = 0
    symbol_mixed_tolerant = 0
    symbol_non_symbol_mixed_tolerant = 0
    wrong_merge_components = 0
    noise_only = 0
    unlabeled_only = 0
    standalone_zero_owned = 0
    max_foreign_fraction = 0.0
    max_minority_fraction = 0.0
    max_non_symbol_fraction = 0.0
    sum_unknown = 0
    sum_overlap = 0
    sum_noise = 0
    per_symbol_fragment_counts = Counter()

    for k, segment in enumerate(segments):
        x1, y1, x2, y2 = map(int, segment.bbox)
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            raise ValueError("Candidate bounding box out of bounds")
        fg = segment.image == 0
        if fg.shape != (y2 - y1, x2 - x1):
            raise ValueError("Candidate crop shape does not match bounding box")
        if not np.array_equal(image[y1:y2, x1:x2][fg], np.zeros(fg.sum(), dtype=np.uint8)):
            raise ValueError("Candidate includes non-ink foreground")
        cover = covered[y1:y2, x1:x2]
        if np.any(cover[fg]):
            raise ValueError("Candidate components overlap in foreground pixels")
        cover[fg] = 1

        lab = gt[y1:y2, x1:x2][fg]
        noisy = noise[y1:y2, x1:x2][fg]
        assigned = lab[(lab > 0) & (lab != OVERLAP_ID)]
        counts = np.bincount(assigned.astype(np.int32), minlength=n_ids + 1)
        ids = np.flatnonzero(counts[1:]) + 1
        dominant = int(ids[np.argmax(counts[ids])]) if len(ids) else 0
        labeled = int(len(assigned))
        secondary = [(int(i), int(counts[i])) for i in ids if i != dominant]
        foreign = sum(v for _, v in secondary)
        worst_minority = max((n / max(1, int(visible[i])) for i, n in secondary), default=0.0)
        significant_symbol_mix = thresholded_mix(
            foreign,
            labeled,
            worst_minority,
            pixel_tolerance,
            comp_fraction,
            symbol_fraction,
        )
        strict = len(ids) > 1

        non_symbol = int(np.count_nonzero(lab == 0))
        noise_pixel_count = int(np.count_nonzero((lab == 0) & (noisy != 0)))
        unknown = int(np.count_nonzero((lab == 0) & (noisy == 0)))
        overlap = int(np.count_nonzero(lab == OVERLAP_ID))
        significant_non_symbol_mix = thresholded_non_symbol_mix(non_symbol, labeled, pixel_tolerance, comp_fraction)
        significant_wrong_merge = bool(significant_symbol_mix or significant_non_symbol_mix)

        mixed_strict += int(strict)
        symbol_mixed_tolerant += int(significant_symbol_mix)
        symbol_non_symbol_mixed_tolerant += int(significant_non_symbol_mix)
        wrong_merge_components += int(significant_wrong_merge)
        noise_only += int(not labeled and noise_pixel_count > 0 and unknown == 0)
        unlabeled_only += int(not labeled and unknown > 0 and noise_pixel_count == 0)
        standalone_zero_owned += int(not labeled and non_symbol > 0)

        own_ink[dominant] += int(counts[dominant]) if dominant else 0
        if dominant:
            per_symbol_fragment_counts[dominant] += 1

        total_symbol_foreign += foreign
        if labeled:
            total_non_symbol_in_symbol_groups += non_symbol
            total_noise_in_symbol_groups += noise_pixel_count
            total_unassigned_in_symbol_groups += unknown
        max_foreign_fraction = max(max_foreign_fraction, foreign / max(1, labeled))
        max_minority_fraction = max(max_minority_fraction, worst_minority)
        max_non_symbol_fraction = max(
            max_non_symbol_fraction,
            non_symbol / max(1, labeled + non_symbol) if labeled else 0.0,
        )
        sum_unknown += unknown
        sum_overlap += overlap
        sum_noise += noise_pixel_count
        ink_pixels = int(fg.sum())

        results.append(
            {
                "component_id": k + 1,
                "bbox": [x1, y1, x2, y2],
                "pixels": ink_pixels,
                "labelled_pixels": labeled,
                "dominant_instance_id": dominant,
                "dominant_token": str(anns[dominant - 1]["token"]) if dominant else "",
                "dominant_pixels": int(counts[dominant]) if dominant else 0,
                "foreign_pixels": foreign,
                "foreign_fraction": round(foreign / max(1, labeled), 6),
                "max_fraction_of_foreign_symbol": round(worst_minority, 6),
                "other_instances": ";".join(f"{i}:{n}" for i, n in secondary),
                "non_symbol_pixels": non_symbol,
                "non_symbol_fraction": round(non_symbol / max(1, labeled + non_symbol), 6) if labeled else 1.0,
                "overlap_pixels": overlap,
                "noise_pixels": noise_pixel_count,
                "unassigned_ink_pixels": unknown,
                "strict_mixed": bool(strict),
                "significant_symbol_mix": bool(significant_symbol_mix),
                "significant_symbol_non_symbol_mix": bool(significant_non_symbol_mix),
                "significant_mixed": significant_wrong_merge,
                "significant_wrong_merge": significant_wrong_merge,
            }
        )

    if not np.all(covered[image == 0]):
        raise ValueError("CCL+splitting lost foreground pixels")
    if np.any(covered[image != 0]):
        raise ValueError("CCL+splitting added foreground pixels")

    vanished_ids = [i for i in range(1, n_ids + 1) if visible[i] == 0]
    swallowed = [
        i
        for i in range(1, n_ids + 1)
        if visible[i]
        and (int(visible[i]) - own_ink[i] > pixel_tolerance)
        and ((int(visible[i]) - own_ink[i]) / int(visible[i]) > symbol_fraction)
    ]
    total_foreground = int(np.count_nonzero(image == 0))
    significant_overlap = sum_overlap > pixel_tolerance and (
        sum_overlap / max(1, int(visible[1:].sum()) + sum_overlap) > comp_fraction
    )

    grouping_recoverable = (
        wrong_merge_components == 0 and not swallowed and not vanished_ids and not significant_overlap
    )
    strict_recoverable = (
        mixed_strict == 0 and symbol_non_symbol_mixed_tolerant == 0 and not vanished_ids and sum_overlap == 0
    )

    return {
        "candidate_components": len(segments),
        "mixed_components_strict": mixed_strict,
        "mixed_components_significant": symbol_mixed_tolerant,
        "symbol_non_symbol_mixed_components_significant": symbol_non_symbol_mixed_tolerant,
        "wrong_merge_components_significant": wrong_merge_components,
        "noise_only_components": noise_only,
        "unlabelled_only_components": unlabeled_only,
        "standalone_zero_owned_components": standalone_zero_owned,
        "foreign_pixels": int(total_symbol_foreign),
        "non_symbol_pixels_in_symbol_groups": int(total_non_symbol_in_symbol_groups),
        "noise_pixels_in_symbol_groups": int(total_noise_in_symbol_groups),
        "unassigned_pixels_in_symbol_groups": int(total_unassigned_in_symbol_groups),
        "max_foreign_fraction": round(max_foreign_fraction, 6),
        "max_fraction_of_foreign_symbol": round(max_minority_fraction, 6),
        "max_non_symbol_fraction_in_symbol_group": round(max_non_symbol_fraction, 6),
        "unknown_ink_pixels": sum_unknown,
        "noise_pixels": sum_noise,
        "ambiguous_overlap_pixels": sum_overlap,
        "significant_ambiguous_overlap": bool(significant_overlap),
        "black_ink_pixels": total_foreground,
        "unknown_ink_percent": round(100 * sum_unknown / max(1, total_foreground), 4),
        "annotated_symbol_percent": round(100 * int(visible[1:].sum()) / max(1, total_foreground), 4),
        "excess_fragments_for_merger": int(sum(max(0, c - 1) for c in per_symbol_fragment_counts.values())),
        "visible_symbols": int(np.count_nonzero(visible[1:])),
        "total_symbols": n_ids,
        "vanished_symbols": len(vanished_ids),
        "vanished_ids": vanished_ids,
        "swallowed_symbols": len(swallowed),
        "swallowed_ids": swallowed,
        "symbol_ink_pixels": int(visible[1:].sum()),
        "grouping_recoverable": bool(grouping_recoverable),
        "strict_recoverable": bool(strict_recoverable),
        "components": results,
    }


def evaluate(results) -> tuple[dict, list[dict]]:
    """Score streamed equation dictionaries and retain per-equation flags."""
    totals = Counter()
    rows = []
    failures = Counter(
        {
            name: 0
            for name in (
                "significant_symbol_symbol_mixing",
                "significant_symbol_noise_mixing",
                "vanished_symbols",
                "swallowed_symbols",
                "significant_ambiguous_overlap",
            )
        }
    )
    for item in results:
        image, gt, noise = item["image"], item["gt"], item["noise"]
        annotations, segments, stats = item["annotations"], item["segments"], item["stats"]
        metrics = analyze(image, gt, noise, annotations, segments, PIXEL_TOLERANCE, COMPONENT_FRACTION, SYMBOL_FRACTION)
        rows.append(
            {
                **equation_flags(item["idx"], item["split"], metrics),
                "source": item["source"],
                "equation": item["equation"],
                "original_components": stats["original_components"],
                "final_candidates": len(segments),
                "cuts": len(segments) - stats["original_components"],
                "wrong_mix_components": metrics["wrong_merge_components_significant"],
                "vanished_symbols": metrics["vanished_symbols"],
                "swallowed_symbols": metrics["swallowed_symbols"],
            }
        )
        totals["total_equations"] += 1
        totals["original_components"] += stats["original_components"]
        totals["final_candidates"] += len(segments)
        totals["cuts"] += len(segments) - stats["original_components"]
        totals["attempted_cuts"] += stats["attempted_cuts"]
        totals["successful_splits"] += stats["successful_splits"]
        totals["recoverable_equations"] += int(metrics["grouping_recoverable"])
        totals["unrecoverable_equations"] += int(not metrics["grouping_recoverable"])
        for reason, key in (
            ("significant_symbol_symbol_mixing", "mixed_components_significant"),
            ("significant_symbol_noise_mixing", "symbol_non_symbol_mixed_components_significant"),
            ("vanished_symbols", "vanished_symbols"),
            ("swallowed_symbols", "swallowed_symbols"),
            ("significant_ambiguous_overlap", "significant_ambiguous_overlap"),
        ):
            failures[reason] += int(bool(metrics[key]))
    count = totals["total_equations"]
    if not count:
        raise ValueError("No equations evaluated")
    summary = {
        **dict(totals),
        "recoverability_percent": round(100 * totals["recoverable_equations"] / count, 4),
        "unrecoverability_percent": round(100 * totals["unrecoverable_equations"] / count, 4),
        "failure_reasons": dict(failures),
        "assumption": "Perfect downstream grouping and classification; standalone noise may be rejected.",
        "tolerances": {
            "pixel_tolerance": PIXEL_TOLERANCE,
            "component_fraction": COMPONENT_FRACTION,
            "symbol_fraction": SYMBOL_FRACTION,
        },
        "cut_definition": "cuts counts additional candidates; attempted_cuts counts proposed seams; successful_splits counts accepted binary splits. Seam-search rejections are not counted as attempted cuts.",
        "failure_definition": "Counts equations per reason; reasons may overlap. Invalid ownership or lost/added ink raises an error instead of producing a partial report.",
    }
    return summary, rows
